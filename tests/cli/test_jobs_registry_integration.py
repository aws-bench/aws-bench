"""End-to-end integration tests for ``aws-bench run``'s registry wiring.

These tests exercise the full chain from the CLI down through the real
``AwsBenchJob.create`` (tasks/scenarios/accounts/exports resolution + the
scenario-reference gate), with only the AWS-facing seams mocked:
``AccountManager`` and ``collect_account_exports`` (in ``aws_bench.task.job``),
``EnvironmentFactory.validate_resource_policies``, the ``TaskClient`` download
seam, and ``AwsBenchJob.run`` (so no Docker / agent ever runs).

Tasks and scenarios are materialized as REAL on-disk directories because
``AwsBenchTask`` and ``Scenario`` both validate at construction.

``test_path_without_dataset_does_not_load_registry`` is the load-bearing
regression: it patches ``AwsBenchRegistry.from_path`` to raise if called, so
any future edit that accidentally loads the registry during a ``--path`` run
will fail this test.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import typer
from harbor.models.job.result import JobResult, JobStats
from harbor.tasks.client import BatchDownloadResult, TaskDownloadResult

from aws_bench.dataset.config import AwsBenchDatasetConfig
from aws_bench.dataset.exceptions import ScenarioReferenceError

FIXTURES = Path(__file__).parent.parent / "dataset" / "fixtures"

# The single task/scenario pair carried by ``registry_valid.json``.
# REGISTRY_TASK_NAME is the registry DESCRIPTOR name: it governs the cloned dir
# basename and the download-seam lookup (``tid.path.name``). The task.toml
# ``[task] name`` is a separate thing — a Harbor ``org/name`` package id — so the
# on-disk task declares REGISTRY_TASK_PACKAGE, not the bare descriptor name.
REGISTRY_TASK_NAME = "lambda-not-reading-appconfig-value"
REGISTRY_TASK_PACKAGE = f"aws-bench/{REGISTRY_TASK_NAME}"
REGISTRY_SCENARIO_NAME = "lambda-with-broken-environment-variables"


# ---------------------------------------------------------------------------
# On-disk task / scenario builders (real dirs — both validate on construction)
# ---------------------------------------------------------------------------


def _make_task_dir(task_dir: Path, *, task_name: str, scenario_id: str) -> Path:
    """Build a minimal valid aws-bench task dir.

    ``AwsBenchTask`` validates at construction (Harbor instruction + test +
    aws-bench env definition + a ``[scenario] scenario_id``), so the dir must be
    fully formed. The ``scenario_id`` is what the post-fetch reference gate in
    ``AwsBenchJob.create`` reads off each task.
    """
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(
        f'[task]\nname = "{task_name}"\n\n[scenario]\nscenario_id = "{scenario_id}"\n'
    )
    (task_dir / "instruction.md").write_text("do it\n")
    (task_dir / "tests").mkdir(exist_ok=True)
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task_dir / "environment").mkdir(exist_ok=True)
    (task_dir / "environment" / "Dockerfile").write_text("FROM scratch\n")
    return task_dir


def _make_scenario_dir(scenario_dir: Path, manifest_name: str) -> Path:
    """Build a minimal valid scenario dir whose manifest declares ``manifest_name``.

    The manifest name is the canonical key the reference gate matches against;
    a descriptor/directory name MAY differ from it.
    """
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "scenario.toml").write_text(
        f"""
schema_version = "1.0"
[scenario]
name = "{manifest_name}"
description = "test"
account_tags = ["PRIMARY"]
regions = ["us-east-1"]
[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
[deploy]
timeout_sec = 60.0
[verify]
timeout_sec = 60.0
[cleanup]
timeout_sec = 60.0
"""
    )
    (scenario_dir / "scenario").mkdir(exist_ok=True)
    (scenario_dir / "scenario" / "Dockerfile").write_text("FROM scratch\n")
    (scenario_dir / "deploy").mkdir(exist_ok=True)
    (scenario_dir / "deploy" / "deploy.sh").write_text("#!/bin/sh\n")
    return scenario_dir


def _fake_job_result() -> JobResult:
    """Minimal ``JobResult`` for ``print_job_results_tables`` / runtime summary."""
    from uuid import uuid4

    now = datetime.now(timezone.utc)
    return JobResult(
        id=uuid4(),
        started_at=now,
        finished_at=now,
        n_total_trials=0,
        stats=JobStats(evals={}),
    )


# ---------------------------------------------------------------------------
# Shared fixture: stub the AWS-facing seams of ``AwsBenchJob.create`` / ``run``
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_external_deps(mocker):
    """Stub external dependencies so tests drive the real ``AwsBenchJob.create``.

    The CLI invokes ``AwsBenchJob.create``, so the AWS-facing seams live in
    ``aws_bench.task.job``. We patch only those seams and stub ``run``;
    task/scenario resolution, the reference gate, metric resolution, and
    trial-config construction all execute for real.
    """
    ns = MagicMock()

    # AwsBenchJob.run → return a real JobResult; the CLI reads timestamps and
    # job._job_result_path off it. No Docker / agent execution.
    ns.run = mocker.patch(
        "aws_bench.task.job.AwsBenchJob.run",
        new=AsyncMock(return_value=_fake_job_result()),
    )

    # Host-env confirmation prompt — no-op for tests.
    ns.confirm_host_env = mocker.patch("aws_bench.cli.jobs._confirm_host_env_access")

    # Skip Docker resource-policy validation (no daemon in tests).
    ns.validate_policies = mocker.patch(
        "aws_bench.task.job.EnvironmentFactory.validate_resource_policies"
    )

    # AccountManager.resolve_test_environment → a real TestEnvironment with one
    # ACTIVE account per (scenario, tag). AwsBenchJob.create reads
    # test_env.account_for(...) and assigns the object to config.test_environment
    # (it rides the resume identity), so a real Pydantic model is required.
    from aws_bench.account_management.models import OrgInfo, ScenarioAccount, TestEnvironment

    def _fake_resolve_test_environment(ou_name, required):
        accounts = {
            scenario_name: {
                next(iter(tags)): ScenarioAccount(
                    account_id="111111111111",
                    email="acct@example.com",
                    scenario_name=scenario_name,
                    account_tag=next(iter(tags)),
                )
            }
            for scenario_name, tags in required.items()
        }
        return TestEnvironment(
            org=OrgInfo(
                org_id="o-1",
                root_id="r-1",
                management_account_id="111111111111",
                management_account_email="mgmt@example.com",
            ),
            ou_id="ou-1",
            ou_name=ou_name,
            accounts=accounts,
        )

    account_manager_instance = MagicMock()
    account_manager_instance.resolve_test_environment = MagicMock(
        side_effect=_fake_resolve_test_environment
    )
    ns.account_manager_cls = mocker.patch(
        "aws_bench.task.job.AccountManager", return_value=account_manager_instance
    )
    ns.account_manager = account_manager_instance

    # collect_account_exports — empty placeholder ledger; create always calls
    # this once with the resolved {account: regions} targets.
    ns.collect_exports = mocker.patch("aws_bench.task.job.collect_account_exports", return_value={})

    # Silence the results-summary renderer.
    mocker.patch("aws_bench.cli.jobs.print_job_results_tables")

    # Preflight seams: run is Docker-only and checks AWS identity before create.
    # No daemon / live credentials in tests, so stub all three.
    mocker.patch("aws_bench.cli.jobs.preflight_docker_cli")
    mocker.patch("aws_bench.cli.jobs.preflight_docker_daemon")
    mocker.patch("aws_bench.cli.jobs.preflight_docker_plugins")
    mocker.patch("aws_bench.cli.jobs.preflight_aws_credentials")

    return ns


def _run_cli_start(**kwargs):
    """Invoke the ``start`` function synchronously.

    We call ``start`` directly rather than going through Typer's argv parser —
    keeping the test focused on behavior, not on argument plumbing.
    """
    from aws_bench.cli.jobs import start

    defaults = {
        "config_path": None,
        "path": None,
        "dataset_name_version": None,
        "registry_url": None,
        "registry_path": None,
        "include_task_names": None,
        "exclude_task_names": None,
        "n_tasks": None,
        "agent_name": None,
        "agent_import_path": None,
        "model_names": None,
        "agent_kwargs": None,
        "agent_env": None,
        "mcp_config": None,
        "skills": None,
        "extra_instruction_paths": None,
        "metric": None,
        "job_name": None,
        "jobs_dir": None,
        "n_attempts": None,
        "n_concurrent": None,
        "max_retries": None,
        "retry_include": None,
        "retry_exclude": None,
        "quiet": False,
        "debug": False,
        "env_file": None,
        "environment_force_build": None,
        "environment_delete": None,
        "verifier_env": None,
        "override_cpus": None,
        "override_memory_mb": None,
        "override_storage_mb": None,
        "override_gpus": None,
        "disable_verification": False,
        "yes": True,
        "verify": False,
        "scenario_path": None,
        # Typer injects the Context at the CLI; a direct call supplies a stub
        # whose empty meta yields no ledger (the no-op path).
        "ctx": MagicMock(meta={}),
    }
    defaults.update(kwargs)
    start(env_name="awsbench-ou", **defaults)


def _patch_task_client_download(mocker, dirs_by_name: dict[str, Path]):
    """Patch ``TaskClient.download_tasks`` at its ``dataset.config`` call site.

    Both ``cache_tasks`` and ``cache_scenarios`` route git refs through
    ``TaskClient().download_tasks(...)``. Each id's path basename matches its
    task/scenario name, so we look the prepared on-disk dir up by basename —
    one patch serves both task and scenario fetches.
    """

    async def fake_download_tasks(self, task_ids, **kwargs):
        results = []
        for tid in task_ids:
            name = tid.path.name
            results.append(
                TaskDownloadResult(
                    path=dirs_by_name[name],
                    download_time_sec=0.0,
                    cached=False,
                    resolved_git_commit_id=getattr(tid, "git_commit_id", None),
                )
            )
        return BatchDownloadResult(results=results, total_time_sec=0.0)

    return mocker.patch(
        "aws_bench.dataset.config.TaskClient.download_tasks",
        new=fake_download_tasks,
    )


def _arrange_registry_task_and_scenario(
    mocker,
    tmp_path,
    *,
    task_scenario_id: str = REGISTRY_SCENARIO_NAME,
    manifest_name: str = REGISTRY_SCENARIO_NAME,
):
    """Materialize the registry's git task + scenario as real dirs behind the download seam.

    The fixture's task ref resolves to basename ``REGISTRY_TASK_NAME`` and its
    scenario ref to basename ``REGISTRY_SCENARIO_NAME``; the patched
    ``download_tasks`` maps each to the dir we build here.
    """
    task_dir = _make_task_dir(
        tmp_path / "fetched" / REGISTRY_TASK_NAME,
        task_name=REGISTRY_TASK_PACKAGE,
        scenario_id=task_scenario_id,
    )
    scenario_dir = _make_scenario_dir(tmp_path / "fetched" / REGISTRY_SCENARIO_NAME, manifest_name)
    _patch_task_client_download(
        mocker,
        {REGISTRY_TASK_NAME: task_dir, REGISTRY_SCENARIO_NAME: scenario_dir},
    )


def _capture_created_job(mocker) -> dict:
    """Wrap the real ``AwsBenchJob.create`` to capture the job it returns.

    Saves the original classmethod, then patches ``create`` with a coroutine
    that delegates to it and stashes the result. Returns a dict that gains a
    ``"job"`` (and ``"config"``) key after the CLI runs. Lets the genuine
    resolution path execute (only the AWS seams are stubbed by the fixture).
    """
    from aws_bench.task.job import AwsBenchJob

    original = AwsBenchJob.create.__func__  # underlying function of the classmethod
    captured: dict = {}

    async def _wrapped_create(config):
        job = await original(AwsBenchJob, config)
        captured["job"] = job
        captured["config"] = config
        return job

    mocker.patch("aws_bench.cli.jobs.AwsBenchJob.create", new=_wrapped_create)
    return captured


# ---------------------------------------------------------------------------
# Dataset-source rejection (validate_run / model validator / CLI guard)
# ---------------------------------------------------------------------------


def test_dataset_with_scenario_path_is_rejected(stub_external_deps, mocker, tmp_path):
    # ``-d`` and ``--scenario-path`` are mutually exclusive — enforced by
    # ``AwsBenchDatasetConfig.validate_dataset_source``.
    from pydantic import ValidationError

    override_root = tmp_path / "local-dev"
    override_root.mkdir()

    with pytest.raises(ValidationError) as exc_info:
        _run_cli_start(
            dataset_name_version="aws-bench@1.0.0",
            registry_path=FIXTURES / "registry_valid.json",
            scenario_path=override_root,
        )

    assert "not both" in str(exc_info.value).lower()


def test_no_dataset_and_no_scenario_path_errors(stub_external_deps, mocker):
    # Neither -d nor --scenario-path nor --path: no CLI source and an empty
    # config.dataset → start() exits 1 with the "no dataset provided" message.
    with pytest.raises(typer.Exit) as exc_info:
        _run_cli_start()  # no -d, no --scenario-path, no --path
    assert exc_info.value.exit_code == 1


def test_package_name_dataset_is_rejected(stub_external_deps, mocker):
    # A package-style ``org/name`` dataset is rejected by the model validator
    # (aws-bench uses a JSON registry, not package datasets).
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        _run_cli_start(dataset_name_version="some-org/aws-bench@1.0.0")
    assert "org/name" in str(exc_info.value).lower() or "package" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Registry validation + scenario fetch: fail-fast behavior
# ---------------------------------------------------------------------------


def test_invalid_registry_fails_fast_before_any_fetch(stub_external_deps, mocker, capsys):
    # Duplicate scenario names fire at registry-load / validate time, before
    # any TaskClient call is reached. The CLI wraps the underlying
    # RegistryValidationError into typer.Exit(1) and prints to stderr/out.
    task_client = mocker.patch("aws_bench.dataset.config.TaskClient")

    with pytest.raises(typer.Exit) as exc_info:
        _run_cli_start(
            dataset_name_version="bad-dupes@1.0.0",
            registry_path=FIXTURES / "registry_duplicate_scenarios.json",
        )

    assert exc_info.value.exit_code == 1
    captured = capsys.readouterr()
    assert "duplicate scenario name" in (captured.out + captured.err).lower()
    # Registry validation short-circuited before any fetch was invoked.
    task_client.assert_not_called()


def test_missing_scenario_reference_fails_fast(stub_external_deps, mocker, tmp_path):
    # A task whose ``[scenario] scenario_id`` matches no scenario in the
    # selected dataset raises ``ScenarioReferenceError`` from
    # ``AwsBenchJob.create``. The CLI wraps it as typer.Exit(1) with the error
    # chained as __cause__.
    _arrange_registry_task_and_scenario(
        mocker, tmp_path, task_scenario_id="this-scenario-does-not-exist"
    )

    with pytest.raises(typer.Exit) as exc_info:
        _run_cli_start(
            dataset_name_version="aws-bench@1.0.0",
            registry_path=FIXTURES / "registry_valid.json",
            jobs_dir=tmp_path / "jobs",
        )
    assert exc_info.value.exit_code == 1

    cause = exc_info.value.__cause__
    assert isinstance(cause, ScenarioReferenceError)
    msg = str(cause)
    assert "this-scenario-does-not-exist" in msg
    assert REGISTRY_SCENARIO_NAME in msg  # listed as available
    assert "Available scenarios" in msg


def test_registry_name_disagrees_with_manifest_name(stub_external_deps, mocker, tmp_path):
    """Locks the canonical-key invariant: the gate matches the manifest name.

    The registry declares the scenario under ``REGISTRY_SCENARIO_NAME``, but the
    cloned manifest declares ``[scenario] name = "different-manifest-name"``. A
    task referencing the manifest name resolves; a task referencing the
    registry-declared name raises ``ScenarioReferenceError``.
    """
    # Task references the registry-declared name → fails (not the manifest name).
    _arrange_registry_task_and_scenario(
        mocker,
        tmp_path,
        task_scenario_id=REGISTRY_SCENARIO_NAME,
        manifest_name="different-manifest-name",
    )
    with pytest.raises(typer.Exit) as exc_info:
        _run_cli_start(
            dataset_name_version="aws-bench@1.0.0",
            registry_path=FIXTURES / "registry_valid.json",
            jobs_dir=tmp_path / "jobs",
        )
    cause = exc_info.value.__cause__
    assert isinstance(cause, ScenarioReferenceError)
    assert "different-manifest-name" in str(cause)  # the available manifest name

    # Task references the canonical manifest name → resolves through create+run.
    _arrange_registry_task_and_scenario(
        mocker,
        tmp_path,
        task_scenario_id="different-manifest-name",
        manifest_name="different-manifest-name",
    )
    _run_cli_start(
        dataset_name_version="aws-bench@1.0.0",
        registry_path=FIXTURES / "registry_valid.json",
        jobs_dir=tmp_path / "jobs2",
    )
    stub_external_deps.run.assert_awaited()


def test_scenario_fetch_failure_aborts_run(stub_external_deps, mocker, tmp_path):
    """A scenario fetch failure is fail-fast: the run aborts with exit 1.

    ``cache_scenarios`` is a batched fetch — a clone failure raises
    ``ScenarioFetchError`` rather than returning a partial result, so the run
    aborts (before the reference gate). The CLI's _run_job wraps it as
    typer.Exit(1).
    """
    from aws_bench.dataset.exceptions import ScenarioFetchError

    # The task still has to materialize so get_task_map succeeds; only the
    # scenario fetch fails.
    task_dir = _make_task_dir(
        tmp_path / "fetched" / REGISTRY_TASK_NAME,
        task_name=REGISTRY_TASK_PACKAGE,
        scenario_id=REGISTRY_SCENARIO_NAME,
    )
    _patch_task_client_download(mocker, {REGISTRY_TASK_NAME: task_dir})

    async def _fail_cache_scenarios(self):
        raise ScenarioFetchError(f"Failed to fetch scenario {REGISTRY_SCENARIO_NAME!r}")

    mocker.patch.object(AwsBenchDatasetConfig, "cache_scenarios", new=_fail_cache_scenarios)

    with pytest.raises(typer.Exit) as exc_info:
        _run_cli_start(
            dataset_name_version="aws-bench@1.0.0",
            registry_path=FIXTURES / "registry_valid.json",
            jobs_dir=tmp_path / "jobs",
        )
    assert exc_info.value.exit_code == 1


def test_run_fails_fast_when_init_was_not_run(stub_external_deps, mocker, tmp_path):
    """`aws-bench run` surfaces a clean error when the OU has no scenario-tagged accounts.

    AccountManager.resolve_test_environment raises AccountResolutionError;
    AwsBenchJob.create lets it propagate, and cli/jobs.py wraps it as
    typer.Exit(1).
    """
    from aws_bench.account_management.exceptions import AccountResolutionError

    _arrange_registry_task_and_scenario(mocker, tmp_path)
    stub_external_deps.account_manager.resolve_test_environment.side_effect = (
        AccountResolutionError(
            f"No accounts in OU 'awsbench-ou' tagged with "
            f"aws-bench:scenario=<{REGISTRY_SCENARIO_NAME}/...>. "
            "Run 'aws-bench env init' to provision."
        )
    )

    with pytest.raises(typer.Exit) as exc_info:
        _run_cli_start(
            dataset_name_version="aws-bench@1.0.0",
            registry_path=FIXTURES / "registry_valid.json",
            jobs_dir=tmp_path / "jobs",
        )
    assert exc_info.value.exit_code == 1


# ---------------------------------------------------------------------------
# --path path: registry must NEVER load when -d is absent
# ---------------------------------------------------------------------------


def test_path_without_dataset_does_not_load_registry(stub_external_deps, mocker, tmp_path):
    # Patch both registry loaders at their config-module call sites; if any
    # future edit loads the registry during a --path run, these stubs raise
    # and the test fails.
    from_path = mocker.patch(
        "aws_bench.dataset.config.AwsBenchRegistry.from_path",
        side_effect=RuntimeError("REGRESSION: registry must not load without -d"),
    )
    from_url = mocker.patch(
        "aws_bench.dataset.config.AwsBenchRegistry.from_url",
        side_effect=RuntimeError("REGRESSION: registry must not load without -d"),
    )

    # ``--path`` (a tasks dir with one real task) + ``--scenario-path`` (a dir
    # with the matching scenario) → fully local resolution, no registry.
    tasks_root = tmp_path / "tasks"
    _make_task_dir(
        tasks_root / "my-task", task_name="aws-bench/my-task", scenario_id="some-scenario-id"
    )
    scenario_root = tmp_path / "local-scenarios"
    _make_scenario_dir(scenario_root / "some-scenario-id", "some-scenario-id")

    _run_cli_start(
        path=tasks_root,
        scenario_path=scenario_root,
        jobs_dir=tmp_path / "jobs",
    )

    # The registry was never touched, and the run reached AwsBenchJob.run.
    from_path.assert_not_called()
    from_url.assert_not_called()
    stub_external_deps.collect_exports.assert_called_once()
    stub_external_deps.run.assert_awaited_once()


# ---------------------------------------------------------------------------
# Metrics: registry metrics seed the dataset-source bucket; --metric appends
# ---------------------------------------------------------------------------


def test_registry_metrics_seed_dataset_source_bucket(stub_external_deps, mocker, tmp_path):
    # Registry-declared metrics land in the job's per-source bucket (keyed by
    # the dataset source name), verified on the REAL job._metrics built by
    # AwsBenchJob._resolve_metrics.
    from harbor.metrics.max import Max
    from harbor.metrics.mean import Mean
    from harbor.models.metric.config import MetricConfig
    from harbor.models.metric.type import MetricType

    _arrange_registry_task_and_scenario(mocker, tmp_path)
    mocker.patch.object(
        AwsBenchDatasetConfig,
        "resolve_metric_configs",
        new=AsyncMock(
            return_value=[
                MetricConfig(type=MetricType.MEAN),
                MetricConfig(type=MetricType.MAX),
            ]
        ),
    )
    captured = _capture_created_job(mocker)

    _run_cli_start(
        dataset_name_version="aws-bench@1.0.0",
        registry_path=FIXTURES / "registry_valid.json",
        jobs_dir=tmp_path / "jobs",
    )

    # The dataset source bucket is keyed by the registry dataset name.
    seeded = captured["job"]._metrics["aws-bench"]
    assert any(isinstance(m, Mean) for m in seeded)
    assert any(isinstance(m, Max) for m in seeded)


def test_cli_metric_appends_to_registry_metrics(stub_external_deps, mocker, tmp_path):
    # A ``--metric`` CLI flag appends on top of the registry metrics in the
    # source bucket (does not replace them).
    from harbor.metrics.max import Max
    from harbor.metrics.mean import Mean
    from harbor.models.metric.config import MetricConfig
    from harbor.models.metric.type import MetricType

    _arrange_registry_task_and_scenario(mocker, tmp_path)
    mocker.patch.object(
        AwsBenchDatasetConfig,
        "resolve_metric_configs",
        new=AsyncMock(return_value=[MetricConfig(type=MetricType.MEAN)]),
    )
    captured = _capture_created_job(mocker)

    _run_cli_start(
        dataset_name_version="aws-bench@1.0.0",
        registry_path=FIXTURES / "registry_valid.json",
        metric=["max"],
        jobs_dir=tmp_path / "jobs",
    )

    seeded = captured["job"]._metrics["aws-bench"]
    types = {type(m) for m in seeded}
    assert Mean in types and Max in types  # registry mean + CLI max


@pytest.mark.asyncio
async def test_sourceless_task_falls_through_to_adhoc_bucket(mocker):
    # A task whose source is None must NOT seed a per-source bucket: it falls
    # through to the framework's adhoc bucket. Verified directly on
    # AwsBenchJob._resolve_metrics.
    from types import SimpleNamespace

    from harbor.metrics.mean import Mean

    from aws_bench.cli.job_config import AwsBenchJobConfig
    from aws_bench.task.job import AwsBenchJob

    config = AwsBenchJobConfig(
        dataset=AwsBenchDatasetConfig(name="aws-bench"), env_name="ou", metrics=[]
    )
    mocker.patch.object(config.dataset, "resolve_metric_configs", new=AsyncMock(return_value=[]))

    result = await AwsBenchJob._resolve_metrics(
        config,
        [SimpleNamespace(source=None)],  # type: ignore[arg-type]
    )

    # No source bucket; only adhoc, floored to Mean.
    assert set(result.keys()) == {"adhoc"}
    assert all(isinstance(m, Mean) for m in result["adhoc"])


# ---------------------------------------------------------------------------
# Retry flags + extra-instruction override reach the resolved config
# ---------------------------------------------------------------------------


def test_retry_include_exclude_flags_reach_retry_config(stub_external_deps, mocker, tmp_path):
    # --retry-include / --retry-exclude set RetryConfig.include_exceptions /
    # exclude_exceptions as sets on the config handed to AwsBenchJob.create.
    # create is fully stubbed here — we only inspect the config the CLI built.
    captured = {}

    async def _capturing_create(config):
        captured["config"] = config
        job = MagicMock()
        job._job_result_path = Path("/tmp/x")
        job.run = AsyncMock(return_value=_fake_job_result())
        return job

    mocker.patch("aws_bench.cli.jobs.AwsBenchJob.create", new=_capturing_create)

    _run_cli_start(
        dataset_name_version="aws-bench@1.0.0",
        registry_path=FIXTURES / "registry_valid.json",
        jobs_dir=tmp_path / "jobs",
        retry_include=["ThrottlingException", "TimeoutError"],
        retry_exclude=["ValueError"],
    )

    config = captured["config"]
    assert config.retry.include_exceptions == {"ThrottlingException", "TimeoutError"}
    assert config.retry.exclude_exceptions == {"ValueError"}


def test_n_concurrent_flag_threads_to_config(stub_external_deps, mocker, tmp_path):
    # -n/--n-concurrent lands on config.n_concurrent_trials handed to create.
    captured = {}

    async def _capturing_create(config):
        captured["config"] = config
        job = MagicMock()
        job._job_result_path = Path("/tmp/x")
        job.run = AsyncMock(return_value=_fake_job_result())
        return job

    mocker.patch("aws_bench.cli.jobs.AwsBenchJob.create", new=_capturing_create)

    _run_cli_start(
        dataset_name_version="aws-bench@1.0.0",
        registry_path=FIXTURES / "registry_valid.json",
        jobs_dir=tmp_path / "jobs",
        n_concurrent=2,
    )

    assert captured["config"].n_concurrent_trials == 2


def test_config_file_n_concurrent_not_clobbered(stub_external_deps, mocker, tmp_path):
    # Without -n, a -c config file's n_concurrent_trials survives (the flag is
    # int | None applied conditionally, so None must not overwrite it).
    captured = {}

    async def _capturing_create(config):
        captured["config"] = config
        job = MagicMock()
        job._job_result_path = Path("/tmp/x")
        job.run = AsyncMock(return_value=_fake_job_result())
        return job

    mocker.patch("aws_bench.cli.jobs.AwsBenchJob.create", new=_capturing_create)

    from aws_bench.cli.job_config import AwsBenchJobConfig

    # The dataset is fully declared in the -c file (registry_path included) so no
    # CLI dataset arg is passed — passing one would rebuild config.dataset and is
    # orthogonal to what this test guards.
    config_file = tmp_path / "job.json"
    config_file.write_text(
        AwsBenchJobConfig(
            dataset=AwsBenchDatasetConfig(
                name="aws-bench",
                version="1.0.0",
                registry_path=FIXTURES / "registry_valid.json",
            ),
            n_concurrent_trials=3,
        ).model_dump_json()
    )

    _run_cli_start(
        config_path=config_file,
        jobs_dir=tmp_path / "jobs",
    )

    assert captured["config"].n_concurrent_trials == 3


def test_extra_instruction_cli_overrides_registry(stub_external_deps, mocker, tmp_path):
    # A CLI --extra-instruction-path wins over the registry's instructions:
    # resolve_instruction_paths must NOT be called, and the CLI path lands on
    # config.extra_instruction_paths.
    _arrange_registry_task_and_scenario(mocker, tmp_path)
    resolve = mocker.patch.object(
        AwsBenchDatasetConfig,
        "resolve_instruction_paths",
        new=AsyncMock(return_value=[Path("/registry/should-not-be-used.md")]),
    )
    cli_instr = tmp_path / "cli.md"
    cli_instr.write_text("operator override")
    captured = _capture_created_job(mocker)

    _run_cli_start(
        dataset_name_version="aws-bench@1.0.0",
        registry_path=FIXTURES / "registry_valid.json",
        extra_instruction_paths=[cli_instr],
        jobs_dir=tmp_path / "jobs",
    )

    resolve.assert_not_called()
    assert captured["config"].extra_instruction_paths == [cli_instr]


# ---------------------------------------------------------------------------
# Smoke: real AwsBenchJob.create end-to-end (local, no AWS) does not crash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_job_create_local_does_not_crash(stub_external_deps, mocker, tmp_path):
    """The REAL ``AwsBenchJob.create`` constructs a ready job from a local dataset.

    Drives the full create path (task listing + validation, scenario
    materialization, the reference gate, metric resolution, account/export
    resolution, trial-config construction) against real on-disk dirs, with
    only the AWS seams (AccountManager / collect_account_exports /
    EnvironmentFactory) stubbed by the fixture. No CLI, no Docker.
    """
    from aws_bench.cli.job_config import AwsBenchJobConfig
    from aws_bench.task.job import AwsBenchJob

    tasks_root = tmp_path / "tasks"
    _make_task_dir(tasks_root / "t1", task_name="aws-bench/t1", scenario_id="ec2-small")
    scenario_root = tmp_path / "scenarios"
    _make_scenario_dir(scenario_root / "ec2-small", "ec2-small")

    config = AwsBenchJobConfig(
        env_name="awsbench-ou",
        verify=False,
        jobs_dir=tmp_path / "jobs",
        job_name="smoke",
        dataset=AwsBenchDatasetConfig(path=tasks_root, scenarios_path=scenario_root),
    )

    job = await AwsBenchJob.create(config)

    assert job is not None
    assert len(job._trial_configs) == 1
    # Each trial got its account mapping from the (stubbed) test environment.
    assert list(job._trial_configs[0].account_mapping.values()) == ["111111111111"]


def test_metrics_instructions_fixture_loads():
    """The registry_metrics_instructions fixture parses with metrics + instruction fields."""
    from aws_bench.dataset.registry import AwsBenchRegistry

    reg = AwsBenchRegistry.from_path(FIXTURES / "registry_metrics_instructions.json")
    spec = reg.get_dataset_spec("metrics-demo", "1.0.0")
    assert len(spec.metrics) == 2
    assert spec.metrics[1].type.value == "uv-script"
    assert spec.metrics[1].kwargs["script_path"] == "metrics/cost/metric.py"
    assert len(spec.extra_instruction_paths) == 1


# ---------------------------------------------------------------------------
# Verification gate: persist config.json + verification.json and exit on failure
# ---------------------------------------------------------------------------


def _arrange_local_task_and_scenario(
    tmp_path, *, scenario_id: str = "ec2-small"
) -> tuple[Path, Path]:
    """Materialize one local task + matching scenario, returning their root dirs.

    Mirrors the local ``--path`` / ``--scenario-path`` driver used by
    ``test_real_job_create_local_does_not_crash``: a tasks root holding one task
    that references ``scenario_id``, and a scenarios root holding the matching
    scenario manifest. Drives the real ``AwsBenchJob.create`` with no registry.
    """
    tasks_root = tmp_path / "tasks"
    _make_task_dir(tasks_root / "t1", task_name="aws-bench/t1", scenario_id=scenario_id)
    scenario_root = tmp_path / "scenarios"
    _make_scenario_dir(scenario_root / scenario_id, scenario_id)
    return tasks_root, scenario_root


def test_verify_failure_persists_config_and_report(stub_external_deps, mocker, tmp_path):
    """A failing verification leaves config.json + verification.json and exits 1.

    ``verify_environment`` is patched at its ``cli.jobs`` call site to return a
    failing report, so the gate's persistence + summary + raise path runs
    independent of any real AWS verification. ``start`` catches the resulting
    ``EnvironmentVerifyError`` in its ``except Exception`` and re-raises
    ``typer.Exit(code=1)``.
    """
    from aws_bench.resource_management.verify.models import (
        AccountVerifyResult,
        RegionVerifyResult,
        VerificationReport,
    )

    failing = VerificationReport(
        passed=False,
        env_name="awsbench-ou",
        results=[
            AccountVerifyResult(
                account_id="123456789012",
                environment_id="ec2-small",
                success=False,
                region_results=[
                    RegionVerifyResult(
                        region="us-east-1",
                        success=False,
                        error_message="drift",
                        suggestion="Run 'aws-bench env reset'",
                    )
                ],
                error_message="drift",
            )
        ],
    )
    verify_mock = mocker.patch(
        "aws_bench.cli.jobs.ResourceManager.verify_environment",
        new=mocker.AsyncMock(return_value=failing),
    )

    tasks_root, scenario_root = _arrange_local_task_and_scenario(tmp_path)
    jobs_dir = tmp_path / "jobs"

    with pytest.raises(typer.Exit) as exc_info:
        _run_cli_start(
            path=tasks_root,
            scenario_path=scenario_root,
            jobs_dir=jobs_dir,
            job_name="verify-fail",
            verify=True,
        )
    assert exc_info.value.exit_code == 1

    # The gate ran and the trials never did.
    verify_mock.assert_awaited_once()
    stub_external_deps.run.assert_not_awaited()

    job_dir = jobs_dir / "verify-fail"
    assert (job_dir / "config.json").exists()
    report = json.loads((job_dir / "verification.json").read_text())
    assert report["passed"] is False
    assert report["results"][0]["account_id"] == "123456789012"


def test_verify_success_proceeds_to_run(stub_external_deps, mocker, tmp_path):
    """A passing verification writes config.json and proceeds to ``job.run``.

    No ``verification.json`` is written on success, and the run reaches the
    (stubbed) ``AwsBenchJob.run``.
    """
    from aws_bench.resource_management.verify.models import VerificationReport

    verify_mock = mocker.patch(
        "aws_bench.cli.jobs.ResourceManager.verify_environment",
        new=mocker.AsyncMock(
            return_value=VerificationReport(passed=True, env_name="awsbench-ou", results=[])
        ),
    )

    tasks_root, scenario_root = _arrange_local_task_and_scenario(tmp_path)
    jobs_dir = tmp_path / "jobs"

    _run_cli_start(
        path=tasks_root,
        scenario_path=scenario_root,
        jobs_dir=jobs_dir,
        job_name="verify-pass",
        verify=True,
    )

    verify_mock.assert_awaited_once()
    stub_external_deps.run.assert_awaited_once()

    job_dir = jobs_dir / "verify-pass"
    assert (job_dir / "config.json").exists()
    assert not (job_dir / "verification.json").exists()


def test_no_verify_env_skips_gate(stub_external_deps, mocker, tmp_path):
    """``--no-verify-env`` (verify=False) skips the gate and proceeds to run.

    ``verify_environment`` must not be called at all, and the run reaches the
    (stubbed) ``AwsBenchJob.run``.
    """
    verify_mock = mocker.patch(
        "aws_bench.cli.jobs.ResourceManager.verify_environment",
        new=mocker.AsyncMock(),
    )

    tasks_root, scenario_root = _arrange_local_task_and_scenario(tmp_path)
    jobs_dir = tmp_path / "jobs"

    _run_cli_start(
        path=tasks_root,
        scenario_path=scenario_root,
        jobs_dir=jobs_dir,
        job_name="no-verify",
        verify=False,
    )

    verify_mock.assert_not_awaited()
    stub_external_deps.run.assert_awaited_once()
