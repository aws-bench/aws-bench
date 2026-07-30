"""Tests for AwsBenchJob — AWS resolution in create + per-trial account/export slicing.

Covers _resolve_metrics (dataset-source bucket, no external seeding),
_init_trial_configs (each trial gets only its account's export slice), and
job-level resume (refuse-on-change via the inherited JobConfig.__eq__, with
aws-bench fields round-tripped by the subclass-parse override).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from harbor.job import Job
from harbor.models.trial.config import TaskConfig

from aws_bench.dataset.task_config import ConcurrencyMode
from aws_bench.task.job import ALLOWED_REGIONS_PLACEHOLDER, AwsBenchJob
from aws_bench.task.trial_config import AwsBenchTrialConfig

# --- _cache_tasks retry override -------------------------------------------


@pytest.mark.asyncio
async def test_cache_tasks_retries_transient_git_failure(monkeypatch):
    """The override retries a throttled base fetch and returns the base result unchanged.

    Backoff is neutralized by the autouse ``_no_git_fetch_backoff`` conftest fixture.
    """
    calls = {"n": 0}
    sentinel = {"id": "result"}  # stands in for dict[TaskIdType, TaskDownloadResult]

    async def flaky_base(task_configs: list[TaskConfig]) -> dict:
        calls["n"] += 1
        if calls["n"] < 3:
            raise subprocess.CalledProcessError(1, ["git"], b"", b"remote hung up")
        return sentinel

    monkeypatch.setattr(Job, "_cache_tasks", staticmethod(flaky_base))
    result = await AwsBenchJob._cache_tasks([])

    assert result is sentinel  # base return value flows through unchanged
    assert calls["n"] == 3


# --- _resolve_metrics ------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_metrics_seeds_dataset_source_bucket(mocker):
    """The dataset source gets its own metric bucket on the resolved map."""
    from harbor.metrics.mean import Mean

    from aws_bench.cli.job_config import AwsBenchJobConfig
    from aws_bench.dataset.config import AwsBenchDatasetConfig

    # _resolve_metrics asserts isinstance(config, AwsBenchJobConfig); use a real one
    # with a real (registry) dataset whose resolve_metric_configs is stubbed.
    config = AwsBenchJobConfig(
        dataset=AwsBenchDatasetConfig(name="aws-bench-all"), env_name="ou", metrics=[]
    )
    mocker.patch.object(config.dataset, "resolve_metric_configs", mocker.AsyncMock(return_value=[]))
    task_configs = [SimpleNamespace(source="aws-bench-all")]

    metrics = await AwsBenchJob._resolve_metrics(config, task_configs)  # type: ignore[arg-type]

    assert "adhoc" in metrics
    assert "aws-bench-all" in metrics
    # Each bucket floored to [Mean()] when no metrics configured.
    assert all(isinstance(m, Mean) for bucket in metrics.values() for m in bucket)


# --- _init_trial_configs: per-account export slicing -----------------------


def _job_skeleton(tmp_path, *, test_environment, exports, task_pairs, n_attempts=1):
    """An AwsBenchJob instance with only the fields _init_trial_configs reads.

    job_dir is a read-only property (config.jobs_dir / config.job_name), so the
    config carries those rather than setting job_dir directly.
    """
    from aws_bench.scenario.locator import ScenarioConfig

    job = AwsBenchJob.__new__(AwsBenchJob)
    job._task_pairs = task_pairs
    # Account-keyed export ledger lives on the job instance (not the config),
    # so it stays out of the job-level resume identity.
    job._account_exports = exports
    # scenario_id -> (descriptor, scenario). The scenario exposes
    # manifest.scenario.regions for the allowed-regions placeholder.
    job._run_pairs = {
        task.config.scenario.scenario_id: (
            ScenarioConfig(
                name=task.config.scenario.scenario_id,
                path=Path(f"scenarios/{task.config.scenario.scenario_id}"),
            ),
            SimpleNamespace(
                name=task.config.scenario.scenario_id,
                manifest=SimpleNamespace(
                    scenario=SimpleNamespace(regions=task.config.scenario.regions)
                ),
            ),
        )
        for _, task in task_pairs
    }
    job._id = None  # type: ignore[assignment]
    job.config = SimpleNamespace(  # type: ignore[assignment]
        jobs_dir=tmp_path,
        job_name="job",
        test_environment=test_environment,
        n_attempts=n_attempts,
        agents=[SimpleNamespace()],
        timeout_multiplier=1.0,
        agent_timeout_multiplier=None,
        verifier_timeout_multiplier=None,
        agent_setup_timeout_multiplier=None,
        environment_build_timeout_multiplier=None,
        environment=SimpleNamespace(),
        # _init_trial_configs deep-copies the verifier per trial; the double only
        # needs to answer model_copy (identity copy is fine for this test).
        verifier=SimpleNamespace(model_copy=lambda deep=False: SimpleNamespace()),
        artifacts=[],
        extra_instruction_paths=[],
        verify=True,
    )
    return job


def _aws_task(scenario_id, mode=ConcurrencyMode.MUTATING, regions=("us-east-1",)):
    return SimpleNamespace(
        config=SimpleNamespace(
            scenario=SimpleNamespace(scenario_id=scenario_id, regions=list(regions)),
            concurrency=SimpleNamespace(mode=mode),
        )
    )


def test_init_trial_configs_gives_each_trial_only_its_account_slice(tmp_path, mocker):
    """Two scenarios → two accounts; each trial gets ONLY its account's exports."""
    # Two (config, task) pairs, each a distinct scenario.
    tc_a = SimpleNamespace(get_task_id=lambda: "id-a")
    tc_b = SimpleNamespace(get_task_id=lambda: "id-b")
    task_pairs = [(tc_a, _aws_task("scenario-a")), (tc_b, _aws_task("scenario-b"))]

    _accounts = {"scenario-a": "111", "scenario-b": "222"}
    test_env = SimpleNamespace(
        account_for=lambda name: _accounts[name],
        mapping_for=lambda name: {"PRIMARY": _accounts[name]},
    )
    exports = {"111": {"BucketA": "a"}, "222": {"BucketB": "b"}}

    job = _job_skeleton(
        tmp_path,
        test_environment=test_env,
        exports=exports,
        task_pairs=task_pairs,
    )
    # AwsBenchTrialConfig validates its task field; bypass with a patched constructor
    # capturing the kwargs each trial is built with.
    built = []

    def _capture(**kwargs):
        built.append(kwargs)
        return MagicMock()

    mocker.patch("aws_bench.task.job.AwsBenchTrialConfig", side_effect=_capture)
    job._init_trial_configs()

    # Drop the injected allowed-regions key to assert purely on the account slice.
    by_account = {
        next(iter(k["account_mapping"].values())): {
            ek: ev
            for ek, ev in k["exports"]["PRIMARY"].items()
            if ek != ALLOWED_REGIONS_PLACEHOLDER
        }
        for k in built
    }
    assert by_account == {"111": {"BucketA": "a"}, "222": {"BucketB": "b"}}


def test_init_trial_configs_empty_slice_when_account_has_no_exports(tmp_path, mocker):
    """An account absent from the exports ledger yields an empty slice (not a KeyError)."""
    tc = SimpleNamespace(get_task_id=lambda: "id-a")
    test_env = SimpleNamespace(
        account_for=lambda name: "111",
        mapping_for=lambda name: {"PRIMARY": "111"},
    )
    job = _job_skeleton(
        tmp_path,
        test_environment=test_env,
        exports={},  # no exports for any account
        task_pairs=[(tc, _aws_task("scenario-a"))],
    )
    built = []
    mocker.patch(
        "aws_bench.task.job.AwsBenchTrialConfig",
        side_effect=lambda **k: built.append(k) or MagicMock(),
    )
    job._init_trial_configs()
    # No account slice existed, so only the injected allowed-regions key remains.
    assert built[0]["exports"] == {"PRIMARY": {ALLOWED_REGIONS_PLACEHOLDER: "us-east-1"}}


# --- allowed-regions placeholder injection ---------------------------------


def test_build_trial_config_injects_allowed_regions_per_scenario(tmp_path, mocker):
    """Each trial's exports carries its own scenario's regions, comma-joined."""
    tc_a = SimpleNamespace(get_task_id=lambda: "id-a")
    tc_b = SimpleNamespace(get_task_id=lambda: "id-b")
    task_pairs = [
        (tc_a, _aws_task("scenario-a", regions=["us-east-1"])),
        (tc_b, _aws_task("scenario-b", regions=["us-east-1", "us-west-2", "ap-northeast-1"])),
    ]
    _accounts = {"scenario-a": "111", "scenario-b": "222"}
    test_env = SimpleNamespace(
        account_for=lambda name: _accounts[name],
        mapping_for=lambda name: {"PRIMARY": _accounts[name]},
    )
    job = _job_skeleton(
        tmp_path,
        test_environment=test_env,
        exports={"111": {}, "222": {}},
        task_pairs=task_pairs,
    )
    built = []
    mocker.patch(
        "aws_bench.task.job.AwsBenchTrialConfig",
        side_effect=lambda **k: built.append(k) or MagicMock(),
    )
    job._init_trial_configs()

    by_account = {
        next(iter(k["account_mapping"].values())): k["exports"]["PRIMARY"][
            ALLOWED_REGIONS_PLACEHOLDER
        ]
        for k in built
    }
    assert by_account == {"111": "us-east-1", "222": "us-east-1, us-west-2, ap-northeast-1"}


def test_build_trial_config_rejects_reserved_key_collision(tmp_path, mocker):
    """A real export stealing the reserved key fails loud at build time."""
    from aws_bench.dataset.exceptions import ScenarioReferenceError

    tc = SimpleNamespace(get_task_id=lambda: "id-a")
    test_env = SimpleNamespace(
        account_for=lambda name: "111",
        mapping_for=lambda name: {"PRIMARY": "111"},
    )
    job = _job_skeleton(
        tmp_path,
        test_environment=test_env,
        # The account's export ledger already defines the reserved key.
        exports={"111": {ALLOWED_REGIONS_PLACEHOLDER: "stolen"}},
        task_pairs=[(tc, _aws_task("scenario-a"))],
    )
    mocker.patch(
        "aws_bench.task.job.AwsBenchTrialConfig",
        side_effect=lambda **k: MagicMock(),
    )
    with pytest.raises(ScenarioReferenceError, match="reserved"):
        job._init_trial_configs()


def test_build_trial_config_rejects_unmapped_scenario(tmp_path, mocker):
    """An unknown scenario (empty mapping) fails loud instead of a credential-less trial."""
    from aws_bench.dataset.exceptions import ScenarioReferenceError

    tc = SimpleNamespace(get_task_id=lambda: "id-a")
    # mapping_for returns {} for an unknown scenario; the guard must catch it.
    test_env = SimpleNamespace(
        account_for=lambda name: "111",
        mapping_for=lambda name: {},
    )
    job = _job_skeleton(
        tmp_path,
        test_environment=test_env,
        exports={"111": {}},
        task_pairs=[(tc, _aws_task("scenario-a"))],
    )
    mocker.patch(
        "aws_bench.task.job.AwsBenchTrialConfig",
        side_effect=lambda **k: MagicMock(),
    )
    with pytest.raises(ScenarioReferenceError, match="No account mapping"):
        job._init_trial_configs()


# --- AwsBenchTrialConfig fields used by the job ----------------------------


def test_trial_config_account_and_exports_are_real_fields():
    """Guard: the fields _init_trial_configs sets exist on AwsBenchTrialConfig."""
    from pathlib import Path

    from harbor.models.trial.config import TaskConfig

    from aws_bench.scenario.locator import ScenarioConfig

    cfg = AwsBenchTrialConfig(
        task=TaskConfig(path=Path("tasks/x")),
        scenario=ScenarioConfig(name="ec2-small", path=Path("scenarios/ec2-small")),
        scenario_id="ec2-small",
        concurrency_mode=ConcurrencyMode.MUTATING,
        account_mapping={"PRIMARY": "111"},
        regions=["us-east-1"],
        exports={"PRIMARY": {"K": "v"}},
    )
    assert cfg.account_mapping == {"PRIMARY": "111"}
    assert cfg.exports == {"PRIMARY": {"K": "v"}}


def test_build_trial_config_sets_scenario_descriptor():
    """_build_trial_config sets scenario from the descriptor resolved in create."""
    from pathlib import Path

    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        VerifierConfig,
    )

    from aws_bench.scenario.locator import ScenarioConfig

    descriptor = ScenarioConfig(name="ec2-small", path=Path("scenarios/ec2-small"))

    job = AwsBenchJob.__new__(AwsBenchJob)
    job._run_pairs = {
        "ec2-small": (
            descriptor,
            SimpleNamespace(
                name="ec2-small",
                manifest=SimpleNamespace(scenario=SimpleNamespace(regions=["us-east-1"])),
            ),
        )
    }
    job._account_exports = {}
    job._id = None  # type: ignore[assignment]
    job.config = SimpleNamespace(  # type: ignore[assignment]
        jobs_dir=Path("jobs"),
        job_name="run-0",
        test_environment=SimpleNamespace(
            account_for=lambda name: "111111111111",
            mapping_for=lambda name: {"PRIMARY": "111111111111"},
        ),
        timeout_multiplier=1.0,
        agent_timeout_multiplier=None,
        verifier_timeout_multiplier=None,
        agent_setup_timeout_multiplier=None,
        environment_build_timeout_multiplier=None,
        environment=EnvironmentConfig(),
        verifier=VerifierConfig(),
        artifacts=[],
        extra_instruction_paths=[],
        verify=True,
    )
    task = SimpleNamespace(
        config=SimpleNamespace(
            scenario=SimpleNamespace(scenario_id="ec2-small"),
            concurrency=SimpleNamespace(mode=ConcurrencyMode.MUTATING),
        )
    )

    cfg = job._build_trial_config(
        task_config=TaskConfig(path=Path("tasks/t")),
        task=task,  # type: ignore[arg-type]
        agent_config=AgentConfig(),
    )
    assert cfg.scenario is descriptor
    assert cfg.scenario_id == "ec2-small"
    assert cfg.regions == ["us-east-1"]


def test_build_trial_config_produces_tag_keyed_export_slice():
    """_build_trial_config keys exports by account tag, each carrying that account's slice."""
    from pathlib import Path

    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        VerifierConfig,
    )

    from aws_bench.scenario.locator import ScenarioConfig

    descriptor = ScenarioConfig(name="ec2-small", path=Path("scenarios/ec2-small"))

    job = AwsBenchJob.__new__(AwsBenchJob)
    job._run_pairs = {
        "ec2-small": (
            descriptor,
            SimpleNamespace(
                name="ec2-small",
                manifest=SimpleNamespace(scenario=SimpleNamespace(regions=["us-east-1"])),
            ),
        )
    }
    job._account_exports = {"111111111111": {"BucketName": "b-123"}}
    job._id = None  # type: ignore[assignment]
    job.config = SimpleNamespace(  # type: ignore[assignment]
        jobs_dir=Path("jobs"),
        job_name="run-0",
        test_environment=SimpleNamespace(
            account_for=lambda name: "111111111111",
            mapping_for=lambda name: {"PRIMARY": "111111111111"},
        ),
        timeout_multiplier=1.0,
        agent_timeout_multiplier=None,
        verifier_timeout_multiplier=None,
        agent_setup_timeout_multiplier=None,
        environment_build_timeout_multiplier=None,
        environment=EnvironmentConfig(),
        verifier=VerifierConfig(),
        artifacts=[],
        extra_instruction_paths=[],
        verify=True,
    )
    task = SimpleNamespace(
        config=SimpleNamespace(
            scenario=SimpleNamespace(scenario_id="ec2-small"),
            concurrency=SimpleNamespace(mode=ConcurrencyMode.MUTATING),
        )
    )

    cfg = job._build_trial_config(
        task_config=TaskConfig(path=Path("tasks/t")),
        task=task,  # type: ignore[arg-type]
        agent_config=AgentConfig(),
    )
    # Drop the injected allowed-regions key to assert purely on the account slice.
    sliced = {
        tag: {k: v for k, v in exports.items() if k != ALLOWED_REGIONS_PLACEHOLDER}
        for tag, exports in cfg.exports.items()
    }
    assert sliced == {"PRIMARY": {"BucketName": "b-123"}}


# --- per-trial verifier isolation (real models, no mocks) ------------------


def test_init_trial_configs_deep_copies_verifier_per_trial(tmp_path):
    """Each trial gets its own VerifierConfig, so a runtime env overlay can't leak.

    Builds real configs (no AwsBenchTrialConfig stub) and mutates one trial's
    verifier.env; the others must be unaffected.
    """
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        VerifierConfig,
    )

    from aws_bench.scenario.locator import ScenarioConfig

    # task must be a real TaskConfig — AwsBenchTrialConfig validates the field.
    task_config = TaskConfig(path=Path("tasks/a"))
    job = AwsBenchJob.__new__(AwsBenchJob)
    job._task_pairs = [(task_config, _aws_task("scenario-a"))]
    job._account_exports = {"111": {"Bucket": "b"}}
    job._run_pairs = {
        "scenario-a": (
            ScenarioConfig(name="scenario-a", path=Path("scenarios/scenario-a")),
            SimpleNamespace(
                name="scenario-a",
                manifest=SimpleNamespace(scenario=SimpleNamespace(regions=["us-east-1"])),
            ),
        )
    }
    job._id = None  # type: ignore[assignment]
    job.config = SimpleNamespace(  # type: ignore[assignment]
        jobs_dir=tmp_path,
        job_name="job",
        test_environment=SimpleNamespace(
            account_for=lambda _name: "111",
            mapping_for=lambda _name: {"PRIMARY": "111"},
        ),
        n_attempts=2,  # two trials of the same task → distinct configs
        agents=[AgentConfig()],
        timeout_multiplier=1.0,
        agent_timeout_multiplier=None,
        verifier_timeout_multiplier=None,
        agent_setup_timeout_multiplier=None,
        environment_build_timeout_multiplier=None,
        environment=EnvironmentConfig(),
        verifier=VerifierConfig(env={"REGION": "us-east-1"}),
        artifacts=[],
        extra_instruction_paths=[],
        verify=True,
    )

    job._init_trial_configs()

    first, second = job._trial_configs
    first.verifier.env["REGION"] = "MUTATED"
    assert second.verifier.env == {"REGION": "us-east-1"}  # untouched


# --- resume: refuse-on-change vs unchanged-succeeds ------------------------


def _real_job_config(tmp_path, *, account_id="111"):
    """A minimal but real AwsBenchJobConfig carrying a resolved test_environment."""
    from aws_bench.account_management.models import OrgInfo, ScenarioAccount, TestEnvironment
    from aws_bench.cli.job_config import AwsBenchJobConfig

    return AwsBenchJobConfig(
        jobs_dir=tmp_path,
        job_name="run-0",
        test_environment=TestEnvironment(
            org=OrgInfo(
                org_id="o-1",
                root_id="r-1",
                management_account_id="999999999999",
                management_account_email="root@example.com",
            ),
            ou_id="ou-1",
            ou_name="awsbench-ou",
            accounts={
                "scenario-a": {
                    "primary": ScenarioAccount(
                        account_id=account_id,
                        email="a@example.com",
                        scenario_name="scenario-a",
                        account_tag="primary",
                    )
                }
            },
        ),
    )


def _resume_job(config):
    """An AwsBenchJob with only what _maybe_init_existing_job reads."""
    job = AwsBenchJob.__new__(AwsBenchJob)
    job.config = config
    return job


def test_resume_unchanged_config_succeeds(tmp_path):
    """An identical persisted config resumes (no FileExistsError, empty prior results)."""
    config = _real_job_config(tmp_path)
    config.jobs_dir.joinpath("run-0").mkdir(parents=True)
    (config.jobs_dir / "run-0" / "config.json").write_text(config.model_dump_json())

    job = _resume_job(config)
    job._maybe_init_existing_job()  # must not raise
    assert job._previous_trial_results == {}


def test_resume_refuses_changed_test_environment(tmp_path):
    """A persisted config whose test_environment differs refuses to resume."""
    persisted = _real_job_config(tmp_path, account_id="111")
    persisted.jobs_dir.joinpath("run-0").mkdir(parents=True)
    (persisted.jobs_dir / "run-0" / "config.json").write_text(persisted.model_dump_json())

    # The live job resolved a DIFFERENT account for the same scenario.
    live = _real_job_config(tmp_path, account_id="222")
    job = _resume_job(live)
    with pytest.raises(FileExistsError):
        job._maybe_init_existing_job()


# --- create(): retains run_scenarios, no longer verifies -------------------


def _make_task_dir(task_dir: Path, *, task_name: str, scenario_id: str) -> Path:
    """Build a minimal valid aws-bench task dir (validates at construction).

    The ``scenario_id`` is what the post-fetch reference gate in
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

    The manifest name is the canonical key the reference gate matches against.
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


def _stub_create_aws_seams(mocker):
    """Stub the AWS-facing seams of ``AwsBenchJob.create`` (no AWS, no Docker).

    Mirrors the seam set the create-exercising integration harness patches:
    Docker resource-policy validation, ``AccountManager.resolve_test_environment``
    (returns a real ``TestEnvironment`` so it can ride the resume identity), and
    ``collect_account_exports``. Task/scenario resolution and the reference gate
    run for real against the on-disk dirs.
    """
    from aws_bench.account_management.models import OrgInfo, ScenarioAccount, TestEnvironment

    mocker.patch("aws_bench.task.job.EnvironmentFactory.validate_resource_policies")

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
    mocker.patch("aws_bench.task.job.AccountManager", return_value=account_manager_instance)
    mocker.patch("aws_bench.task.job.collect_account_exports", return_value={})


@pytest.mark.asyncio
async def test_create_retains_run_scenarios_and_skips_verify(tmp_path, mocker):
    """create() exposes the resolved scenarios and no longer verifies the environment.

    Drives the REAL ``AwsBenchJob.create`` against on-disk task/scenario dirs
    (only the AWS seams stubbed). ``verify=True`` so that, on the pre-refactor
    code, the deleted block would have called ``verify_scenario`` once per
    scenario — asserting it is NOT called proves verification moved out of
    create(). ``verify_scenario`` is patched at its defining module so the guard
    holds regardless of how (or whether) ``task.job`` imports ResourceManager.
    """
    from aws_bench.cli.job_config import AwsBenchJobConfig
    from aws_bench.dataset.config import AwsBenchDatasetConfig

    verify_spy = mocker.patch(
        "aws_bench.resource_management.manager.ResourceManager.verify_scenario",
        new=mocker.AsyncMock(return_value=[]),
    )
    _stub_create_aws_seams(mocker)

    tasks_root = tmp_path / "tasks"
    _make_task_dir(tasks_root / "t1", task_name="aws-bench/t1", scenario_id="ec2-small")
    scenario_root = tmp_path / "scenarios"
    _make_scenario_dir(scenario_root / "ec2-small", "ec2-small")

    config = AwsBenchJobConfig(
        env_name="awsbench-ou",
        verify=True,
        jobs_dir=tmp_path / "jobs",
        job_name="run-0",
        dataset=AwsBenchDatasetConfig(path=tasks_root, scenarios_path=scenario_root),
    )

    job = await AwsBenchJob.create(config)

    assert [s.name for s in job.run_scenarios] == ["ec2-small"]
    verify_spy.assert_not_called()


@pytest.mark.asyncio
async def test_create_collects_exports_from_every_scenario_account(tmp_path, mocker):
    """Export collection targets every account a scenario maps to, not just the first.

    A scenario maps to one account per tag. ``create`` must build the
    ``collect_account_exports`` targets from all of them (``mapping_for``), so the
    per-tag export slice in ``_build_trial_config`` is populated for every tag.
    Stubs the resolved environment to map the scenario to TWO accounts (the
    manifest still validates a single tag; the stub bypasses that gate to exercise
    the multi-account collection path) and asserts both account ids are queried.
    """
    from aws_bench.account_management.models import OrgInfo, ScenarioAccount, TestEnvironment
    from aws_bench.cli.job_config import AwsBenchJobConfig
    from aws_bench.dataset.config import AwsBenchDatasetConfig

    mocker.patch("aws_bench.task.job.EnvironmentFactory.validate_resource_policies")
    mocker.patch(
        "aws_bench.resource_management.manager.ResourceManager.verify_scenario",
        new=mocker.AsyncMock(return_value=[]),
    )

    def _resolve_two_accounts(ou_name, required):
        accounts = {
            scenario_name: {
                "PRIMARY": ScenarioAccount(
                    account_id="111111111111",
                    email="p@example.com",
                    scenario_name=scenario_name,
                    account_tag="PRIMARY",
                ),
                "SECONDARY": ScenarioAccount(
                    account_id="222222222222",
                    email="s@example.com",
                    scenario_name=scenario_name,
                    account_tag="SECONDARY",
                ),
            }
            for scenario_name in required
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
    account_manager_instance.resolve_test_environment = MagicMock(side_effect=_resolve_two_accounts)
    mocker.patch("aws_bench.task.job.AccountManager", return_value=account_manager_instance)
    collect_spy = mocker.patch("aws_bench.task.job.collect_account_exports", return_value={})

    tasks_root = tmp_path / "tasks"
    _make_task_dir(tasks_root / "t1", task_name="aws-bench/t1", scenario_id="ec2-small")
    scenario_root = tmp_path / "scenarios"
    _make_scenario_dir(scenario_root / "ec2-small", "ec2-small")

    config = AwsBenchJobConfig(
        env_name="awsbench-ou",
        jobs_dir=tmp_path / "jobs",
        job_name="run-0",
        dataset=AwsBenchDatasetConfig(path=tasks_root, scenarios_path=scenario_root),
    )

    await AwsBenchJob.create(config)

    collect_spy.assert_called_once()
    targets = collect_spy.call_args.kwargs["targets"]
    assert set(targets) == {"111111111111", "222222222222"}


# --- run logging bridge ----------------------------------------------------


def _logger_bridge_job(tmp_path, *, debug=False, quiet=False):
    """A bare AwsBenchJob carrying only what _init_logger reads."""
    job = AwsBenchJob.__new__(AwsBenchJob)
    (tmp_path / "run-0").mkdir(parents=True, exist_ok=True)
    job.config = SimpleNamespace(  # type: ignore[assignment]
        jobs_dir=tmp_path, job_name="run-0", debug=debug, quiet=quiet
    )
    job.is_resuming = False
    job._log_file_handler = None
    job._console_handler = None
    return job


def test_init_logger_bridges_harbor_tree_onto_rich_handler(tmp_path):
    """_init_logger routes Harbor's run logger through the aws-bench RichHandler.

    The bridge is console-only: it does NOT open a job.log file handler (the CLI
    caller owns job.log via file_logging). It only replaces Harbor's plain
    console handler with the shared RichHandler so console lines share one format.
    """
    import logging

    from rich.logging import RichHandler

    from aws_bench.task.job import _HARBOR_LOGGER_NAME

    job = _logger_bridge_job(tmp_path)
    try:
        job._init_logger()

        # Harbor's tree now carries our RichHandler; the bridge opened no file.
        harbor_handlers = logging.getLogger(_HARBOR_LOGGER_NAME).handlers
        assert any(isinstance(h, RichHandler) for h in harbor_handlers)
        assert job._console_handler is None
        assert job._log_file_handler is None
    finally:
        job._close_logger_handlers()


def test_close_logger_handlers_detaches_the_bridge(tmp_path):
    """Closing removes the bridged handlers from both trees (no leak across jobs)."""
    import logging

    from aws_bench.logging.logger import NAMESPACE
    from aws_bench.task.job import _HARBOR_LOGGER_NAME

    job = _logger_bridge_job(tmp_path)
    job._init_logger()
    file_handler = job._log_file_handler
    aws_console = job._aws_console_handler
    job._close_logger_handlers()

    assert file_handler not in logging.getLogger(NAMESPACE).handlers
    assert aws_console not in logging.getLogger(_HARBOR_LOGGER_NAME).handlers


def test_init_logger_quiet_still_logs_console(tmp_path):
    """--quiet does not silence the log stream: the console handler is still installed.

    Quiet suppresses the live progress display (the base's concern), not logs.
    """
    job = _logger_bridge_job(tmp_path, quiet=True)
    try:
        job._init_logger()
        assert job._aws_console_handler is not None
    finally:
        job._close_logger_handlers()


def test_init_logger_debug_sets_console_to_debug(tmp_path):
    """--debug raises the run console handler to DEBUG."""
    import logging

    job = _logger_bridge_job(tmp_path, debug=True)
    try:
        job._init_logger()
        assert job._aws_console_handler is not None
        assert job._aws_console_handler.level == logging.DEBUG
    finally:
        job._close_logger_handlers()
