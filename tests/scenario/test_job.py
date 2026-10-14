"""Tests for aws_bench.scenario.job.ScenarioJob.

Mocks AccountManager, CredentialProvider, and ScenarioContainer (via the
trial's container injection point) so the discovery + account-resolution +
quota-gate + parallel-trial orchestration + persistence flow can be
exercised without Docker or AWS.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aws_bench.account_management.models import OrgInfo, ScenarioAccount, TestEnvironment
from aws_bench.dataset.config import AwsBenchDatasetConfig
from aws_bench.resource_management.cleanup.models import AccountCleanupResult
from aws_bench.resource_management.exceptions import SnapshotRegionMismatchError
from aws_bench.resource_management.models import (
    QuotaConfiguration,
    QuotaIncreaseResult,
    QuotaStatus,
)
from aws_bench.resource_management.reset.models import ResetResult
from aws_bench.resource_management.verify.models import AccountVerifyResult
from aws_bench.scenario.config import TrialEnvironmentConfig
from aws_bench.scenario.container import ExecResult
from aws_bench.scenario.events import ScenarioPhase
from aws_bench.scenario.exceptions import (
    InsufficientQuotaError,
    ScenarioDiscoveryError,
)
from aws_bench.scenario.job import ScenarioJob
from aws_bench.scenario.job_config import ScenarioJobConfig
from aws_bench.scenario.scenario import Scenario


@pytest.fixture(autouse=True)
def mock_resource_manager():
    """Mock ResourceManager methods to avoid AWS API calls during tests."""
    with (
        patch(
            "aws_bench.resource_management.manager.ResourceManager.snapshot_scenarios",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "aws_bench.resource_management.manager.ResourceManager.verify_scenario",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "aws_bench.resource_management.manager.ResourceManager.reset_scenarios",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "aws_bench.resource_management.manager.ResourceManager.cleanup_scenarios_by_name",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "aws_bench.resource_management.snapshot.manager.SnapshotManager.snapshot_exists",
            return_value=True,
        ),
        patch(
            "aws_bench.resource_management.snapshot.manager.SnapshotManager.validate_pre_setup_snapshot",
            return_value=None,
        ),
    ):
        yield


VALID_TOML = """\
schema_version = "1.0"

[scenario]
name = "{name}"
account_tags = ["PRIMARY"]
regions = ["us-east-1"]
"""


def _make_scenario(root: Path, name: str) -> Path:
    sd = root / name
    sd.mkdir()
    (sd / "scenario.toml").write_text(VALID_TOML.format(name=name))
    (sd / "scenario").mkdir()
    (sd / "scenario" / "Dockerfile").write_text("FROM alpine\n")
    (sd / "deploy").mkdir()
    (sd / "deploy" / "deploy.sh").write_text("#!/bin/sh\nexit 0\n")
    (sd / "verify").mkdir()
    (sd / "verify" / "verify.sh").write_text("#!/bin/sh\nexit 0\n")
    (sd / "cleanup").mkdir()
    (sd / "cleanup" / "cleanup.sh").write_text("#!/bin/sh\nexit 0\n")
    return sd


def _fake_creds():
    cp = MagicMock()
    session = MagicMock()
    fake = MagicMock()
    fake.access_key = "AKIA-T"
    fake.secret_key = "secret"
    fake.token = "token"
    creds = MagicMock()
    creds.get_frozen_credentials.return_value = fake
    session.get_credentials.return_value = creds
    cp.session = session
    return cp


def _fake_account_manager(account_map: dict[str, str]):
    """Account manager whose resolve_test_environment returns a real TestEnvironment.

    Each required (scenario, tag) gets one ACTIVE account; ``account_map`` is
    ``{account_tag: account_id}`` applied to every scenario.
    """
    am = MagicMock()

    def _bulk(_ou_name, required):
        accounts = {
            scenario_name: {
                tag: ScenarioAccount(
                    account_id=account_id,
                    email=f"{scenario_name}-{tag}@example.com",
                    scenario_name=scenario_name,
                    account_tag=tag,
                )
                for tag, account_id in account_map.items()
            }
            for scenario_name in required
        }
        return TestEnvironment(
            org=OrgInfo(
                org_id="o-1",
                root_id="r-1",
                management_account_id="999999999999",
                management_account_email="mgmt@example.com",
            ),
            ou_id="ou-1",
            ou_name=_ou_name,
            accounts=accounts,
        )

    am.resolve_test_environment.side_effect = _bulk
    return am


@pytest.fixture(autouse=True)
def patch_container():
    """Replace ScenarioContainer with a mock everywhere ScenarioTrial creates one."""
    with patch("aws_bench.scenario.trial.ScenarioContainer") as cls:
        instance = MagicMock()
        instance.build = AsyncMock()
        instance.start = AsyncMock()
        instance.run_phase = AsyncMock(return_value=ExecResult(exit_code=0, stdout="ok\n"))
        instance.stop = AsyncMock()
        instance.write_file = AsyncMock()
        cls.return_value = instance
        yield instance


@pytest.fixture(autouse=True)
def mock_account_manager():
    """Stub the trial's AccountManager: contamination reads, tag updates, and the SCP attach."""
    with patch("aws_bench.scenario.trial.AccountManager") as mock_cls:
        instance = mock_cls.return_value
        instance.get_contaminated_accounts.return_value = []
        instance.mark_contaminated = AsyncMock()
        instance.clear_contaminated = AsyncMock()
        yield instance


def _make_job_config(tmp_path: Path, **overrides) -> ScenarioJobConfig:
    # Pop dataset-level kwargs out of overrides if the caller passed them
    # via the legacy field names so the existing tests stay readable.
    include = overrides.pop("include_scenarios", None)
    exclude = overrides.pop("exclude_scenarios", None)
    scenarios_path = overrides.pop("scenarios_path", tmp_path / "scenarios")
    dataset = AwsBenchDatasetConfig(
        scenarios_path=scenarios_path,
        include_scenario_names=include,
        exclude_scenario_names=exclude,
    )
    return ScenarioJobConfig(
        ou_name="test-env",
        dataset=dataset,
        jobs_dir=tmp_path / "out",
        n_concurrent=2,
        environment=TrialEnvironmentConfig(),
        **overrides,
    )


# -- create() factory ------------------------------------------------------


def test_create_resolves_accounts_and_filters_scenarios(tmp_path):
    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")
    _make_scenario(sdir, "lambda-b")
    _make_scenario(sdir, "vpc-c")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path, include_scenarios=["lambda-*"])

    job = asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))

    assert job.config.test_environment is not None
    resolved = job.config.test_environment.to_scenario_account_mappings()
    assert sorted(resolved.keys()) == ["lambda-a", "lambda-b"]
    assert am.resolve_test_environment.call_count == 1


def test_create_does_not_mutate_operator_config(tmp_path):
    """ScenarioJob.create deep-copies before populating test_environment."""
    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path)
    assert config.test_environment is None

    job = asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))

    # Operator's instance untouched; only the job's resolved copy carries the env.
    assert config.test_environment is None
    assert job.config is not config
    assert job.config.test_environment is not None
    assert job.config.test_environment.to_scenario_account_mappings() == {
        "lambda-a": {"PRIMARY": "111"}
    }


def test_create_aborts_when_no_scenarios_match(tmp_path):
    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path, include_scenarios=["nope-*"])

    with pytest.raises(ValueError, match="No scenarios matched the filter"):
        asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))


def test_create_raises_account_resolution_error_on_missing_accounts(tmp_path):
    from aws_bench.account_management.exceptions import AccountResolutionError

    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")

    am = MagicMock()
    am.resolve_test_environment.side_effect = AccountResolutionError("not provisioned")
    config = _make_job_config(tmp_path)

    with pytest.raises(AccountResolutionError, match="not provisioned"):
        asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))


def test_create_raises_when_resolved_tags_miss_manifest_tags(tmp_path):
    """ScenarioJob.create propagates the strict-resolve missing-tag error.

    The strict gate lives on AccountManager.resolve_test_environment;
    when the OU's account-tag set does not cover the manifest's required
    tags it raises AccountResolutionError. ScenarioJob.create no longer
    wraps it — the native exception propagates so callers see the same
    message AccountManager already wrote (with the "run env init" hint).
    """
    from aws_bench.account_management.exceptions import AccountResolutionError

    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")  # manifest declares account_tags=["PRIMARY"]

    am = MagicMock()
    am.resolve_test_environment.side_effect = AccountResolutionError(
        "Missing tag(s): ['PRIMARY']. Run 'aws-bench env init --env-name test-env'."
    )
    config = _make_job_config(tmp_path)

    with pytest.raises(AccountResolutionError) as exc_info:
        asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))
    msg = str(exc_info.value)
    assert "PRIMARY" in msg
    assert "Missing tag(s)" in msg
    assert "env init" in msg  # actionable hint preserved through the wrap


# -- run() ----------------------------------------------------------------


def test_run_executes_phase_across_all_scenarios(tmp_path, patch_container):
    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")
    _make_scenario(sdir, "lambda-b")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path)

    job = asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))
    result = asyncio.run(job.run(ScenarioPhase.DEPLOY))

    assert result.n_total == 2
    assert result.n_succeeded == 2
    assert result.n_failed == 0
    assert result.all_passed
    assert patch_container.run_phase.await_count == 2


def test_run_persists_config_and_result_json(tmp_path):
    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path)

    job = asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))
    asyncio.run(job.run(ScenarioPhase.DEPLOY))

    job_dir = job.paths.job_dir
    assert (job_dir / "config.json").is_file()
    assert (job_dir / "result.json").is_file()
    cfg = json.loads((job_dir / "config.json").read_text())
    accounts = cfg["test_environment"]["accounts"]
    assert accounts.keys() == {"lambda-a"}
    assert accounts["lambda-a"]["PRIMARY"]["account_id"] == "111"


def test_verify_failure_counts_as_failed_trial(tmp_path):
    """A failing verify result (e.g. missing baseline) fails the trial."""
    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path)

    failing = AccountVerifyResult(
        account_id="111",
        environment_id="lambda-a",
        success=False,
        region_results=[],
        error_message="Snapshot not found",
    )
    with patch(
        "aws_bench.resource_management.manager.ResourceManager.verify_scenario",
        new_callable=AsyncMock,
        return_value=[failing],
    ):
        job = asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))
        result = asyncio.run(job.run(ScenarioPhase.VERIFY))

    assert result.n_failed == 1
    assert not result.all_passed
    assert result.trial_results[0].exception_info is not None


def test_reset_failure_counts_as_failed_trial(tmp_path):
    """A failing reset result (drift not reverted, no redeploy) fails the trial."""
    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path)

    failing = ResetResult(success=False, reason="drift could not be reverted")
    with patch(
        "aws_bench.resource_management.manager.ResourceManager.reset_scenarios",
        new_callable=AsyncMock,
        return_value=[failing],
    ):
        job = asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))
        result = asyncio.run(job.run(ScenarioPhase.RESET))

    assert result.n_failed == 1
    assert not result.all_passed
    assert result.trial_results[0].exception_info is not None


def test_cleanup_failure_counts_as_failed_trial(tmp_path):
    """A cleanup result carrying an error fails the trial."""
    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path)

    failing = AccountCleanupResult(account_id="111", summary=None, error="stack DELETE_FAILED")
    with patch(
        "aws_bench.resource_management.manager.ResourceManager.cleanup_scenarios_by_name",
        new_callable=AsyncMock,
        return_value=[failing],
    ):
        job = asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))
        result = asyncio.run(job.run(ScenarioPhase.CLEANUP))

    assert result.n_failed == 1
    assert not result.all_passed
    assert result.trial_results[0].exception_info is not None


def test_run_persists_per_trial_outputs(tmp_path):
    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path)

    job = asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))
    result = asyncio.run(job.run(ScenarioPhase.VERIFY))

    # trial_name is generated per-run with a shortuuid; find it in the result.
    assert len(result.trial_results) == 1
    trial_name = result.trial_results[0].trial_name
    assert trial_name.startswith("lambda-a__")
    trial_dir = job.paths.job_dir / trial_name
    assert (trial_dir / "config.json").is_file()
    assert (trial_dir / "result.json").is_file()


def test_run_failed_phase_marks_n_failed(tmp_path, patch_container):
    """Non-zero exit codes count toward n_failed."""
    patch_container.run_phase = AsyncMock(return_value=ExecResult(exit_code=2, stdout="boom\n"))

    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")
    _make_scenario(sdir, "lambda-b")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path)

    job = asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))
    result = asyncio.run(job.run(ScenarioPhase.DEPLOY))

    assert result.n_failed == 2
    assert not result.all_passed


def test_run_propagates_shutdown_after_persisting_completed_trials(tmp_path, patch_container):
    """A shutdown unwinds the job, but only after the finished trials are recorded.

    ``gather(return_exceptions=True)`` captures the cancelled trial's
    ``OperationCancelled`` as a value. The job coerces every outcome into the
    aggregate FIRST (so a completed sibling's result is not lost), persists it,
    and only then re-raises so the phase aborts. Guards against re-raising before
    the aggregate is built.
    """
    import json as _json

    from aws_bench.exceptions import OperationCancelled

    # The first trial to run its phase cancels; the other completes. Which named
    # scenario is which isn't fixed (trials run concurrently), so assertions below
    # check the partition (one clean / one cancelled), not a specific scenario.
    call_count = {"n": 0}

    async def _run_phase(*_args, **_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OperationCancelled("stop")
        return ExecResult(exit_code=0, stdout="ok\n")

    patch_container.run_phase = AsyncMock(side_effect=_run_phase)

    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")
    _make_scenario(sdir, "lambda-b")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path)

    job = asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))
    with pytest.raises(OperationCancelled):
        asyncio.run(job.run(ScenarioPhase.DEPLOY))

    # The aggregate was built and persisted before the re-raise: the completed
    # sibling's result survives and the cancelled trial is recorded as such —
    # not discarded, not both coerced to failures.
    persisted = _json.loads((job.paths.job_dir / "result.json").read_text())
    results = persisted["trial_results"]
    assert len(results) == 2
    n_clean = sum(1 for tr in results if tr["exception_info"] is None and tr["exit_code"] == 0)
    n_cancelled = sum(
        1
        for tr in results
        if tr["exception_info"] and tr["exception_info"]["exception_type"] == "OperationCancelled"
    )
    assert n_clean == 1
    assert n_cancelled == 1


def test_run_records_trials_when_job_task_cancelled(tmp_path, patch_container):
    """A Ctrl+C that cancels the job task still persists the trials it ran.

    A real signal cancels the ``run()`` task, so ``gather`` re-raises
    ``CancelledError`` and drops its results list. The aggregate must be rebuilt
    from the (now done) trial tasks in ``finally`` — otherwise result.json is
    written with an empty ``trial_results`` while the per-trial dirs hold results.
    Guards against the empty-aggregate-on-Ctrl+C regression.
    """
    import json as _json

    # Park both trials in their phase, then cancel the run task once both are
    # provably suspended — so the cancel lands mid-gather (real Ctrl+C timing),
    # with no reliance on the mock's await_count.
    both_parked = asyncio.Event()
    parked = {"n": 0}

    async def _run_phase(*_args, **_kwargs):
        parked["n"] += 1
        if parked["n"] == 2:
            both_parked.set()
        await asyncio.Event().wait()  # block forever; the cancel unwinds it

    patch_container.run_phase = AsyncMock(side_effect=_run_phase)

    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")
    _make_scenario(sdir, "lambda-b")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path)

    async def _drive():
        job = await ScenarioJob.create(config, _fake_creds(), account_manager=am)
        run_task = asyncio.ensure_future(job.run(ScenarioPhase.DEPLOY))
        await both_parked.wait()
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        return job

    job = asyncio.run(_drive())

    # gather re-raised on cancel, but the aggregate was rebuilt from the tasks:
    # both trials are recorded and attributed to the cancellation, not dropped.
    persisted = _json.loads((job.paths.job_dir / "result.json").read_text())
    assert len(persisted["trial_results"]) == 2
    assert persisted["n_failed"] == 2
    assert all(
        tr["exception_info"]["exception_type"] == "CancelledError"
        for tr in persisted["trial_results"]
    )


def test_task_outcomes_maps_result_exception_and_cancellation():
    """`_task_outcomes` reads each done task as gather(return_exceptions=True) would.

    Directly pins the three branches — success → value, failure → the exception,
    cancellation → a CancelledError value — including the cancelled-task case the
    Ctrl+C fix hinges on (where .result()/.exception() would otherwise raise).
    """
    from aws_bench.scenario.job import _task_outcomes

    async def _raise(exc):
        raise exc

    async def _build():
        done = asyncio.ensure_future(asyncio.sleep(0, result="ok"))
        raised = asyncio.ensure_future(_raise(ValueError("boom")))
        cancelled = asyncio.ensure_future(asyncio.sleep(3600))
        await asyncio.gather(done, raised, return_exceptions=True)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        return _task_outcomes([done, raised, cancelled])

    outcomes = asyncio.run(_build())
    assert outcomes[0] == "ok"
    assert isinstance(outcomes[1], ValueError)
    assert isinstance(outcomes[2], asyncio.CancelledError)


@pytest.mark.parametrize(
    "persist_error",
    [
        OSError("disk full"),  # a disk write failure
        RuntimeError("write blew up"),  # any non-OSError must be swallowed too
    ],
)
def test_persist_failure_during_cancel_does_not_mask_cancellation(
    tmp_path, patch_container, persist_error, caplog
):
    """No result-persist failure during cancel may replace the CancelledError.

    Both the job's and the trials' ``_persist_result`` run in a ``finally`` while
    the cancel unwinds. Every ``write_text`` here is made to fail; the guards must
    swallow it (logging) so the run still aborts as a cancel, with no result file
    written. (The serialization-failure trigger is covered at the trial level by
    ``test_run_cancel_not_masked_by_serialization_failure``.)
    """
    parked = asyncio.Event()

    async def _run_phase(*_args, **_kwargs):
        parked.set()
        await asyncio.Event().wait()  # block forever; the cancel unwinds it

    patch_container.run_phase = AsyncMock(side_effect=_run_phase)

    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path)

    async def _drive():
        job = await ScenarioJob.create(config, _fake_creds(), account_manager=am)
        run_task = asyncio.ensure_future(job.run(ScenarioPhase.DEPLOY))
        await parked.wait()
        # Fail every persist write (job- and trial-level) during the unwind.
        with patch.object(type(job.paths.result_path), "write_text", side_effect=persist_error):
            run_task.cancel()
            # The cancellation wins; the persist errors are swallowed.
            with pytest.raises(asyncio.CancelledError):
                await run_task
        return job

    with caplog.at_level(logging.WARNING, logger="aws_bench.scenario.job"):
        job = asyncio.run(_drive())

    # The guard reached and attempted the write, then swallowed-and-logged it; the
    # cancel propagated and no result file was written (not a masked error).
    assert any("Could not persist job result" in r.message for r in caplog.records)
    assert not (job.paths.job_dir / "result.json").exists()


def test_run_writes_job_log(tmp_path):
    """job.log is written when the caller wraps run() in a file_logging span.

    ScenarioJob.run() no longer opens its own job.log handler — the CLI caller
    (``_run_job_phase``) owns it. This exercises that contract: under the span,
    a run's log lines land in <job_dir>/job.log.
    """
    from aws_bench.logging.logger import file_logging

    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path)

    job = asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))
    log_path = job.paths.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with file_logging(log_path):
        asyncio.run(job.run(ScenarioPhase.DEPLOY))

    assert log_path.is_file()
    assert log_path.stat().st_size > 0


def test_create_aggregates_malformed_toml_into_discovery_error(tmp_path):
    """End-to-end: malformed scenario.toml aggregates into ScenarioDiscoveryError."""
    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "good")

    bad = sdir / "bad-toml"
    bad.mkdir()
    (bad / "scenario.toml").write_text("this is not [valid toml\n")
    (bad / "scenario").mkdir()
    (bad / "scenario" / "Dockerfile").write_text("FROM alpine\n")
    (bad / "deploy").mkdir()
    (bad / "deploy" / "deploy.sh").write_text("#!/bin/sh\n")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path, scenarios_path=sdir)

    with pytest.raises(ScenarioDiscoveryError) as excinfo:
        asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))
    assert "bad-toml" in str(excinfo.value)


# -- _check_quota_sufficiency ---------------------------------------------


def _make_scenario_with_quota(
    sdir_root: Path,
    name: str,
    *,
    quotas: list[dict] | None = None,
) -> Scenario:
    """Create a scenario dir with optional quotas in scenario.toml."""
    sd = sdir_root / name
    sd.mkdir(parents=True, exist_ok=True)
    quota_block = ""
    if quotas:
        for q in quotas:
            quota_block += (
                "\n[[quotas]]\n"
                f'account_tag = "{q["account_tag"]}"\n'
                f'region = "{q["region"]}"\n'
                f'service_code = "{q["service_code"]}"\n'
                f'quota_code = "{q["quota_code"]}"\n'
                f"desired_value = {q['desired_value']}\n"
            )
    toml = (
        'schema_version = "1.0"\n\n'
        "[scenario]\n"
        f'name = "{name}"\n'
        'account_tags = ["PRIMARY"]\n'
        'regions = ["us-east-1"]\n'
        f"{quota_block}"
    )
    (sd / "scenario.toml").write_text(toml)
    (sd / "scenario").mkdir()
    (sd / "scenario" / "Dockerfile").write_text("FROM alpine\n")
    (sd / "deploy").mkdir()
    (sd / "deploy" / "deploy.sh").write_text("#!/bin/sh\nexit 0\n")
    return Scenario(sd)


def test_check_quota_sufficiency_passes_when_all_already_met(tmp_path):
    """No-op fast path when verify_quotas returns ALREADY_MET for everything."""
    scenario = _make_scenario_with_quota(
        tmp_path,
        "alpha",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "ec2",
                "quota_code": "L-1216C47A",
                "desired_value": 50,
            }
        ],
    )
    account_mappings = {"alpha": {"PRIMARY": "111111111111"}}

    cred_provider = MagicMock()
    qm = MagicMock()
    qm.verify_quotas.return_value = [
        QuotaIncreaseResult(
            service_code="ec2",
            quota_code="L-1216C47A",
            desired_value=50.0,
            status=QuotaStatus.ALREADY_MET,
        )
    ]

    with patch("aws_bench.scenario.job.QuotaManager", return_value=qm):
        asyncio.run(
            ScenarioJob._check_quota_sufficiency(
                [scenario], account_mappings, cred_provider, n_concurrent=4
            )
        )

    qm.verify_quotas.assert_called_once()
    config_arg = qm.verify_quotas.call_args.args[0]
    account_arg = qm.verify_quotas.call_args.args[1]
    assert isinstance(config_arg, QuotaConfiguration)
    assert config_arg.region == "us-east-1"
    assert len(config_arg.increases) == 1
    assert config_arg.increases[0].quota_code == "L-1216C47A"
    assert account_arg == "111111111111"


def test_check_quota_sufficiency_aggregates_unmet_quotas_across_scenarios(tmp_path):
    """Two scenarios, each with one insufficient quota -> one error listing both."""
    alpha = _make_scenario_with_quota(
        tmp_path,
        "alpha",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "ec2",
                "quota_code": "L-1216C47A",
                "desired_value": 50,
            }
        ],
    )
    beta = _make_scenario_with_quota(
        tmp_path,
        "beta",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "ec2",
                "quota_code": "L-1216C47B",
                "desired_value": 100,
            }
        ],
    )
    account_mappings = {
        "alpha": {"PRIMARY": "111111111111"},
        "beta": {"PRIMARY": "222222222222"},
    }
    cred_provider = MagicMock()

    def verify_quotas_per_account(config, account_id, role_name):
        desired = config.increases[0].desired_value
        return [
            QuotaIncreaseResult(
                service_code=config.increases[0].service_code,
                quota_code=config.increases[0].quota_code,
                desired_value=desired,
                status=QuotaStatus.ALREADY_PENDING,
                error_message=f"current=8.0, required={desired} (PENDING)",
            )
        ]

    qm = MagicMock()
    qm.verify_quotas.side_effect = verify_quotas_per_account

    with patch("aws_bench.scenario.job.QuotaManager", return_value=qm):
        with pytest.raises(InsufficientQuotaError) as excinfo:
            asyncio.run(
                ScenarioJob._check_quota_sufficiency(
                    [alpha, beta], account_mappings, cred_provider, n_concurrent=4
                )
            )

    failures = excinfo.value.failures
    assert len(failures) == 2
    by_scenario = {sf.scenario_name: sf for sf in failures}
    assert "alpha" in by_scenario and "beta" in by_scenario
    assert by_scenario["alpha"].account_id == "111111111111"
    assert by_scenario["beta"].account_id == "222222222222"
    assert by_scenario["alpha"].result.status == QuotaStatus.ALREADY_PENDING
    # Two scenarios, each with one (account_tag, region) quota group → two work
    # units → two verify_quotas calls.
    assert qm.verify_quotas.call_count == 2


def test_quota_retry_rechecks_once_after_stale_read():
    """A stale read (first check unmet, recheck met) passes after one retry.

    Models Service Quotas eventual consistency: the env-setup gate sees the
    quota as still-unmet on the first read, waits, and the recheck succeeds.
    `asyncio.sleep` is patched so the 60s delay does not slow the test.
    """
    attempts = {"n": 0}

    async def gate(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise InsufficientQuotaError([])
        return None

    with (
        patch.object(ScenarioJob, "_check_quota_sufficiency", side_effect=gate),
        patch("asyncio.sleep", new=AsyncMock()) as slept,
    ):
        asyncio.run(
            ScenarioJob._check_quota_sufficiency_with_retry([], {}, MagicMock(), n_concurrent=1)
        )

    assert attempts["n"] == 2
    assert slept.await_count == 1


def test_quota_retry_reraises_when_still_unmet_after_recheck():
    """A quota unmet on all retry attempts re-raises."""
    attempts = {"n": 0}

    async def gate(*args, **kwargs):
        attempts["n"] += 1
        raise InsufficientQuotaError([])

    with (
        patch.object(ScenarioJob, "_check_quota_sufficiency", side_effect=gate),
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        with pytest.raises(InsufficientQuotaError):
            asyncio.run(
                ScenarioJob._check_quota_sufficiency_with_retry([], {}, MagicMock(), n_concurrent=1)
            )

    assert attempts["n"] == 3


def test_check_quota_sufficiency_no_quotas_is_noop(tmp_path):
    """Scenario with empty [[quotas]] table — no verify_quotas calls at all."""
    scenario = _make_scenario_with_quota(tmp_path, "no-quotas", quotas=None)
    account_mappings = {"no-quotas": {"PRIMARY": "111111111111"}}

    cred_provider = MagicMock()
    qm = MagicMock()

    with patch("aws_bench.scenario.job.QuotaManager", return_value=qm):
        asyncio.run(
            ScenarioJob._check_quota_sufficiency(
                [scenario], account_mappings, cred_provider, n_concurrent=4
            )
        )

    qm.verify_quotas.assert_not_called()


def test_check_quota_sufficiency_groups_quotas_by_region(tmp_path):
    """One verify_quotas call per (account, region) pair.

    Two quotas in different regions of the same scenario -> two calls.
    """
    sd = tmp_path / "multi-region"
    sd.mkdir()
    (sd / "scenario.toml").write_text(
        'schema_version = "1.0"\n\n'
        "[scenario]\n"
        'name = "multi-region"\n'
        'account_tags = ["PRIMARY"]\n'
        'regions = ["us-east-1", "us-west-2"]\n\n'
        "[[quotas]]\n"
        'account_tag = "PRIMARY"\n'
        'region = "us-east-1"\n'
        'service_code = "ec2"\n'
        'quota_code = "L-1216C47A"\n'
        "desired_value = 50\n\n"
        "[[quotas]]\n"
        'account_tag = "PRIMARY"\n'
        'region = "us-west-2"\n'
        'service_code = "ec2"\n'
        'quota_code = "L-1216C47B"\n'
        "desired_value = 100\n"
    )
    (sd / "scenario").mkdir()
    (sd / "scenario" / "Dockerfile").write_text("FROM alpine\n")
    (sd / "deploy").mkdir()
    (sd / "deploy" / "deploy.sh").write_text("#!/bin/sh\nexit 0\n")
    scenario = Scenario(sd)
    account_mappings = {"multi-region": {"PRIMARY": "111111111111"}}

    cred_provider = MagicMock()
    qm = MagicMock()
    qm.verify_quotas.return_value = [
        QuotaIncreaseResult(
            service_code="ec2",
            quota_code="L-1216C47A",
            desired_value=50.0,
            status=QuotaStatus.ALREADY_MET,
        )
    ]

    with patch("aws_bench.scenario.job.QuotaManager", return_value=qm):
        asyncio.run(
            ScenarioJob._check_quota_sufficiency(
                [scenario], account_mappings, cred_provider, n_concurrent=4
            )
        )

    assert qm.verify_quotas.call_count == 2
    regions_called = sorted(call.args[0].region for call in qm.verify_quotas.call_args_list)
    assert regions_called == ["us-east-1", "us-west-2"]


def test_create_raises_insufficient_quota_error_when_quotas_below_required(tmp_path):
    """End-to-end: ScenarioJob.create runs the quota gate after account resolution."""
    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "alpha")

    am = _fake_account_manager({"PRIMARY": "111111111111"})
    config = _make_job_config(tmp_path)

    qm = MagicMock()
    qm.verify_quotas.return_value = [
        QuotaIncreaseResult(
            service_code="ec2",
            quota_code="L-1216C47A",
            desired_value=50.0,
            status=QuotaStatus.ALREADY_PENDING,
            error_message="current=8.0, required=50.0 (PENDING)",
        )
    ]

    sd = sdir / "alpha"
    (sd / "scenario.toml").write_text(
        'schema_version = "1.0"\n\n'
        "[scenario]\n"
        'name = "alpha"\n'
        'account_tags = ["PRIMARY"]\n'
        'regions = ["us-east-1"]\n\n'
        "[[quotas]]\n"
        'account_tag = "PRIMARY"\n'
        'region = "us-east-1"\n'
        'service_code = "ec2"\n'
        'quota_code = "L-1216C47A"\n'
        "desired_value = 50\n"
    )

    # create()'s gate retries once on a stale read; patch the wait so this
    # always-unmet case re-raises without the 60s sleep.
    with (
        patch("aws_bench.scenario.job.QuotaManager", return_value=qm),
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        with pytest.raises(InsufficientQuotaError) as excinfo:
            asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))

    assert len(excinfo.value.failures) == 1
    assert excinfo.value.failures[0].scenario_name == "alpha"
    assert excinfo.value.failures[0].account_id == "111111111111"


def test_create_passes_when_all_quotas_already_met(tmp_path):
    """Same setup but verify_quotas returns ALREADY_MET; create() succeeds."""
    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "alpha")
    sd = sdir / "alpha"
    (sd / "scenario.toml").write_text(
        'schema_version = "1.0"\n\n'
        "[scenario]\n"
        'name = "alpha"\n'
        'account_tags = ["PRIMARY"]\n'
        'regions = ["us-east-1"]\n\n'
        "[[quotas]]\n"
        'account_tag = "PRIMARY"\n'
        'region = "us-east-1"\n'
        'service_code = "ec2"\n'
        'quota_code = "L-1216C47A"\n'
        "desired_value = 50\n"
    )

    am = _fake_account_manager({"PRIMARY": "111111111111"})
    config = _make_job_config(tmp_path)

    qm = MagicMock()
    qm.verify_quotas.return_value = [
        QuotaIncreaseResult(
            service_code="ec2",
            quota_code="L-1216C47A",
            desired_value=50.0,
            status=QuotaStatus.ALREADY_MET,
        )
    ]

    with patch("aws_bench.scenario.job.QuotaManager", return_value=qm):
        job = asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))

    assert job.config.test_environment is not None
    assert job.config.test_environment.to_scenario_account_mappings() == {
        "alpha": {"PRIMARY": "111111111111"}
    }
    qm.verify_quotas.assert_called_once()


# -- init snapshot validation -------------------------------------------------


def test_deploy_fails_on_init_snapshot_mismatch(tmp_path, patch_container, mock_account_manager):
    """Setup refuses a mismatched baseline before the SCP attach and the deploy script."""
    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path)

    job = asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))

    mismatch = SnapshotRegionMismatchError("lambda-a", "111", ["us-east-1"], ["us-west-2"])
    with patch(
        "aws_bench.resource_management.snapshot.manager.SnapshotManager.validate_pre_setup_snapshot",
        side_effect=mismatch,
    ) as validate:
        result = asyncio.run(job.run(ScenarioPhase.DEPLOY))

    assert result.n_failed == 1
    assert not result.all_passed
    validate.assert_called_once_with("lambda-a", "111", ["us-east-1"])
    mock_account_manager.ensure_region_restriction_scp.assert_not_called()
    patch_container.run_phase.assert_not_awaited()
    failure = result.trial_results[0].exception_info
    assert failure is not None
    assert failure.exception_type == "SetupValidationError"
    assert str(mismatch) in failure.exception_message


def test_deploy_reports_corrupt_init_snapshot_under_its_own_type(tmp_path, patch_container):
    """A corrupt baseline surfaces under its own error type, not as a setup-validation failure."""
    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path)

    job = asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))

    with patch(
        "aws_bench.resource_management.snapshot.manager.SnapshotManager.validate_pre_setup_snapshot",
        side_effect=json.JSONDecodeError("corrupt baseline", "{", 0),
    ):
        result = asyncio.run(job.run(ScenarioPhase.DEPLOY))

    assert result.n_failed == 1
    failure = result.trial_results[0].exception_info
    assert failure is not None
    assert failure.exception_type == "JSONDecodeError"


def test_deploy_succeeds_if_init_snapshot_valid(tmp_path, patch_container):
    """Setup proceeds if the shared PRE_SETUP validation succeeds."""
    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path)

    job = asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))
    result = asyncio.run(job.run(ScenarioPhase.DEPLOY))

    assert result.n_succeeded == 1
    assert result.all_passed


# -- cleanup snapshot deletion ------------------------------------------------


def test_cleanup_deletes_snapshots_on_success(tmp_path, patch_container):
    """Successful cleanup should delete both PRE_SETUP and POST_SETUP snapshots."""
    sdir = tmp_path / "scenarios"
    sdir.mkdir()
    _make_scenario(sdir, "lambda-a")

    am = _fake_account_manager({"PRIMARY": "111"})
    config = _make_job_config(tmp_path)

    job = asyncio.run(ScenarioJob.create(config, _fake_creds(), account_manager=am))

    with (
        patch(
            "aws_bench.resource_management.snapshot.manager.SnapshotManager.delete_snapshot"
        ) as mock_delete,
        patch(
            "aws_bench.resource_management.snapshot.manager.SnapshotManager.snapshot_exists",
            return_value=True,
        ),
    ):
        result = asyncio.run(job.run(ScenarioPhase.CLEANUP))

    assert result.all_passed
    # Should have called delete for both PRE_SETUP and POST_SETUP
    assert mock_delete.call_count == 2
