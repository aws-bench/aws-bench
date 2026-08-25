"""Tests for aws_bench.scenario.trial.

Mocks the container layer and the credential provider so the lifecycle
(events, timings, exit-code propagation, exception capture, cancellation,
env-var injection) can be exercised without Docker or AWS.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_core import PydanticSerializationError

from aws_bench.resource_management.cleanup.models import (
    AccountCleanupResult,
    CleanupSummary,
    RegionResult,
)
from aws_bench.resource_management.reset.models import ResetResult
from aws_bench.resource_management.snapshot.models import SnapshotResult
from aws_bench.scenario.config import TrialEnvironmentConfig
from aws_bench.scenario.container import ExecResult
from aws_bench.scenario.events import ScenarioEvent, ScenarioPhase
from aws_bench.scenario.exceptions import CleanupFailedError, ResetFailedError
from aws_bench.scenario.job_config import ScenarioTrialConfig
from aws_bench.scenario.locator import ScenarioConfig
from aws_bench.scenario.results import ScenarioHookEvent
from aws_bench.scenario.scenario import Scenario
from aws_bench.scenario.trial import MANAGEMENT_ROLE_ENV_VAR, ScenarioTrial

VALID_TOML = """\
schema_version = "1.0"

[scenario]
name = "{name}"
account_tags = ["PRIMARY"]
regions = ["us-east-1"]
"""


def _make_scenario_dir(root: Path) -> Path:
    sd = root / "sc"
    sd.mkdir()
    (sd / "scenario.toml").write_text(VALID_TOML.format(name="sc"))
    (sd / "scenario").mkdir()
    (sd / "scenario" / "Dockerfile").write_text("FROM alpine\n")
    (sd / "deploy").mkdir()
    (sd / "deploy" / "deploy.sh").write_text("#!/bin/sh\n")
    return sd


@pytest.fixture
def fake_container():
    c = MagicMock()
    c.build = AsyncMock()
    c.run_phase = AsyncMock(return_value=ExecResult(exit_code=0, stdout="ok\n"))
    c.write_file = AsyncMock()
    # No download_logs — phase outputs land on the host via bind mount.
    # Model is_started like the real container: False until start(), back to
    # False on stop(), so the trial's "start only if not already started" guard
    # is exercised honestly.
    c.is_started = False

    async def _start():
        c.is_started = True

    async def _stop(**_kwargs):
        c.is_started = False

    c.start = AsyncMock(side_effect=_start)
    c.stop = AsyncMock(side_effect=_stop)
    return c


@pytest.fixture
def fake_creds():
    """Mocked CredentialProvider whose session yields stable env creds."""
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


@pytest.fixture(autouse=True)
def mock_resource_manager():
    """Mock ResourceManager methods to avoid AWS API calls during tests."""
    with (
        patch(
            "aws_bench.resource_management.manager.ResourceManager.snapshot_scenarios",
            new_callable=AsyncMock,
            return_value={"sc": [SnapshotResult(account_id="111111111111", success=True)]},
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
            "aws_bench.resource_management.manager.ResourceManager.sweep_scenario_residuals_by_name",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "aws_bench.resource_management.snapshot.manager.SnapshotManager.snapshot_exists",
            return_value=True,
        ),
    ):
        yield


@pytest.fixture(autouse=True)
def mock_account_manager():
    """Stub AccountManager so the trial's org calls never hit AWS.

    Yields the mock AccountManager instance (what ``AccountManager()`` returns)
    so tests can assert on ``ensure_region_restriction_scp``. The contamination
    seams default to no-ops: the async tag writes are awaitable, and the
    read defaults to "clean" so the DEPLOY gate never blocks by accident.
    """
    with patch("aws_bench.scenario.trial.AccountManager") as mock_cls:
        instance = mock_cls.return_value
        instance.mark_contaminated = AsyncMock()
        instance.clear_contaminated = AsyncMock()
        instance.get_contaminated_accounts.return_value = []
        yield instance


def _make_trial_config(scenario_dir: Path, output_dir: Path) -> ScenarioTrialConfig:
    return ScenarioTrialConfig(
        scenario=ScenarioConfig(name="sc", path=scenario_dir),
        trial_name="trial-0",
        output_dir=output_dir,
        environment=TrialEnvironmentConfig(),
        account_mapping={"PRIMARY": "111111111111"},
        ou_name="test-ou",
    )


def _build_trial(tmp_path, fake_container, fake_creds, *, on_event=None) -> ScenarioTrial:
    sd = _make_scenario_dir(tmp_path)
    trial = ScenarioTrial(
        config=_make_trial_config(sd, tmp_path / "out"),
        cred_provider=fake_creds,
        scenario=Scenario(sd),
        container=fake_container,
    )
    if on_event is not None:
        for event in ScenarioEvent:
            trial.add_hook(event, on_event)
    return trial


# -- happy path -----------------------------------------------------------


def test_run_happy_path_records_exit_code_and_timings(tmp_path, fake_container, fake_creds):
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert result.success
    assert result.exit_code == 0
    assert result.phase == ScenarioPhase.DEPLOY
    assert result.started_at is not None and result.finished_at is not None
    assert result.environment_setup is not None
    assert result.execute is not None
    fake_container.build.assert_awaited_once()
    fake_container.start.assert_awaited_once()
    fake_container.run_phase.assert_awaited_once()
    fake_container.stop.assert_awaited_once_with(delete=True)


def test_run_creates_trial_dir(tmp_path, fake_container, fake_creds):
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    asyncio.run(trial.run(ScenarioPhase.DEPLOY))
    assert trial.paths.trial_dir.is_dir()


def test_reset_redeploy_starts_container_when_reset_had_no_script(
    tmp_path, fake_container, fake_creds
):
    """A reset that deletes a stack must start the container before redeploying.

    The scenario has no reset.sh, so RESET skips the build/start branch. When the
    reset reports needs_redeploy, the redeploy runs DEPLOY in the container — which
    requires it to be started. Regression guard: before the fix this raised
    RuntimeError("Container has not been started") and left the stack deleted.
    """
    from aws_bench.resource_management.reset.models import ResetResult

    deleted = ResetResult(success=True, reason="stack deleted", needs_redeploy=True)
    with patch(
        "aws_bench.resource_management.manager.ResourceManager.reset_scenarios",
        new_callable=AsyncMock,
        return_value=[deleted],
    ):
        result = asyncio.run(trial_for_reset(tmp_path, fake_container, fake_creds))

    # The redeploy started the container and ran the DEPLOY script — no
    # "container not started" crash, and the stack-deleted result is marked redeployed.
    fake_container.start.assert_awaited_once()
    assert fake_container.run_phase.await_args.args[0] == ScenarioPhase.DEPLOY
    assert result.resource_results is not None
    assert result.resource_results["reset"][0].redeploy_succeeded is True
    # A successful redeploy leaves the trial passing — the post-redeploy reset
    # check must not re-raise on the (success=True) needs_redeploy result.
    assert result.success
    assert result.exception_info is None


async def trial_for_reset(tmp_path, fake_container, fake_creds):
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    return await trial.run(ScenarioPhase.RESET)


def test_reset_redeploy_retries_on_transient_failure(tmp_path, fake_container, fake_creds):
    """A transient redeploy failure after a successful reset is retried, not fatal.

    Reset's contract is "restore to baseline", which includes recreating the
    deleted stack. A transient deploy failure (e.g. the S3 Tables bucket-name
    deletion cooldown returning a 409) must be retried so the environment is
    actually restored — otherwise reset "succeeds" but leaves the env broken.
    First DEPLOY attempt fails (non-zero exit), second succeeds.
    """
    from aws_bench.resource_management.reset.models import ResetResult

    # run_phase: first DEPLOY call returns non-zero (transient), second returns 0.
    fake_container.run_phase = AsyncMock(
        side_effect=[
            ExecResult(exit_code=1, stdout="TablesBucket ... transitional state ... 409\n"),
            ExecResult(exit_code=0, stdout="ok\n"),
        ]
    )

    deleted = ResetResult(success=True, reason="stack deleted", needs_redeploy=True)
    with (
        patch(
            "aws_bench.resource_management.manager.ResourceManager.reset_scenarios",
            new_callable=AsyncMock,
            return_value=[deleted],
        ),
        # Don't actually sleep between retries.
        patch("aws_bench.scenario.trial.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = asyncio.run(trial_for_reset(tmp_path, fake_container, fake_creds))

    # The redeploy was retried (2 DEPLOY attempts) and ultimately succeeded.
    assert fake_container.run_phase.await_count == 2
    assert result.resource_results is not None
    assert result.resource_results["reset"][0].redeploy_succeeded is True


def test_reset_redeploy_fails_loud_after_exhausting_retries(tmp_path, fake_container, fake_creds):
    """A persistently-failing redeploy still fails loud after all retries.

    The retry handles transient failures; a genuine one must mark the reset
    result failed and re-raise (the env is not restored), not silently pass.
    """
    from aws_bench.resource_management.reset.models import ResetResult
    from aws_bench.scenario.trial import _REDEPLOY_MAX_ATTEMPTS

    fake_container.run_phase = AsyncMock(
        return_value=ExecResult(exit_code=1, stdout="persistent deploy failure\n")
    )

    deleted = ResetResult(success=True, reason="stack deleted", needs_redeploy=True)
    with (
        patch(
            "aws_bench.resource_management.manager.ResourceManager.reset_scenarios",
            new_callable=AsyncMock,
            return_value=[deleted],
        ),
        patch("aws_bench.scenario.trial.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = asyncio.run(trial_for_reset(tmp_path, fake_container, fake_creds))

    # All attempts were made, and the reset result is marked failed (fail loud).
    assert fake_container.run_phase.await_count == _REDEPLOY_MAX_ATTEMPTS
    assert result.resource_results is not None
    assert result.resource_results["reset"][0].redeploy_succeeded is False
    assert result.resource_results["reset"][0].success is False


# -- contamination tagging from reset outcome (SET failed / CLEAR succeeded) --


def test_run_reset_tags_failed_accounts_and_clears_succeeded(tmp_path, fake_container, fake_creds):
    """A failed reset flags its account; a succeeded reset clears its account."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    acct = MagicMock()
    acct.mark_contaminated = AsyncMock()
    acct.clear_contaminated = AsyncMock()
    # The trial builds one AccountManager at construction and reuses it; inject
    # this test's mock onto the already-built instance (patching the class now is
    # too late — construction already happened under _build_trial).
    trial._account_manager = acct
    with patch(
        "aws_bench.resource_management.manager.ResourceManager.reset_scenarios",
        new_callable=AsyncMock,
        return_value=[
            ResetResult(success=False, reason="drift", account_id="111"),
            ResetResult(success=True, reason="ok", account_id="222"),
        ],
    ):
        with pytest.raises(ResetFailedError):
            asyncio.run(trial._run_reset())

    acct.mark_contaminated.assert_awaited_once_with("111")
    acct.clear_contaminated.assert_awaited_once_with("222")


def test_run_reset_all_success_clears_all(tmp_path, fake_container, fake_creds):
    """Every account whose reset succeeded is cleared; none is flagged."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    acct = MagicMock()
    acct.mark_contaminated = AsyncMock()
    acct.clear_contaminated = AsyncMock()
    # The trial builds one AccountManager at construction and reuses it; inject
    # this test's mock onto the already-built instance (patching the class now is
    # too late — construction already happened under _build_trial).
    trial._account_manager = acct
    with patch(
        "aws_bench.resource_management.manager.ResourceManager.reset_scenarios",
        new_callable=AsyncMock,
        return_value=[ResetResult(success=True, reason="ok", account_id="111")],
    ):
        asyncio.run(trial._run_reset())  # no raise

    acct.mark_contaminated.assert_not_awaited()
    acct.clear_contaminated.assert_awaited_once_with("111")


def test_run_reset_tag_write_failure_does_not_mask_reset_error(
    tmp_path, fake_container, fake_creds
):
    """A flag-write failure on the SET path is logged, not raised; reset error wins."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    acct = MagicMock()
    acct.mark_contaminated = AsyncMock(side_effect=RuntimeError("org throttled"))
    acct.clear_contaminated = AsyncMock()
    trial._account_manager = acct
    with patch(
        "aws_bench.resource_management.manager.ResourceManager.reset_scenarios",
        new_callable=AsyncMock,
        return_value=[ResetResult(success=False, reason="drift", account_id="111")],
    ):
        with pytest.raises(ResetFailedError):  # NOT RuntimeError
            asyncio.run(trial._run_reset())
    acct.mark_contaminated.assert_awaited_once_with("111")


def test_run_reset_clear_failure_surfaces_when_reset_otherwise_succeeds(
    tmp_path, fake_container, fake_creds
):
    """A clean account whose flag can't be cleared surfaces the clear error."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    acct = MagicMock()
    acct.mark_contaminated = AsyncMock()
    acct.clear_contaminated = AsyncMock(side_effect=RuntimeError("org throttled"))
    trial._account_manager = acct
    with patch(
        "aws_bench.resource_management.manager.ResourceManager.reset_scenarios",
        new_callable=AsyncMock,
        return_value=[ResetResult(success=True, reason="ok", account_id="111")],
    ):
        with pytest.raises(RuntimeError, match="org throttled"):
            asyncio.run(trial._run_reset())


def test_run_reset_still_failing_reset_wins_over_clear_failure(
    tmp_path, fake_container, fake_creds
):
    """A still-failing reset raises ResetFailedError even if a sibling clear also fails."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    acct = MagicMock()
    acct.mark_contaminated = AsyncMock()
    acct.clear_contaminated = AsyncMock(side_effect=RuntimeError("org throttled"))
    trial._account_manager = acct
    with patch(
        "aws_bench.resource_management.manager.ResourceManager.reset_scenarios",
        new_callable=AsyncMock,
        return_value=[
            ResetResult(success=False, reason="drift", account_id="111"),
            ResetResult(success=True, reason="ok", account_id="222"),
        ],
    ):
        # ResetFailedError (account 111) outranks the clear failure (account 222).
        with pytest.raises(ResetFailedError):
            asyncio.run(trial._run_reset())


def test_reset_redeploy_failure_flags_account_then_raises(tmp_path, fake_container, fake_creds):
    """A failed redeploy leaves the account dirty: it must be flagged, then ResetFailedError.

    The redeploy runs after a needs_redeploy reset. If every attempt fails, the result is
    marked success=False and tagging runs BEFORE the raise — so the now-dirty account is
    flagged rather than silently left clean.
    """
    fake_container.run_phase = AsyncMock(
        return_value=ExecResult(exit_code=1, stdout="deploy boom\n")
    )
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    acct = MagicMock()
    acct.get_contaminated_accounts.return_value = []
    acct.mark_contaminated = AsyncMock()
    acct.clear_contaminated = AsyncMock()
    trial._account_manager = acct

    deleted = ResetResult(
        success=True, reason="stack deleted", needs_redeploy=True, account_id="111"
    )
    with (
        patch(
            "aws_bench.resource_management.manager.ResourceManager.reset_scenarios",
            new_callable=AsyncMock,
            return_value=[deleted],
        ),
        patch("aws_bench.scenario.trial.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = asyncio.run(trial.run(ScenarioPhase.RESET))

    # Redeploy exhausted retries -> result marked failed -> account flagged -> trial failed.
    acct.mark_contaminated.assert_awaited_once_with("111")
    acct.clear_contaminated.assert_not_awaited()
    assert not result.success
    assert result.exception_info is not None


def test_reset_redeploy_not_blocked_by_own_contamination(tmp_path, fake_container, fake_creds):
    """A recovery redeploy must not self-block on the flag the reset is clearing.

    Regression guard: the contamination gate lives in _run_phase_in_container's DEPLOY
    branch, which the post-reset redeploy reuses. When an operator re-runs env reset on
    an already-contaminated account and the reset deletes an un-revertable stack
    (needs_redeploy), the redeploy must proceed — not raise AccountContaminatedError on
    the tag that this very reset clears afterward.
    """
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    acct = MagicMock()
    # Account is still flagged (prior failed run); a naive gate would block the redeploy.
    acct.get_contaminated_accounts.return_value = ["111"]
    acct.mark_contaminated = AsyncMock()
    acct.clear_contaminated = AsyncMock()
    trial._account_manager = acct

    deleted = ResetResult(
        success=True, reason="stack deleted", needs_redeploy=True, account_id="111"
    )
    with patch(
        "aws_bench.resource_management.manager.ResourceManager.reset_scenarios",
        new_callable=AsyncMock,
        return_value=[deleted],
    ):
        result = asyncio.run(trial.run(ScenarioPhase.RESET))

    # Redeploy ran DEPLOY (not blocked), reset succeeded, and the flag was cleared.
    assert fake_container.run_phase.await_args.args[0] == ScenarioPhase.DEPLOY
    assert result.success
    assert result.exception_info is None
    acct.clear_contaminated.assert_awaited_once_with("111")


# -- baseline recapture is gated on a fully-clean reset -------------------


def test_baseline_recapture_allowed_only_when_all_clean():
    """Recapture is allowed only when every result is success and orphan-free."""
    clean = ResetResult(success=True, reason="ok", account_id="111")
    orphan = ResetResult(
        success=False,
        reason="orphan",
        account_id="222",
        unresolved_orphans={"AWS::EC2::Vpc": [{"Identifier": "vpc-1"}]},
    )
    failed = ResetResult(success=False, reason="drift", account_id="333")

    assert ScenarioTrial._baseline_recapture_allowed([clean]) is True
    assert ScenarioTrial._baseline_recapture_allowed([clean, orphan]) is False
    assert ScenarioTrial._baseline_recapture_allowed([clean, failed]) is False
    # A success=True result still blocks recapture if it carries orphans.
    still_orphan = ResetResult(
        success=True,
        reason="ok",
        account_id="444",
        unresolved_orphans={"AWS::EC2::Vpc": [{"Identifier": "vpc-2"}]},
    )
    assert ScenarioTrial._baseline_recapture_allowed([still_orphan]) is False


def test_orphan_identifiers_unions_across_results():
    """The orphan-id union pulls Identifier from every result's orphans."""
    a = ResetResult(
        success=False,
        reason="x",
        account_id="111",
        unresolved_orphans={"AWS::EC2::Vpc": [{"Identifier": "vpc-1"}]},
    )
    b = ResetResult(
        success=False,
        reason="y",
        account_id="222",
        unresolved_orphans={
            "AWS::EC2::Vpc": [{"Identifier": "vpc-2"}],
            "AWS::S3::Bucket": [{"Identifier": "b-1"}, {"NoId": "skip"}],
        },
    )
    clean = ResetResult(success=True, reason="ok", account_id="333")
    assert ScenarioTrial._orphan_identifiers([a, b, clean]) == {"vpc-1", "vpc-2", "b-1"}
    assert ScenarioTrial._orphan_identifiers([clean]) == set()


def test_run_reset_skips_recapture_when_an_account_has_orphans(
    tmp_path, fake_container, fake_creds
):
    """A orphan-carrying account blocks the scenario-wide baseline recapture.

    One account needs_redeploy (stack deleted), another carries unresolved
    orphans (success=False). Recapture would absorb the live orphan, so
    _run_snapshot must not run and the trial fails with ResetFailedError.
    """
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    acct = MagicMock()
    acct.mark_contaminated = AsyncMock()
    acct.clear_contaminated = AsyncMock()
    trial._account_manager = acct

    redeploy = ResetResult(
        success=True, reason="stack deleted", needs_redeploy=True, account_id="111"
    )
    orphan = ResetResult(
        success=False,
        reason="orphan",
        account_id="222",
        unresolved_orphans={"AWS::EC2::Vpc": [{"Identifier": "vpc-1"}]},
    )
    with (
        patch(
            "aws_bench.resource_management.manager.ResourceManager.reset_scenarios",
            new_callable=AsyncMock,
            return_value=[redeploy, orphan],
        ),
        patch.object(trial, "_redeploy_with_retry", new_callable=AsyncMock),
        patch.object(trial, "_run_snapshot", new_callable=AsyncMock) as snap,
    ):
        with pytest.raises(ResetFailedError):
            asyncio.run(trial._run_reset())

    snap.assert_not_awaited()


def test_run_reset_recaptures_when_all_clean(tmp_path, fake_container, fake_creds):
    """A clean reset with a needs_redeploy account recaptures the baseline once."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    acct = MagicMock()
    acct.mark_contaminated = AsyncMock()
    acct.clear_contaminated = AsyncMock()
    trial._account_manager = acct

    redeploy = ResetResult(
        success=True, reason="stack deleted", needs_redeploy=True, account_id="111"
    )
    with (
        patch(
            "aws_bench.resource_management.manager.ResourceManager.reset_scenarios",
            new_callable=AsyncMock,
            return_value=[redeploy],
        ),
        patch.object(trial, "_redeploy_with_retry", new_callable=AsyncMock),
        patch.object(trial, "_run_snapshot", new_callable=AsyncMock) as snap,
    ):
        asyncio.run(trial._run_reset())  # no raise

    snap.assert_awaited_once()


def test_run_passes_account_tag_into_phase_env(tmp_path, fake_container, fake_creds):
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    env_kwarg = fake_container.run_phase.await_args.kwargs["env"]
    assert env_kwarg["PRIMARY"] == "111111111111"
    assert env_kwarg[MANAGEMENT_ROLE_ENV_VAR] == "OrganizationAccountAccessRole"
    # AWS credentials are NOT injected as env vars: the container reads them via
    # ~/.aws/config credential_process (written by ScenarioContainer.start()).
    assert "AWS_ACCESS_KEY_ID" not in env_kwarg
    assert "AWS_SECRET_ACCESS_KEY" not in env_kwarg
    assert "AWS_SESSION_TOKEN" not in env_kwarg
    # aws-bench does NOT set AWS_PROFILE — scripts choose which profile to
    # use, since multi-account scenarios may switch between them.
    assert "AWS_PROFILE" not in env_kwarg


def test_run_merges_phase_env_over_creds(tmp_path, fake_container, fake_creds):
    """Per-phase env vars from scenario.toml [deploy].env are layered on top."""
    sd = _make_scenario_dir(tmp_path)
    # Override the default scenario.toml to add a [deploy] section.
    (sd / "scenario.toml").write_text(
        'schema_version = "1.0"\n\n'
        "[scenario]\n"
        'name = "sc"\n'
        'account_tags = ["PRIMARY"]\n'
        'regions = ["us-east-1"]\n\n'
        "[deploy]\n"
        "timeout_sec = 60.0\n"
        'env = {CUSTOM = "1"}\n'
    )
    cfg = ScenarioTrialConfig(
        scenario=ScenarioConfig(name="sc", path=sd),
        trial_name="trial-0",
        output_dir=tmp_path / "out",
        account_mapping={"PRIMARY": "111111111111"},
        ou_name="test-ou",
    )
    trial = ScenarioTrial(
        config=cfg,
        cred_provider=fake_creds,
        scenario=Scenario(sd),
        container=fake_container,
    )
    asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    env_kwarg = fake_container.run_phase.await_args.kwargs["env"]
    assert env_kwarg["CUSTOM"] == "1"
    assert env_kwarg["PRIMARY"] == "111111111111"


def test_run_applies_operator_overrides_before_build(tmp_path, fake_container, fake_creds):
    """Operator-side overrides (--cpus, etc.) are merged into the env passed to the container."""
    sd = _make_scenario_dir(tmp_path)
    # Override the default scenario.toml to add an [environment] section.
    (sd / "scenario.toml").write_text(
        'schema_version = "1.0"\n\n'
        "[scenario]\n"
        'name = "sc"\n'
        'account_tags = ["PRIMARY"]\n'
        'regions = ["us-east-1"]\n\n'
        "[environment]\n"
        "cpus = 1\n"
        "memory_mb = 1024\n"
    )
    materialized = Scenario(sd)
    cfg = ScenarioTrialConfig(
        scenario=ScenarioConfig(name="sc", path=sd),
        trial_name="trial-0",
        output_dir=tmp_path / "out",
        environment=TrialEnvironmentConfig(override_cpus=4, override_memory_mb=8192),
        account_mapping={"PRIMARY": "111"},
        ou_name="test-ou",
    )
    trial = ScenarioTrial(
        config=cfg,
        cred_provider=fake_creds,
        scenario=materialized,
        container=fake_container,
    )

    # author's config still has cpus=1 (operator overrides don't mutate it)
    assert materialized.manifest.environment.cpus == 1
    # but the merged env that was handed to the container has the overrides
    assert trial._merged_env.cpus == 4
    assert trial._merged_env.memory_mb == 8192


def test_run_no_delete_keeps_container(tmp_path, fake_container, fake_creds):
    sd = _make_scenario_dir(tmp_path)
    cfg = _make_trial_config(sd, tmp_path / "out")
    cfg.environment = TrialEnvironmentConfig(delete=False)
    trial = ScenarioTrial(
        config=cfg,
        cred_provider=fake_creds,
        scenario=Scenario(sd),
        container=fake_container,
    )
    asyncio.run(trial.run(ScenarioPhase.DEPLOY))
    fake_container.stop.assert_awaited_once_with(delete=False)


# -- failure paths --------------------------------------------------------


def test_run_phase_nonzero_exit_marks_failure(tmp_path, fake_container, fake_creds):
    """A non-zero phase exit sets the numeric code and raises NonZeroExitCodeError.

    The code is recorded on the result before the raise, the raise is caught
    by run()'s handler as exception_info, and exception.txt is written.
    """
    fake_container.run_phase = AsyncMock(return_value=ExecResult(exit_code=7, stdout="boom\n"))
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert result.exit_code == 7
    assert not result.success
    assert result.exception_info is not None
    assert result.exception_info.exception_type == "NonZeroExitCodeError"
    assert "deploy phase exited 7" in result.exception_info.exception_message
    exc_file = trial.paths.exception_path
    assert exc_file.is_file()
    assert "NonZeroExitCodeError" in exc_file.read_text()


def test_run_phase_timeout_marks_failure_with_named_error(tmp_path, fake_container, fake_creds):
    """A phase that outruns its budget records PhaseTimeoutError, not a bare error.

    Using the named error means the failure says which phase timed out and after
    how long, instead of a bare asyncio.TimeoutError.
    """
    fake_container.run_phase = AsyncMock(side_effect=TimeoutError())
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert not result.success
    assert result.exception_info is not None
    assert result.exception_info.exception_type == "PhaseTimeoutError"
    assert "deploy phase timed out" in result.exception_info.exception_message


def test_run_skips_snapshot_when_deploy_times_out(tmp_path, fake_container, fake_creds):
    """A deploy timeout must not baseline a half-deployed environment.

    A timeout raises before exit_code is assigned (it stays None), so the
    snapshot must be gated on the script-failure signal, not exit_code alone.
    """
    fake_container.run_phase = AsyncMock(side_effect=TimeoutError())
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    with patch(
        "aws_bench.resource_management.manager.ResourceManager.snapshot_scenarios",
        new_callable=AsyncMock,
    ) as snapshot:
        result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    snapshot.assert_not_awaited()
    assert result.exit_code is None
    assert result.exception_info is not None
    assert result.exception_info.exception_type == "PhaseTimeoutError"


def test_run_skips_snapshot_when_deploy_exits_nonzero(tmp_path, fake_container, fake_creds):
    """A nonzero deploy exit must not capture a baseline snapshot."""
    fake_container.run_phase = AsyncMock(return_value=ExecResult(exit_code=7, stdout="boom\n"))
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    with patch(
        "aws_bench.resource_management.manager.ResourceManager.snapshot_scenarios",
        new_callable=AsyncMock,
    ) as snapshot:
        result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    snapshot.assert_not_awaited()
    assert result.exit_code == 7


def test_run_captures_snapshot_on_deploy_success(tmp_path, fake_container, fake_creds):
    """A clean deploy captures the post-setup baseline snapshot."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    with patch(
        "aws_bench.resource_management.manager.ResourceManager.snapshot_scenarios",
        new_callable=AsyncMock,
        return_value={"sc": []},
    ) as snapshot:
        result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    snapshot.assert_awaited_once()
    assert result.success


def test_run_fails_deploy_on_baseline_capture_failure(tmp_path, fake_container, fake_creds):
    """A failed post-setup baseline fails the deploy trial."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    with patch(
        "aws_bench.resource_management.manager.ResourceManager.snapshot_scenarios",
        new_callable=AsyncMock,
        return_value={
            "sc": [
                SnapshotResult(
                    account_id="111111111111",
                    success=False,
                    error_message="scan blew up",
                )
            ]
        },
    ):
        result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert not result.success
    assert result.exception_info is not None
    assert result.exception_info.exception_type == "SnapshotFailedError"
    # The failing account and its error are surfaced in the recorded exception.
    assert "111111111111" in result.exception_info.exception_message


def test_record_exception_keeps_first_failure(tmp_path, fake_container, fake_creds):
    """The first recorded exception wins; a later one does not overwrite it.

    A failed phase records NonZeroExitCodeError; a subsequent failure (such
    as one raised during cleanup) must not clobber the root-cause traceback.
    """
    from aws_bench.scenario.exceptions import NonZeroExitCodeError

    trial = _build_trial(tmp_path, fake_container, fake_creds)

    # Built and recorded OUTSIDE an active except block: the persisted
    # traceback must come from the exception object, not sys.exc_info()
    # (which would yield "NoneType: None" here).
    try:
        raise NonZeroExitCodeError(phase="deploy", exit_code=7, stdout="boom\n")
    except NonZeroExitCodeError as exc:
        recorded = exc
    trial._record_exception(recorded)
    first_info = trial.result.exception_info
    assert first_info is not None
    assert first_info.exception_type == "NonZeroExitCodeError"
    first_text = trial.paths.exception_path.read_text()
    # The real traceback is persisted, not a degenerate "NoneType: None".
    assert "NonZeroExitCodeError" in first_text
    assert "NoneType: None" not in first_text

    trial._record_exception(RuntimeError("cleanup blew up afterwards"))

    # Same recorded info object and same persisted traceback — not overwritten.
    assert trial.result.exception_info is first_info
    assert trial.paths.exception_path.read_text() == first_text
    assert "cleanup blew up" not in trial.paths.exception_path.read_text()


def test_run_cancel_records_exception_info(tmp_path, fake_container, fake_creds):
    """Cancellation records exception_info before re-raising."""
    fake_container.start = AsyncMock(side_effect=asyncio.CancelledError())
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert trial.result.exception_info is not None
    assert "CancelledError" in trial.result.exception_info.exception_type
    assert trial.paths.exception_path.is_file()


def _fail_cancel_path_writes(error):
    """Patch target: raise ``error`` on the cancel-path writes, pass others through.

    Leaves the pre-run config.json write alone — it runs before any cancel and
    legitimately surfaces its own errors.
    """
    real_write_text = Path.write_text

    def _write(self, *args, **kwargs):
        if self.name in ("result.json", "exception.txt"):
            raise error
        return real_write_text(self, *args, **kwargs)

    return patch.object(Path, "write_text", _write)


def test_run_cancel_not_masked_by_write_failure(tmp_path, fake_container, fake_creds, caplog):
    """A failed result/traceback WRITE in the cancel finally must not mask the cancel.

    ``_persist_result`` / ``_record_exception`` run in ``run``'s finally while the
    cancel propagates; a disk write failure is swallowed (and logged) so the trial
    still aborts as a cancel, with the in-memory record intact.
    """
    fake_container.start = AsyncMock(side_effect=asyncio.CancelledError())
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    with caplog.at_level(logging.WARNING, logger="aws_bench.scenario.trial"):
        with _fail_cancel_path_writes(OSError("disk full")):
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    # Both guarded writes were reached, failed, and logged — not silently dropped.
    assert any("Could not persist result" in r.message for r in caplog.records)
    assert any("Could not persist exception traceback" in r.message for r in caplog.records)
    # The in-memory record survived the write failure.
    assert trial.result.exception_info is not None
    assert "CancelledError" in trial.result.exception_info.exception_type


def test_run_cancel_not_masked_by_serialization_failure(tmp_path, fake_container, fake_creds):
    """A model_dump_json failure in the cancel finally must not mask the cancel.

    ``resource_results`` accepts arbitrary objects (the result model is
    ``arbitrary_types_allowed``), so ``model_dump_json`` can raise
    ``PydanticSerializationError`` at persist time. That serialization failure is
    swallowed so the cancel still propagates — the production non-OSError trigger,
    not a stand-in.
    """
    fake_container.start = AsyncMock(side_effect=asyncio.CancelledError())
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    with patch.object(
        type(trial.result),
        "model_dump_json",
        side_effect=PydanticSerializationError("unserializable"),
    ):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(trial.run(ScenarioPhase.DEPLOY))


def test_run_build_failure_captured_as_exception(tmp_path, fake_container, fake_creds):
    fake_container.build = AsyncMock(side_effect=RuntimeError("docker daemon down"))
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert not result.success
    assert result.exception_info is not None
    assert result.exception_info.exception_type == "RuntimeError"
    assert "docker daemon down" in result.exception_info.exception_message
    # Phase script never ran.
    fake_container.run_phase.assert_not_called()
    # But cleanup ran.
    fake_container.stop.assert_awaited_once()
    # The traceback is persisted for any failure path, not just non-zero exits.
    assert "docker daemon down" in trial.paths.exception_path.read_text()


def test_run_succeeds_without_writing_exception_file(tmp_path, fake_container, fake_creds):
    """A clean run leaves no exception.txt behind."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert result.success
    assert not trial.paths.exception_path.exists()


def test_run_phase_exec_failure_captured_as_exception(tmp_path, fake_container, fake_creds):
    fake_container.run_phase = AsyncMock(side_effect=asyncio.TimeoutError("script timed out"))
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert not result.success
    assert result.exception_info is not None
    assert "TimeoutError" in result.exception_info.exception_type


def test_run_cleanup_runs_even_when_execute_raises(tmp_path, fake_container, fake_creds):
    fake_container.run_phase = AsyncMock(side_effect=RuntimeError("oops"))
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    asyncio.run(trial.run(ScenarioPhase.DEPLOY))
    fake_container.stop.assert_awaited_once()


def test_run_cleanup_failure_recorded_when_phase_succeeds(tmp_path, fake_container, fake_creds):
    """A clean phase followed by a failing container stop records the cleanup error.

    With no earlier lifecycle failure to claim it, the cleanup exception
    becomes exception_info and is persisted to exception.txt.
    """
    fake_container.stop = AsyncMock(side_effect=RuntimeError("docker stop hung"))
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert result.exit_code == 0  # the phase itself succeeded
    assert not result.success  # but the recorded cleanup error fails the trial
    assert result.exception_info is not None
    assert result.exception_info.exception_type == "RuntimeError"
    assert "docker stop hung" in result.exception_info.exception_message
    assert "docker stop hung" in trial.paths.exception_path.read_text()


def test_run_cleanup_cancellation_recorded_and_result_persisted(
    tmp_path, fake_container, fake_creds
):
    """A cancellation during container stop is recorded, and the trial still persists.

    Container stop runs under asyncio.shield in the run() finally block. If the
    stop is cancelled anyway (shield is best-effort), the cancellation is recorded
    on the result rather than re-raised, so the finally still persists the result
    and emits END. Here the phase itself succeeded, so the cleanup cancellation is
    the recorded cause and fails the trial.
    """
    fake_container.stop = AsyncMock(side_effect=asyncio.CancelledError())
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert result.exit_code == 0  # the phase itself succeeded
    assert not result.success  # but the recorded cancellation fails the trial
    assert result.exception_info is not None
    assert result.exception_info.exception_type == "CancelledError"
    assert result.finished_at is not None  # the finally still persisted the result
    assert trial.paths.result_path.exists()


def _cleanup_result(account_id: str, *, orphans: int) -> AccountCleanupResult:
    """Build an AccountCleanupResult whose summary carries ``orphans`` orphaned resources."""
    orphaned = {"AWS::S3::Bucket": [f"bucket-{i}" for i in range(orphans)]} if orphans else {}
    return AccountCleanupResult(
        account_id=account_id,
        summary=CleanupSummary(
            regions=[RegionResult(region="us-east-1", stacks_found=1, stacks_deleted=1)],
            orphaned_resources=orphaned,
        ),
        error=None,
    )


def test_run_cleanup_fails_when_orphans_remain(tmp_path, fake_container, fake_creds):
    """Cleanup that leaves orphaned resources fails the trial via CleanupFailedError."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    with patch(
        "aws_bench.resource_management.manager.ResourceManager.cleanup_scenarios_by_name",
        new_callable=AsyncMock,
        return_value=[_cleanup_result("111111111111", orphans=2)],
    ):
        result = asyncio.run(trial.run(ScenarioPhase.CLEANUP))

    assert not result.success
    assert result.exception_info is not None
    assert result.exception_info.exception_type == "CleanupFailedError"
    # The failure names how many orphaned resources remain.
    assert "2 orphaned resource(s) remain" in result.exception_info.exception_message
    # Results are stored before the raise, so the artifact keeps orphan data.
    assert result.resource_results is not None
    assert result.resource_results["cleanup"][0].summary.total_orphaned == 2


def test_run_cleanup_succeeds_when_no_orphans(tmp_path, fake_container, fake_creds):
    """Cleanup with zero orphaned resources does not fail the trial."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    with patch(
        "aws_bench.resource_management.manager.ResourceManager.cleanup_scenarios_by_name",
        new_callable=AsyncMock,
        return_value=[_cleanup_result("111111111111", orphans=0)],
    ):
        result = asyncio.run(trial.run(ScenarioPhase.CLEANUP))

    assert result.success
    assert result.exception_info is None


def test_run_cleanup_fails_when_scan_incomplete_despite_zero_orphans(
    tmp_path, fake_container, fake_creds
):
    """An incomplete orphan scan fails cleanup with zero orphans and keeps the baselines.

    A region whose post-cleanup scan raised (e.g. the discovery Lambda exhausted its
    retries) leaves total_orphaned at 0 while the account was never verified. Gating on
    orphan count alone would pass the run and then delete the baselines a later
    verify/reset depends on — the verdict must gate on is_clean (which folds in
    scan_incomplete).
    """
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    incomplete = AccountCleanupResult(
        account_id="111111111111",
        summary=CleanupSummary(
            regions=[RegionResult(region="us-east-1", stacks_found=1, stacks_deleted=1)],
            orphaned_resources={},
            scan_incomplete=True,
        ),
        error=None,
    )
    with (
        patch(
            "aws_bench.resource_management.manager.ResourceManager.cleanup_scenarios_by_name",
            new_callable=AsyncMock,
            return_value=[incomplete],
        ),
        patch.object(
            trial, "_delete_snapshots_after_cleanup", new_callable=AsyncMock
        ) as mock_delete,
    ):
        result = asyncio.run(trial.run(ScenarioPhase.CLEANUP))

    assert not result.success
    assert result.exception_info is not None
    assert result.exception_info.exception_type == "CleanupFailedError"
    assert "scan incomplete" in result.exception_info.exception_message
    # The baselines must survive an unverified cleanup.
    mock_delete.assert_not_called()


def test_run_cleanup_fails_when_summary_missing(tmp_path, fake_container, fake_creds):
    """A result with neither a summary nor an error still fails the trial.

    Mirrors display_cleanup_results, which reports such an account as FAILED;
    a missing summary cannot attest the account is clean.
    """
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    no_summary = AccountCleanupResult(account_id="111111111111", summary=None, error=None)
    with patch(
        "aws_bench.resource_management.manager.ResourceManager.cleanup_scenarios_by_name",
        new_callable=AsyncMock,
        return_value=[no_summary],
    ):
        result = asyncio.run(trial.run(ScenarioPhase.CLEANUP))

    assert not result.success
    assert result.exception_info is not None
    assert result.exception_info.exception_type == "CleanupFailedError"
    assert "no summary" in result.exception_info.exception_message


def test_run_cleanup_clears_contamination_on_clean(
    tmp_path, fake_container, fake_creds, mock_account_manager
):
    """A clean cleanup removes the contamination flag from each cleaned account."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    with (
        patch(
            "aws_bench.resource_management.manager.ResourceManager.cleanup_scenarios_by_name",
            new_callable=AsyncMock,
            return_value=[_cleanup_result("111111111111", orphans=0)],
        ),
        patch.object(trial, "_delete_snapshots_after_cleanup", new_callable=AsyncMock),
    ):
        asyncio.run(trial._run_cleanup())

    mock_account_manager.clear_contaminated.assert_awaited_once_with("111111111111")


def test_run_cleanup_keeps_snapshots_when_script_failed(
    tmp_path, fake_container, fake_creds, mock_account_manager
):
    """A cleanup.sh failure must NOT delete the baselines, even when framework cleanup is clean.

    Deleting them on a failed script strands a re-run with no baseline to diff against.
    """
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    with (
        patch(
            "aws_bench.resource_management.manager.ResourceManager.cleanup_scenarios_by_name",
            new_callable=AsyncMock,
            return_value=[_cleanup_result("111111111111", orphans=0)],
        ),
        patch.object(trial, "_delete_snapshots_after_cleanup", new_callable=AsyncMock) as del_snap,
    ):
        asyncio.run(trial._run_cleanup(script_failed=True))

    del_snap.assert_not_awaited()
    # A failed script is not a clean cleanup, so the contamination flag stays set too.
    mock_account_manager.clear_contaminated.assert_not_awaited()


def test_run_cleanup_deletes_snapshots_when_script_succeeded(tmp_path, fake_container, fake_creds):
    """The default path (script succeeded, framework clean) still deletes the baselines."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    with (
        patch(
            "aws_bench.resource_management.manager.ResourceManager.cleanup_scenarios_by_name",
            new_callable=AsyncMock,
            return_value=[_cleanup_result("111111111111", orphans=0)],
        ),
        patch.object(trial, "_delete_snapshots_after_cleanup", new_callable=AsyncMock) as del_snap,
    ):
        asyncio.run(trial._run_cleanup(script_failed=False))

    del_snap.assert_awaited_once()


def test_run_cleanup_not_clean_does_not_clear(
    tmp_path, fake_container, fake_creds, mock_account_manager
):
    """A not-clean cleanup raises and never clears the contamination flag."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    with patch(
        "aws_bench.resource_management.manager.ResourceManager.cleanup_scenarios_by_name",
        new_callable=AsyncMock,
        return_value=[_cleanup_result("111111111111", orphans=2)],
    ):
        with pytest.raises(CleanupFailedError):
            asyncio.run(trial._run_cleanup())

    mock_account_manager.clear_contaminated.assert_not_awaited()


def test_cleanup_runs_residual_sweep_before_cleanup_script(tmp_path, fake_container, fake_creds):
    """CLEANUP sweeps post-setup residuals before running cleanup.sh, then framework cleanup."""
    order: list[str] = []

    trial = _build_trial(tmp_path, fake_container, fake_creds)

    async def fake_sweep(*args, **kwargs):
        order.append("sweep")
        return []

    async def fake_run_phase(phase, **kwargs):
        order.append("cleanup.sh")

    with (
        patch(
            "aws_bench.resource_management.manager.ResourceManager.sweep_scenario_residuals_by_name",
            new_callable=AsyncMock,
            side_effect=fake_sweep,
        ) as mock_sweep,
        patch(
            "aws_bench.resource_management.manager.ResourceManager.cleanup_scenarios_by_name",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_cleanup,
        patch.object(trial._scenario, "has_phase_script", return_value=True),
        patch.object(trial, "_run_phase_in_container", side_effect=fake_run_phase),
    ):
        asyncio.run(trial.run(ScenarioPhase.CLEANUP))

    assert order == ["sweep", "cleanup.sh"]
    mock_sweep.assert_awaited_once()
    # Sweep is scoped to scenario regions and writes to its own pre-sweep subdir.
    assert mock_sweep.await_args is not None
    sweep_kwargs = mock_sweep.await_args.kwargs
    assert sweep_kwargs["regions"] == trial._scenario.manifest.scenario.regions
    assert sweep_kwargs["all_regions"] == trial.config.cleanup_all_regions
    assert sweep_kwargs["output_dir"].name == "pre-sweep"
    # Framework cleanup runs with the in-cleanup Phase-1 sweep disabled.
    assert mock_cleanup.await_args is not None
    assert mock_cleanup.await_args.kwargs["sweep_post_setup"] is False


def test_cleanup_continues_when_pre_cleanup_sweep_raises(tmp_path, fake_container, fake_creds):
    """A failing pre-cleanup sweep is best-effort: cleanup.sh and framework cleanup still run.

    The sweep runs before both, so an unhandled raise would otherwise skip stack deletion.
    """
    order: list[str] = []
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    async def boom_sweep(*args, **kwargs):
        raise RuntimeError("credential resolution failed")

    async def fake_run_phase(phase, **kwargs):
        order.append("cleanup.sh")

    async def fake_cleanup(*args, **kwargs):
        order.append("framework_cleanup")
        return []

    with (
        patch(
            "aws_bench.resource_management.manager.ResourceManager.sweep_scenario_residuals_by_name",
            new_callable=AsyncMock,
            side_effect=boom_sweep,
        ),
        patch(
            "aws_bench.resource_management.manager.ResourceManager.cleanup_scenarios_by_name",
            new_callable=AsyncMock,
            side_effect=fake_cleanup,
        ),
        patch.object(trial._scenario, "has_phase_script", return_value=True),
        patch.object(trial, "_run_phase_in_container", side_effect=fake_run_phase),
    ):
        result = asyncio.run(trial.run(ScenarioPhase.CLEANUP))

    # The sweep failure must NOT abort the phase: both later steps still ran, in order.
    assert order == ["cleanup.sh", "framework_cleanup"]
    # And the best-effort sweep failure is not recorded as the trial's exception.
    assert result.exception_info is None


def test_run_phase_failure_wins_over_cleanup_failure(tmp_path, fake_container, fake_creds):
    """A phase failure is preserved as the root cause even if cleanup also fails."""
    fake_container.run_phase = AsyncMock(return_value=ExecResult(exit_code=7, stdout="boom\n"))
    fake_container.stop = AsyncMock(side_effect=RuntimeError("docker stop hung"))
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    # The non-zero phase exit is the recorded cause, not the later cleanup error.
    assert result.exit_code == 7
    assert result.exception_info is not None
    assert result.exception_info.exception_type == "NonZeroExitCodeError"
    exc_text = trial.paths.exception_path.read_text()
    assert "NonZeroExitCodeError" in exc_text
    assert "docker stop hung" not in exc_text


# -- cancellation ---------------------------------------------------------


def test_run_cancel_emits_cancel_then_reraises(tmp_path, fake_container, fake_creds):
    fake_container.start = AsyncMock(side_effect=asyncio.CancelledError())
    captured: list[ScenarioHookEvent] = []

    async def cb(e: ScenarioHookEvent) -> None:
        captured.append(e)

    trial = _build_trial(tmp_path, fake_container, fake_creds, on_event=cb)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    fired = [e.event for e in captured]
    assert ScenarioEvent.CANCEL in fired
    assert ScenarioEvent.END in fired
    fake_container.stop.assert_awaited_once()


# -- lifecycle events -----------------------------------------------------


def test_run_emits_full_lifecycle_in_order(tmp_path, fake_container, fake_creds):
    captured: list[ScenarioHookEvent] = []

    async def cb(e: ScenarioHookEvent) -> None:
        captured.append(e)

    trial = _build_trial(tmp_path, fake_container, fake_creds, on_event=cb)
    asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert [e.event for e in captured] == [
        ScenarioEvent.START,
        ScenarioEvent.ENVIRONMENT_START,
        ScenarioEvent.PHASE_START,
        ScenarioEvent.END,
    ]
    end = captured[-1]
    assert end.result is not None
    assert end.result.exit_code == 0
    assert end.result.exception_info is None
    assert end.phase == ScenarioPhase.DEPLOY


def test_run_emits_exit_code_on_end_event_for_failed_phase(tmp_path, fake_container, fake_creds):
    fake_container.run_phase = AsyncMock(return_value=ExecResult(exit_code=2, stdout=""))
    captured: list[ScenarioHookEvent] = []

    async def cb(e: ScenarioHookEvent) -> None:
        captured.append(e)

    trial = _build_trial(tmp_path, fake_container, fake_creds, on_event=cb)
    asyncio.run(trial.run(ScenarioPhase.DEPLOY))
    end = next(e for e in captured if e.event == ScenarioEvent.END)
    assert end.result is not None
    assert end.result.exit_code == 2


def test_run_emits_error_on_end_event_for_exception(tmp_path, fake_container, fake_creds):
    fake_container.build = AsyncMock(side_effect=ValueError("boom"))
    captured: list[ScenarioHookEvent] = []

    async def cb(e: ScenarioHookEvent) -> None:
        captured.append(e)

    trial = _build_trial(tmp_path, fake_container, fake_creds, on_event=cb)
    asyncio.run(trial.run(ScenarioPhase.DEPLOY))
    end = next(e for e in captured if e.event == ScenarioEvent.END)
    assert end.result is not None
    assert end.result.exception_info is not None
    assert "boom" in end.result.exception_info.exception_message


def test_run_start_hook_failure_is_swallowed(tmp_path, fake_container, fake_creds):
    """A START-hook failure is logged but swallowed to preserve fault isolation.

    Hook errors cannot break the trial lifecycle. The trial succeeds despite
    the faulty hook, and cleanup still runs.
    """

    async def faulty(event: ScenarioHookEvent) -> None:
        if event.event is ScenarioEvent.START:
            raise RuntimeError("start hook boom")

    trial = _build_trial(tmp_path, fake_container, fake_creds, on_event=faulty)
    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    # Trial succeeds despite hook failure
    assert result.success
    assert result.exception_info is None
    fake_container.stop.assert_awaited_once()


def test_run_end_hook_failure_is_swallowed(tmp_path, fake_container, fake_creds):
    """An END-hook failure is logged but swallowed to preserve fault isolation."""

    async def faulty(event: ScenarioHookEvent) -> None:
        if event.event is ScenarioEvent.END:
            raise RuntimeError("end hook boom")

    trial = _build_trial(tmp_path, fake_container, fake_creds, on_event=faulty)
    # Hook failure is swallowed, trial completes normally
    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))
    assert result.success


def test_run_persists_config_and_result_before_end(tmp_path, fake_container, fake_creds):
    """A finished trial writes config.json and result.json (valid JSON)."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert trial.paths.config_path.is_file()
    assert trial.paths.result_path.is_file()
    json.loads(trial.paths.config_path.read_text())
    persisted = json.loads(trial.paths.result_path.read_text())
    assert persisted["exit_code"] == 0


def test_run_persists_result_even_when_end_hook_raises(tmp_path, fake_container, fake_creds):
    """result.json is written before END is emitted, so a failing END hook cannot lose it.

    Hook failures are swallowed, so the trial completes normally but the result
    was persisted before the hook ran.
    """

    async def faulty(event: ScenarioHookEvent) -> None:
        if event.event is ScenarioEvent.END:
            raise RuntimeError("end hook boom")

    trial = _build_trial(tmp_path, fake_container, fake_creds, on_event=faulty)
    # Hook failure is swallowed, trial completes normally
    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))
    assert result.success

    # Persisted before the END emit, so the record exists and is valid.
    assert trial.paths.result_path.is_file()
    assert json.loads(trial.paths.result_path.read_text())["exit_code"] == 0


# -- per-trial logging ----------------------------------------------------


def test_run_writes_trial_log(tmp_path, fake_container, fake_creds):
    """trial.run(phase) writes <trial_dir>/trial.log via file_logging."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert trial.paths.log_path.is_file()
    assert trial.paths.log_path.stat().st_size > 0


# -- region-restriction SCP (applied before deploy.sh) -------------------


def test_deploy_applies_region_restriction_scp(
    tmp_path, fake_container, fake_creds, mock_account_manager
):
    """Deploy locks the scenario's accounts to its declared regions.

    Applied for exactly the accounts this trial uses, with the scenario's
    declared regions.
    """
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert result.success
    mock_account_manager.ensure_region_restriction_scp.assert_called_once_with(
        "sc", ["us-east-1"], ["111111111111"]
    )


def test_region_restriction_scp_applied_before_deploy_script(
    tmp_path, fake_container, fake_creds, mock_account_manager
):
    """The SCP is applied before deploy.sh runs, so a failing script still gets it.

    The guardrail goes on first, so a non-zero script exit afterward neither
    undoes nor skips the SCP — out-of-region actions were already denied.
    """
    fake_container.run_phase = AsyncMock(return_value=ExecResult(exit_code=7, stdout="boom\n"))
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert not result.success
    mock_account_manager.ensure_region_restriction_scp.assert_called_once_with(
        "sc", ["us-east-1"], ["111111111111"]
    )


def test_scp_failure_aborts_deploy_before_script(
    tmp_path, fake_container, fake_creds, mock_account_manager
):
    """Fail-closed: if the guardrail can't be applied, deploy.sh never runs."""
    mock_account_manager.ensure_region_restriction_scp.side_effect = RuntimeError("scp boom")
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert not result.success
    fake_container.run_phase.assert_not_called()
    assert result.exception_info is not None
    assert "scp boom" in result.exception_info.exception_message


def test_non_deploy_phase_skips_region_restriction_scp(
    tmp_path, fake_container, fake_creds, mock_account_manager
):
    """Region restriction is a deploy-time action; recovery phases never apply it."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    asyncio.run(trial.run(ScenarioPhase.VERIFY))

    mock_account_manager.ensure_region_restriction_scp.assert_not_called()


# -- contamination gate on DEPLOY (env setup) -----------------------------


def test_run_deploy_blocks_on_contaminated_account(
    tmp_path, fake_container, fake_creds, mock_account_manager
):
    """DEPLOY on a contaminated account is refused; the error lands in exception_info."""
    mock_account_manager.get_contaminated_accounts.return_value = ["111111111111"]
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert not result.success
    assert result.exception_info is not None
    assert result.exception_info.exception_type == "AccountContaminatedError"
    # The gate fires before deploy.sh runs.
    fake_container.run_phase.assert_not_called()


def test_run_reset_not_blocked_by_contamination(
    tmp_path, fake_container, fake_creds, mock_account_manager
):
    """A contaminated account never blocks RESET — reset is the recovery path."""
    mock_account_manager.get_contaminated_accounts.return_value = ["111111111111"]
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    # Reset must proceed (recovery path); stub resource-management to a no-op.
    with patch.object(trial, "_run_resource_management", new_callable=AsyncMock):
        result = asyncio.run(trial.run(ScenarioPhase.RESET))

    # Not blocked: no AccountContaminatedError recorded.
    assert (
        result.exception_info is None
        or result.exception_info.exception_type != "AccountContaminatedError"
    )


def test_run_deploy_not_gated_when_accounts_clean(
    tmp_path, fake_container, fake_creds, mock_account_manager
):
    """A clean account passes the gate; DEPLOY proceeds and the read is scoped to it."""
    mock_account_manager.get_contaminated_accounts.return_value = []
    trial = _build_trial(tmp_path, fake_container, fake_creds)

    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert result.success
    mock_account_manager.get_contaminated_accounts.assert_called_once_with(["111111111111"])


# -- stale changeset cleanup ------------------------------------------------


def test_deploy_cleans_stale_changesets(tmp_path, fake_container, fake_creds):
    """Deploy deletes cdk-deploy-change-set from active stacks before running deploy.sh."""
    mock_cfn = MagicMock()
    mock_cfn.get_paginator.return_value.paginate.return_value = [
        {
            "StackSummaries": [
                {"StackName": "my-stack", "StackStatus": "CREATE_COMPLETE"},
            ]
        }
    ]
    mock_cfn.exceptions.ChangeSetNotFoundException = type("ChangeSetNotFound", (Exception,), {})
    mock_session = MagicMock()
    mock_session.client.return_value = mock_cfn
    fake_creds.get_session_for_account.return_value = mock_session

    trial = _build_trial(tmp_path, fake_container, fake_creds)
    asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    mock_cfn.delete_change_set.assert_called_once_with(
        ChangeSetName="cdk-deploy-change-set",
        StackName="my-stack",
    )


def test_deploy_deletes_review_in_progress_stacks(tmp_path, fake_container, fake_creds):
    """Deploy deletes stacks stuck in REVIEW_IN_PROGRESS."""
    mock_cfn = MagicMock()
    mock_cfn.get_paginator.return_value.paginate.return_value = [
        {
            "StackSummaries": [
                {"StackName": "stale-stack", "StackStatus": "REVIEW_IN_PROGRESS"},
            ]
        }
    ]
    mock_session = MagicMock()
    mock_session.client.return_value = mock_cfn
    fake_creds.get_session_for_account.return_value = mock_session

    trial = _build_trial(tmp_path, fake_container, fake_creds)
    asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    mock_cfn.delete_stack.assert_called_once_with(StackName="stale-stack")
    mock_cfn.delete_change_set.assert_not_called()


def test_changeset_cleanup_failure_does_not_block_deploy(tmp_path, fake_container, fake_creds):
    """Changeset cleanup is best-effort; failures don't prevent the deploy."""
    fake_creds.get_session_for_account.side_effect = RuntimeError("creds boom")

    trial = _build_trial(tmp_path, fake_container, fake_creds)
    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    # Deploy still ran and succeeded despite cleanup failure
    assert result.success
    fake_container.run_phase.assert_called_once()


def test_non_deploy_phase_skips_changeset_cleanup(tmp_path, fake_container, fake_creds):
    """Changeset cleanup only runs for DEPLOY, not verify/cleanup/reset."""
    trial = _build_trial(tmp_path, fake_container, fake_creds)
    asyncio.run(trial.run(ScenarioPhase.VERIFY))

    fake_creds.get_session_for_account.assert_not_called()


def test_changeset_cleanup_credential_failure_continues_to_next_account(
    tmp_path, fake_container, fake_creds
):
    """If get_session_for_account fails for one account, others still get cleaned."""
    mock_cfn = MagicMock()
    mock_cfn.get_paginator.return_value.paginate.return_value = [
        {"StackSummaries": [{"StackName": "s", "StackStatus": "CREATE_COMPLETE"}]}
    ]
    mock_session = MagicMock()
    mock_session.client.return_value = mock_cfn

    # First call fails, second succeeds
    fake_creds.get_session_for_account.side_effect = [
        RuntimeError("creds boom"),
        mock_session,
    ]

    # Two accounts in the mapping
    sd = _make_scenario_dir(tmp_path)
    config = _make_trial_config(sd, tmp_path / "out")
    config.account_mapping = {"PRIMARY": "111111111111", "SECONDARY": "222222222222"}
    trial = ScenarioTrial(
        config=config,
        cred_provider=fake_creds,
        scenario=Scenario(sd),
        container=fake_container,
    )

    result = asyncio.run(trial.run(ScenarioPhase.DEPLOY))

    assert result.success
    # 2 calls for changeset cleanup (one per account) + 2 for post-deploy snapshot deletion
    assert fake_creds.get_session_for_account.call_count == 4
    mock_cfn.delete_change_set.assert_called_once()


# -- export placeholder resolution (post-deploy phases) --------------------


def _scenario_with_verify_env(root, env_toml):
    sd = root / "sc"
    sd.mkdir()
    (sd / "scenario.toml").write_text(
        'schema_version = "1.0"\n\n[scenario]\nname = "sc"\n'
        'account_tags = ["PRIMARY"]\nregions = ["us-east-1"]\n\n'
        f"[verify]\nenv = {env_toml}\n"
    )
    (sd / "scenario").mkdir()
    (sd / "scenario" / "Dockerfile").write_text("FROM alpine\n")
    (sd / "deploy").mkdir()
    (sd / "deploy" / "deploy.sh").write_text("#!/bin/sh\n")
    (sd / "verify").mkdir()
    (sd / "verify" / "verify.sh").write_text("#!/bin/sh\n")
    return sd


def _verify_trial(sd, tmp_path, fake_container, fake_creds):
    cfg = ScenarioTrialConfig(
        scenario=ScenarioConfig(name="sc", path=sd),
        trial_name="trial-0",
        output_dir=tmp_path / "out",
        account_mapping={"PRIMARY": "111111111111"},
        ou_name="test-ou",
    )
    return ScenarioTrial(
        config=cfg,
        cred_provider=fake_creds,
        scenario=Scenario(sd),
        container=fake_container,
    )


def test_verify_resolves_qualified_export(tmp_path, fake_container, fake_creds):
    sd = _scenario_with_verify_env(tmp_path, '{BUCKET = "{{PRIMARY::AppBucket}}"}')
    trial = _verify_trial(sd, tmp_path, fake_container, fake_creds)
    with patch(
        "aws_bench.scenario.trial.collect_account_exports",
        return_value={"111111111111": {"AppBucket": "b-1"}},
    ) as mock_collect:
        asyncio.run(trial.run(ScenarioPhase.VERIFY))
    mock_collect.assert_called_once_with({"111111111111": ["us-east-1"]})
    assert fake_container.run_phase.await_args.kwargs["env"]["BUCKET"] == "b-1"


def test_verify_resolves_bare_export_against_sole_tag(tmp_path, fake_container, fake_creds):
    sd = _scenario_with_verify_env(tmp_path, '{BUCKET = "{{AppBucket}}"}')
    trial = _verify_trial(sd, tmp_path, fake_container, fake_creds)
    with patch(
        "aws_bench.scenario.trial.collect_account_exports",
        return_value={"111111111111": {"AppBucket": "b-1"}},
    ):
        asyncio.run(trial.run(ScenarioPhase.VERIFY))
    assert fake_container.run_phase.await_args.kwargs["env"]["BUCKET"] == "b-1"


def test_deploy_skips_export_collection(tmp_path, fake_container, fake_creds):
    sd = _make_scenario_dir(tmp_path)
    (sd / "scenario.toml").write_text(
        'schema_version = "1.0"\n\n[scenario]\nname = "sc"\n'
        'account_tags = ["PRIMARY"]\nregions = ["us-east-1"]\n\n'
        '[deploy]\nenv = {RAW = "{{NotYet}}"}\n'
    )
    trial = _verify_trial(sd, tmp_path, fake_container, fake_creds)
    with patch("aws_bench.scenario.trial.collect_account_exports") as mock_collect:
        asyncio.run(trial.run(ScenarioPhase.DEPLOY))
    mock_collect.assert_not_called()
    assert fake_container.run_phase.await_args.kwargs["env"]["RAW"] == "{{NotYet}}"


def test_verify_missing_export_fails_loud(tmp_path, fake_container, fake_creds):
    sd = _scenario_with_verify_env(tmp_path, '{BUCKET = "{{PRIMARY::Absent}}"}')
    trial = _verify_trial(sd, tmp_path, fake_container, fake_creds)
    with patch(
        "aws_bench.scenario.trial.collect_account_exports",
        return_value={"111111111111": {"Other": "x"}},
    ):
        result = asyncio.run(trial.run(ScenarioPhase.VERIFY))
    assert not result.success
    assert result.exception_info is not None
    assert result.exception_info.exception_type == "PlaceholderMissingError"


def test_verify_resolved_phase_env_wins_over_creds(tmp_path, fake_container, fake_creds):
    """A resolved phase-env value overrides a colliding management-env key.

    Precedence is {**creds, **resolved_phase_env}: the phase env is the author's,
    so it wins over the framework-injected MANAGEMENT_ROLE.
    """
    sd = _scenario_with_verify_env(tmp_path, '{MANAGEMENT_ROLE = "{{PRIMARY::Role}}"}')
    trial = _verify_trial(sd, tmp_path, fake_container, fake_creds)
    with patch(
        "aws_bench.scenario.trial.collect_account_exports",
        return_value={"111111111111": {"Role": "from-export"}},
    ):
        asyncio.run(trial.run(ScenarioPhase.VERIFY))
    assert fake_container.run_phase.await_args.kwargs["env"]["MANAGEMENT_ROLE"] == "from-export"
