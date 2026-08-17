"""Tests for aws_bench.cli.env commands."""

from __future__ import annotations

import signal
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from aws_bench.account_management.exceptions import (
    AccountResolutionError,
)
from aws_bench.account_management.models import (
    OrgInfo,
    ScenarioAccount,
)
from aws_bench.cli.display import display_cleanup_results
from aws_bench.cli.env import _describe_org_account_quota
from aws_bench.cli.main import _handle_shutdown, app
from aws_bench.cli.preflight import PreflightError
from aws_bench.resource_management.cleanup.models import (
    AccountCleanupResult,
    CleanupSummary,
    RegionResult,
)
from aws_bench.resource_management.models import QuotaIncreaseResult, QuotaStatus
from aws_bench.resource_management.reset.models import ResetResult
from aws_bench.resource_management.snapshot.models import SnapshotResult
from aws_bench.resource_management.verify.models import (
    AccountVerifyResult,
    RegionVerifyResult,
)
from aws_bench.scenario.events import ScenarioPhase
from aws_bench.scenario.exceptions import (
    InsufficientQuotaError,
    ScenarioDiscoveryError,
    UnmetQuota,
)
from aws_bench.scenario.provisioning import ProvisionedAccount, ProvisioningSummary
from aws_bench.scenario.results import ScenarioJobResult, ScenarioTrialResult
from aws_bench.utils import concurrent


@pytest.fixture()
def runner():
    """Create a CLI test runner."""
    return CliRunner()


def test_terminate_renders_bracketed_failure_status_without_crashing(runner):
    """A botocore error in a per-account status must not be parsed as Rich markup.

    ``terminate_environment`` returns ``FAILED (<botocore error>)`` statuses; an
    AWS error containing a closing-tag-like ``[/SUSPENDED]`` would raise
    ``MarkupError`` and abort the outcome table mid-render during a destructive
    multi-account close. The status must render verbatim.
    """
    mgr = MagicMock()
    mgr.is_preexisting = False
    mgr._org.get_org_info.return_value = OrgInfo(
        org_id="o-abc",
        management_account_id="111111111111",
        root_id="r-xyz",
        management_account_email="mgmt@example.com",
    )
    mgr._require_ou.return_value = "ou-abc-123"
    mgr._org.list_accounts_in_ou.return_value = [
        {"Id": "222233334444", "Status": "ACTIVE", "Name": "acct"},
    ]
    mgr.terminate_environment.return_value = {
        "222233334444": "FAILED (Account in state [/SUSPENDED] cannot transition)",
    }

    with patch("aws_bench.cli.env.AccountManager", return_value=mgr):
        result = runner.invoke(app, ["env", "terminate", "--env-name", "env-1"], input="y\n")

    # Command exits 1 (an account failed) but must not crash on markup parsing.
    assert result.exit_code == 1
    assert "MarkupError" not in result.output
    assert "[/SUSPENDED]" in result.output


def test_creds_error_with_bracketed_message_does_not_crash(runner):
    """A bracketed botocore error in the creds failure must not be parsed as markup.

    ``BedrockCredentialError`` can wrap a botocore ``ClientError`` whose text
    contains a regex char class or closing tag; the error line must render it
    literally, not raise ``MarkupError`` on the already-failing path.
    """
    from aws_bench.utils.bedrock_credentials import BedrockCredentialError

    err = BedrockCredentialError("Failed: value must match [a-z]+ and not [/closed]")
    with patch("aws_bench.cli.env.generate_bearer_token", side_effect=err):
        result = runner.invoke(app, ["env", "creds"])

    assert result.exit_code == 1
    assert "MarkupError" not in result.output
    assert "[a-z]+" in result.output


@pytest.fixture()
def mock_scenario_accounts():
    """Two scenario-tagged accounts as returned by list_scenario_accounts."""
    return [
        ScenarioAccount(
            account_id="123456789012",
            email="env1@example.com",
            scenario_name="env-1",
            account_tag="main",
            status="ACTIVE",
        ),
        ScenarioAccount(
            account_id="234567890123",
            email="env2@example.com",
            scenario_name="env-2",
            account_tag="main",
            status="ACTIVE",
        ),
    ]


@pytest.fixture()
def mock_cleanup_summary():
    """Create a mock CleanupSummary for successful cleanup."""
    region_result = RegionResult(
        region="us-east-1",
        stacks_found=3,
        stacks_deleted=3,
        stacks_failed=[],
        error="",
    )
    return CleanupSummary(
        regions=[region_result],
        orphaned_resources={},
        run_dir="",
    )


def _cleanup_result_dict(account_id, mock_cleanup_summary, *, failed=False):
    """Build one AccountCleanupResult as the trial attaches it."""
    if failed:
        return AccountCleanupResult(account_id=account_id, summary=None, error="Cleanup failed")
    return AccountCleanupResult(account_id=account_id, summary=mock_cleanup_summary, error=None)


def _mock_cleanup_seams(monkeypatch, *, tmp_path, results_per_trial):
    """Mock ScenarioJob.create + run for cleanup tests.

    ``results_per_trial`` is a list of (scenario_name, list_of_cleanup_dicts).
    """
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_cli", lambda: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_daemon", lambda: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_aws_credentials", lambda cred, **kwargs: None)
    monkeypatch.setattr(
        "aws_bench.cli.env.CredentialProvider.get",
        classmethod(lambda cls: MagicMock()),
    )

    trial_results = []
    any_failed = False
    for scenario_name, cleanup_results in results_per_trial:
        failed = any(r.summary is None for r in cleanup_results)
        any_failed = any_failed or failed
        trial_results.append(
            ScenarioTrialResult(
                scenario_name=scenario_name,
                trial_name=scenario_name,
                phase=ScenarioPhase.CLEANUP,
                started_at=datetime.now(timezone.utc),
                finished_at=datetime.now(timezone.utc),
                exit_code=0,
                resource_results={"cleanup": cleanup_results},
            )
        )

    n_total = len(results_per_trial)
    result = ScenarioJobResult(
        job_name="test",
        job_dir=tmp_path / "jobs" / "test",
        started_at=datetime.now(timezone.utc),
        n_total=n_total,
        n_succeeded=n_total,
        n_failed=0,
        trial_results=trial_results,
    )

    job = MagicMock()
    job.run = AsyncMock(return_value=result)
    create_mock = AsyncMock(return_value=job)
    monkeypatch.setattr("aws_bench.cli.env.ScenarioJob.create", create_mock)

    async def _passthrough(job, phase):
        return await job.run(phase)

    monkeypatch.setattr("aws_bench.cli.env.run_phase_with_progress", _passthrough)
    return create_mock


def test_cleanup_successful(runner, monkeypatch, tmp_path, mock_cleanup_summary):
    """Cleanup runs the CLEANUP phase and reports deleted stacks per account."""
    _mock_cleanup_seams(
        monkeypatch,
        tmp_path=tmp_path,
        results_per_trial=[
            ("env-1", [_cleanup_result_dict("123456789012", mock_cleanup_summary)]),
            ("env-2", [_cleanup_result_dict("234567890123", mock_cleanup_summary)]),
        ],
    )

    result = runner.invoke(
        app, ["env", "cleanup", "--env-name", "test-env", "--scenario-path", str(tmp_path), "--yes"]
    )

    assert result.exit_code == 0
    assert "123456789012" in result.stdout
    assert "234567890123" in result.stdout
    assert "Clean" in result.stdout  # clean teardown detail in table


# ===========================================================================
# setup — containerized scenario deployment
# ===========================================================================
# All setup tests mock at the same seam: ScenarioJob.create + run, plus the
# three preflight checks. Avoids touching Docker, AWS, or the validator path.


def _combined_output(result):
    """Combine stdout + stderr robustly across CliRunner mix_stderr modes.

    Typer's CliRunner mixes stderr into stdout by default; this helper
    stays robust if mix_stderr ever changes.
    """
    out = result.stdout or ""
    try:
        out += result.stderr or ""
    except (ValueError, AttributeError):
        pass
    return out


def _mock_setup_seams(monkeypatch, *, all_passed: bool, raise_acct: bool = False):
    """Patch every external dependency used by env setup.

    Returns the AsyncMock for ScenarioJob.create so the test can assert
    on the config it received.
    """
    # Preflight: all pass
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_cli", lambda: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_daemon", lambda: None)
    monkeypatch.setattr(
        "aws_bench.cli.env.preflight_aws_credentials",
        lambda cred, **kwargs: None,
    )

    # CredentialProvider: stubbed
    monkeypatch.setattr(
        "aws_bench.cli.env.CredentialProvider.get",
        classmethod(lambda cls: MagicMock()),
    )

    # ScenarioJob.create: returns a mock job whose .run yields the result
    if raise_acct:

        async def _create_raises(*args, **kwargs):
            raise AccountResolutionError("Scenario 'foo' has no provisioned accounts.")

        create_mock = AsyncMock(side_effect=_create_raises)
    else:
        result = ScenarioJobResult(
            job_name="test",
            job_dir=Path("/tmp/test"),
            started_at=datetime.now(timezone.utc),
            n_total=1,
            n_succeeded=1 if all_passed else 0,
            n_failed=0 if all_passed else 1,
        )
        job = MagicMock()
        job.run = AsyncMock(return_value=result)
        create_mock = AsyncMock(return_value=job)

    monkeypatch.setattr("aws_bench.cli.env.ScenarioJob.create", create_mock)

    # run_phase_with_progress: pass-through
    async def _passthrough(job, phase):
        return await job.run(phase)

    monkeypatch.setattr(
        "aws_bench.cli.env.run_phase_with_progress",
        _passthrough,
    )

    return create_mock


def test_setup_succeeds_on_all_passed(runner, monkeypatch, tmp_path):
    _mock_setup_seams(monkeypatch, all_passed=True)
    result = runner.invoke(
        app,
        ["env", "setup", "--env-name", "x", "--scenario-path", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "Deployed 1/1" in _combined_output(result)


def test_setup_exits_on_partial_failure(runner, monkeypatch, tmp_path):
    _mock_setup_seams(monkeypatch, all_passed=False)
    result = runner.invoke(
        app,
        ["env", "setup", "--env-name", "x", "--scenario-path", str(tmp_path)],
    )
    assert result.exit_code == 1


def test_setup_fails_when_docker_not_installed(runner, monkeypatch, tmp_path):
    def raise_docker():
        raise PreflightError("docker binary not found on PATH.")

    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_cli", raise_docker)
    result = runner.invoke(
        app,
        ["env", "setup", "--env-name", "x", "--scenario-path", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "docker binary" in _combined_output(result)


def test_setup_account_resolution_error_points_to_init(runner, monkeypatch, tmp_path):
    _mock_setup_seams(monkeypatch, all_passed=True, raise_acct=True)
    result = runner.invoke(
        app,
        ["env", "setup", "--env-name", "myou", "--scenario-path", str(tmp_path)],
    )
    assert result.exit_code == 1
    out = _combined_output(result)
    assert "aws-bench env init" in out
    assert "myou" in out


def test_setup_blocks_on_insufficient_quota(runner, monkeypatch, tmp_path):
    """Setup catches InsufficientQuotaError and prints the env init hint."""
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_cli", lambda: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_daemon", lambda: None)
    monkeypatch.setattr(
        "aws_bench.cli.env.preflight_aws_credentials",
        lambda cred, **kwargs: None,
    )
    monkeypatch.setattr(
        "aws_bench.cli.env.CredentialProvider.get",
        classmethod(lambda cls: MagicMock()),
    )

    unmet = UnmetQuota(
        scenario_name="alpha",
        account_id="111111111111",
        region="us-east-1",
        result=QuotaIncreaseResult(
            service_code="ec2",
            quota_code="L-1216C47A",
            desired_value=50.0,
            status=QuotaStatus.ALREADY_PENDING,
            error_message="current=8.0, required=50.0 (PENDING)",
        ),
    )

    async def _create_raises(*args, **kwargs):
        raise InsufficientQuotaError([unmet])

    monkeypatch.setattr(
        "aws_bench.cli.env.ScenarioJob.create",
        AsyncMock(side_effect=_create_raises),
    )

    result = runner.invoke(
        app,
        ["env", "setup", "--env-name", "myou", "--scenario-path", str(tmp_path)],
    )
    out = _combined_output(result)
    assert result.exit_code == 1
    assert "L-1216C47A" in out
    assert "ALREADY_PENDING" in out
    assert "aws-bench env init" in out
    assert "--wait-for-quotas" in out
    assert "myou" in out


def _stub_preflight_and_creds(monkeypatch):
    """Pass preflight + creds; intended for tests that exercise validator paths."""
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_cli", lambda: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_daemon", lambda: None)
    monkeypatch.setattr(
        "aws_bench.cli.env.preflight_aws_credentials",
        lambda cred, **kwargs: None,
    )
    monkeypatch.setattr(
        "aws_bench.cli.env.CredentialProvider.get",
        classmethod(lambda cls: MagicMock()),
    )


def test_setup_requires_one_of_scenario_path_or_dataset(runner, monkeypatch):
    """Validator: neither --scenario-path nor -d → error."""
    _stub_preflight_and_creds(monkeypatch)
    result = runner.invoke(app, ["env", "setup", "--env-name", "x"])
    assert result.exit_code == 1
    out = _combined_output(result).lower()
    assert "scenario-path" in out or "dataset" in out


def test_setup_rejects_both_scenario_path_and_dataset(runner, monkeypatch, tmp_path):
    _stub_preflight_and_creds(monkeypatch)
    result = runner.invoke(
        app,
        ["env", "setup", "--env-name", "x", "--scenario-path", str(tmp_path), "-d", "awsbench"],
    )
    assert result.exit_code == 1
    assert "not both" in _combined_output(result).lower()


def test_setup_passes_overrides_to_environment_config(runner, monkeypatch, tmp_path):
    create_mock = _mock_setup_seams(monkeypatch, all_passed=True)
    result = runner.invoke(
        app,
        [
            "env",
            "setup",
            "--env-name",
            "x",
            "--scenario-path",
            str(tmp_path),
            "--override-cpus",
            "8",
            "--force-build",
            "--no-delete",
        ],
    )
    assert result.exit_code == 0
    job_config = create_mock.call_args.args[0]
    assert job_config.environment.override_cpus == 8
    assert job_config.environment.force_build is True
    assert job_config.environment.delete is False


def test_setup_passes_retry_config(runner, monkeypatch, tmp_path):
    create_mock = _mock_setup_seams(monkeypatch, all_passed=True)
    result = runner.invoke(
        app,
        [
            "env",
            "setup",
            "--env-name",
            "x",
            "--scenario-path",
            str(tmp_path),
            "--max-retries",
            "2",
            "--retry-include",
            "DockerCLIError",
        ],
    )
    assert result.exit_code == 0
    # Retry policy lives on the job config (config.retry), not a separate
    # create() argument.
    job_cfg = create_mock.call_args.args[0]
    assert job_cfg.retry.max_retries == 2
    assert job_cfg.retry.include_exceptions == {"DockerCLIError"}


def test_setup_passes_dataset_filters(runner, monkeypatch, tmp_path):
    """Scenario filters land on job_config.dataset.{include,exclude}_scenario_names."""
    create_mock = _mock_setup_seams(monkeypatch, all_passed=True)
    result = runner.invoke(
        app,
        [
            "env",
            "setup",
            "--env-name",
            "x",
            "--scenario-path",
            str(tmp_path),
            "--include-scenarios",
            "lambda-*",
            "--exclude-scenarios",
            "lambda-skip",
        ],
    )
    assert result.exit_code == 0
    job_config = create_mock.call_args.args[0]
    assert job_config.dataset.include_scenario_names == ["lambda-*"]
    assert job_config.dataset.exclude_scenario_names == ["lambda-skip"]


def test_cleanup_scenarios_not_found(runner, monkeypatch, tmp_path):
    """Cleanup exits with error when scenario discovery fails."""
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_cli", lambda: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_daemon", lambda: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_aws_credentials", lambda cred, **kwargs: None)
    monkeypatch.setattr(
        "aws_bench.cli.env.CredentialProvider.get",
        classmethod(lambda cls: MagicMock()),
    )

    async def _create_raises(*args, **kwargs):
        errors: list[tuple[Path, Exception]] = [(tmp_path, ValueError("No scenarios found"))]
        raise ScenarioDiscoveryError(errors)

    monkeypatch.setattr(
        "aws_bench.cli.env.ScenarioJob.create", AsyncMock(side_effect=_create_raises)
    )

    result = runner.invoke(
        app,
        ["env", "cleanup", "--env-name", "missing-env", "--scenario-path", str(tmp_path), "--yes"],
    )

    assert result.exit_code == 1
    # Rich may wrap the message at the console width, inserting newlines
    # (and leaving a trailing space) mid-message; collapse all whitespace runs.
    output = " ".join(_combined_output(result).split())
    assert "No scenarios found" in output


def test_cleanup_partial_failures(runner, monkeypatch, tmp_path, mock_cleanup_summary):
    """Cleanup reports per-account failures and exits non-zero."""
    _mock_cleanup_seams(
        monkeypatch,
        tmp_path=tmp_path,
        results_per_trial=[
            ("env-1", [_cleanup_result_dict("123456789012", mock_cleanup_summary)]),
            ("env-2", [_cleanup_result_dict("234567890123", mock_cleanup_summary, failed=True)]),
        ],
    )

    result = runner.invoke(
        app, ["env", "cleanup", "--env-name", "test-env", "--scenario-path", str(tmp_path), "--yes"]
    )

    assert result.exit_code == 1
    assert any("234567890123" in line and "FAILED" in line for line in result.stdout.splitlines())


# ── display_cleanup_results tests ──


def _cleanup_job_result(*cleanup_results) -> ScenarioJobResult:
    """Wrap one trial's AccountCleanupResults in a ScenarioJobResult."""
    return ScenarioJobResult(
        job_name="test",
        job_dir=Path("/tmp/job"),
        started_at=datetime.now(timezone.utc),
        n_total=1,
        trial_results=[
            ScenarioTrialResult(
                scenario_name="test-env",
                trial_name="test-env",
                phase=ScenarioPhase.CLEANUP,
                account_mapping={r.account_id: r.account_id for r in cleanup_results},
                resource_results={"cleanup": list(cleanup_results)},
            )
        ],
    )


def testdisplay_cleanup_results_all_success(mock_cleanup_summary):
    """Test display_cleanup_results returns False on full success."""
    result = _cleanup_job_result(
        AccountCleanupResult("123456789012", mock_cleanup_summary, None),
        AccountCleanupResult("234567890123", mock_cleanup_summary, None),
    )

    has_failures = display_cleanup_results(result)

    assert has_failures is False


def testdisplay_cleanup_results_with_failures(mock_cleanup_summary):
    """Test display_cleanup_results returns True when there are failures."""
    result = _cleanup_job_result(
        AccountCleanupResult("123456789012", mock_cleanup_summary, None),
        AccountCleanupResult("234567890123", None, "Credential error"),
    )

    has_failures = display_cleanup_results(result)

    assert has_failures is True


# ===========================================================================
# verify — containerized scenario-tag flow
# ===========================================================================


def _mock_verify_seams(monkeypatch, *, all_passed: bool, tmp_path):
    """Mock ScenarioJob.create + run for verify tests."""
    # Mock preflight
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_cli", lambda: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_daemon", lambda: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_aws_credentials", lambda cred, **kwargs: None)
    monkeypatch.setattr(
        "aws_bench.cli.env.CredentialProvider.get",
        classmethod(lambda cls: MagicMock()),
    )

    # Build trial results with verify data
    trial_results = []
    for i, scenario_name in enumerate(["env-1", "env-2"]):
        account_id = f"12345678901{i}"
        verify_result = AccountVerifyResult(
            account_id=account_id,
            environment_id=scenario_name,
            success=all_passed,
            region_results=[
                RegionVerifyResult(
                    region="us-east-1",
                    success=all_passed,
                    error_message=None if all_passed else "drift detected",
                )
            ],
        )
        trial_results.append(
            ScenarioTrialResult(
                scenario_name=scenario_name,
                trial_name=scenario_name,
                phase=ScenarioPhase.VERIFY,
                started_at=datetime.now(timezone.utc),
                finished_at=datetime.now(timezone.utc),
                exit_code=0 if all_passed else 1,
                resource_results={"verify": [verify_result]},
            )
        )

    result = ScenarioJobResult(
        job_name="test",
        job_dir=tmp_path / "jobs" / "test",
        started_at=datetime.now(timezone.utc),
        n_total=2,
        n_succeeded=2 if all_passed else 0,
        n_failed=0 if all_passed else 2,
        trial_results=trial_results,
    )

    job = MagicMock()
    job.run = AsyncMock(return_value=result)
    create_mock = AsyncMock(return_value=job)

    monkeypatch.setattr("aws_bench.cli.env.ScenarioJob.create", create_mock)

    # run_phase_with_progress: pass-through
    async def _passthrough(job, phase):
        return await job.run(phase)

    monkeypatch.setattr(
        "aws_bench.cli.env.run_phase_with_progress",
        _passthrough,
    )

    return create_mock


def test_verify_passes(runner, monkeypatch, tmp_path):
    """Verify discovers scenario accounts and reports success per account."""
    _mock_verify_seams(monkeypatch, all_passed=True, tmp_path=tmp_path)

    result = runner.invoke(
        app, ["env", "verify", "--env-name", "test-env", "--scenario-path", str(tmp_path)]
    )

    assert result.exit_code == 0
    assert "All accounts verified successfully" in result.stdout


def test_verify_reports_failure(runner, monkeypatch, tmp_path):
    """Verify exits non-zero when an account fails verification."""
    _mock_verify_seams(monkeypatch, all_passed=False, tmp_path=tmp_path)

    result = runner.invoke(
        app, ["env", "verify", "--env-name", "test-env", "--scenario-path", str(tmp_path)]
    )

    assert result.exit_code == 1
    assert "Verification FAILED" in result.stdout


def test_verify_no_accounts(runner, monkeypatch, tmp_path):
    """Verify exits cleanly when no scenario accounts exist."""
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_cli", lambda: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_daemon", lambda: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_aws_credentials", lambda cred, **kwargs: None)
    monkeypatch.setattr(
        "aws_bench.cli.env.CredentialProvider.get",
        classmethod(lambda cls: MagicMock()),
    )

    async def _create_raises(*args, **kwargs):
        errors: list[tuple[Path, Exception]] = [(tmp_path, ValueError("No scenarios found"))]
        raise ScenarioDiscoveryError(errors)

    create_mock = AsyncMock(side_effect=_create_raises)
    monkeypatch.setattr("aws_bench.cli.env.ScenarioJob.create", create_mock)

    result = runner.invoke(
        app, ["env", "verify", "--env-name", "test-env", "--scenario-path", str(tmp_path)]
    )

    assert result.exit_code == 1
    # Rich may wrap the message at the console width, inserting newlines
    # (and leaving a trailing space) mid-message; collapse all whitespace runs.
    output = " ".join(_combined_output(result).split())
    assert "No scenarios found" in output


# ===========================================================================
# reset — containerized scenario-tag flow
# ===========================================================================


def _mock_reset_seams(monkeypatch, *, all_passed: bool, tmp_path, needs_redeploy: bool = False):
    """Mock ScenarioJob.create + run for reset tests."""
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_cli", lambda: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_daemon", lambda: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_aws_credentials", lambda cred, **kwargs: None)
    monkeypatch.setattr(
        "aws_bench.cli.env.CredentialProvider.get",
        classmethod(lambda cls: MagicMock()),
    )

    trial_results = []
    for scenario_name in ["env-1", "env-2"]:
        reset_result = ResetResult(
            success=all_passed,
            reason="ok" if all_passed else "drift could not be reverted",
            needs_redeploy=needs_redeploy,
        )
        trial_results.append(
            ScenarioTrialResult(
                scenario_name=scenario_name,
                trial_name=scenario_name,
                phase=ScenarioPhase.RESET,
                started_at=datetime.now(timezone.utc),
                finished_at=datetime.now(timezone.utc),
                exit_code=0 if all_passed else 1,
                resource_results={"reset": [reset_result]},
            )
        )

    result = ScenarioJobResult(
        job_name="test",
        job_dir=tmp_path / "jobs" / "test",
        started_at=datetime.now(timezone.utc),
        n_total=2,
        n_succeeded=2 if all_passed else 0,
        n_failed=0 if all_passed else 2,
        trial_results=trial_results,
    )

    job = MagicMock()
    job.run = AsyncMock(return_value=result)
    create_mock = AsyncMock(return_value=job)
    monkeypatch.setattr("aws_bench.cli.env.ScenarioJob.create", create_mock)

    async def _passthrough(job, phase):
        return await job.run(phase)

    monkeypatch.setattr("aws_bench.cli.env.run_phase_with_progress", _passthrough)
    return create_mock


def test_reset_succeeds(runner, monkeypatch, tmp_path):
    """Reset runs the RESET phase and reports a success summary."""
    _mock_reset_seams(monkeypatch, all_passed=True, tmp_path=tmp_path)

    result = runner.invoke(
        app, ["env", "reset", "--env-name", "test-env", "--scenario-path", str(tmp_path), "--yes"]
    )

    assert result.exit_code == 0
    assert "Reset complete: 2 succeeded, 0 failed" in result.stdout


def test_reset_reports_failure(runner, monkeypatch, tmp_path):
    """Reset exits non-zero when an account fails to reset."""
    _mock_reset_seams(monkeypatch, all_passed=False, tmp_path=tmp_path)

    result = runner.invoke(
        app, ["env", "reset", "--env-name", "test-env", "--scenario-path", str(tmp_path), "--yes"]
    )

    assert result.exit_code == 1
    assert "Reset complete: 0 succeeded, 2 failed" in result.stdout


def test_reset_without_source_errors(runner, monkeypatch):
    """Reset requires a scenario source to discover scenarios; none is an error."""
    monkeypatch.setattr("aws_bench.cli.env.preflight_aws_credentials", lambda cred, **kwargs: None)
    monkeypatch.setattr(
        "aws_bench.cli.env.CredentialProvider.get",
        classmethod(lambda cls: MagicMock()),
    )

    result = runner.invoke(app, ["env", "reset", "--env-name", "test-env"])

    assert result.exit_code == 1
    out = _combined_output(result)
    assert "scenario-path" in out or "dataset" in out


def test_reset_redeploys_via_setup_when_source_given(runner, monkeypatch, tmp_path):
    """When reset needs redeploy, the trial automatically handles it within RESET phase.

    The reset trial detects when stacks need redeployment and automatically runs
    the DEPLOY phase followed by snapshot recapture. The CLI no longer needs to
    run a separate DEPLOY job.
    """
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_cli", lambda: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_daemon", lambda: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_aws_credentials", lambda cred, **kwargs: None)
    monkeypatch.setattr(
        "aws_bench.cli.env.CredentialProvider.get",
        classmethod(lambda cls: MagicMock()),
    )

    # RESET job: trial automatically handled redeploy internally
    reset_result = ResetResult(
        success=True,
        reason="Reset complete; deleted stacks recreated via setup",
        needs_redeploy=True,
        redeploy_succeeded=True,
    )
    reset_job_result = ScenarioJobResult(
        job_name="reset",
        job_dir=tmp_path / "jobs" / "reset",
        started_at=datetime.now(timezone.utc),
        n_total=1,
        n_succeeded=1,
        n_failed=0,
        trial_results=[
            ScenarioTrialResult(
                scenario_name="env-1",
                trial_name="env-1",
                phase=ScenarioPhase.RESET,
                started_at=datetime.now(timezone.utc),
                finished_at=datetime.now(timezone.utc),
                exit_code=0,
                resource_results={"reset": [reset_result]},
            )
        ],
    )

    reset_job = MagicMock()
    reset_job.run = AsyncMock(return_value=reset_job_result)

    # Only one job created (RESET), which handles redeploy internally
    create_mock = AsyncMock(return_value=reset_job)
    monkeypatch.setattr("aws_bench.cli.env.ScenarioJob.create", create_mock)

    async def _passthrough(job, phase):
        return await job.run(phase)

    monkeypatch.setattr("aws_bench.cli.env.run_phase_with_progress", _passthrough)

    result = runner.invoke(
        app,
        ["env", "reset", "--env-name", "test-env", "--scenario-path", str(tmp_path), "--yes"],
    )

    assert result.exit_code == 0
    assert "Stacks recreated via setup" in result.stdout
    # Only one job created (RESET), which handles redeploy internally
    assert create_mock.call_count == 1
    reset_job.run.assert_awaited_once()


# ===========================================================================
# env show — quota status display
# ===========================================================================


def test_show_displays_accounts_and_stacks(runner):
    """Env show displays accounts and stack status."""
    org_info = OrgInfo(
        org_id="o-abc",
        root_id="r-root1",
        management_account_id="111111111111",
        management_account_email="mgmt@example.com",
    )
    accounts = [
        ScenarioAccount(
            account_id="222222222222",
            email="test@example.com",
            scenario_name="env-test",
            account_tag="PRIMARY",
            status="ACTIVE",
        ),
    ]

    env = "aws_bench.cli.env"
    with (
        patch(f"{env}.AccountManager") as mock_acct_cls,
        patch(f"{env}.CredentialProvider") as mock_cred_cls,
        patch(f"{env}.list_account_stacks") as mock_stacks,
        patch(
            "aws_bench.resource_management.manager.get_enabled_regions",
            return_value=["us-east-1"],
        ),
    ):
        mock_acct = MagicMock()
        mock_acct_cls.return_value = mock_acct
        mock_acct._org.get_org_info.return_value = org_info
        mock_acct._require_ou.return_value = "ou-test1"
        mock_acct.list_scenario_accounts.return_value = accounts

        mock_cred = MagicMock()
        mock_cred_cls.get.return_value = mock_cred
        mock_session = MagicMock()
        mock_cred.get_session_for_account.return_value = mock_session
        # Quota: no requests
        mock_sq = MagicMock()
        mock_session.client.return_value = mock_sq
        mock_sq.get_paginator.return_value.paginate.return_value = [{"RequestedQuotas": []}]

        mock_stacks.return_value = [
            {"name": "my-stack-us-east-1", "status": "CREATE_COMPLETE", "region": "us-east-1"},
        ]

        result = runner.invoke(app, ["env", "show", "--env-name", "test-env"])

    assert result.exit_code == 0
    assert "111111111111" in result.stdout
    assert "o-abc" in result.stdout
    assert "env-test" in result.stdout
    assert "222222222222" in result.stdout
    assert "CREATE_COMPLETE" in result.stdout
    assert "us-east-1" in result.stdout


def test_show_verifies_quota_current_value(runner):
    """Env show checks requested vs current quota value."""
    org_info = OrgInfo(
        org_id="o-abc",
        root_id="r-root1",
        management_account_id="111111111111",
        management_account_email="mgmt@example.com",
    )
    accounts = [
        ScenarioAccount(
            account_id="222222222222",
            email="test@example.com",
            scenario_name="env-test",
            account_tag="PRIMARY",
            status="ACTIVE",
        ),
    ]

    env = "aws_bench.cli.env"
    with (
        patch(f"{env}.AccountManager") as mock_acct_cls,
        patch(f"{env}.CredentialProvider") as mock_cred_cls,
        patch(f"{env}.list_account_stacks", return_value=[]),
        patch(
            "aws_bench.resource_management.quota_manager.get_enabled_regions",
            return_value=["us-east-1"],
        ),
        patch(
            "aws_bench.resource_management.quota_manager.create_regional_session"
        ) as mock_regional,
    ):
        mock_acct = MagicMock()
        mock_acct_cls.return_value = mock_acct
        mock_acct._org.get_org_info.return_value = org_info
        mock_acct._require_ou.return_value = "ou-test1"
        mock_acct.list_scenario_accounts.return_value = accounts

        mock_cred = MagicMock()
        mock_cred_cls.get.return_value = mock_cred
        mock_session = MagicMock()
        mock_cred.get_session_for_account.return_value = mock_session
        mock_sq = MagicMock()
        # Quota reads go through a per-region session, not the parent session.
        mock_regional.return_value.client.return_value = mock_sq
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "RequestedQuotas": [
                    {
                        "ServiceCode": "ec2",
                        "QuotaCode": "L-1216C47A",
                        "DesiredValue": 8.0,
                        "Status": "CASE_CLOSED",
                    },
                ]
            }
        ]
        mock_sq.get_paginator.return_value = mock_paginator
        mock_sq.get_service_quota.return_value = {
            "Quota": {"Value": 16.0, "QuotaName": "Running On-Demand Standard instances"}
        }

        # Wide terminal so the quota table renders every column (rich drops
        # columns to fit a narrow width).
        result = runner.invoke(
            app, ["env", "show", "--env-name", "test-env"], env={"COLUMNS": "200"}
        )

    assert result.exit_code == 0
    assert "L-1216C47A" in result.stdout
    assert "Running On-Demand Standard instances" in result.stdout
    assert "16" in result.stdout.split("L-1216C47A")[1]  # check "16" appears after the quota code
    assert "✓" in result.stdout


def test_show_no_accounts(runner):
    """Env show with no accounts shows a message."""
    org_info = OrgInfo(
        org_id="o-abc",
        root_id="r-root1",
        management_account_id="111111111111",
        management_account_email="mgmt@example.com",
    )

    env = "aws_bench.cli.env"
    with patch(f"{env}.AccountManager") as mock_acct_cls:
        mock_acct = MagicMock()
        mock_acct_cls.return_value = mock_acct
        mock_acct._org.get_org_info.return_value = org_info
        mock_acct._require_ou.return_value = "ou-test1"
        mock_acct.list_scenario_accounts.return_value = []

        result = runner.invoke(app, ["env", "show", "--env-name", "test-env"])

    assert result.exit_code == 0
    assert "No scenario accounts" in result.stdout


# ===========================================================================
# init — provision accounts and submit quotas (containerized format)
# ===========================================================================


def _mock_init_seams(
    monkeypatch,
    *,
    summary: ProvisioningSummary | None = None,
    raise_discovery: bool = False,
):
    """Patch every external dependency used by env init.

    Returns the AsyncMock for provision_scenarios so tests can assert
    on the args it received.
    """
    monkeypatch.setattr(
        "aws_bench.cli.env.preflight_aws_credentials",
        lambda cred, **kwargs: None,
    )
    monkeypatch.setattr(
        "aws_bench.cli.env.CredentialProvider.get",
        classmethod(lambda cls: MagicMock()),
    )

    # init_organization bootstraps the OU before provisioning; stub it so tests
    # don't hit boto3. Baseline capture lives inside the mocked provision_scenarios.
    monkeypatch.setattr(
        "aws_bench.cli.env.AccountManager",
        lambda *a, **kw: MagicMock(init_organization=MagicMock()),
    )

    # dataset_cfg.get_scenarios returns a non-empty map so init gets past the
    # empty-discovery guard and reaches provisioning.
    async def _get_scenarios(self):
        return {MagicMock(): MagicMock()}

    monkeypatch.setattr(
        "aws_bench.dataset.config.AwsBenchDatasetConfig.get_scenarios",
        _get_scenarios,
    )

    if raise_discovery:

        async def _raises(*args, **kwargs):
            raise ScenarioDiscoveryError([(Path("/tmp/bad"), ValueError("bad toml"))])

        provision_mock = AsyncMock(side_effect=_raises)
    else:
        if summary is None:
            summary = ProvisioningSummary()
            summary.accounts = []
        provision_mock = AsyncMock(return_value=summary)

    # Patch both seams: non-quiet path uses provision_scenarios_with_progress,
    # quiet path uses provision_scenarios. Tests can drive either.
    monkeypatch.setattr(
        "aws_bench.cli.env.provision_scenarios_with_progress",
        provision_mock,
    )
    monkeypatch.setattr(
        "aws_bench.cli.env.provision_scenarios",
        provision_mock,
    )
    return provision_mock


def test_init_succeeds_with_scenario_path(runner, monkeypatch, tmp_path):
    summary = ProvisioningSummary()
    provision_mock = _mock_init_seams(monkeypatch, summary=summary)
    result = runner.invoke(
        app,
        ["env", "init", "--env-name", "x", "--scenario-path", str(tmp_path)],
    )
    assert result.exit_code == 0
    provision_mock.assert_awaited_once()


def test_init_passes_n_concurrent_through(runner, monkeypatch, tmp_path):
    provision_mock = _mock_init_seams(monkeypatch)
    result = runner.invoke(
        app,
        [
            "env",
            "init",
            "--env-name",
            "x",
            "--scenario-path",
            str(tmp_path),
            "-n",
            "8",
        ],
    )
    assert result.exit_code == 0
    assert provision_mock.call_args.kwargs["n_concurrent"] == 8


def test_init_passes_wait_for_quotas_through(runner, monkeypatch, tmp_path):
    provision_mock = _mock_init_seams(monkeypatch)
    result = runner.invoke(
        app,
        [
            "env",
            "init",
            "--env-name",
            "x",
            "--scenario-path",
            str(tmp_path),
            "--wait-for-quotas",
            "--quota-timeout",
            "300",
        ],
    )
    assert result.exit_code == 0
    assert provision_mock.call_args.kwargs["wait_for_quotas"] is True
    assert provision_mock.call_args.kwargs["quota_timeout"] == 300


def test_init_requires_one_of_scenario_path_or_dataset(runner, monkeypatch):
    monkeypatch.setattr(
        "aws_bench.cli.env.preflight_aws_credentials",
        lambda cred, **kwargs: None,
    )
    monkeypatch.setattr(
        "aws_bench.cli.env.CredentialProvider.get",
        classmethod(lambda cls: MagicMock()),
    )
    result = runner.invoke(app, ["env", "init", "--env-name", "x"])
    assert result.exit_code == 1
    out = _combined_output(result).lower()
    assert "scenario-path" in out or "dataset" in out


def test_init_rejects_both_scenario_path_and_dataset(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "aws_bench.cli.env.preflight_aws_credentials",
        lambda cred, **kwargs: None,
    )
    monkeypatch.setattr(
        "aws_bench.cli.env.CredentialProvider.get",
        classmethod(lambda cls: MagicMock()),
    )
    result = runner.invoke(
        app,
        [
            "env",
            "init",
            "--env-name",
            "x",
            "--scenario-path",
            str(tmp_path),
            "-d",
            "awsbench",
        ],
    )
    assert result.exit_code == 1
    assert "not both" in _combined_output(result).lower()


def test_init_exits_on_provisioning_failure(runner, monkeypatch, tmp_path):
    """provision_scenarios raises ScenarioDiscoveryError → exit 1."""
    _mock_init_seams(monkeypatch, raise_discovery=True)
    result = runner.invoke(
        app,
        ["env", "init", "--env-name", "x", "--scenario-path", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "bad toml" in _combined_output(result)


def test_init_passes_dataset_filters(runner, monkeypatch, tmp_path):
    """--include-scenarios and --exclude-scenarios reach dataset_cfg."""
    captured: list = []

    async def _get_scenarios(self):
        captured.append(self)
        return {MagicMock(): MagicMock()}

    _mock_init_seams(monkeypatch)
    monkeypatch.setattr(
        "aws_bench.dataset.config.AwsBenchDatasetConfig.get_scenarios",
        _get_scenarios,
    )

    result = runner.invoke(
        app,
        [
            "env",
            "init",
            "--env-name",
            "x",
            "--scenario-path",
            str(tmp_path),
            "--include-scenarios",
            "foo*",
            "--exclude-scenarios",
            "bar*",
        ],
    )
    assert result.exit_code == 0
    assert captured[0].include_scenario_names == ["foo*"]
    assert captured[0].exclude_scenario_names == ["bar*"]


def _pending_quota_summary(*, waited: bool) -> ProvisioningSummary:
    """A summary with a single quota still pending, tagged with whether we waited."""
    summary = ProvisioningSummary(waited=waited)
    summary.unmet_quotas = [
        UnmetQuota(
            scenario_name="scenario",
            account_id="111111111111",
            region="us-east-1",
            result=QuotaIncreaseResult(
                service_code="ec2",
                quota_code="L-1216C47A",
                desired_value=10.0,
                status=QuotaStatus.ALREADY_PENDING,
            ),
        ),
    ]
    return summary


def test_init_exits_on_quota_wait_failure(runner, monkeypatch, tmp_path):
    """Wait-for-quotas timeout leaves a pending quota; init exits nonzero."""
    _mock_init_seams(monkeypatch, summary=_pending_quota_summary(waited=True))
    result = runner.invoke(
        app,
        [
            "env",
            "init",
            "--env-name",
            "x",
            "--scenario-path",
            str(tmp_path),
            "--wait-for-quotas",
        ],
    )
    assert result.exit_code == 1


def test_init_succeeds_with_pending_quota_when_not_waiting(runner, monkeypatch, tmp_path):
    """Without --wait-for-quotas, a still-pending quota is expected: init exits 0.

    The pending quota is submit-and-don't-wait's normal outcome, surfaced in the
    summary's 'not yet granted' table, not treated as a provisioning failure.
    """
    _mock_init_seams(monkeypatch, summary=_pending_quota_summary(waited=False))
    result = runner.invoke(
        app,
        [
            "env",
            "init",
            "--env-name",
            "x",
            "--scenario-path",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0


def test_init_fails_when_snapshot_capture_reports_failure(runner, monkeypatch, tmp_path):
    """A failed PRE_SETUP snapshot on a provisioned account fails init.

    Capture now runs inside provision_scenarios, so the failure rides back on the
    account's snapshot_result; a failed one flips provisioned False → n_failed > 0
    → all_succeeded False → exit 1.
    """
    summary = ProvisioningSummary()
    summary.accounts = [
        ProvisionedAccount(
            scenario_name="scn-a",
            account_tag="PRIMARY",
            account_id="111111111111",
            snapshot_result=SnapshotResult(
                account_id="111111111111", success=False, error_message="scan failed"
            ),
        )
    ]
    assert summary.all_succeeded is False
    _mock_init_seams(monkeypatch, summary=summary)
    result = runner.invoke(
        app,
        ["env", "init", "--env-name", "x", "--scenario-path", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "scn-a/PRIMARY" in _combined_output(result)


def test_handle_shutdown_raises_keyboard_interrupt_and_sets_flag():
    """A signal flags cooperative cancellation, then raises KeyboardInterrupt.

    The flag lets worker-thread scans unwind; the KeyboardInterrupt lets the
    async stack unwind so finally-blocks persist results.
    """
    concurrent.reset_shutdown()
    try:
        with pytest.raises(KeyboardInterrupt):
            _handle_shutdown(signal.SIGTERM, None)
        assert concurrent.shutdown_requested()
    finally:
        concurrent.reset_shutdown()


# ---------------------------------------------------------------------------
# env show — org account-limit request status line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status, reason, expected_style",
    [
        (QuotaStatus.ALREADY_PENDING, "increase pending", "yellow"),
        (QuotaStatus.CASE_CLOSED, "case closed — increase may still be propagating", "green"),
        (QuotaStatus.APPROVED, "approved — increase may still be propagating", "green"),
        (QuotaStatus.DENIED, "increase denied", "red"),
        (QuotaStatus.FAILED, "no increase requested", "dim"),
    ],
)
def test_describe_org_account_quota(status, reason, expected_style):
    """Pending renders yellow, granted green, and none/failed dim — reason always shown."""
    qm = MagicMock()
    qm.diagnose_org_account_quota = MagicMock(return_value=(status, reason))

    line = _describe_org_account_quota(qm)

    assert reason in line
    assert expected_style in line


def test_describe_org_account_quota_handles_lookup_error():
    """A lookup failure renders as 'unknown' rather than breaking env show."""
    qm = MagicMock()
    qm.diagnose_org_account_quota = MagicMock(side_effect=RuntimeError("boom"))

    assert _describe_org_account_quota(qm) == "[dim]unknown[/dim]"
