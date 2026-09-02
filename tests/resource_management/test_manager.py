"""Tests for top-level resource management API."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.account_management.models import TestEnvironment
from aws_bench.resource_management.cleanup.models import (
    AccountCleanupResult,
    CleanupSummary,
    RegionResult,
)
from aws_bench.resource_management.manager import ResourceManager
from aws_bench.resource_management.snapshot.models import (
    SnapshotResult,
    SnapshotStage,
)
from aws_bench.resource_management.verify.models import (
    AccountVerifyResult,
    RegionVerifyResult,
)
from aws_bench.scenario.scenario import Scenario


def test_cleanup_account_creates_cleanup_manager_and_delegates():
    """Test cleanup_account creates CleanupManager and calls cleanup_all_stacks."""
    mock_cleanup_manager = Mock()
    mock_summary = CleanupSummary(
        regions=[RegionResult(region="us-east-1", stacks_found=5, stacks_deleted=5)]
    )
    mock_cleanup_manager.cleanup_all_stacks = AsyncMock(return_value=mock_summary)

    mock_session = Mock()
    mock_cred_provider = Mock()
    mock_cred_provider.get_session_for_account.return_value = mock_session

    with (
        patch(
            "aws_bench.resource_management.manager.CredentialProvider.get",
            return_value=mock_cred_provider,
        ),
        patch(
            "aws_bench.resource_management.manager.CleanupManager",
            return_value=mock_cleanup_manager,
        ) as mock_cleanup_cls,
    ):
        result = asyncio.run(ResourceManager.cleanup_account("123456789012", regions=["us-east-1"]))

        mock_cred_provider.get_session_for_account.assert_called_once_with(
            "123456789012", "OrganizationAccountAccessRole", "app-rm-cleanup-account"
        )
        mock_cleanup_cls.assert_called_once_with(
            mock_session, env_name=None, account_id="123456789012"
        )
        mock_cleanup_manager.cleanup_all_stacks.assert_called_once_with(regions=["us-east-1"])
        assert result == mock_summary


def test_cleanup_account_discovers_regions_when_none():
    """Test cleanup_account allows region discovery when regions=None."""
    mock_cleanup_manager = Mock()
    mock_cleanup_manager.cleanup_all_stacks = AsyncMock(return_value=CleanupSummary())

    mock_session = Mock()
    mock_cred_provider = Mock()
    mock_cred_provider.get_session_for_account.return_value = mock_session

    with (
        patch(
            "aws_bench.resource_management.manager.CredentialProvider.get",
            return_value=mock_cred_provider,
        ),
        patch(
            "aws_bench.resource_management.manager.CleanupManager",
            return_value=mock_cleanup_manager,
        ),
    ):
        asyncio.run(ResourceManager.cleanup_account("123456789012"))

        mock_cleanup_manager.cleanup_all_stacks.assert_called_once_with(regions=None)


def test_cleanup_stack_creates_cleanup_manager_and_delegates():
    """Test cleanup_stack creates CleanupManager and calls cleanup_stack."""
    mock_cleanup_manager = Mock()
    mock_summary = CleanupSummary(
        regions=[RegionResult(region="us-east-1", stacks_found=1, stacks_deleted=1)]
    )
    mock_cleanup_manager.cleanup_stack = AsyncMock(return_value=mock_summary)

    mock_session = Mock()
    mock_cred_provider = Mock()
    mock_cred_provider.get_session_for_account.return_value = mock_session

    with (
        patch(
            "aws_bench.resource_management.manager.CredentialProvider.get",
            return_value=mock_cred_provider,
        ),
        patch(
            "aws_bench.resource_management.manager.CleanupManager",
            return_value=mock_cleanup_manager,
        ) as mock_cleanup_cls,
    ):
        result = asyncio.run(ResourceManager.cleanup_stack("123456789012", "my-stack"))

        mock_cred_provider.get_session_for_account.assert_called_once_with(
            "123456789012", "OrganizationAccountAccessRole", "app-rm-cleanup-stack"
        )
        mock_cleanup_cls.assert_called_once_with(mock_session, account_id="123456789012")
        mock_cleanup_manager.cleanup_stack.assert_called_once_with("my-stack")
        assert result == mock_summary


def test_cleanup_stack_propagates_value_error():
    """Test cleanup_stack propagates ValueError from stack not found."""
    mock_cleanup_manager = Mock()
    mock_cleanup_manager.cleanup_stack = AsyncMock(
        side_effect=ValueError("Stack 'nonexistent' not found")
    )

    mock_session = Mock()
    mock_cred_provider = Mock()
    mock_cred_provider.get_session_for_account.return_value = mock_session

    with (
        patch(
            "aws_bench.resource_management.manager.CredentialProvider.get",
            return_value=mock_cred_provider,
        ),
        patch(
            "aws_bench.resource_management.manager.CleanupManager",
            return_value=mock_cleanup_manager,
        ),
    ):
        with pytest.raises(ValueError, match="not found"):
            asyncio.run(ResourceManager.cleanup_stack("123456789012", "nonexistent"))


def test_cleanup_account_validates_account_id():
    """Test cleanup_account validates account_id format."""
    with pytest.raises(ValueError, match="12-digit string"):
        asyncio.run(ResourceManager.cleanup_account("invalid"))

    with pytest.raises(ValueError, match="12-digit string"):
        asyncio.run(ResourceManager.cleanup_account("12345"))  # Too short

    with pytest.raises(ValueError, match="non-empty string"):
        asyncio.run(ResourceManager.cleanup_account(""))


def test_cleanup_stack_validates_account_id():
    """Test cleanup_stack validates account_id format."""
    with pytest.raises(ValueError, match="12-digit string"):
        asyncio.run(ResourceManager.cleanup_stack("invalid", "my-stack"))

    with pytest.raises(ValueError, match="12-digit string"):
        asyncio.run(ResourceManager.cleanup_stack("12345", "my-stack"))

    with pytest.raises(ValueError, match="non-empty string"):
        asyncio.run(ResourceManager.cleanup_stack("", "my-stack"))


def test_cleanup_stack_validates_stack_name():
    """Test cleanup_stack validates stack_name format."""
    with pytest.raises(ValueError, match="stack_name must be a non-empty string"):
        asyncio.run(ResourceManager.cleanup_stack("123456789012", ""))

    with pytest.raises(ValueError, match="leading/trailing whitespace"):
        asyncio.run(ResourceManager.cleanup_stack("123456789012", " my-stack"))

    with pytest.raises(ValueError, match="leading/trailing whitespace"):
        asyncio.run(ResourceManager.cleanup_stack("123456789012", "my-stack "))


def _verify_scenario(name: str) -> Scenario:
    # Duck-typed stub: verify_environment only reads .name and .scenario_dir.
    return cast(Scenario, SimpleNamespace(name=name, scenario_dir=f"/tmp/{name}"))


def _verify_test_env() -> TestEnvironment:
    # Duck-typed stub: verify_environment only reads .ou_name and .mapping_for.
    return cast(
        TestEnvironment,
        SimpleNamespace(
            ou_name="prod-bench",
            mapping_for=lambda name: {"PRIMARY": "111111111111"},
        ),
    )


@pytest.mark.asyncio
async def test_verify_environment_all_pass():
    scenarios = [_verify_scenario("ec2-small"), _verify_scenario("s3-versioning")]

    async def fake_verify(*, scenario_name, scenario_dir, account_mapping, region):
        return [
            AccountVerifyResult(
                account_id="111111111111",
                environment_id=scenario_name,
                success=True,
                region_results=[RegionVerifyResult(region="us-east-1", success=True)],
            )
        ]

    with patch(
        "aws_bench.resource_management.manager.ResourceManager.verify_scenario",
        new=AsyncMock(side_effect=fake_verify),
    ):
        report = await ResourceManager.verify_environment(
            _verify_test_env(), scenarios, n_concurrent=4
        )

    assert report.passed is True
    assert report.env_name == "prod-bench"
    assert {r.environment_id for r in report.results} == {"ec2-small", "s3-versioning"}


@pytest.mark.asyncio
async def test_verify_environment_one_failure_marks_not_passed():
    scenarios = [_verify_scenario("ec2-small"), _verify_scenario("ec2-multiregion")]

    async def fake_verify(*, scenario_name, scenario_dir, account_mapping, region):
        ok = scenario_name == "ec2-small"
        return [
            AccountVerifyResult(
                account_id="111111111111",
                environment_id=scenario_name,
                success=ok,
                region_results=[RegionVerifyResult(region="us-east-1", success=ok)],
                error_message=None if ok else "drift",
            )
        ]

    with patch(
        "aws_bench.resource_management.manager.ResourceManager.verify_scenario",
        new=AsyncMock(side_effect=fake_verify),
    ):
        report = await ResourceManager.verify_environment(
            _verify_test_env(), scenarios, n_concurrent=4
        )

    assert report.passed is False
    assert len(report.results) == 2
    assert any(not r.success for r in report.results)


@pytest.mark.asyncio
async def test_verify_environment_empty_scenarios_passes():
    """No scenarios → no verification work, passed=True (vacuously)."""
    with patch(
        "aws_bench.resource_management.manager.ResourceManager.verify_scenario",
        new=AsyncMock(),
    ) as verify_spy:
        report = await ResourceManager.verify_environment(_verify_test_env(), [], n_concurrent=4)

    verify_spy.assert_not_called()
    assert report.passed is True
    assert report.results == []
    assert report.env_name == "prod-bench"


def test_capture_pre_setup_baseline_builds_context_and_delegates():
    """Builds one PRE_SETUP context and delegates to SnapshotManager.snapshot_account.

    The context carries an empty hash and the passed-in regions; the account's
    session comes from the credential provider.
    """
    mock_session = Mock()
    cred_provider = Mock()
    cred_provider.get_session_for_account.return_value = mock_session

    expected = SnapshotResult(account_id="111111111111", success=True)
    mock_mgr = Mock()
    mock_mgr.snapshot_exists.return_value = False
    mock_mgr.snapshot_account.return_value = expected

    with patch("aws_bench.resource_management.manager.SnapshotManager", return_value=mock_mgr):
        result = ResourceManager.capture_pre_setup_baseline(
            "111111111111",
            "scn-a",
            ["us-east-1", "us-west-2"],
            cred_provider=cred_provider,
        )

    assert result is expected
    mock_mgr.snapshot_exists.assert_called_once_with(
        "scn-a", "111111111111", SnapshotStage.PRE_SETUP
    )
    assert cred_provider.get_session_for_account.call_args.args[0] == "111111111111"
    assert cred_provider.get_session_for_account.call_args.args[1] == ORG_ACCESS_ROLE
    session_arg, account_arg, ctx = mock_mgr.snapshot_account.call_args.args
    assert session_arg is mock_session
    assert account_arg == "111111111111"
    assert ctx.stage == SnapshotStage.PRE_SETUP
    assert ctx.scenario_hash == ""
    assert ctx.regions == ["us-east-1", "us-west-2"]
    assert ctx.account_ids == ["111111111111"]


def test_capture_pre_setup_baseline_skips_when_baseline_exists():
    """Idempotency: an existing PRE_SETUP baseline is not recaptured; returns success."""
    cred_provider = Mock()
    mock_mgr = Mock()
    mock_mgr.snapshot_exists.return_value = True

    with patch("aws_bench.resource_management.manager.SnapshotManager", return_value=mock_mgr):
        result = ResourceManager.capture_pre_setup_baseline(
            "111111111111", "scn-a", ["us-east-1"], cred_provider=cred_provider
        )

    assert result.success is True
    assert result.account_id == "111111111111"
    assert result.regions_captured == ["us-east-1"]
    mock_mgr.snapshot_account.assert_not_called()
    cred_provider.get_session_for_account.assert_not_called()


def test_sweep_scenario_residuals_by_name_fans_out_per_account(tmp_path):
    """One CleanupManager.sweep_post_setup_residuals call per account, forwarding regions."""
    fake_provider = Mock()
    fake_provider.get_session_for_account.return_value = Mock()

    created: list[str] = []
    swept_regions: list[list[str] | None] = []

    class FakeCleanupManager:
        def __init__(
            self, session, *, output_dir=None, env_name=None, account_id: str = ""
        ) -> None:
            created.append(account_id)

        async def sweep_post_setup_residuals(
            self, regions: list[str] | None = None, *, all_regions: bool
        ) -> None:
            swept_regions.append(regions)

    with (
        patch(
            "aws_bench.resource_management.manager.CredentialProvider.get",
            return_value=fake_provider,
        ),
        patch("aws_bench.resource_management.manager.CleanupManager", FakeCleanupManager),
    ):
        results = asyncio.run(
            ResourceManager.sweep_scenario_residuals_by_name(
                scenario_name="sc",
                account_mapping={"PRIMARY": "111111111111", "SECONDARY": "222222222222"},
                max_concurrent=2,
                all_regions=False,
                output_dir=tmp_path,
                regions=["us-east-1", "eu-west-1"],
            )
        )

    assert sorted(created) == ["111111111111", "222222222222"]
    # Scenario regions are forwarded to every account's sweep (the no-baseline floor).
    assert swept_regions == [["us-east-1", "eu-west-1"], ["us-east-1", "eu-west-1"]]
    assert all(isinstance(r, AccountCleanupResult) for r in results)
    assert all(r.error is None for r in results)


def test_sweep_scenario_residuals_by_name_captures_account_error(tmp_path):
    """A per-account sweep exception is captured as AccountCleanupResult.error, not raised."""
    fake_provider = Mock()
    fake_provider.get_session_for_account.return_value = Mock()

    class FailingCleanupManager:
        def __init__(
            self, session, *, output_dir=None, env_name=None, account_id: str = ""
        ) -> None:
            pass

        async def sweep_post_setup_residuals(
            self, regions: list[str] | None = None, *, all_regions: bool
        ) -> None:
            raise RuntimeError("sweep blew up")

    with (
        patch(
            "aws_bench.resource_management.manager.CredentialProvider.get",
            return_value=fake_provider,
        ),
        patch("aws_bench.resource_management.manager.CleanupManager", FailingCleanupManager),
    ):
        results = asyncio.run(
            ResourceManager.sweep_scenario_residuals_by_name(
                scenario_name="sc",
                account_mapping={"PRIMARY": "111111111111"},
                max_concurrent=1,
                all_regions=False,
                output_dir=tmp_path,
            )
        )

    assert len(results) == 1
    assert results[0].error is not None
    assert "sweep blew up" in results[0].error
