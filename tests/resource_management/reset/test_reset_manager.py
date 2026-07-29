"""Tests for ResetManager."""

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import boto3
import pytest
from moto import mock_aws

from aws_bench.account_management.models import ScenarioAccount
from aws_bench.resource_management import deferred
from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.reset.manager import ResetManager
from aws_bench.resource_management.reset.models import ResetFailure, ResetResult, RestoreOutcome
from aws_bench.resource_management.snapshot.models import (
    DriftBaseline,
    ResourceDrift,
    Snapshot,
    StackMetadata,
)
from aws_bench.resource_management.verify.models import VerifyResult


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    return tmp_path / "output"


@pytest.fixture
def sample_snapshot():
    """Create a sample snapshot for testing."""
    return Snapshot(
        timestamp=datetime(2026, 4, 22, 10, 30, 0, tzinfo=timezone.utc),
        account_id="123456789012",
        environment_id="env-test",
        scenario_hash="abc123",
        drift_baseline={
            "test-stack": DriftBaseline(
                detection_status="DETECTION_COMPLETE",
                resource_drifts=[ResourceDrift("MyRole", "IN_SYNC", [])],
            )
        },
        stack_metadata={
            "test-stack": StackMetadata(
                status="CREATE_COMPLETE",
                template_hash="sha256:test123",
                parameters={},
                tags={},
            )
        },
        resource_ids={"AWS::IAM::Role": [{"Identifier": "MyRole"}]},
        regions=[],
    )


# ===========================================================================
# ResetManager initialization
# ===========================================================================


@mock_aws
def test_reset_manager_initialization(temp_output_dir):
    """Test ResetManager initialization."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    assert manager._session == session
    assert manager._verify_mgr is not None
    assert manager._cleanup_mgr is not None
    assert manager._stack_restorer is not None


@mock_aws
def test_reset_manager_threads_account_id_into_verify_manager(temp_output_dir):
    """ResetManager passes its account id to the account-level VerifyManager it builds.

    This is the ``__init__`` VerifyManager (``self._verify_mgr``), used for
    ``load_snapshot`` and the region default; its account id keeps that scan-account
    context correct.
    """
    session = boto3.Session(region_name="us-east-1")

    with patch("aws_bench.resource_management.reset.manager.VerifyManager") as mock_verify_cls:
        ResetManager(session, output_dir=temp_output_dir, account_id="123456789012")

    assert mock_verify_cls.call_args.kwargs.get("account_id") == "123456789012"


@mock_aws
def test_reset_region_threads_account_id_into_scanning_verify_manager(
    temp_output_dir, sample_snapshot
):
    """The region-scoped VerifyManager built in _reset_region carries the account id.

    This is the VerifyManager that actually fast-scans the region (region_name set),
    so it is the one whose scan must route to the management-account Lambda. Without
    the account id it degrades to the throttled host path, where a failed lister is
    swallowed into scan_result.failed and skipped by find_new_resources — making the
    reset's verification falsely pass.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir, account_id="123456789012")

    # Patch the class so the region-scoped construction is observable, and make its
    # verify report "already at baseline" so _reset_region returns right after building it.
    with patch("aws_bench.resource_management.reset.manager.VerifyManager") as mock_verify_cls:
        mock_verify_cls._filter_snapshot_to_region = MagicMock(return_value=sample_snapshot)
        mock_verify_cls.return_value.verify_account_state.return_value = VerifyResult(
            success=True, reason="already at baseline"
        )
        asyncio.run(
            manager._reset_region("env-test", "123456789012", sample_snapshot, "us-west-2", None)
        )

    # The scanning VM is the one constructed with region_name set. Assert it also got
    # the account id — that pair is what routes the scan to the Lambda for this region.
    region_calls = [
        c for c in mock_verify_cls.call_args_list if c.kwargs.get("region_name") == "us-west-2"
    ]
    assert len(region_calls) == 1, "expected one region-scoped VerifyManager for us-west-2"
    assert region_calls[0].kwargs.get("account_id") == "123456789012"


# ===========================================================================
# reset_account — region validation
# ===========================================================================


@mock_aws
def test_reset_account_resets_each_snapshot_region(temp_output_dir, sample_snapshot):
    """Reset iterates every region the baseline covers (not the caller's region).

    Reset no longer fails on a region 'mismatch' — it resets exactly the
    regions recorded in the snapshot, verifying each against its own slice.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    snapshot_with_regions = sample_snapshot
    snapshot_with_regions.regions = ["us-east-1", "us-west-2"]

    with (
        patch(
            "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot"
        ) as mock_load,
        patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify,
    ):
        mock_load.return_value = snapshot_with_regions
        # Every region already at baseline -> nothing to do, reset succeeds.
        mock_verify.return_value = VerifyResult(success=True, reason="ok")

        result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    assert result.success
    # Verified once per snapshot region.
    assert mock_verify.call_count == 2


@mock_aws
def test_reset_account_logs_all_region_failures_before_raising(
    temp_output_dir, sample_snapshot, caplog
):
    """When regions fail concurrently, every failure is logged, not just the first."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    snapshot_with_regions = sample_snapshot
    snapshot_with_regions.regions = ["us-east-1", "us-west-2"]

    async def failing_region(env_name, account_id, snapshot, region, scenario_dir):
        raise ResetFailure(reason=f"{region} broke", details={}, suggestion="")

    with (
        patch(
            "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot",
            return_value=snapshot_with_regions,
        ),
        patch.object(manager, "_reset_region", side_effect=failing_region),
        caplog.at_level(logging.ERROR),
    ):
        result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    # A ResetFailure is caught and returned as an unsuccessful result.
    assert not result.success
    # BOTH regions' failures were logged, not just the one that surfaced.
    assert "us-east-1' reset failed" in caplog.text
    assert "us-west-2' reset failed" in caplog.text


@mock_aws
def test_reset_account_region_matches(temp_output_dir, sample_snapshot):
    """Test reset proceeds when region matches snapshot."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    # Set verify manager region and snapshot regions
    manager._verify_mgr._region_name = "us-east-1"
    snapshot_with_regions = sample_snapshot
    snapshot_with_regions.regions = ["us-east-1", "us-west-2"]

    # Mock snapshot and verification
    with patch(
        "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot"
    ) as mock_load:
        with patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify:
            mock_load.return_value = snapshot_with_regions
            mock_verify.return_value = VerifyResult(
                success=True, reason="Account in baseline state"
            )

            result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    assert result.success
    assert "baseline" in result.reason.lower()


@mock_aws
def test_reset_account_no_region_validation_when_no_region_set(temp_output_dir, sample_snapshot):
    """Test reset skips region validation when region_name not set."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    # Ensure verify manager has no region set
    manager._verify_mgr._region_name = None
    snapshot_with_regions = sample_snapshot
    snapshot_with_regions.regions = ["us-west-2"]

    # Mock snapshot and verification
    with patch(
        "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot"
    ) as mock_load:
        with patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify:
            mock_load.return_value = snapshot_with_regions
            mock_verify.return_value = VerifyResult(
                success=True, reason="Account in baseline state"
            )

            result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    assert result.success


@mock_aws
def test_reset_account_skips_region_validation_with_empty_snapshot_regions(
    temp_output_dir, sample_snapshot
):
    """Test reset skips region validation when snapshot has no regions."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    # Set verify manager region but snapshot has no regions
    manager._verify_mgr._region_name = "us-west-2"
    snapshot_without_regions = sample_snapshot
    snapshot_without_regions.regions = []

    # Mock snapshot and verification
    with patch(
        "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot"
    ) as mock_load:
        with patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify:
            mock_load.return_value = snapshot_without_regions
            mock_verify.return_value = VerifyResult(
                success=True, reason="Account in baseline state"
            )

            result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    assert result.success


# ===========================================================================
# reset_account — already in baseline state
# ===========================================================================


@mock_aws
def test_reset_account_already_in_baseline(temp_output_dir, sample_snapshot):
    """Test reset when account is already in baseline state."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    # Mock snapshot and verification
    with patch(
        "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot"
    ) as mock_load:
        with patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify:
            mock_load.return_value = sample_snapshot
            mock_verify.return_value = VerifyResult(
                success=True, reason="Account in baseline state"
            )

            result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    assert result.success
    assert "baseline" in result.reason.lower()


# ===========================================================================
# reset_account — dataset mismatch (unrecoverable)
# ===========================================================================


@mock_aws
def test_reset_account_dataset_mismatch(temp_output_dir, sample_snapshot):
    """Test reset fails when dataset mismatch detected."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    # Mock dataset mismatch
    with patch(
        "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot"
    ) as mock_load:
        with patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify:
            mock_load.return_value = sample_snapshot
            mock_verify.return_value = VerifyResult(
                success=False,
                reason="Dataset version mismatch",
                is_dataset_mismatch=True,
                suggestion="Run cleanup and setup",
            )

            result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    assert not result.success
    assert "mismatch" in result.reason.lower()


# ===========================================================================
# reset_account — delete new resources
# ===========================================================================


@mock_aws
def test_reset_account_deletes_new_resources(temp_output_dir, sample_snapshot):
    """Reset deletes new resources via the shared cleanup pipeline.

    The pipeline (ResourceCleaner) handles CCAPI-unsupported types like
    S3 buckets via service-API custom handlers, unlike raw CCAPI.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    new_resources = {
        "AWS::S3::Bucket": [{"Identifier": "new-bucket"}],
        "AWS::IAM::Role": [{"Identifier": "NewRole"}],
    }

    # Mock verification with new resources
    with patch(
        "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot"
    ) as mock_load:
        with patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify:
            with patch(
                "aws_bench.resource_management.reset.manager.ResourceCleaner"
            ) as mock_cleaner_cls:
                mock_load.return_value = sample_snapshot

                # First call: verification finds new resources
                # Second call: final verification passes
                mock_verify.side_effect = [
                    VerifyResult(
                        success=False,
                        reason="Found 2 new resources",
                        new_resources=new_resources,
                    ),
                    VerifyResult(success=True, reason="Account in baseline state"),
                ]

                # Cleanup pipeline reports no failures.
                cleanup_calls: list[list[str]] = []

                async def _cleanup(stack_resources, *args, **kwargs):
                    cleanup_calls.append([r.resource_type for r in stack_resources])
                    return {}

                mock_cleaner = mock_cleaner_cls.return_value
                mock_cleaner.cleanup = _cleanup

                # Global re-check confirms the IAM role was deleted.
                with patch(
                    "aws_bench.resource_management.reset.manager.CloudControlManager"
                ) as mock_ccm:
                    mock_ccm.return_value.resource_exists.return_value = False
                    result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    assert result.success
    # The regional S3 bucket is deleted in the per-region pass; the global IAM
    # role is deferred and deleted once in the final account-level pass — so the
    # cleaner runs twice, with the two resource classes split across the passes.
    assert cleanup_calls == [["AWS::S3::Bucket"], ["AWS::IAM::Role"]]


# ===========================================================================
# reset_account — fix stack drift
# ===========================================================================


@mock_aws
def test_reset_account_fixes_drift(temp_output_dir, sample_snapshot, tmp_path):
    """Test reset fixes stack drift."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    drift_differences = {
        "test-stack": {
            "baseline": [{"LogicalResourceId": "MyRole", "StackResourceDriftStatus": "IN_SYNC"}],
            "current": [{"LogicalResourceId": "MyRole", "StackResourceDriftStatus": "MODIFIED"}],
        }
    }

    # Mock verification with drift
    with patch(
        "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot"
    ) as mock_load:
        with patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify:
            with patch(
                "aws_bench.resource_management.reset.stack_restorer.StackRestorer.restore_stack"
            ) as mock_restore:
                mock_load.return_value = sample_snapshot

                # First call: verification finds drift
                # Second call: final verification passes
                mock_verify.side_effect = [
                    VerifyResult(
                        success=False,
                        reason="1 stack has different drift",
                        drift_differences=drift_differences,
                    ),
                    VerifyResult(success=True, reason="Account in baseline state"),
                ]

                # Make the mock coroutine
                async def mock_restore_coro(*args, **kwargs):
                    return RestoreOutcome.RESTORED

                mock_restore.side_effect = mock_restore_coro

                result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    assert result.success
    # Verify restore was called
    mock_restore.assert_called_once()


@mock_aws
def test_reset_account_flags_redeploy_when_stack_deleted(temp_output_dir, sample_snapshot):
    """When a stack is deleted for re-setup, reset succeeds with needs_redeploy."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    drift_differences = {
        "test-stack": {
            "baseline": [{"LogicalResourceId": "MyRole", "StackResourceDriftStatus": "IN_SYNC"}],
            "current": [{"LogicalResourceId": "MyRole", "StackResourceDriftStatus": "MODIFIED"}],
        }
    }

    with (
        patch(
            "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot"
        ) as mock_load,
        patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify,
        patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.find_orphan_resources",
            return_value=None,
        ) as mock_orphan,
        patch(
            "aws_bench.resource_management.reset.stack_restorer.StackRestorer.restore_stack"
        ) as mock_restore,
    ):
        mock_load.return_value = sample_snapshot

        # Verification finds drift; the region-wide stack/drift re-verify should NOT
        # run because the stack is deleted (redeploy pending) — but the orphan
        # census still runs as the fail-closed backstop and finds nothing here.
        mock_verify.return_value = VerifyResult(
            success=False,
            reason="1 stack has different drift",
            drift_differences=drift_differences,
        )

        async def mock_restore_coro(*args, **kwargs):
            return RestoreOutcome.DELETED_NEEDS_REDEPLOY

        mock_restore.side_effect = mock_restore_coro

        result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    assert result.success
    assert result.needs_redeploy is True
    assert result.details is not None
    assert result.details["deleted_stacks"] == ["test-stack"]
    # Final stack/drift verify skipped (only the initial verify ran); the orphan
    # census ran as the backstop and was clean.
    assert mock_verify.call_count == 1
    mock_orphan.assert_called_once()


@mock_aws
def test_reset_account_fails_closed_on_orphan_after_stack_delete(temp_output_dir, sample_snapshot):
    """A stack delete must NOT suppress a still-present orphan resource.

    Regression for the loophole: a leaked resource that reset tried and failed to
    delete used to be masked by the deleted-stack early return, then absorbed into a
    fresh baseline. Now the orphan census runs as a fail-closed backstop — reset
    must FAIL and surface the orphan by name in unresolved_orphans.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    drift_differences = {
        "test-stack": {
            "baseline": [{"LogicalResourceId": "MyRole", "StackResourceDriftStatus": "IN_SYNC"}],
            "current": [{"LogicalResourceId": "MyRole", "StackResourceDriftStatus": "MODIFIED"}],
        }
    }
    orphan = {"AWS::EC2::InternetGateway": [{"Identifier": "igw-leaked"}]}

    with (
        patch(
            "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot"
        ) as mock_load,
        patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify,
        patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.find_orphan_resources",
            return_value=VerifyResult(
                success=False,
                reason="Found 1 new resource(s)",
                new_resources=orphan,
                suggestion="Run 'aws-bench env reset' to remove new resources",
            ),
        ) as mock_orphan,
        patch(
            "aws_bench.resource_management.reset.stack_restorer.StackRestorer.restore_stack"
        ) as mock_restore,
    ):
        mock_load.return_value = sample_snapshot
        mock_verify.return_value = VerifyResult(
            success=False,
            reason="1 stack has different drift",
            drift_differences=drift_differences,
        )

        async def mock_restore_coro(*args, **kwargs):
            return RestoreOutcome.DELETED_NEEDS_REDEPLOY

        mock_restore.side_effect = mock_restore_coro

        result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    # The deleted stack did NOT suppress the orphan: reset failed and named it.
    mock_orphan.assert_called_once()
    assert result.success is False
    assert result.needs_redeploy is False
    assert result.unresolved_orphans == orphan


# ===========================================================================
# reset_account — co-occurring new resources + DELETE_FAILED stack
# ===========================================================================


@mock_aws
def test_reset_account_remediates_delete_failed_stack_alongside_new_resources(
    temp_output_dir, sample_snapshot
):
    """A DELETE_FAILED stack is remediated even when new resources also fail.

    Regression for the prod wedge: reset's diagnosis reported only the first
    failing category (3 AWS-managed RAM permissions), so a co-occurring
    DELETE_FAILED stack was never routed to _delete_for_resetup. With the
    aggregating diagnosis, the single VerifyResult carries BOTH categories and
    reset must delete the stack for re-setup.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    aggregated = VerifyResult(
        success=False,
        reason="Found 3 new resource(s); 1 stack(s) have status mismatch",
        new_resources={"AWS::RAM::Permission": [{"Identifier": "arn:aws:ram::aws:permission/x"}]},
        stack_status_failures={
            "test-stack": {"expected": "CREATE_COMPLETE", "actual": "DELETE_FAILED"}
        },
    )

    with (
        patch(
            "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot"
        ) as mock_load,
        patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify,
        patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.find_orphan_resources",
            return_value=None,
        ),
        patch(
            "aws_bench.resource_management.reset.manager.ResetManager._delete_new_resources"
        ) as mock_del_new,
        patch(
            "aws_bench.resource_management.reset.stack_restorer.StackRestorer._delete_for_resetup"
        ) as mock_delete_stack,
    ):
        mock_load.return_value = sample_snapshot
        # Diagnosis returns the aggregated failure; no final stack/drift verify
        # runs because the stack is deleted for redeploy.
        mock_verify.return_value = aggregated

        async def _del_new_coro(*args, **kwargs):
            return {}  # no global resources deferred in this case

        mock_del_new.side_effect = _del_new_coro

        async def _delete_stack_coro(*args, **kwargs):
            return RestoreOutcome.DELETED_NEEDS_REDEPLOY

        mock_delete_stack.side_effect = _delete_stack_coro

        result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    # The DELETE_FAILED stack was routed to deletion-for-resetup...
    mock_delete_stack.assert_called_once()
    assert mock_delete_stack.call_args.args[0] == "test-stack"
    # ...and the new-resource deletion also ran (both categories handled).
    mock_del_new.assert_called_once()
    assert result.success
    assert result.needs_redeploy is True
    assert result.details is not None
    assert result.details["deleted_stacks"] == ["test-stack"]


@mock_aws
def test_reset_account_recreates_drift_undetectable_stack(temp_output_dir, sample_snapshot):
    """A stack that regressed to undetectable drift is deleted for re-setup.

    The benchmark agent can delete a CFN-tracked resource (e.g. an ECS task-def
    revision), after which DetectStackDrift fails forever and in-place revert
    no-ops. Reset must delete the stack so setup recreates it with a fresh,
    drift-readable resource.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    aggregated = VerifyResult(
        success=False,
        reason="1 stack(s) no longer drift-detectable (will be recreated): ecsroll-stack",
        drift_undetectable=["ecsroll-stack"],
    )

    with (
        patch(
            "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot"
        ) as mock_load,
        patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify,
        patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.find_orphan_resources",
            return_value=None,
        ),
        patch(
            "aws_bench.resource_management.reset.stack_restorer.StackRestorer._delete_for_resetup"
        ) as mock_delete_stack,
    ):
        mock_load.return_value = sample_snapshot
        mock_verify.return_value = aggregated

        async def _delete_stack_coro(*args, **kwargs):
            return RestoreOutcome.DELETED_NEEDS_REDEPLOY

        mock_delete_stack.side_effect = _delete_stack_coro

        result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    # Deleted for re-setup (classification already proved it's permanently gone).
    mock_delete_stack.assert_called_once()
    assert mock_delete_stack.call_args.args[0] == "ecsroll-stack"
    assert result.success
    assert result.needs_redeploy is True
    assert result.details is not None
    assert result.details["deleted_stacks"] == ["ecsroll-stack"]
    # Final stack/drift verify skipped (stack deleted for redeploy) — only initial ran.
    assert mock_verify.call_count == 1


@mock_aws
def test_reset_account_drift_undetectable_delete_failure_raises(temp_output_dir, sample_snapshot):
    """If deleting an undetectable stack fails, reset surfaces ResetFailure."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    aggregated = VerifyResult(
        success=False,
        reason="1 stack(s) no longer drift-detectable (will be recreated): stuck-stack",
        drift_undetectable=["stuck-stack"],
    )

    with patch(
        "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot"
    ) as mock_load:
        with patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify:
            with patch(
                "aws_bench.resource_management.reset.stack_restorer."
                "StackRestorer._delete_for_resetup"
            ) as mock_delete_stack:
                mock_load.return_value = sample_snapshot
                mock_verify.return_value = aggregated

                async def _delete_stack_coro(*args, **kwargs):
                    return RestoreOutcome.FAILED

                mock_delete_stack.side_effect = _delete_stack_coro

                result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    # Delete failed → reset_account reports failure (not a silent success).
    assert result.success is False


@mock_aws
def test_reset_account_drift_undetectable_skipped_when_already_status_deleted(
    temp_output_dir, sample_snapshot
):
    """A stack in BOTH stack_status_failures and drift_undetectable is deleted once.

    Phase 2 (status) deletes it; Phase 2b must skip it via already_handled rather
    than re-delete a now-absent stack.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    aggregated = VerifyResult(
        success=False,
        reason="status + undetectable on same stack",
        stack_status_failures={"dup-stack": {"expected": "CREATE_COMPLETE", "actual": "MISSING"}},
        drift_undetectable=["dup-stack"],
    )

    with (
        patch(
            "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot"
        ) as mock_load,
        patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify,
        patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.find_orphan_resources",
            return_value=None,
        ),
        patch(
            "aws_bench.resource_management.reset.stack_restorer.StackRestorer._delete_for_resetup"
        ) as mock_delete_stack,
    ):
        mock_load.return_value = sample_snapshot
        mock_verify.return_value = aggregated

        async def _delete_stack_coro(*args, **kwargs):
            return RestoreOutcome.DELETED_NEEDS_REDEPLOY

        mock_delete_stack.side_effect = _delete_stack_coro

        result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    # The MISSING status stack is handled by Phase 2 (no delete call needed), and
    # Phase 2b must NOT act on it again (already_handled).
    mock_delete_stack.assert_not_called()
    assert result.success
    assert result.details is not None
    assert result.details["deleted_stacks"] == ["dup-stack"]


# ===========================================================================
# reset_account — template-hash mismatch (delete + redeploy)
# ===========================================================================


@mock_aws
def test_reset_account_recreates_template_mismatched_stack(temp_output_dir, sample_snapshot):
    """A stack whose template changed is deleted for re-setup, not looped on.

    The agent edited a stack's CFN template (the recurring databases-and-storage-S3
    case). UsePreviousTemplate drift-revert cannot undo a template change, so reset
    must delete the stack and flag needs_redeploy so setup recreates it from the
    correct template. Regression for RC6: template mismatch used to fall through
    every remediation phase, so final verify re-raised 'template changed'.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    aggregated = VerifyResult(
        success=False,
        reason="Stack databases-and-storage-S3 template changed",
        is_template_mismatch=True,
        template_mismatch_stacks=["databases-and-storage-S3"],
    )

    with (
        patch(
            "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot"
        ) as mock_load,
        patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify,
        patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.find_orphan_resources",
            return_value=None,
        ),
        patch(
            "aws_bench.resource_management.reset.stack_restorer.StackRestorer._delete_for_resetup"
        ) as mock_delete_stack,
    ):
        mock_load.return_value = sample_snapshot
        mock_verify.return_value = aggregated

        async def _delete_stack_coro(*args, **kwargs):
            return RestoreOutcome.DELETED_NEEDS_REDEPLOY

        mock_delete_stack.side_effect = _delete_stack_coro

        result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    # Template-changed stack routed to delete-for-resetup...
    mock_delete_stack.assert_called_once()
    assert mock_delete_stack.call_args.args[0] == "databases-and-storage-S3"
    # ...and reset succeeds with redeploy pending; final stack/drift verify is skipped.
    assert result.success
    assert result.needs_redeploy is True
    assert result.details is not None
    assert result.details["deleted_stacks"] == ["databases-and-storage-S3"]
    assert mock_verify.call_count == 1


@mock_aws
def test_reset_account_template_mismatch_is_recoverable(temp_output_dir, sample_snapshot):
    """A template mismatch must NOT bail to full cleanup like a dataset mismatch.

    is_dataset_mismatch => unrecoverable (cleanup+setup). is_template_mismatch is a
    single-stack problem => recoverable via delete+redeploy. _check_recoverable must
    not raise on it.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    # Should not raise (template mismatch is recoverable).
    manager._check_recoverable(
        VerifyResult(
            success=False,
            reason="Stack x template changed",
            is_template_mismatch=True,
            template_mismatch_stacks=["x"],
        )
    )


@mock_aws
def test_delete_unrecoverable_stacks_includes_template_mismatch(temp_output_dir):
    """Template-mismatched stacks go through the same delete-for-resetup path."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    verify_result = VerifyResult(
        success=False,
        reason="template changed",
        template_mismatch_stacks=["s3-stack"],
    )
    restorer = MagicMock()
    restorer._delete_for_resetup = AsyncMock(return_value=RestoreOutcome.DELETED_NEEDS_REDEPLOY)

    deleted = asyncio.run(manager._delete_unrecoverable_stacks(verify_result, restorer))
    assert deleted == ["s3-stack"]
    restorer._delete_for_resetup.assert_called_once()
    assert restorer._delete_for_resetup.call_args.args[0] == "s3-stack"


# ===========================================================================
# reset_account — stack restoration fails
# ===========================================================================


@mock_aws
def test_reset_account_stack_restoration_fails(temp_output_dir, sample_snapshot, tmp_path):
    """Test reset fails when stack restoration fails."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    drift_differences = {
        "test-stack": {
            "baseline": [{"LogicalResourceId": "MyRole", "StackResourceDriftStatus": "IN_SYNC"}],
            "current": [{"LogicalResourceId": "MyRole", "StackResourceDriftStatus": "MODIFIED"}],
        }
    }

    # Mock verification with drift
    with patch(
        "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot"
    ) as mock_load:
        with patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify:
            with patch(
                "aws_bench.resource_management.reset.stack_restorer.StackRestorer.restore_stack"
            ) as mock_restore:
                mock_load.return_value = sample_snapshot

                # First call: verification finds drift
                mock_verify.return_value = VerifyResult(
                    success=False,
                    reason="1 stack has different drift",
                    drift_differences=drift_differences,
                )

                # Make the mock coroutine return FAILED (restoration fails)
                async def mock_restore_coro(*args, **kwargs):
                    return RestoreOutcome.FAILED

                mock_restore.side_effect = mock_restore_coro

                result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    assert not result.success
    assert "failed to restore" in result.reason.lower()


# ===========================================================================
# reset_account — CDK directory validation
# ===========================================================================


# ===========================================================================
# reset_account — final verification fails
# ===========================================================================


@mock_aws
def test_reset_account_final_verification_fails(temp_output_dir, sample_snapshot):
    """Test reset fails when final verification still fails."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    # Mock verification that never passes
    with patch(
        "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot"
    ) as mock_load:
        with patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify:
            mock_load.return_value = sample_snapshot

            # Both calls fail verification
            mock_verify.side_effect = [
                VerifyResult(success=False, reason="Found new resources", new_resources={}),
                VerifyResult(success=False, reason="Still has issues"),
            ]

            result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    assert not result.success
    assert "verification still failing" in result.reason.lower()


# ===========================================================================
# StackRestorer.restore_stack — revert succeeds
# ===========================================================================


@mock_aws
def test_stack_restorer_restore_stack_attempts_revert(temp_output_dir):
    """Test drift fix attempts changeset creation."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)
    restorer = manager._stack_restorer

    baseline_drift = [{"LogicalResourceId": "MyRole", "StackResourceDriftStatus": "IN_SYNC"}]

    # Mock the CFN client to simulate changeset creation then a successful update.
    with (
        patch.object(restorer._cfn_client, "create_change_set") as mock_create,
        patch.object(
            restorer._cfn_client,
            "describe_change_set",
            return_value={"Status": "CREATE_COMPLETE"},
        ),
        patch.object(
            restorer._cfn_client,
            "describe_stacks",
            return_value={"Stacks": [{"StackStatus": "UPDATE_COMPLETE"}]},
        ),
        patch.object(restorer._cfn_client, "execute_change_set"),
        patch.object(restorer, "_verify_drift_matches_baseline") as mock_verify,
    ):
        mock_create.return_value = None

        async def mock_verify_coro(*args, **kwargs):
            return True

        mock_verify.side_effect = mock_verify_coro

        success = asyncio.run(
            restorer.restore_stack(
                stack_name="test-stack",
                baseline_drift=baseline_drift,
            )
        )

    # Should succeed with revert path
    assert success is RestoreOutcome.RESTORED
    # Verify changeset was attempted
    mock_create.assert_called_once()


# ===========================================================================
# StackRestorer.restore_stack — delete for re-setup when revert impossible
# ===========================================================================


@mock_aws
def test_stack_restorer_deletes_for_resetup_when_revert_fails(temp_output_dir):
    """When drift revert is impossible, the stack is deleted for re-setup."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)
    restorer = manager._stack_restorer

    baseline_drift = [{"LogicalResourceId": "MyRole", "StackResourceDriftStatus": "IN_SYNC"}]

    with patch.object(restorer, "_attempt_drift_revert") as mock_revert:
        with patch.object(restorer._cleanup_mgr, "cleanup_stack") as mock_cleanup:

            async def revert_coro(*args, **kwargs):
                return False

            mock_revert.side_effect = revert_coro

            cleanup_result = MagicMock()
            cleanup_result.all_stacks_succeeded = True

            async def cleanup_coro(*args, **kwargs):
                return cleanup_result

            mock_cleanup.side_effect = cleanup_coro

            outcome = asyncio.run(
                restorer.restore_stack(
                    stack_name="test-stack",
                    baseline_drift=baseline_drift,
                )
            )

    assert outcome is RestoreOutcome.DELETED_NEEDS_REDEPLOY
    mock_cleanup.assert_called_once_with("test-stack")


# ===========================================================================
# _delete_unrecoverable_stacks
# ===========================================================================


@mock_aws
def test_delete_unrecoverable_stacks_no_failures(temp_output_dir):
    """Returns empty when there are neither status failures nor undetectable stacks."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    verify_result = VerifyResult(success=False, reason="drift", stack_status_failures=None)
    restorer = MagicMock()

    deleted = asyncio.run(manager._delete_unrecoverable_stacks(verify_result, restorer))
    assert deleted == []


@mock_aws
def test_delete_unrecoverable_stacks_missing_stack(temp_output_dir):
    """A MISSING stack is flagged for redeploy with no delete call."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    verify_result = VerifyResult(
        success=False,
        reason="status mismatch",
        stack_status_failures={"my-stack": {"expected": "CREATE_COMPLETE", "actual": "MISSING"}},
    )
    restorer = MagicMock()

    deleted = asyncio.run(manager._delete_unrecoverable_stacks(verify_result, restorer))
    assert deleted == ["my-stack"]
    restorer._delete_for_resetup.assert_not_called()


@mock_aws
def test_delete_unrecoverable_stacks_wrong_status_deleted(temp_output_dir):
    """A wrong-status stack is deleted for re-setup."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    verify_result = VerifyResult(
        success=False,
        reason="status mismatch",
        stack_status_failures={
            "bad-stack": {"expected": "CREATE_COMPLETE", "actual": "ROLLBACK_COMPLETE"}
        },
    )
    restorer = MagicMock()
    restorer._delete_for_resetup = AsyncMock(return_value=RestoreOutcome.DELETED_NEEDS_REDEPLOY)

    deleted = asyncio.run(manager._delete_unrecoverable_stacks(verify_result, restorer))
    assert deleted == ["bad-stack"]


@mock_aws
def test_delete_unrecoverable_stacks_undetectable_deleted(temp_output_dir):
    """A drift-undetectable stack goes through the SAME delete-for-resetup path."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    verify_result = VerifyResult(
        success=False,
        reason="undetectable",
        drift_undetectable=["ecsroll-stack"],
    )
    restorer = MagicMock()
    restorer._delete_for_resetup = AsyncMock(return_value=RestoreOutcome.DELETED_NEEDS_REDEPLOY)

    deleted = asyncio.run(manager._delete_unrecoverable_stacks(verify_result, restorer))
    assert deleted == ["ecsroll-stack"]
    restorer._delete_for_resetup.assert_called_once()
    assert restorer._delete_for_resetup.call_args.args[0] == "ecsroll-stack"


@mock_aws
def test_delete_unrecoverable_stacks_dedupes_status_and_undetectable(temp_output_dir):
    """A stack in BOTH status-failures and drift_undetectable is deleted once."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    verify_result = VerifyResult(
        success=False,
        reason="both",
        stack_status_failures={
            "dup-stack": {"expected": "CREATE_COMPLETE", "actual": "DELETE_FAILED"}
        },
        drift_undetectable=["dup-stack"],
    )
    restorer = MagicMock()
    restorer._delete_for_resetup = AsyncMock(return_value=RestoreOutcome.DELETED_NEEDS_REDEPLOY)

    deleted = asyncio.run(manager._delete_unrecoverable_stacks(verify_result, restorer))
    assert deleted == ["dup-stack"]
    restorer._delete_for_resetup.assert_called_once()


@mock_aws
def test_delete_unrecoverable_stacks_delete_fails_raises(temp_output_dir):
    """Raises ResetFailure when a stack delete fails."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    verify_result = VerifyResult(
        success=False,
        reason="status mismatch",
        stack_status_failures={
            "stuck-stack": {"expected": "CREATE_COMPLETE", "actual": "DELETE_FAILED"}
        },
    )
    restorer = MagicMock()
    restorer._delete_for_resetup = AsyncMock(return_value=RestoreOutcome.FAILED)

    with pytest.raises(ResetFailure):
        asyncio.run(manager._delete_unrecoverable_stacks(verify_result, restorer))


@mock_aws
def test_restore_drifted_stacks_skips_already_handled(temp_output_dir):
    """A stack already deleted by _delete_unrecoverable_stacks is NOT drift-restored.

    Regression: with aggregating verify, a DELETE_FAILED stack lands in BOTH
    stack_status_failures AND drift_differences (drift detection is skipped for a
    DELETE_FAILED stack and recorded as a mismatch). Phase 2 deletes it; Phase 3
    must then skip it, otherwise restore_stack errors on the already-gone stack
    ('Stack not found in any region') and the whole reset reports failure.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    verify_result = VerifyResult(
        success=False,
        reason="drift",
        drift_differences={"rds-stack": {"baseline": [], "current": []}},
    )
    restorer = MagicMock()
    restorer.restore_stack = AsyncMock(return_value=RestoreOutcome.RESTORED)

    deleted = asyncio.run(
        manager._restore_drifted_stacks(verify_result, restorer, already_handled={"rds-stack"})
    )

    assert deleted == []
    restorer.restore_stack.assert_not_called()


@mock_aws
def test_restore_drifted_stacks_restores_unhandled(temp_output_dir):
    """A drifted stack NOT already handled is still restored normally."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    verify_result = VerifyResult(
        success=False,
        reason="drift",
        drift_differences={"other-stack": {"baseline": [], "current": []}},
    )
    restorer = MagicMock()
    restorer.restore_stack = AsyncMock(return_value=RestoreOutcome.RESTORED)

    deleted = asyncio.run(
        manager._restore_drifted_stacks(verify_result, restorer, already_handled={"rds-stack"})
    )

    assert deleted == []
    restorer.restore_stack.assert_called_once()


# ===========================================================================
# reset_scenarios (static async method)
# ===========================================================================


def test_reset_scenarios_fans_out(monkeypatch):
    """Test reset_scenarios processes accounts concurrently."""

    async def fake_reset_account(self, env_name, account_id, scenario_dir=None):
        return ResetResult(success=True, reason="OK")

    monkeypatch.setattr(ResetManager, "reset_account", fake_reset_account)
    monkeypatch.setattr(
        "aws_bench.resource_management.reset.manager.CredentialProvider",
        MagicMock(),
    )

    accounts = [
        ScenarioAccount(
            account_id="111111111111",
            email="111@example.com",
            scenario_name="scenario-a",
            account_tag="main",
            status="ACTIVE",
        ),
        ScenarioAccount(
            account_id="222222222222",
            email="222@example.com",
            scenario_name="scenario-a",
            account_tag="main",
            status="ACTIVE",
        ),
    ]

    results = asyncio.run(ResetManager.reset_scenarios("test-ou", accounts, max_concurrent=2))

    assert len(results) == 2
    assert all(isinstance(r, ResetResult) and r.success for r in results)


def test_reset_scenarios_handles_exception(monkeypatch):
    """Test reset_scenarios returns exceptions for failed accounts."""

    async def failing_reset(self, env_name, account_id, scenario_dir=None):
        raise RuntimeError("credential failure")

    monkeypatch.setattr(ResetManager, "reset_account", failing_reset)
    monkeypatch.setattr(
        "aws_bench.resource_management.reset.manager.CredentialProvider",
        MagicMock(),
    )

    accounts = [
        ScenarioAccount(
            account_id="111111111111",
            email="111@example.com",
            scenario_name="env-1",
            account_tag="main",
            status="ACTIVE",
        ),
    ]

    results = asyncio.run(ResetManager.reset_scenarios("test-ou", accounts, max_concurrent=2))

    assert len(results) == 1
    assert isinstance(results[0], Exception)
    assert "credential failure" in str(results[0])


# ===========================================================================
# reset — global-resource deletion ordering (cross-region race fix)
# ===========================================================================


def test_delete_new_resources_defers_globals_and_deletes_only_regional():
    """A region deletes its regional new resources but defers global ones.

    Global (account-wide) types like ``AWS::IAM::Role`` surface in every
    region's scan; deleting one here would let a region without a dependent
    race-delete a global resource another region's dependent still needs. So
    ``_delete_new_resources`` deletes only the region's regional resources,
    marks the globals deferred (so this region's final verify skips them), and
    returns them for the account-level pass.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session)

    verify_result = VerifyResult(
        success=False,
        reason="Found new resources",
        new_resources={
            "AWS::S3::Bucket": [{"Identifier": "regional-bucket"}],
            "AWS::IAM::Role": [{"Identifier": "GlobalRole"}],
        },
    )

    deleted_types: list[str] = []

    async def _cleanup(stack_resources, *args, **kwargs):
        deleted_types.extend(r.resource_type for r in stack_resources)
        return {}

    with patch("aws_bench.resource_management.reset.manager.ResourceCleaner") as mock_cleaner_cls:
        mock_cleaner_cls.return_value.cleanup = _cleanup
        with deferred.deferred_scope():
            globals_seen = asyncio.run(
                manager._delete_new_resources(verify_result, session, "us-east-1")
            )
            # The global role is marked deferred so the region's final verify skips it.
            assert deferred.is_deferred("AWS::IAM::Role", "GlobalRole")
            assert not deferred.is_deferred("AWS::S3::Bucket", "regional-bucket")

    # Only the regional resource was deleted in-region; the global was returned.
    assert deleted_types == ["AWS::S3::Bucket"]
    assert globals_seen == {"AWS::IAM::Role": [{"Identifier": "GlobalRole"}]}


def test_reset_account_deletes_multiregion_global_once_in_final_pass(temp_output_dir):
    """A global resource seen in every region is deleted exactly once, last.

    Reproduces the compute-and-data KB contamination: the KB's global IAM
    execution role appears in all regions. Each region defers it; the final
    account-level pass deletes it once, after every region (incl. the one whose
    KB delete must assume the role) has drained.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    snapshot = Snapshot(
        timestamp=datetime(2026, 4, 22, 10, 30, 0, tzinfo=timezone.utc),
        account_id="123456789012",
        environment_id="env-test",
        scenario_hash="abc123",
        drift_baseline={},
        stack_metadata={},
        resource_ids={},
        regions=["us-east-1", "us-east-2", "ap-southeast-1"],
    )
    # The same global role surfaces in every region's scan.
    global_role = {"AWS::IAM::Role": [{"Identifier": "SharedGlobalRole"}]}

    cleanup_calls: list[list[str]] = []

    async def _cleanup(stack_resources, *args, **kwargs):
        cleanup_calls.append([r.physical_id for r in stack_resources])
        return {}

    def _verify(*args, **kwargs):
        # Diagnosis finds the global role; the region's final verify excludes it
        # once it has been deferred (mirroring the real _check_new_resources,
        # which drops deferred resources). Robust under concurrent regions —
        # no reliance on call ordering.
        if deferred.is_deferred("AWS::IAM::Role", "SharedGlobalRole"):
            return VerifyResult(success=True, reason="clean")
        return VerifyResult(success=False, reason="new", new_resources=global_role)

    with patch(
        "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot",
        return_value=snapshot,
    ):
        with patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state",
            side_effect=_verify,
        ):
            with patch(
                "aws_bench.resource_management.reset.manager.ResourceCleaner"
            ) as mock_cleaner_cls:
                mock_cleaner_cls.return_value.cleanup = _cleanup
                # The final global pass authoritatively re-checks existence; the
                # role was deleted, so it reports gone.
                with patch(
                    "aws_bench.resource_management.reset.manager.CloudControlManager"
                ) as mock_ccm:
                    mock_ccm.return_value.resource_exists.return_value = False
                    result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    assert result.success
    # The role was deleted exactly once (deduped across 3 regions), in one pass.
    assert cleanup_calls == [["SharedGlobalRole"]]


def test_reset_account_raises_when_global_delete_fails(temp_output_dir):
    """A global resource that fails to delete in the final pass fails the reset.

    No per-region final verify guards globals (they are deferred out), so the
    account-level pass must surface a leftover global itself.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    snapshot = Snapshot(
        timestamp=datetime(2026, 4, 22, 10, 30, 0, tzinfo=timezone.utc),
        account_id="123456789012",
        environment_id="env-test",
        scenario_hash="abc123",
        drift_baseline={},
        stack_metadata={},
        resource_ids={},
        regions=["us-east-1"],
    )
    global_role = {"AWS::IAM::Role": [{"Identifier": "StuckRole"}]}

    async def _cleanup(stack_resources, *args, **kwargs):
        # Report every resource as failed to delete.
        return {
            Resource(type=r.resource_type, identifier=r.physical_id): MagicMock()
            for r in stack_resources
        }

    with patch(
        "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot",
        return_value=snapshot,
    ):
        with patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify:
            mock_verify.side_effect = [
                VerifyResult(success=False, reason="new", new_resources=global_role),
                VerifyResult(success=True, reason="clean"),
            ]
            with patch(
                "aws_bench.resource_management.reset.manager.ResourceCleaner"
            ) as mock_cleaner_cls:
                mock_cleaner_cls.return_value.cleanup = _cleanup
                # Authoritative re-check finds the role still present.
                with patch(
                    "aws_bench.resource_management.reset.manager.CloudControlManager"
                ) as mock_ccm:
                    mock_ccm.return_value.resource_exists.return_value = True
                    result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    # The stuck global surfaces as a reset failure (not a false success).
    assert result.success is False
    assert "still present" in (result.reason or "")


def test_reset_account_global_recheck_catches_silent_skip(temp_output_dir):
    """A global the cleanup pipeline SILENTLY SKIPPED (0 failures) still fails reset.

    Guards the F1 collateral-harm case: CCAPI can skip a resource whose existence
    pre-check throttles/errors, reporting zero failures while the resource lives.
    Relying on the pipeline's failure count alone would false-pass with a leftover
    IAM role -> contamination. The authoritative existence re-check must catch it.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    snapshot = Snapshot(
        timestamp=datetime(2026, 4, 22, 10, 30, 0, tzinfo=timezone.utc),
        account_id="123456789012",
        environment_id="env-test",
        scenario_hash="abc123",
        drift_baseline={},
        stack_metadata={},
        resource_ids={},
        regions=["us-east-1"],
    )
    global_role = {"AWS::IAM::Role": [{"Identifier": "SilentlySkippedRole"}]}

    # Pipeline reports ZERO failures (the silent-skip case) ...
    async def _cleanup(stack_resources, *args, **kwargs):
        return {}

    with patch(
        "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot",
        return_value=snapshot,
    ):
        with patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify:
            mock_verify.side_effect = [
                VerifyResult(success=False, reason="new", new_resources=global_role),
                VerifyResult(success=True, reason="clean"),
            ]
            with patch(
                "aws_bench.resource_management.reset.manager.ResourceCleaner"
            ) as mock_cleaner_cls:
                mock_cleaner_cls.return_value.cleanup = _cleanup
                # ... but the role is in fact still present.
                with patch(
                    "aws_bench.resource_management.reset.manager.CloudControlManager"
                ) as mock_ccm:
                    mock_ccm.return_value.resource_exists.return_value = True
                    result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    assert result.success is False
    assert "still present" in (result.reason or "")


def test_reset_account_global_recheck_fails_closed_on_throttle(temp_output_dir):
    """If existence cannot be confirmed (throttle/error), reset fails closed.

    A false "gone" would leak the global into the next task, so an un-confirmable
    check counts the resource as surviving.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    snapshot = Snapshot(
        timestamp=datetime(2026, 4, 22, 10, 30, 0, tzinfo=timezone.utc),
        account_id="123456789012",
        environment_id="env-test",
        scenario_hash="abc123",
        drift_baseline={},
        stack_metadata={},
        resource_ids={},
        regions=["us-east-1"],
    )
    global_role = {"AWS::IAM::Role": [{"Identifier": "ThrottledRole"}]}

    async def _cleanup(stack_resources, *args, **kwargs):
        return {}

    with patch(
        "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot",
        return_value=snapshot,
    ):
        with patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state"
        ) as mock_verify:
            mock_verify.side_effect = [
                VerifyResult(success=False, reason="new", new_resources=global_role),
                VerifyResult(success=True, reason="clean"),
            ]
            with patch(
                "aws_bench.resource_management.reset.manager.ResourceCleaner"
            ) as mock_cleaner_cls:
                mock_cleaner_cls.return_value.cleanup = _cleanup
                with patch(
                    "aws_bench.resource_management.reset.manager.CloudControlManager"
                ) as mock_ccm:
                    mock_ccm.return_value.resource_exists.side_effect = RuntimeError("throttled")
                    result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    assert result.success is False
    assert "still present" in (result.reason or "")
