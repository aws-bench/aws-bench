"""Comprehensive integration tests for CleanupManager."""

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from botocore.exceptions import ClientError

from aws_bench.resource_management.cleanup.manager import CleanupManager
from aws_bench.resource_management.cleanup.models import (
    CleanupSummary,
    DeletionSummary,
    RegionResult,
    SnapshotResources,
    StackDeletionResult,
    StackDeletionStatus,
    StackResource,
)
from aws_bench.resource_management.exceptions import SnapshotNotFoundError
from aws_bench.resource_management.snapshot.models import SnapshotStage


@pytest.fixture
def manager(tmp_path):
    """Create a CleanupManager with mocked session, output dir under tmp_path."""
    session = Mock()
    sts_client = Mock()
    sts_client.get_caller_identity.return_value = {"Account": "123456789012"}

    def mock_client(service, **kwargs):
        if service == "sts":
            return sts_client
        return Mock()

    session.client.side_effect = mock_client
    return CleanupManager(session, output_dir=tmp_path / "cleanup_output")


@pytest.fixture
def mock_run_dir(tmp_path):
    """Create a temporary run directory."""
    run_dir = tmp_path / "cleanup_output" / "test_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# -- cleanup_all_stacks --


def test_cleanup_all_stacks_with_regions(manager, mock_run_dir):
    """Test cleanup_all_stacks with provided regions."""
    manager._output_dir = mock_run_dir
    with (
        patch.object(manager, "_write_metadata"),
        patch.object(manager, "_cleanup_single_region", new_callable=AsyncMock) as mock_cleanup,
        patch.object(manager, "_scan_orphaned_resources", new_callable=AsyncMock, return_value={}),
        patch.object(manager, "_log_summary"),
        patch.object(manager, "_save_summary"),
    ):
        mock_cleanup.return_value = RegionResult(region="us-east-1")
        regions = ["us-east-1", "us-west-2"]

        summary = asyncio.run(manager.cleanup_all_stacks(regions=regions))

        assert len(summary.regions) == 2
        assert summary.run_dir == str(mock_run_dir)
        assert mock_cleanup.call_count == 2


def test_cleanup_all_stacks_discovers_regions(manager):
    """Test cleanup_all_stacks discovers regions when None provided."""
    with (
        patch.object(manager, "_discover_regions", return_value=["us-east-1", "eu-west-1"]),
        patch.object(manager, "_write_metadata"),
        patch.object(manager, "_cleanup_single_region", new_callable=AsyncMock) as mock_cleanup,
        patch.object(manager, "_scan_orphaned_resources", new_callable=AsyncMock, return_value={}),
        patch.object(manager, "_log_summary"),
        patch.object(manager, "_save_summary"),
    ):
        mock_cleanup.return_value = RegionResult(region="us-east-1")

        summary = asyncio.run(manager.cleanup_all_stacks())

        assert len(summary.regions) == 2


def test_cleanup_all_stacks_with_orphans(manager):
    """Test cleanup_all_stacks captures orphaned resources."""
    from aws_bench.resource_management.cleanup.models import AccountScanResult

    orphan_map = {"AWS::S3::Bucket": ["bucket-1"], "AWS::EC2::Instance": ["i-123"]}
    region_counts = {"us-east-1": 2}
    orphan_scan_result = AccountScanResult(
        orphaned_resources=orphan_map, region_counts=region_counts
    )

    with (
        patch.object(manager, "_discover_regions", return_value=["us-east-1"]),
        patch.object(manager, "_write_metadata"),
        patch.object(
            manager,
            "_cleanup_single_region",
            new_callable=AsyncMock,
            return_value=RegionResult(region="us-east-1", stacks_found=2, stacks_deleted=2),
        ),
        patch.object(
            manager,
            "_scan_orphaned_resources",
            new_callable=AsyncMock,
            return_value=orphan_scan_result,
        ),
        patch.object(manager, "_log_summary"),
        patch.object(manager, "_save_summary"),
    ):
        summary = asyncio.run(manager.cleanup_all_stacks())

        assert summary.orphaned_resources == orphan_map
        assert summary.total_orphaned == 2


def test_cleanup_all_stacks_records_region_exception(manager):
    """A region's cleanup exception is recorded on that region, not raised."""
    with (
        patch.object(manager, "_discover_regions", return_value=["us-east-1"]),
        patch.object(manager, "_write_metadata"),
        patch.object(
            manager,
            "_cleanup_single_region",
            new_callable=AsyncMock,
            side_effect=Exception("Test error"),
        ),
        patch.object(manager, "_scan_orphaned_resources", new_callable=AsyncMock, return_value={}),
        patch.object(manager, "_log_summary"),
        patch.object(manager, "_save_summary"),
    ):
        # Should not raise - exceptions are handled gracefully
        summary = asyncio.run(manager.cleanup_all_stacks())

        # Region should have error recorded
        assert len(summary.regions) == 1
        assert summary.regions[0].region == "us-east-1"
        assert summary.regions[0].error == "Test error"


def test_cleanup_all_stacks_all_regions_ignores_baseline(mock_run_dir):
    """all_regions=True cleans every enabled region and bypasses baseline limiting."""
    from aws_bench.resource_management.cleanup.models import AccountScanResult

    session = Mock()
    sts_client = Mock()
    sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
    session.client.return_value = sts_client
    # env_name set -> a baseline snapshot manager exists, but all_regions must ignore it.
    manager = CleanupManager(session, output_dir=mock_run_dir, env_name="test-env")

    with (
        patch.object(
            manager._snapshot_mgr,
            "load_snapshot",
            side_effect=SnapshotNotFoundError("test-env", "123456789012", SnapshotStage.POST_SETUP),
        ),
        patch.object(manager, "_limit_to_baseline_regions", new_callable=AsyncMock) as mock_limit,
        patch.object(
            manager, "_discover_regions", return_value=["us-east-1", "eu-west-1", "ap-south-1"]
        ),
        patch.object(manager, "_write_metadata"),
        patch.object(
            manager,
            "_cleanup_region_in_phases",
            new_callable=AsyncMock,
            return_value=RegionResult(region="us-east-1"),
        ) as mock_cleanup,
        patch.object(
            manager,
            "_scan_orphaned_resources",
            new_callable=AsyncMock,
            return_value=AccountScanResult(orphaned_resources={}, region_counts={}),
        ),
        patch.object(manager, "_log_summary"),
        patch.object(manager, "_save_summary"),
    ):
        summary = asyncio.run(manager.cleanup_all_stacks(all_regions=True))

        # Baseline limiting must be skipped entirely.
        mock_limit.assert_not_called()
        assert mock_cleanup.call_count == 3
        assert len(summary.regions) == 3


def test_cleanup_all_stacks_all_regions_with_explicit_regions(manager):
    """all_regions=True with explicit regions uses them directly (no discovery, no baseline)."""
    from aws_bench.resource_management.cleanup.models import AccountScanResult

    with (
        patch.object(manager, "_limit_to_baseline_regions", new_callable=AsyncMock) as mock_limit,
        patch.object(manager, "_discover_regions") as mock_discover,
        patch.object(manager, "_write_metadata"),
        patch.object(
            manager,
            "_cleanup_single_region",
            new_callable=AsyncMock,
            return_value=RegionResult(region="us-east-1"),
        ) as mock_cleanup,
        patch.object(
            manager,
            "_scan_orphaned_resources",
            new_callable=AsyncMock,
            return_value=AccountScanResult(orphaned_resources={}, region_counts={}),
        ),
        patch.object(manager, "_log_summary"),
        patch.object(manager, "_save_summary"),
    ):
        summary = asyncio.run(manager.cleanup_all_stacks(regions=["us-east-1"], all_regions=True))

        mock_limit.assert_not_called()
        mock_discover.assert_not_called()
        assert mock_cleanup.call_count == 1
        assert len(summary.regions) == 1


def test_cleanup_all_stacks_baseline_path_uses_baseline_regions(mock_run_dir):
    """Regression: default (all_regions=False) cleanup uses baseline regions, not an empty list.

    Reproduces the failure where the orchestrator's baseline path produced an
    empty regions list and raised 'regions list cannot be empty'. With
    regions=None and a baseline snapshot, cleanup must run across the baseline
    regions instead.
    """
    from datetime import datetime, timezone

    from aws_bench.resource_management.cleanup.models import AccountScanResult
    from aws_bench.resource_management.snapshot.models import Snapshot

    session = Mock()
    sts_client = Mock()
    sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
    session.client.return_value = sts_client
    manager = CleanupManager(session, output_dir=mock_run_dir, env_name="test-env")

    baseline = Snapshot(
        timestamp=datetime.now(timezone.utc),
        account_id="123456789012",
        environment_id="test-env",
        scenario_hash="hash_v1",
        drift_baseline={},
        stack_metadata={},
        resource_ids={},
        regions=["ap-northeast-1", "us-east-1", "us-west-2"],
    )

    with (
        patch.object(manager._snapshot_mgr, "load_snapshot", return_value=baseline),
        patch.object(manager, "_write_metadata"),
        patch.object(
            manager,
            "_cleanup_single_region",
            new_callable=AsyncMock,
            return_value=RegionResult(region="us-east-1"),
        ) as mock_cleanup,
        patch.object(
            manager,
            "_scan_orphaned_resources",
            new_callable=AsyncMock,
            return_value=AccountScanResult(orphaned_resources={}, region_counts={}),
        ),
        patch.object(manager, "_log_summary"),
        patch.object(manager, "_save_summary"),
    ):
        # No regions, default all_regions=False -> must resolve to the 3 baseline regions.
        summary = asyncio.run(manager.cleanup_all_stacks())

        assert mock_cleanup.call_count == 3
        cleaned = {call.args[0] for call in mock_cleanup.call_args_list}
        assert cleaned == {"ap-northeast-1", "us-east-1", "us-west-2"}
        assert len(summary.regions) == 3


# -- cleanup_stack --


def test_cleanup_stack_success(manager):
    """Test cleanup_stack successfully deletes a stack."""
    mock_deleter = MagicMock()
    mock_result = StackDeletionResult(stack_name="my-stack", status=StackDeletionStatus.SUCCESS)
    mock_deleter.delete_stack = AsyncMock(return_value=mock_result)

    with (
        patch.object(
            manager, "_find_stack_region", new_callable=AsyncMock, return_value="us-west-2"
        ),
        patch.object(manager, "_write_metadata"),
        patch(
            "aws_bench.resource_management.cleanup.manager.StackDeleter",
            return_value=mock_deleter,
        ),
        patch.object(manager, "_log_summary"),
        patch.object(manager, "_save_summary"),
    ):
        summary = asyncio.run(manager.cleanup_stack("my-stack"))

        assert len(summary.regions) == 1
        assert summary.regions[0].region == "us-west-2"
        assert summary.regions[0].stacks_deleted == 1
        assert summary.regions[0].stacks_failed_count == 0
        # A clean delete abandons nothing.
        assert summary.regions[0].orphan_count == 0
        assert summary.orphaned_resources == {}


def test_cleanup_stack_surfaces_abandoned_as_orphans(manager):
    """A FORCE_DELETE-abandoned resource is surfaced as an orphan on the single-stack path.

    The single-stack path runs no orphan scan, so the abandoned resources riding on
    StackDeletionResult are the only record — they must land in orphaned_resources with
    orphan_count == len(abandoned) so reset can fail closed.
    """
    mock_deleter = MagicMock()
    mock_result = StackDeletionResult(
        stack_name="my-stack",
        status=StackDeletionStatus.SUCCESS,
        abandoned_resources=[
            StackResource(
                logical_id="Igw",
                physical_id="igw-abandoned",
                resource_type="AWS::EC2::InternetGateway",
                status="DELETE_FAILED",
            ),
            # No physical id — falls back to the logical id.
            StackResource(
                logical_id="Bucket",
                physical_id="",
                resource_type="AWS::S3::Bucket",
                status="DELETE_FAILED",
            ),
        ],
    )
    mock_deleter.delete_stack = AsyncMock(return_value=mock_result)

    with (
        patch.object(
            manager, "_find_stack_region", new_callable=AsyncMock, return_value="us-west-2"
        ),
        patch.object(manager, "_write_metadata"),
        patch(
            "aws_bench.resource_management.cleanup.manager.StackDeleter",
            return_value=mock_deleter,
        ),
        patch.object(manager, "_log_summary"),
        patch.object(manager, "_save_summary"),
    ):
        summary = asyncio.run(manager.cleanup_stack("my-stack"))

        assert summary.regions[0].stacks_deleted == 1
        assert summary.regions[0].orphan_count == 2
        assert summary.orphaned_resources == {
            "AWS::EC2::InternetGateway": ["igw-abandoned"],
            "AWS::S3::Bucket": ["Bucket"],
        }


def test_cleanup_stack_not_found(manager):
    """Test cleanup_stack raises ValueError when stack not found."""
    with patch.object(manager, "_find_stack_region", new_callable=AsyncMock, return_value=None):
        with pytest.raises(ValueError, match="Stack 'nonexistent' not found"):
            asyncio.run(manager.cleanup_stack("nonexistent"))


def test_cleanup_stack_failed(manager):
    """Test cleanup_stack handles stack deletion failure."""
    mock_deleter = MagicMock()
    mock_result = StackDeletionResult(
        stack_name="my-stack", status=StackDeletionStatus.FAILED, reason="Timeout"
    )
    mock_deleter.delete_stack = AsyncMock(return_value=mock_result)

    with (
        patch.object(
            manager, "_find_stack_region", new_callable=AsyncMock, return_value="us-east-1"
        ),
        patch.object(manager, "_write_metadata"),
        patch(
            "aws_bench.resource_management.cleanup.manager.StackDeleter",
            return_value=mock_deleter,
        ),
        patch.object(manager, "_log_summary"),
        patch.object(manager, "_save_summary"),
    ):
        summary = asyncio.run(manager.cleanup_stack("my-stack"))

        assert summary.regions[0].stacks_failed_count == 1
        assert summary.regions[0].stacks_failed == ["my-stack"]


# -- Helper methods --


def test_write_metadata_success(manager, tmp_path):
    """Test _write_metadata writes JSON file."""
    run_dir = tmp_path / "output"
    run_dir.mkdir()

    manager._write_metadata(run_dir, {"regions": ["us-east-1"]})

    metadata_file = run_dir / "run_metadata.json"
    assert metadata_file.exists()
    data = json.loads(metadata_file.read_text())
    assert data["account_id"] == "123456789012"
    assert data["regions"] == ["us-east-1"]
    assert "timestamp" in data


def test_write_metadata_handles_os_error(manager, caplog):
    """Test _write_metadata handles OSError gracefully."""
    run_dir = Path("/nonexistent/directory")
    manager._write_metadata(run_dir, {})  # Should not raise


def test_find_stack_region_found(manager):
    """Test _find_stack_region finds stack in second region."""
    with patch.object(manager, "_discover_regions", return_value=["us-east-1", "us-west-2"]):

        def mock_describe(StackName):
            if StackName == "my-stack":
                return {"Stacks": [{"StackName": "my-stack"}]}
            raise ClientError({"Error": {"Code": "ValidationError"}}, "DescribeStacks")

        mock_cfn_us_east = Mock()
        mock_cfn_us_east.describe_stacks.side_effect = ClientError(
            {"Error": {"Code": "ValidationError"}}, "DescribeStacks"
        )

        mock_cfn_us_west = Mock()
        mock_cfn_us_west.describe_stacks.return_value = {"Stacks": [{"StackName": "my-stack"}]}

        def mock_client(service, region_name):
            if region_name == "us-east-1":
                return mock_cfn_us_east
            return mock_cfn_us_west

        manager._session.client.side_effect = mock_client

        region = asyncio.run(manager._find_stack_region("my-stack"))

        assert region == "us-west-2"


def test_find_stack_region_not_found(manager):
    """Test _find_stack_region returns None when stack not found."""
    with patch.object(manager, "_discover_regions", return_value=["us-east-1"]):
        mock_cfn = Mock()
        mock_cfn.describe_stacks.side_effect = ClientError(
            {"Error": {"Code": "StackNotFound"}}, "DescribeStacks"
        )
        # Reset side_effect from fixture and set return_value
        manager._session.client.side_effect = None
        manager._session.client.return_value = mock_cfn

        region = asyncio.run(manager._find_stack_region("nonexistent"))

        assert region is None


def test_cleanup_single_region_success(manager, tmp_path):
    """Test _cleanup_single_region successfully deletes stacks."""
    mock_deleter = Mock()
    mock_deleter.list_stacks.return_value = [{"StackName": "stack1"}, {"StackName": "stack2"}]
    mock_deleter.delete_all_stacks = AsyncMock(
        return_value=DeletionSummary(
            results=[
                StackDeletionResult(stack_name="stack1", status=StackDeletionStatus.SUCCESS),
                StackDeletionResult(
                    stack_name="stack2", status=StackDeletionStatus.FAILED, reason="Error"
                ),
            ]
        )
    )

    with patch(
        "aws_bench.resource_management.cleanup.manager.StackDeleter", return_value=mock_deleter
    ):
        result = asyncio.run(manager._cleanup_single_region("us-east-1", tmp_path))

    assert result.region == "us-east-1"
    assert result.stacks_found == 2
    assert result.stacks_deleted == 1
    assert result.stacks_failed_count == 1
    assert result.stacks_failed == ["stack2"]


def test_cleanup_single_region_no_stacks(manager, tmp_path):
    """Test _cleanup_single_region with no stacks."""
    mock_deleter = Mock()
    mock_deleter.list_stacks.return_value = []

    with patch(
        "aws_bench.resource_management.cleanup.manager.StackDeleter", return_value=mock_deleter
    ):
        result = asyncio.run(manager._cleanup_single_region("us-east-1", tmp_path))

    assert result.stacks_found == 0
    assert result.stacks_deleted == 0


def test_cleanup_single_region_initialization_error(manager, tmp_path):
    """Test _cleanup_single_region handles StackDeleter initialization error."""
    with patch(
        "aws_bench.resource_management.cleanup.manager.StackDeleter",
        side_effect=Exception("Init error"),
    ):
        result = asyncio.run(manager._cleanup_single_region("us-east-1", tmp_path))

    assert result.error == "Init error"
    assert result.stacks_found == 0


def test_scan_orphaned_resources(manager, tmp_path):
    """_scan_orphaned_resources returns the scanner's result unchanged."""
    from aws_bench.resource_management.cleanup.models import AccountScanResult

    # The scanner already projects detected resources down to identifiers, so
    # the manager returns its result as-is.
    scan_result = AccountScanResult(
        orphaned_resources={
            "AWS::S3::Bucket": ["bucket-1"],
            "AWS::EC2::Instance": ["i-123", "i-456"],
        },
        region_counts={"us-east-1": 3},
    )
    mock_scanner = Mock()
    mock_scanner.run.return_value = scan_result

    with patch(
        "aws_bench.resource_management.cleanup.manager.AccountScanner", return_value=mock_scanner
    ):
        result = asyncio.run(manager._scan_orphaned_resources(tmp_path, ["us-east-1"]))

    assert result.orphaned_resources == {
        "AWS::S3::Bucket": ["bucket-1"],
        "AWS::EC2::Instance": ["i-123", "i-456"],
    }
    assert result.region_counts == {"us-east-1": 3}


def test_scan_orphaned_resources_passes_account_id_to_scanner(manager, tmp_path):
    """The orphan scan builds its AccountScanner with the account id (Lambda target).

    Without it AccountScanner's per-region scan degrades to the throttled host
    path; a failed lister then hides orphans behind scan_result.failed.
    """
    from aws_bench.resource_management.cleanup.models import AccountScanResult

    mock_scanner = Mock()
    mock_scanner.run.return_value = AccountScanResult(orphaned_resources={}, region_counts={})

    with patch(
        "aws_bench.resource_management.cleanup.manager.AccountScanner", return_value=mock_scanner
    ) as mock_scanner_cls:
        asyncio.run(manager._scan_orphaned_resources(tmp_path, ["us-east-1"]))

    assert mock_scanner_cls.call_args.kwargs.get("account_id") == "123456789012"


def test_scan_region_resources_passes_account_id_to_scanner(manager):
    """The phase-scan helper builds its AccountScanner with the account id.

    Same invariant as the orphan scan: the destructive cleanup phases diff against
    this scan, so it must run on the Lambda where enumeration failures surface
    rather than the host path where they are silently skipped.
    """
    from aws_bench.resource_management.ccapi.models import ScanResult

    mock_scanner = Mock()
    mock_scanner.scan_region.return_value = ScanResult(detected={}, failed={})

    with patch(
        "aws_bench.resource_management.cleanup.manager.AccountScanner", return_value=mock_scanner
    ) as mock_scanner_cls:
        manager._scan_region_resources("us-east-1")

    assert mock_scanner_cls.call_args.kwargs.get("account_id") == "123456789012"


def test_scan_region_resources_forwards_include_infra(manager):
    """The phase-scan helper forwards include_infra to AccountScanner.scan_region."""
    from aws_bench.resource_management.ccapi.models import ScanResult

    mock_scanner = Mock()
    mock_scanner.scan_region.return_value = ScanResult(detected={}, failed={})

    with patch(
        "aws_bench.resource_management.cleanup.manager.AccountScanner", return_value=mock_scanner
    ):
        manager._scan_region_resources("us-east-1", include_infra=True)

    assert mock_scanner.scan_region.call_args.kwargs.get("include_infra") is True


def test_phase3_scans_with_include_infra_when_region_clean(manager):
    """When the region deleted every stack, phase 3 runs with include_infra=True.

    CDK bootstrap/toolkit *regional* assets (the assets bucket / bootstrap param)
    are absent from the init snapshot, so the phase-3 ``current - init`` diff
    reclaims the CDKToolkit stack's retained leftovers once the region is clear.
    """
    init = SnapshotResources(resource_ids={}, failed_types={})

    with (
        patch.object(
            manager,
            "_cleanup_single_region",
            new_callable=AsyncMock,
            return_value=RegionResult(region="us-east-1", stacks_found=2, stacks_deleted=2),
        ),
        patch.object(manager, "_delete_resources_created_after_setup", new_callable=AsyncMock),
        patch.object(
            manager, "_scan_region_resources", return_value=Mock(detected={}, failed={})
        ) as mock_scan,
        patch.object(manager, "_load_snapshot_resource_ids", return_value=init),
    ):
        asyncio.run(manager._cleanup_region_in_phases("us-east-1", Path("/tmp")))

    assert mock_scan.call_args.kwargs.get("include_infra") is True


def test_phase3_runs_even_when_region_has_a_failed_stack(manager):
    """Phase 3 always runs as last-resort cleanup regardless of phase-2 outcome.

    Orphaned resources (agent-created, not stack-managed) must be reaped even when
    a stack is stuck in DELETE_FAILED. The sweep itself is best-effort (errors
    absorbed by ``_run_sweep_phase``).
    """
    init = SnapshotResources(resource_ids={}, failed_types={})

    with (
        patch.object(
            manager,
            "_cleanup_single_region",
            new_callable=AsyncMock,
            return_value=RegionResult(
                region="us-east-1", stacks_found=2, stacks_deleted=1, stacks_failed=["stuck"]
            ),
        ),
        patch.object(manager, "_delete_resources_created_after_setup", new_callable=AsyncMock),
        patch.object(
            manager, "_scan_region_resources", return_value=Mock(detected={}, failed={})
        ) as mock_scan,
        patch.object(manager, "_load_snapshot_resource_ids", return_value=init),
    ):
        asyncio.run(manager._cleanup_region_in_phases("us-east-1", Path("/tmp")))

    # Phase 3 scan fires even though phase 2 had a failed stack
    mock_scan.assert_called()


def test_log_summary_with_successes_and_failures(manager, caplog):
    """Test _log_summary logs comprehensive summary.

    Per-region rows ("N/M deleted", "no stacks") are DEBUG (technical detail);
    only the aggregate totals and per-region failures surface at INFO/ERROR. Capture
    at DEBUG so the whole summary is visible to the assertions.
    """
    caplog.set_level(logging.DEBUG)

    summary = CleanupSummary(
        regions=[
            RegionResult(
                region="us-east-1",
                stacks_found=3,
                stacks_deleted=2,
                stacks_failed=["stack-fail"],
            ),
            RegionResult(region="us-west-2", stacks_found=0),
            RegionResult(region="eu-west-1", error="Connection timeout"),
        ],
        orphaned_resources={"AWS::S3::Bucket": ["bucket-1"]},
    )

    manager._log_summary(summary)

    log_text = caplog.text
    assert "2/3 deleted" in log_text
    assert "stack-fail" in log_text or "FAILED" in log_text
    assert "no stacks" in log_text
    assert "Connection timeout" in log_text


def test_save_summary(manager, tmp_path):
    """Test _save_summary writes summary JSON."""
    summary = CleanupSummary(
        regions=[RegionResult(region="us-east-1", stacks_found=5, stacks_deleted=5)]
    )

    manager._save_summary(summary, tmp_path)

    summary_file = tmp_path / "summary.json"
    assert summary_file.exists()
    data = json.loads(summary_file.read_text())
    assert len(data["regions"]) == 1
    assert data["regions"][0]["region"] == "us-east-1"


def test_save_summary_handles_os_error(manager, caplog):
    """Test _save_summary handles write errors gracefully."""
    summary = CleanupSummary()
    run_dir = Path("/nonexistent/path")
    manager._save_summary(summary, run_dir)  # Should not raise


def test_cleanup_all_stacks_validates_regions_not_empty(manager):
    """Test cleanup_all_stacks rejects empty regions list."""
    with pytest.raises(ValueError, match="cannot be empty"):
        asyncio.run(manager.cleanup_all_stacks(regions=[]))


def test_cleanup_all_stacks_validates_regions_elements(manager):
    """Test cleanup_all_stacks validates all elements are strings."""
    with pytest.raises(ValueError, match="All regions must be strings"):
        asyncio.run(manager.cleanup_all_stacks(regions=["us-east-1", 123, "us-west-2"]))  # type: ignore
