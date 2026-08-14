"""Tests for VerifyManager."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

from aws_bench.resource_management.exceptions import SnapshotNotFoundError
from aws_bench.resource_management.snapshot.models import (
    DriftBaseline,
    Snapshot,
    StackMetadata,
)
from aws_bench.resource_management.verify.manager import VerifyManager

# ===========================================================================
# VerifyManager initialization
# ===========================================================================


@mock_aws
def test_verify_manager_initialization():
    """Test VerifyManager initialization."""
    session = boto3.Session(region_name="us-east-1")
    manager = VerifyManager(session)

    assert manager._session == session
    assert manager._scan_mgr is not None
    assert manager._stack_inspector is not None
    assert manager._drift_detector is not None


@mock_aws
def test_verify_manager_initialization_with_base_dir():
    """Test VerifyManager initialization."""
    session = boto3.Session(region_name="us-east-1")
    manager = VerifyManager(session)

    assert manager._snapshot_mgr is not None


@mock_aws
def test_verify_manager_initialization_with_region():
    """Test VerifyManager initialization with custom region."""
    session = boto3.Session(region_name="us-east-1")
    manager = VerifyManager(session, region_name="us-west-2")

    assert manager._region_name == "us-west-2"


@mock_aws
@patch("aws_bench.resource_management.verify.manager.make_scanner")
def test_verify_manager_threads_account_id_into_make_scanner(mock_make_scanner):
    """VerifyManager routes its scan to the management-account Lambda.

    make_scanner only targets the Lambda when given an account_id; without it the
    scan silently degrades to the throttled host path, where listers fail into
    scan_result.failed and find_new_resources skips them — so verify would falsely
    pass. The account id is what makes those failures surface.
    """
    session = boto3.Session(region_name="us-east-1")

    VerifyManager(session, region_name="us-east-1", account_id="123456789012")

    assert mock_make_scanner.call_args.kwargs.get("account_id") == "123456789012"


@mock_aws
@patch("aws_bench.resource_management.verify.manager.make_scanner")
def test_verify_manager_account_id_defaults_to_none(mock_make_scanner):
    """Omitting account_id threads None as a keyword (no positional breakage)."""
    session = boto3.Session(region_name="us-east-1")

    VerifyManager(session, region_name="us-east-1")

    assert "account_id" in mock_make_scanner.call_args.kwargs
    assert mock_make_scanner.call_args.kwargs["account_id"] is None


# ===========================================================================
# StackInspector._compute_template_hash
# ===========================================================================


@mock_aws
def test_stack_inspector_compute_template_hash_json():
    """Test computing hash of JSON template."""
    session = boto3.Session(region_name="us-east-1")
    manager = VerifyManager(session)
    inspector = manager._stack_inspector

    template = {"AWSTemplateFormatVersion": "2010-09-09", "Resources": {}}

    hash_result = inspector._compute_template_hash(template)

    # Verify it's a sha256 hash
    assert hash_result.startswith("sha256:")
    assert len(hash_result) == 71  # "sha256:" (7) + 64 hex chars

    # Verify consistency
    hash_result2 = inspector._compute_template_hash(template)
    assert hash_result == hash_result2


@mock_aws
def test_stack_inspector_compute_template_hash_string():
    """Test computing hash of string template."""
    session = boto3.Session(region_name="us-east-1")
    manager = VerifyManager(session)
    inspector = manager._stack_inspector

    template = "AWSTemplateFormatVersion: 2010-09-09\nResources: {}"

    hash_result = inspector._compute_template_hash(template)

    assert hash_result.startswith("sha256:")
    assert len(hash_result) == 71


# ===========================================================================
# StackInspector._list_cloudformation_stacks
# ===========================================================================


@mock_aws
def test_stack_inspector_list_cloudformation_stacks_empty():
    """Test listing stacks when none exist."""
    session = boto3.Session(region_name="us-east-1")
    manager = VerifyManager(session)
    inspector = manager._stack_inspector

    stacks = inspector._list_cloudformation_stacks()

    assert stacks == []


@mock_aws
def test_stack_inspector_list_cloudformation_stacks_filters_deleted():
    """Test that deleted stacks are filtered out."""
    session = boto3.Session(region_name="us-east-1")
    cfn = session.client("cloudformation")

    # Create stack
    cfn.create_stack(
        StackName="test-stack",
        TemplateBody=json.dumps({"AWSTemplateFormatVersion": "2010-09-09", "Resources": {}}),
    )

    # Delete stack
    cfn.delete_stack(StackName="test-stack")

    manager = VerifyManager(session)
    inspector = manager._stack_inspector
    stacks = inspector._list_cloudformation_stacks()

    # Should not include deleted stacks
    assert all(s["StackStatus"] != "DELETE_COMPLETE" for s in stacks)


@mock_aws
def test_stack_inspector_list_cloudformation_stacks_filters_nested():
    """Test that nested stacks are filtered out."""
    session = boto3.Session(region_name="us-east-1")
    cfn = session.client("cloudformation")

    # Create parent stack
    cfn.create_stack(
        StackName="parent-stack",
        TemplateBody=json.dumps({"AWSTemplateFormatVersion": "2010-09-09", "Resources": {}}),
    )

    manager = VerifyManager(session)
    inspector = manager._stack_inspector
    stacks = inspector._list_cloudformation_stacks()

    # Should only include root stacks (no ParentId)
    assert all("ParentId" not in s for s in stacks)


# ===========================================================================
# verify_account_state - full workflow tests
# ===========================================================================


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
def test_verify_account_state_snapshot_not_found(mock_snapshot_mgr_from_sessions):
    """Test verify when snapshot doesn't exist."""
    session = boto3.Session(region_name="us-east-1")

    # Mock snapshot manager to raise SnapshotNotFoundError
    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.side_effect = SnapshotNotFoundError("test-env", "123456789012")
    mock_snapshot_mgr_from_sessions.return_value = mock_snapshot_mgr

    manager = VerifyManager(session)

    # Should raise SnapshotNotFoundError
    try:
        manager.verify_account_state("test-env", "123456789012")
        assert False, "Should have raised SnapshotNotFoundError"
    except SnapshotNotFoundError:
        pass


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
def test_verify_account_state_region_mismatch(mock_snapshot_mgr_from_sessions):
    """Test verify when region doesn't match snapshot regions."""
    session = boto3.Session(region_name="us-east-1")

    # Mock snapshot with regions
    mock_snapshot = Snapshot(
        timestamp=datetime.now(timezone.utc),
        account_id="123456789012",
        environment_id="test-env",
        scenario_hash="v1.0",
        drift_baseline={},
        stack_metadata={},
        resource_ids={},
        regions=["us-west-2"],
    )
    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = mock_snapshot
    mock_snapshot_mgr_from_sessions.return_value = mock_snapshot_mgr

    manager = VerifyManager(session, region_name="us-east-1")

    result = manager.verify_account_state("test-env", "123456789012")

    assert result.success is False
    assert "Region mismatch" in result.reason
    assert result.details is not None
    assert isinstance(result.details, dict)
    assert result.details["verify_region"] == "us-east-1"
    assert result.details["snapshot_regions"] == ["us-west-2"]


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
def test_verify_account_state_region_match(mock_snapshot_mgr_from_sessions):
    """Test verify passes when region matches snapshot regions."""
    session = boto3.Session(region_name="us-east-1")

    # Mock snapshot with matching region
    mock_snapshot = Snapshot(
        timestamp=datetime.now(timezone.utc),
        account_id="123456789012",
        environment_id="test-env",
        scenario_hash="v1.0",
        drift_baseline={},
        stack_metadata={},
        resource_ids={},
        regions=["us-east-1"],
    )
    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = mock_snapshot
    mock_snapshot_mgr_from_sessions.return_value = mock_snapshot_mgr

    # Mock scanner to pass
    with patch("aws_bench.resource_management.verify.manager.make_scanner") as mock_scanner_class:
        from aws_bench.resource_management.ccapi.models import ScanResult

        mock_scanner = MagicMock()
        mock_scanner.scan_resources.return_value = ScanResult(detected={}, failed={})
        mock_scanner_class.return_value = mock_scanner

        # Mock inspector to pass
        with patch(
            "aws_bench.resource_management.verify.manager.StackInspector"
        ) as mock_inspector_class:
            from aws_bench.resource_management.verify.models import (
                StackStatusCheckResult,
                TemplateHashCheckResult,
            )

            mock_inspector = MagicMock()
            mock_inspector.check_stack_status.return_value = StackStatusCheckResult(
                success=True, error_reason="", error_details=None
            )
            mock_inspector.check_template_hash.return_value = TemplateHashCheckResult(
                success=True, error_reason="", error_details=None
            )
            mock_inspector_class.return_value = mock_inspector

            # Mock drift detector to pass
            with patch(
                "aws_bench.resource_management.verify.manager.DriftDetector"
            ) as mock_detector_class:
                from aws_bench.resource_management.verify.models import DriftDetectionResult

                mock_detector = MagicMock()
                mock_detector.detect_and_compare_drift.return_value = DriftDetectionResult(
                    success=True,
                    error_reason="",
                    error_details=None,
                    drift_differences=None,
                )
                mock_detector_class.return_value = mock_detector

                manager = VerifyManager(session, region_name="us-east-1")

                result = manager.verify_account_state("test-env", "123456789012")

                assert result.success is True


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
@patch("aws_bench.resource_management.verify.manager.compute_scenario_hash")
def test_verify_account_state_dataset_mismatch(mock_compute_hash, mock_snapshot_mgr_from_sessions):
    """Test verify when scenario hash doesn't match."""
    import tempfile
    from pathlib import Path

    session = boto3.Session(region_name="us-east-1")

    # Mock snapshot with scenario hash
    mock_snapshot = Snapshot(
        timestamp=datetime.now(timezone.utc),
        account_id="123456789012",
        environment_id="test-env",
        scenario_hash="hash_v1",
        drift_baseline={},
        stack_metadata={},
        resource_ids={},
    )
    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = mock_snapshot
    mock_snapshot_mgr_from_sessions.return_value = mock_snapshot_mgr

    # Mock scenario hash to return different value
    mock_compute_hash.return_value = "hash_v2"

    manager = VerifyManager(session)

    with tempfile.TemporaryDirectory() as tmpdir:
        scenario_dir = Path(tmpdir)
        (scenario_dir / "scenario").mkdir()
        (scenario_dir / "scenario" / "test.txt").write_text("test")

        result = manager.verify_account_state("test-env", "123456789012", scenario_dir=scenario_dir)

    assert result.success is False
    assert result.is_dataset_mismatch is True
    assert "Scenario version mismatch" in result.reason
    assert result.details is not None
    assert isinstance(result.details, dict)
    assert result.details["expected_hash"] == "hash_v1"
    assert result.details["current_hash"] == "hash_v2"


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
@patch("aws_bench.resource_management.verify.manager.make_scanner")
def test_verify_account_state_new_resources_found(
    mock_scanner_class, mock_snapshot_mgr_from_sessions
):
    """Test verify when new resources are found."""
    session = boto3.Session(region_name="us-east-1")

    # Mock snapshot

    mock_snapshot = Snapshot(
        timestamp=datetime.now(timezone.utc),
        account_id="123456789012",
        environment_id="test-env",
        scenario_hash="v1.0",
        drift_baseline={},
        stack_metadata={},
        resource_ids={},
    )
    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = mock_snapshot
    mock_snapshot_mgr_from_sessions.return_value = mock_snapshot_mgr

    # Mock scanner to find new resources
    from aws_bench.resource_management.ccapi.models import ScanResult

    mock_scanner = MagicMock()
    mock_scanner.scan_resources.return_value = ScanResult(
        detected={"AWS::S3::Bucket": [{"Identifier": "new-bucket"}]}, failed={}
    )
    mock_scanner_class.return_value = mock_scanner

    manager = VerifyManager(session)

    result = manager.verify_account_state("test-env", "123456789012")

    assert result.success is False
    assert "Found 1 new resource(s)" in result.reason
    assert result.new_resources is not None


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
@patch("aws_bench.resource_management.verify.manager.make_scanner")
@patch("aws_bench.resource_management.verify.manager.StackInspector")
def test_verify_account_state_stack_status_failed(
    mock_inspector_class, mock_scanner_class, mock_snapshot_mgr_from_sessions
):
    """Test verify when stack status check fails."""
    from aws_bench.resource_management.verify.models import StackStatusCheckResult

    session = boto3.Session(region_name="us-east-1")

    # Mock snapshot
    mock_snapshot = Snapshot(
        timestamp=datetime.now(timezone.utc),
        account_id="123456789012",
        environment_id="test-env",
        scenario_hash="v1.0",
        drift_baseline={},
        stack_metadata={"test-stack": StackMetadata("CREATE_COMPLETE", "sha256:abc")},
        resource_ids={},
    )
    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = mock_snapshot
    mock_snapshot_mgr_from_sessions.return_value = mock_snapshot_mgr

    # Mock scanner - no new resources
    from aws_bench.resource_management.ccapi.models import ScanResult

    mock_scanner = MagicMock()
    mock_scanner.scan_resources.return_value = ScanResult(detected={}, failed={})
    mock_scanner_class.return_value = mock_scanner

    # Mock inspector - status check fails
    mock_inspector = MagicMock()
    mock_inspector.check_stack_status.return_value = StackStatusCheckResult(
        success=False, error_reason="Stack status changed", error_details={}
    )
    mock_inspector_class.return_value = mock_inspector

    manager = VerifyManager(session)

    result = manager.verify_account_state("test-env", "123456789012")

    assert result.success is False
    assert "Stack status changed" in result.reason


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
@patch("aws_bench.resource_management.verify.manager.make_scanner")
@patch("aws_bench.resource_management.verify.manager.StackInspector")
def test_verify_account_state_template_hash_failed(
    mock_inspector_class, mock_scanner_class, mock_snapshot_mgr_from_sessions
):
    """Test verify when template hash check fails."""
    from aws_bench.resource_management.verify.models import (
        StackStatusCheckResult,
        TemplateHashCheckResult,
    )

    session = boto3.Session(region_name="us-east-1")

    # Mock snapshot
    mock_snapshot = Snapshot(
        timestamp=datetime.now(timezone.utc),
        account_id="123456789012",
        environment_id="test-env",
        scenario_hash="v1.0",
        drift_baseline={},
        stack_metadata={"test-stack": StackMetadata("CREATE_COMPLETE", "sha256:abc")},
        resource_ids={},
    )
    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = mock_snapshot
    mock_snapshot_mgr_from_sessions.return_value = mock_snapshot_mgr

    # Mock scanner - no new resources
    from aws_bench.resource_management.ccapi.models import ScanResult

    mock_scanner = MagicMock()
    mock_scanner.scan_resources.return_value = ScanResult(detected={}, failed={})
    mock_scanner_class.return_value = mock_scanner

    # Mock inspector
    mock_inspector = MagicMock()
    mock_inspector.check_stack_status.return_value = StackStatusCheckResult(
        success=True, error_reason="", error_details=None
    )
    mock_inspector.check_template_hash.return_value = TemplateHashCheckResult(
        success=False, error_reason="Template hash changed", error_details={}
    )
    mock_inspector_class.return_value = mock_inspector

    manager = VerifyManager(session)

    result = manager.verify_account_state("test-env", "123456789012")

    assert result.success is False
    assert "Template hash changed" in result.reason


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
@patch("aws_bench.resource_management.verify.manager.make_scanner")
@patch("aws_bench.resource_management.verify.manager.StackInspector")
def test_verify_account_state_template_mismatch_sets_redeploy_fields(
    mock_inspector_class, mock_scanner_class, mock_snapshot_mgr_from_sessions
):
    """The inspector's stack-name dict is surfaced as is_template_mismatch + the stack list.

    Guards the RC6-A seam: a field rename or inverted isinstance would break reset routing
    with every other test still green.
    """
    from aws_bench.resource_management.verify.models import (
        StackStatusCheckResult,
        TemplateHashCheckResult,
    )

    session = boto3.Session(region_name="us-east-1")

    mock_snapshot = Snapshot(
        timestamp=datetime.now(timezone.utc),
        account_id="123456789012",
        environment_id="test-env",
        scenario_hash="v1.0",
        drift_baseline={},
        stack_metadata={"s3-stack": StackMetadata("CREATE_COMPLETE", "sha256:abc")},
        resource_ids={},
    )
    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = mock_snapshot
    mock_snapshot_mgr_from_sessions.return_value = mock_snapshot_mgr

    from aws_bench.resource_management.ccapi.models import ScanResult

    mock_scanner = MagicMock()
    mock_scanner.scan_resources.return_value = ScanResult(detected={}, failed={})
    mock_scanner_class.return_value = mock_scanner

    mock_inspector = MagicMock()
    mock_inspector.check_stack_status.return_value = StackStatusCheckResult(
        success=True, error_reason="", error_details=None
    )
    mock_inspector.check_template_hash.return_value = TemplateHashCheckResult(
        success=False,
        error_reason="Stack s3-stack template changed",
        error_details={"template_mismatch_stacks": ["s3-stack"]},
    )
    mock_inspector_class.return_value = mock_inspector

    result = VerifyManager(session).verify_account_state("test-env", "123456789012")

    assert result.success is False
    assert result.is_template_mismatch is True
    assert result.template_mismatch_stacks == ["s3-stack"]


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
@patch("aws_bench.resource_management.verify.manager.make_scanner")
@patch("aws_bench.resource_management.verify.manager.StackInspector")
def test_verify_account_state_template_read_failure_is_not_remediable(
    mock_inspector_class, mock_scanner_class, mock_snapshot_mgr_from_sessions
):
    """A get_template READ failure (string error_details) must NOT set redeploy fields.

    Reset must never delete a stack it merely couldn't inspect.
    """
    from aws_bench.resource_management.verify.models import (
        StackStatusCheckResult,
        TemplateHashCheckResult,
    )

    session = boto3.Session(region_name="us-east-1")

    mock_snapshot = Snapshot(
        timestamp=datetime.now(timezone.utc),
        account_id="123456789012",
        environment_id="test-env",
        scenario_hash="v1.0",
        drift_baseline={},
        stack_metadata={"s3-stack": StackMetadata("CREATE_COMPLETE", "sha256:abc")},
        resource_ids={},
    )
    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = mock_snapshot
    mock_snapshot_mgr_from_sessions.return_value = mock_snapshot_mgr

    from aws_bench.resource_management.ccapi.models import ScanResult

    mock_scanner = MagicMock()
    mock_scanner.scan_resources.return_value = ScanResult(detected={}, failed={})
    mock_scanner_class.return_value = mock_scanner

    mock_inspector = MagicMock()
    mock_inspector.check_stack_status.return_value = StackStatusCheckResult(
        success=True, error_reason="", error_details=None
    )
    mock_inspector.check_template_hash.return_value = TemplateHashCheckResult(
        success=False,
        error_reason="Failed to verify template for s3-stack",
        error_details="AccessDenied",
    )
    mock_inspector_class.return_value = mock_inspector

    result = VerifyManager(session).verify_account_state("test-env", "123456789012")

    assert result.success is False
    assert result.is_template_mismatch is False
    assert result.template_mismatch_stacks is None


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
@patch("aws_bench.resource_management.verify.manager.make_scanner")
@patch("aws_bench.resource_management.verify.manager.StackInspector")
@patch("aws_bench.resource_management.verify.manager.DriftDetector")
def test_verify_account_state_drift_failed(
    mock_detector_class, mock_inspector_class, mock_scanner_class, mock_snapshot_mgr_from_sessions
):
    """Test verify when drift detection fails."""
    from aws_bench.resource_management.verify.models import (
        DriftDetectionResult,
        StackStatusCheckResult,
        TemplateHashCheckResult,
    )

    session = boto3.Session(region_name="us-east-1")

    # Mock snapshot

    mock_snapshot = Snapshot(
        timestamp=datetime.now(timezone.utc),
        account_id="123456789012",
        environment_id="test-env",
        scenario_hash="v1.0",
        drift_baseline={"test-stack": DriftBaseline("DETECTION_COMPLETE", [])},
        stack_metadata={"test-stack": StackMetadata("CREATE_COMPLETE", "sha256:abc")},
        resource_ids={},
    )
    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = mock_snapshot
    mock_snapshot_mgr_from_sessions.return_value = mock_snapshot_mgr

    # Mock scanner - no new resources
    from aws_bench.resource_management.ccapi.models import ScanResult

    mock_scanner = MagicMock()
    mock_scanner.scan_resources.return_value = ScanResult(detected={}, failed={})
    mock_scanner_class.return_value = mock_scanner

    # Mock inspector - all pass
    mock_inspector = MagicMock()
    mock_inspector.check_stack_status.return_value = StackStatusCheckResult(
        success=True, error_reason="", error_details=None
    )
    mock_inspector.check_template_hash.return_value = TemplateHashCheckResult(
        success=True, error_reason="", error_details=None
    )
    mock_inspector_class.return_value = mock_inspector

    # Mock drift detector - fails
    mock_detector = MagicMock()
    mock_detector.detect_and_compare_drift.return_value = DriftDetectionResult(
        success=False,
        error_reason="Drift detected",
        error_details={},
        drift_differences={"test-stack": {}},
    )
    mock_detector_class.return_value = mock_detector

    manager = VerifyManager(session)

    result = manager.verify_account_state("test-env", "123456789012")

    assert result.success is False
    assert "Drift detected" in result.reason
    assert result.drift_differences is not None


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
@patch("aws_bench.resource_management.verify.manager.make_scanner")
@patch("aws_bench.resource_management.verify.manager.StackInspector")
@patch("aws_bench.resource_management.verify.manager.DriftDetector")
def test_verify_account_state_all_checks_pass(
    mock_detector_class, mock_inspector_class, mock_scanner_class, mock_snapshot_mgr_from_sessions
):
    """Test verify when all checks pass."""
    from aws_bench.resource_management.verify.models import (
        DriftDetectionResult,
        StackStatusCheckResult,
        TemplateHashCheckResult,
    )

    session = boto3.Session(region_name="us-east-1")

    # Mock snapshot

    mock_snapshot = Snapshot(
        timestamp=datetime.now(timezone.utc),
        account_id="123456789012",
        environment_id="test-env",
        scenario_hash="v1.0",
        drift_baseline={"test-stack": DriftBaseline("DETECTION_COMPLETE", [])},
        stack_metadata={"test-stack": StackMetadata("CREATE_COMPLETE", "sha256:abc")},
        resource_ids={},
    )
    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = mock_snapshot
    mock_snapshot_mgr_from_sessions.return_value = mock_snapshot_mgr

    # Mock scanner - no new resources
    from aws_bench.resource_management.ccapi.models import ScanResult

    mock_scanner = MagicMock()
    mock_scanner.scan_resources.return_value = ScanResult(detected={}, failed={})
    mock_scanner_class.return_value = mock_scanner

    # Mock inspector - all pass
    mock_inspector = MagicMock()
    mock_inspector.check_stack_status.return_value = StackStatusCheckResult(
        success=True, error_reason="", error_details=None
    )
    mock_inspector.check_template_hash.return_value = TemplateHashCheckResult(
        success=True, error_reason="", error_details=None
    )
    mock_inspector_class.return_value = mock_inspector

    # Mock drift detector - passes
    mock_detector = MagicMock()
    mock_detector.detect_and_compare_drift.return_value = DriftDetectionResult(
        success=True,
        error_reason="",
        error_details=None,
        drift_differences=None,
    )
    mock_detector_class.return_value = mock_detector

    manager = VerifyManager(session)

    result = manager.verify_account_state("test-env", "123456789012")

    assert result.success is True
    assert "matches post-setup baseline" in result.reason


# ===========================================================================
# verify_account_state — skip_early flag (gate vs reset diagnosis)
# ===========================================================================


def _aggregation_snapshot() -> Snapshot:
    """Snapshot whose baseline a failing new-resources + stack-status run compares against."""
    return Snapshot(
        timestamp=datetime.now(timezone.utc),
        account_id="123456789012",
        environment_id="test-env",
        scenario_hash="v1.0",
        drift_baseline={"test-stack": DriftBaseline("DETECTION_COMPLETE", [])},
        stack_metadata={"test-stack": StackMetadata("CREATE_COMPLETE", "sha256:abc")},
        resource_ids={},
    )


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
@patch("aws_bench.resource_management.verify.manager.make_scanner")
@patch("aws_bench.resource_management.verify.manager.StackInspector")
@patch("aws_bench.resource_management.verify.manager.DriftDetector")
def test_verify_account_state_skip_early_true_short_circuits(
    mock_detector_class, mock_inspector_class, mock_scanner_class, mock_snapshot_mgr_class
):
    """With skip_early=True (the gate), a new-resource failure returns immediately.

    The stack-status check must NOT run, so its failure is not surfaced — this is
    the existing gate behavior and stays the default.
    """
    from aws_bench.resource_management.ccapi.models import ScanResult
    from aws_bench.resource_management.verify.models import StackStatusCheckResult

    session = boto3.Session(region_name="us-east-1")

    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = _aggregation_snapshot()
    mock_snapshot_mgr_class.return_value = mock_snapshot_mgr

    # New resources present -> _check_new_resources (index 1) fails first.
    mock_scanner = MagicMock()
    mock_scanner.scan_resources.return_value = ScanResult(
        detected={"AWS::S3::Bucket": [{"Identifier": "new-bucket"}]}, failed={}
    )
    mock_scanner_class.return_value = mock_scanner

    # Stack status would ALSO fail, but must not be consulted under skip_early.
    mock_inspector = MagicMock()
    mock_inspector.check_stack_status.return_value = StackStatusCheckResult(
        success=False,
        error_reason="1 stack(s) have status mismatch",
        error_details={"test-stack": {"expected": "CREATE_COMPLETE", "actual": "DELETE_FAILED"}},
    )
    mock_inspector_class.return_value = mock_inspector

    manager = VerifyManager(session)
    result = manager.verify_account_state("test-env", "123456789012", skip_early=True)

    assert result.success is False
    assert result.new_resources is not None
    # Short-circuited before stack status: not surfaced, and the check never ran.
    assert result.stack_status_failures is None
    mock_inspector.check_stack_status.assert_not_called()


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
@patch("aws_bench.resource_management.verify.manager.make_scanner")
@patch("aws_bench.resource_management.verify.manager.StackInspector")
@patch("aws_bench.resource_management.verify.manager.DriftDetector")
def test_verify_account_state_skip_early_false_aggregates_all_failures(
    mock_detector_class, mock_inspector_class, mock_scanner_class, mock_snapshot_mgr_class
):
    """With skip_early=False (reset diagnosis), ALL checks run and aggregate.

    A run where new resources, a DELETE_FAILED stack, AND drift all fail must
    return ONE VerifyResult carrying every category, so reset can remediate them
    together (the bug: a DELETE_FAILED stack was masked behind a new-resource
    failure that short-circuited the diagnosis).
    """
    from aws_bench.resource_management.ccapi.models import ScanResult
    from aws_bench.resource_management.verify.models import (
        DriftDetectionResult,
        StackStatusCheckResult,
        TemplateHashCheckResult,
    )

    session = boto3.Session(region_name="us-east-1")

    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = _aggregation_snapshot()
    mock_snapshot_mgr_class.return_value = mock_snapshot_mgr

    mock_scanner = MagicMock()
    mock_scanner.scan_resources.return_value = ScanResult(
        detected={"AWS::S3::Bucket": [{"Identifier": "new-bucket"}]}, failed={}
    )
    mock_scanner_class.return_value = mock_scanner

    mock_inspector = MagicMock()
    mock_inspector.check_stack_status.return_value = StackStatusCheckResult(
        success=False,
        error_reason="1 stack(s) have status mismatch",
        error_details={"test-stack": {"expected": "CREATE_COMPLETE", "actual": "DELETE_FAILED"}},
    )
    mock_inspector.check_template_hash.return_value = TemplateHashCheckResult(
        success=True, error_reason="", error_details=None
    )
    mock_inspector_class.return_value = mock_inspector

    mock_detector = MagicMock()
    mock_detector.detect_and_compare_drift.return_value = DriftDetectionResult(
        success=False,
        error_reason="Drift detected",
        error_details={},
        drift_differences={"test-stack": {"baseline": {}, "current": {}}},
    )
    mock_detector_class.return_value = mock_detector

    manager = VerifyManager(session)
    result = manager.verify_account_state("test-env", "123456789012", skip_early=False)

    assert result.success is False
    # All three remediable categories are populated on the single result.
    assert result.new_resources is not None
    assert result.stack_status_failures is not None
    assert result.stack_status_failures["test-stack"]["actual"] == "DELETE_FAILED"
    assert result.drift_differences is not None
    # The stack-status check actually ran (no short-circuit).
    mock_inspector.check_stack_status.assert_called_once()


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
@patch("aws_bench.resource_management.verify.manager.compute_scenario_hash")
def test_verify_account_state_skip_early_false_dataset_mismatch_stops(
    mock_compute_hash, mock_snapshot_mgr_class
):
    """An unrecoverable dataset-version mismatch returns immediately even when skip_early=False.

    A scenario-hash mismatch means the baseline itself is invalid — there is
    nothing to aggregate against, so reset must still bail to full cleanup.
    """
    import tempfile
    from pathlib import Path

    session = boto3.Session(region_name="us-east-1")

    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = _aggregation_snapshot()
    mock_snapshot_mgr_class.return_value = mock_snapshot_mgr
    mock_compute_hash.return_value = "different_hash"

    manager = VerifyManager(session)
    with tempfile.TemporaryDirectory() as tmpdir:
        scenario_dir = Path(tmpdir)
        (scenario_dir / "scenario").mkdir()
        (scenario_dir / "scenario" / "x.txt").write_text("x")
        result = manager.verify_account_state(
            "test-env", "123456789012", scenario_dir=scenario_dir, skip_early=False
        )

    assert result.success is False
    assert result.is_dataset_mismatch is True


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
@patch("aws_bench.resource_management.verify.manager.make_scanner")
@patch("aws_bench.resource_management.verify.manager.StackInspector")
@patch("aws_bench.resource_management.verify.manager.DriftDetector")
def test_verify_account_state_skip_early_false_all_pass(
    mock_detector_class, mock_inspector_class, mock_scanner_class, mock_snapshot_mgr_class
):
    """With skip_early=False and no issues, the aggregate result is success."""
    from aws_bench.resource_management.ccapi.models import ScanResult
    from aws_bench.resource_management.verify.models import (
        DriftDetectionResult,
        StackStatusCheckResult,
        TemplateHashCheckResult,
    )

    session = boto3.Session(region_name="us-east-1")

    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = _aggregation_snapshot()
    mock_snapshot_mgr_class.return_value = mock_snapshot_mgr

    mock_scanner = MagicMock()
    mock_scanner.scan_resources.return_value = ScanResult(detected={}, failed={})
    mock_scanner_class.return_value = mock_scanner

    mock_inspector = MagicMock()
    mock_inspector.check_stack_status.return_value = StackStatusCheckResult(
        success=True, error_reason="", error_details=None
    )
    mock_inspector.check_template_hash.return_value = TemplateHashCheckResult(
        success=True, error_reason="", error_details=None
    )
    mock_inspector_class.return_value = mock_inspector

    mock_detector = MagicMock()
    mock_detector.detect_and_compare_drift.return_value = DriftDetectionResult(
        success=True, error_reason="", error_details=None, drift_differences=None
    )
    mock_detector_class.return_value = mock_detector

    manager = VerifyManager(session)
    result = manager.verify_account_state("test-env", "123456789012", skip_early=False)

    assert result.success is True


# ===========================================================================
# _check_new_resources — fail closed on an unenumerable baseline type
# ===========================================================================


@mock_aws
@patch("aws_bench.resource_management.verify.manager.make_scanner")
def test_check_new_resources_fails_closed_on_unenumerable_baseline_type(mock_scanner_class):
    """A baseline-tracked type still in scan_result.failed fails verify closed.

    The scanner already retries transient errors, so a persistent failure on a type
    that mattered at setup could hide a leaked resource forever if silently skipped.
    """
    from aws_bench.resource_management.ccapi.models import ScanResult

    session = boto3.Session(region_name="us-east-1")

    mock_scanner = MagicMock()
    mock_scanner.scan_resources.return_value = ScanResult(
        detected={}, failed={"AWS::EC2::InternetGateway": "throttled"}
    )
    mock_scanner_class.return_value = mock_scanner

    manager = VerifyManager(session)
    result = manager._check_new_resources(
        baseline_resource_ids={"AWS::EC2::InternetGateway": [{"Identifier": "igw-1"}]},
        baseline_failed={},
        baseline_empty=set(),
    )

    assert result is not None
    assert result.success is False
    assert "Could not enumerate 1 baseline resource type(s)" in result.reason
    assert isinstance(result.details, dict)
    assert result.details["unenumerable_types"] == ["AWS::EC2::InternetGateway"]


@mock_aws
@patch("aws_bench.resource_management.verify.manager.make_scanner")
def test_check_new_resources_ignores_failed_type_absent_from_baseline(mock_scanner_class):
    """A failed type that was NOT in the baseline does not fail verify.

    The fail-closed rule is scoped to baseline-tracked types; a newly-discovered
    type failing to enumerate is not a leaked-baseline-resource risk.
    """
    from aws_bench.resource_management.ccapi.models import ScanResult

    session = boto3.Session(region_name="us-east-1")

    mock_scanner = MagicMock()
    # The failed type is absent from the baseline (baseline only tracks S3 buckets).
    mock_scanner.scan_resources.return_value = ScanResult(
        detected={}, failed={"AWS::EC2::InternetGateway": "throttled"}
    )
    mock_scanner_class.return_value = mock_scanner

    manager = VerifyManager(session)
    result = manager._check_new_resources(
        baseline_resource_ids={"AWS::S3::Bucket": [{"Identifier": "b1"}]},
        baseline_failed={},
        baseline_empty=set(),
    )

    # No unenumerable baseline type and no new resources -> clean.
    assert result is None


@mock_aws
@patch("aws_bench.resource_management.verify.manager.make_scanner")
def test_check_new_resources_tolerates_region_unavailable_baseline_type(mock_scanner_class):
    """A baseline type failing with a region-unavailability code does NOT fail verify.

    A resource type that cannot exist in the scanned region cannot hide an orphan, so
    a stable region/service-availability error is tolerated rather than failing closed.
    """
    from aws_bench.resource_management.ccapi.models import ScanResult

    session = boto3.Session(region_name="us-east-1")

    mock_scanner = MagicMock()
    mock_scanner.scan_resources.return_value = ScanResult(
        detected={}, failed={"AWS::GameLift::ContainerFleet": "UnsupportedRegionException"}
    )
    mock_scanner_class.return_value = mock_scanner

    manager = VerifyManager(session)
    result = manager._check_new_resources(
        baseline_resource_ids={},
        baseline_failed={},
        baseline_empty={"AWS::GameLift::ContainerFleet"},
    )

    # Tolerated -> no unenumerable failure and no new resources -> clean.
    assert result is None


@mock_aws
@patch("aws_bench.resource_management.verify.manager.make_scanner")
def test_check_new_resources_tolerates_endpoint_connection_error(mock_scanner_class):
    """An endpoint-connection failure (no regional endpoint) is region-unavailability."""
    from aws_bench.resource_management.ccapi.models import ScanResult

    session = boto3.Session(region_name="us-west-2")

    mock_scanner = MagicMock()
    mock_scanner.scan_resources.return_value = ScanResult(
        detected={},
        failed={
            "AWS::CUR::ReportDefinition": (
                'Connect timeout on endpoint URL: "https://cur.us-west-2.amazonaws.com/"'
            )
        },
    )
    mock_scanner_class.return_value = mock_scanner

    manager = VerifyManager(session)
    result = manager._check_new_resources(
        baseline_resource_ids={},
        baseline_failed={},
        baseline_empty={"AWS::CUR::ReportDefinition"},
    )

    assert result is None


@mock_aws
@patch("aws_bench.resource_management.verify.manager.make_scanner")
def test_check_new_resources_fails_closed_on_access_denied(mock_scanner_class):
    """AccessDenied fails closed: a real permission gap, not region-unavailability."""
    from aws_bench.resource_management.ccapi.models import ScanResult

    session = boto3.Session(region_name="us-east-1")

    mock_scanner = MagicMock()
    mock_scanner.scan_resources.return_value = ScanResult(
        detected={}, failed={"AWS::S3::Bucket": "AccessDeniedException"}
    )
    mock_scanner_class.return_value = mock_scanner

    manager = VerifyManager(session)
    result = manager._check_new_resources(
        baseline_resource_ids={"AWS::S3::Bucket": [{"Identifier": "b1"}]},
        baseline_failed={},
        baseline_empty=set(),
    )

    assert result is not None
    assert result.success is False
    assert "Could not enumerate 1 baseline resource type(s)" in result.reason
    assert result.details["unenumerable_types"] == ["AWS::S3::Bucket"]


@mock_aws
@patch("aws_bench.resource_management.verify.manager.make_scanner")
def test_check_new_resources_mixed_tolerates_region_unavailable_but_fails_on_genuine(
    mock_scanner_class,
):
    """A region-unavailable type is tolerated while a genuine failure still fails closed."""
    from aws_bench.resource_management.ccapi.models import ScanResult

    session = boto3.Session(region_name="us-east-1")

    mock_scanner = MagicMock()
    mock_scanner.scan_resources.return_value = ScanResult(
        detected={},
        failed={
            "AWS::GameLift::ContainerFleet": "UnsupportedRegionException",
            "AWS::EC2::InternetGateway": "ThrottlingException",
        },
    )
    mock_scanner_class.return_value = mock_scanner

    manager = VerifyManager(session)
    result = manager._check_new_resources(
        baseline_resource_ids={"AWS::EC2::InternetGateway": [{"Identifier": "igw-1"}]},
        baseline_failed={},
        baseline_empty={"AWS::GameLift::ContainerFleet"},
    )

    assert result is not None
    assert result.success is False
    # Only the genuine (non-region-unavailable) failure is surfaced.
    assert result.details["unenumerable_types"] == ["AWS::EC2::InternetGateway"]


@mock_aws
@patch("aws_bench.resource_management.verify.manager.make_scanner")
def test_check_new_resources_type_with_baseline_resources_fails_closed_despite_region_code(
    mock_scanner_class,
):
    """A type that HAD resources at baseline fails closed even on a region-unavailable code.

    Toleration is scoped to baseline-empty types; a type we tracked resources for must
    never be silently skipped, whatever error its scan returns.
    """
    from aws_bench.resource_management.ccapi.models import ScanResult

    session = boto3.Session(region_name="us-east-1")

    mock_scanner = MagicMock()
    mock_scanner.scan_resources.return_value = ScanResult(
        detected={}, failed={"AWS::GameLift::ContainerFleet": "UnsupportedRegionException"}
    )
    mock_scanner_class.return_value = mock_scanner

    manager = VerifyManager(session)
    result = manager._check_new_resources(
        # The type had resources at baseline (not empty), so it must fail closed.
        baseline_resource_ids={"AWS::GameLift::ContainerFleet": [{"Identifier": "fleet-1"}]},
        baseline_failed={},
        baseline_empty=set(),
    )

    assert result is not None
    assert result.success is False
    assert result.details["unenumerable_types"] == ["AWS::GameLift::ContainerFleet"]


# ===========================================================================
# find_orphan_resources — reset's orphan/scan-health census wrapper
# ===========================================================================


@mock_aws
@patch("aws_bench.resource_management.verify.manager.make_scanner")
def test_find_orphan_resources_flags_orphan_from_snapshot(mock_scanner_class):
    """The wrapper runs only the new-resource census and surfaces a orphan."""
    from aws_bench.resource_management.ccapi.models import ScanResult

    session = boto3.Session(region_name="us-east-1")

    mock_scanner = MagicMock()
    mock_scanner.scan_resources.return_value = ScanResult(
        detected={"AWS::S3::Bucket": [{"Identifier": "orphan-bucket"}]}, failed={}
    )
    mock_scanner_class.return_value = mock_scanner

    snapshot = Snapshot(
        timestamp=datetime.now(timezone.utc),
        account_id="123456789012",
        environment_id="test-env",
        scenario_hash="v1.0",
        drift_baseline={},
        stack_metadata={},
        resource_ids={"AWS::S3::Bucket": []},
        empty_resource_types={"AWS::S3::Bucket"},
    )

    result = VerifyManager(session).find_orphan_resources(snapshot)

    assert result is not None
    assert result.success is False
    assert result.new_resources is not None
    assert "AWS::S3::Bucket" in result.new_resources


# ===========================================================================
# _check_dataset_version — scenario hash comparison
# ===========================================================================


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
def test_check_dataset_version_hash_mismatch(mock_snapshot_mgr_from_sessions):
    """Returns failure when scenario hash mismatches."""
    import tempfile
    from pathlib import Path

    session = boto3.Session(region_name="us-east-1")

    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr_from_sessions.return_value = mock_snapshot_mgr

    with tempfile.TemporaryDirectory() as tmpdir:
        scenario_dir = Path(tmpdir)
        build_context = scenario_dir / "scenario"
        build_context.mkdir()
        (build_context / "Dockerfile").write_text("FROM python:3.12")

        # Snapshot has different hash
        manager = VerifyManager(session)

        # Mock compute_scenario_hash to return a specific hash
        with patch(
            "aws_bench.resource_management.verify.manager.compute_scenario_hash"
        ) as mock_compute:
            mock_compute.return_value = "current_hash_value"

            result = manager._check_dataset_version("different_hash_value", scenario_dir)

            assert result is not None
            assert not result.success
            assert result.reason == "Scenario version mismatch"
            assert result.is_dataset_mismatch
            assert isinstance(result.details, dict)
            assert result.details["expected_hash"] == "different_hash_value"
            assert result.details["current_hash"] == "current_hash_value"


# ===========================================================================
# _filter_snapshot_to_region — per-region baseline scoping
# ===========================================================================


def _multiregion_snapshot():
    """Snapshot with stacks across us-east-1 and us-west-1."""
    return Snapshot(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        account_id="123456789012",
        environment_id="env-test",
        scenario_hash="h1",
        drift_baseline={
            "stack-e1": DriftBaseline(detection_status="DETECTION_COMPLETE", resource_drifts=[]),
            "stack-w1": DriftBaseline(detection_status="DETECTION_COMPLETE", resource_drifts=[]),
        },
        stack_metadata={
            "stack-e1": StackMetadata(
                status="CREATE_COMPLETE", template_hash="sha256:e", region="us-east-1"
            ),
            "stack-w1": StackMetadata(
                status="CREATE_COMPLETE", template_hash="sha256:w", region="us-west-1"
            ),
        },
        resource_ids={"AWS::IAM::Role": [{"Identifier": "r"}]},
        regions=["us-east-1", "us-west-1"],
    )


def test_filter_snapshot_to_region_keeps_only_that_region():
    """Only the region's stacks/drift survive; other regions are dropped."""
    snap = _multiregion_snapshot()
    filtered = VerifyManager._filter_snapshot_to_region(snap, "us-east-1")

    assert set(filtered.stack_metadata) == {"stack-e1"}
    assert set(filtered.drift_baseline) == {"stack-e1"}
    # Resource ids are untouched (scanned live, region-filtered separately).
    assert filtered.resource_ids == snap.resource_ids
    # Original snapshot is not mutated.
    assert set(snap.stack_metadata) == {"stack-e1", "stack-w1"}


def test_filter_snapshot_to_region_legacy_no_region_unchanged():
    """Snapshots without per-stack region info are returned unchanged."""
    snap = _multiregion_snapshot()
    # Simulate a legacy snapshot: clear region on all stacks.
    for meta in snap.stack_metadata.values():
        meta.region = ""

    filtered = VerifyManager._filter_snapshot_to_region(snap, "us-east-1")
    # No filtering: both stacks retained.
    assert set(filtered.stack_metadata) == {"stack-e1", "stack-w1"}


# ===========================================================================
# verify_account_state — snapshot not found path
# ===========================================================================


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
def test_verify_account_state_no_snapshot_exists(mock_snapshot_mgr_class):
    """Test verify returns failure when snapshot_exists check returns False."""
    session = boto3.Session(region_name="us-east-1")

    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.snapshot_exists.return_value = False
    mock_snapshot_mgr_class.return_value = mock_snapshot_mgr

    manager = VerifyManager(session)
    result = manager.verify_account_state("test-env", "123456789012")

    assert result.success is False
    assert "Snapshot not found" in result.reason


# ===========================================================================
# _check_dataset_version — exception path
# ===========================================================================


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
@patch("aws_bench.resource_management.verify.manager.compute_scenario_hash")
def test_check_dataset_version_hash_compute_error(mock_hash, mock_snapshot_mgr_class):
    """Test that hash computation errors skip the check (return None)."""
    session = boto3.Session(region_name="us-east-1")

    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr_class.return_value = mock_snapshot_mgr

    mock_hash.side_effect = OSError("Cannot read scenario dir")

    manager = VerifyManager(session)
    from pathlib import Path

    result = manager._check_dataset_version("abc123", Path("/nonexistent"))

    assert result is None


# ===========================================================================
# _check_stack_status — API error path
# ===========================================================================


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
def test_check_stack_status_api_error(mock_snapshot_mgr_class):
    """Test that stack status check returns proper result on API error."""
    from aws_bench.resource_management.verify.models import StackStatusCheckResult

    session = boto3.Session(region_name="us-east-1")
    mock_snapshot_mgr_class.return_value = MagicMock()

    with patch(
        "aws_bench.resource_management.verify.manager.StackInspector"
    ) as mock_inspector_class:
        mock_inspector = MagicMock()
        mock_inspector.check_stack_status.return_value = StackStatusCheckResult(
            success=False,
            error_reason="Failed to list stacks",
            error_details={"error": "AccessDenied"},
        )
        mock_inspector_class.return_value = mock_inspector

        manager = VerifyManager(session)
        result = manager._check_stack_status({})

        assert result is not None
        assert result.success is False
        assert result.suggestion is not None
        assert "Check AWS permissions" in result.suggestion


# ===========================================================================
# verify_account_multiregion
# ===========================================================================


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
def test_verify_account_multiregion_success(mock_snapshot_mgr_class):
    """Test multiregion verification with all regions passing."""
    session = boto3.Session(region_name="us-east-1")

    mock_snapshot = Snapshot(
        timestamp=datetime.now(timezone.utc),
        account_id="123456789012",
        environment_id="test-env",
        scenario_hash="v1.0",
        drift_baseline={},
        stack_metadata={},
        resource_ids={},
        regions=["us-east-1"],
    )
    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = mock_snapshot
    mock_snapshot_mgr_class.return_value = mock_snapshot_mgr

    with patch.object(VerifyManager, "verify_account_state") as mock_verify:
        from aws_bench.resource_management.verify.models import VerifyResult

        mock_verify.return_value = VerifyResult(success=True, reason="OK")

        manager = VerifyManager(session)
        result = manager.verify_account_multiregion("test-env", "123456789012", "env-1")

    assert result.success is True
    assert result.account_id == "123456789012"
    assert len(result.region_results) == 1
    assert result.region_results[0].region == "us-east-1"
    assert result.region_results[0].success is True


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
def test_verify_account_multiregion_snapshot_not_found(mock_snapshot_mgr_class):
    """Test multiregion verification when snapshot is missing."""
    session = boto3.Session(region_name="us-east-1")

    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.side_effect = SnapshotNotFoundError("test-env", "123456789012")
    mock_snapshot_mgr_class.return_value = mock_snapshot_mgr

    manager = VerifyManager(session)
    result = manager.verify_account_multiregion("test-env", "123456789012", "env-1")

    assert result.success is False
    assert result.error_message is not None
    assert result.region_results == []


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
def test_verify_account_multiregion_unexpected_exception(mock_snapshot_mgr_class):
    """Test multiregion verification handles unexpected exceptions gracefully."""
    session = boto3.Session(region_name="us-east-1")

    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.side_effect = RuntimeError("unexpected")
    mock_snapshot_mgr_class.return_value = mock_snapshot_mgr

    manager = VerifyManager(session)
    result = manager.verify_account_multiregion("test-env", "123456789012", "env-1")

    assert result.success is False
    assert result.error_message is not None
    assert "Verification failed" in result.error_message
    assert result.region_results == []


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
def test_verify_account_multiregion_region_failure(mock_snapshot_mgr_class):
    """Test multiregion verification with one region failing."""
    session = boto3.Session(region_name="us-east-1")

    mock_snapshot = Snapshot(
        timestamp=datetime.now(timezone.utc),
        account_id="123456789012",
        environment_id="test-env",
        scenario_hash="v1.0",
        drift_baseline={},
        stack_metadata={},
        resource_ids={},
        regions=["us-east-1", "us-west-2"],
    )
    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = mock_snapshot
    mock_snapshot_mgr_class.return_value = mock_snapshot_mgr

    # Regions are verified concurrently, so key the outcome off the region
    # (via the per-region VerifyManager's _region_name) rather than call order:
    # us-east-1 passes, us-west-2 fails, regardless of which finishes first.
    def mock_verify(self, *args, **kwargs):
        from aws_bench.resource_management.verify.models import VerifyResult

        if self._region_name == "us-west-2":
            return VerifyResult(success=False, reason="Drift detected")
        return VerifyResult(success=True, reason="OK")

    with patch.object(
        VerifyManager, "verify_account_state", autospec=True, side_effect=mock_verify
    ):
        manager = VerifyManager(session)
        result = manager.verify_account_multiregion("test-env", "123456789012", "env-1")

    assert result.success is False
    assert len(result.region_results) == 2
    # region_results is emitted in input-region order: [us-east-1, us-west-2].
    assert result.region_results[0].region == "us-east-1"
    assert result.region_results[0].success is True
    assert result.region_results[1].region == "us-west-2"
    assert result.region_results[1].success is False


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
def test_verify_account_multiregion_isolates_region_exception(mock_snapshot_mgr_class):
    """A region scan that *raises* is isolated to that region's failed result.

    The other regions' results must survive — a single region blowing up must
    not discard the whole account's verification (the bug the per-region
    try/except guards against).
    """
    session = boto3.Session(region_name="us-east-1")

    mock_snapshot = Snapshot(
        timestamp=datetime.now(timezone.utc),
        account_id="123456789012",
        environment_id="test-env",
        scenario_hash="v1.0",
        drift_baseline={},
        stack_metadata={},
        resource_ids={},
        regions=["us-east-1", "us-west-2"],
    )
    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = mock_snapshot
    mock_snapshot_mgr_class.return_value = mock_snapshot_mgr

    def mock_verify(self, *args, **kwargs):
        from aws_bench.resource_management.verify.models import VerifyResult

        if self._region_name == "us-west-2":
            raise RuntimeError("boom in us-west-2")
        return VerifyResult(success=True, reason="OK")

    with patch.object(
        VerifyManager, "verify_account_state", autospec=True, side_effect=mock_verify
    ):
        manager = VerifyManager(session)
        result = manager.verify_account_multiregion("test-env", "123456789012", "env-1")

    assert result.success is False
    assert len(result.region_results) == 2
    # us-east-1 still verified successfully despite us-west-2 raising.
    assert result.region_results[0].region == "us-east-1"
    assert result.region_results[0].success is True
    assert result.region_results[1].region == "us-west-2"
    assert result.region_results[1].success is False
    error_message = result.region_results[1].error_message
    assert error_message is not None
    assert "boom in us-west-2" in error_message


@mock_aws
@patch("aws_bench.resource_management.verify.manager.SnapshotManager")
def test_verify_account_multiregion_propagates_cancel(mock_snapshot_mgr_class):
    """A shutdown mid-region propagates; it is NOT recorded as a failed region.

    ``OperationCancelled`` is a ``BaseException``, so it bypasses the per-region
    ``except Exception`` and unwinds out of the executor rather than being turned
    into a ``success=False`` region the run would continue past. The inverse of
    test_verify_account_multiregion_isolates_region_exception.
    """
    from aws_bench.exceptions import OperationCancelled

    session = boto3.Session(region_name="us-east-1")

    mock_snapshot = Snapshot(
        timestamp=datetime.now(timezone.utc),
        account_id="123456789012",
        environment_id="test-env",
        scenario_hash="v1.0",
        drift_baseline={},
        stack_metadata={},
        resource_ids={},
        regions=["us-east-1", "us-west-2"],
    )
    mock_snapshot_mgr = MagicMock()
    mock_snapshot_mgr.load_snapshot.return_value = mock_snapshot
    mock_snapshot_mgr_class.return_value = mock_snapshot_mgr

    def mock_verify(self, *args, **kwargs):
        raise OperationCancelled("shutdown")

    with patch.object(
        VerifyManager, "verify_account_state", autospec=True, side_effect=mock_verify
    ):
        manager = VerifyManager(session)
        with pytest.raises(OperationCancelled):
            manager.verify_account_multiregion("test-env", "123456789012", "env-1")
