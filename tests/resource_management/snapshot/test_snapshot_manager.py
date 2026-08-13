# tests/resource_management/snapshot/test_manager.py
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import boto3
import pytest
import tenacity
from botocore.exceptions import ClientError
from moto import mock_aws

from aws_bench.resource_management.exceptions import DriftDetectionError, SnapshotNotFoundError
from aws_bench.resource_management.snapshot.manager import SnapshotManager
from aws_bench.resource_management.snapshot.models import (
    DriftBaseline,
    ResourceDrift,
    Snapshot,
    SnapshotContext,
    SnapshotKey,
    SnapshotStage,
    StackMetadata,
)
from aws_bench.resource_management.storage.exceptions import StorageConflictError
from aws_bench.resource_management.storage.local_backend import LocalStorageBackend
from aws_bench.resource_management.storage.s3_backend import S3StorageBackend


def create_test_manager() -> SnapshotManager:
    """Helper to create SnapshotManager for tests using moto S3."""
    # Mock CredentialProvider to return test sessions
    mock_cred_provider = MagicMock()
    mgmt_session = boto3.Session(region_name="us-east-1")
    mock_cred_provider.get_management_session.return_value = mgmt_session

    patch_path = "aws_bench.resource_management.snapshot.manager.CredentialProvider.get"
    with patch(patch_path, return_value=mock_cred_provider):
        manager = SnapshotManager()
        return manager


@pytest.fixture
def temp_snapshot_dir():
    """Create temporary directory for snapshots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_preexisting_mode_stores_state_on_local_disk(tmp_path, monkeypatch, sample_snapshot):
    """Pre-existing mode resolves the local backend and writes the snapshot under STATE_DIR."""
    monkeypatch.setattr(
        "aws_bench.resource_management.snapshot.manager.STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "aws_bench.resource_management.snapshot.manager.active_account_config",
        lambda: (MagicMock(), Path("accounts.yaml")),
    )
    manager = SnapshotManager()

    assert isinstance(manager._storage, LocalStorageBackend)

    manager.save_snapshot("env-test", sample_snapshot)

    written = tmp_path / "state/snapshots/env-test/post-setup/123456789012/baseline.json"
    assert json.loads(written.read_text())["account_id"] == "123456789012"


def test_managed_mode_still_uses_the_s3_state_bucket(monkeypatch):
    monkeypatch.setattr(
        "aws_bench.resource_management.snapshot.manager.active_account_config", lambda: None
    )
    with mock_aws():
        manager = create_test_manager()
        assert isinstance(manager._storage, S3StorageBackend)


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
                template_hash="sha256:test",
                parameters={"Param1": "Value1"},
                tags={"Env": "test"},
            )
        },
        resource_ids={"AWS::IAM::Role": [{"Identifier": "MyRole"}]},
    )


@mock_aws
def test_snapshot_manager_initialization(temp_snapshot_dir):
    """Test SnapshotManager initialization with lazy S3 backend.

    S3 backend is not initialized until first storage access to avoid
    unnecessary API calls in code paths that only use CloudFormation APIs.
    """
    manager = create_test_manager()
    # S3 backend is not initialized yet (lazy)
    assert manager._backend is None
    assert manager._etags == {}

    # Access _storage property to trigger lazy initialization
    storage = manager._storage
    assert storage is not None
    assert manager._backend is not None


@mock_aws
def test_lazy_initialization_avoids_s3_cost_for_non_storage_operations(temp_snapshot_dir):
    """Test that S3 backend is not initialized on construction.

    Constructing a SnapshotManager must not eagerly pay the S3 initialization
    cost (STS get_caller_identity + head_bucket + config PUTs); the backend is
    created lazily on first storage access. This matters because CloudFormation-
    only code paths (e.g. drift capture) construct a SnapshotManager but never
    touch S3, and multi-region verify creates many of them.
    """
    manager = create_test_manager()

    # S3 backend is not initialized on construction
    assert manager._backend is None

    # Verify S3 is only initialized when storage operations are called
    # (not when CloudFormation-only code paths run)


@mock_aws
def test_save_snapshot(temp_snapshot_dir, sample_snapshot):
    """Test saving a snapshot to S3 and local audit."""
    manager = create_test_manager()
    manager.save_snapshot("test-env", sample_snapshot)

    # Verify S3 file was created (bucket is awsbench-state-{account-id})
    # Note: S3StorageBackend uses "snapshots/" prefix by default
    mgmt_session = boto3.Session(region_name="us-east-1")
    s3 = mgmt_session.client("s3", region_name="us-east-1")
    s3_key = "snapshots/test-env/post-setup/123456789012/baseline.json"
    obj = s3.get_object(Bucket="awsbench-state-123456789012", Key=s3_key)
    data = json.loads(obj["Body"].read())

    assert data["account_id"] == "123456789012"
    assert data["environment_id"] == "env-test"
    assert data["scenario_hash"] == "abc123"


@mock_aws
def test_load_snapshot(temp_snapshot_dir, sample_snapshot):
    """Test loading a snapshot from S3."""
    manager = create_test_manager()

    # Save first
    manager.save_snapshot("test-env", sample_snapshot)

    # Load and verify
    loaded = manager.load_snapshot("test-env", "123456789012")

    assert loaded.account_id == sample_snapshot.account_id
    assert loaded.environment_id == sample_snapshot.environment_id
    assert loaded.scenario_hash == sample_snapshot.scenario_hash
    assert loaded.timestamp == sample_snapshot.timestamp
    assert "test-stack" in loaded.drift_baseline
    assert loaded.drift_baseline["test-stack"].detection_status == "DETECTION_COMPLETE"
    assert loaded.drift_baseline["test-stack"].resource_drifts[0].logical_resource_id == "MyRole"
    assert "test-stack" in loaded.stack_metadata
    assert loaded.stack_metadata["test-stack"].status == "CREATE_COMPLETE"
    assert loaded.stack_metadata["test-stack"].template_hash == "sha256:test"


@mock_aws
def test_load_snapshot_not_found(temp_snapshot_dir):
    """Test loading non-existent snapshot raises error."""
    manager = create_test_manager()

    with pytest.raises(SnapshotNotFoundError) as exc_info:
        manager.load_snapshot("missing-env", "999999999999")

    assert "missing-env" in str(exc_info.value)
    assert "999999999999" in str(exc_info.value)


@mock_aws
def test_load_snapshot_corrupted_json(temp_snapshot_dir, sample_snapshot):
    """Test that JSONDecodeError propagates for corrupted snapshots."""
    manager = create_test_manager()

    # Save valid snapshot first
    manager.save_snapshot("test-env", sample_snapshot)

    # Corrupt the snapshot by writing invalid JSON
    s3 = boto3.client("s3", region_name="us-east-1")
    s3_key = "snapshots/test-env/post-setup/123456789012/baseline.json"
    s3.put_object(Bucket="awsbench-state-123456789012", Key=s3_key, Body=b"corrupted{invalid json")

    # Loading should raise JSONDecodeError, not SnapshotNotFoundError
    with pytest.raises(json.JSONDecodeError):
        manager.load_snapshot("test-env", "123456789012")


@mock_aws
def test_snapshot_exists(temp_snapshot_dir, sample_snapshot):
    """Test checking if snapshot exists."""
    manager = create_test_manager()

    # Before save
    assert not manager.snapshot_exists("test-env", "123456789012")

    # After save
    manager.save_snapshot("test-env", sample_snapshot)
    assert manager.snapshot_exists("test-env", "123456789012")


# ===========================================================================
# capture_snapshot — snapshot capture operations
# ===========================================================================


@mock_aws
def test_capture_snapshot_creates_complete_snapshot(temp_snapshot_dir, mocker):
    """Captures snapshot with all components: drift baseline, metadata, and resources."""
    session = boto3.Session(region_name="us-east-1")
    manager = create_test_manager()

    # Mock create_regional_session to avoid credential checks
    mocker.patch(
        "aws_bench.resource_management.snapshot.manager.create_regional_session",
        return_value=session,
    )

    # Mock CloudFormation client
    mock_cfn = mocker.MagicMock()

    # Mock list_stacks paginator
    mock_list_paginator = mocker.MagicMock()
    mock_list_paginator.paginate.return_value = [
        {
            "StackSummaries": [
                {
                    "StackName": "test-stack-us-east-1",
                    "StackStatus": "CREATE_COMPLETE",
                }
            ]
        }
    ]

    # Mock drift detection
    mock_cfn.detect_stack_drift.return_value = {"StackDriftDetectionId": "detection-123"}

    # Mock paginator for describe_stack_resource_drifts
    mock_drift_paginator = mocker.MagicMock()
    mock_drift_paginator.paginate.return_value = [
        {
            "StackResourceDrifts": [
                {
                    "LogicalResourceId": "MyRole",
                    "StackResourceDriftStatus": "IN_SYNC",
                    "PropertyDifferences": [],
                }
            ]
        }
    ]

    # Mock get_paginator to return appropriate paginators
    def get_paginator_side_effect(operation):
        if operation == "list_stacks":
            return mock_list_paginator
        elif operation == "describe_stack_resource_drifts":
            return mock_drift_paginator
        return mocker.MagicMock()

    mock_cfn.get_paginator.side_effect = get_paginator_side_effect

    # Mock describe_stacks
    mock_cfn.describe_stacks.return_value = {
        "Stacks": [
            {
                "StackName": "test-stack-us-east-1",
                "StackStatus": "CREATE_COMPLETE",
                "Parameters": [{"ParameterKey": "Param1", "ParameterValue": "Value1"}],
                "Tags": [{"Key": "Env", "Value": "test"}],
            }
        ]
    }

    # Mock get_template
    mock_cfn.get_template.return_value = {"TemplateBody": {"Resources": {}}}

    mock_cfn.describe_stack_drift_detection_status.return_value = {
        "DetectionStatus": "DETECTION_COMPLETE"
    }

    # Mock get_stack_resource_drifts
    mocker.patch(
        "aws_bench.resource_management.snapshot.drift.get_stack_resource_drifts",
        return_value=[
            {
                "LogicalResourceId": "MyRole",
                "StackResourceDriftStatus": "IN_SYNC",
                "PropertyDifferences": [],
            }
        ],
    )

    # Mock the fast-scan scanner
    mock_scanner = mocker.MagicMock()
    mock_scan_result = mocker.MagicMock()
    mock_scan_result.detected = {"AWS::IAM::Role": [{"Identifier": "MyRole"}]}
    mock_scanner.scan_resources.return_value = mock_scan_result

    mocker.patch(
        "aws_bench.resource_management.snapshot.manager.get_drift_client",
        return_value=mock_cfn,
    )
    mocker.patch(
        "aws_bench.resource_management.snapshot.manager.make_scanner",
        return_value=mock_scanner,
    )

    # Capture snapshot
    snapshot = manager.capture_snapshot(
        session,
        "123456789012",
        "env-test",
        "abc123",
        "us-east-1",
    )

    # Verify snapshot structure
    assert snapshot.account_id == "123456789012"
    assert snapshot.environment_id == "env-test"
    assert snapshot.scenario_hash == "abc123"
    assert "test-stack-us-east-1" in snapshot.drift_baseline
    assert "test-stack-us-east-1" in snapshot.stack_metadata
    assert "AWS::IAM::Role" in snapshot.resource_ids

    # Verify drift baseline
    drift = snapshot.drift_baseline["test-stack-us-east-1"]
    assert drift.detection_status == "DETECTION_COMPLETE"
    assert len(drift.resource_drifts) == 1
    assert drift.resource_drifts[0].logical_resource_id == "MyRole"

    # Verify stack metadata
    metadata = snapshot.stack_metadata["test-stack-us-east-1"]
    assert metadata.status == "CREATE_COMPLETE"
    assert metadata.parameters == {"Param1": "Value1"}
    assert metadata.tags == {"Env": "test"}

    # Verify the fast-scan was called for this region.
    mock_scanner.scan_resources.assert_called_once_with(region="us-east-1")


@mock_aws
def test_capture_snapshot_skips_deleted_stacks(temp_snapshot_dir, mocker):
    """Skips stacks with DELETE_COMPLETE status."""
    session = boto3.Session(region_name="us-east-1")
    manager = create_test_manager()

    # Mock create_regional_session to avoid credential checks
    mocker.patch(
        "aws_bench.resource_management.snapshot.manager.create_regional_session",
        return_value=session,
    )

    # Mock CloudFormation client
    mock_cfn = mocker.MagicMock()

    # Mock list_stacks with one deleted stack
    mock_list_paginator = mocker.MagicMock()
    mock_list_paginator.paginate.return_value = [
        {
            "StackSummaries": [
                {
                    "StackName": "deleted-stack",
                    "StackStatus": "DELETE_COMPLETE",
                },
                {
                    "StackName": "active-stack",
                    "StackStatus": "CREATE_COMPLETE",
                },
            ]
        }
    ]

    # Mock drift detection for active stack
    mock_cfn.detect_stack_drift.return_value = {"StackDriftDetectionId": "detection-123"}

    # Mock drift paginator
    mock_drift_paginator = mocker.MagicMock()
    mock_drift_paginator.paginate.return_value = [{"StackResourceDrifts": []}]

    def get_paginator_side_effect(operation):
        if operation == "list_stacks":
            return mock_list_paginator
        elif operation == "describe_stack_resource_drifts":
            return mock_drift_paginator
        return mocker.MagicMock()

    mock_cfn.get_paginator.side_effect = get_paginator_side_effect

    mock_cfn.describe_stacks.return_value = {
        "Stacks": [{"StackName": "active-stack", "StackStatus": "CREATE_COMPLETE"}]
    }
    mock_cfn.get_template.return_value = {"TemplateBody": {}}

    mock_cfn.describe_stack_drift_detection_status.return_value = {
        "DetectionStatus": "DETECTION_COMPLETE"
    }

    # Mock get_stack_resource_drifts
    mocker.patch(
        "aws_bench.resource_management.snapshot.drift.get_stack_resource_drifts",
        return_value=[],
    )

    mock_scanner = mocker.MagicMock()
    mock_scanner.scan_resources.return_value = mocker.MagicMock(detected={})

    mocker.patch(
        "aws_bench.resource_management.snapshot.manager.get_drift_client",
        return_value=mock_cfn,
    )
    mocker.patch(
        "aws_bench.resource_management.snapshot.manager.make_scanner",
        return_value=mock_scanner,
    )

    snapshot = manager.capture_snapshot(
        session,
        "123456789012",
        "env-test",
        "abc123",
        "us-east-1",
    )

    # Only active-stack should be in the snapshot
    assert "active-stack" in snapshot.drift_baseline
    assert "deleted-stack" not in snapshot.drift_baseline


@mock_aws
def test_capture_snapshot_skips_nested_stacks(temp_snapshot_dir, mocker):
    """Skips nested stacks (those with ParentId)."""
    session = boto3.Session(region_name="us-east-1")
    manager = create_test_manager()

    # Mock create_regional_session to avoid credential checks
    mocker.patch(
        "aws_bench.resource_management.snapshot.manager.create_regional_session",
        return_value=session,
    )

    mock_cfn = mocker.MagicMock()

    mock_list_paginator = mocker.MagicMock()
    mock_list_paginator.paginate.return_value = [
        {
            "StackSummaries": [
                {
                    "StackName": "parent-stack",
                    "StackStatus": "CREATE_COMPLETE",
                },
                {
                    "StackName": "nested-stack",
                    "StackStatus": "CREATE_COMPLETE",
                    "ParentId": "parent-stack-id",
                },
            ]
        }
    ]

    mock_cfn.detect_stack_drift.return_value = {"StackDriftDetectionId": "detection-123"}

    # Mock drift paginator
    mock_drift_paginator = mocker.MagicMock()
    mock_drift_paginator.paginate.return_value = [{"StackResourceDrifts": []}]

    def get_paginator_side_effect(operation):
        if operation == "list_stacks":
            return mock_list_paginator
        elif operation == "describe_stack_resource_drifts":
            return mock_drift_paginator
        return mocker.MagicMock()

    mock_cfn.get_paginator.side_effect = get_paginator_side_effect

    mock_cfn.describe_stacks.return_value = {
        "Stacks": [{"StackName": "parent-stack", "StackStatus": "CREATE_COMPLETE"}]
    }
    mock_cfn.get_template.return_value = {"TemplateBody": {}}

    mock_cfn.describe_stack_drift_detection_status.return_value = {
        "DetectionStatus": "DETECTION_COMPLETE"
    }

    # Mock get_stack_resource_drifts
    mocker.patch(
        "aws_bench.resource_management.snapshot.drift.get_stack_resource_drifts",
        return_value=[],
    )

    mock_scanner = mocker.MagicMock()
    mock_scanner.scan_resources.return_value = mocker.MagicMock(detected={})

    mocker.patch(
        "aws_bench.resource_management.snapshot.manager.get_drift_client",
        return_value=mock_cfn,
    )
    mocker.patch(
        "aws_bench.resource_management.snapshot.manager.make_scanner",
        return_value=mock_scanner,
    )

    snapshot = manager.capture_snapshot(
        session,
        "123456789012",
        "env-test",
        "abc123",
        "us-east-1",
    )

    # Only parent-stack should be in the snapshot
    assert "parent-stack" in snapshot.drift_baseline
    assert "nested-stack" not in snapshot.drift_baseline


def test_capture_snapshot_passes_account_id_to_make_scanner(mocker):
    """capture_snapshot threads its target account_id into make_scanner.

    This activates the Lambda scan path for the snapshot (make_scanner routes to
    LambdaScanner when given an account_id; the host scan is still the fallback).
    """
    from aws_bench.resource_management.snapshot import manager as m

    made = mocker.patch.object(m, "make_scanner")
    made.return_value.scan_resources.return_value = mocker.MagicMock(
        detected={}, failed={}, empty=set()
    )
    mocker.patch.object(m, "get_drift_client")
    mocker.patch.object(m, "create_regional_session")

    mgr = create_test_manager()
    mocker.patch.object(mgr, "_list_active_stacks", return_value=[])
    mocker.patch.object(mgr, "_capture_drift_baselines", return_value={})
    mocker.patch.object(mgr, "_capture_stack_metadata", return_value={})

    mgr.capture_snapshot(mocker.MagicMock(), "111111111111", "env", "hash", "us-east-1")

    _, kwargs = made.call_args
    assert kwargs.get("account_id") == "111111111111"


# ===========================================================================
# Optimistic Concurrency Control
# ===========================================================================


@mock_aws
def test_save_snapshot_stores_etag(temp_snapshot_dir, sample_snapshot):
    """Stores etag when saving snapshot."""
    manager = create_test_manager()

    manager.save_snapshot("test-env", sample_snapshot)

    etag_key = SnapshotKey("test-env", "123456789012", SnapshotStage.POST_SETUP)
    assert etag_key in manager._etags
    assert isinstance(manager._etags[etag_key], str)
    assert len(manager._etags[etag_key]) == 32  # MD5 hex digest (S3 ETags are MD5)


@mock_aws
def test_load_snapshot_stores_etag(temp_snapshot_dir, sample_snapshot):
    """Stores etag when loading snapshot."""
    manager = create_test_manager()

    manager.save_snapshot("test-env", sample_snapshot)

    manager2 = create_test_manager()
    manager2.load_snapshot("test-env", "123456789012")

    etag_key = SnapshotKey("test-env", "123456789012", SnapshotStage.POST_SETUP)
    assert etag_key in manager2._etags
    assert isinstance(manager2._etags[etag_key], str)
    assert len(manager2._etags[etag_key]) == 32  # MD5 hex digest (S3 ETags are MD5)


@mock_aws
def test_save_snapshot_succeeds_when_no_modification(temp_snapshot_dir, sample_snapshot):
    """Saves successfully when snapshot has not been modified."""
    manager = create_test_manager()

    manager.save_snapshot("test-env", sample_snapshot)
    loaded = manager.load_snapshot("test-env", "123456789012")

    loaded.environment_id = "env-updated"
    manager.save_snapshot("test-env", loaded)

    reloaded = manager.load_snapshot("test-env", "123456789012")
    assert reloaded.environment_id == "env-updated"


@mock_aws
def test_save_snapshot_raises_when_modified_externally(temp_snapshot_dir, sample_snapshot):
    """Raises StorageConflictError when snapshot was modified by another process."""
    manager1 = create_test_manager()
    manager2 = create_test_manager()

    manager1.save_snapshot("test-env", sample_snapshot)

    snapshot1 = manager1.load_snapshot("test-env", "123456789012")
    snapshot2 = manager2.load_snapshot("test-env", "123456789012")

    snapshot1.environment_id = "env-updated-1"
    manager1.save_snapshot("test-env", snapshot1)

    snapshot2.environment_id = "env-updated-2"
    with pytest.raises(StorageConflictError) as exc_info:
        manager2.save_snapshot("test-env", snapshot2)

    assert exc_info.value.key  # Verify it has the key attribute


@mock_aws
def test_save_snapshot_succeeds_for_new_snapshot_without_load(temp_snapshot_dir, sample_snapshot):
    """Saves successfully when creating a new snapshot without prior load."""
    manager = create_test_manager()

    # Should not raise - creating new snapshot without prior load
    manager.save_snapshot("test-env", sample_snapshot)


@mock_aws
def test_save_snapshot_updates_etag_after_save(temp_snapshot_dir, sample_snapshot):
    """Updates etag after successful save."""
    manager = create_test_manager()

    manager.save_snapshot("test-env", sample_snapshot)
    etag_key = SnapshotKey("test-env", "123456789012", SnapshotStage.POST_SETUP)
    first_etag = manager._etags[etag_key]

    loaded = manager.load_snapshot("test-env", "123456789012")
    loaded.environment_id = "env-updated"
    manager.save_snapshot("test-env", loaded)

    second_etag = manager._etags[etag_key]
    assert second_etag != first_etag


# ===========================================================================
# capture_snapshot_multiregion — merge across regions
# ===========================================================================


def _region_snapshot(region, *, stack, resource_id):
    """Build a single-region Snapshot like capture_snapshot returns."""
    return Snapshot(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        account_id="123456789012",
        environment_id="env-test",
        scenario_hash="hash1",
        drift_baseline={
            stack: DriftBaseline(detection_status="DETECTION_COMPLETE", resource_drifts=[])
        },
        stack_metadata={stack: StackMetadata(status="CREATE_COMPLETE", template_hash="sha256:x")},
        resource_ids={"AWS::IAM::Role": [{"Identifier": resource_id}]},
        regions=[region],
        failed_resource_types={},
        empty_resource_types=set(),
    )


@mock_aws
def test_capture_snapshot_multiregion_merges_all_regions(temp_snapshot_dir, mocker):
    """Merges per-region captures into one snapshot covering every region."""
    manager = create_test_manager()

    per_region = {
        "us-east-1": _region_snapshot("us-east-1", stack="stack-e1", resource_id="role-e1"),
        "us-west-1": _region_snapshot("us-west-1", stack="stack-w1", resource_id="role-w1"),
        "us-west-2": _region_snapshot("us-west-2", stack="stack-w2", resource_id="role-w2"),
    }

    def capture_side_effect(_scan_session, _account_id, _environment_id, _scenario_hash, region):
        return per_region[region]

    mocker.patch.object(manager, "capture_snapshot", side_effect=capture_side_effect)

    session = boto3.Session(region_name="us-east-1")
    merged = manager.capture_snapshot_multiregion(
        session,
        "123456789012",
        "env-test",
        "hash1",
        ["us-east-1", "us-west-1", "us-west-2"],
    )

    # All regions present in the merged baseline (the original bug: only the
    # last region survived).
    assert merged.regions == ["us-east-1", "us-west-1", "us-west-2"]
    # Stacks and resources unioned across regions.
    assert set(merged.stack_metadata) == {"stack-e1", "stack-w1", "stack-w2"}
    ids = {r["Identifier"] for r in merged.resource_ids["AWS::IAM::Role"]}
    assert ids == {"role-e1", "role-w1", "role-w2"}


@mock_aws
def test_capture_snapshot_multiregion_single_region(temp_snapshot_dir, mocker):
    """A single region yields that region's snapshot unchanged."""
    manager = create_test_manager()

    snap = _region_snapshot("us-east-1", stack="stack-e1", resource_id="role-e1")
    mocker.patch.object(manager, "capture_snapshot", return_value=snap)

    session = boto3.Session(region_name="us-east-1")
    merged = manager.capture_snapshot_multiregion(
        session,
        "123456789012",
        "env-test",
        "hash1",
        ["us-east-1"],
    )

    assert merged.regions == ["us-east-1"]


@mock_aws
def test_capture_snapshot_multiregion_requires_regions(temp_snapshot_dir):
    """Empty region list is rejected."""
    manager = create_test_manager()

    session = boto3.Session(region_name="us-east-1")
    with pytest.raises(ValueError, match="regions must be non-empty"):
        manager.capture_snapshot_multiregion(
            session,
            "123456789012",
            "env-test",
            "hash1",
            [],
        )


@patch("aws_bench.resource_management.snapshot.manager.detect_stacks_drift")
def test_capture_drift_baselines_raises_on_detection_failed(mock_detect):
    """A stack still DETECTION_FAILED after retries aborts capture.

    Saving a baseline missing a stack's true drift state would read as a false
    drift mismatch on a later verify, so capture fails instead.
    """
    manager = create_test_manager()

    mock_detect.return_value = {
        "good-stack": DriftBaseline(detection_status="DETECTION_COMPLETE", resource_drifts=[]),
        "failed-stack": DriftBaseline(detection_status="DETECTION_FAILED", resource_drifts=[]),
    }

    stacks = [
        {"StackName": "good-stack", "StackStatus": "CREATE_COMPLETE"},
        {"StackName": "failed-stack", "StackStatus": "CREATE_COMPLETE"},
    ]

    with pytest.raises(DriftDetectionError, match="failed-stack"):
        manager._capture_drift_baselines(MagicMock(), stacks)


@patch("aws_bench.resource_management.snapshot.manager.detect_stacks_drift")
def test_capture_drift_baselines_returns_when_all_measured(mock_detect):
    """When every stack is COMPLETE or SKIPPED, the baseline is returned intact."""
    manager = create_test_manager()

    mock_detect.return_value = {
        "good-stack": DriftBaseline(detection_status="DETECTION_COMPLETE", resource_drifts=[]),
        "skipped-stack": DriftBaseline(
            detection_status="SKIPPED_ROLLBACK_COMPLETE", resource_drifts=[]
        ),
    }

    stacks = [
        {"StackName": "good-stack", "StackStatus": "CREATE_COMPLETE"},
        {"StackName": "skipped-stack", "StackStatus": "ROLLBACK_COMPLETE"},
    ]

    baseline = manager._capture_drift_baselines(MagicMock(), stacks)

    assert set(baseline) == {"good-stack", "skipped-stack"}


# ===========================================================================
# snapshot_account — OBSERVABILITY local-file capture (no S3)
# ===========================================================================


@mock_aws
def test_snapshot_account_writes_local_file_and_skips_s3(temp_snapshot_dir, mocker):
    """With ctx.output_dir set, snapshot goes to a local JSON file, never to S3."""
    manager = create_test_manager()

    captured = _region_snapshot("us-east-1", stack="stack-e1", resource_id="role-e1")
    mocker.patch.object(manager, "capture_snapshot_multiregion", return_value=captured)
    save_spy = mocker.patch.object(manager, "save_snapshot")

    ctx = SnapshotContext(
        scenario_id="my-scenario",
        scenario_hash="",
        regions=["us-east-1"],
        stage=SnapshotStage.OBSERVABILITY,
        account_ids=["123456789012"],
        output_dir=temp_snapshot_dir,
    )

    session = boto3.Session(region_name="us-east-1")
    result = manager.snapshot_account(session, "123456789012", ctx)

    expected_path = temp_snapshot_dir / "my-scenario" / "123456789012.json"
    assert result.success is True
    assert result.output_path == str(expected_path)
    assert result.regions_captured == ["us-east-1"]
    # S3 save must NOT be called for an OBSERVABILITY (output_dir) capture.
    save_spy.assert_not_called()
    # The file exists and holds the captured snapshot.
    assert expected_path.exists()
    data = json.loads(expected_path.read_text())
    assert data["account_id"] == "123456789012"
    assert data["environment_id"] == "env-test"


# ===========================================================================
# Fresh-account subscription retry — _list_active_stacks self-heals OptInRequired
#
# _list_active_stacks is the snapshot's first CloudFormation call, so it is the
# one place an unconverged subscription surfaces (OptInRequired) on a fresh
# account; the retry lives there, not on the whole capture_snapshot orchestration.
# ===========================================================================


def _client_error(code: str, operation: str = "ListStacks") -> ClientError:
    """Build a botocore ClientError carrying ``code`` in Error.Code."""
    return ClientError(
        {"Error": {"Code": code, "Message": "needs a subscription for the service"}},
        operation,
    )


@pytest.fixture
def _instant_stacks_retry(mocker):
    """Neutralize _list_active_stacks's backoff so retry tests don't actually sleep."""
    # tenacity attaches the .retry controller at runtime; the stubs don't model it.
    mocker.patch.object(
        SnapshotManager._list_active_stacks.retry,  # type: ignore[attr-defined]
        "wait",
        tenacity.wait_none(),
    )


def _cfn_with_paginate(mocker, *, paginate_side_effect):
    """A CFN mock whose list_stacks paginator applies ``paginate_side_effect``.

    Returns the paginator so a test can assert attempt counts: _list_active_stacks
    calls ``paginate`` once per attempt. A list entry that is an exception raises;
    a value is returned.
    """
    mock_cfn = mocker.MagicMock()
    paginator = mocker.MagicMock()
    paginator.paginate.side_effect = paginate_side_effect
    mock_cfn.get_paginator.return_value = paginator
    return mock_cfn, paginator


def test_list_active_stacks_retries_then_succeeds_on_optin_required(_instant_stacks_retry, mocker):
    """A fresh-account OptInRequired on the first calls clears and the listing succeeds."""
    manager = create_test_manager()

    # First two ListStacks calls fail with OptInRequired, third returns no stacks.
    mock_cfn, paginator = _cfn_with_paginate(
        mocker,
        paginate_side_effect=[
            _client_error("OptInRequired"),
            _client_error("OptInRequired"),
            [{"StackSummaries": []}],
        ],
    )

    stacks = manager._list_active_stacks(mock_cfn)

    assert stacks == []
    assert paginator.paginate.call_count == 3


def test_list_active_stacks_does_not_retry_non_transient_error(_instant_stacks_retry, mocker):
    """A non-subscription ClientError (AccessDenied) surfaces on the first attempt."""
    manager = create_test_manager()

    mock_cfn, paginator = _cfn_with_paginate(
        mocker, paginate_side_effect=_client_error("AccessDenied")
    )

    with pytest.raises(ClientError) as exc_info:
        manager._list_active_stacks(mock_cfn)

    assert exc_info.value.response["Error"]["Code"] == "AccessDenied"
    # No retry: exactly one attempt.
    assert paginator.paginate.call_count == 1


def test_list_active_stacks_reraises_original_error_after_budget(mocker):
    """When the transient error never clears, the budget stops retry and reraises.

    Overrides stop with stop_after_attempt so the budget is deterministic, and
    asserts the ORIGINAL OptInRequired ClientError surfaces (reraise=True), not a
    tenacity RetryError — so snapshot_account records the real cause.
    """
    manager = create_test_manager()

    # tenacity attaches the .retry controller at runtime; the stubs don't model it.
    controller = SnapshotManager._list_active_stacks.retry  # type: ignore[attr-defined]
    mocker.patch.object(controller, "wait", tenacity.wait_none())
    mocker.patch.object(controller, "stop", tenacity.stop_after_attempt(3))

    mock_cfn, paginator = _cfn_with_paginate(
        mocker, paginate_side_effect=_client_error("OptInRequired")
    )

    with pytest.raises(ClientError) as exc_info:
        manager._list_active_stacks(mock_cfn)

    assert exc_info.value.response["Error"]["Code"] == "OptInRequired"
    assert paginator.paginate.call_count == 3


# -- Snapshot fails on transient server errors -----------------------------------


def test_capture_snapshot_aborts_on_transient_server_errors(temp_snapshot_dir, mocker):
    """capture_snapshot aborts when a lister failed with a transient server error.

    A snapshot missing resources to a throttle/5xx would drive wrong verify/reset/cleanup calls.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = create_test_manager()

    mocker.patch(
        "aws_bench.resource_management.snapshot.manager.create_regional_session",
        return_value=session,
    )
    mock_cfn = mocker.MagicMock()
    mock_list_paginator = mocker.MagicMock()
    mock_list_paginator.paginate.return_value = [{"StackSummaries": []}]
    mock_cfn.get_paginator.return_value = mock_list_paginator
    mocker.patch(
        "aws_bench.resource_management.snapshot.manager.get_drift_client",
        return_value=mock_cfn,
    )

    # One lister failed with a transient (throttle) error string — must abort the snapshot.
    mock_scanner = mocker.MagicMock()
    mock_scanner.scan_resources.return_value = mocker.MagicMock(
        detected={}, failed={"ec2:DescribeInstances": "ThrottlingException: rate exceeded"}
    )
    mocker.patch(
        "aws_bench.resource_management.snapshot.manager.make_scanner",
        return_value=mock_scanner,
    )

    with pytest.raises(RuntimeError, match="transient server errors"):
        manager.capture_snapshot(session, "123456789012", "env-test", "abc123", "us-east-1")


def test_capture_snapshot_does_not_abort_on_non_transient_failure(temp_snapshot_dir, mocker):
    """A permanent lister failure (e.g. AccessDenied) does NOT abort — it rides through as failed.

    Broadening the abort to any failure would spuriously fail every run over a lone AccessDenied.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = create_test_manager()

    mocker.patch(
        "aws_bench.resource_management.snapshot.manager.create_regional_session",
        return_value=session,
    )
    mock_cfn = mocker.MagicMock()
    mock_list_paginator = mocker.MagicMock()
    mock_list_paginator.paginate.return_value = [{"StackSummaries": []}]
    mock_cfn.get_paginator.return_value = mock_list_paginator
    mocker.patch(
        "aws_bench.resource_management.snapshot.manager.get_drift_client",
        return_value=mock_cfn,
    )

    failed = {"iam:ListRoles": "AccessDenied: not authorized"}
    mock_scanner = mocker.MagicMock()
    mock_scanner.scan_resources.return_value = mocker.MagicMock(
        detected={}, failed=failed, empty=set()
    )
    mocker.patch(
        "aws_bench.resource_management.snapshot.manager.make_scanner",
        return_value=mock_scanner,
    )

    snapshot = manager.capture_snapshot(session, "123456789012", "env-test", "abc123", "us-east-1")

    assert snapshot.failed_resource_types == failed
