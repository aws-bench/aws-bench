from datetime import datetime, timezone

from aws_bench.resource_management.exceptions import SnapshotNotFoundError
from aws_bench.resource_management.snapshot.models import (
    DriftBaseline,
    ResourceDrift,
    Snapshot,
    SnapshotStage,
    StackMetadata,
)


def test_resource_drift_creation():
    """Test ResourceDrift dataclass creation."""
    drift = ResourceDrift(
        logical_resource_id="MySecurityGroup",
        stack_resource_drift_status="MODIFIED",
        property_differences=[{"PropertyPath": "/IpPermissions", "Expected": "[]"}],
    )

    assert drift.logical_resource_id == "MySecurityGroup"
    assert drift.stack_resource_drift_status == "MODIFIED"
    assert len(drift.property_differences) == 1


def test_drift_baseline_creation():
    """Test DriftBaseline dataclass creation."""
    drift = ResourceDrift(
        logical_resource_id="MyRole",
        stack_resource_drift_status="IN_SYNC",
        property_differences=[],
    )

    baseline = DriftBaseline(detection_status="DETECTION_COMPLETE", resource_drifts=[drift])

    assert baseline.detection_status == "DETECTION_COMPLETE"
    assert len(baseline.resource_drifts) == 1
    assert baseline.resource_drifts[0].logical_resource_id == "MyRole"


def test_stack_metadata_creation():
    """Test StackMetadata dataclass creation."""
    metadata = StackMetadata(
        status="CREATE_COMPLETE",
        template_hash="sha256:abc123",
        parameters={"Param1": "Value1"},
        tags={"Environment": "test"},
    )

    assert metadata.status == "CREATE_COMPLETE"
    assert metadata.template_hash == "sha256:abc123"
    assert metadata.parameters["Param1"] == "Value1"
    assert metadata.tags["Environment"] == "test"


def test_snapshot_creation():
    """Test Snapshot dataclass creation."""
    timestamp = datetime.now(timezone.utc)

    snapshot = Snapshot(
        timestamp=timestamp,
        account_id="123456789012",
        environment_id="env-basic-ec2",
        scenario_hash="abc123def456",
        drift_baseline={"stack1": DriftBaseline("DETECTION_COMPLETE", [])},
        stack_metadata={"stack1": StackMetadata("CREATE_COMPLETE", "sha256:xyz", {}, {})},
        resource_ids={"AWS::IAM::Role": [{"Identifier": "MyRole"}]},
    )

    assert snapshot.account_id == "123456789012"
    assert snapshot.environment_id == "env-basic-ec2"
    assert snapshot.scenario_hash == "abc123def456"
    assert "stack1" in snapshot.drift_baseline
    assert "stack1" in snapshot.stack_metadata
    assert "AWS::IAM::Role" in snapshot.resource_ids


def test_snapshot_not_found_error():
    """Test SnapshotNotFoundError exception."""
    error = SnapshotNotFoundError("env-test", "123456789012")

    assert "env-test" in str(error)
    assert "123456789012" in str(error)


def test_snapshot_not_found_error_names_init_for_pre_setup():
    """The remedy for a missing PRE_SETUP baseline is env init, not env setup."""
    error = SnapshotNotFoundError("env-test", "123456789012", SnapshotStage.PRE_SETUP)

    assert "aws-bench env init" in str(error)
