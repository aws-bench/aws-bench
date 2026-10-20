"""Spike test for the stacks-first reset deletion ordering (design Section A).

Encodes the DESIRED behavior before `_reset_region` is reordered:

- A new ``AWS::CloudFormation::Stack`` in ``new_resources`` must be deleted via the
  reset's existing whole-stack path (``StackRestorer._delete_for_resetup``, which
  waits for terminal deletion), NOT via the individual CCAPI sweep.
- That stack entry must be stripped from the set handed to the individual sweep
  (``ResourceCleaner.cleanup``), so the sweep never tries to CCAPI-delete a stack
  that CloudFormation is cascading.

Before the stacks-first reorder these assertions failed (the reset deleted the
stack entry individually via the sweep and never routed it to
``_delete_for_resetup``). The reorder in ``_reset_region`` closes that gap.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from aws_bench.resource_management.reset.manager import ResetManager
from aws_bench.resource_management.reset.models import ResetupDeletion, RestoreOutcome
from aws_bench.resource_management.snapshot.models import (
    DriftBaseline,
    ResourceDrift,
    Snapshot,
    StackMetadata,
)
from aws_bench.resource_management.verify.models import VerifyResult


@pytest.fixture
def temp_output_dir(tmp_path):
    return tmp_path / "output"


@pytest.fixture
def sample_snapshot():
    """A baseline snapshot whose only baseline resource is an IAM role (no stacks)."""
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


@mock_aws
def test_new_stack_deleted_via_stack_path_not_individual_sweep(temp_output_dir, sample_snapshot):
    """A new CFN stack is routed to _delete_for_resetup and excluded from the sweep."""
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    # Census reports a new agent-created stack PLUS a standalone resource. Under the
    # stacks-first design the stack is deleted (cascade) and the bucket is swept.
    new_resources = {
        "AWS::CloudFormation::Stack": [{"Identifier": "agent-created-stack"}],
        "AWS::S3::Bucket": [{"Identifier": "standalone-bucket"}],
    }

    swept_types: list[str] = []

    async def _cleanup(stack_resources, *args, **kwargs):
        swept_types.extend(r.resource_type for r in stack_resources)
        return {}

    deleted_via_stack_path: list[str] = []

    async def _delete_for_resetup(self, stack_name):  # noqa: ANN001
        deleted_via_stack_path.append(stack_name)
        return ResetupDeletion(RestoreOutcome.DELETED_NEEDS_REDEPLOY)

    with (
        patch(
            "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot",
            return_value=sample_snapshot,
        ),
        patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state",
            side_effect=[
                VerifyResult(
                    success=False,
                    reason="Found new resources incl. a stack",
                    new_resources=new_resources,
                ),
                VerifyResult(success=True, reason="Account in baseline state"),
            ],
        ),
        patch("aws_bench.resource_management.reset.manager.ResourceCleaner") as mock_cleaner_cls,
        patch(
            "aws_bench.resource_management.reset.stack_restorer.StackRestorer._delete_for_resetup",
            _delete_for_resetup,
        ),
        # A stack was deleted -> _reset_region runs the orphan-census backstop
        # instead of the final verify; None means "clean, nothing abandoned".
        patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.find_orphan_resources",
            return_value=None,
        ),
        patch("aws_bench.resource_management.reset.manager.CloudControlManager") as mock_ccm,
    ):
        mock_cleaner_cls.return_value.cleanup = _cleanup
        mock_ccm.return_value.resource_exists.return_value = False
        result = asyncio.run(manager.reset_account("test-env", "123456789012"))

    assert result.success
    # The new stack was deleted via the whole-stack path...
    assert "agent-created-stack" in deleted_via_stack_path
    # ...and was NOT handed to the individual CCAPI sweep.
    assert "AWS::CloudFormation::Stack" not in swept_types
    # The standalone resource still goes through the sweep.
    assert "AWS::S3::Bucket" in swept_types


def test_is_service_managed_studio_stack_predicate():
    """environment-*-flink-studio stacks (name or ARN) are service-managed; others aren't."""
    from aws_bench.resource_management.cleanup.models import is_service_managed_studio_stack

    assert is_service_managed_studio_stack("environment-abc123-flink-studio")
    assert is_service_managed_studio_stack("environment-abc123-flink-studio-notebook")
    assert is_service_managed_studio_stack(
        "arn:aws:cloudformation:us-east-1:123456789012:stack/environment-x-flink-studio/gg-uuid"
    )
    assert not is_service_managed_studio_stack("agent-created-stack")
    assert not is_service_managed_studio_stack("environment-abc-something-else")
    assert not is_service_managed_studio_stack(
        "arn:aws:cloudformation:us-east-1:123456789012:stack/my-normal-stack/gg-uuid"
    )


@mock_aws
def test_service_managed_studio_stack_not_whole_deleted(temp_output_dir, sample_snapshot):
    """A new environment-*-flink-studio stack is excluded from the stacks-first delete.

    Deleting it whole goes DELETE_FAILED (AWS removes it only when the owning KDA
    app is deleted), so it must NOT be routed to _delete_for_resetup here.
    """
    session = boto3.Session(region_name="us-east-1")
    manager = ResetManager(session, output_dir=temp_output_dir)

    new_resources = {
        "AWS::CloudFormation::Stack": [{"Identifier": "environment-xyz-flink-studio"}],
        "AWS::S3::Bucket": [{"Identifier": "standalone-bucket"}],
    }

    async def _cleanup(stack_resources, *args, **kwargs):
        return {}

    deleted_via_stack_path: list[str] = []

    async def _delete_for_resetup(self, stack_name):  # noqa: ANN001
        deleted_via_stack_path.append(stack_name)
        return ResetupDeletion(RestoreOutcome.DELETED_NEEDS_REDEPLOY)

    with (
        patch(
            "aws_bench.resource_management.verify.manager.SnapshotManager.load_snapshot",
            return_value=sample_snapshot,
        ),
        patch(
            "aws_bench.resource_management.verify.manager.VerifyManager.verify_account_state",
            side_effect=[
                VerifyResult(
                    success=False,
                    reason="Found a studio stack",
                    new_resources=new_resources,
                ),
                VerifyResult(success=True, reason="Account in baseline state"),
            ],
        ),
        patch("aws_bench.resource_management.reset.manager.ResourceCleaner") as mock_cleaner_cls,
        patch(
            "aws_bench.resource_management.reset.stack_restorer.StackRestorer._delete_for_resetup",
            _delete_for_resetup,
        ),
        patch("aws_bench.resource_management.reset.manager.CloudControlManager") as mock_ccm,
    ):
        mock_cleaner_cls.return_value.cleanup = _cleanup
        mock_ccm.return_value.resource_exists.return_value = False
        asyncio.run(manager.reset_account("test-env", "123456789012"))

    # The Studio stack was NOT whole-deleted (excluded from the stacks-first pass).
    assert "environment-xyz-flink-studio" not in deleted_via_stack_path
