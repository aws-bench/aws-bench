"""Tests for aws_bench.resource_management.cleanup.stack_deleter."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from aws_bench.resource_management.ccapi.manager import Resource
from aws_bench.resource_management.ccapi.models import DeletionFailureEvent
from aws_bench.resource_management.cleanup.handlers.cross_service import (
    EniReapResult,
    IpamPoolReapResult,
    VpcPublicAddressWedgeResult,
)
from aws_bench.resource_management.cleanup.models import (
    ExistenceStatus,
    ResourceVerificationResult,
    StackDeletionResult,
    StackDeletionStatus,
    StackResource,
)
from aws_bench.resource_management.cleanup.stack_deleter import StackDeleter
from aws_bench.resource_management.deferred import deferred_scope, is_deferred


@pytest.fixture()
def deleter(tmp_path):
    session = MagicMock()
    session.region_name = "us-east-1"
    session.client.return_value = MagicMock()
    return StackDeleter(session, manifest_path=tmp_path / "manifest.json")


# -- list_stacks --


def test_list_stacks_returns_active_stacks(deleter):
    deleter._client.get_paginator.return_value.paginate.return_value = [
        {
            "StackSummaries": [
                {"StackName": "app-stack", "StackStatus": "CREATE_COMPLETE"},
                {"StackName": "deleted-stack", "StackStatus": "DELETE_COMPLETE"},
                {"StackName": "nested", "StackStatus": "CREATE_COMPLETE", "ParentId": "parent"},
            ]
        }
    ]
    stacks = deleter.list_stacks()
    assert len(stacks) == 1
    assert stacks[0]["StackName"] == "app-stack"


def test_list_stacks_raises_on_client_error(deleter):
    deleter._client.get_paginator.return_value.paginate.side_effect = ClientError(
        {"Error": {"Code": "ValidationError", "Message": "fail"}}, "ListStacks"
    )
    with pytest.raises(ClientError):
        deleter.list_stacks()


# -- get_stack_resources --


def test_get_stack_resources_returns_resources(deleter):
    deleter._client.get_paginator.return_value.paginate.return_value = [
        {
            "StackResourceSummaries": [
                {
                    "LogicalResourceId": "Bucket",
                    "PhysicalResourceId": "my-bucket",
                    "ResourceType": "AWS::S3::Bucket",
                    "ResourceStatus": "CREATE_COMPLETE",
                },
            ]
        }
    ]
    resources = deleter.get_stack_resources("my-stack")
    assert len(resources) == 1
    assert resources[0].logical_id == "Bucket"
    assert resources[0].physical_id == "my-bucket"


def test_get_stack_resources_returns_empty_for_deleted_stack(deleter):
    deleter._client.get_paginator.return_value.paginate.side_effect = ClientError(
        {"Error": {"Code": "ValidationError", "Message": "Stack does not exist"}},
        "ListStackResources",
    )
    resources = deleter.get_stack_resources("gone-stack")
    assert resources == []


def test_get_stack_resources_raises_on_non_existence_error(deleter):
    deleter._client.get_paginator.return_value.paginate.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "forbidden"}}, "ListStackResources"
    )
    with pytest.raises(ClientError):
        deleter.get_stack_resources("my-stack")


# -- _stack_exists --


def test_stack_exists_returns_true_for_active_stack(deleter):
    deleter._client.describe_stacks.return_value = {"Stacks": [{"StackStatus": "CREATE_COMPLETE"}]}
    assert deleter._stack_exists("my-stack") is True


def test_stack_exists_returns_false_for_deleted_stack(deleter):
    deleter._client.describe_stacks.return_value = {"Stacks": [{"StackStatus": "DELETE_COMPLETE"}]}
    assert deleter._stack_exists("my-stack") is False


def test_stack_exists_returns_false_when_not_found(deleter):
    deleter._client.describe_stacks.side_effect = ClientError(
        {"Error": {"Code": "ValidationError", "Message": "does not exist"}}, "DescribeStacks"
    )
    assert deleter._stack_exists("my-stack") is False


def test_stack_exists_reraises_non_existence_error(deleter):
    deleter._client.describe_stacks.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "forbidden"}}, "DescribeStacks"
    )
    with pytest.raises(ClientError):
        deleter._stack_exists("my-stack")


# -- _try_delete_stack --


def test_try_delete_stack_success(deleter):
    deleter._client.describe_stacks.side_effect = ClientError(
        {"Error": {"Code": "ValidationError", "Message": "does not exist"}}, "DescribeStacks"
    )
    with patch.object(deleter._cleaner, "cleanup", new_callable=AsyncMock, return_value={}):
        status = asyncio.run(deleter._try_delete_stack("my-stack", []))
    assert status == "DELETE_COMPLETE"
    deleter._client.delete_stack.assert_called_once_with(StackName="my-stack")


def test_try_delete_stack_returns_failed_on_client_error(deleter):
    deleter._client.delete_stack.side_effect = ClientError(
        {"Error": {"Code": "ValidationError", "Message": "fail"}}, "DeleteStack"
    )
    with patch.object(deleter._cleaner, "cleanup", new_callable=AsyncMock, return_value={}):
        status = asyncio.run(deleter._try_delete_stack("my-stack", []))
    assert status == "DELETE_FAILED"


def test_try_delete_stack_skips_stuck_cleanup_by_default(deleter):
    deleter._client.describe_stacks.side_effect = ClientError(
        {"Error": {"Code": "ValidationError", "Message": "does not exist"}}, "DescribeStacks"
    )
    with patch.object(deleter._cleaner, "cleanup", new_callable=AsyncMock) as mock_cleanup:
        asyncio.run(deleter._try_delete_stack("my-stack", []))
    assert mock_cleanup.call_count == 1
    _, kwargs = mock_cleanup.call_args_list[0]
    assert kwargs.get("prepare") is True
    assert kwargs.get("handle_stuck", False) is False


# -- _disable_termination_protection --


def test_disable_termination_protection_when_enabled(deleter):
    deleter._client.describe_stacks.return_value = {
        "Stacks": [{"EnableTerminationProtection": True}]
    }
    deleter._disable_termination_protection("my-stack")
    deleter._client.update_termination_protection.assert_called_once_with(
        EnableTerminationProtection=False, StackName="my-stack"
    )


def test_disable_termination_protection_noop_when_disabled(deleter):
    deleter._client.describe_stacks.return_value = {
        "Stacks": [{"EnableTerminationProtection": False}]
    }
    deleter._disable_termination_protection("my-stack")
    deleter._client.update_termination_protection.assert_not_called()


def test_disable_termination_protection_handles_error(deleter):
    deleter._client.describe_stacks.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "forbidden"}}, "DescribeStacks"
    )
    deleter._disable_termination_protection("my-stack")


# -- _get_failure_reason --


def test_get_failure_reason_returns_reason(deleter):
    deleter._client.describe_stacks.return_value = {
        "Stacks": [{"StackStatusReason": "Resource failed"}]
    }
    reason = deleter._get_failure_reason("my-stack")
    assert reason == "Resource failed"


def test_get_failure_reason_returns_unknown_on_error(deleter):
    deleter._client.describe_stacks.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "forbidden"}}, "DescribeStacks"
    )
    reason = deleter._get_failure_reason("my-stack")
    assert reason == "unknown"


def test_get_failure_reason_returns_unknown_when_no_reason(deleter):
    deleter._client.describe_stacks.return_value = {"Stacks": [{}]}
    reason = deleter._get_failure_reason("my-stack")
    assert reason == "unknown"


# -- _get_stack_status --


def test_get_stack_status_returns_status_and_reason(deleter):
    deleter._client.describe_stacks.return_value = {
        "Stacks": [{"StackStatus": "DELETE_IN_PROGRESS", "StackStatusReason": "Deleting"}]
    }
    status, reason = asyncio.run(deleter._get_stack_status("my-stack"))
    assert status == "DELETE_IN_PROGRESS"
    assert reason == "Deleting"


def test_get_stack_status_returns_complete_when_not_found(deleter):
    deleter._client.describe_stacks.side_effect = ClientError(
        {"Error": {"Code": "ValidationError", "Message": "does not exist"}}, "DescribeStacks"
    )
    status, reason = asyncio.run(deleter._get_stack_status("my-stack"))
    assert status == "DELETE_COMPLETE"
    assert reason == ""


def test_get_stack_status_returns_complete_when_empty_stacks(deleter):
    deleter._client.describe_stacks.return_value = {"Stacks": []}
    status, reason = asyncio.run(deleter._get_stack_status("my-stack"))
    assert status == "DELETE_COMPLETE"
    assert reason == ""


def test_get_stack_status_reraises_non_existence_error(deleter):
    deleter._client.describe_stacks.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "forbidden"}}, "DescribeStacks"
    )
    with pytest.raises(ClientError):
        asyncio.run(deleter._get_stack_status("my-stack"))


# -- _poll_deletion --


def test_poll_deletion_returns_complete(deleter):
    with patch.object(
        deleter, "_get_stack_status", new_callable=AsyncMock, return_value=("DELETE_COMPLETE", "")
    ):
        status = asyncio.run(deleter._poll_deletion("my-stack", timeout=5))
    assert status == "DELETE_COMPLETE"


def test_poll_deletion_returns_failed(deleter):
    with patch.object(
        deleter,
        "_get_stack_status",
        new_callable=AsyncMock,
        return_value=("DELETE_FAILED", "reason"),
    ):
        status = asyncio.run(deleter._poll_deletion("my-stack", timeout=5))
    assert status == "DELETE_FAILED"


def test_poll_deletion_returns_export_blocked_on_complete_state(deleter):
    with patch.object(
        deleter, "_get_stack_status", new_callable=AsyncMock, return_value=("UPDATE_COMPLETE", "")
    ):
        status = asyncio.run(deleter._poll_deletion("my-stack", timeout=5))
    assert status == "EXPORT_BLOCKED"


def test_poll_deletion_times_out(deleter):
    with patch.object(
        deleter,
        "_get_stack_status",
        new_callable=AsyncMock,
        return_value=("DELETE_IN_PROGRESS", ""),
    ):
        status = asyncio.run(deleter._poll_deletion("my-stack", timeout=0))
    assert status == "DELETE_FAILED"


def test_poll_deletion_loops_until_complete(deleter):
    """Test that polling loop executes and increases interval."""
    mock_status = AsyncMock()
    mock_status.side_effect = [
        ("DELETE_IN_PROGRESS", ""),
        ("DELETE_IN_PROGRESS", ""),
        ("DELETE_COMPLETE", ""),
    ]
    with (
        patch.object(deleter, "_get_stack_status", mock_status),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        status = asyncio.run(deleter._poll_deletion("my-stack", timeout=100))
    assert status == "DELETE_COMPLETE"
    assert mock_status.call_count == 3


# -- _build_result --


def test_build_result_success(deleter):
    resources = [
        StackResource(
            logical_id="Bucket",
            physical_id="my-bucket",
            resource_type="AWS::S3::Bucket",
            status="CREATE_COMPLETE",
        )
    ]
    result = deleter._build_result("my-stack", "DELETE_COMPLETE", resources)
    assert result.stack_name == "my-stack"
    assert result.status == StackDeletionStatus.SUCCESS
    assert result.reason == ""
    assert result.resources == resources


def test_build_result_failed(deleter):
    deleter._client.describe_stacks.return_value = {"Stacks": [{"StackStatusReason": "Failed"}]}
    resources = []
    result = deleter._build_result("my-stack", "DELETE_FAILED", resources)
    assert result.stack_name == "my-stack"
    assert result.status == StackDeletionStatus.FAILED
    assert result.reason == "Failed"


# -- delete_all_stacks --


def test_delete_all_stacks_deletes_env_then_infra(deleter):
    deleter._client.get_paginator.return_value.paginate.return_value = [
        {
            "StackSummaries": [
                {"StackName": "app-stack", "StackStatus": "CREATE_COMPLETE"},
                {"StackName": "CDKToolkit", "StackStatus": "CREATE_COMPLETE"},
            ]
        }
    ]
    result = StackDeletionResult("s", StackDeletionStatus.SUCCESS)
    with patch.object(
        deleter, "delete_stack", new_callable=AsyncMock, return_value=result
    ) as mock_delete:
        summary = asyncio.run(deleter.delete_all_stacks())
    assert mock_delete.call_count == 2
    assert summary.results[0].stack_name == "s"


def test_delete_all_stacks_handles_exception_in_gather(deleter):
    deleter._client.get_paginator.return_value.paginate.return_value = [
        {
            "StackSummaries": [
                {"StackName": "app-stack", "StackStatus": "CREATE_COMPLETE"},
            ]
        }
    ]

    async def raise_error(name):
        raise RuntimeError("Unexpected error")

    with patch.object(deleter, "delete_stack", side_effect=raise_error):
        summary = asyncio.run(deleter.delete_all_stacks())
    failed = summary.results[0]
    assert failed.status == StackDeletionStatus.FAILED
    assert "Unexpected error" in failed.reason


def test_delete_all_stacks_skips_infra_on_env_failure(deleter):
    deleter._client.get_paginator.return_value.paginate.return_value = [
        {
            "StackSummaries": [
                {"StackName": "app-stack", "StackStatus": "CREATE_COMPLETE"},
                {"StackName": "CDKToolkit", "StackStatus": "CREATE_COMPLETE"},
            ]
        }
    ]
    call_count = [0]

    def describe_side_effect(**kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:
            return {
                "Stacks": [{"StackStatus": "CREATE_COMPLETE", "EnableTerminationProtection": False}]
            }
        return {"Stacks": [{"StackStatus": "DELETE_FAILED", "StackStatusReason": "stuck"}]}

    deleter._client.describe_stacks.side_effect = describe_side_effect
    with (
        patch.object(deleter._cleaner, "cleanup", new_callable=AsyncMock, return_value={}),
    ):
        deleter._verifier.verify_resources = AsyncMock(return_value=[])
        summary = asyncio.run(deleter.delete_all_stacks())
    assert any(r.stack_name == "app-stack" for r in summary.results)
    assert not any(r.stack_name == "CDKToolkit" for r in summary.results)


def test_maybe_delete_infra_deferred_stack_blocks_cdktoolkit(deleter):
    """A still-present DEFERRED stack must block CDKToolkit deletion.

    Regression guard: a deferred CDK stack still needs the cfn-exec-role (in the
    CDKToolkit stack) for its eventual DeleteStack retry. Deleting CDKToolkit now
    would strand it — CloudFormation could no longer assume the role to finish it.
    """
    deferred = StackDeletionResult("app", StackDeletionStatus.FAILED, deferred=True)
    infra = [{"StackName": "CDKToolkit"}]
    delete_fn = AsyncMock()
    with patch.object(deleter, "_stack_exists", return_value=True):
        result = asyncio.run(deleter._maybe_delete_infra(infra, [deferred], delete_fn))
    assert result == []
    delete_fn.assert_not_called()


def test_maybe_delete_infra_runs_once_deferred_stack_is_gone(deleter):
    """Once the deferred stack is truly gone, CDKToolkit is deleted."""
    deferred = StackDeletionResult("app", StackDeletionStatus.FAILED, deferred=True)
    infra = [{"StackName": "CDKToolkit"}]

    async def _delete_fn(name):
        return StackDeletionResult(name, StackDeletionStatus.SUCCESS)

    with patch.object(deleter, "_stack_exists", return_value=False):
        result = asyncio.run(deleter._maybe_delete_infra(infra, [deferred], _delete_fn))
    assert [r.stack_name for r in result] == ["CDKToolkit"]


# -- _verify_deletion --


def test_verify_deletion_noop_no_resources(deleter):
    result = StackDeletionResult(stack_name="s", status=StackDeletionStatus.SUCCESS, resources=[])
    asyncio.run(deleter._verify_deletion(result))


def test_verify_deletion_no_surviving(deleter):
    r = StackResource(
        logical_id="A",
        physical_id="p1",
        resource_type="AWS::S3::Bucket",
        status="CREATE_COMPLETE",
    )
    result = StackDeletionResult(stack_name="s", status=StackDeletionStatus.SUCCESS, resources=[r])
    deleter._verifier.verify_resources = AsyncMock(
        return_value=[
            ResourceVerificationResult(
                logical_id="A",
                physical_id="p1",
                resource_type="AWS::S3::Bucket",
                cfn_status="CREATE_COMPLETE",
                existence_status=ExistenceStatus.ABSENT,
            )
        ]
    )
    asyncio.run(deleter._verify_deletion(result))
    assert result.status == StackDeletionStatus.SUCCESS


# -- delete_stack full flow --


def test_delete_stack_success_flow(deleter):
    not_found = ClientError(
        {"Error": {"Code": "ValidationError", "Message": "does not exist"}}, "DescribeStacks"
    )

    call_count = 0

    def describe_side_effect(**kwargs):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return {
                "Stacks": [{"StackStatus": "CREATE_COMPLETE", "EnableTerminationProtection": False}]
            }
        raise not_found

    deleter._client.describe_stacks.side_effect = describe_side_effect
    deleter._client.get_paginator.return_value.paginate.return_value = [
        {"StackResourceSummaries": []}
    ]
    deleter._verifier.verify_resources = AsyncMock(return_value=[])
    with patch.object(deleter._cleaner, "cleanup", new_callable=AsyncMock, return_value={}):
        result = asyncio.run(deleter.delete_stack("my-stack"))
    assert result.status == StackDeletionStatus.SUCCESS


# -- Failure capture --


def test_delete_stack_stores_failures_in_manifest(deleter):
    """Test that deletion failures are stored in the manifest."""
    not_found = ClientError(
        {"Error": {"Code": "ValidationError", "Message": "does not exist"}}, "DescribeStacks"
    )

    call_count = 0

    def describe_side_effect(**kwargs):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return {
                "Stacks": [{"StackStatus": "CREATE_COMPLETE", "EnableTerminationProtection": False}]
            }
        raise not_found

    deleter._client.describe_stacks.side_effect = describe_side_effect
    deleter._client.get_paginator.return_value.paginate.return_value = [
        {"StackResourceSummaries": []}
    ]
    deleter._verifier.verify_resources = AsyncMock(return_value=[])

    # Mock cleanup to return failures on retry
    resource = Resource(type="AWS::Lambda::Function", identifier="my-function")
    failure_event = DeletionFailureEvent(status_message="Function is still in use")
    mock_failures = {resource: failure_event}

    with patch.object(
        deleter._cleaner, "cleanup", new_callable=AsyncMock, return_value=mock_failures
    ):
        result = asyncio.run(deleter.delete_stack("my-stack"))

    # Check that failures were captured in manifest
    assert "my-stack" in deleter._manifest
    # No failures should be in manifest for first successful attempt
    assert result.status == StackDeletionStatus.SUCCESS


# -- Manifest tracking for cleanup stages --


def test_try_delete_stack_records_prepare_stage_in_manifest(deleter):
    """The prepare stage is recorded in the manifest's cleanup_stages."""
    deleter._client.describe_stacks.side_effect = ClientError(
        {"Error": {"Code": "ValidationError", "Message": "does not exist"}}, "DescribeStacks"
    )

    with patch.object(deleter._cleaner, "cleanup", new_callable=AsyncMock, return_value={}):
        asyncio.run(deleter._try_delete_stack("my-stack", []))

    assert deleter._manifest["my-stack"]["cleanup_stages"] == {"prepare": True}


def test_delete_stack_not_found_returns_early(deleter):
    """Test that delete_stack returns NOT_FOUND status when stack doesn't exist."""
    deleter._client.describe_stacks.side_effect = ClientError(
        {"Error": {"Code": "ValidationError", "Message": "does not exist"}}, "DescribeStacks"
    )

    result = asyncio.run(deleter.delete_stack("nonexistent-stack"))

    assert result.status == StackDeletionStatus.NOT_FOUND
    assert result.stack_name == "nonexistent-stack"


# -- failure-event logging --


def test_describe_recent_delete_failures_keeps_latest_failed_or_stuck(deleter):
    deleter._client.describe_stack_events.return_value = {
        "StackEvents": [  # newest first
            {
                "LogicalResourceId": "Vpc",
                "PhysicalResourceId": "vpc-1",
                "ResourceType": "AWS::EC2::VPC",
                "ResourceStatus": "DELETE_IN_PROGRESS",
                "ResourceStatusReason": "has dependencies",
            },
            {
                "LogicalResourceId": "Subnet",
                "PhysicalResourceId": "subnet-1",
                "ResourceType": "AWS::EC2::Subnet",
                "ResourceStatus": "DELETE_FAILED",
                "ResourceStatusReason": "has a dependent object (eni-abc)",
            },
            {  # latest event for Bucket is COMPLETE -> excluded
                "LogicalResourceId": "Bucket",
                "PhysicalResourceId": "b",
                "ResourceType": "AWS::S3::Bucket",
                "ResourceStatus": "DELETE_COMPLETE",
                "ResourceStatusReason": "",
            },
            {  # older in-progress for the same Bucket -> must be ignored
                "LogicalResourceId": "Bucket",
                "PhysicalResourceId": "b",
                "ResourceType": "AWS::S3::Bucket",
                "ResourceStatus": "DELETE_IN_PROGRESS",
                "ResourceStatusReason": "",
            },
        ]
    }
    events = deleter._describe_recent_delete_failures("stack")
    statuses = {(e["logical_id"], e["status"]) for e in events}
    assert ("Vpc", "DELETE_IN_PROGRESS") in statuses
    assert ("Subnet", "DELETE_FAILED") in statuses
    assert all(e["logical_id"] != "Bucket" for e in events)


def test_describe_recent_delete_failures_stack_gone_returns_empty(deleter):
    deleter._client.describe_stack_events.side_effect = ClientError(
        {"Error": {"Code": "ValidationError", "Message": "does not exist"}},
        "DescribeStackEvents",
    )
    assert deleter._describe_recent_delete_failures("stack") == []


def test_log_stack_failure_events_persists_and_dedupes(deleter):
    deleter._client.describe_stack_events.return_value = {
        "StackEvents": [
            {
                "LogicalResourceId": "Subnet",
                "PhysicalResourceId": "subnet-1",
                "ResourceType": "AWS::EC2::Subnet",
                "ResourceStatus": "DELETE_FAILED",
                "ResourceStatusReason": "dependent object",
            },
        ]
    }
    asyncio.run(deleter._log_stack_failure_events("stack"))
    assert deleter._manifest["stack"]["failure_events"][0]["logical_id"] == "Subnet"

    deleter._client.describe_stack_events.reset_mock()
    asyncio.run(deleter._log_stack_failure_events("stack"))  # deduped: no second fetch
    deleter._client.describe_stack_events.assert_not_called()


# -- ENI reaper integration --


def test_collect_vpc_ids_dedupes_and_skips_blank(deleter):
    resources = [
        StackResource("Vpc1", "vpc-1", "AWS::EC2::VPC", "DELETE_FAILED"),
        StackResource("Vpc1Dup", "vpc-1", "AWS::EC2::VPC", "DELETE_FAILED"),
        StackResource("Subnet", "subnet-1", "AWS::EC2::Subnet", "DELETE_FAILED"),
        StackResource("VpcNoId", "", "AWS::EC2::VPC", "DELETE_FAILED"),
    ]
    assert deleter._collect_resource_physical_ids(resources, "AWS::EC2::VPC") == ["vpc-1"]


def test_reap_and_retry_networking_success(deleter):
    result = StackDeletionResult(
        stack_name="eks-stack",
        status=StackDeletionStatus.FAILED,
        resources=[StackResource("Vpc", "vpc-1", "AWS::EC2::VPC", "DELETE_FAILED")],
    )
    deleter._poll_deletion = AsyncMock(return_value="DELETE_COMPLETE")
    with (
        patch(
            "aws_bench.resource_management.cleanup.stack_deleter.reap_vpc_enis",
            return_value=EniReapResult(deleted=["eni-1"]),
        ) as mock_reap,
        patch(
            "aws_bench.resource_management.cleanup.stack_deleter.reap_vpc_security_groups",
            return_value=[],
        ) as mock_sg,
    ):
        asyncio.run(deleter._reap_and_retry_networking("eks-stack", result))
    mock_reap.assert_called_once()
    # Leftover non-default SGs are drained before the re-drive so the VPC deletes.
    mock_sg.assert_called_once()
    assert mock_sg.call_args.args[1] == ["vpc-1"]
    deleter._client.delete_stack.assert_called_once_with(StackName="eks-stack")
    assert result.status == StackDeletionStatus.SUCCESS
    assert deleter._manifest["eks-stack"]["eni_reap"]["deleted"] == ["eni-1"]


def test_reap_and_retry_networking_noop_without_vpc(deleter):
    result = StackDeletionResult(
        stack_name="s3-stack",
        status=StackDeletionStatus.FAILED,
        resources=[StackResource("Bucket", "b", "AWS::S3::Bucket", "DELETE_FAILED")],
    )
    with patch("aws_bench.resource_management.cleanup.stack_deleter.reap_vpc_enis") as mock_reap:
        asyncio.run(deleter._reap_and_retry_networking("s3-stack", result))
    mock_reap.assert_not_called()
    assert result.status == StackDeletionStatus.FAILED


def test_reap_and_retry_networking_defers_when_requester_enis_remain(deleter):
    # Only requester-managed ENIs remain (their owner hasn't released them) and the
    # stack's outstanding blocker is its subnet — an eventually-deletable state. The
    # stack is DEFERRED (not force-retried): DeleteStack is not re-issued, the result
    # keeps FAILED status but is flagged deferred, and the stack + its networking
    # resources + the leftover ENI are recorded in the run's deferred registry.
    result = StackDeletionResult(
        stack_name="opensearch-stack",
        status=StackDeletionStatus.FAILED,
        resources=[
            StackResource("Vpc", "vpc-1", "AWS::EC2::VPC", "DELETE_FAILED"),
            StackResource("Subnet", "subnet-1", "AWS::EC2::Subnet", "DELETE_FAILED"),
            StackResource("Sg", "sg-1", "AWS::EC2::SecurityGroup", "DELETE_FAILED"),
        ],
    )
    deleter._poll_deletion = AsyncMock()
    deleter._client.describe_stack_events.return_value = {
        "StackEvents": [
            {
                "LogicalResourceId": "Subnet",
                "PhysicalResourceId": "subnet-1",
                "ResourceType": "AWS::EC2::Subnet",
                "ResourceStatus": "DELETE_FAILED",
                "ResourceStatusReason": "has dependencies and cannot be deleted",
            }
        ]
    }
    deleter._client.describe_stacks.return_value = {
        "Stacks": [{"StackId": "arn:aws:cloudformation:us-east-1:1:stack/opensearch-stack/uuid"}]
    }
    with (
        deferred_scope(),
        patch(
            "aws_bench.resource_management.cleanup.stack_deleter.reap_vpc_enis",
            return_value=EniReapResult(remaining=["eni-os"]),
        ),
    ):
        asyncio.run(deleter._reap_and_retry_networking("opensearch-stack", result))

        deleter._client.delete_stack.assert_not_called()
        deleter._poll_deletion.assert_not_awaited()
        assert result.status == StackDeletionStatus.FAILED
        assert result.deferred is True
        # The stack (by ARN) and every networking resource + leftover ENI are deferred,
        # so the post-cleanup orphan scan excludes them.
        assert is_deferred(
            "AWS::CloudFormation::Stack",
            "arn:aws:cloudformation:us-east-1:1:stack/opensearch-stack/uuid",
        )
        assert is_deferred("AWS::EC2::VPC", "vpc-1")
        assert is_deferred("AWS::EC2::Subnet", "subnet-1")
        assert is_deferred("AWS::EC2::SecurityGroup", "sg-1")
        assert is_deferred("AWS::EC2::NetworkInterface", "eni-os")
    assert deleter._manifest["opensearch-stack"]["deferred"]["remaining_enis"] == ["eni-os"]


def test_reap_and_retry_networking_not_deferred_on_non_networking_blocker(deleter):
    # A requester-managed ENI remains, but the stack also has a non-networking
    # DELETE_FAILED resource (a stuck custom resource) — a genuine, non-self-healing
    # blocker. The stack must NOT be deferred; it stays plainly FAILED for the
    # force-delete last resort, and nothing is recorded in the deferred registry.
    result = StackDeletionResult(
        stack_name="bedrock-stack",
        status=StackDeletionStatus.FAILED,
        resources=[StackResource("Vpc", "vpc-1", "AWS::EC2::VPC", "DELETE_FAILED")],
    )
    deleter._poll_deletion = AsyncMock()
    deleter._client.describe_stack_events.return_value = {
        "StackEvents": [
            {
                "LogicalResourceId": "SeedDoc",
                "PhysicalResourceId": "seed",
                "ResourceType": "Custom::AWS",
                "ResourceStatus": "DELETE_FAILED",
                "ResourceStatusReason": "did not receive a response",
            }
        ]
    }
    with (
        deferred_scope(),
        patch(
            "aws_bench.resource_management.cleanup.stack_deleter.reap_vpc_enis",
            return_value=EniReapResult(remaining=["eni-x"]),
        ),
    ):
        asyncio.run(deleter._reap_and_retry_networking("bedrock-stack", result))

        deleter._client.delete_stack.assert_not_called()
        assert result.status == StackDeletionStatus.FAILED
        assert result.deferred is False
        assert not is_deferred("AWS::EC2::VPC", "vpc-1")
        assert not is_deferred("AWS::EC2::NetworkInterface", "eni-x")


def test_reap_and_retry_networking_retries_when_vpc_left_eni_clear(deleter):
    # The reaper deleted/detached nothing itself (reaped_any is False) because the
    # EKS control-plane X-ENIs were released by their owning service *while the
    # reaper waited* — leaving the VPC ENI-clear (remaining == []). The first
    # DeleteStack failed only because those interfaces were still attached, so the
    # retry must fire and now completes. This is the exact case observed live.
    result = StackDeletionResult(
        stack_name="eks-stack",
        status=StackDeletionStatus.FAILED,
        resources=[StackResource("Vpc", "vpc-1", "AWS::EC2::VPC", "DELETE_FAILED")],
    )
    deleter._poll_deletion = AsyncMock(return_value="DELETE_COMPLETE")
    with patch(
        "aws_bench.resource_management.cleanup.stack_deleter.reap_vpc_enis",
        return_value=EniReapResult(),
    ) as mock_reap:
        asyncio.run(deleter._reap_and_retry_networking("eks-stack", result))
    mock_reap.assert_called_once()
    deleter._client.delete_stack.assert_called_once_with(StackName="eks-stack")
    assert result.status == StackDeletionStatus.SUCCESS


def test_reap_and_retry_networking_clears_wedge_before_reap_and_records_manifest(deleter):
    # The single-stack retry path must clear the NAT/EIP/IGW wedge (which the shared
    # hooks never run here) BEFORE the ENI reap — a NAT gateway holds its own ENI —
    # and record what it did in the manifest alongside the eni_reap entry.
    result = StackDeletionResult(
        stack_name="net-stack",
        status=StackDeletionStatus.FAILED,
        resources=[StackResource("Vpc", "vpc-1", "AWS::EC2::VPC", "DELETE_FAILED")],
    )
    deleter._poll_deletion = AsyncMock(return_value="DELETE_COMPLETE")
    call_order: list[str] = []
    wedge = VpcPublicAddressWedgeResult(
        nat_deleted=["nat-1"], eips_released=["eipalloc-1"], igws_deleted=["igw-1"]
    )
    with (
        patch(
            "aws_bench.resource_management.cleanup.stack_deleter.clear_vpc_public_address_wedge",
            side_effect=lambda *a, **k: call_order.append("wedge") or wedge,
        ) as mock_wedge,
        patch(
            "aws_bench.resource_management.cleanup.stack_deleter.reap_vpc_enis",
            side_effect=lambda *a, **k: (
                call_order.append("enis") or EniReapResult(deleted=["eni-1"])
            ),
        ) as mock_reap,
    ):
        asyncio.run(deleter._reap_and_retry_networking("net-stack", result))

    mock_wedge.assert_called_once()
    mock_reap.assert_called_once()
    assert call_order == ["wedge", "enis"]  # wedge cleared before the ENI reap
    deleter._client.delete_stack.assert_called_once_with(StackName="net-stack")
    assert result.status == StackDeletionStatus.SUCCESS
    teardown = deleter._manifest["net-stack"]["nat_eip_igw_teardown"]
    assert teardown["nat_deleted"] == ["nat-1"]
    assert teardown["eips_released"] == ["eipalloc-1"]
    assert teardown["igws_deleted"] == ["igw-1"]
    assert teardown["remaining"] == []


def test_reap_and_retry_networking_no_retry_when_only_wedge_orphans(deleter):
    # The wedge could not fully clear (a NAT gateway never reached 'deleted') and the
    # ENI reap found nothing to do: no progress + orphans means a retry would stall,
    # so DeleteStack is not re-issued and the orphan is recorded.
    result = StackDeletionResult(
        stack_name="net-stack",
        status=StackDeletionStatus.FAILED,
        resources=[StackResource("Vpc", "vpc-1", "AWS::EC2::VPC", "DELETE_FAILED")],
    )
    deleter._poll_deletion = AsyncMock()
    with (
        patch(
            "aws_bench.resource_management.cleanup.stack_deleter.clear_vpc_public_address_wedge",
            return_value=VpcPublicAddressWedgeResult(remaining=["nat-slow"]),
        ),
        patch(
            "aws_bench.resource_management.cleanup.stack_deleter.reap_vpc_enis",
            return_value=EniReapResult(),
        ),
    ):
        asyncio.run(deleter._reap_and_retry_networking("net-stack", result))
    deleter._client.delete_stack.assert_not_called()
    deleter._poll_deletion.assert_not_awaited()
    assert result.status == StackDeletionStatus.FAILED
    assert deleter._manifest["net-stack"]["nat_eip_igw_teardown"]["remaining"] == ["nat-slow"]


def test_reap_and_retry_networking_retries_when_only_wedge_made_progress(deleter):
    # The ENI reap found nothing (reaped_any False) but the wedge deleted a NAT gateway
    # (cleared_any True), so progress WAS made — the retry must still fire.
    result = StackDeletionResult(
        stack_name="net-stack",
        status=StackDeletionStatus.FAILED,
        resources=[StackResource("Vpc", "vpc-1", "AWS::EC2::VPC", "DELETE_FAILED")],
    )
    deleter._poll_deletion = AsyncMock(return_value="DELETE_COMPLETE")
    with (
        patch(
            "aws_bench.resource_management.cleanup.stack_deleter.clear_vpc_public_address_wedge",
            return_value=VpcPublicAddressWedgeResult(nat_deleted=["nat-1"]),
        ),
        patch(
            "aws_bench.resource_management.cleanup.stack_deleter.reap_vpc_enis",
            return_value=EniReapResult(),
        ),
    ):
        asyncio.run(deleter._reap_and_retry_networking("net-stack", result))
    deleter._client.delete_stack.assert_called_once_with(StackName="net-stack")
    assert result.status == StackDeletionStatus.SUCCESS


def test_reap_and_retry_networking_skips_when_not_failed(deleter):
    result = StackDeletionResult(
        stack_name="eks-stack",
        status=StackDeletionStatus.SUCCESS,
        resources=[StackResource("Vpc", "vpc-1", "AWS::EC2::VPC", "DELETE_COMPLETE")],
    )
    with patch("aws_bench.resource_management.cleanup.stack_deleter.reap_vpc_enis") as mock_reap:
        asyncio.run(deleter._reap_and_retry_networking("eks-stack", result))
    mock_reap.assert_not_called()


def test_delete_stack_reaps_and_logs_events_on_failure(deleter):
    deleter._stack_exists = MagicMock(return_value=True)
    deleter._prepare_stack_for_deletion = AsyncMock(
        return_value=[StackResource("Vpc", "vpc-1", "AWS::EC2::VPC", "DELETE_FAILED")]
    )
    deleter._try_delete_stack = AsyncMock(return_value="DELETE_FAILED")
    deleter._verify_deletion = AsyncMock()
    deleter._get_failure_reason = MagicMock(return_value="stuck on ENI")
    deleter._reap_and_retry_networking = AsyncMock()
    deleter._force_delete_failed_stack = AsyncMock()
    deleter._log_stack_failure_events = AsyncMock()

    result = asyncio.run(deleter.delete_stack("eks-stack"))

    assert result.status == StackDeletionStatus.FAILED
    deleter._reap_and_retry_networking.assert_awaited_once()
    # A still-FAILED stack must reach the force-delete last resort.
    deleter._force_delete_failed_stack.assert_awaited_once()
    deleter._log_stack_failure_events.assert_awaited_once_with("eks-stack")


def test_delete_stack_skips_event_logging_when_reaper_recovers(deleter):
    deleter._stack_exists = MagicMock(return_value=True)
    deleter._prepare_stack_for_deletion = AsyncMock(
        return_value=[StackResource("Vpc", "vpc-1", "AWS::EC2::VPC", "DELETE_FAILED")]
    )
    deleter._try_delete_stack = AsyncMock(return_value="DELETE_FAILED")
    deleter._verify_deletion = AsyncMock()
    deleter._get_failure_reason = MagicMock(return_value="stuck on ENI")
    deleter._force_delete_failed_stack = AsyncMock()
    deleter._log_stack_failure_events = AsyncMock()

    def _recover(_stack_name, result):
        result.status = StackDeletionStatus.SUCCESS

    deleter._reap_and_retry_networking = AsyncMock(side_effect=_recover)

    result = asyncio.run(deleter.delete_stack("eks-stack"))

    assert result.status == StackDeletionStatus.SUCCESS
    # Reaper already recovered the stack, so the force-delete last resort must be skipped.
    deleter._force_delete_failed_stack.assert_not_awaited()
    deleter._log_stack_failure_events.assert_not_awaited()


def test_delete_stack_reaches_ipam_reap_on_failed_ipam_stack(deleter):
    """delete_stack routes a DELETE_FAILED IPAM stack through the IPAM child-pool reap.

    This is the entry point reset relies on: reset's _delete_unrecoverable_stacks ->
    StackRestorer._delete_for_resetup -> CleanupManager.cleanup_stack -> this
    StackDeleter.delete_stack. So an IPAM stack reset finds in DELETE_FAILED must hit
    _reap_and_retry_ipam here — with the networking reap a no-op (no VPCs). A successful
    reap+redrive flips the stack to SUCCESS without falling through to force-delete.
    """
    deleter._stack_exists = MagicMock(return_value=True)
    deleter._prepare_stack_for_deletion = AsyncMock(
        return_value=[StackResource("Pool", "ipam-pool-1", "AWS::EC2::IPAMPool", "DELETE_FAILED")]
    )
    deleter._try_delete_stack = AsyncMock(return_value="DELETE_FAILED")
    deleter._get_failure_reason = MagicMock(return_value="child pool allocation")
    deleter._force_delete_failed_stack = AsyncMock()
    deleter._log_stack_failure_events = AsyncMock()
    deleter._verify_deletion = AsyncMock()
    # Real networking reaper is a no-op here (no VPCs); the real IPAM reaper reaps a
    # leaked child pool and the redrive poll reports the stack gone -> SUCCESS.
    deleter._poll_deletion = AsyncMock(return_value="DELETE_COMPLETE")

    with patch(
        "aws_bench.resource_management.cleanup.stack_deleter.reap_ipam_child_pools",
        return_value=IpamPoolReapResult(deleted=["ipam-pool-child"]),
    ) as mock_ipam_reap:
        result = asyncio.run(deleter.delete_stack("ipam-stack"))

    # The reaper reached via the delete_stack entry point (not a direct call) reaped
    # under the stack's own pool as parent, and the stack recovered without force-delete.
    mock_ipam_reap.assert_called_once()
    assert mock_ipam_reap.call_args.args[1] == ["ipam-pool-1"]
    assert result.status == StackDeletionStatus.SUCCESS
    deleter._force_delete_failed_stack.assert_not_awaited()
    deleter._log_stack_failure_events.assert_not_awaited()
    assert deleter._manifest["ipam-stack"]["ipam_child_pool_reap"]["deleted"] == ["ipam-pool-child"]


def test_delete_stack_ipam_reap_precedes_force_delete_when_reap_fails(deleter):
    """When the IPAM reap cannot clear the blocker, delete_stack still falls to force-delete.

    Locks the ordering delete_stack promises for an IPAM stack: reap first, and only if
    the stack is still FAILED afterwards does the force-delete last resort run. Networking
    reap is a no-op (no VPCs).
    """
    deleter._stack_exists = MagicMock(return_value=True)
    deleter._prepare_stack_for_deletion = AsyncMock(
        return_value=[StackResource("Pool", "ipam-pool-1", "AWS::EC2::IPAMPool", "DELETE_FAILED")]
    )
    deleter._try_delete_stack = AsyncMock(return_value="DELETE_FAILED")
    deleter._get_failure_reason = MagicMock(return_value="child pool allocation")
    deleter._force_delete_failed_stack = AsyncMock()
    deleter._log_stack_failure_events = AsyncMock()
    deleter._verify_deletion = AsyncMock()

    # Nothing reaped -> _reap_and_retry_ipam leaves the result FAILED (no redrive),
    # so the force-delete last resort must run.
    with patch(
        "aws_bench.resource_management.cleanup.stack_deleter.reap_ipam_child_pools",
        return_value=IpamPoolReapResult(remaining=["ipam-pool-child"]),
    ) as mock_ipam_reap:
        result = asyncio.run(deleter.delete_stack("ipam-stack"))

    mock_ipam_reap.assert_called_once()
    assert result.status == StackDeletionStatus.FAILED
    deleter._force_delete_failed_stack.assert_awaited_once()


# -- _force_delete_failed_stack --


def test_force_delete_failed_stack_success_flips_to_success(deleter):
    """A DELETE_FAILED stack force-deletes to SUCCESS and records abandoned resources."""
    # Still DELETE_FAILED when the force branch checks, then gone after force-delete.
    deleter._client.describe_stacks.return_value = {"Stacks": [{"StackStatus": "DELETE_FAILED"}]}
    deleter._client.get_paginator.return_value.paginate.return_value = [
        {
            "StackResourceSummaries": [
                {
                    "LogicalResourceId": "Bucket",
                    "PhysicalResourceId": "my-bucket",
                    "ResourceType": "AWS::S3::Bucket",
                    "ResourceStatus": "DELETE_FAILED",
                },
                {
                    "LogicalResourceId": "Fn",
                    "PhysicalResourceId": "my-fn",
                    "ResourceType": "AWS::Lambda::Function",
                    "ResourceStatus": "DELETE_COMPLETE",
                },
            ]
        }
    ]
    result = StackDeletionResult(stack_name="my-stack", status=StackDeletionStatus.FAILED)
    # Drive the sweep so Bucket stays a stuck survivor: it verifies EXISTS and the
    # cleaner fails to delete it. abandoned_resources then carries only that survivor.
    failed_resource = Resource(type="AWS::S3::Bucket", identifier="my-bucket")
    with (
        patch.object(
            deleter, "_poll_deletion", new_callable=AsyncMock, return_value="DELETE_COMPLETE"
        ),
        patch.object(
            deleter._verifier,
            "verify_resources",
            new_callable=AsyncMock,
            return_value=[
                ResourceVerificationResult(
                    logical_id="Bucket",
                    physical_id="my-bucket",
                    resource_type="AWS::S3::Bucket",
                    cfn_status="DELETE_FAILED",
                    existence_status=ExistenceStatus.EXISTS,
                )
            ],
        ),
        patch.object(
            deleter._cleaner,
            "cleanup",
            new_callable=AsyncMock,
            return_value={failed_resource: DeletionFailureEvent(status_message="AccessDenied")},
        ),
    ):
        asyncio.run(deleter._force_delete_failed_stack("my-stack", result))

    deleter._client.delete_stack.assert_called_once_with(
        StackName="my-stack", DeletionMode="FORCE_DELETE_STACK"
    )
    assert result.status == StackDeletionStatus.SUCCESS
    assert result.reason == ""
    assert "Bucket" in deleter._manifest["my-stack"]["force_deleted"]
    assert "Fn" not in deleter._manifest["my-stack"]["force_deleted"]
    # Only the survivor the sweep could not delete rides on the result as an orphan;
    # everything the sweep reaped is excluded so it does not false-positive.
    assert [r.logical_id for r in result.abandoned_resources] == ["Bucket"]
    assert result.abandoned_resources[0].resource_type == "AWS::S3::Bucket"
    assert result.abandoned_resources[0].physical_id == "my-bucket"


def test_force_delete_failed_stack_no_op_when_not_delete_failed(deleter):
    """No force-delete when the stack is no longer DELETE_FAILED."""
    deleter._client.describe_stacks.return_value = {
        "Stacks": [{"StackStatus": "DELETE_IN_PROGRESS"}]
    }
    result = StackDeletionResult(stack_name="my-stack", status=StackDeletionStatus.FAILED)
    asyncio.run(deleter._force_delete_failed_stack("my-stack", result))
    deleter._client.delete_stack.assert_not_called()
    assert result.status == StackDeletionStatus.FAILED


def test_force_delete_failed_stack_stays_failed_when_force_incomplete(deleter):
    """If the force-delete does not reach DELETE_COMPLETE, result stays FAILED."""
    deleter._client.describe_stacks.return_value = {"Stacks": [{"StackStatus": "DELETE_FAILED"}]}
    deleter._client.get_paginator.return_value.paginate.return_value = [
        {"StackResourceSummaries": []}
    ]
    result = StackDeletionResult(stack_name="my-stack", status=StackDeletionStatus.FAILED)
    with patch.object(
        deleter, "_poll_deletion", new_callable=AsyncMock, return_value="DELETE_FAILED"
    ):
        asyncio.run(deleter._force_delete_failed_stack("my-stack", result))
    assert result.status == StackDeletionStatus.FAILED


def test_force_delete_failed_stack_swallows_client_error(deleter):
    """A ClientError on the force delete_stack is logged, not raised; result stays FAILED."""
    deleter._client.describe_stacks.return_value = {"Stacks": [{"StackStatus": "DELETE_FAILED"}]}
    deleter._client.get_paginator.return_value.paginate.return_value = [
        {"StackResourceSummaries": []}
    ]
    deleter._client.delete_stack.side_effect = ClientError(
        {"Error": {"Code": "ValidationError", "Message": "nope"}}, "DeleteStack"
    )
    result = StackDeletionResult(stack_name="my-stack", status=StackDeletionStatus.FAILED)
    asyncio.run(deleter._force_delete_failed_stack("my-stack", result))
    assert result.status == StackDeletionStatus.FAILED


def test_force_delete_failed_stack_swallows_client_error_from_poll(deleter):
    """A ClientError while polling is caught too, not just on delete_stack.

    Regression guard: the poll must sit inside the try/except (the single-stack path
    has no gather() net), so a mid-poll throttle leaves the result FAILED.
    """
    deleter._client.describe_stacks.return_value = {"Stacks": [{"StackStatus": "DELETE_FAILED"}]}
    deleter._client.get_paginator.return_value.paginate.return_value = [
        {"StackResourceSummaries": []}
    ]
    result = StackDeletionResult(stack_name="my-stack", status=StackDeletionStatus.FAILED)
    with patch.object(
        deleter,
        "_poll_deletion",
        new_callable=AsyncMock,
        side_effect=ClientError(
            {"Error": {"Code": "Throttling", "Message": "rate exceeded"}}, "DescribeStacks"
        ),
    ):
        asyncio.run(deleter._force_delete_failed_stack("my-stack", result))
    assert result.status == StackDeletionStatus.FAILED


def test_force_delete_failed_stack_finalizes_when_no_resources_remain(deleter):
    """Resources already gone, stack state lags: finalizes to SUCCESS, no force_deleted key.

    The former ``_finalize_failed_stack`` case — nothing to abandon.
    """
    deleter._client.describe_stacks.return_value = {"Stacks": [{"StackStatus": "DELETE_FAILED"}]}
    deleter._client.get_paginator.return_value.paginate.return_value = [
        {"StackResourceSummaries": []}
    ]
    result = StackDeletionResult(stack_name="my-stack", status=StackDeletionStatus.FAILED)
    with patch.object(
        deleter, "_poll_deletion", new_callable=AsyncMock, return_value="DELETE_COMPLETE"
    ):
        asyncio.run(deleter._force_delete_failed_stack("my-stack", result))

    assert result.status == StackDeletionStatus.SUCCESS
    assert result.reason == ""
    assert "force_deleted" not in deleter._manifest.get("my-stack", {})
    assert result.abandoned_resources == []


# -- _sweep_force_abandoned --


def _abandoned_snapshot() -> list[StackResource]:
    """Pre-force-delete resource snapshot mirroring the orphaned-bucket incident.

    The custom resource is DELETE_FAILED; the bucket it guarded was never
    attempted (DELETE_SKIPPED at force time, still CREATE_COMPLETE in the
    snapshot). Both must be swept; the DELETE_COMPLETE one must not.
    """
    return [
        StackResource(
            logical_id="AutoDeleteCR",
            physical_id="cr-physical-id",
            resource_type="Custom::S3AutoDeleteObjects",
            status="DELETE_FAILED",
        ),
        StackResource(
            logical_id="ALBLogsBucket",
            physical_id="tigris-logs-111111111111",
            resource_type="AWS::S3::Bucket",
            status="CREATE_COMPLETE",
        ),
        StackResource(
            logical_id="Fn",
            physical_id="my-fn",
            resource_type="AWS::Lambda::Function",
            status="DELETE_COMPLETE",
        ),
    ]


def _verification(res: StackResource, status: ExistenceStatus) -> ResourceVerificationResult:
    return ResourceVerificationResult(
        logical_id=res.logical_id,
        physical_id=res.physical_id,
        resource_type=res.resource_type,
        cfn_status=res.status,
        existence_status=status,
    )


def test_sweep_force_abandoned_deletes_surviving_resources(deleter):
    """Survivors of a force-delete (EXISTS) are fed to the cleaner; gone ones are not.

    Regression: the orphaned fixed-name bucket incident. The bucket was
    DELETE_SKIPPED (snapshot status CREATE_COMPLETE, never attempted), so any
    sweep keyed off DELETE_FAILED alone would miss it.
    """
    snapshot = _abandoned_snapshot()
    verifications = [
        _verification(snapshot[0], ExistenceStatus.SKIPPED),  # Custom:: unverifiable
        _verification(snapshot[1], ExistenceStatus.EXISTS),  # the orphaned bucket
        _verification(snapshot[2], ExistenceStatus.ABSENT),  # deleted fine
    ]
    with (
        patch.object(deleter._verifier, "verify_resources", new_callable=AsyncMock) as mock_verify,
        patch.object(
            deleter._cleaner, "cleanup", new_callable=AsyncMock, return_value={}
        ) as mock_cleanup,
    ):
        mock_verify.return_value = verifications
        asyncio.run(deleter._sweep_force_abandoned("my-stack", snapshot))

    # Only the not-DELETE_COMPLETE resources are verified.
    verify_args, _ = mock_verify.call_args_list[0]
    assert {r.logical_id for r in verify_args[0]} == {"AutoDeleteCR", "ALBLogsBucket"}
    # Only the EXISTS survivor is cleaned, via prepare (empty bucket) + delete.
    mock_cleanup.assert_awaited_once()
    cleanup_args, cleanup_kwargs = mock_cleanup.call_args_list[0]
    assert [r.logical_id for r in cleanup_args[0]] == ["ALBLogsBucket"]
    assert cleanup_kwargs == {
        "prepare": True,
        "custom_delete": True,
        "ccapi_fallback": True,
    }
    assert deleter._manifest["my-stack"]["force_abandoned_swept"] == ["ALBLogsBucket"]


def test_sweep_force_abandoned_noop_when_nothing_survives(deleter):
    """All abandoned resources verify ABSENT: no cleaner call, no manifest key."""
    snapshot = _abandoned_snapshot()
    verifications = [
        _verification(snapshot[0], ExistenceStatus.ABSENT),
        _verification(snapshot[1], ExistenceStatus.ABSENT),
    ]
    with (
        patch.object(
            deleter._verifier,
            "verify_resources",
            new_callable=AsyncMock,
            return_value=verifications,
        ),
        patch.object(deleter._cleaner, "cleanup", new_callable=AsyncMock) as mock_cleanup,
    ):
        asyncio.run(deleter._sweep_force_abandoned("my-stack", snapshot))

    mock_cleanup.assert_not_awaited()
    assert "force_abandoned_swept" not in deleter._manifest.get("my-stack", {})


def test_sweep_force_abandoned_attempts_unverified_resources(deleter):
    """An UNKNOWN resource is swept; a SKIPPED one is not.

    UNKNOWN means the existence check could not answer, not that the resource is gone, and
    deleting an already-absent resource is a no-op. SKIPPED asserts CCAPI cannot act on the
    type, so a delete through it is equally impossible.
    """
    snapshot = _abandoned_snapshot()
    verifications = [
        _verification(snapshot[0], ExistenceStatus.SKIPPED),
        _verification(snapshot[1], ExistenceStatus.UNKNOWN),
        _verification(snapshot[2], ExistenceStatus.ABSENT),
    ]
    with (
        patch.object(
            deleter._verifier,
            "verify_resources",
            new_callable=AsyncMock,
            return_value=verifications,
        ),
        patch.object(
            deleter._cleaner, "cleanup", new_callable=AsyncMock, return_value={}
        ) as mock_cleanup,
    ):
        asyncio.run(deleter._sweep_force_abandoned("my-stack", snapshot))

    mock_cleanup.assert_awaited_once()
    cleanup_args, _ = mock_cleanup.call_args_list[0]
    assert [r.logical_id for r in cleanup_args[0]] == ["ALBLogsBucket"]
    assert deleter._manifest["my-stack"]["force_abandoned_swept"] == ["ALBLogsBucket"]


def test_sweep_force_abandoned_leaves_unchecked_subresources_to_their_parent(deleter):
    """An UNCHECKED_SUBRESOURCE candidate is not swept.

    A sub-resource cannot be verified or deleted without its parent's context, so it is
    reclaimed by deleting the parent rather than attempted on its own.
    """
    snapshot = _abandoned_snapshot()
    verifications = [
        _verification(snapshot[0], ExistenceStatus.UNCHECKED_SUBRESOURCE),
        _verification(snapshot[1], ExistenceStatus.UNCHECKED_SUBRESOURCE),
    ]
    with (
        patch.object(
            deleter._verifier,
            "verify_resources",
            new_callable=AsyncMock,
            return_value=verifications,
        ),
        patch.object(deleter._cleaner, "cleanup", new_callable=AsyncMock) as mock_cleanup,
    ):
        stuck = asyncio.run(deleter._sweep_force_abandoned("my-stack", snapshot))

    mock_cleanup.assert_not_awaited()
    assert stuck == []


def test_sweep_force_abandoned_logs_dropped_candidates(deleter, caplog):
    """Candidates excluded from the sweep are logged even when others are swept."""
    snapshot = _abandoned_snapshot()
    verifications = [
        _verification(snapshot[0], ExistenceStatus.SKIPPED),
        _verification(snapshot[1], ExistenceStatus.EXISTS),
    ]
    with (
        patch.object(
            deleter._verifier,
            "verify_resources",
            new_callable=AsyncMock,
            return_value=verifications,
        ),
        patch.object(deleter._cleaner, "cleanup", new_callable=AsyncMock, return_value={}),
        caplog.at_level(
            logging.DEBUG, logger="aws_bench.resource_management.cleanup.stack_deleter"
        ),
    ):
        asyncio.run(deleter._sweep_force_abandoned("my-stack", snapshot))

    assert "not swept" in caplog.text
    assert "AutoDeleteCR" in caplog.text


def test_sweep_force_abandoned_is_best_effort(deleter):
    """A sweep failure is swallowed (WARN), never raised to the caller."""
    snapshot = _abandoned_snapshot()
    with patch.object(
        deleter._verifier,
        "verify_resources",
        new_callable=AsyncMock,
        side_effect=ClientError({"Error": {"Code": "Throttling", "Message": "x"}}, "Verify"),
    ):
        asyncio.run(deleter._sweep_force_abandoned("my-stack", snapshot))  # must not raise


def test_sweep_force_abandoned_records_cleanup_failures(deleter):
    """Resources the cleaner could not delete are recorded, and the sweep stays quiet."""
    snapshot = _abandoned_snapshot()
    verifications = [_verification(snapshot[1], ExistenceStatus.EXISTS)]
    failed_resource = Resource(type="AWS::S3::Bucket", identifier="tigris-logs-111111111111")
    with (
        patch.object(
            deleter._verifier,
            "verify_resources",
            new_callable=AsyncMock,
            return_value=verifications,
        ),
        patch.object(
            deleter._cleaner,
            "cleanup",
            new_callable=AsyncMock,
            return_value={failed_resource: DeletionFailureEvent(status_message="AccessDenied")},
        ),
    ):
        asyncio.run(deleter._sweep_force_abandoned("my-stack", snapshot[1:2]))

    assert deleter._manifest["my-stack"]["force_abandoned_sweep_failures"] == [
        "tigris-logs-111111111111"
    ]


def test_force_delete_success_triggers_abandoned_sweep(deleter):
    """After a force-delete flips the result to SUCCESS, the sweep runs on the snapshot."""
    deleter._client.describe_stacks.return_value = {"Stacks": [{"StackStatus": "DELETE_FAILED"}]}
    deleter._client.get_paginator.return_value.paginate.return_value = [
        {
            "StackResourceSummaries": [
                {
                    "LogicalResourceId": "Bucket",
                    "PhysicalResourceId": "my-bucket",
                    "ResourceType": "AWS::S3::Bucket",
                    "ResourceStatus": "DELETE_FAILED",
                },
            ]
        }
    ]
    result = StackDeletionResult(stack_name="my-stack", status=StackDeletionStatus.FAILED)
    with (
        patch.object(
            deleter, "_poll_deletion", new_callable=AsyncMock, return_value="DELETE_COMPLETE"
        ),
        patch.object(deleter, "_sweep_force_abandoned", new_callable=AsyncMock) as mock_sweep,
    ):
        asyncio.run(deleter._force_delete_failed_stack("my-stack", result))

    assert result.status == StackDeletionStatus.SUCCESS
    mock_sweep.assert_awaited_once()
    (swept_stack, swept_resources), _ = mock_sweep.call_args_list[0]
    assert swept_stack == "my-stack"
    assert [r.logical_id for r in swept_resources] == ["Bucket"]


def test_force_delete_failure_skips_abandoned_sweep(deleter):
    """No sweep when the force-delete itself did not reach DELETE_COMPLETE."""
    deleter._client.describe_stacks.return_value = {"Stacks": [{"StackStatus": "DELETE_FAILED"}]}
    deleter._client.get_paginator.return_value.paginate.return_value = [
        {"StackResourceSummaries": []}
    ]
    result = StackDeletionResult(stack_name="my-stack", status=StackDeletionStatus.FAILED)
    with (
        patch.object(
            deleter, "_poll_deletion", new_callable=AsyncMock, return_value="DELETE_FAILED"
        ),
        patch.object(deleter, "_sweep_force_abandoned", new_callable=AsyncMock) as mock_sweep,
    ):
        asyncio.run(deleter._force_delete_failed_stack("my-stack", result))

    assert result.status == StackDeletionStatus.FAILED
    mock_sweep.assert_not_awaited()


def test_export_blocked_stacks_retried_after_importers_deleted(deleter):
    """Stacks blocked by export dependencies are retried after the initial batch.

    The first attempt returns EXPORT_BLOCKED (importer still live). After the
    initial batch deletes the importer, the retry should succeed.
    """
    from aws_bench.resource_management.cleanup.stack_deleter import _EXPORT_BLOCKED_REASON

    blocked_result = StackDeletionResult(
        stack_name="networking-stack",
        status=StackDeletionStatus.FAILED,
        reason=_EXPORT_BLOCKED_REASON,
    )
    success_result = StackDeletionResult(
        stack_name="networking-stack",
        status=StackDeletionStatus.SUCCESS,
    )

    # Simulate: first batch returns the blocked result, retry succeeds
    with patch.object(
        deleter,
        "_delete_stacks_concurrently",
        new_callable=AsyncMock,
        side_effect=[[blocked_result], [success_result]],
    ):
        with patch.object(deleter, "_maybe_delete_infra", new_callable=AsyncMock, return_value=[]):
            summary = asyncio.run(deleter.delete_all_stacks())

    assert any(
        r.stack_name == "networking-stack" and r.status == StackDeletionStatus.SUCCESS
        for r in summary.results
    )


# -- IPAM child-pool reaper integration --


def test_collect_ipam_pool_ids_dedupes_and_skips_other_types(deleter):
    resources = [
        StackResource("Pool1", "ipam-pool-1", "AWS::EC2::IPAMPool", "DELETE_FAILED"),
        StackResource("Pool1Dup", "ipam-pool-1", "AWS::EC2::IPAMPool", "DELETE_FAILED"),
        StackResource("Pool2", "ipam-pool-2", "AWS::EC2::IPAMPool", "DELETE_FAILED"),
        StackResource("Vpc", "vpc-1", "AWS::EC2::VPC", "DELETE_FAILED"),
        StackResource("PoolNoId", "", "AWS::EC2::IPAMPool", "DELETE_FAILED"),
    ]
    assert deleter._collect_resource_physical_ids(resources, "AWS::EC2::IPAMPool") == [
        "ipam-pool-1",
        "ipam-pool-2",
    ]


def test_reap_and_retry_ipam_success(deleter):
    result = StackDeletionResult(
        stack_name="ipam-stack",
        status=StackDeletionStatus.FAILED,
        resources=[StackResource("Pool", "ipam-pool-1", "AWS::EC2::IPAMPool", "DELETE_FAILED")],
    )
    deleter._poll_deletion = AsyncMock(return_value="DELETE_COMPLETE")
    with patch(
        "aws_bench.resource_management.cleanup.stack_deleter.reap_ipam_child_pools",
        return_value=IpamPoolReapResult(deleted=["ipam-pool-child"]),
    ) as mock_reap:
        asyncio.run(deleter._reap_and_retry_ipam("ipam-stack", result))
    mock_reap.assert_called_once()
    # The stack's own pool ids are BOTH the parents to search under AND the CFN-owned
    # set to exclude from reaping.
    assert mock_reap.call_args.args[1] == ["ipam-pool-1"]
    assert mock_reap.call_args.kwargs["stack_owned_pool_ids"] == {"ipam-pool-1"}
    deleter._client.delete_stack.assert_called_once_with(StackName="ipam-stack")
    assert result.status == StackDeletionStatus.SUCCESS
    assert deleter._manifest["ipam-stack"]["ipam_child_pool_reap"]["deleted"] == ["ipam-pool-child"]


def test_reap_and_retry_ipam_no_redrive_when_reap_fails(deleter):
    # Nothing DELETED to unblock the parent — the reaper reported the child FAILED
    # (still present). DeleteStack is NOT re-issued and the result stays plainly FAILED
    # for the force-delete last resort. No defer is set on the result.
    result = StackDeletionResult(
        stack_name="ipam-stack",
        status=StackDeletionStatus.FAILED,
        resources=[StackResource("Pool", "ipam-pool-1", "AWS::EC2::IPAMPool", "DELETE_FAILED")],
    )
    deleter._poll_deletion = AsyncMock()
    with patch(
        "aws_bench.resource_management.cleanup.stack_deleter.reap_ipam_child_pools",
        return_value=IpamPoolReapResult(remaining=["ipam-pool-child"]),
    ):
        asyncio.run(deleter._reap_and_retry_ipam("ipam-stack", result))

    deleter._client.delete_stack.assert_not_called()
    deleter._poll_deletion.assert_not_awaited()
    assert result.status == StackDeletionStatus.FAILED
    assert result.deferred is False
    reap_manifest = deleter._manifest["ipam-stack"]["ipam_child_pool_reap"]
    assert reap_manifest["remaining"] == ["ipam-pool-child"]
    assert "deferred" not in reap_manifest


def test_reap_and_retry_ipam_noop_without_pool(deleter):
    result = StackDeletionResult(
        stack_name="s3-stack",
        status=StackDeletionStatus.FAILED,
        resources=[StackResource("Bucket", "b", "AWS::S3::Bucket", "DELETE_FAILED")],
    )
    with patch(
        "aws_bench.resource_management.cleanup.stack_deleter.reap_ipam_child_pools"
    ) as mock_reap:
        asyncio.run(deleter._reap_and_retry_ipam("s3-stack", result))
    mock_reap.assert_not_called()
    deleter._client.delete_stack.assert_not_called()
    assert result.status == StackDeletionStatus.FAILED


def test_reap_and_retry_ipam_skips_when_not_failed(deleter):
    result = StackDeletionResult(
        stack_name="ipam-stack",
        status=StackDeletionStatus.SUCCESS,
        resources=[StackResource("Pool", "ipam-pool-1", "AWS::EC2::IPAMPool", "DELETE_COMPLETE")],
    )
    with patch(
        "aws_bench.resource_management.cleanup.stack_deleter.reap_ipam_child_pools"
    ) as mock_reap:
        asyncio.run(deleter._reap_and_retry_ipam("ipam-stack", result))
    mock_reap.assert_not_called()


def test_reap_and_retry_ipam_poll_client_error_does_not_propagate(deleter):
    # A throttled describe_stacks during the retry poll must not abort the delete task
    # (which would bypass _force_delete_failed_stack). The ClientError is swallowed and
    # the result stays FAILED for the force-delete fallback.
    result = StackDeletionResult(
        stack_name="ipam-stack",
        status=StackDeletionStatus.FAILED,
        resources=[StackResource("Pool", "ipam-pool-1", "AWS::EC2::IPAMPool", "DELETE_FAILED")],
    )
    deleter._poll_deletion = AsyncMock(
        side_effect=ClientError({"Error": {"Code": "Throttling"}}, "DescribeStacks")
    )
    with patch(
        "aws_bench.resource_management.cleanup.stack_deleter.reap_ipam_child_pools",
        return_value=IpamPoolReapResult(deleted=["ipam-pool-child"]),
    ):
        asyncio.run(deleter._reap_and_retry_ipam("ipam-stack", result))
    deleter._client.delete_stack.assert_called_once_with(StackName="ipam-stack")
    assert result.status == StackDeletionStatus.FAILED


def test_reap_and_retry_networking_poll_client_error_does_not_propagate(deleter):
    # The shared re-drive helper guards the poll for the networking path too.
    result = StackDeletionResult(
        stack_name="eks-stack",
        status=StackDeletionStatus.FAILED,
        resources=[StackResource("Vpc", "vpc-1", "AWS::EC2::VPC", "DELETE_FAILED")],
    )
    deleter._poll_deletion = AsyncMock(
        side_effect=ClientError({"Error": {"Code": "Throttling"}}, "DescribeStacks")
    )
    with (
        patch(
            "aws_bench.resource_management.cleanup.stack_deleter.reap_vpc_enis",
            return_value=EniReapResult(deleted=["eni-1"]),
        ),
        patch(
            "aws_bench.resource_management.cleanup.stack_deleter.reap_vpc_security_groups",
            return_value=[],
        ),
    ):
        asyncio.run(deleter._reap_and_retry_networking("eks-stack", result))
    deleter._client.delete_stack.assert_called_once_with(StackName="eks-stack")
    assert result.status == StackDeletionStatus.FAILED


# -- _delete_stack_kwargs --


def test_delete_stack_kwargs_uses_correct_role_arn_parameter(tmp_path):
    """CloudFormation DeleteStack uses 'RoleARN' (all-caps), not 'RoleArn'."""
    session = MagicMock()
    session.region_name = "us-east-1"
    role_arn = "arn:aws:iam::123456789012:role/cfn-service-execution"
    deleter = StackDeleter(session, manifest_path=tmp_path / "manifest.json", cfn_role_arn=role_arn)
    # Sanity: the ops role validated and was retained.
    assert deleter._cfn_role_arn == role_arn

    kwargs = deleter._delete_stack_kwargs("MyStack")

    assert kwargs["StackName"] == "MyStack"
    assert "RoleARN" in kwargs
    assert "RoleArn" not in kwargs
    assert kwargs["RoleARN"] == role_arn


def test_delete_stack_kwargs_omits_role_when_unconfigured(deleter):
    """Without a validated ops role, no RoleARN/RoleArn is injected."""
    assert deleter._cfn_role_arn is None
    kwargs = deleter._delete_stack_kwargs("MyStack", RetainResources=["Foo"])
    assert kwargs == {"StackName": "MyStack", "RetainResources": ["Foo"]}
    assert "RoleARN" not in kwargs
    assert "RoleArn" not in kwargs
