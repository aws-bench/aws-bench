"""Tests for resource_cleaner — cleanup pipeline orchestration."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.models import (
    CustomDeletionResult,
    HandlerResult,
    HandlerStatus,
    StackResource,
    to_ccapi_resources,
)
from aws_bench.resource_management.cleanup.resource_cleaner import (
    DELETE_BEFORE_PREPARE_TYPES,
    ResourceCleaner,
    format_sample_list,
    partition_delete_before_prepare,
    truncate_for_log,
)

# -- Utility functions --


def test_truncate_for_log_short_text():
    assert truncate_for_log("hello", 10) == "hello"


def test_truncate_for_log_long_text():
    assert truncate_for_log("hello world", 8) == "hello..."


def test_format_sample_list_empty():
    assert format_sample_list([]) == "(none)"


def test_format_sample_list_few_items():
    assert format_sample_list([1, 2]) == "1, 2"


def test_format_sample_list_many_items():
    assert format_sample_list([1, 2, 3, 4, 5], limit=3) == "1, 2, 3 (+2 more)"


def test_format_sample_list_with_custom_formatter():
    result = format_sample_list(["a", "b"], format_fn=lambda x: x.upper())
    assert result == "A, B"


# -- to_ccapi_resources (models.py line 117) --


def test_to_ccapi_resources_skips_empty_physical_id():
    resources = [
        StackResource("L1", "p1", "AWS::S3::Bucket", "CREATE_COMPLETE"),
        StackResource("L2", "", "AWS::IAM::Role", "CREATE_COMPLETE"),
    ]
    result = to_ccapi_resources(resources)
    assert len(result) == 1
    assert result[0].identifier == "p1"


# -- cleanup: prepare --


def test_cleanup_prepare_calls_prepare_all():
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "p1", "AWS::S3::Bucket", "CREATE_COMPLETE")]
    with patch.object(cleaner, "_prepare_all") as mock:
        asyncio.run(cleaner.cleanup(resources, prepare=True))
    mock.assert_called_once()


def test_cleanup_prepare_skips_when_no_resources():
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "", "AWS::S3::Bucket", "CREATE_COMPLETE")]
    with patch.object(cleaner, "_prepare_all") as mock:
        asyncio.run(cleaner.cleanup(resources, prepare=True))
    mock.assert_not_called()


# -- cleanup: handle_stuck --


def test_cleanup_handle_stuck():
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "p1", "AWS::S3::Bucket", "DELETE_FAILED")]
    with patch.object(cleaner, "_handle_failed") as mock:
        asyncio.run(cleaner.cleanup(resources, handle_stuck=True))
    # Only called if FAILED_RESOURCE_HANDLERS is non-empty (it is, from handler imports)
    mock.assert_called_once()


# -- cleanup: validation --


def test_cleanup_raises_when_all_flags_false():
    """Test that cleanup raises ValueError when no operations are enabled."""
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "p1", "AWS::S3::Bucket", "CREATE_COMPLETE")]
    with pytest.raises(ValueError, match="At least one cleanup operation must be enabled"):
        asyncio.run(cleaner.cleanup(resources))


def test_cleanup_returns_empty_without_delete_flags():
    """Test that cleanup with only prepare=True returns empty (no deletions requested)."""
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "p1", "AWS::S3::Bucket", "CREATE_COMPLETE")]
    result = asyncio.run(cleaner.cleanup(resources, prepare=True))
    assert result == {}


# -- cleanup: custom_delete only --


def test_cleanup_custom_delete_only():
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "p1", "AWS::S3::Bucket", "CREATE_COMPLETE")]
    with (
        patch.object(cleaner, "_resolve_hooks", return_value=[]),
    ):
        result = asyncio.run(cleaner.cleanup(resources, custom_delete=True))
    assert result == {}


def test_cleanup_custom_delete_with_resolved():
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "p1", "AWS::S3::Bucket", "CREATE_COMPLETE")]
    resolved = [Resource(type="AWS::S3::Bucket", identifier="p1")]
    mock_result = MagicMock(skipped=[], failed={})
    with (
        patch.object(cleaner, "_resolve_hooks", return_value=resolved),
        patch.object(cleaner, "_custom_delete", return_value=mock_result),
    ):
        result = asyncio.run(cleaner.cleanup(resources, custom_delete=True))
    assert result == {}


# -- cleanup: ccapi_fallback --


def test_cleanup_ccapi_fallback():
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "p1", "AWS::S3::Bucket", "CREATE_COMPLETE")]
    resolved = [Resource(type="AWS::S3::Bucket", identifier="p1")]
    mock_result = MagicMock(skipped=resolved, failed={})
    with (
        patch.object(cleaner, "_resolve_hooks", return_value=resolved),
        patch.object(cleaner, "_custom_delete", return_value=mock_result),
        patch(
            "aws_bench.resource_management.cleanup.resource_cleaner.CloudControlManager"
        ) as mock_ccm_cls,
    ):
        mock_ccm_cls.return_value.delete_resources.return_value = {}
        result = asyncio.run(cleaner.cleanup(resources, custom_delete=True, ccapi_fallback=True))
    assert result == {}


def test_cleanup_custom_delete_claims_cluster_ccapi_gets_the_groups():
    cleaner = ResourceCleaner(MagicMock())
    cluster = StackResource("Cluster", "c-1", "AWS::RDS::DBCluster", "CREATE_COMPLETE")
    subnet_group = StackResource("Sng", "sng-1", "AWS::RDS::DBSubnetGroup", "CREATE_COMPLETE")
    param_group = StackResource(
        "Pg", "pg-1", "AWS::RDS::DBClusterParameterGroup", "CREATE_COMPLETE"
    )
    resources = [cluster, subnet_group, param_group]

    def _fake_cluster_delete(resource, session):
        return HandlerResult(resource.identifier, resource.type, "delete", HandlerStatus.SUCCESS)

    with (
        patch(
            "aws_bench.resource_management.cleanup.resource_cleaner.CUSTOM_DELETION_REGISTRY",
            {"AWS::RDS::DBCluster": _fake_cluster_delete},
        ),
        patch(
            "aws_bench.resource_management.cleanup.resource_cleaner.CloudControlManager"
        ) as mock_ccm_cls,
    ):
        mock_ccm_cls.return_value.delete_resources.return_value = {}
        result = asyncio.run(cleaner.cleanup(resources, custom_delete=True, ccapi_fallback=True))

    ccapi_args = mock_ccm_cls.return_value.delete_resources.call_args.args[0]
    ccapi_types = {r.type for r in ccapi_args}
    assert ccapi_types == {"AWS::RDS::DBSubnetGroup", "AWS::RDS::DBClusterParameterGroup"}
    assert "AWS::RDS::DBCluster" not in ccapi_types
    assert result == {}


# -- _prepare_all --


def test_prepare_all_runs_handler():
    cleaner = ResourceCleaner(MagicMock())
    resource = Resource(type="AWS::S3::Bucket", identifier="b1")
    handler_result = HandlerResult("b1", "AWS::S3::Bucket", "prepare", HandlerStatus.SUCCESS)
    with patch(
        "aws_bench.resource_management.cleanup.resource_cleaner.PREPARE_REGISTRY",
        {"AWS::S3::Bucket": MagicMock(return_value=handler_result)},
    ):
        results = cleaner._prepare_all([resource])
    assert results[0].status == HandlerStatus.SUCCESS


def test_prepare_all_skips_unregistered():
    cleaner = ResourceCleaner(MagicMock())
    resource = Resource(type="AWS::Unknown::Thing", identifier="x")
    with patch("aws_bench.resource_management.cleanup.resource_cleaner.PREPARE_REGISTRY", {}):
        results = cleaner._prepare_all([resource])
    assert results[0].status == HandlerStatus.SKIPPED


def test_prepare_all_handles_exception():
    cleaner = ResourceCleaner(MagicMock())
    resource = Resource(type="AWS::S3::Bucket", identifier="b1")
    with patch(
        "aws_bench.resource_management.cleanup.resource_cleaner.PREPARE_REGISTRY",
        {"AWS::S3::Bucket": MagicMock(side_effect=Exception("boom"))},
    ):
        results = cleaner._prepare_all([resource])
    assert results[0].status == HandlerStatus.FAILED


# -- _custom_delete --


def test_custom_delete_skips_unregistered():
    cleaner = ResourceCleaner(MagicMock())
    resource = Resource(type="AWS::Unknown::Thing", identifier="x")
    with patch(
        "aws_bench.resource_management.cleanup.resource_cleaner.CUSTOM_DELETION_REGISTRY", {}
    ):
        result = cleaner._custom_delete([resource])
    assert resource in result.skipped


def test_custom_delete_runs_handler():
    cleaner = ResourceCleaner(MagicMock())
    resource = Resource(type="AWS::S3::Bucket", identifier="b1")
    with patch(
        "aws_bench.resource_management.cleanup.resource_cleaner.CUSTOM_DELETION_REGISTRY",
        {"AWS::S3::Bucket": MagicMock(return_value=None)},
    ):
        result = cleaner._custom_delete([resource])
    assert resource in result.succeeded


def test_custom_delete_records_handler_failure_result():
    cleaner = ResourceCleaner(MagicMock())
    resource = Resource(type="AWS::S3::Bucket", identifier="b1")
    failed_result = HandlerResult("b1", "AWS::S3::Bucket", "delete", HandlerStatus.FAILED, "err")
    with patch(
        "aws_bench.resource_management.cleanup.resource_cleaner.CUSTOM_DELETION_REGISTRY",
        {"AWS::S3::Bucket": MagicMock(return_value=failed_result)},
    ):
        result = cleaner._custom_delete([resource])
    assert resource in result.failed


def test_custom_delete_records_exception():
    cleaner = ResourceCleaner(MagicMock())
    resource = Resource(type="AWS::S3::Bucket", identifier="b1")
    with patch(
        "aws_bench.resource_management.cleanup.resource_cleaner.CUSTOM_DELETION_REGISTRY",
        {"AWS::S3::Bucket": MagicMock(side_effect=Exception("boom"))},
    ):
        result = cleaner._custom_delete([resource])
    assert resource in result.failed


# -- _resolve_hooks --


def test_resolve_hooks_no_hooks():
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "p1", "AWS::S3::Bucket", "CREATE_COMPLETE")]
    with patch("aws_bench.resource_management.cleanup.resource_cleaner.PRE_DELETE_HOOKS", {}):
        result = cleaner._resolve_hooks(resources)
    assert len(result) == 1


def test_resolve_hooks_discovers_new_resources():
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "vpc-1", "AWS::EC2::VPC", "CREATE_COMPLETE")]
    discovered = Resource(type="AWS::EC2::NetworkInterface", identifier="eni-1")
    hook = MagicMock(return_value=[discovered])
    with patch(
        "aws_bench.resource_management.cleanup.resource_cleaner.PRE_DELETE_HOOKS",
        {"AWS::EC2::VPC": hook},
    ):
        result = cleaner._resolve_hooks(resources)
    assert any(r.identifier == "eni-1" for r in result)


def test_resolve_hooks_deduplicates():
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "p1", "AWS::S3::Bucket", "CREATE_COMPLETE")]
    dup = Resource(type="AWS::S3::Bucket", identifier="p1")
    hook = MagicMock(return_value=[dup])
    with patch(
        "aws_bench.resource_management.cleanup.resource_cleaner.PRE_DELETE_HOOKS",
        {"AWS::S3::Bucket": hook},
    ):
        result = cleaner._resolve_hooks(resources)
    assert len(result) == 1


def test_resolve_hooks_does_not_deduplicate_different_types_same_identifier():
    """Different resource types with same identifier should not be deduplicated."""
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "my-resource", "AWS::S3::Bucket", "CREATE_COMPLETE")]
    # Hook discovers a different resource type with the same identifier
    discovered_eni = Resource(type="AWS::EC2::NetworkInterface", identifier="my-resource")
    hook = MagicMock(return_value=[discovered_eni])
    with patch(
        "aws_bench.resource_management.cleanup.resource_cleaner.PRE_DELETE_HOOKS",
        {"AWS::S3::Bucket": hook},
    ):
        result = cleaner._resolve_hooks(resources)
    # Should have both: the original bucket and the discovered ENI
    assert len(result) == 2
    types = {r.type for r in result}
    assert "AWS::S3::Bucket" in types
    assert "AWS::EC2::NetworkInterface" in types
    # Both should have the same identifier
    assert all(r.identifier == "my-resource" for r in result)


def test_resolve_hooks_handles_exception():
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "vpc-1", "AWS::EC2::VPC", "CREATE_COMPLETE")]
    hook = MagicMock(side_effect=Exception("boom"))
    with patch(
        "aws_bench.resource_management.cleanup.resource_cleaner.PRE_DELETE_HOOKS",
        {"AWS::EC2::VPC": hook},
    ):
        result = cleaner._resolve_hooks(resources)
    assert len(result) == 1


# -- _handle_failed --


def test_handle_failed_runs_matching_handler():
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "p1", "Custom::Thing", "DELETE_FAILED")]
    handler = MagicMock()
    with patch(
        "aws_bench.resource_management.cleanup.resource_cleaner.FAILED_RESOURCE_HANDLERS",
        [(50, "Custom::", handler)],  # (priority, pattern, handler)
    ):
        cleaner._handle_failed(resources)
    handler.assert_called_once()


def test_handle_failed_noop_when_none_failed():
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "p1", "AWS::S3::Bucket", "CREATE_COMPLETE")]
    cleaner._handle_failed(resources)


def test_handle_failed_handles_exception():
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "p1", "Custom::Thing", "DELETE_FAILED")]
    handler = MagicMock(side_effect=Exception("boom"))
    with patch(
        "aws_bench.resource_management.cleanup.resource_cleaner.FAILED_RESOURCE_HANDLERS",
        [(50, "Custom::", handler)],  # (priority, pattern, handler)
    ):
        cleaner._handle_failed(resources)


def test_handle_failed_executes_in_priority_order():
    """Verify handlers execute in priority order (lower priority first)."""
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "p1", "Custom::AWS::Thing", "DELETE_FAILED")]

    execution_order = []
    handler_low = MagicMock(side_effect=lambda *args: execution_order.append("low"))
    handler_high = MagicMock(side_effect=lambda *args: execution_order.append("high"))

    with patch(
        "aws_bench.resource_management.cleanup.resource_cleaner.FAILED_RESOURCE_HANDLERS",
        [
            # List must be sorted by priority (maintained by decorator in production)
            (10, "Custom::", handler_low),  # Lower priority (executes first)
            (50, "Custom::AWS", handler_high),  # Higher priority (executes second)
        ],
    ):
        cleaner._handle_failed(resources)

    # Both handlers should match "Custom::AWS::Thing" via prefix matching
    assert handler_low.call_count == 1
    assert handler_high.call_count == 1
    # Lower priority (10) should execute before higher priority (50)
    assert execution_order == ["low", "high"]


# -- _log_failures --


def test_log_failures():
    resource = Resource(type="AWS::S3::Bucket", identifier="b1")
    from aws_bench.resource_management.ccapi.models import DeletionFailureEvent

    ResourceCleaner._log_failures({resource: DeletionFailureEvent("err")})


# -- Logging tests --


def test_cleanup_logs_prepared_resources():
    """Test that successful prepare operations are logged."""
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "p1", "AWS::S3::Bucket", "CREATE_COMPLETE")]
    handler_result = HandlerResult("p1", "AWS::S3::Bucket", "prepare", HandlerStatus.SUCCESS)

    mock_log = MagicMock()
    with (
        patch(
            "aws_bench.resource_management.cleanup.resource_cleaner.PREPARE_REGISTRY",
            {"AWS::S3::Bucket": MagicMock(return_value=handler_result)},
        ),
        patch("aws_bench.resource_management.cleanup.resource_cleaner.logger", mock_log),
    ):
        asyncio.run(cleaner.cleanup(resources, prepare=True))

    # Check that prepare success was logged
    info_calls = [call for call in mock_log.debug.call_args_list if "Prepared" in str(call)]
    assert len(info_calls) > 0


def test_cleanup_logs_custom_deleted_resources():
    """Test that successful custom deletions are logged."""
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "p1", "AWS::S3::Bucket", "CREATE_COMPLETE")]
    resolved = [Resource(type="AWS::S3::Bucket", identifier="p1")]
    mock_result = MagicMock(skipped=[], succeeded=[resolved[0]], failed={})

    mock_log = MagicMock()
    with (
        patch.object(cleaner, "_resolve_hooks", return_value=resolved),
        patch.object(cleaner, "_custom_delete", return_value=mock_result),
        patch("aws_bench.resource_management.cleanup.resource_cleaner.logger", mock_log),
    ):
        asyncio.run(cleaner.cleanup(resources, custom_delete=True))

    # Check that custom delete success was logged (per-item detail → DEBUG)
    debug_calls = [call for call in mock_log.debug.call_args_list if "Custom-deleted" in str(call)]
    assert len(debug_calls) > 0


def test_cleanup_logs_ccapi_deleted_resources():
    """Test that successful CCAPI deletions are logged."""
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "p1", "AWS::S3::Bucket", "CREATE_COMPLETE")]
    resolved = [Resource(type="AWS::S3::Bucket", identifier="p1")]
    mock_result = MagicMock(skipped=resolved, succeeded=[], failed={})

    mock_log = MagicMock()
    with (
        patch.object(cleaner, "_resolve_hooks", return_value=resolved),
        patch.object(cleaner, "_custom_delete", return_value=mock_result),
        patch(
            "aws_bench.resource_management.cleanup.resource_cleaner.CloudControlManager"
        ) as mock_ccm_cls,
        patch("aws_bench.resource_management.cleanup.resource_cleaner.logger", mock_log),
    ):
        mock_ccm_cls.return_value.delete_resources.return_value = {}  # No failures = success
        asyncio.run(cleaner.cleanup(resources, custom_delete=True, ccapi_fallback=True))

    # Check that CCAPI delete success was logged (per-item detail → DEBUG)
    debug_calls = [call for call in mock_log.debug.call_args_list if "CCAPI-deleted" in str(call)]
    assert len(debug_calls) > 0


def test_cleanup_logs_stuck_resources():
    """Test that stuck resource handling is logged."""
    cleaner = ResourceCleaner(MagicMock())
    resources = [StackResource("L1", "p1", "AWS::S3::Bucket", "DELETE_FAILED")]

    mock_log = MagicMock()
    with (
        patch.object(cleaner, "_handle_failed"),
        patch("aws_bench.resource_management.cleanup.resource_cleaner.logger", mock_log),
    ):
        asyncio.run(cleaner.cleanup(resources, handle_stuck=True))

    # Check that stuck resource handling was logged
    info_calls = [call for call in mock_log.debug.call_args_list if "stuck" in str(call).lower()]
    assert len(info_calls) > 0


# -- nodegroup-first dependency barrier --


def test_delete_before_prepare_types_contains_nodegroup():
    assert "AWS::EKS::Nodegroup" in DELETE_BEFORE_PREPARE_TYPES


def test_partition_delete_before_prepare_splits_nodegroups():
    nodegroup = StackResource("Ng", "c1|ng1", "AWS::EKS::Nodegroup", "CREATE_COMPLETE")
    bucket = StackResource("B", "b1", "AWS::S3::Bucket", "CREATE_COMPLETE")
    barrier, rest = partition_delete_before_prepare([nodegroup, bucket])
    assert barrier == [nodegroup]
    assert rest == [bucket]


def test_barrier_nodegroup_deletes_before_asg_prepare_starts():
    """The nodegroup custom deletion must complete before any ASG prepare runs."""
    cleaner = ResourceCleaner(MagicMock())
    nodegroup = StackResource("Ng", "c1|ng1", "AWS::EKS::Nodegroup", "CREATE_COMPLETE")
    asg = StackResource("Asg", "asg-1", "AWS::AutoScaling::AutoScalingGroup", "CREATE_COMPLETE")
    resources = [nodegroup, asg]

    order: list[tuple[str, tuple[str, ...]]] = []

    def fake_custom_delete(res, *args, **kwargs):
        order.append(("custom_delete", tuple(r.type for r in res)))
        return CustomDeletionResult(skipped=[], succeeded=list(res), failed={})

    def fake_prepare_all(res, *args, **kwargs):
        order.append(("prepare_all", tuple(r.type for r in res)))
        return []

    with (
        patch.object(cleaner, "_custom_delete", side_effect=fake_custom_delete),
        patch.object(cleaner, "_prepare_all", side_effect=fake_prepare_all),
    ):
        result = asyncio.run(cleaner.cleanup(resources, prepare=True, custom_delete=True))

    assert result == {}
    # First operation is the barrier custom-delete of the nodegroup alone.
    assert order[0] == ("custom_delete", ("AWS::EKS::Nodegroup",))
    first_prepare = next(i for i, (kind, _) in enumerate(order) if kind == "prepare_all")
    assert first_prepare > 0
    # The nodegroup is never routed through prepare, and the ASG is.
    for kind, types in order:
        if kind == "prepare_all":
            assert "AWS::EKS::Nodegroup" not in types
    assert any(
        kind == "prepare_all" and "AWS::AutoScaling::AutoScalingGroup" in types
        for kind, types in order
    )


def test_barrier_success_continues_prepare_custom_and_ccapi():
    """A handled nodegroup is removed; remaining prepare/custom/CCAPI work proceeds."""
    cleaner = ResourceCleaner(MagicMock())
    nodegroup = StackResource("Ng", "c1|ng1", "AWS::EKS::Nodegroup", "CREATE_COMPLETE")
    bucket = StackResource("B", "b1", "AWS::S3::Bucket", "CREATE_COMPLETE")
    resources = [nodegroup, bucket]

    def fake_ng_delete(resource, session):
        return HandlerResult(resource.identifier, resource.type, "delete", HandlerStatus.SUCCESS)

    with (
        patch(
            "aws_bench.resource_management.cleanup.resource_cleaner.CUSTOM_DELETION_REGISTRY",
            {"AWS::EKS::Nodegroup": fake_ng_delete},
        ),
        patch.object(cleaner, "_prepare_all", return_value=[]) as mock_prepare,
        patch(
            "aws_bench.resource_management.cleanup.resource_cleaner.CloudControlManager"
        ) as mock_ccm_cls,
    ):
        mock_ccm_cls.return_value.delete_resources.return_value = {}
        result = asyncio.run(
            cleaner.cleanup(resources, prepare=True, custom_delete=True, ccapi_fallback=True)
        )

    assert result == {}
    # Prepare ran on the remaining resource only — never the nodegroup.
    prepared_types = {r.type for r in mock_prepare.call_args.args[0]}
    assert prepared_types == {"AWS::S3::Bucket"}
    # The unregistered bucket fell through to CCAPI; the nodegroup did not.
    ccapi_types = {r.type for r in mock_ccm_cls.return_value.delete_resources.call_args.args[0]}
    assert "AWS::S3::Bucket" in ccapi_types
    assert "AWS::EKS::Nodegroup" not in ccapi_types


def test_barrier_nodegroup_failure_blocks_prepare_and_ccapi():
    """A failing nodegroup handler fails closed: the failure is returned, nothing else runs."""
    cleaner = ResourceCleaner(MagicMock())
    nodegroup = StackResource("Ng", "c1|ng1", "AWS::EKS::Nodegroup", "CREATE_COMPLETE")
    asg = StackResource("Asg", "asg-1", "AWS::AutoScaling::AutoScalingGroup", "CREATE_COMPLETE")
    resources = [nodegroup, asg]

    def fake_ng_delete(resource, session):
        return HandlerResult(
            resource.identifier, resource.type, "delete", HandlerStatus.FAILED, "drain stuck"
        )

    with (
        patch(
            "aws_bench.resource_management.cleanup.resource_cleaner.CUSTOM_DELETION_REGISTRY",
            {"AWS::EKS::Nodegroup": fake_ng_delete},
        ),
        patch.object(cleaner, "_prepare_all") as mock_prepare,
        patch(
            "aws_bench.resource_management.cleanup.resource_cleaner.CloudControlManager"
        ) as mock_ccm_cls,
    ):
        result = asyncio.run(
            cleaner.cleanup(resources, prepare=True, custom_delete=True, ccapi_fallback=True)
        )

    assert len(result) == 1
    failed = next(iter(result))
    assert failed.type == "AWS::EKS::Nodegroup"
    assert result[failed].status_message == "drain stuck"
    mock_prepare.assert_not_called()
    mock_ccm_cls.return_value.delete_resources.assert_not_called()


def test_barrier_unregistered_nodegroup_handler_fails_closed():
    """No registered handler (skipped by _custom_delete) is a fail-closed failure."""
    cleaner = ResourceCleaner(MagicMock())
    nodegroup = StackResource("Ng", "c1|ng1", "AWS::EKS::Nodegroup", "CREATE_COMPLETE")

    with (
        patch(
            "aws_bench.resource_management.cleanup.resource_cleaner.CUSTOM_DELETION_REGISTRY",
            {},
        ),
        patch.object(cleaner, "_prepare_all") as mock_prepare,
        patch(
            "aws_bench.resource_management.cleanup.resource_cleaner.CloudControlManager"
        ) as mock_ccm_cls,
    ):
        result = asyncio.run(
            cleaner.cleanup([nodegroup], prepare=True, custom_delete=True, ccapi_fallback=True)
        )

    assert len(result) == 1
    failed = next(iter(result))
    assert failed.type == "AWS::EKS::Nodegroup"
    assert "No custom deletion handler registered" in result[failed].status_message
    mock_prepare.assert_not_called()
    mock_ccm_cls.return_value.delete_resources.assert_not_called()


def test_prepare_only_does_not_invoke_barrier():
    """prepare-only never triggers the barrier; the nodegroup flows through prepare unchanged."""
    cleaner = ResourceCleaner(MagicMock())
    nodegroup = StackResource("Ng", "c1|ng1", "AWS::EKS::Nodegroup", "CREATE_COMPLETE")

    with (
        patch.object(cleaner, "_delete_before_prepare") as mock_barrier,
        patch.object(cleaner, "_prepare_all", return_value=[]) as mock_prepare,
    ):
        asyncio.run(cleaner.cleanup([nodegroup], prepare=True))

    mock_barrier.assert_not_called()
    prepared_types = {r.type for r in mock_prepare.call_args.args[0]}
    assert prepared_types == {"AWS::EKS::Nodegroup"}


def test_custom_delete_only_does_not_invoke_barrier():
    """custom-delete-only never triggers the barrier; existing ordering/behavior is retained."""
    cleaner = ResourceCleaner(MagicMock())
    nodegroup = StackResource("Ng", "c1|ng1", "AWS::EKS::Nodegroup", "CREATE_COMPLETE")
    handler = MagicMock(
        return_value=HandlerResult("c1|ng1", "AWS::EKS::Nodegroup", "delete", HandlerStatus.SUCCESS)
    )

    with (
        patch.object(cleaner, "_delete_before_prepare") as mock_barrier,
        patch(
            "aws_bench.resource_management.cleanup.resource_cleaner.CUSTOM_DELETION_REGISTRY",
            {"AWS::EKS::Nodegroup": handler},
        ),
    ):
        result = asyncio.run(cleaner.cleanup([nodegroup], custom_delete=True))

    mock_barrier.assert_not_called()
    handler.assert_called_once()
    assert result == {}
