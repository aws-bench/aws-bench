"""Tests for cross-service cleanup helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from aws_bench.resource_management.cleanup.handlers.cross_service import (
    EniReapResult,
    IpamPoolReapResult,
    VpcPublicAddressWedgeResult,
    _delete_dms_instances_in_vpcs,
    _delete_eks_resource_group,
    _delete_redshift_resources,
    _delete_vpc_endpoints_in_vpcs,
    _eks_sub_resources_gone,
    _force_delete_iam_role,
    _make_cluster_gone_check,
    _prepare_security_group,
    cleanup_redshift,
    cleanup_stuck_custom_resource_deps,
    clear_igw_public_address_wedge,
    clear_vpc_public_address_wedge,
    delete_dms_instances,
    delete_eks_clusters,
    discover_vpc_dynamic_resources,
    drain_sg_eni_references,
    reap_ipam_child_pools,
)
from aws_bench.resource_management.cleanup.handlers.ipam import (
    PoolDeleteOutcome,
    PoolDeleteResult,
)
from aws_bench.resource_management.cleanup.models import StackResource
from aws_bench.resource_management.deferred import deferred_scope

CROSS = "aws_bench.resource_management.cleanup.handlers.cross_service"


def _paginator(pages: list[dict]) -> MagicMock:
    """A mock boto3 paginator whose .paginate() yields the given pages."""
    paginator = MagicMock()
    paginator.paginate.return_value = pages
    return paginator


# -- IAM role --


def test_force_delete_iam_role_detaches_policies_and_deletes():
    iam = MagicMock()

    # Mock paginators
    attached_paginator = MagicMock()
    attached_paginator.paginate.return_value = [
        {"AttachedPolicies": [{"PolicyArn": "arn:aws:iam::123456789012:policy/p1"}]}
    ]

    inline_paginator = MagicMock()
    inline_paginator.paginate.return_value = [{"PolicyNames": ["inline1"]}]

    profiles_paginator = MagicMock()
    profiles_paginator.paginate.return_value = [
        {"InstanceProfiles": [{"InstanceProfileName": "profile1"}]}
    ]

    def get_paginator(operation):
        if operation == "list_attached_role_policies":
            return attached_paginator
        elif operation == "list_role_policies":
            return inline_paginator
        elif operation == "list_instance_profiles_for_role":
            return profiles_paginator

    iam.get_paginator.side_effect = get_paginator

    _force_delete_iam_role(iam, "test-role")

    iam.detach_role_policy.assert_called_once()
    iam.delete_role_policy.assert_called_once_with(RoleName="test-role", PolicyName="inline1")
    iam.remove_role_from_instance_profile.assert_called_once_with(
        RoleName="test-role", InstanceProfileName="profile1"
    )
    iam.delete_role.assert_called_once_with(RoleName="test-role")


def test_force_delete_iam_role_handles_error_gracefully():
    iam = MagicMock()
    iam.get_paginator.side_effect = Exception("no such role")
    _force_delete_iam_role(iam, "gone-role")


# -- Stuck custom resource deps --


def test_cleanup_stuck_deletes_lambda_function():
    session = MagicMock()
    lam_client = MagicMock()
    iam_client = MagicMock()

    def client_factory(svc, **kwargs):
        if svc == "lambda":
            return lam_client
        return iam_client

    session.client.side_effect = client_factory

    resources = [
        StackResource(
            logical_id="Fn", physical_id="my-fn", resource_type="AWS::Lambda::Function", status=""
        )
    ]
    cleanup_stuck_custom_resource_deps(session, resources, "us-east-1")
    lam_client.delete_function.assert_called_once_with(FunctionName="my-fn")


def test_cleanup_stuck_skips_empty_physical_id():
    session = MagicMock()
    lam_client = MagicMock()
    iam_client = MagicMock()
    session.client.side_effect = lambda svc, **kw: lam_client if svc == "lambda" else iam_client

    resources = [
        StackResource(
            logical_id="Fn", physical_id="", resource_type="AWS::Lambda::Function", status=""
        )
    ]
    cleanup_stuck_custom_resource_deps(session, resources, "us-east-1")
    lam_client.delete_function.assert_not_called()


def test_cleanup_stuck_handles_client_creation_failure():
    session = MagicMock()
    session.client.side_effect = Exception("no endpoint")
    cleanup_stuck_custom_resource_deps(
        session,
        [
            StackResource(
                logical_id="Fn", physical_id="fn", resource_type="AWS::Lambda::Function", status=""
            )
        ],
        "us-east-1",
    )


def test_cleanup_stuck_handles_lambda_delete_error():
    session = MagicMock()
    lam = MagicMock()
    iam = MagicMock()
    session.client.side_effect = lambda svc, **kw: lam if svc == "lambda" else iam
    lam.delete_function.side_effect = Exception("fail")
    cleanup_stuck_custom_resource_deps(
        session,
        [
            StackResource(
                logical_id="Fn", physical_id="fn", resource_type="AWS::Lambda::Function", status=""
            )
        ],
        "us-east-1",
    )


def test_cleanup_stuck_deletes_iam_role():
    session = MagicMock()
    lam = MagicMock()
    iam = MagicMock()
    session.client.side_effect = lambda svc, **kw: lam if svc == "lambda" else iam
    iam.list_attached_role_policies.return_value = {"AttachedPolicies": []}
    iam.list_role_policies.return_value = {"PolicyNames": []}
    iam.list_instance_profiles_for_role.return_value = {"InstanceProfiles": []}
    cleanup_stuck_custom_resource_deps(
        session,
        [
            StackResource(
                logical_id="R", physical_id="role1", resource_type="AWS::IAM::Role", status=""
            )
        ],
        "us-east-1",
    )
    iam.delete_role.assert_called_once()


# -- EKS --


def test_delete_eks_clusters_deletes_cluster():
    session = MagicMock()
    eks = MagicMock()
    session.client.return_value = eks
    eks.exceptions.ResourceNotFoundException = type("ResourceNotFoundException", (Exception,), {})
    eks.list_nodegroups.return_value = {"nodegroups": []}
    eks.list_fargate_profiles.return_value = {"fargateProfileNames": []}
    # Cluster gone on first poll (describe_cluster raises NotFound).
    eks.describe_cluster.side_effect = eks.exceptions.ResourceNotFoundException()
    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.wait_until",
        return_value=True,
    ):
        unresolved = delete_eks_clusters(session, {"cluster1"})
    eks.delete_cluster.assert_called_once()
    # Confirmed gone -> nothing unresolved.
    assert unresolved == set()


def test_delete_eks_clusters_handles_not_found():
    session = MagicMock()
    eks = MagicMock()
    session.client.return_value = eks
    eks.exceptions.ResourceNotFoundException = type("ResourceNotFoundException", (Exception,), {})
    eks.list_nodegroups.return_value = {"nodegroups": []}
    eks.list_fargate_profiles.return_value = {"fargateProfileNames": []}
    eks.delete_cluster.side_effect = eks.exceptions.ResourceNotFoundException()
    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.wait_until",
        return_value=True,
    ):
        delete_eks_clusters(session, {"cluster1"})


def test_delete_eks_clusters_handles_generic_error():
    session = MagicMock()
    eks = MagicMock()
    session.client.return_value = eks
    eks.exceptions.ResourceNotFoundException = type("ResourceNotFoundException", (Exception,), {})
    # A sub-resource list error is swallowed internally (logged, treated as empty),
    # so the cluster delete still submits and, confirmed gone, resolves cleanly.
    eks.list_nodegroups.side_effect = Exception("fail")
    eks.describe_cluster.side_effect = eks.exceptions.ResourceNotFoundException()
    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.wait_until",
        return_value=True,
    ):
        unresolved = delete_eks_clusters(session, {"cluster1"})
    eks.delete_cluster.assert_called_once()
    assert unresolved == set()


def test_delete_eks_clusters_timeout():
    session = MagicMock()
    eks = MagicMock()
    session.client.return_value = eks
    eks.exceptions.ResourceNotFoundException = type("ResourceNotFoundException", (Exception,), {})
    eks.list_nodegroups.return_value = {"nodegroups": []}
    eks.list_fargate_profiles.return_value = {"fargateProfileNames": []}
    # Cluster never disappears, so the poll runs then hits the deadline.
    eks.describe_cluster.return_value = {"cluster": {"status": "DELETING"}}
    # monotonic ticks: deadline calc, one loop pass, then past-deadline to exit.
    cross = "aws_bench.resource_management.cleanup.handlers.cross_service"
    with (
        patch(f"{cross}.wait_until", return_value=True),
        patch(f"{cross}.time.monotonic", side_effect=[0.0, 1.0, 10_000.0]),
        patch(f"{cross}.time.sleep") as mock_sleep,
    ):
        unresolved = delete_eks_clusters(session, {"cluster1"})
    # Poll body ran, and the never-gone cluster is reported as timed out.
    eks.describe_cluster.assert_called()
    mock_sleep.assert_called_once()
    assert unresolved == {"cluster1"}


def test_delete_eks_clusters_final_sweep_catches_late_deletion():
    """A cluster that becomes gone by the final sweep is not mislabeled as timed out."""
    session = MagicMock()
    eks = MagicMock()
    session.client.return_value = eks
    eks.exceptions.ResourceNotFoundException = type("ResourceNotFoundException", (Exception,), {})
    eks.list_nodegroups.return_value = {"nodegroups": []}
    eks.list_fargate_profiles.return_value = {"fargateProfileNames": []}
    # Still present during the in-loop poll, gone by the post-loop final sweep.
    eks.describe_cluster.side_effect = [
        {"cluster": {"status": "DELETING"}},
        eks.exceptions.ResourceNotFoundException(),
    ]
    cross = "aws_bench.resource_management.cleanup.handlers.cross_service"
    with (
        patch(f"{cross}.wait_until", return_value=True),
        patch(f"{cross}.time.monotonic", side_effect=[0.0, 1.0, 10_000.0]),
        patch(f"{cross}.time.sleep"),
    ):
        unresolved = delete_eks_clusters(session, {"cluster1"})
    # Final sweep saw it gone -> not reported as unresolved.
    assert unresolved == set()


def test_delete_eks_clusters_continues_on_generic_error():
    session = MagicMock()
    eks = MagicMock()
    session.client.return_value = eks
    eks.exceptions.ResourceNotFoundException = type("ResourceNotFoundException", (Exception,), {})
    eks.list_nodegroups.return_value = {"nodegroups": []}
    eks.list_fargate_profiles.return_value = {"fargateProfileNames": []}
    eks.delete_cluster.side_effect = RuntimeError("generic")
    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.wait_until",
        return_value=True,
    ):
        unresolved = delete_eks_clusters(session, {"c1", "c2"})
    # Both submits failed -> both reported unresolved.
    assert unresolved == {"c1", "c2"}


def test_delete_eks_resource_group_deletes_and_waits():
    eks = MagicMock()
    eks.list_nodegroups.return_value = {"nodegroups": ["ng1"]}
    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.wait_until",
        return_value=True,
    ):
        _delete_eks_resource_group(
            eks,
            "c1",
            "list_nodegroups",
            "nodegroups",
            "delete_nodegroup",
            "nodegroupName",
            "nodegroup",
            timeout=10,
        )
    eks.delete_nodegroup.assert_called_once()


def test_delete_eks_resource_group_handles_list_error():
    eks = MagicMock()
    eks.list_nodegroups.side_effect = Exception("fail")
    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.wait_until",
        return_value=True,
    ):
        _delete_eks_resource_group(
            eks,
            "c1",
            "list_nodegroups",
            "nodegroups",
            "delete_nodegroup",
            "nodegroupName",
            "nodegroup",
            timeout=10,
        )


def test_delete_eks_resource_group_handles_delete_error():
    eks = MagicMock()
    eks.list_nodegroups.return_value = {"nodegroups": ["ng1"]}
    eks.delete_nodegroup.side_effect = Exception("fail")
    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.wait_until",
        return_value=True,
    ):
        _delete_eks_resource_group(
            eks,
            "c1",
            "list_nodegroups",
            "nodegroups",
            "delete_nodegroup",
            "nodegroupName",
            "nodegroup",
            timeout=10,
        )


def test_eks_sub_resources_gone_true_when_empty():
    eks = MagicMock()
    eks.list_nodegroups.return_value = {"nodegroups": []}
    assert _eks_sub_resources_gone(eks, "c1", "list_nodegroups", "nodegroups") is True


def test_eks_sub_resources_gone_false_when_present():
    eks = MagicMock()
    eks.list_nodegroups.return_value = {"nodegroups": ["ng1"]}
    assert _eks_sub_resources_gone(eks, "c1", "list_nodegroups", "nodegroups") is False


def test_eks_sub_resources_gone_true_on_not_found():
    eks = MagicMock()
    eks.exceptions.ResourceNotFoundException = type("ResourceNotFoundException", (Exception,), {})
    eks.list_nodegroups.side_effect = eks.exceptions.ResourceNotFoundException()
    assert _eks_sub_resources_gone(eks, "c1", "list_nodegroups", "nodegroups") is True


def test_eks_sub_resources_gone_false_on_error():
    eks = MagicMock()
    eks.exceptions.ResourceNotFoundException = type("ResourceNotFoundException", (Exception,), {})
    eks.list_nodegroups.side_effect = RuntimeError("fail")
    assert _eks_sub_resources_gone(eks, "c1", "list_nodegroups", "nodegroups") is False


def test_make_cluster_gone_check_true_when_not_found():
    eks = MagicMock()
    eks.exceptions.ResourceNotFoundException = type("ResourceNotFoundException", (Exception,), {})
    eks.describe_cluster.side_effect = eks.exceptions.ResourceNotFoundException()
    assert _make_cluster_gone_check(eks, "c1")() is True


def test_make_cluster_gone_check_false_when_exists():
    eks = MagicMock()
    eks.describe_cluster.return_value = {}
    assert _make_cluster_gone_check(eks, "c1")() is False


def test_make_cluster_gone_check_false_on_error():
    eks = MagicMock()
    eks.exceptions.ResourceNotFoundException = type("ResourceNotFoundException", (Exception,), {})
    eks.describe_cluster.side_effect = RuntimeError("fail")
    assert _make_cluster_gone_check(eks, "c1")() is False


# -- DMS replication instances --


def test_delete_dms_instances_deletes_and_confirms_gone():
    session = MagicMock()
    dms = MagicMock()
    session.client.return_value = dms
    arn = "arn:aws:dms:us-east-1:123456789012:rep:ABC"
    # Gone on the first poll (describe returns no instances), so the wait loop
    # exits on its own — no timing patch needed.
    dms.describe_replication_instances.return_value = {"ReplicationInstances": []}
    unresolved = delete_dms_instances(session, {arn})
    dms.delete_replication_instance.assert_called_once_with(ReplicationInstanceArn=arn)
    assert unresolved == set()


def test_delete_dms_instances_not_found_is_gone_without_polling():
    """ResourceNotFoundFault means the instance is already gone: not polled, resolved."""
    session = MagicMock()
    dms = MagicMock()
    session.client.return_value = dms
    dms.delete_replication_instance.side_effect = ClientError(
        {"Error": {"Code": "ResourceNotFoundFault"}}, "DeleteReplicationInstance"
    )
    arn = "arn:aws:dms:us-east-1:123456789012:rep:GONE"
    unresolved = delete_dms_instances(session, {arn})
    assert unresolved == set()
    # Already gone -> no need to poll its existence.
    dms.describe_replication_instances.assert_not_called()


def test_delete_dms_instances_invalid_state_is_polled_not_assumed_gone():
    """An instance DMS refuses to delete is polled, not assumed gone.

    InvalidResourceStateFault means the instance still EXISTS (mid-op / task
    attached), so it must be polled — if it persists it surfaces as unresolved,
    never a silent success (the wedge this code exists to prevent).
    """
    arn = "arn:aws:dms:us-east-1:123456789012:rep:STUCK"
    cross = "aws_bench.resource_management.cleanup.handlers.cross_service"

    # (a) still present on poll -> reported unresolved (not a silent success).
    session = MagicMock()
    dms = MagicMock()
    session.client.return_value = dms
    dms.delete_replication_instance.side_effect = ClientError(
        {"Error": {"Code": "InvalidResourceStateFault"}}, "DeleteReplicationInstance"
    )
    dms.describe_replication_instances.return_value = {
        "ReplicationInstances": [{"ReplicationInstanceArn": arn}]
    }
    with patch(f"{cross}._DMS_INSTANCE_TIMEOUT", 0), patch(f"{cross}.time.sleep"):
        unresolved = delete_dms_instances(session, {arn})
    assert unresolved == {arn}
    dms.describe_replication_instances.assert_called()  # it WAS polled

    # (b) merely already-deleting -> gone on poll -> resolved.
    session = MagicMock()
    dms = MagicMock()
    session.client.return_value = dms
    dms.delete_replication_instance.side_effect = ClientError(
        {"Error": {"Code": "InvalidResourceStateFault"}}, "DeleteReplicationInstance"
    )
    dms.describe_replication_instances.return_value = {"ReplicationInstances": []}
    unresolved = delete_dms_instances(session, {arn})
    assert unresolved == set()


def test_delete_dms_instances_reports_submit_failure_as_unresolved():
    session = MagicMock()
    dms = MagicMock()
    session.client.return_value = dms
    dms.exceptions.ResourceNotFoundFault = type("ResourceNotFoundFault", (Exception,), {})
    dms.delete_replication_instance.side_effect = RuntimeError("boom")
    arn = "arn:aws:dms:us-east-1:123456789012:rep:FAIL"
    unresolved = delete_dms_instances(session, {arn})
    assert unresolved == {arn}


def test_delete_dms_instances_timeout_reports_unresolved():
    session = MagicMock()
    dms = MagicMock()
    session.client.return_value = dms
    arn = "arn:aws:dms:us-east-1:123456789012:rep:SLOW"
    # Never disappears: describe keeps returning the instance, so the wait loop
    # hits its deadline. Use a zero timeout so it times out immediately without
    # coupling the test to internal clock-call counts.
    dms.describe_replication_instances.return_value = {
        "ReplicationInstances": [
            {"ReplicationInstanceArn": arn, "ReplicationInstanceStatus": "deleting"}
        ]
    }
    cross = "aws_bench.resource_management.cleanup.handlers.cross_service"
    with (
        patch(f"{cross}._DMS_INSTANCE_TIMEOUT", 0),
        patch(f"{cross}.time.sleep"),
    ):
        unresolved = delete_dms_instances(session, {arn})
    assert unresolved == {arn}


def test_delete_dms_instances_in_vpcs_finds_and_deletes():
    session = MagicMock()
    dms = MagicMock()
    ec2 = MagicMock()
    session.client.side_effect = lambda svc: dms if svc == "dms" else ec2
    # The target VPC currently contains subnet-a (subnet-z belongs to another VPC).
    # Resolved by a vpc-id-filtered paginator, so a sibling subnet already deleted
    # mid-teardown can't poison the lookup.
    ec2.get_paginator.return_value.paginate.return_value = [{"Subnets": [{"SubnetId": "subnet-a"}]}]
    # Two subnet groups; only sg-in-vpc has a subnet in the target VPC.
    dms.get_paginator.side_effect = lambda op: {
        "describe_replication_subnet_groups": _paginator(
            [
                {
                    "ReplicationSubnetGroups": [
                        {
                            "ReplicationSubnetGroupIdentifier": "sg-in-vpc",
                            "Subnets": [{"SubnetIdentifier": "subnet-a"}],
                        },
                        {
                            "ReplicationSubnetGroupIdentifier": "sg-other",
                            "Subnets": [{"SubnetIdentifier": "subnet-z"}],
                        },
                    ]
                }
            ]
        ),
        "describe_replication_instances": _paginator(
            [
                {
                    "ReplicationInstances": [
                        {
                            "ReplicationInstanceArn": "arn:in-vpc",
                            "ReplicationSubnetGroup": {
                                "ReplicationSubnetGroupIdentifier": "sg-in-vpc"
                            },
                        },
                        {
                            "ReplicationInstanceArn": "arn:other",
                            "ReplicationSubnetGroup": {
                                "ReplicationSubnetGroupIdentifier": "sg-other"
                            },
                        },
                    ]
                }
            ]
        ),
    }[op]
    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.delete_dms_instances"
    ) as mock_del:
        _delete_dms_instances_in_vpcs(session, ["vpc-1"])
    mock_del.assert_called_once_with(session, {"arn:in-vpc"})


def test_delete_dms_instances_in_vpcs_noop_when_no_instances_in_vpc():
    session = MagicMock()
    dms = MagicMock()
    ec2 = MagicMock()
    session.client.side_effect = lambda svc: dms if svc == "dms" else ec2
    # The target VPC has a subnet, but no replication subnet group uses it.
    ec2.get_paginator.return_value.paginate.return_value = [{"Subnets": [{"SubnetId": "subnet-a"}]}]
    dms.get_paginator.return_value.paginate.return_value = [
        {
            "ReplicationSubnetGroups": [
                {
                    "ReplicationSubnetGroupIdentifier": "sg-other",
                    "Subnets": [{"SubnetIdentifier": "subnet-z"}],
                }
            ]
        }
    ]
    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.delete_dms_instances"
    ) as mock_del:
        _delete_dms_instances_in_vpcs(session, ["vpc-1"])
    mock_del.assert_not_called()


def test_delete_dms_instances_in_vpcs_handles_error():
    session = MagicMock()
    session.client.return_value.get_paginator.side_effect = Exception("dms not available")
    # Must not raise — a discovery failure is logged and swallowed.
    _delete_dms_instances_in_vpcs(session, ["vpc-1"])


# -- VPC endpoints --


def test_delete_vpc_endpoints_deletes_all_types_in_vpc():
    ec2 = MagicMock()
    session = MagicMock()
    session.client.return_value = ec2
    # Both interface (pins subnet via ENI) and gateway (pins VPC via route-table)
    # endpoints in the VPC are returned by the vpc-id-filtered describe and deleted.
    ec2.get_paginator.return_value.paginate.return_value = [
        {
            "VpcEndpoints": [
                {"VpcEndpointId": "vpce-iface", "VpcEndpointType": "Interface", "VpcId": "vpc-1"},
                {"VpcEndpointId": "vpce-gw", "VpcEndpointType": "Gateway", "VpcId": "vpc-1"},
            ]
        }
    ]
    ec2.delete_vpc_endpoints.return_value = {"Unsuccessful": []}
    # Poll: endpoints report gone immediately.
    ec2.describe_vpc_endpoints.return_value = {"VpcEndpoints": []}
    _delete_vpc_endpoints_in_vpcs(session, ["vpc-1"])
    ec2.delete_vpc_endpoints.assert_called_once_with(VpcEndpointIds=["vpce-iface", "vpce-gw"])


def test_delete_vpc_endpoints_noop_when_none_in_vpc():
    ec2 = MagicMock()
    session = MagicMock()
    session.client.return_value = ec2
    # vpc-id-filtered describe returns nothing for the target VPC.
    ec2.get_paginator.return_value.paginate.return_value = [{"VpcEndpoints": []}]
    _delete_vpc_endpoints_in_vpcs(session, ["vpc-1"])
    ec2.delete_vpc_endpoints.assert_not_called()


def test_delete_vpc_endpoints_surfaces_partial_failure(caplog):
    """A per-endpoint delete failure is surfaced, not silently swallowed.

    delete_vpc_endpoints does not raise on a per-endpoint failure; it returns it
    in Unsuccessful. That must be logged, not silently leave a pinned subnet.
    """
    import logging

    ec2 = MagicMock()
    session = MagicMock()
    session.client.return_value = ec2
    ec2.get_paginator.return_value.paginate.return_value = [
        {"VpcEndpoints": [{"VpcEndpointId": "vpce-stuck", "VpcEndpointType": "Interface"}]}
    ]
    ec2.delete_vpc_endpoints.return_value = {
        "Unsuccessful": [
            {
                "ResourceId": "vpce-stuck",
                "Error": {"Code": "DependencyViolation", "Message": "in use"},
            }
        ]
    }
    ec2.describe_vpc_endpoints.return_value = {"VpcEndpoints": []}
    with caplog.at_level(logging.WARNING):
        _delete_vpc_endpoints_in_vpcs(session, ["vpc-1"])
    assert "vpce-stuck" in caplog.text
    assert "DependencyViolation" in caplog.text


def test_delete_vpc_endpoints_handles_error():
    session = MagicMock()
    session.client.return_value.get_paginator.side_effect = Exception("boom")
    # Must not raise — discovery failure is logged and swallowed.
    _delete_vpc_endpoints_in_vpcs(session, ["vpc-1"])


def test_vpc_endpoint_gone_matches_capitalized_deleted_state():
    """The gone-check matches the 'Deleted' state case-insensitively.

    EC2 reports the state as 'Deleted' (capitalized), so a lingering deleted
    endpoint must be recognized as gone and exit at once.
    """
    from aws_bench.resource_management.cleanup.handlers.cross_service import _vpc_endpoint_gone

    ec2 = MagicMock()
    ec2.describe_vpc_endpoints.return_value = {
        "VpcEndpoints": [{"VpcEndpointId": "vpce-x", "State": "Deleted"}]
    }
    assert _vpc_endpoint_gone(ec2, "vpce-x") is True
    # A still-live endpoint is not gone.
    ec2.describe_vpc_endpoints.return_value = {
        "VpcEndpoints": [{"VpcEndpointId": "vpce-x", "State": "Available"}]
    }
    assert _vpc_endpoint_gone(ec2, "vpce-x") is False


def test_sweep_gone_swallows_raising_predicate():
    """A raising is_gone is treated as 'not gone yet', not fatal.

    One id's transient error must not abort the poll for the rest of the set.
    """
    from aws_bench.resource_management.cleanup.handlers.cross_service import _sweep_gone

    remaining = {"a", "b"}

    def is_gone(id_: str) -> bool:
        if id_ == "a":
            raise RuntimeError("transient")
        return True  # "b" is gone

    _sweep_gone(remaining, is_gone, "thing")
    # "b" swept; "a" retained (its error was swallowed, not fatal).
    assert remaining == {"a"}


# -- Redshift --


def test_cleanup_redshift_deletes_workgroups_and_namespaces():
    session = MagicMock()
    rs = MagicMock()
    session.client.return_value = rs
    rs.list_workgroups.return_value = {"workgroups": [{"workgroupName": "wg1"}]}
    rs.list_namespaces.return_value = {"namespaces": [{"namespaceName": "ns1"}]}

    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.wait_until",
        return_value=True,
    ):
        cleanup_redshift(session, "us-east-1")

    # The redshift-serverless client is built for the stack's region only.
    session.client.assert_called_once_with("redshift-serverless", region_name="us-east-1")
    rs.delete_workgroup.assert_called_once()
    rs.delete_namespace.assert_called_once()


def test_cleanup_redshift_handles_unavailable_region(caplog):
    import logging

    from botocore.exceptions import EndpointConnectionError

    session = MagicMock()
    rs = MagicMock()
    session.client.return_value = rs
    # Region doesn't offer redshift-serverless — expected, swallowed at debug.
    rs.list_workgroups.side_effect = EndpointConnectionError(endpoint_url="https://x")
    with (
        patch(
            "aws_bench.resource_management.cleanup.handlers.cross_service.wait_until",
            return_value=True,
        ),
        caplog.at_level(logging.WARNING),
    ):
        cleanup_redshift(session, "eu-west-1")
    # An unavailable region must NOT log a warning.
    assert "cleanup failed" not in caplog.text


def test_cleanup_redshift_surfaces_real_fault_as_warning(caplog):
    """A genuine API fault (throttle/access-denied) is logged at warning, not hidden at debug."""
    import logging

    session = MagicMock()
    rs = MagicMock()
    session.client.return_value = rs
    rs.list_workgroups.side_effect = Exception("ThrottlingException")
    with (
        patch(
            "aws_bench.resource_management.cleanup.handlers.cross_service.wait_until",
            return_value=True,
        ),
        caplog.at_level(logging.WARNING),
    ):
        cleanup_redshift(session, "us-east-1")
    assert "Redshift Serverless cleanup failed in us-east-1" in caplog.text


def test_delete_redshift_resources_deletes():
    client = MagicMock()
    client.list_workgroups.return_value = {"workgroups": [{"workgroupName": "wg1"}]}
    _delete_redshift_resources(
        client,
        "us-east-1",
        "list_workgroups",
        "workgroups",
        "workgroupName",
        "delete_workgroup",
        "workgroup",
    )
    client.delete_workgroup.assert_called_once()


def test_delete_redshift_resources_handles_delete_error():
    client = MagicMock()
    client.list_workgroups.return_value = {"workgroups": [{"workgroupName": "wg1"}]}
    client.delete_workgroup.side_effect = Exception("fail")
    _delete_redshift_resources(
        client,
        "us-east-1",
        "list_workgroups",
        "workgroups",
        "workgroupName",
        "delete_workgroup",
        "workgroup",
    )


def test_delete_redshift_resources_skips_malformed_item_and_continues():
    """A malformed item (missing name key) is skipped, not fatal — the rest still delete."""
    client = MagicMock()
    client.list_workgroups.return_value = {
        "workgroups": [{"workgroupName": "wg1"}, {"wrong": "shape"}, {"workgroupName": "wg2"}]
    }
    _delete_redshift_resources(
        client,
        "us-east-1",
        "list_workgroups",
        "workgroups",
        "workgroupName",
        "delete_workgroup",
        "workgroup",
    )
    # Both well-formed items were deleted; the malformed one didn't abort the batch.
    assert client.delete_workgroup.call_count == 2
    client.delete_workgroup.assert_any_call(workgroupName="wg1")
    client.delete_workgroup.assert_any_call(workgroupName="wg2")


# -- VPC dynamic resource discovery --


def test_delete_eks_clusters_in_vpcs_finds_and_deletes():
    from aws_bench.resource_management.cleanup.handlers.cross_service import (
        _delete_eks_clusters_in_vpcs,
    )

    session = MagicMock()
    eks = MagicMock()
    session.client.return_value = eks
    eks.list_clusters.return_value = {"clusters": ["c1", "c2"]}
    eks.describe_cluster.side_effect = [
        {"cluster": {"resourcesVpcConfig": {"vpcId": "vpc-1"}}},
        {"cluster": {"resourcesVpcConfig": {"vpcId": "vpc-other"}}},
    ]
    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.delete_eks_clusters"
    ) as mock_del:
        _delete_eks_clusters_in_vpcs(session, ["vpc-1"])
    mock_del.assert_called_once_with(session, {"c1"})


def test_delete_eks_clusters_in_vpcs_handles_error():
    from aws_bench.resource_management.cleanup.handlers.cross_service import (
        _delete_eks_clusters_in_vpcs,
    )

    session = MagicMock()
    session.client.return_value.list_clusters.side_effect = Exception("fail")
    _delete_eks_clusters_in_vpcs(session, ["vpc-1"])


def test_discover_efs_mount_targets_finds_targets():
    from aws_bench.resource_management.cleanup.handlers.cross_service import (
        _discover_efs_mount_targets,
    )

    session = MagicMock()
    ec2 = MagicMock()
    efs = MagicMock()
    session.client.side_effect = lambda svc: efs if svc == "efs" else ec2
    ec2.get_paginator.return_value.paginate.return_value = [{"Subnets": [{"SubnetId": "subnet-1"}]}]
    efs.get_paginator.return_value.paginate.return_value = [
        {"FileSystems": [{"FileSystemId": "fs-1"}]}
    ]
    efs.describe_mount_targets.return_value = {
        "MountTargets": [{"MountTargetId": "fsmt-1", "SubnetId": "subnet-1"}]
    }
    result = _discover_efs_mount_targets(session, ["vpc-1"])
    assert len(result) == 1
    assert result[0].type == "AWS::EFS::MountTarget"
    assert result[0].identifier == "fsmt-1"


def test_discover_efs_mount_targets_skips_other_subnets():
    from aws_bench.resource_management.cleanup.handlers.cross_service import (
        _discover_efs_mount_targets,
    )

    session = MagicMock()
    ec2 = MagicMock()
    efs = MagicMock()
    session.client.side_effect = lambda svc: efs if svc == "efs" else ec2
    ec2.get_paginator.return_value.paginate.return_value = [{"Subnets": [{"SubnetId": "subnet-1"}]}]
    efs.get_paginator.return_value.paginate.return_value = [
        {"FileSystems": [{"FileSystemId": "fs-1"}]}
    ]
    efs.describe_mount_targets.return_value = {
        "MountTargets": [{"MountTargetId": "fsmt-1", "SubnetId": "subnet-other"}]
    }
    assert _discover_efs_mount_targets(session, ["vpc-1"]) == []


def test_discover_efs_mount_targets_handles_error():
    from aws_bench.resource_management.cleanup.handlers.cross_service import (
        _discover_efs_mount_targets,
    )

    session = MagicMock()
    session.client.side_effect = Exception("fail")
    assert _discover_efs_mount_targets(session, ["vpc-1"]) == []


def test_discover_security_groups_filters_default():
    from aws_bench.resource_management.cleanup.handlers.cross_service import (
        _discover_security_groups,
    )

    ec2 = MagicMock()
    ec2.get_paginator.return_value.paginate.return_value = [
        {
            "SecurityGroups": [
                {"GroupId": "sg-1", "GroupName": "default"},
                {"GroupId": "sg-2", "GroupName": "my-sg"},
            ]
        }
    ]
    result = _discover_security_groups(ec2, ["vpc-1"])
    assert len(result) == 1
    assert result[0].identifier == "sg-2"


def test_discover_security_groups_handles_error():
    from aws_bench.resource_management.cleanup.handlers.cross_service import (
        _discover_security_groups,
    )

    ec2 = MagicMock()
    ec2.get_paginator.return_value.paginate.side_effect = Exception("fail")
    assert _discover_security_groups(ec2, ["vpc-1"]) == []


def test_discover_security_groups_revokes_circular_rules_before_returning():
    """Circular EFS NFS SGs: each SG's ingress+egress rules are revoked before delete.

    Two mount-target SGs reference each other (inbound-nfs ingress -> outbound-nfs SG,
    outbound-nfs egress -> inbound-nfs SG). Neither can be deleted until the other's
    referencing rule is gone, so discovery must revoke both SGs' rules first to break
    the cycle. The default SG is never touched.
    """
    from aws_bench.resource_management.cleanup.handlers.cross_service import (
        _discover_security_groups,
    )

    inbound = {
        "GroupId": "sg-inbound",
        "GroupName": "security-group-for-inbound-nfs-d-vvpgghrq38so",
        "IpPermissions": [{"IpProtocol": "tcp", "UserIdGroupPairs": [{"GroupId": "sg-outbound"}]}],
        "IpPermissionsEgress": [],
    }
    outbound = {
        "GroupId": "sg-outbound",
        "GroupName": "security-group-for-outbound-nfs-d-vvpgghrq38so",
        "IpPermissions": [],
        "IpPermissionsEgress": [
            {"IpProtocol": "tcp", "UserIdGroupPairs": [{"GroupId": "sg-inbound"}]}
        ],
    }
    default_sg = {
        "GroupId": "sg-default",
        "GroupName": "default",
        "IpPermissions": [{"IpProtocol": "-1", "UserIdGroupPairs": [{"GroupId": "sg-default"}]}],
        "IpPermissionsEgress": [],
    }

    ec2 = MagicMock()
    ec2.get_paginator.return_value.paginate.return_value = [
        {"SecurityGroups": [inbound, outbound, default_sg]}
    ]

    result = _discover_security_groups(ec2, ["vpc-1"])

    # Both non-default SGs are returned for deletion; default filtered out.
    assert {r.identifier for r in result} == {"sg-inbound", "sg-outbound"}
    # The inbound SG's ingress rule was revoked (breaks the cycle).
    ec2.revoke_security_group_ingress.assert_any_call(
        GroupId="sg-inbound", IpPermissions=inbound["IpPermissions"]
    )
    # The outbound SG's egress rule was revoked.
    ec2.revoke_security_group_egress.assert_any_call(
        GroupId="sg-outbound", IpPermissions=outbound["IpPermissionsEgress"]
    )
    # The default SG's rules are never revoked.
    for call in ec2.revoke_security_group_ingress.call_args_list:
        assert call.kwargs.get("GroupId") != "sg-default"


def test_discover_security_groups_skips_revoke_when_no_rules():
    """An SG with no rules triggers no revoke calls (nothing to break)."""
    from aws_bench.resource_management.cleanup.handlers.cross_service import (
        _discover_security_groups,
    )

    ec2 = MagicMock()
    ec2.get_paginator.return_value.paginate.return_value = [
        {"SecurityGroups": [{"GroupId": "sg-2", "GroupName": "my-sg"}]}
    ]
    result = _discover_security_groups(ec2, ["vpc-1"])
    assert [r.identifier for r in result] == ["sg-2"]
    ec2.revoke_security_group_ingress.assert_not_called()
    ec2.revoke_security_group_egress.assert_not_called()


def test_revoke_sg_rules_swallows_errors():
    """A revoke failure on one SG is logged, not raised, and does not block the rest."""
    from aws_bench.resource_management.cleanup.handlers.cross_service import _revoke_sg_rules

    ec2 = MagicMock()
    ec2.revoke_security_group_ingress.side_effect = Exception("boom")
    sgs = [
        {
            "GroupId": "sg-a",
            "IpPermissions": [{"IpProtocol": "-1"}],
            "IpPermissionsEgress": [{"IpProtocol": "-1"}],
        }
    ]
    # Must not raise despite the ingress revoke failing.
    _revoke_sg_rules(ec2, sgs)
    # Egress revoke still attempted even though ingress failed.
    ec2.revoke_security_group_egress.assert_called_once_with(
        GroupId="sg-a", IpPermissions=[{"IpProtocol": "-1"}]
    )


# -- Failed resource handlers --


def test_handle_stuck_custom_resources():
    from aws_bench.resource_management.cleanup.handlers.cross_service import (
        _handle_stuck_custom_resources,
    )

    session = MagicMock()
    lam = MagicMock()
    iam = MagicMock()
    session.client.side_effect = lambda svc, **kw: lam if svc == "lambda" else iam
    failed = []
    all_resources = [
        MagicMock(
            resource_type="AWS::Lambda::Function",
            physical_id="fn1",
            status="DELETE_FAILED",
        )
    ]  # type: ignore[list-item]
    # region is threaded from the stack's cleanup, not guessed from the session.
    _handle_stuck_custom_resources(failed, all_resources, session, "eu-west-1")  # type: ignore[arg-type]
    lam.delete_function.assert_called_once()
    session.client.assert_any_call("lambda", region_name="eu-west-1")


def test_handle_stuck_custom_aws_resources_cleans_only_stack_region():
    from aws_bench.resource_management.cleanup.handlers.cross_service import (
        _handle_stuck_custom_aws_resources,
    )

    session = MagicMock()
    rs = MagicMock()
    rs.list_workgroups.return_value = {"workgroups": []}
    rs.list_namespaces.return_value = {"namespaces": []}
    session.client.return_value = rs

    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.wait_until",
        return_value=True,
    ):
        _handle_stuck_custom_aws_resources([], [], session, "ap-south-1")

    # Redshift cleanup targets the stack's region only — no all-region scan.
    session.client.assert_called_once_with("redshift-serverless", region_name="ap-south-1")


def test_cleanup_stuck_custom_resource_deps_unsupported_type():
    """Test cleanup_stuck_custom_resource_deps skips unsupported resource types."""
    from aws_bench.resource_management.cleanup.handlers.cross_service import (
        cleanup_stuck_custom_resource_deps,
    )
    from aws_bench.resource_management.cleanup.models import StackResource

    resources = [
        StackResource("L1", "unsupported-id", "AWS::UnsupportedType::Thing", "DELETE_FAILED")
    ]
    session = MagicMock()
    lam = MagicMock()
    iam = MagicMock()
    session.client.side_effect = lambda svc, **kwargs: lam if svc == "lambda" else iam
    # Should not raise, just skip
    cleanup_stuck_custom_resource_deps(session, resources, "us-east-1")


def test_discover_vpc_dynamic_resources():
    """The hook reaps ENIs and returns discovered SGs for later deletion."""
    session = MagicMock()
    ec2 = MagicMock()
    session.client.return_value = ec2
    # _discover_security_groups uses the describe_security_groups paginator.
    ec2.get_paginator.return_value.paginate.return_value = [
        {"SecurityGroups": [{"GroupId": "sg-1", "GroupName": "not-default"}]}
    ]

    cross = "aws_bench.resource_management.cleanup.handlers.cross_service"
    with (
        patch(f"{cross}._delete_eks_clusters_in_vpcs"),
        patch(f"{cross}._delete_dms_instances_in_vpcs"),
        patch(f"{cross}._delete_vpc_endpoints_in_vpcs"),
        patch(f"{cross}._discover_load_balancers_in_vpcs", return_value=[]),
        patch(f"{cross}._discover_efs_mount_targets", return_value=[]),
        patch(f"{cross}.reap_vpc_enis", return_value=EniReapResult()) as mock_reap,
    ):
        resources = discover_vpc_dynamic_resources(["vpc-123"], session)

    mock_reap.assert_called_once()
    assert ("AWS::EC2::SecurityGroup", "sg-1") in {(r.type, r.identifier) for r in resources}


def test_discover_vpc_dynamic_resources_defers_pinned_resources_when_enis_remain():
    """Leftover requester-managed ENIs → defer them plus the VPC/subnets/SGs they pin.

    When the reaper leaves requester-managed ENIs (service-owned — Lambda Hyperplane, EKS X-ENIs,
    ELB/VPC-endpoint interfaces — released asynchronously by their owner, past StackDeleter's
    bounded wait), the ENIs and the VPC/subnets/non-default SGs they pin are marked deferred so the
    post-cleanup orphan scan excludes them (a later run reaps them once the owner releases the ENIs)
    rather than failing the run. Distinct from the circular-SG revocation (which makes SGs
    immediately deletable) and from the out-of-scope proactive drain.
    """
    session = MagicMock()
    ec2 = MagicMock()
    session.client.return_value = ec2
    cross = "aws_bench.resource_management.cleanup.handlers.cross_service"
    sg = MagicMock(identifier="sg-1")
    with (
        patch(f"{cross}._delete_eks_clusters_in_vpcs"),
        patch(f"{cross}._delete_dms_instances_in_vpcs"),
        patch(f"{cross}._delete_vpc_endpoints_in_vpcs"),
        patch(f"{cross}._discover_load_balancers_in_vpcs", return_value=[]),
        patch(f"{cross}._discover_efs_mount_targets", return_value=[]),
        patch(f"{cross}._discover_security_groups", return_value=[sg]),
        patch(f"{cross}._subnets_in_vpcs", return_value={"subnet-1"}),
        patch(f"{cross}.reap_vpc_enis", return_value=EniReapResult(remaining=["eni-x"])),
        patch(f"{cross}.mark_deferred") as mock_defer,
    ):
        discover_vpc_dynamic_resources(["vpc-123"], session)

    deferred = {(call.args[0], call.args[1]) for call in mock_defer.call_args_list}
    assert ("AWS::EC2::VPC", "vpc-123") in deferred
    assert ("AWS::EC2::Subnet", "subnet-1") in deferred
    assert ("AWS::EC2::SecurityGroup", "sg-1") in deferred
    assert ("AWS::EC2::NetworkInterface", "eni-x") in deferred


def test_discover_vpc_dynamic_resources_no_defer_when_no_enis_remain():
    """No leftover ENIs → nothing is deferred (the common clean path)."""
    session = MagicMock()
    session.client.return_value = MagicMock()
    cross = "aws_bench.resource_management.cleanup.handlers.cross_service"
    with (
        patch(f"{cross}._delete_eks_clusters_in_vpcs"),
        patch(f"{cross}._delete_dms_instances_in_vpcs"),
        patch(f"{cross}._delete_vpc_endpoints_in_vpcs"),
        patch(f"{cross}._discover_load_balancers_in_vpcs", return_value=[]),
        patch(f"{cross}._discover_efs_mount_targets", return_value=[]),
        patch(f"{cross}._discover_security_groups", return_value=[]),
        patch(f"{cross}.reap_vpc_enis", return_value=EniReapResult()),
        patch(f"{cross}.mark_deferred") as mock_defer,
    ):
        discover_vpc_dynamic_resources(["vpc-123"], session)
    mock_defer.assert_not_called()


def test_discover_vpc_dynamic_resources_reaps_dms_before_enis():
    """Reap DMS instances before ENI discovery.

    Their RequesterManaged ENIs are released first so the subnets can delete.
    """
    session = MagicMock()
    session.client.return_value = MagicMock()
    cross = "aws_bench.resource_management.cleanup.handlers.cross_service"
    call_order: list[str] = []
    with (
        patch(f"{cross}._delete_eks_clusters_in_vpcs"),
        patch(
            f"{cross}._delete_dms_instances_in_vpcs",
            side_effect=lambda *a, **k: call_order.append("dms"),
        ),
        patch(
            f"{cross}._delete_vpc_endpoints_in_vpcs",
            side_effect=lambda *a, **k: call_order.append("vpce"),
        ),
        patch(f"{cross}._discover_efs_mount_targets", return_value=[]),
        patch(
            f"{cross}.reap_vpc_enis",
            side_effect=lambda *a, **k: call_order.append("enis") or EniReapResult(),
        ),
        patch(f"{cross}._discover_security_groups", return_value=[]),
    ):
        discover_vpc_dynamic_resources(["vpc-123"], session)
    # Both ENI-holding resources reaped before the ENI reap; order between them
    # doesn't matter, only that the ENI reap comes last.
    assert call_order[-1] == "enis"
    assert set(call_order[:-1]) == {"dms", "vpce"}


def test_discover_load_balancers_in_vpcs_finds_elbv2_and_classic():
    """ELBv2 (by VpcId) and classic ELB (by VPCId) in the target VPCs are returned."""
    from aws_bench.resource_management.cleanup.handlers.cross_service import (
        _discover_load_balancers_in_vpcs,
    )

    elbv2 = MagicMock()
    elb = MagicMock()

    def mock_client(service, **_kw):
        return elbv2 if service == "elbv2" else elb

    session = MagicMock()
    session.client.side_effect = mock_client

    elbv2_paginator = MagicMock()
    elbv2_paginator.paginate.return_value = [
        {
            "LoadBalancers": [
                {"LoadBalancerArn": "arn:...:loadbalancer/app/in/1", "VpcId": "vpc-123"},
                {"LoadBalancerArn": "arn:...:loadbalancer/app/out/2", "VpcId": "vpc-other"},
            ]
        }
    ]
    elb_paginator = MagicMock()
    elb_paginator.paginate.return_value = [
        {
            "LoadBalancerDescriptions": [
                {"LoadBalancerName": "classic-in", "VPCId": "vpc-123"},
                {"LoadBalancerName": "classic-out", "VPCId": "vpc-other"},
            ]
        }
    ]
    elbv2.get_paginator.return_value = elbv2_paginator
    elb.get_paginator.return_value = elb_paginator

    resources = _discover_load_balancers_in_vpcs(session, ["vpc-123"])

    by_type = {(r.type, r.identifier) for r in resources}
    assert (
        "AWS::ElasticLoadBalancingV2::LoadBalancer",
        "arn:...:loadbalancer/app/in/1",
    ) in by_type
    assert ("AWS::ElasticLoadBalancing::LoadBalancer", "classic-in") in by_type
    # Load balancers in other VPCs are excluded.
    assert not any("out" in ident for _t, ident in by_type)


def test_discover_load_balancers_in_vpcs_swallows_errors():
    """A discovery error on one LB API is logged and swallowed, not raised."""
    from aws_bench.resource_management.cleanup.handlers.cross_service import (
        _discover_load_balancers_in_vpcs,
    )

    elbv2 = MagicMock()
    elb = MagicMock()

    def mock_client(service, **_kw):
        return elbv2 if service == "elbv2" else elb

    session = MagicMock()
    session.client.side_effect = mock_client

    # elbv2 raises; classic returns one in-VPC LB.
    elbv2.get_paginator.side_effect = Exception("elbv2 boom")
    elb_paginator = MagicMock()
    elb_paginator.paginate.return_value = [
        {"LoadBalancerDescriptions": [{"LoadBalancerName": "classic-in", "VPCId": "vpc-123"}]}
    ]
    elb.get_paginator.return_value = elb_paginator

    resources = _discover_load_balancers_in_vpcs(session, ["vpc-123"])
    assert [(r.type, r.identifier) for r in resources] == [
        ("AWS::ElasticLoadBalancing::LoadBalancer", "classic-in")
    ]


def test_discover_vpc_dynamic_resources_reaps_load_balancers_before_enis():
    """Load balancers are discovered before ENIs so their ENIs release first."""
    from aws_bench.resource_management.cleanup.handlers.cross_service import (
        discover_vpc_dynamic_resources,
    )

    session = MagicMock()
    session.client.return_value = MagicMock()
    cross = "aws_bench.resource_management.cleanup.handlers.cross_service"
    call_order: list[str] = []
    with (
        patch(f"{cross}._delete_eks_clusters_in_vpcs"),
        patch(f"{cross}._delete_dms_instances_in_vpcs"),
        patch(f"{cross}._delete_vpc_endpoints_in_vpcs"),
        patch(
            f"{cross}._discover_load_balancers_in_vpcs",
            side_effect=lambda *a, **k: call_order.append("lbs") or [],
        ),
        patch(f"{cross}._discover_efs_mount_targets", return_value=[]),
        patch(
            f"{cross}.reap_vpc_enis",
            side_effect=lambda *a, **k: call_order.append("enis") or EniReapResult(),
        ),
        patch(f"{cross}._discover_security_groups", return_value=[]),
    ):
        discover_vpc_dynamic_resources(["vpc-123"], session)
    assert call_order.index("lbs") < call_order.index("enis")


def test_delete_eks_clusters_in_vpcs_describe_cluster_error():
    """Test _delete_eks_clusters_in_vpcs handles describe_cluster errors."""
    from aws_bench.resource_management.cleanup.handlers.cross_service import (
        _delete_eks_clusters_in_vpcs,
    )

    session = MagicMock()
    eks = MagicMock()
    session.client.return_value = eks

    eks.list_clusters.return_value = {"clusters": ["cluster-1"]}
    # describe_cluster raises exception
    eks.describe_cluster.side_effect = Exception("API error")

    # Should not raise, just skip the cluster
    _delete_eks_clusters_in_vpcs(session, ["vpc-123"])


def test_discover_efs_mount_targets_no_subnets():
    """Test _discover_efs_mount_targets returns empty list when no subnets."""
    from aws_bench.resource_management.cleanup.handlers.cross_service import (
        _discover_efs_mount_targets,
    )

    session = MagicMock()
    ec2 = MagicMock()
    efs = MagicMock()

    def mock_client(service):
        if service == "ec2":
            return ec2
        return efs

    session.client.side_effect = mock_client

    # No subnets in the VPC
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Subnets": []}]
    ec2.get_paginator.return_value = paginator

    resources = _discover_efs_mount_targets(session, ["vpc-123"])
    assert resources == []


# -- Security-group ENI-reference drain --


def _eni_paginator(enis: list[dict]) -> MagicMock:
    """A mock ec2 whose describe_network_interfaces paginator yields ``enis``."""
    ec2 = MagicMock()
    ec2.get_paginator.return_value.paginate.return_value = [{"NetworkInterfaces": enis}]
    return ec2


def test_drain_sg_eni_references_deletes_available_eni():
    """An ``available`` ENI referencing the SG is deleted, freeing the SG."""
    ec2 = _eni_paginator([{"NetworkInterfaceId": "eni-1", "Status": "available"}])
    still_pinned = drain_sg_eni_references(ec2, ["sg-1"])
    ec2.delete_network_interface.assert_called_once_with(NetworkInterfaceId="eni-1")
    assert still_pinned == []
    # It queries by group-id (region-wide), not vpc-id, so cross-stack pins are found.
    _, kwargs = ec2.get_paginator.return_value.paginate.call_args
    assert kwargs["Filters"] == [{"Name": "group-id", "Values": ["sg-1"]}]


def test_drain_sg_eni_references_rewrites_customer_in_use_eni_groups():
    """An in-use customer ENI keeps its other SGs but the target SG is dropped."""
    ec2 = _eni_paginator(
        [
            {
                "NetworkInterfaceId": "eni-2",
                "Status": "in-use",
                "RequesterManaged": False,
                "VpcId": "vpc-1",
                "Groups": [{"GroupId": "sg-1"}, {"GroupId": "sg-other"}],
            }
        ]
    )
    still_pinned = drain_sg_eni_references(ec2, ["sg-1"])
    ec2.modify_network_interface_attribute.assert_called_once_with(
        NetworkInterfaceId="eni-2", Groups=["sg-other"]
    )
    ec2.delete_network_interface.assert_not_called()
    assert still_pinned == []


def test_drain_sg_eni_references_substitutes_default_sg_when_only_group():
    """If the target SG is the ENI's only group, the VPC default SG is substituted."""
    ec2 = _eni_paginator(
        [
            {
                "NetworkInterfaceId": "eni-3",
                "Status": "in-use",
                "RequesterManaged": False,
                "VpcId": "vpc-1",
                "Groups": [{"GroupId": "sg-1"}],
            }
        ]
    )
    ec2.describe_security_groups.return_value = {"SecurityGroups": [{"GroupId": "sg-default"}]}
    drain_sg_eni_references(ec2, ["sg-1"])
    ec2.modify_network_interface_attribute.assert_called_once_with(
        NetworkInterfaceId="eni-3", Groups=["sg-default"]
    )


def test_drain_sg_eni_references_defers_requester_managed_eni():
    """A requester-managed ENI cannot be touched; its SG is returned as still-pinned."""
    ec2 = _eni_paginator(
        [{"NetworkInterfaceId": "eni-4", "Status": "in-use", "RequesterManaged": True}]
    )
    still_pinned = drain_sg_eni_references(ec2, ["sg-1"])
    ec2.delete_network_interface.assert_not_called()
    ec2.modify_network_interface_attribute.assert_not_called()
    assert still_pinned == ["sg-1"]


def test_drain_sg_eni_references_survives_discovery_error():
    """A describe failure on one SG is swallowed (best-effort), not raised."""
    ec2 = MagicMock()
    ec2.get_paginator.return_value.paginate.side_effect = Exception("boom")
    assert drain_sg_eni_references(ec2, ["sg-1"]) == []


def test_prepare_security_group_revokes_and_drains():
    """The prepare handler revokes rules and drains ENI references for a non-default SG."""
    from aws_bench.resource_management.ccapi.models import Resource
    from aws_bench.resource_management.cleanup.models import HandlerStatus

    ec2 = MagicMock()
    ec2.describe_security_groups.return_value = {
        "SecurityGroups": [
            {
                "GroupId": "sg-1",
                "GroupName": "WorkerNodeSecurityGroup",
                "IpPermissions": [{"IpProtocol": "-1"}],
                "IpPermissionsEgress": [],
            }
        ]
    }
    # No ENIs reference the SG -> nothing to drain, not deferred.
    ec2.get_paginator.return_value.paginate.return_value = [{"NetworkInterfaces": []}]
    session = MagicMock()
    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.build_client",
        return_value=ec2,
    ):
        result = _prepare_security_group(
            Resource(type="AWS::EC2::SecurityGroup", identifier="sg-1"), session
        )
    ec2.revoke_security_group_ingress.assert_called_once()
    assert result.status == HandlerStatus.SUCCESS


def test_prepare_security_group_skips_default():
    """The default SG is left untouched (CFN/CCAPI delete it with the VPC)."""
    from aws_bench.resource_management.ccapi.models import Resource
    from aws_bench.resource_management.cleanup.models import HandlerStatus

    ec2 = MagicMock()
    ec2.describe_security_groups.return_value = {
        "SecurityGroups": [{"GroupId": "sg-def", "GroupName": "default"}]
    }
    session = MagicMock()
    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.build_client",
        return_value=ec2,
    ):
        result = _prepare_security_group(
            Resource(type="AWS::EC2::SecurityGroup", identifier="sg-def"), session
        )
    assert result.status == HandlerStatus.SKIPPED
    ec2.revoke_security_group_ingress.assert_not_called()


def test_prepare_security_group_defers_when_requester_managed_eni_pins_it():
    """A requester-managed ENI still pinning the SG -> deferred for a later re-drive."""
    from aws_bench.resource_management.ccapi.models import Resource
    from aws_bench.resource_management.cleanup.models import HandlerStatus

    ec2 = MagicMock()
    ec2.describe_security_groups.return_value = {
        "SecurityGroups": [
            {"GroupId": "sg-1", "GroupName": "WorkerNodeSecurityGroup", "IpPermissions": []}
        ]
    }
    ec2.get_paginator.return_value.paginate.return_value = [
        {"NetworkInterfaces": [{"NetworkInterfaceId": "eni-x", "RequesterManaged": True}]}
    ]
    session = MagicMock()
    with (
        patch(
            "aws_bench.resource_management.cleanup.handlers.cross_service.build_client",
            return_value=ec2,
        ),
        patch(
            "aws_bench.resource_management.cleanup.handlers.cross_service.mark_deferred"
        ) as mock_defer,
    ):
        result = _prepare_security_group(
            Resource(type="AWS::EC2::SecurityGroup", identifier="sg-1"), session
        )
    mock_defer.assert_called_once_with("AWS::EC2::SecurityGroup", "sg-1")
    assert result.status == HandlerStatus.SUCCESS


def test_prepare_security_group_skips_missing_sg():
    """A not-found SG is a clean SKIP (already gone)."""
    from aws_bench.resource_management.ccapi.models import Resource
    from aws_bench.resource_management.cleanup.models import HandlerStatus

    ec2 = MagicMock()
    ec2.describe_security_groups.side_effect = ClientError(
        {"Error": {"Code": "InvalidGroup.NotFound", "Message": "gone"}}, "DescribeSecurityGroups"
    )
    session = MagicMock()
    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.build_client",
        return_value=ec2,
    ):
        result = _prepare_security_group(
            Resource(type="AWS::EC2::SecurityGroup", identifier="sg-gone"), session
        )
    assert result.status == HandlerStatus.SKIPPED


# -- IPAM child-pool reaper --


@pytest.fixture(autouse=True)
def _fast_ipam_drain():
    """Shrink the IPAM drain budget/interval and neutralize its sleep so reaper tests run fast.

    The reaper binds ``_DEPROVISION_BUDGET_SEC`` in the cross_service namespace (its own
    shared deadline), so patch both that copy and the ipam module's for the drain itself.
    """
    ipam = "aws_bench.resource_management.cleanup.handlers.ipam"
    cross = "aws_bench.resource_management.cleanup.handlers.cross_service"
    with (
        patch(f"{ipam}._DEPROVISION_BUDGET_SEC", 0.05),
        patch(f"{ipam}._DEPROVISION_POLL_INTERVAL_SEC", 0.001),
        patch(f"{ipam}.time.sleep"),
        patch(f"{cross}._DEPROVISION_BUDGET_SEC", 0.05),
    ):
        yield


def _ipam_ec2(pools: list[dict], cidrs_by_pool: dict[str, list[list[dict]]]) -> MagicMock:
    """Build a mock EC2 client for reap_ipam_child_pools.

    ``pools`` is the full DescribeIpamPools result. ``cidrs_by_pool`` maps a pool id
    to its GetIpamPoolCidrs page-lists, popped in call order (two per delete: initial
    list then post-deprovision wait).
    """
    ec2 = MagicMock()
    describe_pools_paginator = MagicMock()
    describe_pools_paginator.paginate.return_value = [{"IpamPools": pools}]

    # Per get_ipam_pool_cidrs paginator, pop the next scripted page-list for the
    # pool id passed to paginate(IpamPoolId=...). get_ipam_pool_allocations always
    # reports no allocation (these reaper tests exercise the CIDR path, not the wait).
    def get_paginator(operation: str) -> MagicMock:
        if operation == "describe_ipam_pools":
            return describe_pools_paginator
        if operation == "get_ipam_pool_allocations":
            alloc_paginator = MagicMock()
            alloc_paginator.paginate.return_value = [{"IpamPoolAllocations": []}]
            return alloc_paginator
        assert operation == "get_ipam_pool_cidrs"
        cidr_paginator = MagicMock()

        def paginate(IpamPoolId: str, **_kw):
            pages = cidrs_by_pool.get(IpamPoolId, [])
            page = pages.pop(0) if pages else []
            return [{"IpamPoolCidrs": page}]

        cidr_paginator.paginate.side_effect = paginate
        return cidr_paginator

    ec2.get_paginator.side_effect = get_paginator
    ec2.delete_ipam_pool.return_value = {"IpamPool": {"State": "delete-complete"}}
    # Confirm-gone poll: describe_ipam_pools reports the pool absent by default so a
    # deleted child confirms DELETED. Tests that keep a pool present override this.
    ec2.describe_ipam_pools.return_value = {"IpamPools": []}
    return ec2


def _session_returning(ec2: MagicMock) -> MagicMock:
    session = MagicMock()
    session.client.return_value = ec2
    return session


def test_reap_ipam_child_pools_deprovisions_and_deletes_child():
    """A parent with one sourced child (provisioned CIDR) -> child CIDR deprovisioned + deleted."""
    parent = "ipam-pool-parent"
    child = "ipam-pool-child"
    pools = [
        {"IpamPoolId": parent, "SourceIpamPoolId": None},
        {"IpamPoolId": child, "SourceIpamPoolId": parent},
    ]
    cidrs_by_pool = {
        # child: initial list (provisioned) then post-deprovision wait (gone)
        child: [
            [{"Cidr": "10.1.0.0/24", "State": "provisioned"}],
            [],
        ],
    }
    ec2 = _ipam_ec2(pools, cidrs_by_pool)

    result = reap_ipam_child_pools(_session_returning(ec2), [parent])

    ec2.deprovision_ipam_pool_cidr.assert_called_once_with(IpamPoolId=child, Cidr="10.1.0.0/24")
    ec2.delete_ipam_pool.assert_called_once_with(IpamPoolId=child)
    assert result.deleted == [child]
    assert result.remaining == []
    assert result.reaped_any is True


def test_reap_ipam_child_pools_deletes_blocking_vpc():
    """A leaked child whose allocation is a VPC -> the reaper deletes that VPC to free it.

    Proves the shared drain's blocking-VPC deletion is reached through the reaper path too
    (a leaked child's allocation is also a VPC), so the reaper converges in one pass.
    """
    parent = "ipam-pool-parent"
    child = "ipam-pool-child"
    pools = [
        {"IpamPoolId": parent, "SourceIpamPoolId": None},
        {"IpamPoolId": child, "SourceIpamPoolId": parent},
    ]
    # Child keeps a provisioned CIDR until its VPC allocation releases, then deprovisions.
    cidrs_by_pool = {
        child: [
            [{"Cidr": "10.1.0.0/24", "State": "provisioned"}],
            [{"Cidr": "10.1.0.0/24", "State": "provisioned"}],
            [],
        ],
    }
    # Allocation present (a VPC in the client's region) for the first poll, then released.
    alloc_pages = [
        {
            "IpamPoolAllocations": [
                {"ResourceType": "vpc", "ResourceId": "vpc-9", "ResourceRegion": "us-east-1"}
            ]
        },
        {"IpamPoolAllocations": []},
    ]
    ec2 = _ipam_ec2(pools, cidrs_by_pool)
    ec2.meta.region_name = "us-east-1"
    base_get_paginator = ec2.get_paginator.side_effect

    def get_paginator(operation: str) -> MagicMock:
        if operation == "get_ipam_pool_allocations":
            paginator = MagicMock()
            page = alloc_pages.pop(0) if len(alloc_pages) > 1 else alloc_pages[0]
            paginator.paginate.return_value = [page]
            return paginator
        return base_get_paginator(operation)

    ec2.get_paginator.side_effect = get_paginator

    result = reap_ipam_child_pools(_session_returning(ec2), [parent])

    ec2.delete_vpc.assert_called_once_with(VpcId="vpc-9")
    ec2.delete_ipam_pool.assert_called_once_with(IpamPoolId=child)
    assert result.deleted == [child]


def test_reap_ipam_child_pools_recurses_grandchild_first():
    """A child with its own grandchild -> grandchild reaped BEFORE the child (bottom-up)."""
    parent = "ipam-pool-parent"
    child = "ipam-pool-child"
    grandchild = "ipam-pool-grandchild"
    pools = [
        {"IpamPoolId": parent, "SourceIpamPoolId": None},
        {"IpamPoolId": child, "SourceIpamPoolId": parent},
        {"IpamPoolId": grandchild, "SourceIpamPoolId": child},
    ]
    cidrs_by_pool = {
        child: [[], []],
        grandchild: [[], []],
    }
    ec2 = _ipam_ec2(pools, cidrs_by_pool)

    result = reap_ipam_child_pools(_session_returning(ec2), [parent])

    # Grandchild must be deleted before its parent (the child).
    delete_order = [call.kwargs["IpamPoolId"] for call in ec2.delete_ipam_pool.call_args_list]
    assert delete_order == [grandchild, child]
    assert result.deleted == [grandchild, child]


def test_reap_ipam_child_pools_delete_error_lands_in_remaining_and_never_raises():
    """A delete that never confirms gone does not raise; the child is FAILED -> ``remaining``."""
    parent = "ipam-pool-parent"
    child = "ipam-pool-child"
    pools = [
        {"IpamPoolId": parent, "SourceIpamPoolId": None},
        {"IpamPoolId": child, "SourceIpamPoolId": parent},
    ]
    # Drain sees no CIDRs; delete is rejected and the pool never vanishes -> FAILED.
    cidrs_by_pool = {child: [[]]}
    ec2 = _ipam_ec2(pools, cidrs_by_pool)
    ec2.delete_ipam_pool.side_effect = ClientError(
        {"Error": {"Code": "InvalidParameterValue", "Message": "Cannot delete pool with CIDRs"}},
        "DeleteIpamPool",
    )
    ec2.describe_ipam_pools.return_value = {"IpamPools": [{"IpamPoolId": child}]}

    result = reap_ipam_child_pools(_session_returning(ec2), [parent])

    assert result.deleted == []
    assert result.remaining == [child]
    assert result.reaped_any is False


def test_reap_ipam_child_pools_surprise_exception_lands_in_remaining_and_never_raises():
    """A non-boto surprise from deprovision_and_delete_pool is caught, not propagated.

    Best-effort contract: a raised RuntimeError is swallowed and the child recorded
    in ``remaining`` (not deleted).
    """
    parent = "ipam-pool-parent"
    child = "ipam-pool-child"
    pools = [
        {"IpamPoolId": parent, "SourceIpamPoolId": None},
        {"IpamPoolId": child, "SourceIpamPoolId": parent},
    ]
    ec2 = _ipam_ec2(pools, {child: [[], []]})

    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.deprovision_and_delete_pool",
        side_effect=RuntimeError("surprise"),
    ):
        result = reap_ipam_child_pools(_session_returning(ec2), [parent])

    assert result.deleted == []
    assert result.remaining == [child]
    assert result.reaped_any is False


def test_reap_ipam_child_pools_discovery_error_returns_empty():
    """A describe_ipam_pools failure is swallowed (best-effort) and yields an empty result."""
    ec2 = MagicMock()
    ec2.get_paginator.side_effect = ClientError(
        {"Error": {"Code": "InternalError", "Message": "boom"}}, "DescribeIpamPools"
    )
    result = reap_ipam_child_pools(_session_returning(ec2), ["ipam-pool-parent"])
    assert result.deleted == []
    assert result.remaining == []


def test_reap_ipam_child_pools_no_parents_is_noop():
    """No parent pool ids -> no client built, empty result."""
    session = MagicMock()
    result = reap_ipam_child_pools(session, [])
    assert isinstance(result, IpamPoolReapResult)
    assert result.deleted == []
    session.client.assert_not_called()


def test_reap_ipam_child_pools_cyclic_source_graph_terminates():
    """A cyclic SourceIpamPoolId graph does not recurse infinitely; each pool deleted once."""
    parent = "ipam-pool-parent"
    a = "ipam-pool-a"
    b = "ipam-pool-b"
    # A sources from parent, B sources from A, and A also sources from B (cycle A<->B).
    pools = [
        {"IpamPoolId": parent, "SourceIpamPoolId": None},
        {"IpamPoolId": a, "SourceIpamPoolId": parent},
        {"IpamPoolId": b, "SourceIpamPoolId": a},
        {"IpamPoolId": a, "SourceIpamPoolId": b},
    ]
    cidrs_by_pool = {a: [[], []], b: [[], []]}
    ec2 = _ipam_ec2(pools, cidrs_by_pool)

    result = reap_ipam_child_pools(_session_returning(ec2), [parent])

    # Terminates and deletes each distinct pool exactly once (visited-set guard).
    deleted_ids = [call.kwargs["IpamPoolId"] for call in ec2.delete_ipam_pool.call_args_list]
    assert sorted(deleted_ids) == [a, b]
    assert sorted(result.deleted) == [a, b]


def test_reap_ipam_child_pools_skips_stack_owned_child():
    """A child in ``stack_owned_pool_ids`` is CFN-owned, not a leak: never reaped.

    The parent sources two children; only the non-owned one is deprovisioned and
    deleted. The stack-owned child is never passed to deprovision/delete.
    """
    parent = "ipam-pool-parent"
    owned_child = "ipam-pool-owned"
    leaked_child = "ipam-pool-leaked"
    pools = [
        {"IpamPoolId": parent, "SourceIpamPoolId": None},
        {"IpamPoolId": owned_child, "SourceIpamPoolId": parent},
        {"IpamPoolId": leaked_child, "SourceIpamPoolId": parent},
    ]
    cidrs_by_pool = {
        leaked_child: [
            [{"Cidr": "10.2.0.0/24", "State": "provisioned"}],
            [],
        ],
    }
    ec2 = _ipam_ec2(pools, cidrs_by_pool)

    result = reap_ipam_child_pools(
        _session_returning(ec2),
        [parent],
        stack_owned_pool_ids={parent, owned_child},
    )

    # Only the leaked child is reaped; the stack-owned child is never touched.
    ec2.delete_ipam_pool.assert_called_once_with(IpamPoolId=leaked_child)
    deprovisioned = {
        call.kwargs["IpamPoolId"] for call in ec2.deprovision_ipam_pool_cidr.call_args_list
    }
    assert owned_child not in deprovisioned
    assert result.deleted == [leaked_child]
    assert result.remaining == []


def test_reap_ipam_child_pools_skips_stack_owned_child_subtree():
    """A stack-owned child's own descendants are the stack's problem, never reaped as leaks."""
    parent = "ipam-pool-parent"
    owned_child = "ipam-pool-owned"
    owned_grandchild = "ipam-pool-owned-grandchild"
    pools = [
        {"IpamPoolId": parent, "SourceIpamPoolId": None},
        {"IpamPoolId": owned_child, "SourceIpamPoolId": parent},
        {"IpamPoolId": owned_grandchild, "SourceIpamPoolId": owned_child},
    ]
    ec2 = _ipam_ec2(pools, {})

    result = reap_ipam_child_pools(
        _session_returning(ec2),
        [parent],
        stack_owned_pool_ids={parent, owned_child},
    )

    # Neither the owned child nor its (recursed) subtree is deleted.
    ec2.delete_ipam_pool.assert_not_called()
    assert result.deleted == []
    assert result.remaining == []


def test_reap_ipam_child_pools_unconfirmed_child_is_failed_not_deferred():
    """A child whose delete never confirms gone is FAILED (remaining), never deferred.

    The old defer path is gone: a still-present child is reported in ``remaining`` and
    ``reaped_any`` is False, so the caller does not re-drive on it alone.
    """
    parent = "ipam-pool-parent"
    child = "ipam-pool-child"
    pools = [
        {"IpamPoolId": parent, "SourceIpamPoolId": None},
        {"IpamPoolId": child, "SourceIpamPoolId": parent},
    ]
    # Drain sees no CIDRs; delete is rejected and the pool never vanishes -> FAILED.
    ec2 = _ipam_ec2(pools, {child: [[]]})
    ec2.delete_ipam_pool.side_effect = ClientError(
        {"Error": {"Code": "InvalidParameterValue", "Message": "CIDR still deprovisioning"}},
        "DeleteIpamPool",
    )
    ec2.describe_ipam_pools.return_value = {"IpamPools": [{"IpamPoolId": child}]}

    with deferred_scope() as entries:
        result = reap_ipam_child_pools(_session_returning(ec2), [parent])
        # Nothing is deferred anymore.
        assert ("AWS::EC2::IPAMPool", child) not in entries

    assert result.deleted == []
    assert result.remaining == [child]
    assert result.reaped_any is False


def test_reap_ipam_child_pools_already_gone_child_counts_as_deleted():
    """An ALREADY_GONE child (removed by a prior pass) counts as successfully reaped."""
    parent = "ipam-pool-parent"
    child = "ipam-pool-child"
    pools = [
        {"IpamPoolId": parent, "SourceIpamPoolId": None},
        {"IpamPoolId": child, "SourceIpamPoolId": parent},
    ]
    ec2 = _ipam_ec2(pools, {child: [[], []]})

    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.deprovision_and_delete_pool",
        return_value=PoolDeleteResult(PoolDeleteOutcome.ALREADY_GONE, "IPAM pool already gone"),
    ):
        result = reap_ipam_child_pools(_session_returning(ec2), [parent])

    assert result.deleted == [child]
    assert result.remaining == []
    assert result.reaped_any is True


def test_reap_ipam_child_pools_shares_one_deadline_across_children():
    """Two still-allocated leaked children share ONE drain budget, not one each.

    Children are walked sequentially, so a per-pool budget would cost K × budget. The
    reaper threads ONE ``time.monotonic()`` deadline into every deprovision_and_delete_pool
    call, so the whole tree is bounded by ~one budget (MAX). Asserting an IDENTICAL deadline
    for both children proves the shared budget without a real wall-clock wait.
    """
    parent = "ipam-pool-parent"
    child_a = "ipam-pool-a"
    child_b = "ipam-pool-b"
    pools = [
        {"IpamPoolId": parent, "SourceIpamPoolId": None},
        {"IpamPoolId": child_a, "SourceIpamPoolId": parent},
        {"IpamPoolId": child_b, "SourceIpamPoolId": parent},
    ]
    ec2 = _ipam_ec2(pools, {})

    deadlines: dict[str, float | None] = {}

    def spy(_client, pool_id, *, deadline=None):
        deadlines[pool_id] = deadline
        return PoolDeleteResult(
            PoolDeleteOutcome.FAILED, "IPAM pool allocation not released within budget"
        )

    with patch(
        "aws_bench.resource_management.cleanup.handlers.cross_service.deprovision_and_delete_pool",
        side_effect=spy,
    ):
        reap_ipam_child_pools(_session_returning(ec2), [parent])

    # Both children got the SAME non-None deadline -> one shared budget, not two.
    assert deadlines[child_a] is not None
    assert deadlines[child_a] == deadlines[child_b]


# -- NAT / EIP / IGW "mapped public address" wedge --


def _wedge_ec2(
    *,
    nats: list[dict] | None = None,
    nat_poll_state: str = "deleted",
    addresses: list[dict] | None = None,
    igws: list[dict] | None = None,
) -> tuple[MagicMock, list[str]]:
    """A mock ec2 client for the wedge teardown plus an ordered destructive-call recorder.

    ``nats``/``igws`` are what the vpc-id-filtered paginators return; ``nat_poll_state``
    is the State the NAT poll (``describe_nat_gateways``) reports; ``addresses`` is the
    EIP list ``describe_addresses`` returns.
    """
    ec2 = MagicMock()
    order: list[str] = []

    nat_pag = _paginator([{"NatGateways": nats or []}])
    igw_pag = _paginator([{"InternetGateways": igws or []}])

    def get_paginator(op: str) -> MagicMock:
        if op == "describe_nat_gateways":
            return nat_pag
        if op == "describe_internet_gateways":
            return igw_pag
        raise AssertionError(f"unexpected paginator {op}")

    ec2.get_paginator.side_effect = get_paginator
    ec2.describe_nat_gateways.return_value = {"NatGateways": [{"State": nat_poll_state}]}
    ec2.describe_addresses.return_value = {"Addresses": addresses or []}

    for name in (
        "delete_nat_gateway",
        "disassociate_address",
        "release_address",
        "detach_internet_gateway",
        "delete_internet_gateway",
    ):
        getattr(ec2, name).side_effect = (lambda n: lambda **kw: order.append(n))(name)
    return ec2, order


def test_clear_wedge_deletes_nat_releases_eips_deletes_igw_in_order():
    """NAT gateway → EIP (disassociate+release / release-only) → IGW detach+delete, in order."""
    session = MagicMock()
    ec2, order = _wedge_ec2(
        nats=[{"NatGatewayId": "nat-1", "State": "available"}],
        nat_poll_state="deleted",
        addresses=[
            {"AllocationId": "eipalloc-1", "AssociationId": "eipassoc-1"},
            {"AllocationId": "eipalloc-2"},  # no association -> release only
        ],
        igws=[{"InternetGatewayId": "igw-1", "Attachments": [{"VpcId": "vpc-1"}]}],
    )
    session.client.return_value = ec2

    result = clear_vpc_public_address_wedge(session, ["vpc-1"])

    assert result.nat_deleted == ["nat-1"]
    assert set(result.eips_released) == {"eipalloc-1", "eipalloc-2"}
    assert result.igws_deleted == ["igw-1"]
    assert result.remaining == []
    assert result.cleared_any is True

    # Only the associated EIP is disassociated; both are released.
    ec2.disassociate_address.assert_called_once_with(AssociationId="eipassoc-1")
    assert ec2.release_address.call_count == 2
    ec2.detach_internet_gateway.assert_called_once_with(InternetGatewayId="igw-1", VpcId="vpc-1")
    ec2.delete_internet_gateway.assert_called_once_with(InternetGatewayId="igw-1")

    # Dependency order: NAT first, then EIP release, then IGW detach, then IGW delete.
    assert order.index("delete_nat_gateway") < order.index("release_address")
    assert order.index("disassociate_address") < order.index("release_address")
    assert order.index("release_address") < order.index("detach_internet_gateway")
    assert order.index("detach_internet_gateway") < order.index("delete_internet_gateway")


def test_clear_wedge_empty_vpc_is_noop_without_destructive_calls():
    """A VPC with no NAT/EIP/IGW returns an empty result and makes no destructive calls."""
    session = MagicMock()
    ec2, order = _wedge_ec2(nats=[], addresses=[], igws=[])
    session.client.return_value = ec2

    result = clear_vpc_public_address_wedge(session, ["vpc-1"])

    assert result.cleared_any is False
    assert result.remaining == []
    assert order == []
    ec2.delete_nat_gateway.assert_not_called()
    ec2.disassociate_address.assert_not_called()
    ec2.release_address.assert_not_called()
    ec2.detach_internet_gateway.assert_not_called()
    ec2.delete_internet_gateway.assert_not_called()


def test_clear_wedge_empty_vpc_ids_returns_empty_without_client():
    session = MagicMock()
    result = clear_vpc_public_address_wedge(session, ["", None])  # type: ignore[list-item]
    assert result.cleared_any is False and result.remaining == []
    session.client.assert_not_called()


def test_clear_wedge_nat_never_deleted_lands_in_remaining():
    """A NAT gateway that never reaches 'deleted' before timeout is reported as remaining."""
    session = MagicMock()
    ec2, _order = _wedge_ec2(
        nats=[{"NatGatewayId": "nat-slow", "State": "available"}],
        nat_poll_state="deleting",  # never reaches "deleted"
        addresses=[],
        igws=[],
    )
    session.client.return_value = ec2
    with patch(f"{CROSS}._NAT_GATEWAY_TIMEOUT", 0), patch(f"{CROSS}.time.sleep"):
        result = clear_vpc_public_address_wedge(session, ["vpc-1"])
    assert result.nat_deleted == []
    assert result.remaining == ["nat-slow"]
    assert result.cleared_any is False


def test_clear_wedge_nat_already_gone_is_treated_as_deleted():
    """delete_nat_gateway NatGatewayNotFound -> recorded as deleted, not failed."""
    session = MagicMock()
    ec2, _order = _wedge_ec2(
        nats=[{"NatGatewayId": "nat-gone", "State": "available"}], addresses=[], igws=[]
    )
    ec2.delete_nat_gateway.side_effect = ClientError(
        {"Error": {"Code": "NatGatewayNotFound"}}, "DeleteNatGateway"
    )
    session.client.return_value = ec2
    result = clear_vpc_public_address_wedge(session, ["vpc-1"])
    assert result.nat_deleted == ["nat-gone"]
    assert result.remaining == []


def test_clear_wedge_eip_release_failure_lands_in_remaining():
    """An EIP that fails to release surfaces in remaining (never a silent success)."""
    session = MagicMock()
    ec2, _order = _wedge_ec2(nats=[], addresses=[{"AllocationId": "eipalloc-stuck"}], igws=[])
    ec2.release_address.side_effect = ClientError(
        {"Error": {"Code": "AuthFailure"}}, "ReleaseAddress"
    )
    session.client.return_value = ec2
    result = clear_vpc_public_address_wedge(session, ["vpc-1"])
    assert result.eips_released == []
    assert result.remaining == ["eipalloc-stuck"]


def test_clear_wedge_igw_detach_failure_lands_in_remaining():
    """An IGW whose detach fails surfaces in remaining and is not deleted."""
    session = MagicMock()
    ec2, _order = _wedge_ec2(
        nats=[],
        addresses=[],
        igws=[{"InternetGatewayId": "igw-stuck", "Attachments": [{"VpcId": "vpc-1"}]}],
    )
    ec2.detach_internet_gateway.side_effect = ClientError(
        {"Error": {"Code": "DependencyViolation"}}, "DetachInternetGateway"
    )
    session.client.return_value = ec2
    result = clear_vpc_public_address_wedge(session, ["vpc-1"])
    assert result.igws_deleted == []
    assert result.remaining == ["igw-stuck"]
    ec2.delete_internet_gateway.assert_not_called()


def test_clear_wedge_eip_disassociate_failure_lands_in_remaining():
    """An EIP whose disassociate fails is not released and surfaces in remaining."""
    session = MagicMock()
    ec2, _order = _wedge_ec2(
        nats=[],
        addresses=[{"AllocationId": "eipalloc-stuck", "AssociationId": "eipassoc-stuck"}],
        igws=[],
    )
    ec2.disassociate_address.side_effect = ClientError(
        {"Error": {"Code": "AuthFailure"}}, "DisassociateAddress"
    )
    session.client.return_value = ec2
    result = clear_vpc_public_address_wedge(session, ["vpc-1"])
    assert result.eips_released == []
    assert result.remaining == ["eipalloc-stuck"]
    ec2.release_address.assert_not_called()


def test_clear_wedge_igw_delete_failure_after_detach_lands_in_remaining():
    """An IGW that detaches but fails to delete surfaces in remaining, not deleted."""
    session = MagicMock()
    ec2, _order = _wedge_ec2(
        nats=[],
        addresses=[],
        igws=[{"InternetGatewayId": "igw-stuck", "Attachments": [{"VpcId": "vpc-1"}]}],
    )
    ec2.delete_internet_gateway.side_effect = ClientError(
        {"Error": {"Code": "DependencyViolation"}}, "DeleteInternetGateway"
    )
    session.client.return_value = ec2
    result = clear_vpc_public_address_wedge(session, ["vpc-1"])
    assert result.igws_deleted == []
    assert result.remaining == ["igw-stuck"]
    ec2.detach_internet_gateway.assert_called_once_with(
        InternetGatewayId="igw-stuck", VpcId="vpc-1"
    )


def test_clear_wedge_igw_already_detached_and_gone_codes_are_idempotent():
    """Gateway.NotAttached on detach and NotFound on delete are treated as success."""
    session = MagicMock()
    ec2, _order = _wedge_ec2(
        nats=[],
        addresses=[],
        igws=[{"InternetGatewayId": "igw-1", "Attachments": [{"VpcId": "vpc-1"}]}],
    )
    ec2.detach_internet_gateway.side_effect = ClientError(
        {"Error": {"Code": "Gateway.NotAttached"}}, "DetachInternetGateway"
    )
    ec2.delete_internet_gateway.side_effect = ClientError(
        {"Error": {"Code": "InvalidInternetGatewayID.NotFound"}}, "DeleteInternetGateway"
    )
    session.client.return_value = ec2
    result = clear_vpc_public_address_wedge(session, ["vpc-1"])
    assert result.igws_deleted == ["igw-1"]
    assert result.remaining == []


def test_clear_wedge_discovery_errors_are_swallowed():
    """A describe failure on each phase is logged and swallowed, not raised."""
    session = MagicMock()
    ec2 = MagicMock()
    ec2.get_paginator.side_effect = Exception("boom")
    ec2.describe_addresses.side_effect = Exception("boom")
    session.client.return_value = ec2
    result = clear_vpc_public_address_wedge(session, ["vpc-1"])
    assert result.cleared_any is False and result.remaining == []


def test_clear_igw_wedge_resolves_vpc_from_attachments_and_delegates():
    """The IGW entry point resolves the VPC via describe_internet_gateways then delegates."""
    session = MagicMock()
    ec2 = MagicMock()
    ec2.describe_internet_gateways.return_value = {
        "InternetGateways": [{"InternetGatewayId": "igw-1", "Attachments": [{"VpcId": "vpc-9"}]}]
    }
    # default_vpc_ids() iterates a describe_vpcs paginator; stub it empty (no default VPC).
    ec2.get_paginator.return_value.paginate.return_value = iter([{"Vpcs": []}])
    session.client.return_value = ec2
    with patch(f"{CROSS}.clear_vpc_public_address_wedge") as mock_clear:
        clear_igw_public_address_wedge(session, ["igw-1"])
    ec2.describe_internet_gateways.assert_called_once_with(InternetGatewayIds=["igw-1"])
    mock_clear.assert_called_once_with(session, ["vpc-9"], region=None)


def test_clear_igw_wedge_detached_igw_is_noop():
    """A detached IGW (no VPC attachment) resolves to no VPC and does nothing."""
    session = MagicMock()
    ec2 = MagicMock()
    ec2.describe_internet_gateways.return_value = {
        "InternetGateways": [{"InternetGatewayId": "igw-1", "Attachments": []}]
    }
    # default_vpc_ids() iterates a describe_vpcs paginator; stub it empty (no default VPC).
    ec2.get_paginator.return_value.paginate.return_value = iter([{"Vpcs": []}])
    session.client.return_value = ec2
    with patch(f"{CROSS}.clear_vpc_public_address_wedge") as mock_clear:
        result = clear_igw_public_address_wedge(session, ["igw-1"])
    mock_clear.assert_not_called()
    assert result.cleared_any is False


def test_clear_igw_wedge_empty_ids_returns_empty_without_client():
    session = MagicMock()
    result = clear_igw_public_address_wedge(session, [""])
    assert result.cleared_any is False
    session.client.assert_not_called()


def test_discover_vpc_dynamic_resources_clears_wedge_before_eni_reap():
    """The VPC hook clears the NAT/EIP/IGW wedge before the ENI reap.

    A NAT gateway holds its own ENI, so clearing it first means the ENI reap won't
    trip over it.
    """
    session = MagicMock()
    session.client.return_value = MagicMock()
    call_order: list[str] = []
    with (
        patch(f"{CROSS}._delete_eks_clusters_in_vpcs"),
        patch(f"{CROSS}._delete_dms_instances_in_vpcs"),
        patch(f"{CROSS}._delete_vpc_endpoints_in_vpcs"),
        patch(f"{CROSS}._discover_load_balancers_in_vpcs", return_value=[]),
        patch(f"{CROSS}._discover_efs_mount_targets", return_value=[]),
        patch(f"{CROSS}._discover_security_groups", return_value=[]),
        patch(
            f"{CROSS}.clear_vpc_public_address_wedge",
            side_effect=lambda *a, **k: call_order.append("wedge") or VpcPublicAddressWedgeResult(),
        ),
        patch(
            f"{CROSS}.reap_vpc_enis",
            side_effect=lambda *a, **k: call_order.append("enis") or EniReapResult(),
        ),
    ):
        discover_vpc_dynamic_resources(["vpc-123"], session)
    assert call_order.index("wedge") < call_order.index("enis")
