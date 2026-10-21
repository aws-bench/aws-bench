"""Tests for the standalone-path network-interface cleanup handlers.

The reset's new-resource sweep hands a single ENI / attachment identifier
(classically a requester-managed ECS/ELB/Lambda-VPC/AppStream interface still
draining). CCAPI cannot delete it; the stack/VPC paths DEFER such interfaces so
they self-heal (~20-40 min after the owner is torn down). These handlers bring
that same deferral to the standalone path — delete when genuinely reapable, else
defer; never a bounded wait, never fail-closed on a self-healing ENI.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import CUSTOM_DELETION_REGISTRY
from aws_bench.resource_management.cleanup.handlers.ec2_network_interface import (
    _delete_network_interface,
    _delete_network_interface_attachment,
)
from aws_bench.resource_management.cleanup.models import HandlerStatus
from aws_bench.resource_management.deferred import deferred_scope, is_deferred

_ENI_TYPE = "AWS::EC2::NetworkInterface"
_ATTACH_TYPE = "AWS::EC2::NetworkInterfaceAttachment"


def _eni(eni_id: str = "eni-0123456789abcdef0") -> Resource:
    return Resource(type=_ENI_TYPE, identifier=eni_id)


def _attach(att_id: str = "ela-attach-0cd7ec32b98b29d4d") -> Resource:
    return Resource(type=_ATTACH_TYPE, identifier=att_id)


def _ec2_describing(interfaces: list[dict]) -> MagicMock:
    ec2 = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [{"NetworkInterfaces": interfaces}]
    ec2.get_paginator.return_value = paginator
    return ec2


def _ec2_with_subnet_associations(
    interfaces: list[dict],
    *,
    rt_assocs: list[dict] | None = None,
    nacl_assocs: list[dict] | None = None,
) -> MagicMock:
    """EC2 mock routing describe calls by op name.

    describe_network_interfaces -> interfaces; describe_route_tables -> one route
    table carrying rt_assocs; describe_network_acls -> one NACL carrying nacl_assocs.
    """
    ec2 = MagicMock()

    def _get_paginator(op: str) -> MagicMock:
        paginator = MagicMock()
        if op == "describe_network_interfaces":
            paginator.paginate.return_value = [{"NetworkInterfaces": interfaces}]
        elif op == "describe_route_tables":
            paginator.paginate.return_value = [{"RouteTables": [{"Associations": rt_assocs or []}]}]
        elif op == "describe_network_acls":
            paginator.paginate.return_value = [
                {"NetworkAcls": [{"Associations": nacl_assocs or []}]}
            ]
        else:
            paginator.paginate.return_value = []
        return paginator

    ec2.get_paginator.side_effect = _get_paginator
    return ec2


def _session(ec2: MagicMock) -> MagicMock:
    session = MagicMock()
    session.client.return_value = ec2
    return session


def test_handlers_are_registered():
    assert _ENI_TYPE in CUSTOM_DELETION_REGISTRY
    assert _ATTACH_TYPE in CUSTOM_DELETION_REGISTRY


def test_available_eni_is_deleted_not_deferred():
    ec2 = _ec2_describing([{"NetworkInterfaceId": "eni-1", "Status": "available"}])
    with deferred_scope():
        result = _delete_network_interface(_eni("eni-1"), _session(ec2))
        assert result.status is HandlerStatus.SUCCESS
        assert not is_deferred(_ENI_TYPE, "eni-1")
    ec2.delete_network_interface.assert_called_once_with(NetworkInterfaceId="eni-1")


def test_already_gone_is_success():
    ec2 = _ec2_describing([])
    with deferred_scope():
        result = _delete_network_interface(_eni("eni-gone"), _session(ec2))
        assert result.status is HandlerStatus.SUCCESS
        assert not is_deferred(_ENI_TYPE, "eni-gone")
    ec2.delete_network_interface.assert_not_called()


def test_requester_managed_in_use_is_deferred_not_failed():
    ec2 = _ec2_describing(
        [{"NetworkInterfaceId": "eni-2", "Status": "in-use", "RequesterManaged": True}]
    )
    with deferred_scope():
        result = _delete_network_interface(_eni("eni-2"), _session(ec2))
        # Deferred, not failed, not waited.
        assert result.status is HandlerStatus.SUCCESS
        assert is_deferred(_ENI_TYPE, "eni-2")
    ec2.delete_network_interface.assert_not_called()
    ec2.detach_network_interface.assert_not_called()


def test_requester_managed_eni_also_defers_pinned_subnet_associations():
    """The subnet the ENI pins has its rtb/nacl associations deferred too.

    These associations (not the ENI) were the recorded prod verdict, so deferring
    only the ENI would still fail the run.
    """
    _RT = "AWS::EC2::SubnetRouteTableAssociation"
    _NACL = "AWS::EC2::SubnetNetworkAclAssociation"
    ec2 = _ec2_with_subnet_associations(
        [
            {
                "NetworkInterfaceId": "eni-6",
                "Status": "in-use",
                "RequesterManaged": True,
                "SubnetId": "subnet-abc",
            }
        ],
        rt_assocs=[{"SubnetId": "subnet-abc", "RouteTableAssociationId": "rtbassoc-111"}],
        nacl_assocs=[{"SubnetId": "subnet-abc", "NetworkAclAssociationId": "aclassoc-222"}],
    )
    with deferred_scope():
        result = _delete_network_interface(_eni("eni-6"), _session(ec2))
        assert result.status is HandlerStatus.SUCCESS
        assert is_deferred(_ENI_TYPE, "eni-6")
        assert is_deferred(_RT, "rtbassoc-111")
        assert is_deferred(_NACL, "aclassoc-222")


def test_association_defer_skips_other_subnets():
    """An association on a DIFFERENT subnet than the ENI's is not deferred."""
    _RT = "AWS::EC2::SubnetRouteTableAssociation"
    ec2 = _ec2_with_subnet_associations(
        [
            {
                "NetworkInterfaceId": "eni-7",
                "Status": "in-use",
                "RequesterManaged": True,
                "SubnetId": "subnet-abc",
            }
        ],
        # describe is filtered server-side, but guard client-side too: a stray
        # association for a different subnet must not be deferred.
        rt_assocs=[{"SubnetId": "subnet-OTHER", "RouteTableAssociationId": "rtbassoc-999"}],
    )
    with deferred_scope():
        _delete_network_interface(_eni("eni-7"), _session(ec2))
        assert not is_deferred(_RT, "rtbassoc-999")


def test_association_describe_failure_is_best_effort():
    """A describe failure while deferring associations does not fail the ENI handler."""
    ec2 = MagicMock()

    def _get_paginator(op: str) -> MagicMock:
        paginator = MagicMock()
        if op == "describe_network_interfaces":
            paginator.paginate.return_value = [
                {
                    "NetworkInterfaces": [
                        {
                            "NetworkInterfaceId": "eni-8",
                            "Status": "in-use",
                            "RequesterManaged": True,
                            "SubnetId": "subnet-abc",
                        }
                    ]
                }
            ]
        else:
            paginator.paginate.side_effect = ClientError(
                {"Error": {"Code": "UnauthorizedOperation", "Message": "denied"}}, op
            )
        return paginator

    ec2.get_paginator.side_effect = _get_paginator
    with deferred_scope():
        result = _delete_network_interface(_eni("eni-8"), _session(ec2))
        # ENI still deferred; association describe failure swallowed.
        assert result.status is HandlerStatus.SUCCESS
        assert is_deferred(_ENI_TYPE, "eni-8")


def test_customer_managed_in_use_is_detached_then_deferred():
    ec2 = _ec2_describing(
        [
            {
                "NetworkInterfaceId": "eni-3",
                "Status": "in-use",
                "RequesterManaged": False,
                "Attachment": {"AttachmentId": "eni-attach-3", "Status": "attached"},
            }
        ]
    )
    with deferred_scope():
        result = _delete_network_interface(_eni("eni-3"), _session(ec2))
        assert result.status is HandlerStatus.SUCCESS
        assert is_deferred(_ENI_TYPE, "eni-3")
    ec2.detach_network_interface.assert_called_once_with(AttachmentId="eni-attach-3", Force=True)


def test_available_delete_failure_defers_rather_than_failing():
    ec2 = _ec2_describing([{"NetworkInterfaceId": "eni-4", "Status": "available"}])
    ec2.delete_network_interface.side_effect = ClientError(
        {"Error": {"Code": "InvalidParameterValue", "Message": "in use race"}},
        "DeleteNetworkInterface",
    )
    with deferred_scope():
        result = _delete_network_interface(_eni("eni-4"), _session(ec2))
        # _delete_eni swallows the error (returns False) -> fall through to defer.
        assert result.status is HandlerStatus.SUCCESS
        assert is_deferred(_ENI_TYPE, "eni-4")


def test_describe_error_fails_closed():
    ec2 = MagicMock()
    ec2.get_paginator.return_value.paginate.side_effect = ClientError(
        {"Error": {"Code": "UnauthorizedOperation", "Message": "denied"}},
        "DescribeNetworkInterfaces",
    )
    with deferred_scope():
        result = _delete_network_interface(_eni("eni-5"), _session(ec2))
        assert result.status is HandlerStatus.FAILED
        assert not is_deferred(_ENI_TYPE, "eni-5")


def test_ela_attachment_is_deferred():
    with deferred_scope():
        result = _delete_network_interface_attachment(_attach(), _session(MagicMock()))
        assert result.status is HandlerStatus.SUCCESS
        assert is_deferred(_ATTACH_TYPE, "ela-attach-0cd7ec32b98b29d4d")


def test_eni_attachment_is_deferred():
    with deferred_scope():
        result = _delete_network_interface_attachment(
            _attach("eni-attach-0123456789abcdef0"), _session(MagicMock())
        )
        assert result.status is HandlerStatus.SUCCESS
        assert is_deferred(_ATTACH_TYPE, "eni-attach-0123456789abcdef0")
