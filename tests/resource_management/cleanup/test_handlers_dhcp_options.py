"""Tests for the DHCP options set cleanup handler.

AWS Managed AD (and similar services) create a non-default DHCP options set and
associate it with a VPC. ``delete_dhcp_options`` returns ``DependencyViolation``
while any VPC references the set, so the handler must disassociate every
referencing VPC (reverting it to the ``"default"`` set) before deleting. The set
associated with the account's default VPC must never be touched.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import CUSTOM_DELETION_REGISTRY
from aws_bench.resource_management.cleanup.handlers.vpc import _delete_dhcp_options
from aws_bench.resource_management.cleanup.models import HandlerStatus

_DHCP_TYPE = "AWS::EC2::DHCPOptions"


def _resource(dopt_id: str = "dopt-leaked") -> Resource:
    return Resource(type=_DHCP_TYPE, identifier=dopt_id)


def _ec2_mock(*, default_dopt_id: str | None, vpcs_by_dopt: dict[str, list[str]]):
    """Build an EC2 client mock driven by the ``describe_vpcs`` filter.

    - ``is-default=true`` -> the default VPC (carrying ``default_dopt_id``), if any.
    - ``dhcp-options-id=<id>`` -> the VPCs associated with that set.
    """
    ec2 = MagicMock()

    def _paginate(Filters):  # noqa: N803 — boto3 kwarg name
        key = Filters[0]["Name"]
        if key == "is-default":
            if default_dopt_id is None:
                return [{"Vpcs": []}]
            return [{"Vpcs": [{"VpcId": "vpc-default", "DhcpOptionsId": default_dopt_id}]}]
        if key == "dhcp-options-id":
            dopt_id = Filters[0]["Values"][0]
            vpc_ids = vpcs_by_dopt.get(dopt_id, [])
            return [{"Vpcs": [{"VpcId": v} for v in vpc_ids]}]
        raise AssertionError(f"unexpected filter {key}")

    paginator = MagicMock()
    paginator.paginate.side_effect = _paginate
    ec2.get_paginator.return_value = paginator
    return ec2


def _session_with(ec2: MagicMock) -> MagicMock:
    session = MagicMock()
    session.client.return_value = ec2
    return session


def _not_found(op: str) -> ClientError:
    return ClientError({"Error": {"Code": "InvalidDhcpOptionsID.NotFound", "Message": "gone"}}, op)


def test_handler_is_registered():
    assert _DHCP_TYPE in CUSTOM_DELETION_REGISTRY


def test_disassociates_then_deletes_single_vpc():
    ec2 = _ec2_mock(default_dopt_id="dopt-default", vpcs_by_dopt={"dopt-leaked": ["vpc-a"]})
    result = _delete_dhcp_options(_resource(), _session_with(ec2))

    assert result.status is HandlerStatus.SUCCESS
    ec2.associate_dhcp_options.assert_called_once_with(DhcpOptionsId="default", VpcId="vpc-a")
    ec2.delete_dhcp_options.assert_called_once_with(DhcpOptionsId="dopt-leaked")


def test_disassociates_every_referencing_vpc():
    ec2 = _ec2_mock(
        default_dopt_id="dopt-default", vpcs_by_dopt={"dopt-leaked": ["vpc-a", "vpc-b"]}
    )
    result = _delete_dhcp_options(_resource(), _session_with(ec2))

    assert result.status is HandlerStatus.SUCCESS
    calls = ec2.associate_dhcp_options.call_args_list
    assert {c.kwargs["VpcId"] for c in calls} == {"vpc-a", "vpc-b"}
    assert all(c.kwargs["DhcpOptionsId"] == "default" for c in calls)
    ec2.delete_dhcp_options.assert_called_once_with(DhcpOptionsId="dopt-leaked")


def test_default_vpc_set_is_never_touched():
    # The target set IS the default VPC's current set -> must be skipped untouched.
    ec2 = _ec2_mock(default_dopt_id="dopt-leaked", vpcs_by_dopt={"dopt-leaked": ["vpc-default"]})
    result = _delete_dhcp_options(_resource("dopt-leaked"), _session_with(ec2))

    assert result.status is HandlerStatus.SKIPPED
    ec2.associate_dhcp_options.assert_not_called()
    ec2.delete_dhcp_options.assert_not_called()


def test_unassociated_set_deletes_without_disassociate():
    ec2 = _ec2_mock(default_dopt_id="dopt-default", vpcs_by_dopt={"dopt-leaked": []})
    result = _delete_dhcp_options(_resource(), _session_with(ec2))

    assert result.status is HandlerStatus.SUCCESS
    ec2.associate_dhcp_options.assert_not_called()
    ec2.delete_dhcp_options.assert_called_once_with(DhcpOptionsId="dopt-leaked")


def test_already_gone_set_is_success():
    ec2 = _ec2_mock(default_dopt_id="dopt-default", vpcs_by_dopt={"dopt-leaked": []})
    ec2.delete_dhcp_options.side_effect = _not_found("DeleteDhcpOptions")
    result = _delete_dhcp_options(_resource(), _session_with(ec2))

    assert result.status is HandlerStatus.SUCCESS


def test_vanished_vpc_during_disassociate_is_tolerated():
    ec2 = _ec2_mock(default_dopt_id="dopt-default", vpcs_by_dopt={"dopt-leaked": ["vpc-a"]})
    ec2.associate_dhcp_options.side_effect = ClientError(
        {"Error": {"Code": "InvalidVpcID.NotFound", "Message": "gone"}}, "AssociateDhcpOptions"
    )
    result = _delete_dhcp_options(_resource(), _session_with(ec2))

    # VPC gone -> no longer a blocker -> proceed to delete and succeed.
    assert result.status is HandlerStatus.SUCCESS
    ec2.delete_dhcp_options.assert_called_once_with(DhcpOptionsId="dopt-leaked")


def test_disassociate_failure_fails_closed():
    ec2 = _ec2_mock(default_dopt_id="dopt-default", vpcs_by_dopt={"dopt-leaked": ["vpc-a"]})
    ec2.associate_dhcp_options.side_effect = ClientError(
        {"Error": {"Code": "UnauthorizedOperation", "Message": "denied"}},
        "AssociateDhcpOptions",
    )
    result = _delete_dhcp_options(_resource(), _session_with(ec2))

    assert result.status is HandlerStatus.FAILED
    ec2.delete_dhcp_options.assert_not_called()


def test_delete_failure_fails_closed():
    ec2 = _ec2_mock(default_dopt_id="dopt-default", vpcs_by_dopt={"dopt-leaked": ["vpc-a"]})
    ec2.delete_dhcp_options.side_effect = ClientError(
        {"Error": {"Code": "DependencyViolation", "Message": "still referenced"}},
        "DeleteDhcpOptions",
    )
    result = _delete_dhcp_options(_resource(), _session_with(ec2))

    assert result.status is HandlerStatus.FAILED
