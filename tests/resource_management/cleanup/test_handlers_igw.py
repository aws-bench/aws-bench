"""Tests for the VPN Gateway detach pre-delete hook.

The Internet Gateway hook now clears the NAT/EIP/IGW "mapped public address" wedge
(see ``test_handlers_cross_service.py`` and ``test_handlers_vpc.py``); its default-VPC
skip is covered by the cross-service wedge tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from aws_bench.resource_management.cleanup.handlers.vpc import _detach_vpn_gateways
from aws_bench.resource_management.cleanup.models import StackResource


def _igw(physical_id: str) -> StackResource:
    return StackResource("L", physical_id, "AWS::EC2::InternetGateway", "CREATE_COMPLETE")


def _vgw(physical_id: str) -> StackResource:
    return StackResource("L", physical_id, "AWS::EC2::VPNGateway", "CREATE_COMPLETE")


def _client_error(code: str, op: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "x"}}, op)


# ---------------------------------------------------------------------------
# VPN Gateway (no default-VPC guard — AWS never attaches a VGW to the default VPC)
# ---------------------------------------------------------------------------


def test_vpn_hook_no_matching_resources_returns_empty_without_client():
    session = MagicMock()
    result = _detach_vpn_gateways([_igw("igw-1")], session)
    assert result == []
    session.client.assert_not_called()


def test_vpn_hook_detaches_only_attached_vpc():
    ec2 = MagicMock()
    ec2.describe_vpn_gateways.return_value = {
        "VpnGateways": [
            {
                "VpnGatewayId": "vgw-1",
                "VpcAttachments": [
                    {"VpcId": "vpc-1", "State": "attached"},
                    {"VpcId": "vpc-2", "State": "detached"},
                ],
            }
        ]
    }
    session = MagicMock()
    session.client.return_value = ec2

    result = _detach_vpn_gateways([_vgw("vgw-1")], session)

    assert result == []
    ec2.describe_vpn_gateways.assert_called_once_with(VpnGatewayIds=["vgw-1"])
    ec2.detach_vpn_gateway.assert_called_once_with(VpnGatewayId="vgw-1", VpcId="vpc-1")
    # VPN path must not consult default-VPC status (no paginated describe_vpcs lookup).
    ec2.get_paginator.assert_not_called()


def test_vpn_hook_swallows_not_found_on_describe():
    ec2 = MagicMock()
    ec2.describe_vpn_gateways.side_effect = _client_error(
        "InvalidVpnGatewayID.NotFound", "DescribeVpnGateways"
    )
    session = MagicMock()
    session.client.return_value = ec2

    result = _detach_vpn_gateways([_vgw("vgw-1")], session)

    assert result == []
    ec2.detach_vpn_gateway.assert_not_called()
