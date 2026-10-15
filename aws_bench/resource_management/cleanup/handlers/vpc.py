"""VPC-network pre-delete hooks.

Three hooks run before CloudControl (CCAPI) deletes VPC-network resources:

* ``AWS::EC2::VPC`` — discover ENIs, EFS mount targets, and security groups
  attached to the VPC so they are deleted alongside it.
* ``AWS::EC2::InternetGateway`` — clear the NAT/EIP "mapped public address"
  wedge, then detach and delete the IGW. CCAPI's ``DeleteResource`` maps to
  ``DeleteInternetGateway``/``DetachInternetGateway``, which raise
  ``DependencyViolation`` while a NAT gateway or EIP is still mapped in the VPC,
  so NAT gateways and EIPs are torn down first (NAT → EIP → IGW). An IGW attached
  to the account's **default VPC** is skipped so the default VPC's connectivity is
  preserved (and it is thus never deleted).
* ``AWS::EC2::VPNGateway`` — the same detach-before-delete requirement, via
  ``DetachVpnGateway`` before ``DeleteVpnGateway``. (No default-VPC guard: AWS
  does not attach a VPN gateway to the default VPC.)

The gateway hooks return no new resources: the gateways are already in the
deletion set; the hooks only make them deletable (like emptying a bucket before
deleting it). They run only in the CCAPI sweep/reset path (``_resolve_hooks`` is
reached only when ``custom_delete`` or ``ccapi_fallback`` is set), not in the
stack-deleter prepare-only pass, so they never pre-empt CloudFormation's own
gateway teardown during a normal stack delete.

Both hooks are best-effort: EC2 calls are wrapped so a missing gateway or a
transient detach failure logs a warning and lets the subsequent CCAPI delete (or
the next sweep) proceed rather than aborting the whole cleanup.
"""

from __future__ import annotations

from typing import Any

import boto3
from botocore.exceptions import ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import (
    pre_delete_hook,
    resource_handler,
)
from aws_bench.resource_management.cleanup.handlers.cross_service import (
    clear_igw_public_address_wedge,
    discover_vpc_dynamic_resources,
)
from aws_bench.resource_management.cleanup.models import (
    HandlerResult,
    HandlerStatus,
    StackResource,
)
from aws_bench.resource_management.fastscan.listers.custom_listers import default_vpcs
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)

# AWS's sentinel for AssociateDhcpOptions meaning "revert this VPC to the
# Amazon-provided default DNS/DHCP config" — detaches a custom set without
# needing another set to swap in.
_DEFAULT_DHCP_OPTIONS = "default"

# Gateway↔VPC attachment states that still hold the gateway to a VPC and thus
# block deletion. An attachment in any other state (notably ``detaching`` /
# ``detached``) needs no action.
_ATTACHED_STATES = {"available", "attached", "attaching"}


@pre_delete_hook("AWS::EC2::VPC")
def _discover_vpc_dynamic_resources(
    resources: list[StackResource], session: boto3.Session
) -> list[Resource]:
    """Find ENIs, EFS mount targets, and security groups attached to VPCs."""
    vpc_ids = [
        resource.physical_id
        for resource in resources
        if resource.resource_type == "AWS::EC2::VPC" and resource.physical_id
    ]
    if not vpc_ids:
        return []
    return discover_vpc_dynamic_resources(vpc_ids, session)


@pre_delete_hook("AWS::EC2::InternetGateway")
def _clear_igw_public_address_wedge(
    resources: list[StackResource], session: boto3.Session
) -> list[Resource]:
    """Clear the NAT/EIP wedge blocking a flagged IGW, then detach+delete it.

    CCAPI's level map omits IGW/NAT/EIP, so DeleteInternetGateway/DetachInternetGateway
    fails with DependencyViolation "mapped public address(es)" while a NAT gateway/EIP is
    still mapped. Resolving the IGW's VPC and clearing NAT → EIP → IGW here removes the
    wedge. An IGW on the account's default VPC is skipped (preserving default connectivity).
    Discovers no new resources (the teardown deletes the IGW itself), so returns []; the
    teardown is idempotent, so a second pass (e.g. the VPC hook already handled the VPC) is
    a clean no-op.
    """
    igw_ids = [
        resource.physical_id
        for resource in resources
        if resource.resource_type == "AWS::EC2::InternetGateway" and resource.physical_id
    ]
    if not igw_ids:
        return []
    clear_igw_public_address_wedge(session, igw_ids)
    return []


@pre_delete_hook("AWS::EC2::VPNGateway")
def _detach_vpn_gateways(resources: list[StackResource], session: boto3.Session) -> list[Resource]:
    """Detach VPN Gateways from their VPCs so CCAPI can delete them.

    ``DeleteVpnGateway`` shares the IGW detach-before-delete requirement: a VGW
    still attached to a VPC cannot be deleted. Returns no new resources.
    """
    vgw_ids = [
        resource.physical_id
        for resource in resources
        if resource.resource_type == "AWS::EC2::VPNGateway" and resource.physical_id
    ]
    if not vgw_ids:
        return []
    ec2 = build_client(session, "ec2")
    for vgw_id in vgw_ids:
        _detach_one_vpn_gateway(ec2, vgw_id)
    return []


def _detach_one_vpn_gateway(ec2: Any, vgw_id: str) -> None:
    """Detach a single VPN gateway from every VPC it is attached to (best-effort)."""
    try:
        resp = ec2.describe_vpn_gateways(VpnGatewayIds=[vgw_id])
    except ClientError as e:
        if not _is_not_found(e):
            logger.warning("DescribeVpnGateways failed for %s: %s", vgw_id, e)
        return
    for vgw in resp.get("VpnGateways", []):
        for attachment in vgw.get("VpcAttachments", []):
            vpc_id = attachment.get("VpcId")
            if not vpc_id or attachment.get("State") not in _ATTACHED_STATES:
                continue
            try:
                ec2.detach_vpn_gateway(VpnGatewayId=vgw_id, VpcId=vpc_id)
                logger.debug("Detached VPN gateway %s from VPC %s", vgw_id, vpc_id)
            except ClientError as e:
                if not _is_not_found(e):
                    logger.warning("DetachVpnGateway %s from %s failed: %s", vgw_id, vpc_id, e)


def _is_not_found(error: ClientError) -> bool:
    """Return True for an EC2 ``Invalid...ID.NotFound`` error (gateway already gone)."""
    code = error.response.get("Error", {}).get("Code", "")
    return code.endswith(".NotFound")


@resource_handler("AWS::EC2::DHCPOptions", role="delete")
def _delete_dhcp_options(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Disassociate a DHCP options set from its VPCs, then delete it.

    AWS Managed AD (and similar services) creates a non-default DHCP options set
    and associates it with a VPC. ``delete_dhcp_options`` returns
    ``DependencyViolation`` for as long as any VPC references the set, and nothing
    on the reset/cleanup path clears that association — so without this handler the
    set falls to the CCAPI fallback and never deletes, failing the reset closed.

    This handler issues the explicit teardown the CCAPI path cannot: for every VPC
    referencing the set, ``associate_dhcp_options(DhcpOptionsId="default", ...)``
    reverts the VPC to Amazon-provided DNS/DHCP (no replacement set needed), then
    ``delete_dhcp_options`` removes the now-unreferenced set.

    The set associated with the account's **default VPC** is never touched: the
    fast-scan lister already excludes it from discovery, but this handler re-guards
    independently (it now issues mutating calls) by reading the default VPC's
    current ``DhcpOptionsId`` — swapping the default VPC's set would degrade its
    DNS/DHCP config.

    Idempotent and safe to retry: an already-gone set is treated as success, and a
    VPC already on ``"default"`` needs no disassociate. A set still referenced after
    the disassociate pass (e.g. a VPC we could not detach) fails closed.
    """
    dopt_id = resource.identifier
    ec2 = build_client(session, "ec2")

    # Re-guard: never touch the set the default VPC currently uses.
    try:
        default_dopt_ids = {
            vpc["DhcpOptionsId"] for vpc in default_vpcs(ec2) if vpc.get("DhcpOptionsId")
        }
    except ClientError as e:
        return _dhcp_result(
            dopt_id, HandlerStatus.FAILED, f"Could not resolve default VPC DHCP options: {e}"
        )
    if dopt_id in default_dopt_ids:
        return _dhcp_result(
            dopt_id,
            HandlerStatus.SKIPPED,
            "DHCP options set is associated with the default VPC; left untouched",
        )

    # Find every VPC referencing this set and detach it.
    try:
        vpc_ids = _vpcs_using_dhcp_options(ec2, dopt_id)
    except ClientError as e:
        return _dhcp_result(dopt_id, HandlerStatus.FAILED, f"DescribeVpcs failed: {e}")

    for vpc_id in vpc_ids:
        try:
            ec2.associate_dhcp_options(DhcpOptionsId=_DEFAULT_DHCP_OPTIONS, VpcId=vpc_id)
            logger.debug("Disassociated DHCP options %s from VPC %s", dopt_id, vpc_id)
        except ClientError as e:
            if _is_not_found(e):
                continue  # VPC gone — no longer a blocker
            return _dhcp_result(
                dopt_id,
                HandlerStatus.FAILED,
                f"Could not disassociate from VPC {vpc_id}: {e}",
            )

    # Delete the now-unreferenced set.
    try:
        ec2.delete_dhcp_options(DhcpOptionsId=dopt_id)
    except ClientError as e:
        if _is_not_found(e):
            return _dhcp_result(dopt_id, HandlerStatus.SUCCESS, "DHCP options set already gone")
        return _dhcp_result(dopt_id, HandlerStatus.FAILED, f"delete_dhcp_options failed: {e}")

    logger.debug("Deleted DHCP options set %s", dopt_id)
    return _dhcp_result(
        dopt_id, HandlerStatus.SUCCESS, f"Disassociated from {len(vpc_ids)} VPC(s) and deleted"
    )


def _vpcs_using_dhcp_options(ec2: Any, dopt_id: str) -> list[str]:
    """Return the ids of every VPC currently associated with ``dopt_id``."""
    return [
        vpc["VpcId"]
        for page in ec2.get_paginator("describe_vpcs").paginate(
            Filters=[{"Name": "dhcp-options-id", "Values": [dopt_id]}]
        )
        for vpc in page.get("Vpcs", [])
        if vpc.get("VpcId")
    ]


def _dhcp_result(dopt_id: str, status: HandlerStatus, message: str) -> HandlerResult:
    """Build a HandlerResult for the DHCP options delete handler."""
    return HandlerResult(
        resource_id=dopt_id,
        resource_type="AWS::EC2::DHCPOptions",
        action="delete",
        status=status,
        message=message,
    )
