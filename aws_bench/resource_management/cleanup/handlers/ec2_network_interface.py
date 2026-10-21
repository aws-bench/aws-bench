"""Network-interface cleanup handlers for the standalone (reset) path.

A leaked ``AWS::EC2::NetworkInterface`` / ``AWS::EC2::NetworkInterfaceAttachment``
— classically the requester-managed ENIs an ECS/ELB/Lambda-VPC or AppStream
deployment attaches — surfaces in the reset's new-resource sweep. CCAPI cannot
delete a requester-managed in-use ENI, and we are not allowed to manage an
ELB ``ela-attach`` attachment at all.

The stack-delete and VPC pre-delete paths already handle these correctly: when a
requester-managed ENI pins a subnet/VPC they **defer** it (``mark_deferred``) so
the fail-closed verify and orphan scan exclude it, and a later run reaps it after
AWS releases it (requester-managed release takes ~20-40 min — far longer than any
reset would wait). The STANDALONE single-resource reset path was the one path with
no deferral: it fell to the CCAPI fallback, failed, and failed the whole reset on
a resource that self-heals.

These handlers bring the same deferral to the standalone path:

* ENI ``available`` -> delete immediately (genuinely reapable now);
* ENI customer-managed ``in-use`` -> force-detach so it can be deleted next pass,
  then defer this pass;
* ENI requester-managed ``in-use`` -> **defer** (only the owning service releases
  it; a later run reaps it);
* ``NetworkInterfaceAttachment`` -> **defer** (an ``ela-attach`` is ELB-owned and
  unmanageable by us; an ``eni-attach`` is released when its ENI is).

Deferral (not a bounded wait, not fail-closed) matches the stack/VPC paths and the
prod evidence that these are self-healing, not persistent orphans.
"""

from __future__ import annotations

from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.handlers.cross_service import (
    _delete_eni,
    _detach_eni,
)
from aws_bench.resource_management.cleanup.models import HandlerResult, HandlerStatus
from aws_bench.resource_management.deferred import mark_deferred
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)

_ENI_TYPE = "AWS::EC2::NetworkInterface"
_ENI_ATTACHMENT_TYPE = "AWS::EC2::NetworkInterfaceAttachment"
_SUBNET_RT_ASSOC_TYPE = "AWS::EC2::SubnetRouteTableAssociation"
_SUBNET_NACL_ASSOC_TYPE = "AWS::EC2::SubnetNetworkAclAssociation"
_ENI_NOT_FOUND = "InvalidNetworkInterfaceID.NotFound"


def _result(
    identifier: str, resource_type: str, status: HandlerStatus, message: str = ""
) -> HandlerResult:
    return HandlerResult(
        resource_id=identifier,
        resource_type=resource_type,
        action="delete",
        status=status,
        message=message,
    )


def _describe_one(ec2: Any, eni_id: str) -> dict | None:
    """Return the ENI dict, or None if it no longer exists."""
    try:
        pages = ec2.get_paginator("describe_network_interfaces").paginate(
            NetworkInterfaceIds=[eni_id]
        )
        for page in pages:
            for eni in page.get("NetworkInterfaces", []):
                return eni
        return None
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") == _ENI_NOT_FOUND:
            return None
        raise


def _defer_pinned_subnet_associations(ec2: Any, subnet_id: str) -> None:
    """Defer the route-table/NACL associations of a subnet an ENI pins.

    A ``SubnetRouteTableAssociation`` / ``SubnetNetworkAclAssociation`` exists as
    long as its subnet exists; while a requester-managed ENI pins the subnet the
    subnet cannot be deleted, so these associations persist and the standalone
    sweep flags them as new. The prod verdict for ENI-pinned runs was frequently
    these associations, NOT the ENI — so deferring the ENI alone is insufficient.
    Defer them under the exact id form the fast-scan listers report
    (``rtbassoc-...`` / ``aclassoc-...``) so ``exclude_deferred`` matches; a later
    run (after the owner releases the ENI and the subnet is deleted) sees them gone.

    Best-effort: a describe failure is logged and skipped — the association simply
    is not deferred, degrading to today's behavior rather than aborting cleanup.
    """
    if not subnet_id:
        return
    try:
        for page in ec2.get_paginator("describe_route_tables").paginate(
            Filters=[{"Name": "association.subnet-id", "Values": [subnet_id]}]
        ):
            for rt in page.get("RouteTables", []):
                for assoc in rt.get("Associations", []):
                    if assoc.get("SubnetId") == subnet_id and assoc.get("RouteTableAssociationId"):
                        mark_deferred(_SUBNET_RT_ASSOC_TYPE, assoc["RouteTableAssociationId"])
    except (ClientError, BotoCoreError) as e:
        logger.debug("Could not defer route-table associations for subnet '%s': %s", subnet_id, e)
    try:
        for page in ec2.get_paginator("describe_network_acls").paginate(
            Filters=[{"Name": "association.subnet-id", "Values": [subnet_id]}]
        ):
            for nacl in page.get("NetworkAcls", []):
                for assoc in nacl.get("Associations", []):
                    if assoc.get("SubnetId") == subnet_id and assoc.get("NetworkAclAssociationId"):
                        mark_deferred(_SUBNET_NACL_ASSOC_TYPE, assoc["NetworkAclAssociationId"])
    except (ClientError, BotoCoreError) as e:
        logger.debug("Could not defer NACL associations for subnet '%s': %s", subnet_id, e)


@resource_handler(_ENI_TYPE, role="delete")
def _delete_network_interface(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete a reapable ENI, or DEFER a requester-managed one for a later run.

    Mirrors the stack/VPC paths: delete when ``available``, force-detach a
    customer-managed in-use ENI (then defer this pass so it is reaped next run),
    and defer a requester-managed in-use ENI (only its owner can release it). No
    bounded wait and no fail-closed: a requester-managed ENI self-heals ~20-40 min
    after the owner is torn down, which is what the deferral registry is for.
    """
    eni_id = resource.identifier
    try:
        ec2 = build_client(session, "ec2")
        eni = _describe_one(ec2, eni_id)
        if eni is None:
            return _result(
                eni_id, _ENI_TYPE, HandlerStatus.SUCCESS, "Network interface already gone"
            )

        status = eni.get("Status", "")
        if status == "available" and _delete_eni(ec2, eni_id):
            return _result(eni_id, _ENI_TYPE, HandlerStatus.SUCCESS)

        if status != "available" and not eni.get("RequesterManaged", False):
            # Customer-managed in-use: force-detach now; it becomes deletable next
            # pass. Defer this pass so the current run is not failed on it.
            _detach_eni(ec2, eni_id, eni.get("Attachment") or {})

        # available-but-delete-failed, customer-managed just-detached, or
        # requester-managed in-use: defer and let a later run reap it. Also defer
        # the pinned subnet's route-table/NACL associations — while the ENI pins
        # the subnet those associations persist and would otherwise fail the run
        # (they, not the ENI, were the recorded prod verdict).
        mark_deferred(_ENI_TYPE, eni_id)
        _defer_pinned_subnet_associations(ec2, eni.get("SubnetId", ""))
        logger.debug("Deferred ENI '%s' (status=%s) for reap by a later run", eni_id, status)
        return _result(
            eni_id,
            _ENI_TYPE,
            HandlerStatus.SUCCESS,
            "Deferred: requester-managed/in-use ENI (and its subnet's associations) "
            "released by its owner later",
        )
    except ClientError as e:
        return _result(eni_id, _ENI_TYPE, HandlerStatus.FAILED, f"Failed to handle ENI: {e}")
    except BotoCoreError as e:
        return _result(
            eni_id, _ENI_TYPE, HandlerStatus.FAILED, f"Connection error handling ENI: {e}"
        )


@resource_handler(_ENI_ATTACHMENT_TYPE, role="delete")
def _delete_network_interface_attachment(
    resource: Resource, session: boto3.Session
) -> HandlerResult:
    """Defer an ENI attachment — it is released when its owner releases the ENI.

    An ``ela-attach-...`` is ELB-owned ("You are not allowed to manage 'ela-attach'
    attachments"); an ``eni-attach-...`` for a requester-managed ENI is released
    when the owning service releases the interface. Neither is ours to delete
    directly, so defer so the fail-closed verify/orphan scan excludes it and a
    later run (after the ENI is gone) sees it already removed.
    """
    attachment_id = resource.identifier
    mark_deferred(_ENI_ATTACHMENT_TYPE, attachment_id)
    logger.debug("Deferred ENI attachment '%s' (released with its owner's ENI)", attachment_id)
    return _result(
        attachment_id,
        _ENI_ATTACHMENT_TYPE,
        HandlerStatus.SUCCESS,
        "Deferred: attachment released when its owning service releases the ENI",
    )
