"""Cross-service cleanup helpers.

Standalone functions for multi-resource cleanup scenarios, plus
registered failed-resource handlers for Custom:: resources.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import boto3
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError

from aws_bench.constants import DEFAULT_REGION
from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import LOG_TRUNCATE_MEDIUM, Resource
from aws_bench.resource_management.cleanup.handler_registry import (
    failed_resource_handler,
    resource_handler,
)
from aws_bench.resource_management.cleanup.handlers.ipam import (
    _DEPROVISION_BUDGET_SEC,
    PoolDeleteOutcome,
    deprovision_and_delete_pool,
)
from aws_bench.resource_management.cleanup.models import HandlerResult, HandlerStatus, StackResource
from aws_bench.resource_management.deferred import mark_deferred
from aws_bench.resource_management.fastscan.listers.custom_listers import default_vpc_ids
from aws_bench.resource_management.utils.polling import wait_until
from aws_bench.utils.concurrent import build_client, raise_if_shutdown

logger = get_logger(__name__)

_EKS_CLUSTER_TIMEOUT = 600
_EKS_FARGATE_TIMEOUT = 300
_EKS_POLL_INTERVAL = 15

# DMS replication instances delete asynchronously (~5-10 min) and hold a
# RequesterManaged ENI in their subnet group's subnets until gone.
_DMS_INSTANCE_TIMEOUT = 900
_DMS_POLL_INTERVAL = 20

_REDSHIFT_WORKGROUP_TIMEOUT = 600
_REDSHIFT_POLL_INTERVAL = 15

# Interface VPC endpoints release their RequesterManaged ENI only once fully
# deleted; deletion is async, so wait before the ENI/subnet teardown proceeds.
_VPCE_DELETE_TIMEOUT = 300
_VPCE_POLL_INTERVAL = 10

# delete_nat_gateway is async (~1-2 min to reach "deleted"); a NAT gateway holds its
# EIP association and its own ENI until gone, so it must be awaited before EIP release
# and before the ENI reap.
_NAT_GATEWAY_TIMEOUT = 300
_NAT_GATEWAY_POLL_INTERVAL = 10
_NAT_GATEWAY_DELETED_STATE = "deleted"
_NAT_GATEWAY_NOT_FOUND_CODE = "NatGatewayNotFound"
# EIP alloc/association ids AWS reports as already gone (idempotent release).
_EIP_ALLOC_GONE_CODES = ("InvalidAllocationID.NotFound", "InvalidAddress.NotFound")
_EIP_ASSOC_GONE_CODES = ("InvalidAssociationID.NotFound",)
# IGW faults meaning the detach/delete already effectively happened.
_IGW_NOT_FOUND_CODE = "InvalidInternetGatewayID.NotFound"
_IGW_NOT_ATTACHED_CODE = "Gateway.NotAttached"


# ── Async delete-and-poll skeleton ───────────────────────────────────


class _AlreadyGone(Exception):
    """Sentinel a ``submit`` callable raises to mean the resource is already gone.

    Lets a submit wrapper map a service-specific "not found / already deleting"
    fault to the skeleton's already-gone path without leaking the boto3 exception
    type into ``_submit_and_await``.
    """


def _submit_and_await(
    ids: set[str],
    *,
    submit: Callable[[str], None],
    is_gone: Callable[[str], bool],
    timeout: float,
    interval: float,
    label: str,
    already_gone_exc: type[BaseException] | tuple[type[BaseException], ...] = (),
) -> set[str]:
    """Submit an async delete for each id, then poll until all are gone.

    Shared skeleton for resources whose delete is asynchronous and whose
    dependent (e.g. a RequesterManaged ENI) is released only once the resource
    is fully gone: submit every delete first, then poll the surviving set
    against one shared deadline (wall-clock ~max, not sum).

    Args:
        ids: Resource identifiers to delete.
        submit: Issues the delete for one id.
        is_gone: Whether one id is now fully gone (sweeps the surviving set).
        timeout: Seconds to wait for all deletes to finish.
        interval: Seconds between polls.
        label: Human name for log lines (e.g. "EKS cluster").
        already_gone_exc: Exception(s) from ``submit`` meaning "already deleted"
            (recorded as done, not failed).

    Returns:
        Ids that could not be confirmed deleted (submit failed or timed out); a
        non-empty set lets the caller treat it as a failure, not success.
    """
    deleting: list[str] = []
    failed: set[str] = set()
    for id_ in ids:
        try:
            submit(id_)
            logger.debug(f"Deleting {label} '{id_}'")
            deleting.append(id_)
        except already_gone_exc:
            logger.debug(f"{label} '{id_}' already deleted")
        except Exception as e:
            logger.warning(f"Failed to delete {label} '{id_}': {e}")
            failed.add(id_)

    timed_out = _await_gone(
        set(deleting), is_gone=is_gone, timeout=timeout, interval=interval, label=label
    )
    return failed | timed_out


def _await_gone(
    ids: set[str],
    *,
    is_gone: Callable[[str], bool],
    timeout: float,
    interval: float,
    label: str,
) -> set[str]:
    """Poll ``ids`` against one shared deadline until each is gone.

    Use when the delete was already submitted (e.g. a batch delete API) and only
    the wait-until-released is needed. Returns the ids still present at timeout.
    """
    remaining = set(ids)
    deadline = time.monotonic() + timeout
    while remaining and time.monotonic() < deadline:
        raise_if_shutdown()
        _sweep_gone(remaining, is_gone, label)
        if remaining:
            time.sleep(interval)
    # Final sweep: catch an id that went gone during the last sleep before
    # labeling it timed out.
    _sweep_gone(remaining, is_gone, label)
    for id_ in remaining:
        logger.warning(f"{label} '{id_}' deletion timed out")
    return remaining


def _sweep_gone(remaining: set[str], is_gone: Callable[[str], bool], label: str) -> None:
    """Discard from ``remaining`` every id that ``is_gone`` reports as deleted.

    A raising ``is_gone`` is treated as "not gone yet" (logged), not fatal — one
    id's transient error must not abort the poll for the rest of the set.
    """
    for id_ in list(remaining):
        try:
            gone = is_gone(id_)
        except Exception as e:
            logger.debug(f"Checking whether {label} '{id_}' is gone failed: {e}")
            continue
        if gone:
            logger.debug(f"{label} '{id_}' deleted")
            remaining.discard(id_)


# ── EKS ──────────────────────────────────────────────────────────────


def delete_eks_clusters(session: boto3.Session, cluster_names: set[str]) -> set[str]:
    """Delete EKS clusters (nodegroups → Fargate profiles → cluster) and wait.

    delete_cluster is async (~10 min), so submit every cluster first then poll
    the set against one shared deadline — the cluster-delete wait is batched
    (wall-clock ~max, not sum); per-cluster sub-resource teardown is not.

    Returns clusters that could not be confirmed deleted (submit failed or timed
    out); a non-empty set lets the caller treat it as a failure, not success.
    """
    eks = build_client(session, "eks")

    # Submit: sub-resources (intra-cluster dependency) then the cluster delete.
    def _submit(name: str) -> None:
        _delete_eks_sub_resources(eks, name)
        eks.delete_cluster(name=name)

    return _submit_and_await(
        cluster_names,
        submit=_submit,
        is_gone=lambda name: _make_cluster_gone_check(eks, name)(),
        timeout=_EKS_CLUSTER_TIMEOUT,
        interval=_EKS_POLL_INTERVAL,
        label="EKS cluster",
        already_gone_exc=eks.exceptions.ResourceNotFoundException,
    )


def _delete_eks_sub_resources(eks: Any, cluster_name: str) -> None:
    _delete_eks_resource_group(
        eks,
        cluster_name,
        "list_nodegroups",
        "nodegroups",
        "delete_nodegroup",
        "nodegroupName",
        "nodegroup",
        timeout=_EKS_CLUSTER_TIMEOUT,
    )
    _delete_eks_resource_group(
        eks,
        cluster_name,
        "list_fargate_profiles",
        "fargateProfileNames",
        "delete_fargate_profile",
        "fargateProfileName",
        "Fargate profile",
        timeout=_EKS_FARGATE_TIMEOUT,
    )


def _delete_eks_resource_group(
    eks: Any,
    cluster_name: str,
    list_method: str,
    list_key: str,
    delete_method: str,
    delete_param: str,
    label: str,
    *,
    timeout: float,
) -> None:
    try:
        names = getattr(eks, list_method)(clusterName=cluster_name).get(list_key, [])
    except Exception as exc:
        logger.warning("Failed to list %ss for '%s': %s", label, cluster_name, exc)
        names = []
    for name in names:
        try:
            getattr(eks, delete_method)(clusterName=cluster_name, **{delete_param: name})
            logger.debug("Deleting EKS %s '%s/%s'", label, cluster_name, name)
        except Exception as e:
            logger.warning("Could not delete %s '%s/%s': %s", label, cluster_name, name, e)
    wait_until(
        lambda: _eks_sub_resources_gone(eks, cluster_name, list_method, list_key),
        timeout=timeout,
        interval=_EKS_POLL_INTERVAL,
    )


def _eks_sub_resources_gone(eks: Any, cluster_name: str, method: str, key: str) -> bool:
    try:
        return not getattr(eks, method)(clusterName=cluster_name).get(key)
    except eks.exceptions.ResourceNotFoundException:
        return True
    except Exception as e:
        logger.debug("Polling %s for '%s' failed: %s", method, cluster_name, e)
        return False


def _make_cluster_gone_check(eks: Any, name: str):
    def check() -> bool:
        try:
            eks.describe_cluster(name=name)
            return False
        except eks.exceptions.ResourceNotFoundException:
            return True
        except Exception as e:
            logger.debug("Polling cluster '%s' status failed: %s", name, e)
            return False

    return check


# ── DMS replication instances ────────────────────────────────────────


# delete_replication_instance fault meaning the instance is already gone (skip the
# wait — nothing to release). NOT InvalidResourceStateFault: that means the instance
# still EXISTS (mid-create/modify, already-deleting, or a task is still attached), so
# it must be polled — if it is merely already-deleting it will confirm gone, and if it
# genuinely refuses to delete it must surface as unresolved, never a silent success.
_DMS_ALREADY_GONE_CODE = "ResourceNotFoundFault"
_DMS_STILL_PRESENT_CODE = "InvalidResourceStateFault"


def delete_dms_instances(session: boto3.Session, instance_arns: set[str]) -> set[str]:
    """Delete DMS replication instances and wait until they are gone.

    A replication instance holds a RequesterManaged ENI in each subnet of its
    replication subnet group; that ENI is not force-detachable, so the only way
    to free the subnet (and let its VPC delete) is to delete the owning instance.
    delete_replication_instance is async (~5-10 min), so submit every instance
    first then poll the set against one shared deadline.

    Returns instances that could not be confirmed deleted (submit failed or timed
    out); a non-empty set lets the caller treat it as a failure, not success.
    """
    dms = build_client(session, "dms")

    def _submit(arn: str) -> None:
        try:
            dms.delete_replication_instance(ReplicationInstanceArn=arn)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == _DMS_ALREADY_GONE_CODE:
                raise _AlreadyGone from e
            # Still present (already-deleting / mid-op / task attached): fall
            # through so the arn is polled — a real absence confirms gone, a
            # persistent one times out and is reported, never silently dropped.
            if code == _DMS_STILL_PRESENT_CODE:
                logger.debug(
                    f"DMS replication instance '{arn}' not yet deletable ({code}); will poll"
                )
                return
            raise

    return _submit_and_await(
        instance_arns,
        submit=_submit,
        is_gone=lambda arn: _dms_instance_gone(dms, arn),
        timeout=_DMS_INSTANCE_TIMEOUT,
        interval=_DMS_POLL_INTERVAL,
        label="DMS replication instance",
        already_gone_exc=_AlreadyGone,
    )


def _dms_instance_gone(dms: Any, arn: str) -> bool:
    """Whether the replication instance is fully gone (transient errors -> not yet)."""
    try:
        resp = dms.describe_replication_instances(
            Filters=[{"Name": "replication-instance-arn", "Values": [arn]}]
        )
    except dms.exceptions.ResourceNotFoundFault:
        return True
    except Exception as e:
        logger.debug(f"Polling DMS replication instance '{arn}' failed: {e}")
        return False
    return not resp.get("ReplicationInstances")


def _delete_dms_instances_in_vpcs(session: boto3.Session, vpc_ids: list[str]) -> None:
    """Delete DMS replication instances whose subnet group lives in ``vpc_ids``.

    A replication instance is placed in a replication subnet group, whose subnets
    belong to a VPC. When that VPC is being torn down, the instance's ENI pins the
    subnet in DELETE_FAILED, so the instance must go first. Failures are logged and
    swallowed — discovery is best-effort and must not abort the VPC cleanup.
    """
    try:
        dms = build_client(session, "dms")
        ec2 = build_client(session, "ec2")

        # Subnets that currently live in the target VPCs — resolved once by
        # vpc-id filter, so a sibling subnet already deleted mid-teardown can't
        # poison the lookup (describe_subnets by explicit id fails the whole call
        # on one stale id).
        subnets_in_vpc = _subnets_in_vpcs(ec2, vpc_ids)

        # Which replication subnet groups still have a subnet in the target VPCs?
        groups_in_vpc: set[str] = set()
        for page in dms.get_paginator("describe_replication_subnet_groups").paginate():
            for group in page.get("ReplicationSubnetGroups", []):
                group_subnets = {
                    s["SubnetIdentifier"]
                    for s in group.get("Subnets", [])
                    if s.get("SubnetIdentifier")
                }
                if group_subnets & subnets_in_vpc:
                    groups_in_vpc.add(group["ReplicationSubnetGroupIdentifier"])

        if not groups_in_vpc:
            return

        # Delete every replication instance placed in one of those subnet groups.
        to_delete: set[str] = set()
        for page in dms.get_paginator("describe_replication_instances").paginate():
            for instance in page.get("ReplicationInstances", []):
                group = instance.get("ReplicationSubnetGroup", {})
                if group.get("ReplicationSubnetGroupIdentifier") in groups_in_vpc:
                    to_delete.add(instance["ReplicationInstanceArn"])

        if to_delete:
            delete_dms_instances(session, to_delete)
    except Exception as e:
        logger.warning(f"DMS replication instance discovery for VPCs failed: {e}")


def _subnets_in_vpcs(ec2: Any, vpc_ids: list[str]) -> set[str]:
    """Return the ids of subnets currently in ``vpc_ids``.

    Filtered by vpc-id (not queried by explicit subnet id) so a subnet already
    deleted during VPC teardown never fails the call and hides a sibling that is
    still present and pinned.
    """
    try:
        subnet_ids: set[str] = set()
        for page in ec2.get_paginator("describe_subnets").paginate(
            Filters=[{"Name": "vpc-id", "Values": vpc_ids}]
        ):
            for subnet in page.get("Subnets", []):
                subnet_ids.add(subnet["SubnetId"])
        return subnet_ids
    except Exception as e:
        logger.debug(f"Could not describe subnets for VPCs {vpc_ids}: {e}")
        return set()


# ── VPC endpoints ────────────────────────────────────────────────────


def _delete_vpc_endpoints_in_vpcs(session: boto3.Session, vpc_ids: list[str]) -> None:
    """Delete every VPC endpoint in ``vpc_ids`` so the VPC and subnets can delete.

    An agent-created endpoint (not part of the scenario template) wedges the
    stack two ways: an interface endpoint places a RequesterManaged ENI in each
    subnet (not force-detachable, pins the subnet), and a gateway endpoint holds
    a route-table association (pins the VPC). Both types are deleted so neither
    the subnet nor the VPC is left in DELETE_FAILED, and we wait until they are
    gone — the interface ENI releases only on full deletion, so returning early
    would let the very next ENI/subnet teardown see a still-pinned subnet.
    Failures are logged and swallowed — discovery is best-effort and must not
    abort the VPC cleanup.
    """
    try:
        ec2 = build_client(session, "ec2")
        endpoint_ids = [
            endpoint["VpcEndpointId"]
            for page in ec2.get_paginator("describe_vpc_endpoints").paginate(
                Filters=[{"Name": "vpc-id", "Values": vpc_ids}]
            )
            for endpoint in page.get("VpcEndpoints", [])
        ]
        if not endpoint_ids:
            return
        logger.debug(f"Deleting {len(endpoint_ids)} VPC endpoint(s): {endpoint_ids}")

        # delete_vpc_endpoints does not raise on a per-endpoint failure — it
        # returns them in Unsuccessful; surface those instead of silently
        # leaving a pinned subnet.
        resp = ec2.delete_vpc_endpoints(VpcEndpointIds=endpoint_ids)
        for item in resp.get("Unsuccessful", []):
            err = item.get("Error", {})
            logger.warning(
                f"Failed to delete VPC endpoint '{item.get('ResourceId')}': "
                f"{err.get('Code')} {err.get('Message')}"
            )

        # Wait until the endpoints are gone so their ENIs are released before the
        # caller discovers ENIs / deletes subnets (the batch delete above already
        # submitted them, so only the wait remains).
        _await_gone(
            set(endpoint_ids),
            is_gone=lambda vpce_id: _vpc_endpoint_gone(ec2, vpce_id),
            timeout=_VPCE_DELETE_TIMEOUT,
            interval=_VPCE_POLL_INTERVAL,
            label="VPC endpoint",
        )
    except Exception as e:
        logger.warning(f"VPC endpoint discovery for VPCs failed: {e}")


def _vpc_endpoint_gone(ec2: Any, vpce_id: str) -> bool:
    """Whether the VPC endpoint is gone or fully deleted (transient errors -> not yet)."""
    try:
        resp = ec2.describe_vpc_endpoints(VpcEndpointIds=[vpce_id])
    except ClientError as e:
        # AWS reports an unknown endpoint id as gone.
        if "NotFound" in e.response.get("Error", {}).get("Code", ""):
            return True
        logger.debug(f"Polling VPC endpoint '{vpce_id}' failed: {e}")
        return False
    except Exception as e:
        logger.debug(f"Polling VPC endpoint '{vpce_id}' failed: {e}")
        return False
    endpoints = resp.get("VpcEndpoints", [])
    # EC2 reports the state as "Deleted" (capitalized); compare case-insensitively
    # so a lingering already-deleted endpoint exits at once instead of waiting out
    # the full timeout.
    return not endpoints or all(e.get("State", "").lower() == "deleted" for e in endpoints)


# ── NAT gateway / EIP / IGW "mapped public address" wedge ────────────
#
# CCAPI's ordered level map omits IGW/NAT/EIP/VPCGatewayAttachment, so it deletes
# them first with no detach and no dependency ordering. DetachInternetGateway then
# fails with DependencyViolation "has some mapped public address(es)" because a NAT
# gateway / EIP is still mapped in the VPC — the IGW (and its VPC) survive forever.
# This clears the wedge in dependency order: NAT gateways → EIPs → IGW detach+delete.


@dataclass
class VpcPublicAddressWedgeResult:
    """Outcome of clearing the NAT/EIP/IGW "mapped public address" wedge in some VPCs."""

    nat_deleted: list[str] = field(default_factory=list)
    eips_released: list[str] = field(default_factory=list)
    igws_deleted: list[str] = field(default_factory=list)
    remaining: list[str] = field(default_factory=list)
    """NAT gateways / EIPs / IGWs that did not clear (delete failed or timed out).

    Non-empty means the wedge is NOT fully cleared; the caller must surface these so
    the fail-closed reset logic fails rather than absorbing a surviving orphan."""

    @property
    def cleared_any(self) -> bool:
        """True if at least one NAT gateway, EIP, or IGW was removed."""
        return bool(self.nat_deleted or self.eips_released or self.igws_deleted)


def clear_vpc_public_address_wedge(
    session: boto3.Session, vpc_ids: list[str], *, region: str | None = None
) -> VpcPublicAddressWedgeResult:
    """Delete NAT gateways, release EIPs, then detach+delete IGWs in ``vpc_ids``.

    Dependency order clears the "mapped public address(es)" DependencyViolation:
    NAT first (async, awaited to ``deleted`` — it holds an ENI + EIP), then surviving
    VPC EIPs, then each attached IGW. Idempotent. Anything that fails to clear lands
    in ``remaining`` rather than being reported as success over a survivor.

    Args:
        session: Boto3 session scoped to the target account.
        vpc_ids: VPCs whose NAT/EIP/IGW should be cleared.
        region: EC2 region; falls back to the session's region.

    Returns:
        What was removed and what remained (non-empty ``remaining`` = not fully cleared).
    """
    vpc_ids = [v for v in vpc_ids if v]
    result = VpcPublicAddressWedgeResult()
    if not vpc_ids:
        return result

    ec2 = build_client(session, "ec2", region_name=region)

    # Phase 1: NAT gateways (hold the EIP association and their own ENI) go first.
    _clear_nat_gateways(ec2, vpc_ids, result)
    # Phase 2: release any VPC-scoped EIP the NAT gateway didn't auto-release.
    _release_vpc_eips(ec2, vpc_ids, result)
    # Phase 3: detach + delete the now-unblocked IGWs.
    _clear_internet_gateways(ec2, vpc_ids, result)

    if result.cleared_any or result.remaining:
        logger.info(
            f"Public-address wedge for {vpc_ids}: {len(result.nat_deleted)} NAT deleted, "
            f"{len(result.eips_released)} EIP released, {len(result.igws_deleted)} IGW deleted, "
            f"{len(result.remaining)} still present"
        )
    return result


def clear_igw_public_address_wedge(
    session: boto3.Session, igw_ids: list[str], *, region: str | None = None
) -> VpcPublicAddressWedgeResult:
    """Resolve each IGW's attached VPC, then clear the NAT/EIP/IGW wedge for those VPCs.

    Entry point for the ``AWS::EC2::InternetGateway`` pre-delete hook, where the IGW
    itself (not the VPC) is the flagged resource. Maps ``igw_ids`` to their VPC via
    ``describe_internet_gateways`` Attachments and delegates to
    :func:`clear_vpc_public_address_wedge`, which detaches + deletes the IGW once NAT
    gateways and EIPs are gone. A detached IGW has no VPC wedge (CCAPI can delete it
    directly), so it resolves to no VPC and is a no-op here.
    """
    igw_ids = [i for i in igw_ids if i]
    if not igw_ids:
        return VpcPublicAddressWedgeResult()
    ec2 = build_client(session, "ec2", region_name=region)
    vpc_ids = _vpcs_for_igws(ec2, igw_ids)
    # Never detach/delete the account's default-VPC IGW: it is AWS-created, not a leak.
    try:
        protected_vpc_ids = default_vpc_ids(ec2)
    except (ClientError, BotoCoreError) as e:
        logger.warning(f"DescribeVpcs (is-default) failed; not protecting default VPCs: {e}")
        protected_vpc_ids = set()
    vpc_ids = [v for v in vpc_ids if v not in protected_vpc_ids]
    if not vpc_ids:
        return VpcPublicAddressWedgeResult()
    return clear_vpc_public_address_wedge(session, vpc_ids, region=region)


def _vpcs_for_igws(ec2: Any, igw_ids: list[str]) -> list[str]:
    """Return the VPC ids the given IGWs are attached to (deduped; [] on error)."""
    try:
        resp = ec2.describe_internet_gateways(InternetGatewayIds=igw_ids)
    except Exception as e:
        logger.warning(f"Could not resolve VPCs for IGWs {igw_ids}: {e}")
        return []
    vpc_ids: list[str] = []
    for igw in resp.get("InternetGateways", []):
        for attachment in igw.get("Attachments", []):
            vpc_id = attachment.get("VpcId")
            if vpc_id and vpc_id not in vpc_ids:
                vpc_ids.append(vpc_id)
    return vpc_ids


def _clear_nat_gateways(ec2: Any, vpc_ids: list[str], result: VpcPublicAddressWedgeResult) -> None:
    """Delete every non-deleted NAT gateway in ``vpc_ids`` and await terminal ``deleted``."""
    nat_ids = _describe_nat_gateway_ids(ec2, vpc_ids)
    if not nat_ids:
        return

    def _submit(nat_id: str) -> None:
        try:
            ec2.delete_nat_gateway(NatGatewayId=nat_id)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code", "") == _NAT_GATEWAY_NOT_FOUND_CODE:
                raise _AlreadyGone from e
            raise

    unresolved = _submit_and_await(
        nat_ids,
        submit=_submit,
        is_gone=lambda nat_id: _nat_gateway_gone(ec2, nat_id),
        timeout=_NAT_GATEWAY_TIMEOUT,
        interval=_NAT_GATEWAY_POLL_INTERVAL,
        label="NAT gateway",
        already_gone_exc=_AlreadyGone,
    )
    result.nat_deleted.extend(sorted(nat_ids - unresolved))
    result.remaining.extend(sorted(unresolved))


def _describe_nat_gateway_ids(ec2: Any, vpc_ids: list[str]) -> set[str]:
    """Return ids of NAT gateways in ``vpc_ids`` that are not already deleted/deleting."""
    try:
        nat_ids: set[str] = set()
        for page in ec2.get_paginator("describe_nat_gateways").paginate(
            Filter=[{"Name": "vpc-id", "Values": vpc_ids}]
        ):
            for nat in page.get("NatGateways", []):
                if nat.get("State") not in ("deleted", "deleting") and nat.get("NatGatewayId"):
                    nat_ids.add(nat["NatGatewayId"])
        return nat_ids
    except Exception as e:
        logger.warning(f"NAT gateway discovery for {vpc_ids} failed: {e}")
        return set()


def _nat_gateway_gone(ec2: Any, nat_id: str) -> bool:
    """Whether a NAT gateway is fully ``deleted`` (transient errors -> not yet gone)."""
    try:
        resp = ec2.describe_nat_gateways(NatGatewayIds=[nat_id])
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") == _NAT_GATEWAY_NOT_FOUND_CODE:
            return True
        logger.debug(f"Polling NAT gateway '{nat_id}' failed: {e}")
        return False
    except Exception as e:
        logger.debug(f"Polling NAT gateway '{nat_id}' failed: {e}")
        return False
    nats = resp.get("NatGateways", [])
    return not nats or all(n.get("State") == _NAT_GATEWAY_DELETED_STATE for n in nats)


def _release_vpc_eips(ec2: Any, vpc_ids: list[str], result: VpcPublicAddressWedgeResult) -> None:
    """Disassociate (if associated) then release every EIP still mapped into ``vpc_ids``.

    A NAT-gateway-owned EIP is auto-released when the NAT gateway is deleted, so only
    the addresses that survive Phase 1 are handled here — an EIP still associated to a
    network interface in a target VPC. Each address that fails to release lands in
    ``remaining`` so a surviving public address is never reported as cleared.
    """
    for addr in _describe_vpc_eips(ec2, vpc_ids):
        alloc_id = addr.get("AllocationId")
        if not alloc_id:
            continue
        if not _disassociate_eip(ec2, addr):
            result.remaining.append(alloc_id)
            continue
        if _release_eip(ec2, alloc_id):
            result.eips_released.append(alloc_id)
        else:
            result.remaining.append(alloc_id)


def _describe_vpc_eips(ec2: Any, vpc_ids: list[str]) -> list[dict]:
    """Return EIPs mapped to a network interface in ``vpc_ids`` (or [] on error).

    Scoped by the ``network-interface-vpc-id`` filter, which returns only the
    associated addresses — exactly the "mapped public address(es)" pinning the VPC.
    An unassociated EIP has no VPC and cannot be attributed to this teardown.
    """
    try:
        return ec2.describe_addresses(
            Filters=[{"Name": "network-interface-vpc-id", "Values": vpc_ids}]
        ).get("Addresses", [])
    except Exception as e:
        logger.warning(f"EIP discovery for {vpc_ids} failed: {e}")
        return []


def _disassociate_eip(ec2: Any, addr: dict) -> bool:
    """Disassociate an EIP if it has an association. True if now free (or was already)."""
    association_id = addr.get("AssociationId")
    if not association_id:
        return True
    try:
        ec2.disassociate_address(AssociationId=association_id)
        logger.info(f"Disassociated EIP '{addr.get('AllocationId')}' (assoc {association_id})")
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") in _EIP_ASSOC_GONE_CODES:
            return True
        logger.warning(f"Could not disassociate EIP '{addr.get('AllocationId')}': {e}")
        return False
    except Exception as e:
        logger.warning(f"Could not disassociate EIP '{addr.get('AllocationId')}': {e}")
        return False


def _release_eip(ec2: Any, alloc_id: str) -> bool:
    """Release an EIP by allocation id. True if released or already gone."""
    try:
        ec2.release_address(AllocationId=alloc_id)
        logger.info(f"Released EIP '{alloc_id}'")
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") in _EIP_ALLOC_GONE_CODES:
            return True
        logger.warning(f"Could not release EIP '{alloc_id}': {e}")
        return False
    except Exception as e:
        logger.warning(f"Could not release EIP '{alloc_id}': {e}")
        return False


def _clear_internet_gateways(
    ec2: Any, vpc_ids: list[str], result: VpcPublicAddressWedgeResult
) -> None:
    """Detach then delete every IGW attached to ``vpc_ids`` (now unblocked by NAT/EIP)."""
    for igw_id, attached_vpc in _describe_attached_igws(ec2, vpc_ids):
        if not _detach_internet_gateway(ec2, igw_id, attached_vpc):
            result.remaining.append(igw_id)
            continue
        if _delete_internet_gateway(ec2, igw_id):
            result.igws_deleted.append(igw_id)
        else:
            result.remaining.append(igw_id)


def _describe_attached_igws(ec2: Any, vpc_ids: list[str]) -> list[tuple[str, str]]:
    """Return (igw_id, vpc_id) for every IGW attached to one of ``vpc_ids`` (or [] on error)."""
    try:
        vpc_set = set(vpc_ids)
        pairs: list[tuple[str, str]] = []
        for page in ec2.get_paginator("describe_internet_gateways").paginate(
            Filters=[{"Name": "attachment.vpc-id", "Values": vpc_ids}]
        ):
            for igw in page.get("InternetGateways", []):
                igw_id = igw.get("InternetGatewayId")
                if not igw_id:
                    continue
                for attachment in igw.get("Attachments", []):
                    if attachment.get("VpcId") in vpc_set:
                        pairs.append((igw_id, attachment["VpcId"]))
        return pairs
    except Exception as e:
        logger.warning(f"IGW discovery for {vpc_ids} failed: {e}")
        return []


def _detach_internet_gateway(ec2: Any, igw_id: str, vpc_id: str) -> bool:
    """Detach an IGW from its VPC. True if detached or already detached/gone."""
    try:
        ec2.detach_internet_gateway(InternetGatewayId=igw_id, VpcId=vpc_id)
        logger.info(f"Detached IGW '{igw_id}' from '{vpc_id}'")
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in (_IGW_NOT_ATTACHED_CODE, _IGW_NOT_FOUND_CODE):
            return True
        logger.warning(f"Could not detach IGW '{igw_id}' from '{vpc_id}': {e}")
        return False
    except Exception as e:
        logger.warning(f"Could not detach IGW '{igw_id}' from '{vpc_id}': {e}")
        return False


def _delete_internet_gateway(ec2: Any, igw_id: str) -> bool:
    """Delete a detached IGW. True if deleted or already gone."""
    try:
        ec2.delete_internet_gateway(InternetGatewayId=igw_id)
        logger.info(f"Deleted IGW '{igw_id}'")
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") == _IGW_NOT_FOUND_CODE:
            return True
        logger.warning(f"Could not delete IGW '{igw_id}': {e}")
        return False
    except Exception as e:
        logger.warning(f"Could not delete IGW '{igw_id}': {e}")
        return False


# ── Redshift Serverless ──────────────────────────────────────────────


def cleanup_redshift(session: boto3.Session, region: str) -> None:
    """Delete Redshift Serverless workgroups then namespaces in one region.

    A failed Custom::AWS resource leaves its orphans in the stack's own region
    (the resource's backing Lambda is regional), so this cleans only that
    region rather than scanning every enabled region. Intra-region order —
    workgroups must drain before namespaces can be deleted — is preserved.
    """
    try:
        rs = build_client(session, "redshift-serverless", region_name=region)
        _delete_redshift_resources(
            rs,
            region,
            "list_workgroups",
            "workgroups",
            "workgroupName",
            "delete_workgroup",
            "workgroup",
        )
        # Wait for workgroups to finish deleting before deleting namespaces
        wait_until(
            lambda: not rs.list_workgroups().get("workgroups"),
            timeout=_REDSHIFT_WORKGROUP_TIMEOUT,
            interval=_REDSHIFT_POLL_INTERVAL,
        )
        _delete_redshift_resources(
            rs,
            region,
            "list_namespaces",
            "namespaces",
            "namespaceName",
            "delete_namespace",
            "namespace",
        )
    except EndpointConnectionError as e:
        # Region doesn't offer Redshift Serverless — expected, nothing to clean.
        logger.debug("Redshift Serverless not available in %s: %s", region, e)
    except Exception as e:
        # A real fault may leave orphans — surface it, don't hide as "not available".
        logger.warning("Redshift Serverless cleanup failed in %s: %s", region, e)


def _delete_redshift_resources(
    client,
    region: str,
    list_method: str,
    list_key: str,
    name_key: str,
    delete_method: str,
    label: str,
) -> None:
    for item in getattr(client, list_method)().get(list_key, []):
        # .get not [] so a malformed item skips only itself, not the whole batch.
        name = item.get(name_key)
        if not name:
            logger.warning(
                "Redshift %s entry missing '%s' in %s: %r", label, name_key, region, item
            )
            continue
        try:
            getattr(client, delete_method)(**{name_key: name})
            logger.debug("Deleted Redshift %s '%s' in %s", label, name, region)
        except Exception as e:
            logger.warning("Failed to delete Redshift %s '%s' in %s: %s", label, name, region, e)


# ── Stuck Custom::* resource dependencies ────────────────────────────


def cleanup_stuck_custom_resource_deps(
    session: boto3.Session,
    resources: list[StackResource],
    region: str,
) -> None:
    """Delete Lambda/IAM left behind by failed Custom:: resources."""
    try:
        lam = build_client(session, "lambda", region_name=region)
        iam = build_client(session, "iam")
    except Exception as exc:
        logger.warning(
            "Failed to create clients for custom resource cleanup in %s: %s", region, exc
        )
        return

    for resource in resources:
        rtype, pid = resource.resource_type, resource.physical_id
        if not pid:
            continue
        if rtype == "AWS::Lambda::Function":
            try:
                lam.delete_function(FunctionName=pid)
                logger.debug("Deleted stuck Lambda '%s'", pid[:LOG_TRUNCATE_MEDIUM])
            except Exception as e:
                logger.warning("Failed to delete Lambda '%s': %s", pid[:LOG_TRUNCATE_MEDIUM], e)
        elif rtype == "AWS::IAM::Role":
            _force_delete_iam_role(iam, pid)
        else:
            logger.debug("Skipping unsupported resource type '%s' for custom cleanup", rtype)


def detach_iam_role_dependencies(iam: BaseClient, role_name: str) -> None:
    """Detach managed/inline policies and remove the role from instance profiles.

    These are the dependencies that block a plain ``DeleteRole`` — most notably
    an instance-profile membership, which makes the delete fail with "Cannot
    delete entity, must remove roles from instance profile first". Used both as
    the prep step before force-deleting a stuck role and as the IAM::Role prepare
    handler on the reset new-resource path.
    """
    # Detach managed policies
    paginator = iam.get_paginator("list_attached_role_policies")
    for page in paginator.paginate(RoleName=role_name):
        for policy in page.get("AttachedPolicies", []):
            iam.detach_role_policy(RoleName=role_name, PolicyArn=policy["PolicyArn"])

    # Delete inline policies
    paginator = iam.get_paginator("list_role_policies")
    for page in paginator.paginate(RoleName=role_name):
        for policy_name in page.get("PolicyNames", []):
            iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)

    # Remove from instance profiles
    paginator = iam.get_paginator("list_instance_profiles_for_role")
    for page in paginator.paginate(RoleName=role_name):
        for profile in page.get("InstanceProfiles", []):
            iam.remove_role_from_instance_profile(
                RoleName=role_name, InstanceProfileName=profile["InstanceProfileName"]
            )


def _force_delete_iam_role(iam: Any, role_name: str) -> None:
    try:
        detach_iam_role_dependencies(iam, role_name)
        iam.delete_role(RoleName=role_name)
        logger.debug("Deleted stuck IAM role '%s'", role_name[:LOG_TRUNCATE_MEDIUM])
    except Exception as e:
        logger.warning("Failed to delete IAM role '%s': %s", role_name[:LOG_TRUNCATE_MEDIUM], e)


# ── VPC dynamic resource discovery ───────────────────────────────────


def discover_vpc_dynamic_resources(vpc_ids: list[str], session: boto3.Session) -> list:
    """Find EKS clusters, EFS mount targets, ENIs, and security groups attached to VPCs."""
    ec2 = build_client(session, "ec2")
    resources: list = []

    # Delete EKS clusters in these VPCs first — they hold ENIs and SGs
    _delete_eks_clusters_in_vpcs(session, vpc_ids)

    # Delete DMS replication instances in these VPCs before ENI discovery — their
    # RequesterManaged ENIs are not force-detachable and would otherwise pin the
    # subnet in DELETE_FAILED.
    _delete_dms_instances_in_vpcs(session, vpc_ids)

    # Delete VPC endpoints in these VPCs for the same reason — an agent-created
    # endpoint (not in the scenario template) pins the subnet (interface ENI) or
    # the VPC (gateway route-table association).
    _delete_vpc_endpoints_in_vpcs(session, vpc_ids)

    # Clear the NAT/EIP/IGW "mapped public address" wedge before the ENI reap: a NAT
    # gateway holds its own ENI (which would otherwise block the reap) plus its EIP,
    # and CCAPI's level map omits IGW/NAT/EIP so DetachInternetGateway would fail with
    # DependencyViolation. Ordered NAT → EIP → IGW clears it; orphans surface below.
    clear_vpc_public_address_wedge(session, vpc_ids)

    # Discover load balancers in these VPCs. An ALB/NLB created out-of-band by an
    # in-cluster controller (e.g. the EKS Auto Mode / AWS Load Balancer Controller
    # provisioning an ALB for a Kubernetes Ingress) is NOT a CloudFormation resource
    # and is never garbage-collected when the cluster is deleted, so it lingers and
    # its service-managed ENIs pin the subnets in DELETE_FAILED. Returned (not deleted
    # inline) so the elbv2 custom handler tears each down via the elbv2 API and WAITS
    # for terminal deletion (ENI release) in the custom-delete phase, which runs before
    # CCAPI deletes the subnets/VPC.
    resources.extend(_discover_load_balancers_in_vpcs(session, vpc_ids))

    # Discover EFS mount targets via API (must be deleted before their ENIs can be released)
    resources.extend(_discover_efs_mount_targets(session, vpc_ids))

    # Actively reap leftover ENIs: available → delete, customer-managed in-use →
    # detach. Requester-managed ENIs (EKS X-ENIs, and the LB interfaces whose load
    # balancer the elbv2 handler tears down in a later phase) are left for their
    # owner — hence wait_for_release=False here; StackDeleter does the patient wait.
    sg_resources = _discover_security_groups(ec2, vpc_ids)
    reap = reap_vpc_enis(session, vpc_ids, wait_for_release=False)
    if reap.remaining:
        logger.debug(
            "VPC reap left %d requester-managed ENI(s) for their owner to release",
            len(reap.remaining),
        )
        # These leftover ENIs are service-owned (Lambda Hyperplane, EKS X-ENIs,
        # ELB / VPC-endpoint interfaces) and are released by AWS only *after* the
        # owning resource is gone — for which deletes have already been issued above.
        # That release is asynchronous (Lambda Hyperplane ENIs routinely take 20-40
        # min, well past StackDeleter's bounded wait), so the ENIs — and the VPC,
        # subnets, and non-default security groups they pin (DependencyViolation) —
        # can still be present when this run finishes. They ARE eventually deletable,
        # so defer them (mirrors the Lambda@Edge deferral in handlers/lambda_.py): the
        # post-cleanup orphan scan excludes deferred ids instead of failing the run,
        # and a later run reaps them once the owner has released the ENIs.
        for vpc_id in vpc_ids:
            mark_deferred("AWS::EC2::VPC", vpc_id)
        for subnet_id in _subnets_in_vpcs(ec2, vpc_ids):
            mark_deferred("AWS::EC2::Subnet", subnet_id)
        for sg in sg_resources:
            mark_deferred("AWS::EC2::SecurityGroup", sg.identifier)
        for eni_id in reap.remaining:
            mark_deferred("AWS::EC2::NetworkInterface", eni_id)
    resources.extend(sg_resources)
    return resources


def _discover_load_balancers_in_vpcs(session: boto3.Session, vpc_ids: list[str]) -> list:
    """Find ELBv2 (ALB/NLB) and classic ELB load balancers in ``vpc_ids``."""
    vpc_set = set(vpc_ids)
    resources: list = []

    # ELBv2 (Application / Network / Gateway load balancers)
    try:
        elbv2 = build_client(session, "elbv2")
        for page in elbv2.get_paginator("describe_load_balancers").paginate():
            for lb in page.get("LoadBalancers", []):
                if lb.get("VpcId") in vpc_set and lb.get("LoadBalancerArn"):
                    resources.append(
                        Resource(
                            type="AWS::ElasticLoadBalancingV2::LoadBalancer",
                            identifier=lb["LoadBalancerArn"],
                        )
                    )
    except Exception as e:
        logger.warning("ELBv2 load balancer discovery for VPCs failed: %s", e)

    # Classic ELB (elb): CCAPI supports AWS::ElasticLoadBalancing::LoadBalancer.
    try:
        elb = build_client(session, "elb")
        for page in elb.get_paginator("describe_load_balancers").paginate():
            for lb in page.get("LoadBalancerDescriptions", []):
                if lb.get("VPCId") in vpc_set and lb.get("LoadBalancerName"):
                    resources.append(
                        Resource(
                            type="AWS::ElasticLoadBalancing::LoadBalancer",
                            identifier=lb["LoadBalancerName"],
                        )
                    )
    except Exception as e:
        logger.warning("Classic ELB discovery for VPCs failed: %s", e)

    if resources:
        logger.debug("Discovered %d load balancer(s) pinning VPC ENIs", len(resources))
    return resources


def _delete_eks_clusters_in_vpcs(session: boto3.Session, vpc_ids: list[str]) -> None:
    """Find and delete EKS clusters whose VPC is in vpc_ids."""
    try:
        eks = build_client(session, "eks")
        cluster_names = eks.list_clusters().get("clusters", [])
        vpc_set = set(vpc_ids)
        to_delete: set[str] = set()
        for name in cluster_names:
            try:
                desc = eks.describe_cluster(name=name)
                cluster_vpc = desc.get("cluster", {}).get("resourcesVpcConfig", {}).get("vpcId")
                if cluster_vpc in vpc_set:
                    to_delete.add(name)
            except Exception as e:
                logger.debug("Could not describe EKS cluster '%s': %s", name, e)
        if to_delete:
            delete_eks_clusters(session, to_delete)
    except Exception as e:
        logger.warning("EKS cluster discovery for VPCs failed: %s", e)


def _discover_efs_mount_targets(session: boto3.Session, vpc_ids: list[str]) -> list:
    """Find EFS mount targets in the given VPCs via the EFS API."""
    try:
        efs = build_client(session, "efs")
        ec2 = build_client(session, "ec2")
        # Get all subnets in these VPCs
        subnet_ids: set[str] = set()
        for page in ec2.get_paginator("describe_subnets").paginate(
            Filters=[{"Name": "vpc-id", "Values": vpc_ids}]
        ):
            for subnet in page.get("Subnets", []):
                subnet_ids.add(subnet["SubnetId"])
        if not subnet_ids:
            return []
        # List all file systems, then their mount targets
        resources: list = []
        for page in efs.get_paginator("describe_file_systems").paginate():
            for fs in page.get("FileSystems", []):
                fs_id = fs["FileSystemId"]
                for mt in efs.describe_mount_targets(FileSystemId=fs_id).get("MountTargets", []):
                    if mt.get("SubnetId") in subnet_ids:
                        resources.append(
                            Resource(type="AWS::EFS::MountTarget", identifier=mt["MountTargetId"])
                        )
        return resources
    except Exception as e:
        logger.warning("EFS mount target discovery failed: %s", e)
        return []


def _discover_security_groups(ec2, vpc_ids: list[str]) -> list:
    try:
        sgs: list[dict] = []
        paginator = ec2.get_paginator("describe_security_groups")
        for page in paginator.paginate(Filters=[{"Name": "vpc-id", "Values": vpc_ids}]):
            sgs.extend(page.get("SecurityGroups", []))
        non_default = [sg for sg in sgs if sg["GroupName"] != "default"]
        # Break cross-references before the SGs are handed to the deleter. EFS
        # auto-creates a pair of mount-target SGs (inbound-nfs / outbound-nfs) whose
        # rules point at each other, so neither can be deleted (DependencyViolation)
        # until the other's referencing rule is gone — and the VPC they live in stays
        # pinned indefinitely. Revoking every non-default SG's own ingress+egress rules
        # first severs the cycle; it is idempotent and safe since the SGs are being torn
        # down. (Unlike ENI-pinned resources this never self-resolves, so it cannot be
        # left to a deferral/later run.)
        _revoke_sg_rules(ec2, non_default)
        # Revoking rules only breaks SG<->SG cross-references. A non-default SG can
        # still be undeletable because an ENI lists it in its group set ("has a
        # dependent object") — classically a cross-stack WorkerNodeSecurityGroup
        # pinned by leftover, now-``available`` EKS worker-node ENIs. Sever those
        # ENI references (region-wide, by group-id) so the subsequent delete succeeds.
        drain_sg_eni_references(ec2, [sg["GroupId"] for sg in non_default])
        return [
            Resource(type="AWS::EC2::SecurityGroup", identifier=sg["GroupId"]) for sg in non_default
        ]
    except Exception as e:
        logger.warning("SG discovery failed: %s", e)
        return []


def _revoke_sg_rules(ec2, security_groups: list[dict]) -> None:
    """Revoke each SG's ingress and egress rules to break cross-references before delete.

    Reuses the rules already returned by ``describe_security_groups`` (no extra API
    call). Best-effort per rule set: a revoke failure on one SG (or one direction) is
    logged and skipped so it never aborts the surrounding VPC cleanup, and egress is
    still attempted even if ingress failed.
    """
    for sg in security_groups:
        sg_id = sg.get("GroupId")
        if not sg_id:
            continue
        ingress = sg.get("IpPermissions") or []
        if ingress:
            try:
                ec2.revoke_security_group_ingress(GroupId=sg_id, IpPermissions=ingress)
                logger.debug("Revoked %d ingress rule(s) on SG '%s'", len(ingress), sg_id)
            except Exception as e:
                logger.debug("Could not revoke ingress on SG '%s': %s", sg_id, e)
        egress = sg.get("IpPermissionsEgress") or []
        if egress:
            try:
                ec2.revoke_security_group_egress(GroupId=sg_id, IpPermissions=egress)
                logger.debug("Revoked %d egress rule(s) on SG '%s'", len(egress), sg_id)
            except Exception as e:
                logger.debug("Could not revoke egress on SG '%s': %s", sg_id, e)


# ── Security-group ENI-reference drain ───────────────────────────────
#
# EC2 refuses to delete a security group while ANY network interface lists it in
# its group set ("resource sg-... has a dependent object"). Revoking the SG's own
# rules (above) only breaks SG<->SG references; it does nothing about ENIs that
# reference the SG. The classic offender is a cross-stack ``WorkerNodeSecurityGroup``
# pinned by leftover EKS worker-node ENIs: after the cluster/nodegroup is deleted
# those ENIs go ``available`` but still reference the SG, and they can live in the
# *cluster's* VPC — not the SG-owning stack's VPC — so the vpc-scoped ENI reaper
# never sees them. The drain below is SG-centric (queried by group-id, region-wide)
# so it finds those cross-stack references and severs them before the SG delete.


def drain_sg_eni_references(ec2: Any, sg_ids: list[str]) -> list[str]:
    """Sever every ENI reference to ``sg_ids`` so the SGs become deletable.

    ENIs are queried by ``group-id`` REGION-WIDE (not by vpc-id) so cross-stack /
    cross-VPC references — e.g. leftover EKS worker-node ENIs in the cluster's VPC
    pinning a ``WorkerNodeSecurityGroup`` owned by another stack — are found. Per ENI:

      - ``available``                 -> delete it (releases the reference).
      - ``in-use``, customer-managed  -> rewrite its ``Groups`` to drop this SG (no
        detach needed); substitute the VPC default SG if that would leave it empty.
      - ``in-use``, requester-managed -> only the owning service (EKS X-ENI, ELB,
        VPC endpoint) can release it; the SG id is returned so the caller can defer
        it and re-drive once the owner has released the interface.

    Best-effort and idempotent: a per-SG or per-ENI failure is logged and skipped so
    it never aborts the surrounding cleanup. Returns the SG ids still pinned by a
    requester-managed ENI.
    """
    still_pinned: list[str] = []
    for sg_id in sg_ids:
        if not sg_id:
            continue
        try:
            enis: list[dict] = []
            paginator = ec2.get_paginator("describe_network_interfaces")
            for page in paginator.paginate(Filters=[{"Name": "group-id", "Values": [sg_id]}]):
                enis.extend(page.get("NetworkInterfaces", []))
        except Exception as e:  # noqa: BLE001
            logger.warning("ENI-by-SG discovery failed for '%s': %s", sg_id, e)
            continue

        pinned = False
        for eni in enis:
            eni_id = eni.get("NetworkInterfaceId", "")
            if not eni_id:
                continue
            if eni.get("Status", "") == "available":
                _delete_eni(ec2, eni_id)
            elif eni.get("RequesterManaged", False):
                # Service-owned (EKS X-ENI / ELB / VPC endpoint): can neither delete
                # nor modify it; only its owner releases it, asynchronously.
                pinned = True
            else:
                _drop_sg_from_eni(ec2, eni, sg_id)
        if pinned:
            still_pinned.append(sg_id)
    return still_pinned


def _drop_sg_from_eni(ec2: Any, eni: dict, sg_id: str) -> None:
    """Remove one SG from an in-use, customer-managed ENI's group set (no detach).

    An ENI must keep at least one security group, so if ``sg_id`` is its only group
    we substitute the ENI's VPC default SG. Best-effort: a failure is logged and left
    for a later pass.
    """
    eni_id = eni.get("NetworkInterfaceId", "")
    groups = [g["GroupId"] for g in eni.get("Groups", []) if g.get("GroupId") != sg_id]
    if not groups:
        default_sg = _default_sg_for_vpc(ec2, eni.get("VpcId", ""))
        if not default_sg:
            logger.debug("No default SG for ENI '%s'; cannot drop SG '%s'", eni_id, sg_id)
            return
        groups = [default_sg]
    try:
        ec2.modify_network_interface_attribute(NetworkInterfaceId=eni_id, Groups=groups)
        logger.debug("Dropped SG '%s' from ENI '%s'", sg_id, eni_id)
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not drop SG '%s' from ENI '%s': %s", sg_id, eni_id, e)


def _default_sg_for_vpc(ec2: Any, vpc_id: str) -> str | None:
    """Return the default security group id for ``vpc_id`` (or None if unavailable)."""
    if not vpc_id:
        return None
    try:
        resp = ec2.describe_security_groups(
            Filters=[
                {"Name": "vpc-id", "Values": [vpc_id]},
                {"Name": "group-name", "Values": ["default"]},
            ]
        )
        groups = resp.get("SecurityGroups", [])
        return groups[0]["GroupId"] if groups else None
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not resolve default SG for VPC '%s': %s", vpc_id, e)
        return None


@resource_handler("AWS::EC2::SecurityGroup", role="prepare")
def _prepare_security_group(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Make a security group deletable before CloudFormation/CCAPI deletes it.

    Runs on the stack's own resources during the prepare stage — before the CFN
    ``DeleteStack`` — so a stack-owned SG (e.g. ``WorkerNodeSecurityGroup``) is
    unpinned by the time CloudFormation tries to delete it. Revokes the SG's own
    rules (breaks SG<->SG cross-references) and drains ENI references (breaks the
    "has a dependent object" failure). The default SG is left untouched — CCAPI/CFN
    skip it and it is deleted with the VPC. If a requester-managed ENI still pins the
    SG, it is deferred so the post-cleanup scan does not fail the run and a later
    re-drive reaps it once the owner releases the interface.
    """
    sg_id = resource.identifier
    ec2 = build_client(session, "ec2")
    try:
        resp = ec2.describe_security_groups(GroupIds=[sg_id])
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("InvalidGroup.NotFound", "InvalidGroupId.Malformed"):
            return HandlerResult(
                resource_id=sg_id,
                resource_type=resource.type,
                action="prepare",
                status=HandlerStatus.SKIPPED,
                message="Security group not found",
            )
        raise
    groups = resp.get("SecurityGroups", [])
    if not groups or groups[0].get("GroupName") == "default":
        return HandlerResult(
            resource_id=sg_id,
            resource_type=resource.type,
            action="prepare",
            status=HandlerStatus.SKIPPED,
            message="Default or missing security group; left untouched",
        )

    _revoke_sg_rules(ec2, groups)
    still_pinned = drain_sg_eni_references(ec2, [sg_id])
    if still_pinned:
        mark_deferred("AWS::EC2::SecurityGroup", sg_id)
        return HandlerResult(
            resource_id=sg_id,
            resource_type=resource.type,
            action="prepare",
            status=HandlerStatus.SUCCESS,
            message="Revoked rules; deferred (requester-managed ENI still attached)",
        )
    return HandlerResult(
        resource_id=sg_id,
        resource_type=resource.type,
        action="prepare",
        status=HandlerStatus.SUCCESS,
        message="Revoked rules and drained ENI references",
    )


# ── ENI reaper ───────────────────────────────────────────────────────
#
# Leftover ENIs are what keep a VPC's subnets/security-groups (and the VPC) in
# DELETE_FAILED after their owner is gone, so CloudFormation's DeleteSubnet/DeleteVpc
# fails with DependencyViolation and the stack stalls in DELETE_IN_PROGRESS. The reaper
# actively clears them so the (re-)delete can complete.

# Bounded so a never-releasing requester-managed ENI cannot hang teardown forever.
REAP_TIMEOUT_SEC = 300
REAP_INTERVAL_SEC = 15
_ENI_NOT_FOUND = "InvalidNetworkInterfaceID.NotFound"
_ENI_ATTACHED_STATES = ("attached", "attaching")


@dataclass
class EniReapResult:
    """Outcome of reaping the ENIs in one or more VPCs."""

    deleted: list[str] = field(default_factory=list)
    detached: list[str] = field(default_factory=list)
    remaining: list[str] = field(default_factory=list)
    """ENIs still present when the reaper stopped — requester-managed interfaces
    whose owning service had not released them yet. Non-empty means the VPC may
    still be undeletable; the caller should surface these."""

    @property
    def reaped_any(self) -> bool:
        """True if the reaper deleted or detached at least one ENI."""
        return bool(self.deleted or self.detached)


def reap_vpc_enis(
    session: boto3.Session,
    vpc_ids: list[str],
    *,
    region: str | None = None,
    timeout: float = REAP_TIMEOUT_SEC,
    interval: float = REAP_INTERVAL_SEC,
    wait_for_release: bool = True,
) -> EniReapResult:
    """Delete/detach leftover ENIs in ``vpc_ids`` until none are reapable.

    ``available`` ENIs are deleted; customer-managed ``in-use`` ENIs are
    force-detached (then deleted next pass); requester-managed ``in-use`` ENIs
    (EKS X-ENIs, VPC-endpoint/ELB interfaces) can only be released by their owner.

    ``wait_for_release`` controls what happens once only requester-managed ENIs
    remain:
      - ``True`` (StackDeleter's patient retry) keeps polling until the owner
    releases them or ``timeout`` elapses;
      - ``False`` (the VPC pre-delete hook, where e.g. the load balancer isn't
    deleted until a later phase) returns as soon as no further progress can be made,
    reporting them in ``remaining``.
    """
    vpc_ids = [v for v in vpc_ids if v]
    if not vpc_ids:
        return EniReapResult()

    ec2 = build_client(session, "ec2", region_name=region)
    efs = build_client(session, "efs", region_name=region)
    result = EniReapResult()
    deleted: set[str] = set()
    detached: set[str] = set()
    # EFS mount-target ENIs whose mount target we deleted. Deletion is async, so we
    # do NOT count them done until a later poll shows them actually gone — that keeps
    # them in ``remaining`` so the loop waits them out (poll-until-gone), which lets
    # the caller's re-drive delete the subnet/VPC in one pass instead of leaning on
    # CloudFormation's slow internal retry.
    efs_releasing: set[str] = set()
    deadline = time.monotonic() + timeout

    while True:
        raise_if_shutdown()
        enis = _describe_vpc_enis(ec2, vpc_ids)
        if not enis:
            break

        progressed = False
        for eni in enis:
            eni_id = eni.get("NetworkInterfaceId", "")
            if not eni_id:
                continue
            if eni.get("Status", "") == "available":
                if _delete_eni(ec2, eni_id):
                    deleted.add(eni_id)
                    progressed = True
            elif eni.get("RequesterManaged", False):
                # Most requester-managed ENIs (EKS X-ENIs, ELB, VPC-endpoint) can
                # only be released by their owning service. EFS mount-target ENIs
                # are the exception: WE can release them by deleting the mount
                # target. Issue the delete once, then keep polling until the ENI
                # actually disappears (below) rather than assuming it is gone.
                if eni_id not in efs_releasing and _release_efs_mount_target_eni(efs, eni):
                    efs_releasing.add(eni_id)
                    progressed = True
                continue
            elif _detach_eni(ec2, eni_id, eni.get("Attachment") or {}):
                detached.add(eni_id)
                progressed = True

        # An EFS ENI whose mount target we deleted stays in ``remaining`` until it is
        # actually gone, so the loop waits it out instead of breaking early.
        remaining = {e.get("NetworkInterfaceId", "") for e in enis} - deleted
        if not remaining:
            break
        # Nothing more we can do this pass (only requester-managed ENIs left).
        # Callers that won't wait for an external owner stop here; patient callers
        # keep polling until the deadline for the owner to release them.
        if not progressed and not wait_for_release:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(interval)

    deleted_now = _describe_vpc_enis(ec2, vpc_ids)
    present_now = {
        e.get("NetworkInterfaceId", "") for e in deleted_now if e.get("NetworkInterfaceId")
    }
    # An EFS ENI we released that is now gone counts as reaped, so the caller
    # re-drives DeleteStack rather than deferring the stack.
    deleted |= efs_releasing - present_now
    result.deleted = sorted(deleted)
    result.detached = sorted(detached)
    result.remaining = sorted(present_now)
    if result.deleted or result.detached or result.remaining:
        logger.debug(
            "ENI reap for %s: %d deleted, %d detached, %d still present",
            vpc_ids,
            len(result.deleted),
            len(result.detached),
            len(result.remaining),
        )
    return result


def reap_vpc_security_groups(
    session: boto3.Session, vpc_ids: list[str], *, region: str | None = None
) -> list[str]:
    """Delete leftover *orphan* security groups in ``vpc_ids`` so the VPC can delete.

    After the ENI reaper frees a VPC's interfaces, a leftover NON-default security
    group can still block ``DeleteVpc`` with DependencyViolation — typically one a
    managed service created inside the VPC and did not remove, so CloudFormation
    never owned it and no stack will delete it.

    To keep the blast radius minimal this deletes ONLY security groups that are not
    owned by a CloudFormation stack (no ``aws:cloudformation:stack-id`` tag). A
    stack-owned SG — whether the failing stack's or another still-deleting stack's —
    is left untouched: its own stack's DeleteStack removes it. The default SG is
    never deleted (it goes with the VPC). Rules on the orphan SGs and the default SG
    are revoked first so a cross-reference to an orphan cannot block its delete, and
    ENI references are severed; stack-owned SGs are never mutated.

    Best-effort and idempotent: a per-SG failure is logged and skipped. Returns the
    orphan SG ids it could not delete.
    """
    vpc_ids = [v for v in vpc_ids if v]
    if not vpc_ids:
        return []
    ec2 = build_client(session, "ec2", region_name=region)
    try:
        sgs: list[dict] = []
        paginator = ec2.get_paginator("describe_security_groups")
        for page in paginator.paginate(Filters=[{"Name": "vpc-id", "Values": vpc_ids}]):
            sgs.extend(page.get("SecurityGroups", []))
    except (ClientError, BotoCoreError) as e:
        logger.warning("Security-group discovery for %s failed: %s", vpc_ids, e)
        return []

    def _cfn_owned(sg: dict) -> bool:
        return any(t.get("Key") == "aws:cloudformation:stack-id" for t in (sg.get("Tags") or []))

    # Only service-created orphans (no CFN stack tag) are ours to delete; a
    # stack-owned SG is deleted by its own stack, not pre-empted here.
    orphans = [
        sg
        for sg in sgs
        if sg.get("GroupName") != "default" and sg.get("GroupId") and not _cfn_owned(sg)
    ]
    if not orphans:
        return []

    default_sgs = [sg for sg in sgs if sg.get("GroupName") == "default"]
    # Revoke rules on the orphan SGs and the (non-stack) default SG so a rule that
    # references an orphan — including a default-SG rule — cannot block its delete.
    # Stack-owned SGs are intentionally left unmodified.
    _revoke_sg_rules(ec2, orphans + default_sgs)
    orphan_ids = [sg["GroupId"] for sg in orphans]
    drain_sg_eni_references(ec2, orphan_ids)

    remaining: list[str] = []
    for sg_id in orphan_ids:
        try:
            ec2.delete_security_group(GroupId=sg_id)
            logger.debug("Deleted orphan SG '%s' to unblock VPC delete", sg_id)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("InvalidGroup.NotFound", "InvalidGroupId.Malformed"):
                continue
            logger.debug("Could not delete SG '%s': %s", sg_id, e)
            remaining.append(sg_id)
        except BotoCoreError as e:
            logger.debug("Could not delete SG '%s': %s", sg_id, e)
            remaining.append(sg_id)
    if remaining:
        logger.debug("SG reap for %s: %d orphan SG(s) still undeletable", vpc_ids, len(remaining))
    return remaining


def _describe_vpc_enis(ec2: Any, vpc_ids: list[str]) -> list[dict]:
    """Return all network interfaces in ``vpc_ids`` (every status), or [] on error."""
    try:
        enis: list[dict] = []
        paginator = ec2.get_paginator("describe_network_interfaces")
        for page in paginator.paginate(Filters=[{"Name": "vpc-id", "Values": vpc_ids}]):
            enis.extend(page.get("NetworkInterfaces", []))
        return enis
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") == _ENI_NOT_FOUND:
            return []
        logger.warning("ENI discovery for %s failed: %s", vpc_ids, e)
        return []
    except Exception as e:  # noqa: BLE001
        logger.warning("ENI discovery for %s failed: %s", vpc_ids, e)
        return []


def _delete_eni(ec2: Any, eni_id: str) -> bool:
    """Delete an available ENI. Returns True if deleted or already gone."""
    try:
        ec2.delete_network_interface(NetworkInterfaceId=eni_id)
        logger.debug("Reaped ENI '%s'", eni_id)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == _ENI_NOT_FOUND:
            return True
        # In-use race (flipped back to attached) or not-permitted: leave for next pass.
        logger.debug("Could not delete ENI '%s' [%s]: %s", eni_id, code, e)
        return False
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not delete ENI '%s': %s", eni_id, e)
        return False


def _detach_eni(ec2: Any, eni_id: str, attachment: dict) -> bool:
    """Force-detach a customer-managed in-use ENI so it can be deleted next pass.

    Returns True if a detach was issued. A missing attachment id or a detach error
    yields False (nothing to do / retry next pass).
    """
    attachment_id = attachment.get("AttachmentId")
    if not attachment_id or attachment.get("Status") not in _ENI_ATTACHED_STATES:
        return False
    try:
        ec2.detach_network_interface(AttachmentId=attachment_id, Force=True)
        logger.debug("Force-detached ENI '%s' (attachment %s)", eni_id, attachment_id)
        return True
    except ClientError as e:
        logger.debug("Could not detach ENI '%s': %s", eni_id, e)
        return False
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not detach ENI '%s': %s", eni_id, e)
        return False


# EFS mount-target id embedded in the ENI's description: "EFS mount target for
# fs-... (fsmt-...)". Deleting that mount target is the only way to release the
# interface — EFS mount-target ENIs cannot be detached or deleted directly.
_EFS_MOUNT_TARGET_RE = re.compile(r"fsmt-[0-9a-f]+")


def _release_efs_mount_target_eni(efs: Any, eni: dict) -> bool:
    """Release an EFS mount-target ENI by deleting its owning mount target.

    Unlike EKS/ELB/VPC-endpoint requester-managed ENIs (only the owner releases
    those), an EFS mount-target ENI is freed by deleting its mount target, which
    unpins the subnet/VPC. The mount-target id is parsed from the ENI description.
    Returns True if a delete was issued (or the mount target was already gone) so
    the reaper counts it as progress and keeps polling until the ENI disappears.
    """
    if eni.get("InterfaceType") != "efs":
        return False
    match = _EFS_MOUNT_TARGET_RE.search(eni.get("Description", "") or "")
    if not match:
        return False
    mt_id = match.group(0)
    try:
        efs.delete_mount_target(MountTargetId=mt_id)
        logger.debug(
            "Deleted EFS mount target %s to release ENI %s",
            mt_id,
            eni.get("NetworkInterfaceId", ""),
        )
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") == "MountTargetNotFound":
            return True
        logger.debug("Could not delete EFS mount target %s: %s", mt_id, e)
        return False
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not delete EFS mount target %s: %s", mt_id, e)
        return False


# ── IPAM child-pool reaper ───────────────────────────────────────────
#
# A leaked child pool holds an allocation that blocks its parent's CIDR deprovision
# during DeleteStack; this reaper drains those non-stack children bottom-up (the IPAM
# analogue of reap_vpc_enis).


@dataclass
class IpamPoolReapResult:
    """Outcome of reaping the leaked child pools of one or more parent pools."""

    deleted: list[str] = field(default_factory=list)
    """Child pools deleted (or confirmed already gone) so the parent can drain."""
    remaining: list[str] = field(default_factory=list)
    """Child pools not confirmed gone (FAILED); the caller surfaces these."""

    @property
    def reaped_any(self) -> bool:
        """True if at least one child was deleted (confirmed gone) so a re-drive can proceed."""
        return bool(self.deleted)


def reap_ipam_child_pools(
    session: boto3.Session,
    parent_pool_ids: list[str],
    *,
    region: str | None = None,
    stack_owned_pool_ids: Iterable[str] = (),
) -> IpamPoolReapResult:
    """Delete leaked child pools sourced from ``parent_pool_ids`` so the parents can delete.

    Best-effort (never raises); reaps bottom-up so no descendant allocation blocks a
    parent. A child in ``stack_owned_pool_ids`` is CFN-owned, not a leak, so it and its
    subtree are skipped.
    """
    parent_pool_ids = [p for p in parent_pool_ids if p]
    if not parent_pool_ids:
        return IpamPoolReapResult()

    try:
        ec2 = build_client(session, "ec2", region_name=region)
    except Exception as e:  # noqa: BLE001 — best-effort contract: never propagate
        logger.warning(f"IPAM child-pool reap could not build ec2 client: {e}")
        return IpamPoolReapResult()

    all_pools = _describe_all_ipam_pools(ec2)
    # source id -> children, so a child's own children can be reaped before it.
    children_by_source: dict[str, list[str]] = {}
    for pool in all_pools:
        source = pool.get("SourceIpamPoolId")
        pool_id = pool.get("IpamPoolId")
        if source and pool_id:
            children_by_source.setdefault(source, []).append(pool_id)

    owned = set(stack_owned_pool_ids)
    result = IpamPoolReapResult()
    visited: set[str] = set()
    # One shared deadline for the whole reap so a K-child tree costs ~one budget, not K×.
    deadline = time.monotonic() + _DEPROVISION_BUDGET_SEC
    for parent_id in parent_pool_ids:
        _reap_leaked_child_pools(
            ec2, parent_id, children_by_source, owned, result, visited, deadline
        )
    if result.deleted or result.remaining:
        logger.debug(
            f"IPAM child-pool reap for {parent_pool_ids}: {len(result.deleted)} deleted, "
            f"{len(result.remaining)} still present"
        )
    return result


def _reap_leaked_child_pools(
    ec2: Any,
    parent_id: str,
    children_by_source: dict[str, list[str]],
    stack_owned_pool_ids: set[str],
    result: IpamPoolReapResult,
    visited: set[str],
    deadline: float,
) -> None:
    """Reap every leaked child pool sourced from ``parent_id``, grandchildren first.

    CFN-owned children are skipped; ``visited`` guards a cyclic source graph and
    ``deadline`` is the reap's shared drain budget. Each child is confirmed gone
    (``deleted``) or FAILED (``remaining``) — never deferred.
    """
    for child_id in children_by_source.get(parent_id, []):
        if child_id in visited:
            continue
        visited.add(child_id)
        if child_id in stack_owned_pool_ids:
            # CFN-owned, not a leak: CloudFormation deletes it (and its subtree).
            logger.debug(f"Skipping stack-owned IPAM child pool '{child_id}' (not a leak)")
            continue
        # Reap this child's own children first so its CIDR can deprovision.
        _reap_leaked_child_pools(
            ec2, child_id, children_by_source, stack_owned_pool_ids, result, visited, deadline
        )
        try:
            outcome = deprovision_and_delete_pool(ec2, child_id, deadline=deadline)
        except Exception as e:  # noqa: BLE001 — best-effort contract: never propagate
            # deprovision_and_delete_pool maps every fault to an outcome; a raise is
            # unexpected — record as still-present and keep reaping the rest.
            result.remaining.append(child_id)
            logger.warning(f"Unexpected error reaping IPAM child pool '{child_id}': {e}")
            continue
        if outcome.outcome in (PoolDeleteOutcome.DELETED, PoolDeleteOutcome.ALREADY_GONE):
            result.deleted.append(child_id)
            logger.debug(f"Reaped leaked IPAM child pool '{child_id}': {outcome.message}")
        else:
            result.remaining.append(child_id)
            logger.warning(f"Could not reap leaked IPAM child pool '{child_id}': {outcome.message}")


def _describe_all_ipam_pools(ec2: Any) -> list[dict]:
    """Return every IPAM pool in the region, or [] on error (source-pool match is client-side)."""
    try:
        pools: list[dict] = []
        for page in ec2.get_paginator("describe_ipam_pools").paginate():
            pools.extend(page.get("IpamPools", []))
        return pools
    except Exception as e:  # noqa: BLE001
        logger.warning(f"IPAM pool discovery failed: {e}")
        return []


# ── Registered failed-resource handlers ──────────────────────────────
#
# Priority ordering ensures predictable execution when multiple handlers match.
# Generic handlers (low priority) run before specific handlers (high priority),
# allowing specialized cleanup to build on generic cleanup.


# ── EKS Cluster cleanup ──────────────────────────────────────────────


@resource_handler("AWS::EKS::Cluster", role="prepare")
def _prepare_eks_cluster(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete EKS cluster (nodegroups, Fargate profiles, cluster) before CCAPI deletion.

    EKS clusters hold ENIs and security groups that cannot be deleted while the
    cluster exists. Running this as a prepare step ensures the cluster and its
    managed resources are gone before CCAPI attempts to delete orphaned ENIs.
    """
    cluster_name = resource.identifier
    try:
        unresolved = delete_eks_clusters(session, {cluster_name})
        if unresolved:
            # Not SUCCESS: a still-present cluster holds ENIs/SGs, so CCAPI would
            # fail later with a confusing dependency error.
            return HandlerResult(
                resource_id=cluster_name,
                resource_type=resource.type,
                action="prepare",
                status=HandlerStatus.FAILED,
                message=f"EKS cluster '{cluster_name}' not confirmed deleted",
            )
        return HandlerResult(
            resource_id=cluster_name,
            resource_type=resource.type,
            action="prepare",
            status=HandlerStatus.SUCCESS,
            message=f"Deleted EKS cluster '{cluster_name}' and sub-resources",
        )
    except Exception as e:
        return HandlerResult(
            resource_id=cluster_name,
            resource_type=resource.type,
            action="prepare",
            status=HandlerStatus.FAILED,
            message=str(e),
        )


# ── EKS Add-on cleanup ───────────────────────────────────────────────

_EKS_ADDON_NOT_FOUND_CODES = ("ResourceNotFoundException",)


@resource_handler("AWS::EKS::Addon", role="delete")
def _delete_eks_addon(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete an EKS add-on, splitting the composite ``clusterName|addonName`` id."""
    identifier = resource.identifier
    if "|" not in identifier:
        return HandlerResult(
            resource_id=identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"EKS add-on id '{identifier}' is not 'clusterName|addonName'",
        )
    cluster_name, addon_name = identifier.split("|", 1)
    try:
        build_client(session, "eks").delete_addon(clusterName=cluster_name, addonName=addon_name)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") in _EKS_ADDON_NOT_FOUND_CODES:
            return HandlerResult(
                resource_id=identifier,
                resource_type=resource.type,
                action="delete",
                status=HandlerStatus.SUCCESS,
                message="EKS add-on already gone",
            )
        return HandlerResult(
            resource_id=identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Failed to delete EKS add-on: {e}",
        )
    except BotoCoreError as e:
        return HandlerResult(
            resource_id=identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Connection error deleting EKS add-on: {e}",
        )
    logger.debug(f"Deleted EKS add-on '{identifier[:LOG_TRUNCATE_MEDIUM]}'")
    return HandlerResult(
        resource_id=identifier,
        resource_type=resource.type,
        action="delete",
        status=HandlerStatus.SUCCESS,
    )


@failed_resource_handler("Custom::", priority=10)
def _handle_stuck_custom_resources(failed, all_resources, session, region=None):
    """Clean up Lambda/IAM left behind by failed Custom:: resources.

    Priority 10 (generic): Runs first to handle common Custom resource dependencies.
    Matches ALL Custom:: resources including Custom::AWS, Custom::Lambda, etc.
    """
    remaining = [resource for resource in all_resources if resource.status != "DELETE_COMPLETE"]
    cleanup_stuck_custom_resource_deps(
        session,
        remaining,
        region or session.region_name or DEFAULT_REGION,
    )


@failed_resource_handler("Custom::AWS", priority=50)
def _handle_stuck_custom_aws_resources(failed, all_resources, session, region=None):
    """Clean up Redshift resources left by a failed Custom::AWS resource.

    Priority 50 (specific): Runs after generic Custom:: handler to handle
    AWS-managed custom resources that leave service-specific orphans. The
    orphans live in the stack's region, so clean only that one.
    """
    cleanup_redshift(session, region or session.region_name or DEFAULT_REGION)
