"""IoT cleanup handlers.

Agent-created IoT resources are not stack-managed, survive normal CFN cleanup,
and CCAPI cannot delete them, so each needs a custom delete handler.

``AWS::IoT::ThingGroup`` — the group may contain things (added by the agent);
these must be removed before the group can be deleted:
1. Remove all things from the group (``RemoveThingFromThingGroup``).
2. Delete the group (``DeleteThingGroup``).

``AWS::IoT::Policy`` — IoT requires detach-before-delete, and a policy cannot be
deleted while non-default versions exist:
1. Detach the policy from every target (``DetachPolicy``).
2. Delete every non-default policy version (``DeletePolicyVersion``).
3. Delete the policy (``DeletePolicy``).
"""

from __future__ import annotations

import boto3
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.models import HandlerResult, HandlerStatus
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)

_NOT_FOUND_CODES = ("ResourceNotFoundException", "NoSuchEntity")


def _thing_group_name_from_arn(arn: str) -> str:
    """Extract the thing group name from its ARN.

    The lister uses id_field="groupArn", so the identifier is always the full ARN:
    arn:aws:iot:<region>:<account>:thinggroup/<name>
    """
    return arn.rsplit("/", 1)[-1]


def _remove_all_things_from_group(client: BaseClient, group_name: str) -> int:
    """Remove every thing from the group. Returns the count removed."""
    removed = 0
    paginator = client.get_paginator("list_things_in_thing_group")
    for page in paginator.paginate(thingGroupName=group_name):
        for thing_name in page.get("things", []):
            try:
                client.remove_thing_from_thing_group(
                    thingGroupName=group_name, thingName=thing_name
                )
                removed += 1
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in _NOT_FOUND_CODES:
                    continue
                raise
    return removed


@resource_handler("AWS::IoT::ThingGroup", role="prepare")
def _prepare(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Remove all things from the group so it can be deleted."""
    group_name = _thing_group_name_from_arn(resource.identifier)
    client = build_client(session, "iot")
    try:
        removed = _remove_all_things_from_group(client, group_name)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in _NOT_FOUND_CODES:
            return HandlerResult(
                resource_id=resource.identifier,
                resource_type=resource.type,
                action="prepare",
                status=HandlerStatus.SKIPPED,
                message="Thing group not found (already deleted)",
            )
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="prepare",
            status=HandlerStatus.FAILED,
            message=f"Failed to remove things from group '{group_name}': {e}",
        )
    except BotoCoreError as e:
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="prepare",
            status=HandlerStatus.FAILED,
            message=f"Connection error removing things from group '{group_name}': {e}",
        )
    msg = f"Removed {removed} thing(s) from group" if removed else "Group already empty"
    return HandlerResult(
        resource_id=resource.identifier,
        resource_type=resource.type,
        action="prepare",
        status=HandlerStatus.SUCCESS,
        message=msg,
    )


@resource_handler("AWS::IoT::ThingGroup", role="delete")
def _delete(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete the (now-empty) thing group."""
    group_name = _thing_group_name_from_arn(resource.identifier)
    client = build_client(session, "iot")
    try:
        client.delete_thing_group(thingGroupName=group_name)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in _NOT_FOUND_CODES:
            return HandlerResult(
                resource_id=resource.identifier,
                resource_type=resource.type,
                action="delete",
                status=HandlerStatus.SUCCESS,
                message="Thing group already gone",
            )
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Failed to delete thing group '{group_name}': {e}",
        )
    except BotoCoreError as e:
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Connection error deleting thing group '{group_name}': {e}",
        )
    logger.debug(f"Deleted IoT ThingGroup '{group_name}'")
    return HandlerResult(
        resource_id=resource.identifier,
        resource_type=resource.type,
        action="delete",
        status=HandlerStatus.SUCCESS,
    )


def _policy_name_from_arn(arn: str) -> str:
    """Extract the policy name from its ARN.

    The identifier is the policy ARN
    (``arn:aws:iot:<region>:<account>:policy/<name>``); a bare name without a
    ``/`` is returned unchanged.
    """
    return arn.rsplit("/", 1)[-1]


def _detach_all_targets(client: BaseClient, policy_name: str) -> int:
    """Detach the policy from every target. Returns the count detached."""
    detached = 0
    paginator = client.get_paginator("list_targets_for_policy")
    for page in paginator.paginate(policyName=policy_name):
        for target in page.get("targets", []):
            try:
                client.detach_policy(policyName=policy_name, target=target)
                detached += 1
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in _NOT_FOUND_CODES:
                    continue
                raise
    return detached


def _delete_non_default_versions(client: BaseClient, policy_name: str) -> int:
    """Delete every non-default policy version. Returns the count deleted.

    ``DeletePolicy`` fails while non-default versions exist; the default version
    is removed by ``DeletePolicy`` itself and must not be deleted here.
    """
    deleted = 0
    versions = client.list_policy_versions(policyName=policy_name).get("policyVersions", [])
    for version in versions:
        if version.get("isDefaultVersion"):
            continue
        try:
            client.delete_policy_version(
                policyName=policy_name, policyVersionId=version["versionId"]
            )
            deleted += 1
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in _NOT_FOUND_CODES:
                continue
            raise
    return deleted


@resource_handler("AWS::IoT::Policy", role="delete")
def _delete_policy(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Detach the policy from all targets, drop non-default versions, then delete it."""
    policy_name = _policy_name_from_arn(resource.identifier)
    client = build_client(session, "iot")
    try:
        detached = _detach_all_targets(client, policy_name)
        removed_versions = _delete_non_default_versions(client, policy_name)
        client.delete_policy(policyName=policy_name)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in _NOT_FOUND_CODES:
            return HandlerResult(
                resource_id=resource.identifier,
                resource_type=resource.type,
                action="delete",
                status=HandlerStatus.SUCCESS,
                message="IoT policy already gone",
            )
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Failed to delete IoT policy '{policy_name}': {e}",
        )
    except BotoCoreError as e:
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Connection error deleting IoT policy '{policy_name}': {e}",
        )
    logger.debug(
        f"Deleted IoT Policy '{policy_name}' (detached {detached} target(s), "
        f"removed {removed_versions} non-default version(s))"
    )
    return HandlerResult(
        resource_id=resource.identifier,
        resource_type=resource.type,
        action="delete",
        status=HandlerStatus.SUCCESS,
    )
