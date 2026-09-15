"""EKS Nodegroup cleanup handler.

Handles standalone nodegroup deletion. Nodegroup deletion is async —
the nodegroup enters DELETING status and takes several minutes to drain.
ResourceInUseException indicates the nodegroup is already being deleted.
"""

from __future__ import annotations

import boto3
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.models import HandlerResult, HandlerStatus
from aws_bench.resource_management.utils.polling import wait_until
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)

_NOT_FOUND_CODES = ("ResourceNotFoundException",)
_IN_USE_CODES = ("ResourceInUseException",)

# Bounded terminal-deletion polling so a stuck nodegroup cannot hang cleanup
# forever: 40 attempts x 15s = up to 600s, ample for a nodegroup to drain its
# instances and be removed.
_WAITER_TIMEOUT_SEC = 600
_WAITER_INTERVAL_SEC = 15


def _wait_for_terminal_deletion(client: BaseClient, cluster: str, nodegroup: str) -> None:
    """Block until the nodegroup is actually gone.

    ``delete_nodegroup`` is asynchronous: the nodegroup enters ``DELETING`` and
    ``describe_nodegroup`` keeps returning it until it has fully drained and been
    removed. Poll until ``describe_nodegroup`` reports ``ResourceNotFoundException``.

    Raises:
        ClientError: If the nodegroup is still present after the bounded wait, so
            the caller maps the result to FAILED rather than hanging.
    """

    def _gone() -> bool:
        try:
            client.describe_nodegroup(clusterName=cluster, nodegroupName=nodegroup)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code", "") in _NOT_FOUND_CODES:
                return True  # Fully deleted.
            raise  # transient (throttling/etc.) — wait_until swallows and retries
        return False

    if wait_until(_gone, timeout=_WAITER_TIMEOUT_SEC, interval=_WAITER_INTERVAL_SEC):
        return
    # Still present after the bounded wait — raise so the caller maps to FAILED.
    raise ClientError(
        {
            "Error": {
                "Code": "DeletionTimeout",
                "Message": f"nodegroup still present after {_WAITER_TIMEOUT_SEC}s",
            }
        },
        "DescribeNodegroup",
    )


@resource_handler("AWS::EKS::Nodegroup", role="delete")
def _delete(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete the EKS nodegroup."""
    parts = resource.identifier.split("|", 1)
    if len(parts) != 2:
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message="Invalid identifier format, expected 'clusterName|nodegroupName'",
        )
    cluster, nodegroup = parts
    client = build_client(session, "eks")
    try:
        client.delete_nodegroup(clusterName=cluster, nodegroupName=nodegroup)
        message = "Deleted nodegroup"
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in _NOT_FOUND_CODES:
            return HandlerResult(
                resource_id=resource.identifier,
                resource_type=resource.type,
                action="delete",
                status=HandlerStatus.SKIPPED,
                message="Nodegroup or cluster not found",
            )
        if code not in _IN_USE_CODES:
            return HandlerResult(
                resource_id=resource.identifier,
                resource_type=resource.type,
                action="delete",
                status=HandlerStatus.FAILED,
                message=f"Failed to delete nodegroup: {e}",
            )
        # ResourceInUseException: the nodegroup is already deleting — still wait
        # for it to reach terminal deletion before reporting success.
        message = "Nodegroup was already deleting; deletion completed"
    except BotoCoreError as e:
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Connection error: {e}",
        )

    # Deletion is asynchronous — block until the nodegroup is actually gone so a
    # caller (e.g. post-run reset verification) never races a still-DELETING one.
    try:
        _wait_for_terminal_deletion(client, cluster, nodegroup)
    except ClientError as e:
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Nodegroup deletion did not complete: {e}",
        )
    except BotoCoreError as e:
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Connection error: {e}",
        )

    return HandlerResult(
        resource_id=resource.identifier,
        resource_type=resource.type,
        action="delete",
        status=HandlerStatus.SUCCESS,
        message=message,
    )
