"""EKS Nodegroup cleanup handler.

Handles standalone nodegroup deletion. Nodegroup deletion is async —
the nodegroup enters DELETING status and takes several minutes to drain.
ResourceInUseException indicates the nodegroup is already being deleted.
"""

from __future__ import annotations

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.models import HandlerResult, HandlerStatus
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)

_NOT_FOUND_CODES = ("ResourceNotFoundException",)
_IN_USE_CODES = ("ResourceInUseException",)


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
        if code in _IN_USE_CODES:
            return HandlerResult(
                resource_id=resource.identifier,
                resource_type=resource.type,
                action="delete",
                status=HandlerStatus.SUCCESS,
                message="Nodegroup already deleting",
            )
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Failed to delete nodegroup: {e}",
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
        message="Initiated nodegroup deletion",
    )
