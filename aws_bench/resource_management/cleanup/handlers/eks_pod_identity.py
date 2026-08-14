"""EKS Pod Identity Association cleanup handler.

Deletes orphan pod identity associations. These are sub-resources of EKS
clusters, keyed by the composite identifier ``clusterName|associationId``.
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


@resource_handler("AWS::EKS::PodIdentityAssociation", role="delete")
def _delete(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete the pod identity association."""
    parts = resource.identifier.split("|", 1)
    if len(parts) != 2:
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message="Invalid identifier format, expected 'clusterName|associationId'",
        )
    cluster, association_id = parts
    client = build_client(session, "eks")
    try:
        client.delete_pod_identity_association(clusterName=cluster, associationId=association_id)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in _NOT_FOUND_CODES:
            return HandlerResult(
                resource_id=resource.identifier,
                resource_type=resource.type,
                action="delete",
                status=HandlerStatus.SKIPPED,
                message="Association or cluster not found",
            )
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Failed to delete pod identity association: {e}",
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
        message="Deleted pod identity association",
    )
