"""S3 bucket policy cleanup handler.

Deletes a standalone bucket policy that persists after task cleanup.
The S3 Bucket handler already removes policies as a side-effect of bucket
teardown, but orphan policies on surviving buckets need independent deletion.
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

_NOT_FOUND_CODES = ("NoSuchBucket", "NoSuchBucketPolicy", "404")


@resource_handler("AWS::S3::BucketPolicy", role="delete")
def _delete(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete the bucket policy."""
    client = build_client(session, "s3")
    try:
        client.delete_bucket_policy(Bucket=resource.identifier)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in _NOT_FOUND_CODES:
            return HandlerResult(
                resource_id=resource.identifier,
                resource_type=resource.type,
                action="delete",
                status=HandlerStatus.SKIPPED,
                message="Bucket or policy not found",
            )
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Failed to delete bucket policy: {e}",
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
        message="Deleted bucket policy",
    )
