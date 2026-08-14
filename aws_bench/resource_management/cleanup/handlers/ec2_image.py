"""EC2 Image (AMI) cleanup handler.

Deregisters the AMI and deletes associated EBS snapshots to prevent
orphan resource leaks.
"""

from __future__ import annotations

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import LOG_TRUNCATE_MEDIUM, Resource
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.models import HandlerResult, HandlerStatus
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)

_NOT_FOUND_CODES = ("InvalidAMIID.NotFound", "InvalidAMIID.Unavailable")
_SNAPSHOT_NOT_FOUND_CODES = ("InvalidSnapshot.NotFound",)


@resource_handler("AWS::EC2::Image", role="delete")
def _delete(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Deregister the AMI and delete its backing EBS snapshots."""
    client = build_client(session, "ec2")
    image_id = resource.identifier

    # Step 1: Describe image to capture snapshot IDs before deregistering.
    try:
        resp = client.describe_images(ImageIds=[image_id])
        images = resp.get("Images", [])
        if not images:
            return HandlerResult(
                resource_id=image_id,
                resource_type=resource.type,
                action="delete",
                status=HandlerStatus.SKIPPED,
                message="Image not found",
            )
        snapshot_ids = [
            bdm["Ebs"]["SnapshotId"]
            for bdm in images[0].get("BlockDeviceMappings", [])
            if "Ebs" in bdm and "SnapshotId" in bdm["Ebs"]
        ]
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in _NOT_FOUND_CODES:
            return HandlerResult(
                resource_id=image_id,
                resource_type=resource.type,
                action="delete",
                status=HandlerStatus.SKIPPED,
                message="Image not found",
            )
        return HandlerResult(
            resource_id=image_id,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Failed to describe image: {e}",
        )
    except BotoCoreError as e:
        return HandlerResult(
            resource_id=image_id,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Connection error describing image: {e}",
        )

    # Step 2: Deregister the AMI.
    try:
        client.deregister_image(ImageId=image_id)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code not in _NOT_FOUND_CODES:
            return HandlerResult(
                resource_id=image_id,
                resource_type=resource.type,
                action="delete",
                status=HandlerStatus.FAILED,
                message=f"Failed to deregister image: {e}",
            )
        # Already gone — still attempt snapshot cleanup below.

    # Step 3: Delete associated snapshots.
    failed_snapshots: list[str] = []
    for snap_id in snapshot_ids:
        try:
            client.delete_snapshot(SnapshotId=snap_id)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in _SNAPSHOT_NOT_FOUND_CODES:
                continue
            failed_snapshots.append(f"{snap_id}: {code}")
            logger.warning(
                "Failed to delete snapshot '%s' for image '%s': %s",
                snap_id,
                image_id[:LOG_TRUNCATE_MEDIUM],
                e,
            )

    if failed_snapshots:
        return HandlerResult(
            resource_id=image_id,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Deregistered image but {len(failed_snapshots)} snapshot(s) failed: "
            f"{'; '.join(failed_snapshots)}",
        )
    return HandlerResult(
        resource_id=image_id,
        resource_type=resource.type,
        action="delete",
        status=HandlerStatus.SUCCESS,
        message=f"Deregistered image and deleted {len(snapshot_ids)} snapshot(s)",
    )
