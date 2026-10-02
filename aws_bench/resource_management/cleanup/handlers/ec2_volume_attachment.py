"""EC2 VolumeAttachment cleanup handler.

CCAPI cannot delete ``AWS::EC2::VolumeAttachment`` (the scan logs
``Skip … not supported by CCAPI``), so an EBS volume an agent/scenario left
attached — e.g. an EKS node's data volume after its managed node group is torn
down — keeps its attachment. That attachment in turn blocks the CCAPI
``AWS::EC2::Volume`` delete (``VolumeInUse``), so both the attachment and often
the volume behind it leak and fail the post-reset orphan re-check.

The fast-scan lister emits the VOLUME id (``vol-…``) as the attachment's
identifier (from ``describe_volumes`` → ``Attachments[].VolumeId``), so this
handler detaches that volume:

1. Describe the volume; a missing volume, or one already detached, is treated as
   success (the attachment is already gone).
2. ``DetachVolume`` with ``Force=True`` — during the custom-delete pass the
   owning instance may still be running/terminating, and a teardown detach is
   intentionally destructive.
3. Wait for the volume to reach ``available`` so the subsequent CCAPI
   ``AWS::EC2::Volume`` delete does not race a still-``detaching`` volume.
"""

from __future__ import annotations

import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.models import HandlerResult, HandlerStatus
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)

_VOLUME_NOT_FOUND_CODES = ("InvalidVolume.NotFound",)
# DetachVolume rejects a volume that is not attached (already ``available``).
_ALREADY_DETACHED_CODES = ("IncorrectState",)
# Wait for the volume to leave ``detaching`` after the detach so the CCAPI volume
# delete does not race it. Bounded (~2 min); a still-attached volume after the
# window is a genuine failure worth surfacing.
_DETACH_WAIT_ATTEMPTS = 40
_DETACH_WAIT_DELAY_SEC = 3


def _code(error: ClientError) -> str:
    return error.response.get("Error", {}).get("Code", "")


def _success(resource: Resource, message: str = "") -> HandlerResult:
    return HandlerResult(
        resource_id=resource.identifier,
        resource_type=resource.type,
        action="delete",
        status=HandlerStatus.SUCCESS,
        message=message,
    )


def _failed(resource: Resource, message: str) -> HandlerResult:
    return HandlerResult(
        resource_id=resource.identifier,
        resource_type=resource.type,
        action="delete",
        status=HandlerStatus.FAILED,
        message=message,
    )


def _active_attachments(volume: dict) -> list[dict]:
    """Attachments that still bind the volume to an instance."""
    return [a for a in volume.get("Attachments", []) if a.get("InstanceId")]


def _wait_until_detached(ec2: object, volume_id: str) -> bool:
    """Poll until the volume is ``available`` / gone. Returns False on timeout."""
    for attempt in range(_DETACH_WAIT_ATTEMPTS):
        try:
            resp = ec2.describe_volumes(VolumeIds=[volume_id])  # type: ignore[attr-defined]
        except ClientError as e:
            if _code(e) in _VOLUME_NOT_FOUND_CODES:
                return True  # deleted out from under us — desired end state
            raise
        volumes = resp.get("Volumes", [])
        if not volumes or not _active_attachments(volumes[0]):
            return True
        if attempt < _DETACH_WAIT_ATTEMPTS - 1:
            time.sleep(_DETACH_WAIT_DELAY_SEC)
    return False


@resource_handler("AWS::EC2::VolumeAttachment", role="delete")
def _delete_volume_attachment(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Detach the volume so its attachment (and the CCAPI volume delete) can clear."""
    volume_id = resource.identifier
    ec2 = build_client(session, "ec2")

    try:
        resp = ec2.describe_volumes(VolumeIds=[volume_id])
    except ClientError as e:
        if _code(e) in _VOLUME_NOT_FOUND_CODES:
            return _success(resource, "Volume already gone")
        return _failed(resource, f"Failed to describe volume: {e}")
    except BotoCoreError as e:
        return _failed(resource, f"Connection error describing volume: {e}")

    volumes = resp.get("Volumes", [])
    if not volumes:
        return _success(resource, "Volume already gone")
    attachments = _active_attachments(volumes[0])
    if not attachments:
        return _success(resource, "Volume already detached")

    try:
        for attachment in attachments:
            ec2.detach_volume(VolumeId=volume_id, InstanceId=attachment["InstanceId"], Force=True)
    except ClientError as e:
        if _code(e) in _VOLUME_NOT_FOUND_CODES or _code(e) in _ALREADY_DETACHED_CODES:
            return _success(resource, "Volume already detached")
        return _failed(resource, f"Failed to detach volume: {e}")
    except BotoCoreError as e:
        return _failed(resource, f"Connection error detaching volume: {e}")

    try:
        detached = _wait_until_detached(ec2, volume_id)
    except ClientError as e:
        return _failed(resource, f"Failed to confirm volume detach: {e}")
    except BotoCoreError as e:
        return _failed(resource, f"Connection error confirming volume detach: {e}")
    if not detached:
        return _failed(resource, "Volume still attached after detach")

    logger.debug("Detached volume '%s'", volume_id)
    return _success(resource)
