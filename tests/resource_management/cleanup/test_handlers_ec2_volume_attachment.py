"""Tests for the EC2 VolumeAttachment cleanup handler.

moto models the attach/detach lifecycle: a volume attached to an instance is
``in-use`` and carries an ``Attachments`` entry until detached, after which it is
``available`` with no attachments. A SUCCESS from the handler therefore proves it
detached the volume so the subsequent CCAPI ``AWS::EC2::Volume`` delete can clear.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError
from moto import mock_aws

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import CUSTOM_DELETION_REGISTRY
from aws_bench.resource_management.cleanup.handlers.ec2_volume_attachment import (
    _delete_volume_attachment,
)
from aws_bench.resource_management.cleanup.models import HandlerStatus

_REGION = "us-east-1"
# Amazon Linux 2 AMI moto recognizes for run_instances.
_AMI = "ami-12c6146b"


def _resource(volume_id: str) -> Resource:
    return Resource(type="AWS::EC2::VolumeAttachment", identifier=volume_id)


def _attached_volume(ec2: object) -> str:
    """Create an instance + volume, attach them, return the volume id (``in-use``)."""
    instance_id = ec2.run_instances(ImageId=_AMI, MinCount=1, MaxCount=1)[  # type: ignore[attr-defined]
        "Instances"
    ][0]["InstanceId"]
    volume_id = ec2.create_volume(AvailabilityZone=f"{_REGION}a", Size=1)["VolumeId"]  # type: ignore[attr-defined]
    ec2.attach_volume(VolumeId=volume_id, InstanceId=instance_id, Device="/dev/sdh")  # type: ignore[attr-defined]
    return volume_id


def _volume_state(ec2: object, volume_id: str) -> str:
    return ec2.describe_volumes(VolumeIds=[volume_id])["Volumes"][0]["State"]  # type: ignore[attr-defined]


def _not_found(op: str) -> ClientError:
    return ClientError({"Error": {"Code": "InvalidVolume.NotFound"}}, op)


def _mock_session(client: MagicMock) -> MagicMock:
    session = MagicMock()
    session.client.return_value = client
    return session


# -- registration --


def test_handler_registered_for_volume_attachment_type():
    """The delete handler must be registered so the scan does not fall through to CCAPI.

    CCAPI cannot delete ``AWS::EC2::VolumeAttachment``; without this registration an
    attached volume leaks and blocks the volume delete (the bug this handler fixes).
    """
    import aws_bench.resource_management.cleanup.handlers  # noqa: F401

    assert "AWS::EC2::VolumeAttachment" in CUSTOM_DELETION_REGISTRY


# -- happy path (moto) --


@mock_aws
def test_detaches_attached_volume():
    """An attached volume is detached and left ``available``."""
    ec2 = boto3.client("ec2", region_name=_REGION)
    volume_id = _attached_volume(ec2)
    assert _volume_state(ec2, volume_id) == "in-use"

    result = _delete_volume_attachment(_resource(volume_id), boto3.Session(region_name=_REGION))

    assert result.status == HandlerStatus.SUCCESS
    assert _volume_state(ec2, volume_id) == "available"


@mock_aws
def test_already_detached_volume_is_success():
    """A volume with no attachments is a no-op success (nothing to detach)."""
    ec2 = boto3.client("ec2", region_name=_REGION)
    volume_id = ec2.create_volume(AvailabilityZone=f"{_REGION}a", Size=1)["VolumeId"]

    result = _delete_volume_attachment(_resource(volume_id), boto3.Session(region_name=_REGION))

    assert result.status == HandlerStatus.SUCCESS
    assert _volume_state(ec2, volume_id) == "available"


@mock_aws
def test_missing_volume_is_idempotent_success():
    """A volume that no longer exists yields SUCCESS (idempotent)."""
    result = _delete_volume_attachment(
        _resource("vol-0deadbeef0000000"), boto3.Session(region_name=_REGION)
    )

    assert result.status == HandlerStatus.SUCCESS


# -- failure / edge paths (mocked clients) --


def test_describe_not_found_is_success():
    """A not-found error on describe means the volume is gone → SUCCESS."""
    client = MagicMock()
    client.describe_volumes.side_effect = _not_found("DescribeVolumes")

    result = _delete_volume_attachment(_resource("vol-1"), _mock_session(client))

    assert result.status == HandlerStatus.SUCCESS
    client.detach_volume.assert_not_called()


def test_describe_client_error_is_failed():
    """A non-not-found error on describe maps to FAILED."""
    client = MagicMock()
    client.describe_volumes.side_effect = ClientError(
        {"Error": {"Code": "UnauthorizedOperation"}}, "DescribeVolumes"
    )

    result = _delete_volume_attachment(_resource("vol-1"), _mock_session(client))

    assert result.status == HandlerStatus.FAILED


def test_detach_botocore_error_is_failed():
    """A connection-level error during detach maps to FAILED."""
    client = MagicMock()
    client.describe_volumes.return_value = {
        "Volumes": [{"Attachments": [{"InstanceId": "i-1", "State": "attached"}]}]
    }
    client.detach_volume.side_effect = EndpointConnectionError(endpoint_url="https://ec2")

    result = _delete_volume_attachment(_resource("vol-1"), _mock_session(client))

    assert result.status == HandlerStatus.FAILED


def test_detach_already_detached_race_is_success():
    """An ``IncorrectState`` (already detached between describe and detach) is SUCCESS."""
    client = MagicMock()
    client.describe_volumes.return_value = {
        "Volumes": [{"Attachments": [{"InstanceId": "i-1", "State": "attached"}]}]
    }
    client.detach_volume.side_effect = ClientError(
        {"Error": {"Code": "IncorrectState"}}, "DetachVolume"
    )

    result = _delete_volume_attachment(_resource("vol-1"), _mock_session(client))

    assert result.status == HandlerStatus.SUCCESS


def test_still_attached_after_detach_is_failed(monkeypatch):
    """If the volume never leaves ``in-use`` after detach, surface FAILED (not a false pass)."""
    client = MagicMock()
    client.describe_volumes.return_value = {
        "Volumes": [{"Attachments": [{"InstanceId": "i-1", "State": "attached"}]}]
    }
    # detach accepted, but every subsequent describe still shows it attached.

    import aws_bench.resource_management.cleanup.handlers.ec2_volume_attachment as mod

    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)  # do not sleep through the wait
    monkeypatch.setattr(mod, "_DETACH_WAIT_ATTEMPTS", 3)
    result = _delete_volume_attachment(_resource("vol-1"), _mock_session(client))

    assert result.status == HandlerStatus.FAILED
