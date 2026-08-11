"""Tests for EC2 Image (AMI) cleanup handler."""

from __future__ import annotations

from unittest.mock import MagicMock

from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handlers.ec2_image import (
    _delete as _delete_image,
)
from aws_bench.resource_management.cleanup.models import HandlerStatus

_IMAGE_ID = "ami-0123456789abcdef0"


def _resource(identifier: str = _IMAGE_ID) -> Resource:
    return Resource(type="AWS::EC2::Image", identifier=identifier)


def _image_with_snapshots(*snap_ids: str) -> dict:
    return {
        "Images": [
            {
                "ImageId": _IMAGE_ID,
                "BlockDeviceMappings": [{"Ebs": {"SnapshotId": sid}} for sid in snap_ids],
            }
        ]
    }


class TestDeleteImage:
    def test_deregisters_ami_and_deletes_snapshots(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_images.return_value = _image_with_snapshots("snap-aaa", "snap-bbb")

        result = _delete_image(_resource(), session)

        client.deregister_image.assert_called_once_with(ImageId=_IMAGE_ID)
        assert client.delete_snapshot.call_count == 2
        client.delete_snapshot.assert_any_call(SnapshotId="snap-aaa")
        client.delete_snapshot.assert_any_call(SnapshotId="snap-bbb")
        assert result.status == HandlerStatus.SUCCESS
        assert "2 snapshot(s)" in result.message

    def test_ami_with_no_snapshots(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_images.return_value = {
            "Images": [{"ImageId": _IMAGE_ID, "BlockDeviceMappings": []}]
        }

        result = _delete_image(_resource(), session)

        client.deregister_image.assert_called_once_with(ImageId=_IMAGE_ID)
        client.delete_snapshot.assert_not_called()
        assert result.status == HandlerStatus.SUCCESS
        assert "0 snapshot(s)" in result.message

    def test_skips_when_image_not_found_on_describe(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_images.side_effect = ClientError(
            {"Error": {"Code": "InvalidAMIID.NotFound"}}, "DescribeImages"
        )

        result = _delete_image(_resource(), session)

        assert result.status == HandlerStatus.SKIPPED
        assert "not found" in result.message

    def test_skips_when_no_images_returned(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_images.return_value = {"Images": []}

        result = _delete_image(_resource(), session)

        assert result.status == HandlerStatus.SKIPPED
        assert "not found" in result.message
        client.deregister_image.assert_not_called()

    def test_deregister_not_found_still_cleans_snapshots(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_images.return_value = _image_with_snapshots("snap-aaa")
        client.deregister_image.side_effect = ClientError(
            {"Error": {"Code": "InvalidAMIID.NotFound"}}, "DeregisterImage"
        )

        result = _delete_image(_resource(), session)

        client.delete_snapshot.assert_called_once_with(SnapshotId="snap-aaa")
        assert result.status == HandlerStatus.SUCCESS

    def test_snapshot_not_found_continues(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_images.return_value = _image_with_snapshots("snap-aaa", "snap-bbb")
        client.delete_snapshot.side_effect = [
            ClientError({"Error": {"Code": "InvalidSnapshot.NotFound"}}, "DeleteSnapshot"),
            None,
        ]

        result = _delete_image(_resource(), session)

        assert result.status == HandlerStatus.SUCCESS
        assert client.delete_snapshot.call_count == 2

    def test_partial_snapshot_failure(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_images.return_value = _image_with_snapshots("snap-aaa", "snap-bbb")
        client.delete_snapshot.side_effect = [
            None,
            ClientError({"Error": {"Code": "UnauthorizedOperation"}}, "DeleteSnapshot"),
        ]

        result = _delete_image(_resource(), session)

        assert result.status == HandlerStatus.FAILED
        assert "1 snapshot(s) failed" in result.message

    def test_fails_on_deregister_error(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_images.return_value = _image_with_snapshots("snap-aaa")
        client.deregister_image.side_effect = ClientError(
            {"Error": {"Code": "UnauthorizedOperation"}}, "DeregisterImage"
        )

        result = _delete_image(_resource(), session)

        assert result.status == HandlerStatus.FAILED
        assert "Failed to deregister image" in result.message

    def test_fails_on_connection_error(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_images.side_effect = BotoCoreError()

        result = _delete_image(_resource(), session)

        assert result.status == HandlerStatus.FAILED
        assert "Connection error" in result.message
