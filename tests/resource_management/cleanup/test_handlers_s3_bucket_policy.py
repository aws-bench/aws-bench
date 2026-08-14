"""Tests for S3 bucket policy cleanup handler."""

from __future__ import annotations

from unittest.mock import MagicMock

from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handlers.s3_bucket_policy import (
    _delete as _delete_bucket_policy,
)
from aws_bench.resource_management.cleanup.models import HandlerStatus

_BUCKET = "my-test-bucket"


def _resource(identifier: str = _BUCKET) -> Resource:
    return Resource(type="AWS::S3::BucketPolicy", identifier=identifier)


class TestDeleteBucketPolicy:
    def test_deletes_bucket_policy_successfully(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client

        result = _delete_bucket_policy(_resource(), session)

        client.delete_bucket_policy.assert_called_once_with(Bucket=_BUCKET)
        assert result.status == HandlerStatus.SUCCESS
        assert "Deleted bucket policy" in result.message

    def test_skips_when_bucket_not_found(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_bucket_policy.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket"}}, "DeleteBucketPolicy"
        )

        result = _delete_bucket_policy(_resource(), session)

        assert result.status == HandlerStatus.SKIPPED
        assert "not found" in result.message

    def test_skips_when_no_policy(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_bucket_policy.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucketPolicy"}}, "DeleteBucketPolicy"
        )

        result = _delete_bucket_policy(_resource(), session)

        assert result.status == HandlerStatus.SKIPPED
        assert "not found" in result.message

    def test_fails_on_access_denied(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_bucket_policy.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "DeleteBucketPolicy"
        )

        result = _delete_bucket_policy(_resource(), session)

        assert result.status == HandlerStatus.FAILED
        assert "Failed to delete bucket policy" in result.message

    def test_fails_on_connection_error(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_bucket_policy.side_effect = BotoCoreError()

        result = _delete_bucket_policy(_resource(), session)

        assert result.status == HandlerStatus.FAILED
        assert "Connection error" in result.message
