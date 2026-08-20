"""Tests for DynamoDB cleanup handlers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handlers.dynamodb import (
    _disable_deletion_protection,
    _prepare_global_table,
    _prepare_table,
)
from aws_bench.resource_management.cleanup.models import HandlerStatus

_NOT_FOUND = ClientError(
    {"Error": {"Code": "ResourceNotFoundException", "Message": "not found"}}, "DescribeTable"
)


@pytest.fixture(autouse=True)
def _fast_wait_until():
    """Patch the poll so tests never sleep; default success."""
    with patch(
        "aws_bench.resource_management.cleanup.handlers.dynamodb.wait_until",
        return_value=True,
    ) as mock:
        yield mock


def _patch_client(client: MagicMock):
    return patch(
        "aws_bench.resource_management.cleanup.handlers.dynamodb.build_client",
        return_value=client,
    )


def test_disable_deletion_protection_disables_and_succeeds():
    client = MagicMock()
    client.describe_table.return_value = {
        "Table": {"TableStatus": "ACTIVE", "DeletionProtectionEnabled": True}
    }
    with _patch_client(client):
        result = _disable_deletion_protection(MagicMock(), "t1", "AWS::DynamoDB::Table")
    client.update_table.assert_called_once_with(TableName="t1", DeletionProtectionEnabled=False)
    assert result.status == HandlerStatus.SUCCESS


def test_disable_deletion_protection_noop_when_already_disabled():
    client = MagicMock()
    client.describe_table.return_value = {
        "Table": {"TableStatus": "ACTIVE", "DeletionProtectionEnabled": False}
    }
    with _patch_client(client):
        result = _disable_deletion_protection(MagicMock(), "t1", "AWS::DynamoDB::Table")
    client.update_table.assert_not_called()
    assert result.status == HandlerStatus.SUCCESS


def test_disable_deletion_protection_skips_missing_table():
    client = MagicMock()
    client.describe_table.side_effect = _NOT_FOUND
    with _patch_client(client):
        result = _disable_deletion_protection(MagicMock(), "gone", "AWS::DynamoDB::Table")
    client.update_table.assert_not_called()
    assert result.status == HandlerStatus.SKIPPED


def test_disable_deletion_protection_times_out(_fast_wait_until):
    _fast_wait_until.return_value = False
    client = MagicMock()
    client.describe_table.return_value = {
        "Table": {"TableStatus": "ACTIVE", "DeletionProtectionEnabled": True}
    }
    with _patch_client(client):
        result = _disable_deletion_protection(MagicMock(), "t1", "AWS::DynamoDB::Table")
    assert result.status == HandlerStatus.FAILED


def test_prepare_handlers_pass_identifier_as_table_name():
    client = MagicMock()
    client.describe_table.return_value = {
        "Table": {"TableStatus": "ACTIVE", "DeletionProtectionEnabled": True}
    }
    with _patch_client(client):
        # both scanned types resolve to the same table NAME; both should disable protection
        r_table = _prepare_table(
            Resource(type="AWS::DynamoDB::Table", identifier="storage-app-objects"), MagicMock()
        )
        r_global = _prepare_global_table(
            Resource(type="AWS::DynamoDB::GlobalTable", identifier="storage-app-objects"),
            MagicMock(),
        )
    assert r_table.status == HandlerStatus.SUCCESS
    assert r_global.status == HandlerStatus.SUCCESS
    client.update_table.assert_called_with(
        TableName="storage-app-objects", DeletionProtectionEnabled=False
    )
