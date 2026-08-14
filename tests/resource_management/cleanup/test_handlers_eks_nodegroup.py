"""Tests for EKS Nodegroup cleanup handler."""

from __future__ import annotations

from unittest.mock import MagicMock

from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handlers.eks_nodegroup import (
    _delete as _delete_nodegroup,
)
from aws_bench.resource_management.cleanup.models import HandlerStatus


def _resource(identifier: str = "my-cluster|my-nodegroup") -> Resource:
    return Resource(type="AWS::EKS::Nodegroup", identifier=identifier)


class TestDeleteNodegroup:
    def test_deletes_nodegroup_successfully(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client

        result = _delete_nodegroup(_resource(), session)

        client.delete_nodegroup.assert_called_once_with(
            clusterName="my-cluster", nodegroupName="my-nodegroup"
        )
        assert result.status == HandlerStatus.SUCCESS
        assert "Initiated nodegroup deletion" in result.message

    def test_skips_when_not_found(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_nodegroup.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException"}}, "DeleteNodegroup"
        )

        result = _delete_nodegroup(_resource(), session)

        assert result.status == HandlerStatus.SKIPPED
        assert "not found" in result.message

    def test_succeeds_when_already_deleting(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_nodegroup.side_effect = ClientError(
            {"Error": {"Code": "ResourceInUseException"}}, "DeleteNodegroup"
        )

        result = _delete_nodegroup(_resource(), session)

        assert result.status == HandlerStatus.SUCCESS
        assert "already deleting" in result.message

    def test_fails_on_access_denied(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_nodegroup.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException"}}, "DeleteNodegroup"
        )

        result = _delete_nodegroup(_resource(), session)

        assert result.status == HandlerStatus.FAILED
        assert "Failed to delete nodegroup" in result.message

    def test_fails_on_invalid_identifier(self):
        session = MagicMock()

        result = _delete_nodegroup(_resource("no-pipe-separator"), session)

        assert result.status == HandlerStatus.FAILED
        assert "Invalid identifier format" in result.message

    def test_fails_on_connection_error(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_nodegroup.side_effect = BotoCoreError()

        result = _delete_nodegroup(_resource(), session)

        assert result.status == HandlerStatus.FAILED
        assert "Connection error" in result.message
