"""Tests for EKS Pod Identity Association cleanup handler."""

from __future__ import annotations

from unittest.mock import MagicMock

from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handlers.eks_pod_identity import (
    _delete as _delete_pod_identity,
)
from aws_bench.resource_management.cleanup.models import HandlerStatus


def _resource(identifier: str = "my-cluster|assoc-12345") -> Resource:
    return Resource(type="AWS::EKS::PodIdentityAssociation", identifier=identifier)


class TestDeletePodIdentity:
    def test_deletes_association_successfully(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client

        result = _delete_pod_identity(_resource(), session)

        client.delete_pod_identity_association.assert_called_once_with(
            clusterName="my-cluster", associationId="assoc-12345"
        )
        assert result.status == HandlerStatus.SUCCESS
        assert "Deleted pod identity association" in result.message

    def test_skips_when_not_found(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_pod_identity_association.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException"}},
            "DeletePodIdentityAssociation",
        )

        result = _delete_pod_identity(_resource(), session)

        assert result.status == HandlerStatus.SKIPPED
        assert "not found" in result.message

    def test_fails_on_access_denied(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_pod_identity_association.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException"}},
            "DeletePodIdentityAssociation",
        )

        result = _delete_pod_identity(_resource(), session)

        assert result.status == HandlerStatus.FAILED
        assert "Failed to delete pod identity association" in result.message

    def test_fails_on_invalid_identifier(self):
        session = MagicMock()

        result = _delete_pod_identity(_resource("no-pipe-separator"), session)

        assert result.status == HandlerStatus.FAILED
        assert "Invalid identifier format" in result.message

    def test_fails_on_connection_error(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_pod_identity_association.side_effect = BotoCoreError()

        result = _delete_pod_identity(_resource(), session)

        assert result.status == HandlerStatus.FAILED
        assert "Connection error" in result.message

    def test_parses_composite_identifier_correctly(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client

        result = _delete_pod_identity(_resource("prod-cluster|a-abcdef123456"), session)

        client.delete_pod_identity_association.assert_called_once_with(
            clusterName="prod-cluster", associationId="a-abcdef123456"
        )
        assert result.status == HandlerStatus.SUCCESS
