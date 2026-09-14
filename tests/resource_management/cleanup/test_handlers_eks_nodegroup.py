"""Tests for EKS Nodegroup cleanup handler."""

from __future__ import annotations

from unittest.mock import MagicMock

from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handlers.eks_nodegroup import (
    _delete as _delete_nodegroup,
)
from aws_bench.resource_management.cleanup.models import HandlerStatus

_HANDLER = "aws_bench.resource_management.cleanup.handlers.eks_nodegroup"


def _resource(identifier: str = "my-cluster|my-nodegroup") -> Resource:
    return Resource(type="AWS::EKS::Nodegroup", identifier=identifier)


def _not_found(op: str = "DescribeNodegroup") -> ClientError:
    return ClientError({"Error": {"Code": "ResourceNotFoundException"}}, op)


def _present() -> dict:
    return {"nodegroup": {"nodegroupName": "my-nodegroup", "status": "DELETING"}}


class TestDeleteNodegroup:
    def test_deletes_nodegroup_and_waits_until_gone(self, monkeypatch):
        """A freshly submitted delete must wait for terminal deletion before success."""
        monkeypatch.setattr(f"{_HANDLER}._WAITER_INTERVAL_SEC", 0)
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        # Present on the first poll, gone (ResourceNotFound) on the second.
        client.describe_nodegroup.side_effect = [_present(), _not_found()]

        result = _delete_nodegroup(_resource(), session)

        client.delete_nodegroup.assert_called_once_with(
            clusterName="my-cluster", nodegroupName="my-nodegroup"
        )
        assert client.describe_nodegroup.call_count >= 2
        assert result.status == HandlerStatus.SUCCESS
        assert "Deleted nodegroup" in result.message

    def test_waits_until_gone_when_already_deleting(self, monkeypatch):
        """ResourceInUseException means it is already deleting — still wait for it to go."""
        monkeypatch.setattr(f"{_HANDLER}._WAITER_INTERVAL_SEC", 0)
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_nodegroup.side_effect = ClientError(
            {"Error": {"Code": "ResourceInUseException"}}, "DeleteNodegroup"
        )
        client.describe_nodegroup.side_effect = [_present(), _not_found()]

        result = _delete_nodegroup(_resource(), session)

        assert client.describe_nodegroup.call_count >= 2
        assert result.status == HandlerStatus.SUCCESS
        assert "already deleting" in result.message

    def test_tolerates_transient_polling_error_then_succeeds(self, monkeypatch):
        """A transient error while polling is swallowed; the wait continues to completion."""
        monkeypatch.setattr(f"{_HANDLER}._WAITER_INTERVAL_SEC", 0)
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        # Transient throttle, then the nodegroup is gone.
        client.describe_nodegroup.side_effect = [
            ClientError({"Error": {"Code": "ThrottlingException"}}, "DescribeNodegroup"),
            _not_found(),
        ]

        result = _delete_nodegroup(_resource(), session)

        assert result.status == HandlerStatus.SUCCESS
        assert "Deleted nodegroup" in result.message

    def test_fails_when_nodegroup_never_terminally_deletes(self, monkeypatch):
        """A nodegroup stuck DELETING past the bounded wait maps to FAILED, not a hang."""
        monkeypatch.setattr(f"{_HANDLER}._WAITER_TIMEOUT_SEC", 0)
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_nodegroup.return_value = _present()

        result = _delete_nodegroup(_resource(), session)

        assert result.status == HandlerStatus.FAILED
        assert "did not complete" in result.message

    def test_fails_on_persistent_polling_error(self, monkeypatch):
        """A persistent polling error is retried until the bounded wait elapses -> FAILED."""
        monkeypatch.setattr(f"{_HANDLER}._WAITER_INTERVAL_SEC", 0)
        monkeypatch.setattr(f"{_HANDLER}._WAITER_TIMEOUT_SEC", 0.05)
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_nodegroup.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException"}}, "DescribeNodegroup"
        )

        result = _delete_nodegroup(_resource(), session)

        assert result.status == HandlerStatus.FAILED
        assert "did not complete" in result.message

    def test_skips_when_not_found(self):
        """A not-found at the delete call is skipped — nothing left to wait on."""
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_nodegroup.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException"}}, "DeleteNodegroup"
        )

        result = _delete_nodegroup(_resource(), session)

        client.describe_nodegroup.assert_not_called()
        assert result.status == HandlerStatus.SKIPPED
        assert "not found" in result.message

    def test_fails_on_access_denied(self):
        """A non-not-found error from delete_nodegroup maps to FAILED, no poll."""
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_nodegroup.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException"}}, "DeleteNodegroup"
        )

        result = _delete_nodegroup(_resource(), session)

        client.describe_nodegroup.assert_not_called()
        assert result.status == HandlerStatus.FAILED
        assert "Failed to delete nodegroup" in result.message

    def test_fails_on_invalid_identifier(self):
        session = MagicMock()

        result = _delete_nodegroup(_resource("no-pipe-separator"), session)

        assert result.status == HandlerStatus.FAILED
        assert "Invalid identifier format" in result.message

    def test_fails_on_connection_error(self):
        """A BotoCoreError at the delete call maps to FAILED, no poll."""
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_nodegroup.side_effect = BotoCoreError()

        result = _delete_nodegroup(_resource(), session)

        client.describe_nodegroup.assert_not_called()
        assert result.status == HandlerStatus.FAILED
        assert "Connection error" in result.message
