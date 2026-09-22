"""Tests for the ACM Private CA (``AWS::ACMPCA::CertificateAuthority``) handler.

moto's ``acmpca`` backend does not model the ACTIVE-state deletion constraint
(``delete_certificate_authority`` sets ``DELETED`` regardless of status and
never yields ``PENDING_DELETION``), so it cannot exercise the disable-then-delete
ordering this handler exists to enforce. These tests therefore drive the ACM PCA
client with ``MagicMock`` — the same approach the directory-service handler tests
use — so the ordering and error mapping are asserted directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

from botocore.exceptions import ClientError, EndpointConnectionError

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import CUSTOM_DELETION_REGISTRY
from aws_bench.resource_management.cleanup.handlers.acmpca import _delete_certificate_authority
from aws_bench.resource_management.cleanup.models import HandlerStatus

_CA_ARN = "arn:aws:acm-pca:us-east-1:111122223333:certificate-authority/abc-123"


def _resource() -> Resource:
    return Resource(type="AWS::ACMPCA::CertificateAuthority", identifier=_CA_ARN)


def _client_with_status(status: str) -> MagicMock:
    client = MagicMock()
    client.describe_certificate_authority.return_value = {
        "CertificateAuthority": {"Status": status}
    }
    return client


def _session_with(client: MagicMock) -> MagicMock:
    session = MagicMock()
    session.client.side_effect = lambda service, **_kw: client
    return session


# -- registration --


def test_handler_registered_for_acmpca_certificate_authority_type():
    """The delete handler must be registered so an ACTIVE CA is not left to CCAPI.

    CCAPI cannot delete an ACTIVE/EXPIRED CA, so without this registration the CA
    leaks and fails the reset (the bug this handler fixes).
    """
    import aws_bench.resource_management.cleanup.handlers  # noqa: F401

    assert "AWS::ACMPCA::CertificateAuthority" in CUSTOM_DELETION_REGISTRY


# -- disable-then-delete ordering --


def test_active_ca_is_disabled_then_scheduled_for_deletion():
    """An ACTIVE CA is DISABLED first, then deletion is scheduled (min restore window)."""
    client = _client_with_status("ACTIVE")

    result = _delete_certificate_authority(_resource(), _session_with(client))

    assert result.status == HandlerStatus.SUCCESS
    # Disable must precede delete: an ACTIVE CA cannot be deleted directly.
    assert client.mock_calls == [
        call.describe_certificate_authority(CertificateAuthorityArn=_CA_ARN),
        call.update_certificate_authority(CertificateAuthorityArn=_CA_ARN, Status="DISABLED"),
        call.delete_certificate_authority(
            CertificateAuthorityArn=_CA_ARN, PermanentDeletionTimeInDays=7
        ),
    ]


def test_expired_ca_is_deleted_without_disabling():
    """An EXPIRED CA is directly deletable — disabling it would be rejected.

    UpdateCertificateAuthority accepts only ACTIVE/DISABLED CAs, and
    DeleteCertificateAuthority lists EXPIRED among the directly-deletable states.
    """
    client = _client_with_status("EXPIRED")

    result = _delete_certificate_authority(_resource(), _session_with(client))

    assert result.status == HandlerStatus.SUCCESS
    client.update_certificate_authority.assert_not_called()
    client.delete_certificate_authority.assert_called_once_with(
        CertificateAuthorityArn=_CA_ARN, PermanentDeletionTimeInDays=7
    )


def test_disabled_ca_is_deleted_without_disabling():
    """A CA already in a directly-deletable state is deleted without a disable call."""
    client = _client_with_status("DISABLED")

    result = _delete_certificate_authority(_resource(), _session_with(client))

    assert result.status == HandlerStatus.SUCCESS
    client.update_certificate_authority.assert_not_called()
    client.delete_certificate_authority.assert_called_once_with(
        CertificateAuthorityArn=_CA_ARN, PermanentDeletionTimeInDays=7
    )


def test_pending_certificate_ca_is_deleted_without_disabling():
    """A PENDING_CERTIFICATE CA is directly deletable (no disable needed)."""
    client = _client_with_status("PENDING_CERTIFICATE")

    result = _delete_certificate_authority(_resource(), _session_with(client))

    assert result.status == HandlerStatus.SUCCESS
    client.update_certificate_authority.assert_not_called()
    client.delete_certificate_authority.assert_called_once()


# -- idempotent already-gone states --


def test_pending_deletion_ca_is_success_without_further_calls():
    """A CA already scheduled for deletion is SUCCESS — the lister excludes it."""
    client = _client_with_status("PENDING_DELETION")

    result = _delete_certificate_authority(_resource(), _session_with(client))

    assert result.status == HandlerStatus.SUCCESS
    client.update_certificate_authority.assert_not_called()
    client.delete_certificate_authority.assert_not_called()


def test_deleted_ca_is_success_without_further_calls():
    """A CA already DELETED is SUCCESS."""
    client = _client_with_status("DELETED")

    result = _delete_certificate_authority(_resource(), _session_with(client))

    assert result.status == HandlerStatus.SUCCESS
    client.delete_certificate_authority.assert_not_called()


def test_not_found_on_describe_is_success():
    """A CA that no longer exists (not-found on describe) is SUCCESS (idempotent)."""
    client = MagicMock()
    client.describe_certificate_authority.side_effect = ClientError(
        {"Error": {"Code": "ResourceNotFoundException"}}, "DescribeCertificateAuthority"
    )

    result = _delete_certificate_authority(_resource(), _session_with(client))

    assert result.status == HandlerStatus.SUCCESS
    client.delete_certificate_authority.assert_not_called()


def test_not_found_on_delete_is_success():
    """A CA that vanishes between describe and delete is SUCCESS."""
    client = _client_with_status("DISABLED")
    client.delete_certificate_authority.side_effect = ClientError(
        {"Error": {"Code": "ResourceNotFoundException"}}, "DeleteCertificateAuthority"
    )

    result = _delete_certificate_authority(_resource(), _session_with(client))

    assert result.status == HandlerStatus.SUCCESS


# -- failure paths --


def test_invalid_state_on_delete_is_failed():
    """A delete rejected for state is FAILED (surfaced, not silently dropped)."""
    client = _client_with_status("DISABLED")
    client.delete_certificate_authority.side_effect = ClientError(
        {"Error": {"Code": "InvalidStateException"}}, "DeleteCertificateAuthority"
    )

    result = _delete_certificate_authority(_resource(), _session_with(client))

    assert result.status == HandlerStatus.FAILED


def test_botocore_error_is_failed():
    """A connection-level BotoCoreError maps to FAILED."""
    client = MagicMock()
    client.describe_certificate_authority.side_effect = EndpointConnectionError(
        endpoint_url="https://acm-pca"
    )

    result = _delete_certificate_authority(_resource(), _session_with(client))

    assert result.status == HandlerStatus.FAILED
