"""Tests for the Directory Service directory cleanup handler."""

from __future__ import annotations

from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import (
    CUSTOM_DELETION_REGISTRY,
    PREPARE_REGISTRY,
)
from aws_bench.resource_management.cleanup.handlers import _pca_connector_ad, directory_service
from aws_bench.resource_management.cleanup.handlers.directory_service import _delete, _prepare
from aws_bench.resource_management.cleanup.models import HandlerStatus

# A Managed Microsoft AD directory id — the scan flags every edition (Simple AD,
# Managed Microsoft AD, AD Connector) under AWS::DirectoryService::SimpleAD.
_DIR_ID = "d-9a675a669d"


def _resource() -> Resource:
    return Resource(type="AWS::DirectoryService::SimpleAD", identifier=_DIR_ID)


def _not_found(op: str = "DescribeDirectories") -> ClientError:
    return ClientError({"Error": {"Code": "EntityDoesNotExistException"}}, op)


# -- registration --


def test_handler_registered_for_simple_ad_type():
    """The delete handler must be registered so the scan does not fall through to CCAPI.

    Every directory edition is scanned under this single CFN type, so registering
    it is what lets an agent-created Managed Microsoft AD be torn down at all.
    """
    import aws_bench.resource_management.cleanup.handlers  # noqa: F401

    assert "AWS::DirectoryService::SimpleAD" in CUSTOM_DELETION_REGISTRY


# -- _delete --


def test_delete_success_waits_for_terminal_deletion(monkeypatch):
    """The handler must WAIT for terminal deletion before reporting success.

    ``delete_directory`` is async and ``describe_directories`` keeps listing a
    ``Deleting`` directory, so a handler that returned immediately would let the
    post-run reset verification re-flag the still-present directory.
    """
    monkeypatch.setattr(
        "aws_bench.resource_management.cleanup.handlers.directory_service._WAITER_INTERVAL_SEC",
        0,
    )
    session = MagicMock()
    client = MagicMock()
    session.client.return_value = client
    # Directory still present on the first poll, gone (EntityDoesNotExist) on the second.
    client.describe_directories.side_effect = [
        {"DirectoryDescriptions": [{"DirectoryId": _DIR_ID, "Stage": "Deleting"}]},
        _not_found(),
    ]
    result = _delete(_resource(), session)
    client.delete_directory.assert_called_once_with(DirectoryId=_DIR_ID)
    assert client.describe_directories.call_count >= 2
    assert result.status == HandlerStatus.SUCCESS


def test_delete_treats_empty_description_list_as_gone(monkeypatch):
    """An empty DirectoryDescriptions list also means the directory is gone."""
    monkeypatch.setattr(
        "aws_bench.resource_management.cleanup.handlers.directory_service._WAITER_INTERVAL_SEC",
        0,
    )
    session = MagicMock()
    client = MagicMock()
    session.client.return_value = client
    client.describe_directories.return_value = {"DirectoryDescriptions": []}
    result = _delete(_resource(), session)
    assert result.status == HandlerStatus.SUCCESS


def test_delete_already_gone_is_success():
    """A not-found at the delete call is success — nothing left to wait on."""
    session = MagicMock()
    client = MagicMock()
    session.client.return_value = client
    client.delete_directory.side_effect = _not_found("DeleteDirectory")
    result = _delete(_resource(), session)
    # Already gone at the delete call: no terminal-deletion poll needed.
    client.describe_directories.assert_not_called()
    assert result.status == HandlerStatus.SUCCESS


def test_delete_failure_on_other_error():
    """A non-not-found error from delete_directory maps to FAILED, no poll."""
    session = MagicMock()
    client = MagicMock()
    session.client.return_value = client
    client.delete_directory.side_effect = ClientError(
        {"Error": {"Code": "ClientException"}}, "DeleteDirectory"
    )
    result = _delete(_resource(), session)
    client.describe_directories.assert_not_called()
    assert result.status == HandlerStatus.FAILED


def test_delete_fails_when_directory_never_terminally_deletes(monkeypatch):
    """A directory stuck in Deleting past the bounded wait maps to FAILED, not a hang."""
    # Collapse the bounded poll to a single immediate check so the test is fast.
    monkeypatch.setattr(
        "aws_bench.resource_management.cleanup.handlers.directory_service._WAITER_TIMEOUT_SEC",
        0,
    )
    session = MagicMock()
    client = MagicMock()
    session.client.return_value = client
    # describe_directories always returns the directory -> it never disappears.
    client.describe_directories.return_value = {
        "DirectoryDescriptions": [{"DirectoryId": _DIR_ID, "Stage": "Deleting"}]
    }
    result = _delete(_resource(), session)
    assert result.status == HandlerStatus.FAILED


# -- _prepare (remove PCA Connector AD authorized applications) --


def _pca_client(
    *, connectors: list[dict] | None = None, registrations: list[dict] | None = None
) -> MagicMock:
    """A pca-connector-ad client mock whose get_* immediately reports gone."""
    client = MagicMock()
    conn_pag = MagicMock()
    conn_pag.paginate.return_value = [{"Connectors": connectors or []}]
    reg_pag = MagicMock()
    reg_pag.paginate.return_value = [{"DirectoryRegistrations": registrations or []}]
    client.get_paginator.side_effect = lambda op: {
        "list_connectors": conn_pag,
        "list_directory_registrations": reg_pag,
    }[op]
    # A deleted connector/registration reads back as not-found (pca-connector-ad
    # uses ResourceNotFoundException, unlike Directory Service) -> _gone() True.
    pca_not_found = ClientError({"Error": {"Code": "ResourceNotFoundException"}}, "Get")
    client.get_connector.side_effect = pca_not_found
    client.get_directory_registration.side_effect = pca_not_found
    return client


def test_prepare_registered_for_simple_ad_type():
    """The prepare handler must be registered so authorized apps are removed first."""
    import aws_bench.resource_management.cleanup.handlers  # noqa: F401

    assert "AWS::DirectoryService::SimpleAD" in PREPARE_REGISTRY


def test_prepare_success_when_no_authorized_apps():
    """No PCA apps -> SUCCESS (not SKIPPED) so the directory delete still runs."""
    session = MagicMock()
    session.client.return_value = _pca_client()
    result = _prepare(_resource(), session)
    # SUCCESS (not SKIPPED): a SKIPPED prepare would abort DeleteDirectory entirely.
    assert result.status == HandlerStatus.SUCCESS


def test_prepare_removes_authorized_apps_and_succeeds(monkeypatch):
    """A connector/registration for the directory is removed; prepare is SUCCESS."""
    monkeypatch.setattr(_pca_connector_ad, "_WAITER_INTERVAL_SEC", 0)
    pca = _pca_client(
        connectors=[{"Arn": "conn-1", "DirectoryId": _DIR_ID}],
        registrations=[{"Arn": "reg-1", "DirectoryId": _DIR_ID}],
    )
    session = MagicMock()
    session.client.return_value = pca
    result = _prepare(_resource(), session)
    assert result.status == HandlerStatus.SUCCESS
    pca.delete_connector.assert_called_once_with(ConnectorArn="conn-1")
    pca.delete_directory_registration.assert_called_once_with(DirectoryRegistrationArn="reg-1")


def test_prepare_failed_on_real_delete_error(monkeypatch):
    """A non-not-found error removing an authorized app maps to FAILED."""
    monkeypatch.setattr(_pca_connector_ad, "_WAITER_INTERVAL_SEC", 0)
    pca = _pca_client(connectors=[{"Arn": "conn-1", "DirectoryId": _DIR_ID}])
    pca.delete_connector.side_effect = ClientError(
        {"Error": {"Code": "ThrottlingException"}}, "DeleteConnector"
    )
    session = MagicMock()
    session.client.return_value = pca
    result = _prepare(_resource(), session)
    assert result.status == HandlerStatus.FAILED


def test_full_delete_removes_pca_apps_before_deleting_directory(monkeypatch):
    """The chained prepare removes authorized apps BEFORE DeleteDirectory is called.

    This is the ordering the fix guarantees: DeleteDirectory otherwise fails with
    "still has authorized applications".
    """
    monkeypatch.setattr(_pca_connector_ad, "_WAITER_INTERVAL_SEC", 0)
    monkeypatch.setattr(directory_service, "_WAITER_INTERVAL_SEC", 0)

    pca = _pca_client(connectors=[{"Arn": "conn-1", "DirectoryId": _DIR_ID}])
    ds = MagicMock()
    ds.describe_directories.side_effect = _not_found()  # directory gone right after delete

    manager = MagicMock()
    manager.attach_mock(pca, "pca")
    manager.attach_mock(ds, "ds")

    session = MagicMock()
    session.client.side_effect = lambda service, **_kw: {"pca-connector-ad": pca, "ds": ds}[service]

    full_delete = CUSTOM_DELETION_REGISTRY["AWS::DirectoryService::SimpleAD"]
    result = full_delete(_resource(), session)

    assert result.status == HandlerStatus.SUCCESS
    ds.delete_directory.assert_called_once_with(DirectoryId=_DIR_ID)
    op_names = [c[0] for c in manager.mock_calls]
    assert op_names.index("pca.delete_connector") < op_names.index("ds.delete_directory")
