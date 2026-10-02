"""Tests for the PCA Connector for AD teardown helpers.

moto has no ``pca-connector-ad`` backend, so these tests drive the client with
``MagicMock`` (the same approach the directory-service handler tests use).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from aws_bench.resource_management.cleanup.handlers import _pca_connector_ad
from aws_bench.resource_management.cleanup.handlers._pca_connector_ad import (
    remove_directory_authorized_apps,
)

_DIR_ID = "d-9a675a669d"
_OTHER_DIR_ID = "d-000000abcd"


def _not_found(op: str) -> ClientError:
    return ClientError({"Error": {"Code": "ResourceNotFoundException"}}, op)


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
    # A deleted connector/registration reads back as not-found -> _gone() True.
    client.get_connector.side_effect = _not_found("GetConnector")
    client.get_directory_registration.side_effect = _not_found("GetDirectoryRegistration")
    return client


def _session_with(client: MagicMock) -> MagicMock:
    session = MagicMock()
    session.client.side_effect = lambda service, **_kw: client
    return session


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch):
    """Collapse the terminal-deletion poll interval so tests do not sleep."""
    monkeypatch.setattr(_pca_connector_ad, "_WAITER_INTERVAL_SEC", 0)


def test_removes_only_apps_for_the_target_directory():
    """Connectors/registrations for other directories are left untouched."""
    client = _pca_client(
        connectors=[
            {"Arn": "conn-target", "DirectoryId": _DIR_ID},
            {"Arn": "conn-other", "DirectoryId": _OTHER_DIR_ID},
        ],
        registrations=[
            {"Arn": "reg-target", "DirectoryId": _DIR_ID},
            {"Arn": "reg-other", "DirectoryId": _OTHER_DIR_ID},
        ],
    )

    removed = remove_directory_authorized_apps(_session_with(client), _DIR_ID)

    assert removed == 2
    client.delete_connector.assert_called_once_with(ConnectorArn="conn-target")
    client.delete_directory_registration.assert_called_once_with(
        DirectoryRegistrationArn="reg-target"
    )


def test_connectors_deleted_before_registrations():
    """The connector is removed before its directory registration."""
    client = _pca_client(
        connectors=[{"Arn": "conn-1", "DirectoryId": _DIR_ID}],
        registrations=[{"Arn": "reg-1", "DirectoryId": _DIR_ID}],
    )

    remove_directory_authorized_apps(_session_with(client), _DIR_ID)

    op_names = [c[0] for c in client.mock_calls]
    assert op_names.index("delete_connector") < op_names.index("delete_directory_registration")


def test_no_apps_removes_nothing():
    """A directory with no PCA Connector AD apps removes nothing and does not error."""
    client = _pca_client()

    removed = remove_directory_authorized_apps(_session_with(client), _DIR_ID)

    assert removed == 0
    client.delete_connector.assert_not_called()
    client.delete_directory_registration.assert_not_called()


def test_list_failure_is_best_effort_and_removes_nothing():
    """If listing the PCA Connector AD API fails, teardown is a no-op, not an error."""
    client = _pca_client(registrations=[{"Arn": "reg-1", "DirectoryId": _DIR_ID}])
    conn_pag = MagicMock()
    conn_pag.paginate.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException"}}, "ListConnectors"
    )
    reg_pag = MagicMock()
    reg_pag.paginate.return_value = [
        {"DirectoryRegistrations": [{"Arn": "reg-1", "DirectoryId": _DIR_ID}]}
    ]
    client.get_paginator.side_effect = lambda op: {
        "list_connectors": conn_pag,
        "list_directory_registrations": reg_pag,
    }[op]

    removed = remove_directory_authorized_apps(_session_with(client), _DIR_ID)

    # Connector listing failed (best-effort skip); the registration still removed.
    assert removed == 1
    client.delete_connector.assert_not_called()
    client.delete_directory_registration.assert_called_once_with(DirectoryRegistrationArn="reg-1")


def test_delete_not_found_counts_as_already_gone():
    """A connector that vanishes before delete is not counted and does not error."""
    client = _pca_client(connectors=[{"Arn": "conn-1", "DirectoryId": _DIR_ID}])
    client.delete_connector.side_effect = _not_found("DeleteConnector")

    removed = remove_directory_authorized_apps(_session_with(client), _DIR_ID)

    assert removed == 0


def test_real_delete_error_propagates():
    """A non-not-found delete failure propagates so the caller maps it to FAILED."""
    client = _pca_client(connectors=[{"Arn": "conn-1", "DirectoryId": _DIR_ID}])
    client.delete_connector.side_effect = ClientError(
        {"Error": {"Code": "ThrottlingException"}}, "DeleteConnector"
    )

    with pytest.raises(ClientError):
        remove_directory_authorized_apps(_session_with(client), _DIR_ID)


def test_delete_that_never_terminally_deletes_raises(monkeypatch):
    """A connector that never disappears within the bounded wait raises (fail-closed)."""
    monkeypatch.setattr(_pca_connector_ad, "_WAITER_TIMEOUT_SEC", 0)
    client = _pca_client(connectors=[{"Arn": "conn-1", "DirectoryId": _DIR_ID}])
    # get_connector keeps returning the connector -> it never reads back as gone.
    client.get_connector.side_effect = None
    client.get_connector.return_value = {"Connector": {"Arn": "conn-1", "Status": "DELETING"}}

    with pytest.raises(ClientError):
        remove_directory_authorized_apps(_session_with(client), _DIR_ID)
