"""PCA Connector for AD teardown helpers.

A private-CA connector for Active Directory registers itself as an *authorized
application* on its AWS Managed Microsoft AD directory. Two resource types hold
that authorization:

* ``AWS::PCAConnectorAD::Connector`` — the connector itself (``create_connector``
  takes a ``DirectoryId``), and
* ``AWS::PCAConnectorAD::DirectoryRegistration`` — the directory registration
  (``create_directory_registration`` takes a ``DirectoryId``).

While either exists the directory ``DeleteDirectory`` call is rejected with
"Cannot delete the directory because it still has authorized applications". Both
are ordinary CCAPI-deletable orphans, but CCAPI deletion runs *after* the
custom-delete ``DeleteDirectory``, so the directory delete races ahead and fails.

These helpers remove both types for a given directory and wait for terminal
deletion, so the directory-service ``prepare`` handler can clear the
authorization before ``DeleteDirectory`` is attempted. Connectors are deleted
before directory registrations (connector first, its registration second).
"""

from __future__ import annotations

import boto3
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.utils.polling import wait_until
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)

_CLIENT_NAME = "pca-connector-ad"
_NOT_FOUND_CODES = ("ResourceNotFoundException",)

_WAITER_TIMEOUT_SEC = 300
_WAITER_INTERVAL_SEC = 10


def _error_code(error: ClientError) -> str:
    return error.response.get("Error", {}).get("Code", "")


def _list_arns_for_directory(
    client: BaseClient, op_name: str, result_key: str, directory_id: str
) -> list[str]:
    """Return the ARNs of ``op_name`` items whose ``DirectoryId`` matches.

    Listing is best-effort: if the PCA Connector AD API is unavailable in the
    region (or errors), there is nothing this helper can remove, so it logs and
    returns an empty list rather than blocking the directory teardown.
    """
    try:
        arns: list[str] = []
        for page in client.get_paginator(op_name).paginate():
            for item in page.get(result_key, []):
                if item.get("DirectoryId") == directory_id and item.get("Arn"):
                    arns.append(item["Arn"])
        return arns
    except (ClientError, BotoCoreError) as e:
        logger.warning("pca-connector-ad.%s skipped for %s: %s", op_name, directory_id, e)
        return []


def _delete_and_wait(
    client: BaseClient, delete_op: str, get_op: str, id_param: str, arn: str
) -> int:
    """Delete one connector/registration and wait for it to be gone.

    Returns 1 if a delete was issued, 0 if it was already gone. Raises on a real
    (non not-found) delete failure, or if the resource never terminally deletes
    within the bounded wait, so the caller maps it to FAILED (fail-closed).
    """
    try:
        getattr(client, delete_op)(**{id_param: arn})
    except ClientError as e:
        if _error_code(e) in _NOT_FOUND_CODES:
            return 0
        raise

    def _gone() -> bool:
        try:
            getattr(client, get_op)(**{id_param: arn})
        except ClientError as e:
            if _error_code(e) in _NOT_FOUND_CODES:
                return True
            raise
        return False

    if wait_until(_gone, timeout=_WAITER_TIMEOUT_SEC, interval=_WAITER_INTERVAL_SEC):
        return 1
    # Still present after the bounded wait — raise so the caller maps it to FAILED
    # (fail-closed) rather than reporting a not-actually-removed app as removed.
    raise ClientError(
        {
            "Error": {
                "Code": "DeletionTimeout",
                "Message": f"{arn} still present after {_WAITER_TIMEOUT_SEC}s",
            }
        },
        get_op,
    )


def remove_directory_authorized_apps(session: boto3.Session, directory_id: str) -> int:
    """Remove every PCA Connector AD connector/registration for ``directory_id``.

    Returns the number of authorized applications removed. Raises ``ClientError``
    / ``BotoCoreError`` on a real delete failure.
    """
    client = build_client(session, _CLIENT_NAME)
    connectors = _list_arns_for_directory(client, "list_connectors", "Connectors", directory_id)
    registrations = _list_arns_for_directory(
        client, "list_directory_registrations", "DirectoryRegistrations", directory_id
    )

    removed = 0
    for arn in connectors:
        removed += _delete_and_wait(
            client, "delete_connector", "get_connector", "ConnectorArn", arn
        )
    for arn in registrations:
        removed += _delete_and_wait(
            client,
            "delete_directory_registration",
            "get_directory_registration",
            "DirectoryRegistrationArn",
            arn,
        )
    return removed
