"""DynamoDB cleanup handlers.

A table created with ``DeletionProtectionEnabled`` rejects DeleteTable with
``ValidationException: Resource cannot be deleted as it is currently protected
against deletion. Disable deletion protection first.`` — so the cleanup/reset
sweep leaves it as an orphan forever and fails the run (observed on
create-vpc-valkey-dynamodb, which enables deletion protection).

The prepare hook disables deletion protection so the subsequent CCAPI delete
succeeds; no custom delete handler is needed. The same physical table surfaces
under two scanned types — ``AWS::DynamoDB::Table`` (ListTables) and, when it is a
global table, ``AWS::DynamoDB::GlobalTable`` (ListGlobalTables) — and both
identifiers are the table NAME, so the same prepare is registered for both.
Deletion protection is a per-(regional-)table property; disabling it here in the
region being swept makes that replica deletable, and deleting it dissolves the
global table.
"""

from __future__ import annotations

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.models import HandlerResult, HandlerStatus
from aws_bench.resource_management.utils.polling import wait_until
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)

_TABLE_ACTIVE_TIMEOUT = 120
_TABLE_ACTIVE_INTERVAL = 5


def _table_deletable(client, table_name: str) -> bool:
    """True once the table is ACTIVE with deletion protection disabled.

    ``update_table`` puts the table in UPDATING briefly; DeleteTable rejects a
    non-ACTIVE table, so wait for it to settle. A vanished table (already deleted)
    counts as deletable — nothing left to protect.
    """
    try:
        table = client.describe_table(TableName=table_name)["Table"]
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") == "ResourceNotFoundException":
            return True
        logger.debug("Error checking table '%s': %s", table_name, e)
        return False
    return table.get("TableStatus") == "ACTIVE" and not table.get("DeletionProtectionEnabled")


def _disable_deletion_protection(
    session: boto3.Session, table_name: str, resource_type: str
) -> HandlerResult:
    """Disable deletion protection on ``table_name`` so it can be deleted."""
    client = build_client(session, "dynamodb")
    try:
        table = client.describe_table(TableName=table_name)["Table"]
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") == "ResourceNotFoundException":
            return HandlerResult(
                resource_id=table_name,
                resource_type=resource_type,
                action="prepare",
                status=HandlerStatus.SKIPPED,
                message="Table not found",
            )
        return HandlerResult(
            resource_id=table_name,
            resource_type=resource_type,
            action="prepare",
            status=HandlerStatus.FAILED,
            message=f"Error describing table: {e}",
        )
    except BotoCoreError as e:
        return HandlerResult(
            resource_id=table_name,
            resource_type=resource_type,
            action="prepare",
            status=HandlerStatus.FAILED,
            message=f"Connection error: {e}",
        )

    if not table.get("DeletionProtectionEnabled"):
        return HandlerResult(
            resource_id=table_name,
            resource_type=resource_type,
            action="prepare",
            status=HandlerStatus.SUCCESS,
            message="Deletion protection already disabled",
        )

    try:
        client.update_table(TableName=table_name, DeletionProtectionEnabled=False)
    except (ClientError, BotoCoreError) as e:
        return HandlerResult(
            resource_id=table_name,
            resource_type=resource_type,
            action="prepare",
            status=HandlerStatus.FAILED,
            message=f"Failed to disable deletion protection: {e}",
        )

    if not wait_until(
        lambda: _table_deletable(client, table_name),
        timeout=_TABLE_ACTIVE_TIMEOUT,
        interval=_TABLE_ACTIVE_INTERVAL,
    ):
        return HandlerResult(
            resource_id=table_name,
            resource_type=resource_type,
            action="prepare",
            status=HandlerStatus.FAILED,
            message="Timed out waiting for table to become deletable",
        )

    return HandlerResult(
        resource_id=table_name,
        resource_type=resource_type,
        action="prepare",
        status=HandlerStatus.SUCCESS,
        message="Deletion protection disabled",
    )


# The lister emits the table NAME for both types (ListTables → TableName,
# ListGlobalTables → GlobalTableName), which is what describe_table/update_table want.
@resource_handler("AWS::DynamoDB::Table", role="prepare")
def _prepare_table(resource: Resource, session: boto3.Session) -> HandlerResult:
    return _disable_deletion_protection(session, resource.identifier, resource.type)


@resource_handler("AWS::DynamoDB::GlobalTable", role="prepare")
def _prepare_global_table(resource: Resource, session: boto3.Session) -> HandlerResult:
    return _disable_deletion_protection(session, resource.identifier, resource.type)
