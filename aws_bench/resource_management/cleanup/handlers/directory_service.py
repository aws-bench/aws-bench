"""AWS Directory Service directory cleanup handler.

An agent-created directory (``ds:CreateDirectory`` / ``CreateMicrosoftAD``) is not
stack-managed and survives normal CFN cleanup, so the post-run reset/cleanup scan
flags it as an orphan.

The ``ds:describe_directories`` fast-scan lister maps EVERY directory edition —
Simple AD, Managed Microsoft AD, and AD Connector — to the single CloudFormation
type ``AWS::DirectoryService::SimpleAD`` (that is the type the resource is scanned
and flagged under, regardless of its real ``Type``). So this one handler covers
all editions, and ``ds:delete_directory`` is the correct teardown API for every
one of them. (This is why a Managed Microsoft AD orphan surfaces in the scan as
``AWS::DirectoryService::SimpleAD``.)

Registering this handler also removes a region-dependent leak: the reset pipeline
otherwise falls back to the CloudControl API (CCAPI) to delete this type, but
CCAPI's read handler for ``AWS::DirectoryService::SimpleAD`` is only present in
some regions (e.g. us-east-1) and absent in others (e.g. us-east-2, where CCAPI
returns ``UnsupportedActionException``). When the existence probe hits an
unsupported region the resource is silently skipped and leaks. This custom
handler runs in the ``custom_delete`` phase, BEFORE the CCAPI fallback, so the
directory is torn down deterministically in every region.

The directory's requester-managed ENIs — the domain-controller network
interfaces AWS attaches in the VPC — are released by AWS as part of directory
deletion. They are already dropped from the reset verify set by the AWS-managed
ownership probe (``RequesterManaged`` ENIs are service-owned), so no separate ENI
handler is needed here.

The delete is asynchronous: ``delete_directory`` returns immediately while the
directory transitions through ``Deleting``, and ``describe_directories`` keeps
listing it until it is fully gone. The reset re-verifies exactly once, right
after deletion, so the delete step must WAIT for terminal deletion or the still-
``Deleting`` directory is re-flagged as a new resource and reset fails.

A directory that still has a PCA Connector AD connector or directory
registration attached cannot be deleted ("still has authorized applications").
The ``prepare`` handler removes those authorized applications first — see
``_pca_connector_ad`` — so ``DeleteDirectory`` succeeds.
"""

from __future__ import annotations

import boto3
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.handlers._pca_connector_ad import (
    remove_directory_authorized_apps,
)
from aws_bench.resource_management.cleanup.handlers._service_delete import service_delete
from aws_bench.resource_management.cleanup.models import HandlerResult, HandlerStatus
from aws_bench.resource_management.utils.polling import wait_until

_NOT_FOUND_CODES = ("EntityDoesNotExistException",)

_WAITER_TIMEOUT_SEC = 600
_WAITER_INTERVAL_SEC = 15


def _wait_for_terminal_deletion(client: BaseClient, directory_id: str) -> None:
    """Block until the directory is actually gone.

    Raises:
        ClientError: If the directory is still present after the bounded wait, so
            ``service_delete`` maps the result to FAILED (rather than hanging).
    """

    def _gone() -> bool:
        try:
            described = client.describe_directories(DirectoryIds=[directory_id])
        except ClientError as e:
            if e.response.get("Error", {}).get("Code", "") in _NOT_FOUND_CODES:
                return True  # Fully deleted.
            raise  # transient (throttling/etc.) — wait_until swallows and retries
        return not described.get("DirectoryDescriptions", [])

    if wait_until(_gone, timeout=_WAITER_TIMEOUT_SEC, interval=_WAITER_INTERVAL_SEC):
        return
    # Still present after the bounded wait — raise so service_delete maps to FAILED.
    raise ClientError(
        {
            "Error": {
                "Code": "DeletionTimeout",
                "Message": f"directory still present after {_WAITER_TIMEOUT_SEC}s",
            }
        },
        "DescribeDirectories",
    )


@resource_handler("AWS::DirectoryService::SimpleAD", role="prepare")
def _prepare(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Remove the directory's PCA Connector AD authorized applications first.

    A directory that still has a PCA Connector AD connector or directory
    registration attached cannot be deleted (``DeleteDirectory`` fails with "still
    has authorized applications"). Those types are CCAPI-deletable, but CCAPI runs
    only AFTER the custom-delete ``DeleteDirectory``, so the delete races ahead and
    fails. The prepare phase runs before custom-delete, so removing them here
    unblocks the directory teardown deterministically.

    Returns SUCCESS whether or not any app was removed (so the delete still runs);
    only a real removal failure maps to FAILED.
    """
    try:
        removed = remove_directory_authorized_apps(session, resource.identifier)
    except (ClientError, BotoCoreError) as e:
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="prepare",
            status=HandlerStatus.FAILED,
            message=f"Failed to remove PCA Connector AD authorized applications: {e}",
        )
    message = (
        f"Removed {removed} PCA Connector AD authorized application(s)"
        if removed
        else "No PCA Connector AD authorized applications"
    )
    return HandlerResult(
        resource_id=resource.identifier,
        resource_type=resource.type,
        action="prepare",
        status=HandlerStatus.SUCCESS,
        message=message,
    )


@resource_handler("AWS::DirectoryService::SimpleAD", role="delete")
def _delete(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete the directory via ``ds:delete_directory``, then wait for it to be gone."""
    return service_delete(
        resource,
        session,
        client_name="ds",
        op_name="delete_directory",
        id_param="DirectoryId",
        not_found_codes=_NOT_FOUND_CODES,
        already_gone_message="Directory already gone",
        log_label="Directory Service directory",
        post_delete=lambda client: _wait_for_terminal_deletion(client, resource.identifier),
    )
