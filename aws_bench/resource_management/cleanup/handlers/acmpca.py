"""ACM Private CA (``AWS::ACMPCA::CertificateAuthority``) cleanup handler.

An agent-created private CA is flagged as an orphan by the fast-scan lister
(``list_certificate_authorities``), but Cloud Control (CCAPI) cannot tear it
down: a CA in the ``ACTIVE`` state is "not in a valid state for deletion", so
``DeleteCertificateAuthority`` is rejected and the CA leaks, failing the post-run
reset fail-closed.

Deletion is a two-step state machine:

1. ``UpdateCertificateAuthority(Status=DISABLED)`` — an ``ACTIVE`` CA must be
   disabled before it can be deleted. CAs already in a directly-deletable state
   (``CREATING``, ``PENDING_CERTIFICATE``, ``DISABLED``, ``EXPIRED``, ``FAILED``)
   are deleted as-is. (``UpdateCertificateAuthority`` itself only accepts an
   ``ACTIVE`` or ``DISABLED`` CA, so disabling is scoped to exactly ``ACTIVE``.)
2. ``DeleteCertificateAuthority(PermanentDeletionTimeInDays=7)`` — deletion is
   *scheduled*, not immediate: the CA moves to ``PENDING_DELETION`` for a restore
   window (7 days is the minimum). That is enough to clear the orphan: the CA
   lister keeps only live states and excludes ``PENDING_DELETION``, so a
   scheduled CA no longer surfaces as an orphan and the reset re-verify passes.

A CA already gone, already ``DELETED``, or already ``PENDING_DELETION`` maps to
SUCCESS (idempotent). Any other failure maps to FAILED so it is surfaced rather
than silently dropped.
"""

from __future__ import annotations

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import LOG_TRUNCATE_MEDIUM, Resource
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.models import HandlerResult, HandlerStatus
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)

_NOT_FOUND_CODES = ("ResourceNotFoundException",)

# Minimum restore window ACM PCA accepts for a scheduled deletion.
_PERMANENT_DELETION_DAYS = 7

# The only state a CA must leave (via DISABLED) before it can be deleted.
# ``UpdateCertificateAuthority`` accepts only ACTIVE/DISABLED CAs, and
# ``DeleteCertificateAuthority`` lists EXPIRED among the directly-deletable
# states, so disabling is scoped to exactly ACTIVE.
_DISABLE_BEFORE_DELETE_STATES = ("ACTIVE",)

# States for which no delete call is needed — the CA is already gone or already
# scheduled for deletion (and thus excluded from the orphan scan).
_ALREADY_TERMINAL_STATES = ("DELETED", "PENDING_DELETION")


def _success(resource: Resource, message: str = "") -> HandlerResult:
    return HandlerResult(
        resource_id=resource.identifier,
        resource_type=resource.type,
        action="delete",
        status=HandlerStatus.SUCCESS,
        message=message,
    )


def _failed(resource: Resource, message: str) -> HandlerResult:
    return HandlerResult(
        resource_id=resource.identifier,
        resource_type=resource.type,
        action="delete",
        status=HandlerStatus.FAILED,
        message=message,
    )


def _error_code(error: ClientError) -> str:
    return error.response.get("Error", {}).get("Code", "")


@resource_handler("AWS::ACMPCA::CertificateAuthority", role="delete")
def _delete_certificate_authority(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Disable (if needed) then schedule deletion of a private CA.

    CCAPI cannot delete an ``ACTIVE`` CA (it must be disabled first), so this
    handler drives the disable-then-delete state machine via the ACM PCA API. CAs
    already in a directly-deletable state, including ``EXPIRED``, are deleted
    as-is.
    """
    arn = resource.identifier
    try:
        client = build_client(session, "acm-pca")
        described = client.describe_certificate_authority(CertificateAuthorityArn=arn)
        status = described.get("CertificateAuthority", {}).get("Status", "")

        if status in _ALREADY_TERMINAL_STATES:
            return _success(resource, f"CA already {status.lower()}")

        if status in _DISABLE_BEFORE_DELETE_STATES:
            client.update_certificate_authority(CertificateAuthorityArn=arn, Status="DISABLED")

        client.delete_certificate_authority(
            CertificateAuthorityArn=arn,
            PermanentDeletionTimeInDays=_PERMANENT_DELETION_DAYS,
        )
    except ClientError as e:
        if _error_code(e) in _NOT_FOUND_CODES:
            return _success(resource, "CA already gone")
        return _failed(resource, f"Failed to delete ACM PCA certificate authority: {e}")
    except BotoCoreError as e:
        return _failed(resource, f"Connection error deleting ACM PCA certificate authority: {e}")

    logger.debug(f"Scheduled deletion of ACM PCA CA '{arn[:LOG_TRUNCATE_MEDIUM]}'")
    return _success(resource)
