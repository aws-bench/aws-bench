"""IAM role cleanup handler.

A task-created ``AWS::IAM::Role`` that is a member of an instance profile cannot
be deleted by raw CCAPI: ``DeleteRole`` fails with "Cannot delete entity, must
remove roles from instance profile first" (observed with EC2ImageBuilderRole /
EC2SSMRole / EMR_EC2_DefaultRole). The dependency
removal already existed in ``cross_service`` but only ran on the rarely-used
``handle_stuck`` path; this prepare handler runs it on the reset new-resource
path so the subsequent CCAPI delete succeeds.

Registered as a *prepare* handler only — the actual delete still flows through
the CCAPI fallback, which keeps the service-linked / protected-role skip guards
(``Deleter._check_deletable``). The prepare step applies the same guards so it
never strips policies off a service-linked role or OrganizationAccountAccessRole.
"""

from __future__ import annotations

import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import (
    PROTECTED_IAM_ROLE_NAMES,
    SERVICE_ROLE_PREFIX,
    Resource,
)
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.handlers.cross_service import (
    detach_iam_role_dependencies,
)
from aws_bench.resource_management.cleanup.models import HandlerResult, HandlerStatus
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)


@resource_handler("AWS::IAM::Role", role="prepare")
def _prepare(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Remove the role from instance profiles and detach its policies."""
    role_name = resource.identifier
    if role_name.startswith(SERVICE_ROLE_PREFIX) or role_name in PROTECTED_IAM_ROLE_NAMES:
        # Mirrors Deleter._check_deletable: never touch service-linked or
        # cross-account-access roles. CCAPI will skip the delete too.
        return HandlerResult(
            resource_id=role_name,
            resource_type=resource.type,
            action="prepare",
            status=HandlerStatus.SKIPPED,
            message="Protected or service-linked role; left untouched",
        )
    iam = build_client(session, "iam")
    try:
        detach_iam_role_dependencies(iam, role_name)
    except iam.exceptions.NoSuchEntityException:
        return HandlerResult(
            resource_id=role_name,
            resource_type=resource.type,
            action="prepare",
            status=HandlerStatus.SKIPPED,
            message="Role not found",
        )
    except (ClientError, BotoCoreError) as e:
        # Log with a stack trace before returning FAILED: the cleaner only logs handlers
        # that RAISE, so a returned-FAILED result would otherwise leave this — the exact
        # instance-profile/policy-detach failure this handler exists to fix — with no
        # traceable record. Mirrors cross_service._force_delete_iam_role's logging.
        logger.warning(f"Failed to detach IAM role dependencies for '{role_name}': {e}")
        return HandlerResult(
            resource_id=role_name,
            resource_type=resource.type,
            action="prepare",
            status=HandlerStatus.FAILED,
            message=f"Failed to detach role dependencies: {e}",
        )
    return HandlerResult(
        resource_id=role_name,
        resource_type=resource.type,
        action="prepare",
        status=HandlerStatus.SUCCESS,
        message="Detached policies and removed from instance profiles",
    )


_DELETE_CONFLICT_CODES = ("DeleteConflict",)
_NOT_FOUND_CODES = ("NoSuchEntity",)
_MAX_DELETE_ATTEMPTS = 3
_DELETE_RETRY_DELAY_SEC = 15


@resource_handler("AWS::IAM::Role", role="delete")
def _delete(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete the IAM role, retrying on DeleteConflict.

    DeleteConflict occurs when the role is still referenced by an async-draining
    resource (e.g. an EKS nodegroup whose instances haven't fully terminated).
    A short retry with backoff gives the referencing resource time to release.
    """
    role_name = resource.identifier
    if role_name.startswith(SERVICE_ROLE_PREFIX) or role_name in PROTECTED_IAM_ROLE_NAMES:
        return HandlerResult(
            resource_id=role_name,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.SKIPPED,
            message="Protected or service-linked role; left untouched",
        )
    iam = build_client(session, "iam")
    last_error = ""
    for attempt in range(1, _MAX_DELETE_ATTEMPTS + 1):
        try:
            iam.delete_role(RoleName=role_name)
            return HandlerResult(
                resource_id=role_name,
                resource_type=resource.type,
                action="delete",
                status=HandlerStatus.SUCCESS,
                message="Deleted role",
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in _NOT_FOUND_CODES:
                return HandlerResult(
                    resource_id=role_name,
                    resource_type=resource.type,
                    action="delete",
                    status=HandlerStatus.SKIPPED,
                    message="Role not found",
                )
            if code in _DELETE_CONFLICT_CODES and attempt < _MAX_DELETE_ATTEMPTS:
                logger.debug(
                    "DeleteConflict on role '%s' (attempt %d/%d), retrying in %ds",
                    role_name,
                    attempt,
                    _MAX_DELETE_ATTEMPTS,
                    _DELETE_RETRY_DELAY_SEC,
                )
                time.sleep(_DELETE_RETRY_DELAY_SEC)
                last_error = str(e)
                continue
            return HandlerResult(
                resource_id=role_name,
                resource_type=resource.type,
                action="delete",
                status=HandlerStatus.FAILED,
                message=f"Failed to delete role: {e}",
            )
        except BotoCoreError as e:
            return HandlerResult(
                resource_id=role_name,
                resource_type=resource.type,
                action="delete",
                status=HandlerStatus.FAILED,
                message=f"Connection error: {e}",
            )
    # Should not reach here, but safety net
    return HandlerResult(
        resource_id=role_name,
        resource_type=resource.type,
        action="delete",
        status=HandlerStatus.FAILED,
        message=f"Exhausted {_MAX_DELETE_ATTEMPTS} attempts: {last_error}",
    )
