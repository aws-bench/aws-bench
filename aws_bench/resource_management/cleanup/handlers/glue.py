"""Glue Database cleanup handler.

Lake Formation gates DDL on a governed Data Catalog resource behind its own permission
table: ``glue:DeleteDatabase`` requires the Lake Formation ``DROP`` permission on that
database, and IAM alone does not satisfy it — an ``AdministratorAccess`` principal is
still denied. Lake Formation grants that permission implicitly to the principal that
*creates* a resource, so a database created by one role is undeletable by another.

The prepare step grants ``DROP`` to the CloudFormation execution role and to the caller's
own role, on the database and on a wildcard over its tables. Granting requires the caller
to be a Lake Formation data lake administrator; when it is not, the grant is skipped and
the delete proceeds unchanged, since the caller may already hold ``DROP``.
"""

from __future__ import annotations

import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.account_management.preexisting import effective_cfn_role
from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.models import HandlerResult, HandlerStatus
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)

_GRANT_PERMISSIONS = ("DROP",)

# The database or the target principal does not exist.
_NOT_FOUND_CODES = ("EntityNotFoundException",)

_CANNOT_GRANT_CODES = ("AccessDeniedException",)

# Outcomes that leave the environment no worse than an absent handler.
_BENIGN_CODES = frozenset(_NOT_FOUND_CODES + _CANNOT_GRANT_CODES)

# Contention on the Lake Formation permission table, which clears on its own. botocore
# models no retry for lakeformation, so it is applied here.
_RETRY_CODES = ("ConcurrentModificationException",)
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SEC = 1.0


def _caller_principal(session: boto3.Session, identity_arn: str) -> str | None:
    """Return the caller's Lake Formation principal ARN, or None if it has none.

    ``get_caller_identity`` reports an ``sts`` assumed-role ARN, which Lake Formation
    rejects. The IAM role ARN it maps to carries the role's path, which the ``sts`` form
    omits. A user ARN is already a valid principal and passes through unchanged.
    """
    fields = identity_arn.split(":")
    resource = fields[5] if len(fields) > 5 else ""
    if resource.startswith("user/"):
        return identity_arn
    if not resource.startswith("assumed-role/"):
        return None
    role_name = resource.split("/")[1]
    if not role_name:
        return None
    try:
        return build_client(session, "iam").get_role(RoleName=role_name)["Role"]["Arn"]
    except (ClientError, BotoCoreError) as exc:
        logger.debug("Could not resolve caller role '%s': %s", role_name, exc)
        return None


def _delete_principals(session: boto3.Session) -> list[str]:
    """Principal ARNs of the roles that perform the delete, ops role first, deduped.

    Raises:
        LookupError: If the caller's account cannot be determined.
    """
    identity = build_client(session, "sts").get_caller_identity()
    account = identity.get("Account")
    if not account:
        raise LookupError("get_caller_identity returned no Account")
    principals = [f"arn:aws:iam::{account}:role/{effective_cfn_role()}"]
    caller = _caller_principal(session, identity.get("Arn", ""))
    if caller and caller not in principals:
        principals.append(caller)
    return principals


def _grant(client: object, principal: str, target: dict) -> str | None:
    """Grant the permissions on one target. Returns None on success, else an error code."""
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            client.grant_permissions(  # type: ignore[attr-defined]
                Principal={"DataLakePrincipalIdentifier": principal},
                Resource=target,
                Permissions=list(_GRANT_PERMISSIONS),
                PermissionsWithGrantOption=[],
            )
            return None
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in _RETRY_CODES and attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(_RETRY_BACKOFF_SEC * (attempt + 1))
                continue
            return code
        except BotoCoreError as exc:
            return type(exc).__name__
    return None


@resource_handler("AWS::Glue::Database", role="prepare")
def _prepare(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Grant Lake Formation DROP on the database and its tables before deletion."""
    database = resource.identifier
    database_target = {"Database": {"Name": database}}
    table_target = {"Table": {"DatabaseName": database, "TableWildcard": {}}}

    def _result(status: HandlerStatus, message: str) -> HandlerResult:
        if status is not HandlerStatus.SUCCESS:
            logger.warning("Glue database '%s' prepare %s: %s", database, status.value, message)
        return HandlerResult(
            resource_id=database,
            resource_type=resource.type,
            action="prepare",
            status=status,
            message=message,
        )

    try:
        client = build_client(session, "lakeformation")
        principals = _delete_principals(session)
    except (ClientError, BotoCoreError, LookupError) as exc:
        return _result(HandlerStatus.FAILED, f"Could not resolve deleting principals: {exc}")

    # Every principal is attempted even after one is rejected: Lake Formation rejects a
    # grant to a principal that does not exist, and some accounts have no ops role.
    database_granted: list[str] = []
    errors: list[str] = []
    for principal in principals:
        for target in (database_target, table_target):
            code = _grant(client, principal, target)
            if code is not None:
                errors.append(code)
            elif target is database_target:
                database_granted.append(principal)

    # Only DROP on the database authorizes DeleteDatabase; a table-wildcard grant does not.
    if database_granted:
        message = (
            f"Granted {'/'.join(_GRANT_PERMISSIONS)} on '{database}' "
            f"to {', '.join(database_granted)}"
        )
        if errors:
            message += f" ({len(errors)} rejected: {', '.join(sorted(set(errors)))})"
        return _result(HandlerStatus.SUCCESS, message)

    codes = ", ".join(sorted(set(errors))) or "no principal to grant to"
    if errors and all(code in _BENIGN_CODES for code in errors):
        return _result(
            HandlerStatus.SKIPPED,
            f"No Lake Formation grant possible on '{database}' ({codes}); "
            "deletion proceeds without one",
        )
    return _result(
        HandlerStatus.FAILED,
        f"Failed to grant Lake Formation permissions on '{database}': {codes}",
    )
