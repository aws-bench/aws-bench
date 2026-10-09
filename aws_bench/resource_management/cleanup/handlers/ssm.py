"""SSM Patch Baseline cleanup handler.

CloudControl (CCAPI) cannot operate on ``AWS::SSM::PatchBaseline`` — a
``get_resource`` / ``delete_resource`` returns ``InvalidRequestException``, which
the CCAPI deleter classifies as "unsupported" and SKIPS. With no custom handler,
an agent-created patch baseline survives the reset as an orphan and contaminates
the account. This handler deletes it via the SSM service API.

The scanned identifier is the ``BaselineId`` — either a bare ``pb-...`` id or a
full ARN. AWS ships PREDEFINED baselines (e.g. ``AWS-DefaultPatchBaseline``) whose
``BaselineId`` is an ARN under the AWS-owned account (``733109147000``); those are
undeletable and must never be targeted. The handler re-guards by ownership: a
baseline whose identifier is an ARN in an account OTHER than the caller's is
AWS-owned and skipped. A bare ``pb-...`` id or an ARN in the caller's account is a
custom, deletable baseline.

``DeletePatchBaseline`` is rejected (``ResourceInUseException``) while the baseline
is still registered to any patch group. A prepare handler therefore deregisters
every patch group bound to the baseline (``describe_patch_groups`` ->
``deregister_patch_baseline_for_patch_group``) before the delete, mirroring the
CodeBuild resource-policy and DHCP disassociate-then-delete patterns. A baseline
with no registrations (the common agent-created case) is a clean no-op.
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

_PATCH_BASELINE_TYPE = "AWS::SSM::PatchBaseline"
# delete_patch_baseline errors meaning the baseline is already gone.
_NOT_FOUND_CODES = ("DoesNotExistException", "InvalidResourceId")


def _owner_account_if_arn(baseline_id: str) -> str | None:
    """Return the account component if ``baseline_id`` is an ARN, else None.

    A patch-baseline ARN is
    ``arn:aws:ssm:<region>:<account>:patchbaseline/pb-...`` (account is field 4).
    A bare ``pb-...`` id has no account and returns None (caller-owned by default).
    """
    if not baseline_id.startswith("arn:"):
        return None
    parts = baseline_id.split(":")
    return parts[4] if len(parts) > 4 and parts[4] else None


def _baseline_pb_id(baseline_id: str) -> str:
    """Return the bare ``pb-...`` id from a baseline id that may be an ARN.

    ``describe_patch_groups`` reports ``BaselineIdentity.BaselineId`` as a bare
    ``pb-...``, while the scanned identifier may be a full ARN; compare on the
    ``pb-...`` tail so both forms match.
    """
    return baseline_id.rsplit("/", 1)[-1] if "/" in baseline_id else baseline_id


def _is_aws_owned(baseline_id: str, session: boto3.Session) -> bool | None:
    """True if the baseline is an AWS-predefined (foreign-account) ARN.

    Returns None if ownership cannot be determined (caller account unresolvable),
    so callers can fail closed rather than guess. A bare ``pb-...`` id is
    caller-owned by definition and resolves without an STS call.
    """
    owner = _owner_account_if_arn(baseline_id)
    if owner is None:
        return False
    caller_account = build_client(session, "sts").get_caller_identity().get("Account")
    if not caller_account:
        return None
    return owner != caller_account


def _result(
    baseline_id: str, status: HandlerStatus, action: str = "delete", message: str = ""
) -> HandlerResult:
    return HandlerResult(
        resource_id=baseline_id,
        resource_type=_PATCH_BASELINE_TYPE,
        action=action,
        status=status,
        message=message,
    )


@resource_handler(_PATCH_BASELINE_TYPE, role="prepare")
def _prepare_patch_baseline(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Deregister every patch group bound to the baseline so delete is not rejected.

    ``DeletePatchBaseline`` throws ``ResourceInUseException`` while any patch group
    is registered to the baseline. This finds those registrations
    (``describe_patch_groups``) and removes each
    (``deregister_patch_baseline_for_patch_group``). AWS-predefined baselines are
    left untouched (the delete handler skips them anyway); a baseline with no
    registrations is a clean no-op.
    """
    baseline_id = resource.identifier
    try:
        if _is_aws_owned(baseline_id, session):
            return _result(
                baseline_id,
                HandlerStatus.SKIPPED,
                action="prepare",
                message="AWS-owned predefined patch baseline; nothing to prepare",
            )
        ssm = build_client(session, "ssm")
        target_pb = _baseline_pb_id(baseline_id)
        for page in ssm.get_paginator("describe_patch_groups").paginate():
            for mapping in page.get("Mappings", []):
                identity = mapping.get("BaselineIdentity", {})
                if _baseline_pb_id(identity.get("BaselineId", "")) != target_pb:
                    continue
                patch_group = mapping.get("PatchGroup")
                if not patch_group:
                    continue
                ssm.deregister_patch_baseline_for_patch_group(
                    BaselineId=baseline_id, PatchGroup=patch_group
                )
                logger.debug(
                    "Deregistered patch group '%s' from baseline '%s'", patch_group, target_pb
                )
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") in _NOT_FOUND_CODES:
            return _result(
                baseline_id,
                HandlerStatus.SUCCESS,
                action="prepare",
                message="Patch baseline already gone; nothing to prepare",
            )
        return _result(
            baseline_id,
            HandlerStatus.FAILED,
            action="prepare",
            message=f"Failed to deregister patch groups: {e}",
        )
    except BotoCoreError as e:
        return _result(
            baseline_id,
            HandlerStatus.FAILED,
            action="prepare",
            message=f"Connection error deregistering patch groups: {e}",
        )
    return _result(baseline_id, HandlerStatus.SUCCESS, action="prepare")


@resource_handler(_PATCH_BASELINE_TYPE, role="delete")
def _delete_patch_baseline(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete a custom SSM patch baseline via the service API (CCAPI cannot).

    AWS-predefined baselines (ARN under a different account) are skipped untouched.
    """
    baseline_id = resource.identifier

    try:
        aws_owned = _is_aws_owned(baseline_id, session)
    except (ClientError, BotoCoreError) as e:
        return _result(
            baseline_id,
            HandlerStatus.FAILED,
            message=f"Could not resolve caller account to classify baseline ownership: {e}",
        )
    if aws_owned is None:
        return _result(
            baseline_id,
            HandlerStatus.FAILED,
            message="Could not resolve caller account to classify baseline ownership",
        )
    if aws_owned:
        return _result(
            baseline_id,
            HandlerStatus.SKIPPED,
            message="AWS-owned predefined patch baseline; left untouched",
        )

    try:
        build_client(session, "ssm").delete_patch_baseline(BaselineId=baseline_id)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") in _NOT_FOUND_CODES:
            return _result(
                baseline_id, HandlerStatus.SUCCESS, message="Patch baseline already gone"
            )
        return _result(
            baseline_id, HandlerStatus.FAILED, message=f"Failed to delete patch baseline: {e}"
        )
    except BotoCoreError as e:
        return _result(
            baseline_id,
            HandlerStatus.FAILED,
            message=f"Connection error deleting patch baseline: {e}",
        )

    logger.debug("Deleted SSM patch baseline '%s'", baseline_id[:LOG_TRUNCATE_MEDIUM])
    return _result(baseline_id, HandlerStatus.SUCCESS)
