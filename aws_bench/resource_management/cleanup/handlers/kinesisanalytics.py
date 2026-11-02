"""Kinesis Data Analytics cleanup handlers.

An agent that builds a Managed Service for Apache Flink "Studio" application
(a ``AWS::KinesisAnalyticsV2::Application`` notebook, "deployed as an
application" into a service-managed ``environment-*-flink-studio`` CloudFormation
stack) leaves Kinesis Analytics resources that normal reset cannot remove:

- The scanner lists the one Studio application **twice** — as
  ``AWS::KinesisAnalyticsV2::Application`` and as ``AWS::KinesisAnalytics::Application``
  (v1). The v2 handler deletes it; the v1 entry's ``DescribeApplication`` then
  raises ``UnsupportedOperationException`` ("created/updated by kinesisanalyticsv2
  SDK"). The v1 handler treats that as already-handled rather than a failure.
- **Stack-managed applications must be deleted via their STACK.** When the agent
  deploys the Studio app through a CloudFormation stack, the stack contains an
  ``ApplicationCloudWatchLoggingOption`` child whose delete requires the parent
  application to exist. Deleting the application natively first therefore sends
  the stack's teardown to ``DELETE_FAILED`` (``ResourceNotFoundException`` on the
  logging option). The handler detects stack membership
  (``describe_stack_resources`` by physical id — managing stacks carry
  agent-chosen names, so name patterns are unreliable) and deletes the stack,
  letting CloudFormation cascade in the correct order; an already-broken
  ``DELETE_FAILED`` stack is retried with ``RetainResources``.
- ``DeleteApplication`` returns success while the application stays **visible to
  the reset scanner's ``list_applications``** (under both the v2 and v1 types) for
  minutes — the core residual for the unmanaged (no-stack) variant. After any
  delete (native or stack-cascaded) the handler polls that same listing until the
  application clears from both types.

Both application APIs delete the same way, and both require the application's
``CreateTimestamp`` (a conditional token that guards against a stale delete):

1. ``DescribeApplication`` to read ``ApplicationDetail.CreateTimestamp``.
2. ``DeleteApplication(ApplicationName=..., CreateTimestamp=...)``.

Custom delete handlers run before the CCAPI fallback in the reset pipeline, so
removing the applications here lets the reset succeed.
"""

from __future__ import annotations

import time

import boto3
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError, WaiterError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.models import (
    HandlerResult,
    HandlerStatus,
    is_service_managed_studio_stack,
)
from aws_bench.resource_management.fastscan.runtime import collect
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)

_NOT_FOUND_CODES = ("ResourceNotFoundException",)
# Raised by the v1 API for an application the v2 SDK created/manages.
_V2_MANAGED_CODE = "UnsupportedOperationException"
# A deleted KDA application stays VISIBLE to the reset scanner's list_applications
# for minutes (DeleteApplication is async), and a stack-backed Studio app's managed
# stack self-completes over a similar window. Poll the scanner listing (and the
# managed stack, when present) for true absence, backing off up to a grace window
# sized to comfortably exceed the observed latency. The window intentionally does
# NOT target the multi-hour horizon seen when a delete is re-issued late; tune
# _TEARDOWN_MAX_ATTEMPTS if the measured pure latency ever approaches it.
_TEARDOWN_INITIAL_DELAY = 10
_TEARDOWN_MAX_DELAY = 60
_TEARDOWN_BACKOFF = 2.0
_TEARDOWN_MAX_ATTEMPTS = 18  # ~16 min total with the backoff above
_TERMINAL_STACK_STATUSES = ("DELETE_COMPLETE",)
# CloudFormation resource types under which a KDA application appears in a stack.
_KDA_CFN_APP_TYPES = (
    "AWS::KinesisAnalyticsV2::Application",
    "AWS::KinesisAnalytics::Application",
)
# Bounded wait for a stack-managed teardown (DeleteStack cascade): ~10 min per wait.
_STACK_DELETE_WAITER_CONFIG = {"Delay": 15, "MaxAttempts": 40}


def _application_name_from_identifier(identifier: str) -> str:
    """Extract the application name from the scanned identifier.

    The v1 lister uses ``id_field="ApplicationARN"`` and the v2 lister
    ``id_field="ApplicationName"``, so the identifier may be either the full ARN
    (``arn:aws:kinesisanalytics:<region>:<account>:application/<name>``) or a bare
    name. ``DeleteApplication`` takes the name in both APIs; a bare name (no
    ``/``) is returned unchanged.
    """
    return identifier.rsplit("/", 1)[-1]


def _is_v2_managed_error(error: ClientError) -> bool:
    """True when a v1 API call was rejected because the app is v2-SDK-managed."""
    err = error.response.get("Error", {})
    if err.get("Code", "") != _V2_MANAGED_CODE:
        return False
    return "kinesisanalyticsv2" in err.get("Message", "").lower()


def _delete_application(
    resource: Resource,
    session: boto3.Session,
    *,
    service: str,
    label: str,
    v2_managed_is_handled: bool = False,
) -> HandlerResult:
    """Delete a Kinesis Analytics (v1 or v2) application.

    **Ordering is load-bearing.** If the application is managed by a
    CloudFormation stack, the STACK is deleted and CloudFormation cascades — the
    application is never deleted natively. Deleting the app first orphans the
    stack's ``AWS::KinesisAnalyticsV2::ApplicationCloudWatchLoggingOption`` child
    (its delete needs the parent app to exist and raises
    ``ResourceNotFoundException``), sending the stack to ``DELETE_FAILED``.

    For an unmanaged application, ``DeleteApplication`` requires the
    ``CreateTimestamp`` from ``DescribeApplication``; both the v1
    (``kinesisanalytics``) and v2 (``kinesisanalyticsv2``) APIs share this shape.

    ``v2_managed_is_handled`` (v1 only): the scanner lists a v2 Studio application
    under both types, so a v1 call rejected with ``UnsupportedOperationException``
    ("use kinesisanalyticsv2 SDK") is the SAME app the v2 handler deletes — report
    SUCCESS (deduped) rather than a failure.
    """
    name = _application_name_from_identifier(resource.identifier)
    cfn_client = build_client(session, "cloudformation")
    stack_id = _managing_stack_id(cfn_client, name)
    if stack_id is not None:
        return _delete_managing_stack(resource, session, cfn_client, stack_id, name)
    client: BaseClient = build_client(session, service)
    try:
        detail = client.describe_application(ApplicationName=name)["ApplicationDetail"]
        client.delete_application(ApplicationName=name, CreateTimestamp=detail["CreateTimestamp"])
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in _NOT_FOUND_CODES:
            return HandlerResult(
                resource_id=resource.identifier,
                resource_type=resource.type,
                action="delete",
                status=HandlerStatus.SUCCESS,
                message=f"{label} application already gone",
            )
        if v2_managed_is_handled and _is_v2_managed_error(e):
            return HandlerResult(
                resource_id=resource.identifier,
                resource_type=resource.type,
                action="delete",
                status=HandlerStatus.SUCCESS,
                message="v2-SDK-managed application; deleted via the v2 handler",
            )
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Failed to delete {label} application '{name}': {e}",
        )
    except BotoCoreError as e:
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Connection error deleting {label} application '{name}': {e}",
        )
    logger.debug(f"Deleted {label} application '{name}'")
    # DeleteApplication returns success while the application stays visible to the
    # reset scanner's list_applications for minutes; wait for it to actually clear.
    _await_kda_teardown(resource, session)
    return HandlerResult(
        resource_id=resource.identifier,
        resource_type=resource.type,
        action="delete",
        status=HandlerStatus.SUCCESS,
    )


def _managing_stack_id(cfn_client: BaseClient, name: str) -> str | None:
    """Return the StackId of the CloudFormation stack managing this application.

    Detection is by stack MEMBERSHIP (``describe_stack_resources`` with the
    application name as ``PhysicalResourceId``), not by stack-name pattern —
    observed managing stacks carry agent-chosen names with no common prefix.
    Returns None for an unmanaged application or when membership cannot be
    determined (falls back to the native delete path).
    """
    try:
        resp = cfn_client.describe_stack_resources(PhysicalResourceId=name)
    except (ClientError, BotoCoreError):
        return None
    resources = resp.get("StackResources", []) if isinstance(resp, dict) else []
    for stack_resource in resources:
        if stack_resource.get("ResourceType") in _KDA_CFN_APP_TYPES:
            return stack_resource.get("StackId") or stack_resource.get("StackName")
    return None


def _delete_failed_logical_ids(cfn_client: BaseClient, stack_id: str) -> list[str]:
    """Logical IDs of the stack's resources currently in DELETE_FAILED."""
    try:
        resp = cfn_client.describe_stack_resources(StackName=stack_id)
    except (ClientError, BotoCoreError):
        return []
    return [
        r["LogicalResourceId"]
        for r in resp.get("StackResources", [])
        if r.get("ResourceStatus") == "DELETE_FAILED" and r.get("LogicalResourceId")
    ]


def _delete_managing_stack(
    resource: Resource,
    session: boto3.Session,
    cfn_client: BaseClient,
    stack_id: str,
    name: str,
) -> HandlerResult:
    """Delete a stack-managed KDA application by deleting its STACK (cascade).

    CloudFormation removes the ``ApplicationCloudWatchLoggingOption`` child while
    the application still exists, then the application itself — the ordering a
    native app-first delete breaks. If the stack still lands in ``DELETE_FAILED``
    (e.g. it was already broken by an earlier app-first delete), retry once with
    ``RetainResources`` for the stuck logical IDs — those children are virtual
    once the application is gone, so retaining them leaks nothing live (and the
    reset's re-verify re-detects anything that does still exist).
    """
    waiter = cfn_client.get_waiter("stack_delete_complete")
    try:
        cfn_client.delete_stack(StackName=stack_id)
        try:
            waiter.wait(StackName=stack_id, WaiterConfig=dict(_STACK_DELETE_WAITER_CONFIG))
        except WaiterError:
            retained = _delete_failed_logical_ids(cfn_client, stack_id)
            if not retained:
                raise
            logger.warning(
                "Stack '%s' hit DELETE_FAILED on %s; retrying with RetainResources",
                stack_id,
                retained,
            )
            cfn_client.delete_stack(StackName=stack_id, RetainResources=retained)
            waiter.wait(StackName=stack_id, WaiterConfig=dict(_STACK_DELETE_WAITER_CONFIG))
    except (ClientError, WaiterError, BotoCoreError) as e:
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Failed to delete managing stack '{stack_id}' for application '{name}': {e}",
        )
    logger.debug(f"Deleted managing stack '{stack_id}' for application '{name}'")
    # The cascaded application delete has the same async listing lag as a native
    # delete; wait for the scanner listing to clear before the reset re-verifies.
    _await_kda_teardown(resource, session)
    return HandlerResult(
        resource_id=resource.identifier,
        resource_type=resource.type,
        action="delete",
        status=HandlerStatus.SUCCESS,
        message=f"Deleted via managing CloudFormation stack '{stack_id}'",
    )


def _application_listed(client: BaseClient, id_field: str, name: str) -> bool:
    """True if ``name`` still appears in this service's ``list_applications`` output.

    Polls the SAME source the reset scanner reads (``collect`` over
    ``list_applications`` / ``ApplicationSummaries``), not ``DescribeApplication``
    — the control-plane describe reports the app gone immediately while the lister
    keeps returning it for minutes. ``id_field`` is ``ApplicationName`` (v2) or
    ``ApplicationARN`` (v1); both normalise to the bare name for comparison. A
    failed listing counts as "still present" so the poll keeps waiting rather than
    declaring a false absence.
    """
    try:
        listed = collect(client, "list_applications", "ApplicationSummaries", id_field)
    except (ClientError, BotoCoreError):
        return True
    return any(_application_name_from_identifier(str(app_id)) == name for app_id in listed)


def _stack_not_found(error: ClientError) -> bool:
    """True when describe_stacks reports the stack no longer exists."""
    err = error.response.get("Error", {})
    return err.get("Code", "") == "ValidationError" and "does not exist" in err.get("Message", "")


def _stack_absent(cfn_client: BaseClient, name: str) -> bool:
    """True once the service-managed stack is gone or fully DELETE_COMPLETE."""
    try:
        stacks = cfn_client.describe_stacks(StackName=name).get("Stacks", [])
    except ClientError as e:
        return _stack_not_found(e)
    except BotoCoreError:
        return False
    return all(s.get("StackStatus") in _TERMINAL_STACK_STATUSES for s in stacks)


def _await_kda_teardown(resource: Resource, session: boto3.Session) -> None:
    """Wait until a deleted KDA application clears the reset scanner's listing.

    ``kinesisanalyticsv2:DeleteApplication`` returns success while the application
    remains visible to the scanner's ``list_applications`` for minutes — the core
    residual, present in BOTH Studio task variants (stack-backed and stack-less).
    This wait is therefore UNCONDITIONAL after any real delete: poll the v2 AND v1
    ``list_applications`` (the app is listed under both types) until it clears from
    both. For the stack-backed variant (a ``environment-*-flink-studio`` app name),
    additionally wait for the service-managed backing stack to clear — no
    ``DeleteStack`` is issued (AWS tears it down as the application is removed; the
    transient ``DELETE_FAILED`` self-completes).

    Poll with capped exponential backoff over a grace window sized to comfortably
    exceed the observed removal latency. Best-effort: on timeout the residual is
    left to the reset's fail-closed re-verify.
    """
    name = _application_name_from_identifier(resource.identifier)
    v2_client = build_client(session, "kinesisanalyticsv2")
    v1_client = build_client(session, "kinesisanalytics")
    cfn_client = (
        build_client(session, "cloudformation") if is_service_managed_studio_stack(name) else None
    )
    v2_gone = v1_gone = stack_gone = False
    delay = _TEARDOWN_INITIAL_DELAY
    for _ in range(_TEARDOWN_MAX_ATTEMPTS):
        v2_gone = not _application_listed(v2_client, "ApplicationName", name)
        v1_gone = not _application_listed(v1_client, "ApplicationARN", name)
        stack_gone = cfn_client is None or _stack_absent(cfn_client, name)
        if v2_gone and v1_gone and stack_gone:
            logger.debug("Kinesis Analytics application '%s' cleared from the scanner", name)
            return
        time.sleep(delay)
        delay = min(delay * _TEARDOWN_BACKOFF, _TEARDOWN_MAX_DELAY)
    logger.warning(
        "Kinesis Analytics application '%s' still present after the grace window "
        "(v2_listed_gone=%s, v1_listed_gone=%s, stack_gone=%s); leaving it to the "
        "reset re-verify",
        name,
        v2_gone,
        v1_gone,
        stack_gone,
    )


@resource_handler("AWS::KinesisAnalyticsV2::Application", role="delete")
def _delete_v2(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete a Kinesis Data Analytics v2 (Managed Flink / Studio) application.

    The delete waits (in ``_delete_application``) for the application to clear the
    scanner's listing — and, for a stack-backed Studio app, for its service-managed
    backing stack — never issuing a direct ``DeleteStack``.
    """
    return _delete_application(
        resource, session, service="kinesisanalyticsv2", label="Kinesis Analytics v2"
    )


@resource_handler("AWS::KinesisAnalytics::Application", role="delete")
def _delete_v1(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete a Kinesis Data Analytics v1 application (not supported by CCAPI).

    A v2 Studio application is listed under this type too; the v1 API rejects it
    with ``UnsupportedOperationException``, which is treated as already handled by
    the v2 handler rather than a failure (no wait — the v2 handler's delete waits).
    """
    return _delete_application(
        resource,
        session,
        service="kinesisanalytics",
        label="Kinesis Analytics v1",
        v2_managed_is_handled=True,
    )
