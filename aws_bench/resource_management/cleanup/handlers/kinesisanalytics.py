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
- The application's backing ``environment-*-flink-studio`` stack is
  **service-managed**: AWS removes both the application and the stack
  **asynchronously** (over several minutes) once the application is deleted, and
  a direct ``DeleteStack`` goes terminal ``DELETE_FAILED``. After deleting the v2
  application the handler polls for the true absence of both — it never issues a
  ``DeleteStack`` for the service-managed stack.

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
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.models import (
    HandlerResult,
    HandlerStatus,
    is_service_managed_studio_stack,
)
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)

_NOT_FOUND_CODES = ("ResourceNotFoundException",)
# Raised by the v1 API for an application the v2 SDK created/manages.
_V2_MANAGED_CODE = "UnsupportedOperationException"
# The KDA Studio application AND its service-managed backing stack are torn down
# ASYNCHRONOUSLY after DeleteApplication — the stack may transiently show
# DELETE_FAILED but self-completes, so no direct DeleteStack is issued. Poll for
# the true absence of both, backing off up to a grace window sized to comfortably
# exceed the (minutes-scale) teardown latency. The window intentionally does NOT
# target the multi-hour horizon seen when a framework re-issues the delete late;
# tune _TEARDOWN_MAX_ATTEMPTS if the measured pure latency ever approaches it.
_TEARDOWN_INITIAL_DELAY = 10
_TEARDOWN_MAX_DELAY = 60
_TEARDOWN_BACKOFF = 2.0
_TEARDOWN_MAX_ATTEMPTS = 18  # ~16 min total with the backoff above
_TERMINAL_STACK_STATUSES = ("DELETE_COMPLETE",)


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

    ``DeleteApplication`` requires the ``CreateTimestamp`` from
    ``DescribeApplication``; both the v1 (``kinesisanalytics``) and v2
    (``kinesisanalyticsv2``) APIs share this shape.

    ``v2_managed_is_handled`` (v1 only): the scanner lists a v2 Studio application
    under both types, so a v1 call rejected with ``UnsupportedOperationException``
    ("use kinesisanalyticsv2 SDK") is the SAME app the v2 handler deletes — report
    SUCCESS (deduped) rather than a failure.
    """
    name = _application_name_from_identifier(resource.identifier)
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
    return HandlerResult(
        resource_id=resource.identifier,
        resource_type=resource.type,
        action="delete",
        status=HandlerStatus.SUCCESS,
    )


def _application_absent(kda_client: BaseClient, name: str) -> bool:
    """True once the v2 application no longer exists (its async delete completed)."""
    try:
        kda_client.describe_application(ApplicationName=name)
    except ClientError as e:
        return e.response.get("Error", {}).get("Code", "") in _NOT_FOUND_CODES
    except BotoCoreError:
        return False
    return False


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


def _await_studio_teardown(resource: Resource, session: boto3.Session) -> None:
    """Wait for a deleted Studio application AND its service-managed stack to vanish.

    ``DeleteApplication`` on a Managed Flink "Studio" application triggers AWS to
    tear down both the application and its ``environment-*-flink-studio`` backing
    stack **asynchronously**, over several minutes. The stack may transiently show
    ``DELETE_FAILED`` but self-completes, so no ``DeleteStack`` is issued here — a
    direct delete only races the service. Poll (with backoff) for the true absence
    of both so the reset's re-verify — which lists both the application and the
    stack — does not fail-close before the async teardown finishes.

    Best-effort: on timeout the residual is left to the reset's fail-closed
    re-verify. Non-Studio applications have no managed stack and are skipped.
    """
    name = _application_name_from_identifier(resource.identifier)
    if not is_service_managed_studio_stack(name):
        return
    kda_client = build_client(session, "kinesisanalyticsv2")
    cfn_client = build_client(session, "cloudformation")
    app_gone = stack_gone = False
    delay = _TEARDOWN_INITIAL_DELAY
    for _ in range(_TEARDOWN_MAX_ATTEMPTS):
        app_gone = _application_absent(kda_client, name)
        stack_gone = _stack_absent(cfn_client, name)
        if app_gone and stack_gone:
            logger.debug("Kinesis Analytics Studio teardown complete for '%s'", name)
            return
        time.sleep(delay)
        delay = min(delay * _TEARDOWN_BACKOFF, _TEARDOWN_MAX_DELAY)
    logger.warning(
        "Kinesis Analytics Studio '%s' not fully torn down within the grace window "
        "(application_gone=%s, stack_gone=%s); leaving it to the reset re-verify",
        name,
        app_gone,
        stack_gone,
    )


@resource_handler("AWS::KinesisAnalyticsV2::Application", role="delete")
def _delete_v2(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete a Kinesis Data Analytics v2 (Managed Flink / Studio) application.

    After the application is deleted, wait for AWS to asynchronously tear down the
    application and its service-managed ``environment-*-flink-studio`` backing
    stack (if any) — never issuing a direct ``DeleteStack``.
    """
    result = _delete_application(
        resource, session, service="kinesisanalyticsv2", label="Kinesis Analytics v2"
    )
    if result.status is HandlerStatus.SUCCESS:
        _await_studio_teardown(resource, session)
    return result


@resource_handler("AWS::KinesisAnalytics::Application", role="delete")
def _delete_v1(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete a Kinesis Data Analytics v1 application (not supported by CCAPI).

    A v2 Studio application is listed under this type too; the v1 API rejects it
    with ``UnsupportedOperationException``, which is treated as already handled by
    the v2 handler rather than a failure.
    """
    return _delete_application(
        resource,
        session,
        service="kinesisanalytics",
        label="Kinesis Analytics v1",
        v2_managed_is_handled=True,
    )
