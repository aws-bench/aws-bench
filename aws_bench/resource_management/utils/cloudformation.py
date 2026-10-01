"""CloudFormation utility functions."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from botocore.client import BaseClient
from botocore.exceptions import ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.exceptions import PollTimeout
from aws_bench.utils.concurrent import raise_if_shutdown

logger = get_logger(__name__)

_DEFAULT_TIMEOUT_SEC = 300
_DEFAULT_POLL_INTERVAL_SEC = 5

DRIFT_DETECTION_COMPLETE = "DETECTION_COMPLETE"
DRIFT_DETECTION_FAILED = "DETECTION_FAILED"

STACK_UPDATE_TIMEOUT_SEC = 1200  # was the stack_update_complete waiter: Delay=30 × MaxAttempts=40
CHANGESET_TIMEOUT_SEC = 300  # was change_set_create_complete: Delay=5 × MaxAttempts=60


def is_stack_settled(status: str) -> bool:
    """A stack operation is terminal once its status stops being in-progress."""
    return not status.endswith("_IN_PROGRESS")


def poll_for_terminal_status(
    read_status: Callable[[], str],
    *,
    is_terminal: Callable[[str], bool],
    description: str,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    interval_sec: float = _DEFAULT_POLL_INTERVAL_SEC,
) -> str:
    """Poll ``read_status`` until ``is_terminal`` accepts it; return that status.

    The cancellable replacement for ``waiter.wait()``: botocore waiters block in
    one uninterruptible call, so on a worker thread (under ``asyncio.to_thread``)
    a shutdown signal can't reach them. This polls the ``Describe*`` API directly
    and checks the cooperative flag between attempts.

    ``read_status`` returns the current status string; ``is_terminal`` decides
    when to stop. Raises ``PollTimeout`` if no terminal status is reached within
    ``timeout_sec``.
    """
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        raise_if_shutdown()
        status = read_status()
        if is_terminal(status):
            return status
        raise_if_shutdown()
        time.sleep(interval_sec)
    raise PollTimeout(f"{description} did not reach a terminal status within {timeout_sec:.0f}s")


def wait_for_drift_detection(
    cfn_client: BaseClient,
    detection_id: str,
    *,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    interval_sec: float = _DEFAULT_POLL_INTERVAL_SEC,
) -> str:
    """Poll drift detection to a terminal status; return ``DETECTION_COMPLETE``/``_FAILED``.

    Raises ``PollTimeout`` if neither is reached within ``timeout_sec``.
    """
    return poll_for_terminal_status(
        lambda: cfn_client.describe_stack_drift_detection_status(
            StackDriftDetectionId=detection_id
        )["DetectionStatus"],
        is_terminal=lambda s: s in (DRIFT_DETECTION_COMPLETE, DRIFT_DETECTION_FAILED),
        description=f"drift detection {detection_id}",
        timeout_sec=timeout_sec,
        interval_sec=interval_sec,
    )


def wait_for_stack_update(
    cfn_client: BaseClient,
    stack_name: str,
    *,
    timeout_sec: float = STACK_UPDATE_TIMEOUT_SEC,
) -> str:
    """Poll a stack update to a terminal status; return that ``StackStatus``.

    Success is ``UPDATE_COMPLETE``; any other terminal status is a failure.
    Raises ``PollTimeout`` if the stack stays in-progress past ``timeout_sec``.
    """
    return poll_for_terminal_status(
        lambda: cfn_client.describe_stacks(StackName=stack_name)["Stacks"][0]["StackStatus"],
        is_terminal=is_stack_settled,
        description=f"stack {stack_name} update",
        timeout_sec=timeout_sec,
        interval_sec=30,
    )


def wait_for_changeset_creation(
    cfn_client: BaseClient,
    stack_name: str,
    change_set_name: str,
    *,
    timeout_sec: float = CHANGESET_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Poll changeset creation to a terminal status; return the final DescribeChangeSet response.

    The caller inspects ``Status``/``StatusReason`` — CloudFormation reports an
    empty changeset as ``FAILED`` with a telltale reason, which only the caller
    can interpret. Raises ``PollTimeout`` if creation doesn't settle in time.
    """
    result: dict[str, Any] = {}

    def read_status() -> str:
        nonlocal result
        result = cfn_client.describe_change_set(ChangeSetName=change_set_name, StackName=stack_name)
        return result["Status"]

    poll_for_terminal_status(
        read_status,
        is_terminal=lambda s: s in ("CREATE_COMPLETE", "FAILED"),
        description=f"change set {change_set_name}",
        timeout_sec=timeout_sec,
    )
    return result


def is_stack_not_found(exc: ClientError) -> bool:
    """Check if exception indicates stack does not exist.

    CloudFormation returns ValidationError for multiple scenarios. We check both
    the error code and the message to distinguish "stack not found" from other
    ValidationError cases (e.g., ARN validation failures).

    Note: This applies to read operations (describe_stacks, get_template, etc.).
    delete_stack is idempotent and does NOT raise an error for missing stacks.

    Args:
        exc: The ClientError exception to check

    Returns:
        True if the error indicates the stack was not found

    Examples:
        Stack not found (read operations):
            Error Code: ValidationError
            Message: "Stack with id my-stack does not exist"
            Returns: True

        ARN validation error:
            Error Code: ValidationError
            Message: "Resource account doesn't match"
            Returns: False

        delete_stack on missing stack:
            No exception raised (idempotent operation)
    """
    error_code = exc.response.get("Error", {}).get("Code", "")
    return error_code == "ValidationError" and "does not exist" in str(exc).lower()


def disable_termination_protection(cfn_client: BaseClient, stack_name: str) -> bool:
    """Turn termination protection off on ``stack_name`` if it is on.

    CloudFormation refuses ``DeleteStack`` with a ``ValidationError`` while a
    stack has termination protection enabled, and agents do create stacks with
    ``EnableTerminationProtection`` set. Every deletion path (``env cleanup``
    and the reset's native ``DeleteStack`` fallback) runs this first.

    Errors are logged and swallowed: the caller's ``DeleteStack`` surfaces the
    real failure.

    Args:
        cfn_client: CloudFormation client
        stack_name: Stack name or ARN

    Returns:
        True if protection was on and has been turned off, False otherwise
    """
    try:
        resp = cfn_client.describe_stacks(StackName=stack_name)
        if not resp["Stacks"][0].get("EnableTerminationProtection", False):
            return False
        cfn_client.update_termination_protection(
            EnableTerminationProtection=False, StackName=stack_name
        )
        logger.debug("Disabled termination protection on '%s'.", stack_name)
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("AccessDenied", "UnauthorizedOperation", "InvalidClientTokenId"):
            logger.error(
                "Permission denied when disabling termination protection on '%s': %s",
                stack_name,
                error_code,
            )
        elif is_stack_not_found(e):
            logger.debug("Stack '%s' not found when disabling termination protection", stack_name)
        else:
            logger.debug("Could not disable termination protection on '%s': %s", stack_name, e)
        return False


def get_stack_resource_drifts(cfn_client: BaseClient, stack_name: str) -> list[dict[str, Any]]:
    """Get all resource drifts for a stack, handling pagination manually.

    The describe_stack_resource_drifts operation does not support boto3's
    automatic pagination, so we must paginate manually using NextToken.

    Args:
        cfn_client: CloudFormation client
        stack_name: Name of the stack

    Returns:
        List of all resource drifts across all pages
    """
    drifts: list[dict[str, Any]] = []
    next_token: str | None = None

    while True:
        kwargs: dict[str, Any] = {"StackName": stack_name}
        if next_token:
            kwargs["NextToken"] = next_token

        response = cfn_client.describe_stack_resource_drifts(**kwargs)
        drifts.extend(response.get("StackResourceDrifts", []))

        next_token = response.get("NextToken")
        if not next_token:
            break

    return drifts
