"""Shared retry policies and transient-error classifiers."""

from __future__ import annotations

import subprocess
from collections.abc import Awaitable, Callable
from typing import TypeVar

import tenacity
from botocore.exceptions import ClientError

from aws_bench.account_management.constants import FRESH_ACCOUNT_TRANSIENT_CODES

_T = TypeVar("_T")


def is_fresh_account_transient(exc: BaseException) -> bool:
    """True when ``exc`` is a not-yet-converged subscription error on a new account.

    The code set lives in ``account_management.constants`` (a dep-free leaf), shared with the
    in-Lambda engine sweep which cannot import this tenacity-backed module.
    """
    return (
        isinstance(exc, ClientError)
        and exc.response.get("Error", {}).get("Code") in FRESH_ACCOUNT_TRANSIENT_CODES
    )


def is_scp_access_denied(exc: BaseException) -> bool:
    """Identify explicit SCP rejections, not ordinary IAM access denials."""
    if not isinstance(exc, ClientError):
        return False
    error = exc.response.get("Error", {})
    return (
        error.get("Code")
        in {
            "AccessDenied",
            "AccessDeniedException",
            "UnauthorizedOperation",
        }
        and "service control policy" in error.get("Message", "").lower()
    )


def is_region_access_transient(exc: BaseException) -> bool:
    """Identify subscription, regional authentication, or SCP propagation failures."""
    return (
        is_fresh_account_transient(exc)
        or is_scp_access_denied(exc)
        or (
            isinstance(exc, ClientError)
            and exc.response.get("Error", {}).get("Code") == "AuthFailure"
        )
    )


@tenacity.retry(
    stop=tenacity.stop_after_delay(180),
    wait=tenacity.wait_exponential(min=5, max=30) + tenacity.wait_random(0, 5),
    retry=tenacity.retry_if_exception(is_region_access_transient),
    reraise=True,
)
def retrying_region_read(operation: Callable[[], _T]) -> _T:
    """Retry a read-only regional probe while access converges; never use for writes."""
    return operation()


@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(min=5, max=60) + tenacity.wait_random(0, 10),
    retry=tenacity.retry_if_exception_type(subprocess.CalledProcessError),
    reraise=True,
)
async def retrying_git_fetch(fetch: Callable[[], Awaitable[_T]]) -> _T:
    """Run a git fetch thunk, retrying transient failures; reraise the original on exhaustion."""
    return await fetch()
