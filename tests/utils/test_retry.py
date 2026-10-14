"""Retry policy tests; conftest's autouse ``_no_git_fetch_backoff`` skips git-fetch waits."""

import subprocess

import pytest
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.utils.retry import (
    is_fresh_account_transient,
    is_region_access_transient,
    is_scp_access_denied,
    retrying_git_fetch,
)


def _client_error(code: str) -> ClientError:
    """Build a botocore ClientError carrying ``code`` in Error.Code."""
    return ClientError(
        {"Error": {"Code": code, "Message": "needs a subscription for the service"}},
        "ListStacks",
    )


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_client_error("OptInRequired"), True),
        (_client_error("SubscriptionRequiredException"), True),
        # Fresh account whose STS/IAM identity has not propagated yet.
        (_client_error("InvalidClientTokenId"), True),
        # AccessDenied and NotFound are ambiguous — must NOT auto-retry.
        (_client_error("AccessDenied"), False),
        (_client_error("ResourceNotFoundException"), False),
        (_client_error("ValidationError"), False),
        (BotoCoreError(), False),
        (RuntimeError("boom"), False),
    ],
)
def test_is_fresh_account_transient_classifies_fresh_account_codes(exc, expected):
    """Subscription-convergence codes and the fresh-account STS-propagation error retry."""
    assert is_fresh_account_transient(exc) is expected


@pytest.mark.asyncio
async def test_returns_result_without_retry_on_success():
    """A thunk that succeeds first try runs exactly once and returns its value."""
    calls = {"n": 0}

    async def fetch():
        calls["n"] += 1
        return "downloaded"

    assert await retrying_git_fetch(fetch) == "downloaded"
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_retries_called_process_error_then_succeeds():
    """Transient CalledProcessError is retried; a later success is returned."""
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise subprocess.CalledProcessError(1, ["git"], b"", b"remote hung up")
        return "ok"

    assert await retrying_git_fetch(flaky) == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_exhausts_attempts_and_reraises_original():
    """A persistent CalledProcessError retries up to the cap, then reraises the original.

    ``reraise=True`` surfaces the underlying git error (with its stderr) rather
    than a tenacity RetryError, so each caller's typed-error wrapping still sees
    the real failure.
    """
    calls = {"n": 0}

    async def always_fail():
        calls["n"] += 1
        raise subprocess.CalledProcessError(1, ["git"], b"", b"persistent throttle")

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        await retrying_git_fetch(always_fail)
    # The ORIGINAL error is reraised (stderr preserved), not a tenacity RetryError.
    assert exc_info.value.stderr == b"persistent throttle"
    # 5 attempts total (the configured stop_after_attempt cap).
    assert calls["n"] == 5


@pytest.mark.asyncio
async def test_does_not_retry_other_exceptions():
    """Non-CalledProcessError failures fall straight through (fail-fast preserved)."""
    calls = {"n": 0}

    async def hard_fail():
        calls["n"] += 1
        raise FileNotFoundError("missing local path")

    with pytest.raises(FileNotFoundError):
        await retrying_git_fetch(hard_fail)
    assert calls["n"] == 1


@pytest.mark.parametrize("code", ["AccessDenied", "AccessDeniedException", "UnauthorizedOperation"])
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("with an explicit deny in a service control policy", True),
        ("Denied by a SERVICE CONTROL POLICY", True),
        ("no identity-based policy allows this action", False),
        ("explicit deny in a permissions boundary", False),
        ("", False),
    ],
)
def test_scp_access_denied_is_precise(code: str, message: str, expected: bool) -> None:
    exc = ClientError({"Error": {"Code": code, "Message": message}}, "ListStacks")
    assert is_scp_access_denied(exc) is expected
    assert is_region_access_transient(exc) is expected
    assert not is_fresh_account_transient(exc)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_client_error("OptInRequired"), True),
        (_client_error("SubscriptionRequiredException"), True),
        (_client_error("InvalidClientTokenId"), True),
        (_client_error("AccessDenied"), False),
        (ClientError({"Error": {"Code": "AccessDenied"}}, "ListStacks"), False),
        (
            ClientError(
                {"Error": {"Code": "ValidationError", "Message": "service control policy"}},
                "ListStacks",
            ),
            False,
        ),
        (BotoCoreError(), False),
        (RuntimeError("service control policy"), False),
    ],
)
def test_region_access_transient_classifier(exc: BaseException, expected: bool) -> None:
    assert is_region_access_transient(exc) is expected
    assert not is_scp_access_denied(exc)
