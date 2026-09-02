"""Tests for aws_bench.utils.credentials_provider."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import boto3
import pytest

from aws_bench.utils.credentials_provider import (
    MAX_SESSION_NAME_LEN,
    SESSION_NAME_PREFIX,
    CredentialProvider,
    _create_refreshable_session,
    build_aws_credentials_file,
    build_session_name,
    create_regional_session,
    enforce_session_name,
    session_to_env_credentials,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the CredentialProvider singleton between tests."""
    CredentialProvider.reset()
    yield
    CredentialProvider.reset()


# ── singleton ──


def test_get_returns_same_instance():
    """Returns the same instance on repeated calls."""
    a = CredentialProvider.get()
    b = CredentialProvider.get()
    assert a is b


def test_reset_clears_instance():
    """Reset creates a fresh instance on next get call."""
    a = CredentialProvider.get()
    CredentialProvider.reset()
    b = CredentialProvider.get()
    assert a is not b


def test_get_uses_provided_session():
    """Uses the provided boto3 session on first call."""
    mock_session = MagicMock()
    provider = CredentialProvider.get(session=mock_session)
    assert provider.session is mock_session


# ── get_caller_account_id ──


def test_get_caller_account_id_returns_account_id():
    """Returns the account ID from STS get_caller_identity."""
    mock_session = MagicMock()
    mock_session.client.return_value.get_caller_identity.return_value = {"Account": "123456789012"}
    provider = CredentialProvider(session=mock_session)
    assert provider.get_caller_account_id() == "123456789012"


def test_get_caller_account_id_caches():
    """Caches the account ID after the first call."""
    mock_session = MagicMock()
    mock_sts = mock_session.client.return_value
    mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

    provider = CredentialProvider(session=mock_session)
    provider.get_caller_account_id()
    provider.get_caller_account_id()
    mock_sts.get_caller_identity.assert_called_once()


def test_sts_property_creates_fresh_client_each_time():
    """The _sts property creates a fresh STS client on each access."""
    mock_session = MagicMock()
    mock_sts_1 = MagicMock()
    mock_sts_2 = MagicMock()
    mock_session.client.side_effect = [mock_sts_1, mock_sts_2]

    provider = CredentialProvider(session=mock_session)

    # Access _sts property twice
    sts1 = provider._sts
    sts2 = provider._sts

    # Should have created two STS clients
    assert mock_session.client.call_count == 2
    assert sts1 is mock_sts_1
    assert sts2 is mock_sts_2


# ── assume_role ──


def test_assume_role_returns_credentials():
    """Returns credentials dict from STS assume_role response."""
    mock_session = MagicMock()
    mock_sts = mock_session.client.return_value
    mock_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "AKIA_TEST",
            "SecretAccessKey": "SECRET_TEST",
            "SessionToken": "TOKEN_TEST",
        }
    }
    provider = CredentialProvider(session=mock_session)
    creds = provider.assume_role("111111111111", "TestRole", "app-test-session")

    assert creds["AWS_ACCESS_KEY_ID"] == "AKIA_TEST"
    assert creds["AWS_SECRET_ACCESS_KEY"] == "SECRET_TEST"
    assert creds["AWS_SESSION_TOKEN"] == "TOKEN_TEST"


def test_assume_role_truncates_session_name_to_64_chars():
    """Truncates session name to 64 characters for AWS limit."""
    mock_session = MagicMock()
    mock_sts = mock_session.client.return_value
    mock_sts.assume_role.return_value = {
        "Credentials": {"AccessKeyId": "AK", "SecretAccessKey": "SK", "SessionToken": "ST"}
    }
    provider = CredentialProvider(session=mock_session)
    provider.assume_role("111111111111", "TestRole", "app-" + "a" * 100)

    call_kwargs = mock_sts.assume_role.call_args[1]
    assert len(call_kwargs["RoleSessionName"]) == 64


# ── _create_refreshable_session ──


def test_create_refreshable_session_returns_session():
    """Creates a boto3 session with refreshable credentials."""
    mock_parent_session = MagicMock()
    mock_sts = MagicMock()
    mock_parent_session.client.return_value = mock_sts
    future_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    mock_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "AK_TEST",
            "SecretAccessKey": "SK_TEST",
            "SessionToken": "ST_TEST",
            "Expiration": future_expiry,
        }
    }

    session = _create_refreshable_session(
        mock_parent_session, "arn:aws:iam::123456789012:role/TestRole", "app-test-session"
    )

    assert isinstance(session, boto3.Session)
    assert session.region_name == "us-east-1"


def test_create_refreshable_session_uses_correct_params():
    """Uses correct parameters for assume_role calls."""
    mock_parent_session = MagicMock()
    mock_sts = MagicMock()
    mock_parent_session.client.return_value = mock_sts
    future_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    mock_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "AK_TEST",
            "SecretAccessKey": "SK_TEST",
            "SessionToken": "ST_TEST",
            "Expiration": future_expiry,
        }
    }

    session = _create_refreshable_session(
        mock_parent_session,
        "arn:aws:iam::123456789012:role/TestRole",
        "app-test-session-name",
    )

    # Trigger credential fetch by accessing credentials
    creds = session.get_credentials()
    creds.get_frozen_credentials()

    # Verify assume_role was called with correct parameters
    mock_sts.assume_role.assert_called_once()
    call_kwargs = mock_sts.assume_role.call_args.kwargs
    assert call_kwargs["RoleArn"] == "arn:aws:iam::123456789012:role/TestRole"
    assert call_kwargs["RoleSessionName"] == "app-test-session-name"


def test_create_refreshable_session_creates_fresh_sts_client_on_each_refresh():
    """Every refresh creates a new STS client from the parent session.

    This is intentional defense in depth — re-creating the STS client per
    refresh forces boto3 to re-resolve parent credentials, avoiding any
    staleness in the captured client across long-running operations.
    """
    mock_parent_session = MagicMock()
    call_count = {"n": 0}

    def make_sts(_service):
        call_count["n"] += 1
        sts = MagicMock()
        future_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": f"AK_{call_count['n']}",
                "SecretAccessKey": f"SK_{call_count['n']}",
                "SessionToken": f"ST_{call_count['n']}",
                "Expiration": future_expiry,
            }
        }
        return sts

    mock_parent_session.client.side_effect = make_sts

    session = _create_refreshable_session(
        mock_parent_session, "arn:aws:iam::123456789012:role/TestRole", "app-test-session"
    )

    # First credential fetch creates first STS client
    creds = session.get_credentials()
    frozen1 = creds.get_frozen_credentials()
    assert frozen1.access_key == "AK_1"
    assert call_count["n"] == 1

    # Trigger refresh - should create a NEW STS client
    refresh_func = getattr(creds, "_refresh_using", None)
    assert refresh_func is not None
    refreshed = refresh_func()
    assert refreshed["access_key"] == "AK_2"
    assert call_count["n"] == 2


# ── get_session_for_account ──


@patch("aws_bench.utils.credentials_provider._create_refreshable_session")
def test_get_session_for_account_returns_session(mock_create_session):
    """Returns a boto3 Session with refreshable credentials."""
    mock_session = MagicMock()
    mock_created_session = MagicMock()
    mock_create_session.return_value = mock_created_session

    provider = CredentialProvider(session=mock_session)
    result = provider.get_session_for_account("111111111111", "TestRole", "app-sess")

    assert result is mock_created_session
    mock_create_session.assert_called_once_with(
        mock_session, "arn:aws:iam::111111111111:role/TestRole", "app-sess", "us-east-1"
    )


# ── chain_assume_role ──


@patch("aws_bench.utils.credentials_provider.boto3.Session")
def test_chain_assume_role_without_role_returns_org_creds(mock_session_cls):
    """Without role_name, returns org access role credentials (single hop)."""
    mock_session = MagicMock()
    mock_sts = mock_session.client.return_value
    mock_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "ORG_AK",
            "SecretAccessKey": "ORG_SK",
            "SessionToken": "ORG_ST",
        }
    }
    provider = CredentialProvider(session=mock_session)
    creds = provider.chain_assume_role(account_id="111111111111", session_name="app-sess")

    assert creds["AWS_ACCESS_KEY_ID"] == "ORG_AK"
    mock_sts.assume_role.assert_called_once()  # only hop 1


@patch("aws_bench.utils.credentials_provider.boto3.Session")
def test_chain_assume_role_with_org_role_skips_second_hop(mock_session_cls):
    """Passing ORG_ACCESS_ROLE explicitly still does a single hop."""
    from aws_bench.account_management.constants import ORG_ACCESS_ROLE

    mock_session = MagicMock()
    mock_sts = mock_session.client.return_value
    mock_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "ORG_AK",
            "SecretAccessKey": "ORG_SK",
            "SessionToken": "ORG_ST",
        }
    }
    provider = CredentialProvider(session=mock_session)
    creds = provider.chain_assume_role(
        account_id="111111111111", session_name="app-sess", role_name=ORG_ACCESS_ROLE
    )

    assert creds["AWS_ACCESS_KEY_ID"] == "ORG_AK"
    mock_sts.assume_role.assert_called_once()  # only hop 1
    assert mock_sts.assume_role.call_args.kwargs["RoleSessionName"] == "app-sess"


@patch("aws_bench.utils.credentials_provider.boto3.Session")
def test_chain_assume_role_with_role_performs_two_hops(mock_session_cls):
    """With role_name, chains through org role then assumes the target role."""
    mock_session = MagicMock()
    mock_sts = mock_session.client.return_value
    mock_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "ORG_AK",
            "SecretAccessKey": "ORG_SK",
            "SessionToken": "ORG_ST",
        }
    }

    mock_member_sts = MagicMock()
    mock_member_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "TASK_AK",
            "SecretAccessKey": "TASK_SK",
            "SessionToken": "TASK_ST",
        }
    }
    mock_session_cls.return_value.client.return_value = mock_member_sts

    provider = CredentialProvider(session=mock_session)
    creds = provider.chain_assume_role(
        account_id="111111111111", session_name="app-sess", role_name="TaskRole"
    )

    assert creds["AWS_ACCESS_KEY_ID"] == "TASK_AK"
    # Hop 1: org role via the provider's own STS
    mock_sts.assume_role.assert_called_once()
    # Hop 2: target role via the member session STS
    mock_member_sts.assume_role.assert_called_once()
    assert "TaskRole" in mock_member_sts.assume_role.call_args[1]["RoleArn"]


@patch("aws_bench.utils.credentials_provider.boto3.Session")
def test_chain_assume_role_first_hop_failure(mock_session_cls):
    """First hop failure is propagated with error logging."""
    from botocore.exceptions import ClientError

    mock_session = MagicMock()
    mock_sts = mock_session.client.return_value
    error = ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "AssumeRole")
    mock_sts.assume_role.side_effect = error

    provider = CredentialProvider(session=mock_session)

    with pytest.raises(ClientError):
        provider.chain_assume_role(
            account_id="111111111111", session_name="app-sess", role_name="TaskRole"
        )


@patch("aws_bench.utils.credentials_provider.boto3.Session")
def test_chain_assume_role_second_hop_failure(mock_session_cls):
    """Second hop failure is propagated with error logging."""
    from botocore.exceptions import ClientError

    mock_session = MagicMock()
    mock_sts = mock_session.client.return_value
    # First hop succeeds
    mock_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "ORG_AK",
            "SecretAccessKey": "ORG_SK",
            "SessionToken": "ORG_ST",
        }
    }

    # Second hop fails
    mock_member_sts = MagicMock()
    error = ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "AssumeRole")
    mock_member_sts.assume_role.side_effect = error
    mock_session_cls.return_value.client.return_value = mock_member_sts

    provider = CredentialProvider(session=mock_session)

    with pytest.raises(ClientError):
        provider.chain_assume_role(
            account_id="111111111111", session_name="app-sess", role_name="TaskRole"
        )


# ── create_regional_session ──


def test_create_regional_session_shares_credentials():
    """Regional session shares the same credential provider."""
    mock_parent_session = MagicMock()
    mock_sts = MagicMock()
    mock_parent_session.client.return_value = mock_sts
    future_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    mock_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "AK_TEST",
            "SecretAccessKey": "SK_TEST",
            "SessionToken": "ST_TEST",
            "Expiration": future_expiry,
        }
    }

    # Create parent session with refreshable credentials
    parent_session = _create_refreshable_session(
        mock_parent_session, "arn:aws:iam::123456789012:role/TestRole", "app-test-session"
    )

    # Create regional session
    regional_session = create_regional_session(parent_session, "us-west-2")

    # Both sessions should share the same credential provider
    assert regional_session._session._credentials is parent_session._session._credentials
    assert regional_session.region_name == "us-west-2"


def test_create_regional_session_preserves_refreshable_credentials():
    """Regional session credentials remain cached like parent."""
    mock_parent_session = MagicMock()
    mock_sts = MagicMock()
    mock_parent_session.client.return_value = mock_sts
    future_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    mock_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "AK_TEST",
            "SecretAccessKey": "SK_TEST",
            "SessionToken": "ST_TEST",
            "Expiration": future_expiry,
        }
    }

    # Create parent session
    parent_session = _create_refreshable_session(
        mock_parent_session, "arn:aws:iam::123456789012:role/TestRole", "app-test-session"
    )

    # Get credentials from parent
    parent_creds = parent_session.get_credentials().get_frozen_credentials()

    # Create regional session
    regional_session = create_regional_session(parent_session, "us-west-2")

    # Get credentials from regional session
    regional_creds = regional_session.get_credentials().get_frozen_credentials()

    # Should be the same credentials (cached, not refreshed)
    assert regional_creds.access_key == parent_creds.access_key
    assert regional_creds.secret_key == parent_creds.secret_key
    assert regional_creds.token == parent_creds.token


# ── session_to_env_credentials ──


def test_session_to_env_credentials_returns_env_dict():
    """session_to_env_credentials returns all three env-var keys from frozen creds."""
    frozen = MagicMock()
    frozen.access_key = "AKIA-FOO"
    frozen.secret_key = "bar"
    frozen.token = "baz"

    creds = MagicMock()
    creds.get_frozen_credentials.return_value = frozen

    session = MagicMock()
    session.get_credentials.return_value = creds

    result = session_to_env_credentials(session)

    assert result == {
        "AWS_ACCESS_KEY_ID": "AKIA-FOO",
        "AWS_SECRET_ACCESS_KEY": "bar",
        "AWS_SESSION_TOKEN": "baz",
    }


def test_session_to_env_credentials_empty_token_defaults_to_blank():
    """A None session token becomes an empty string to satisfy downstream dict usage."""
    frozen = MagicMock()
    frozen.access_key = "AKIA-FOO"
    frozen.secret_key = "bar"
    frozen.token = None

    creds = MagicMock()
    creds.get_frozen_credentials.return_value = frozen

    session = MagicMock()
    session.get_credentials.return_value = creds

    result = session_to_env_credentials(session)
    assert result["AWS_SESSION_TOKEN"] == ""


def test_session_to_env_credentials_raises_credential_error_when_no_credentials():
    """session_to_env_credentials raises CredentialError when session has none."""
    from aws_bench.exceptions import CredentialError

    session = MagicMock()
    session.get_credentials.return_value = None

    with pytest.raises(CredentialError, match="no credentials"):
        session_to_env_credentials(session)


def test_session_to_env_credentials_forces_refresh_for_refreshable_creds():
    """Refreshable creds are re-minted (mandatory refresh) before freezing.

    The snapshot is static, so the recipient must start with a full-duration
    credential rather than the parent's remaining lifetime.
    """
    frozen = MagicMock(access_key="AKIA-FRESH", secret_key="s", token="t")
    # spec= limits the mock to real RefreshableCredentials attributes, so
    # _protected_refresh exists (and is asserted) while absent ones would not.
    from botocore.credentials import RefreshableCredentials

    creds = MagicMock(spec=RefreshableCredentials)
    creds.get_frozen_credentials.return_value = frozen

    session = MagicMock()
    session.get_credentials.return_value = creds

    result = session_to_env_credentials(session)

    creds._protected_refresh.assert_called_once_with(is_mandatory=True)
    assert result["AWS_ACCESS_KEY_ID"] == "AKIA-FRESH"


def test_session_to_env_credentials_skips_refresh_for_static_creds():
    """Static creds (no _protected_refresh) are frozen as-is, no refresh attempted."""
    from botocore.credentials import Credentials

    frozen = MagicMock(access_key="AKIA-STATIC", secret_key="s", token=None)
    # A plain Credentials object has no _protected_refresh — must pass through.
    creds = MagicMock(spec=Credentials)
    creds.get_frozen_credentials.return_value = frozen
    assert not hasattr(creds, "_protected_refresh")

    session = MagicMock()
    session.get_credentials.return_value = creds

    result = session_to_env_credentials(session)
    assert result["AWS_ACCESS_KEY_ID"] == "AKIA-STATIC"
    assert result["AWS_SESSION_TOKEN"] == ""


# ── session_to_credential_process ──


def test_session_to_credential_process_emits_process_json():
    """Returns the credential_process shape: Version, keys, and RFC3339 Expiration."""
    from datetime import datetime, timezone

    from aws_bench.utils.credentials_provider import session_to_credential_process

    frozen = MagicMock(access_key="AKIA-P", secret_key="sk", token="tok")
    creds = MagicMock()
    creds.get_frozen_credentials.return_value = frozen
    creds._expiry_time = datetime(2099, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    session = MagicMock()
    session.get_credentials.return_value = creds

    result = session_to_credential_process(session)
    assert result == {
        "Version": 1,
        "AccessKeyId": "AKIA-P",
        "SecretAccessKey": "sk",
        "SessionToken": "tok",
        "Expiration": "2099-01-02T03:04:05+00:00",
    }


def test_session_to_credential_process_raises_without_expiry():
    """Creds with no expiry can't form a refreshing credential_process — must raise."""
    from botocore.credentials import Credentials

    from aws_bench.exceptions import CredentialError
    from aws_bench.utils.credentials_provider import session_to_credential_process

    frozen = MagicMock(access_key="AKIA", secret_key="sk", token=None)
    creds = MagicMock(spec=Credentials)  # no _expiry_time attribute
    creds.get_frozen_credentials.return_value = frozen
    session = MagicMock()
    session.get_credentials.return_value = creds

    with pytest.raises(CredentialError, match="no expiry"):
        session_to_credential_process(session)


# ── wait_for_role ──


def test_wait_for_role_succeeds_immediately():
    """Returns immediately when assume_role succeeds on first try."""
    mock_session = MagicMock()
    mock_sts = mock_session.client.return_value
    mock_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "AK",
            "SecretAccessKey": "SK",
            "SessionToken": "ST",
        }
    }
    provider = CredentialProvider(session=mock_session)
    provider.wait_for_role("111111111111", "TestRole", timeout=10, interval=1)
    mock_sts.assume_role.assert_called_once()


# ── default client config (retry policy) ──


def test_regional_session_stamps_adaptive_retry_default():
    """A client built from a provider session defaults to adaptive retries."""
    mock_parent = MagicMock()
    mock_sts = mock_parent.client.return_value
    future_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    mock_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "AK",
            "SecretAccessKey": "SK",
            "SessionToken": "ST",
            "Expiration": future_expiry,
        }
    }
    session = _create_refreshable_session(
        mock_parent, "arn:aws:iam::123456789012:role/TestRole", "app-test"
    )
    regional = create_regional_session(session, "us-west-2")

    client = regional.client("sts")
    assert client.meta.config.retries["mode"] == "adaptive"
    # botocore Config max_attempts=8 (retries) surfaces as 9 total attempts.
    assert client.meta.config.retries["total_max_attempts"] == 9


def test_explicit_client_config_overrides_retry_default():
    """A client passing its own retries wins over the session default (merge)."""
    from botocore.config import Config

    mock_parent = MagicMock()
    mock_sts = mock_parent.client.return_value
    future_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    mock_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "AK",
            "SecretAccessKey": "SK",
            "SessionToken": "ST",
            "Expiration": future_expiry,
        }
    }
    session = _create_refreshable_session(
        mock_parent, "arn:aws:iam::123456789012:role/TestRole", "app-test"
    )

    # A timeout-only config keeps the inherited adaptive retries (field-by-field merge).
    timeout_client = session.client("sts", config=Config(connect_timeout=5))
    assert timeout_client.meta.config.retries["mode"] == "adaptive"
    assert timeout_client.meta.config.connect_timeout == 5

    # A config with its own retries wins.
    tuned_client = session.client(
        "sts", config=Config(retries={"max_attempts": 8, "mode": "adaptive"})
    )
    assert tuned_client.meta.config.retries["total_max_attempts"] == 9


def test_wait_for_role_retries_on_access_denied():
    """Retries when AssumeRole returns AccessDenied, then succeeds."""
    from botocore.exceptions import ClientError

    mock_session = MagicMock()
    mock_sts = mock_session.client.return_value

    access_denied = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "Not authorized"}}, "AssumeRole"
    )
    success = {
        "Credentials": {
            "AccessKeyId": "AK",
            "SecretAccessKey": "SK",
            "SessionToken": "ST",
        }
    }
    mock_sts.assume_role.side_effect = [access_denied, access_denied, success]

    provider = CredentialProvider(session=mock_session)
    provider.wait_for_role("111111111111", "TestRole", timeout=30, interval=0)

    assert mock_sts.assume_role.call_count == 3


def test_wait_for_role_raises_on_timeout():
    """Raises CredentialError when role never becomes assumable."""
    from botocore.exceptions import ClientError

    from aws_bench.exceptions import CredentialError as AwsCredentialError

    mock_session = MagicMock()
    mock_sts = mock_session.client.return_value
    mock_sts.assume_role.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "Not authorized"}}, "AssumeRole"
    )

    provider = CredentialProvider(session=mock_session)
    with pytest.raises(AwsCredentialError, match="not assumable after"):
        provider.wait_for_role("111111111111", "TestRole", timeout=0, interval=0)


def test_wait_for_role_propagates_non_access_denied_errors():
    """Non-AccessDenied ClientErrors propagate immediately."""
    from botocore.exceptions import ClientError

    mock_session = MagicMock()
    mock_sts = mock_session.client.return_value
    mock_sts.assume_role.side_effect = ClientError(
        {"Error": {"Code": "MalformedPolicyDocument", "Message": "bad"}}, "AssumeRole"
    )

    provider = CredentialProvider(session=mock_session)
    with pytest.raises(ClientError, match="MalformedPolicyDocument"):
        provider.wait_for_role("111111111111", "TestRole", timeout=10, interval=0)


def test_wait_for_role_raises_on_shutdown():
    """A shutdown unwinds the retry loop instead of polling for the full timeout."""
    from aws_bench.exceptions import OperationCancelled
    from aws_bench.utils import concurrent

    mock_session = MagicMock()
    mock_sts = mock_session.client.return_value

    provider = CredentialProvider(session=mock_session)
    concurrent.reset_shutdown()
    concurrent.request_shutdown()
    try:
        with pytest.raises(OperationCancelled):
            provider.wait_for_role("111111111111", "TestRole", timeout=300, interval=5)
        # Bailed at the loop-entry checkpoint, before probing STS.
        mock_sts.assume_role.assert_not_called()
    finally:
        concurrent.reset_shutdown()


# ── build_aws_credentials_file ──


def _creds(n: str) -> dict[str, str]:
    return {
        "AWS_ACCESS_KEY_ID": f"AKIA{n}",
        "AWS_SECRET_ACCESS_KEY": f"secret{n}",
        "AWS_SESSION_TOKEN": f"token{n}",
    }


def test_build_aws_credentials_file_one_block_per_tag():
    body = build_aws_credentials_file({"PRIMARY": _creds("1"), "SECONDARY": _creds("2")})
    assert "[PRIMARY]" in body
    assert "[SECONDARY]" in body
    assert "aws_access_key_id=AKIA1" in body
    assert "aws_session_token=token2" in body


def test_build_aws_credentials_file_writes_no_default_block():
    """No [default] for any tag count: a profile must always be named explicitly."""
    single = build_aws_credentials_file({"PRIMARY": _creds("1")})
    multi = build_aws_credentials_file({"PRIMARY": _creds("1"), "SECONDARY": _creds("2")})
    assert "[default]" not in single
    assert "[default]" not in multi
    assert "[PRIMARY]" in single


def test_build_aws_credentials_file_omits_role_arn_and_credential_source():
    """Static creds only — never the credential_source/role_arn chaining form."""
    body = build_aws_credentials_file({"PRIMARY": _creds("1")})
    assert "role_arn" not in body
    assert "credential_source" not in body


def test_build_aws_credentials_file_empty_mapping_returns_empty_string():
    assert build_aws_credentials_file({}) == ""


# ── build_session_name (the single CloudTrail naming constructor) ──


def test_build_session_name_prepends_prefix_and_joins():
    """Composes app-<segments> with hyphens from the shared prefix constant."""
    assert build_session_name("rm", "cleanup") == "app-rm-cleanup"
    assert build_session_name("quota", "verify", "123456") == "app-quota-verify-123456"
    assert build_session_name("show") == "app-show"


def test_build_session_name_uses_prefix_constant():
    """The prefix comes from SESSION_NAME_PREFIX, the single source of truth."""
    assert build_session_name("x").startswith(SESSION_NAME_PREFIX + "-")


def test_build_session_name_truncates_to_sts_limit():
    """Composed names are truncated to STS's 64-char limit."""
    result = build_session_name("a" * 100)
    assert len(result) == MAX_SESSION_NAME_LEN
    assert result.startswith("app-")


def test_build_session_name_output_passes_enforce():
    """Anything build_session_name produces satisfies the choke-point validator."""
    name = build_session_name("rm", "reset")
    assert enforce_session_name(name) == name


# ── enforce_session_name (the single CloudTrail naming choke point) ──


@pytest.mark.parametrize(
    "name",
    [
        "app-rm-cleanup",
        "app-rm-cleanup-stack",
        "app-rm-reset",
        "app-rm-verify",
        "app-rm-snapshot-post_setup",
        "app-role-probe",
        "app-show",
        "app-exports-123456",
        "app-quota-123456",
        "app-quota-verify-123456",
        "app-quota-show-123456",
        "app-org-123456",
        "app-agent-aws-introspection-list-ec-instances",
    ],
)
def test_enforce_session_name_accepts_convention(name):
    """Every name following the app- convention passes through unchanged."""
    assert enforce_session_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "cleanup-account",
        "reset-account",
        "verify-account",
        "snapshot-post_setup",
        "verify",
        "reset",
        "cleanup",
        "AWSBench-rm-cleanup",  # wrong case: prefix is lowercase
        "",
    ],
)
def test_enforce_session_name_rejects_missing_prefix(name):
    """Names without the app- prefix are rejected before reaching STS."""
    with pytest.raises(ValueError, match="must start with 'app-'"):
        enforce_session_name(name)


def test_enforce_session_name_truncates_to_sts_limit():
    """Over-long names are truncated to STS's 64-char limit, prefix preserved."""
    long_name = "app-" + "a" * 100
    result = enforce_session_name(long_name)
    assert len(result) == MAX_SESSION_NAME_LEN
    assert result.startswith("app-")


def test_assume_role_enforces_session_name_convention():
    """The public assume_role path rejects a non-conforming name before calling STS.

    Guards the choke point: a future caller passing a bad RoleSessionName fails
    fast rather than writing an unattributable CloudTrail entry.
    """
    mock_session = MagicMock()
    provider = CredentialProvider(session=mock_session)
    with pytest.raises(ValueError, match="must start with 'app-'"):
        provider.assume_role("123456789012", "SomeRole", "bad-session-name")
    # STS must never be invoked with an invalid session name.
    mock_session.client.return_value.assume_role.assert_not_called()
