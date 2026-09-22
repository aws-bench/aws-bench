"""Tests for aws_bench.utils.credentials_provider."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from threading import Event
from types import MappingProxyType
from unittest.mock import MagicMock, patch

import boto3
import pytest
from botocore.client import BaseClient
from botocore.configloader import load_config
from botocore.credentials import Credentials, DeferredRefreshableCredentials, RefreshableCredentials
from botocore.exceptions import ClientError, NoCredentialsError

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.account_management.preexisting import ACCOUNT_CONFIG_ENV_VAR
from aws_bench.exceptions import CredentialError
from aws_bench.utils.credentials_provider import (
    CREDENTIAL_ENV_VARS,
    CREDS_DIR,
    MAX_SESSION_NAME_LEN,
    SESSION_NAME_PREFIX,
    CredentialProvider,
    _create_refreshable_session,
    build_aws_config,
    build_aws_credentials_file,
    build_session_name,
    check_static_profiles,
    create_regional_session,
    credential_command,
    credential_env,
    enforce_session_name,
    mint_credentials,
    session_to_credential_process,
    session_to_env_credentials,
    validate_account_tag,
)


@pytest.fixture(autouse=True)
def _isolate_aws(tmp_path, monkeypatch):
    """Use synthetic credentials and refuse any unmocked cloud call."""
    for key in (*CREDENTIAL_ENV_VARS, ACCOUNT_CONFIG_ENV_VAR):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "HOST_TEST_KEY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "HOST_TEST_SECRET")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "HOST_TEST_TOKEN")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "unused-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "unused-credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")

    def unexpected_call(*args, **kwargs):
        pytest.fail("Unmocked AWS call")

    monkeypatch.setattr(BaseClient, "_make_api_call", unexpected_call)


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
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)

    def refresh():
        assert creds._refresh_lock.locked()
        return {
            "access_key": "AKIA-FRESH",
            "secret_key": "fresh-secret",
            "token": "fresh-token",
            "expiry_time": expiry.isoformat(),
        }

    creds = RefreshableCredentials(
        "AKIA-OLD",
        "old-secret",
        "old-token",
        expiry,
        refresh,
        "test",
    )
    session = boto3.Session(region_name="us-east-1")
    session._session._credentials = creds

    result = session_to_env_credentials(session)

    assert result == {
        "AWS_ACCESS_KEY_ID": "AKIA-FRESH",
        "AWS_SECRET_ACCESS_KEY": "fresh-secret",
        "AWS_SESSION_TOKEN": "fresh-token",
    }


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
    assert build_session_name("session") == "app-session"
    assert build_session_name("session", "123456") == "app-session-123456"


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
    name = build_session_name("session", "123456")
    assert enforce_session_name(name) == name


# ── enforce_session_name (the single CloudTrail naming choke point) ──


@pytest.mark.parametrize(
    "name",
    [
        "app-session",
        "app-session-123456",
        "app-session-00000000-0000-0000-0000-000000000001",
        "app-test-session-name",  # arbitrary caller-provided name, still valid
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


def _session_with_credentials(creds) -> boto3.Session:
    session = boto3.Session(region_name="us-east-1")
    session._session._credentials = creds
    return session


def _refreshable_credentials(
    generation: str, expiry: datetime | None, refresh=None
) -> RefreshableCredentials:
    def unexpected_refresh():
        pytest.fail("Unexpected credential refresh")

    return RefreshableCredentials(
        f"{generation}_KEY",
        f"{generation}_SECRET",
        f"{generation}_TOKEN",
        expiry,
        refresh or unexpected_refresh,
        "test",
    )


def _metadata(generation: str, expiry: datetime) -> dict[str, str]:
    return {
        "access_key": f"{generation}_KEY",
        "secret_key": f"{generation}_SECRET",
        "token": f"{generation}_TOKEN",
        "expiry_time": expiry.isoformat(),
    }


def test_credentials_directory_is_relative_to_container_home():
    assert CREDS_DIR == PurePosixPath(".aws/creds")
    assert PurePosixPath("/home/runner") / CREDS_DIR == PurePosixPath("/home/runner/.aws/creds")


def test_credential_env_blocks_the_exact_spec_sources():
    expected = (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_CREDENTIAL_EXPIRATION",
        "AWS_ACCOUNT_ID",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CREDENTIAL_FILE",
        "BOTO_CONFIG",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_EC2_METADATA_SERVICE_ENDPOINT",
        "AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
    )
    assert CREDENTIAL_ENV_VARS == expected
    assert len(CREDENTIAL_ENV_VARS) == 21
    result = credential_env("PRIMARY")
    assert result == {
        **dict.fromkeys(expected, ""),
        "AWS_PROFILE": "PRIMARY",
        "AWS_EC2_METADATA_DISABLED": "true",
    }


def test_credential_env_preserves_model_region_and_host_environment():
    original = {
        **dict.fromkeys(CREDENTIAL_ENV_VARS, "OLD_SECRET"),
        "AWS_EC2_METADATA_DISABLED": "false",
        "AWS_BEARER_TOKEN_BEDROCK": "MODEL_TOKEN",
        "AWS_REGION": "eu-west-1",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_MAX_ATTEMPTS": "3",
        "HOME": "/home/runner",
        "PATH": "/usr/bin",
        "CUSTOM": "value",
    }
    before = dict(os.environ)
    result = credential_env("PRIMARY", MappingProxyType(original))
    assert "OLD_SECRET" not in result.values()
    assert original["AWS_SECRET_ACCESS_KEY"] == "OLD_SECRET"
    assert result["AWS_DEFAULT_PROFILE"] == ""
    for key in (
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_MAX_ATTEMPTS",
        "HOME",
        "PATH",
        "CUSTOM",
    ):
        assert result[key] == original[key]
    assert dict(os.environ) == before


@pytest.mark.parametrize("inherited", ["OLD_SECRET", ""])
@pytest.mark.parametrize("transport_overlay", [False, True])
def test_credential_command_unsets_sources_in_real_child_shell(inherited, transport_overlay):
    env = {
        **dict.fromkeys(CREDENTIAL_ENV_VARS, inherited),
        "PATH": os.defpath,
        "AWS_BEARER_TOKEN_BEDROCK": "MODEL_TOKEN",
        "AWS_REGION": "eu-west-1",
        "AWS_DEFAULT_REGION": "us-east-1",
        "CUSTOM": "kept",
    }
    if transport_overlay:
        env.update(credential_env("PRIMARY"))
    script = "import json, os; print(json.dumps(dict(os.environ)))"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    result = subprocess.run(
        ["sh", "-c", credential_command(command, "PRIMARY")],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    child = json.loads(result.stdout)
    assert child["AWS_PROFILE"] == "PRIMARY"
    assert child["AWS_EC2_METADATA_DISABLED"] == "true"
    assert not (set(CREDENTIAL_ENV_VARS) - {"AWS_PROFILE"}).intersection(child)
    for key in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION", "AWS_DEFAULT_REGION", "CUSTOM"):
        assert child[key] == env[key]
    assert "OLD_SECRET" not in credential_command(command, "PRIMARY")


def test_credential_command_preserves_shell_script_and_exit_status():
    command = 'cat <<\'EOF\'\n"quoted" $literal `literal`\nEOF\nprintf "%s" "$AWS_PROFILE"\nexit 17'
    wrapped = credential_command(command, "Primary_1")
    result = subprocess.run(
        ["sh", "-c", wrapped], env={"PATH": os.defpath}, capture_output=True, text=True
    )
    assert wrapped.endswith(command)
    assert result.returncode == 17
    assert result.stdout == '"quoted" $literal `literal`\nPrimary_1'


@pytest.mark.parametrize("readonly", ["AWS_PROFILE", "AWS_EC2_METADATA_DISABLED"])
def test_credential_command_stops_if_environment_cleanup_fails(readonly):
    result = subprocess.run(
        ["sh", "-c", f"readonly {readonly}=OLD\n" + credential_command("echo UNSAFE", "PRIMARY")],
        env={"PATH": os.defpath},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "UNSAFE" not in result.stdout


@pytest.mark.parametrize("tag", ["A", "PRIMARY", "primary", "Primary_1", "A" * 32, "default"])
def test_shared_account_tag_validation_accepts_safe_names(tag):
    assert validate_account_tag(tag) == tag


@pytest.mark.parametrize(
    "tag",
    [
        "",
        "1PRIMARY",
        "_PRIMARY",
        "A" * 33,
        "PRIMARY\n",
        "PRIMARY\r",
        "PRI\0MARY",
        "PŘIMARY",
        "../PRIMARY",
        "PRIMARY.json",
        "PRIMARY; echo BAD",
        "PRIMARY'\"",
        "PRIMARY$(echo BAD)",
        "PRIMARY-2",
        "HOME",
        "PATH",
        "MANAGEMENT_ROLE",
        "AWS_EC2_METADATA_DISABLED",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        *CREDENTIAL_ENV_VARS,
    ],
)
def test_all_credential_entrypoints_reject_unsafe_tags_before_cloud_calls(tag):
    provider = MagicMock()
    calls = [
        lambda: credential_env(tag),
        lambda: credential_command("true", tag),
        lambda: build_aws_config([tag]),
        lambda: build_aws_credentials_file({tag: _creds("1")}),
        lambda: check_static_profiles([tag], ""),
        lambda: mint_credentials(provider, {tag: "111122223333"}, "TaskRole", "app-session"),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="Invalid account_tag"):
            call()
    provider.get_chained_session_for_account.assert_not_called()


def _parsed_config(tmp_path: Path, content: str) -> dict:
    path = tmp_path / "config"
    path.write_text(content)
    return load_config(str(path))


def test_build_aws_config_creates_only_process_profiles(tmp_path):
    body = build_aws_config(iter(["PRIMARY", "Secondary_2"]), region="eu-west-1")
    parsed = _parsed_config(tmp_path, body)["profiles"]
    assert set(parsed) == {"PRIMARY", "Secondary_2"}
    for tag, profile in parsed.items():
        assert profile == {
            "credential_process": f"""sh -c 'cat "$HOME/.aws/creds/{tag}.json"'""",
            "region": "eu-west-1",
        }
    assert build_aws_config(["PRIMARY"]) == (
        "[profile PRIMARY]\ncredential_process = sh -c 'cat \"$HOME/.aws/creds/PRIMARY.json\"'\n"
    )
    assert build_aws_config([], existing="# untouched\n") == "# untouched\n"


@pytest.mark.parametrize("header", ["profile PRIMARY", 'profile "PRIMARY"', "profile 'PRIMARY'"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_build_aws_config_preserves_text_and_existing_region(tmp_path, header, newline):
    existing = newline.join(
        [
            "# Keep comments and formatting",
            "[profile OTHER]",
            "aws_access_key_id = OTHER_SECRET",
            "region = us-west-2",
            "",
            f"[{header}]",
            "region=ap-south-1",
            "output = json",
            "s3 =",
            "  addressing_style = path",
            "# 100% preserved",
            "[plugins]",
            "custom = plugin",
            "",
        ]
    )
    body = build_aws_config(["PRIMARY", "SECONDARY"], existing, region="eu-west-1")
    addition = "credential_process = sh -c 'cat \"$HOME/.aws/creds/PRIMARY.json\"'\n"
    assert body.replace(addition, "", 1).startswith(existing)
    parsed = _parsed_config(tmp_path, body)["profiles"]
    assert parsed["PRIMARY"]["region"] == "ap-south-1"
    assert parsed["PRIMARY"]["s3"]["addressing_style"] == "path"
    assert parsed["SECONDARY"]["region"] == "eu-west-1"
    assert parsed["OTHER"]["aws_access_key_id"] == "OTHER_SECRET"
    assert build_aws_config(["PRIMARY", "SECONDARY"], body, region="us-east-1") == body


@pytest.mark.parametrize("indent", ["  ", "\t"], ids=["spaces", "tab"])
@pytest.mark.parametrize(
    "first_option, addition, expected_region",
    [
        (
            "region = eu-west-1",
            """credential_process = sh -c 'cat "$HOME/.aws/creds/PRIMARY.json"'""",
            "eu-west-1",
        ),
        (
            """credential_process = sh -c 'cat "$HOME/.aws/creds/PRIMARY.json"'""",
            "region = us-east-1",
            "us-east-1",
        ),
    ],
    ids=["add-process", "add-region"],
)
def test_build_aws_config_preserves_indented_profile_settings(
    tmp_path, monkeypatch, indent, first_option, addition, expected_region
):
    existing = (
        "[profile OTHER]\nregion=us-west-2\n"
        "[profile PRIMARY]\n    ; keep this comment\n\n"
        f"{indent}{first_option}\n"
        f"{indent}output = json\n"
        f"{indent}s3 =\n"
        f"{indent}  addressing_style = path\n"
        "[plugins]\ncustom=kept\n"
    )
    body = build_aws_config(["PRIMARY"], existing, region="us-east-1")
    parsed = _parsed_config(tmp_path, body)
    assert parsed["profiles"]["PRIMARY"] == {
        "credential_process": """sh -c 'cat "$HOME/.aws/creds/PRIMARY.json"'""",
        "region": expected_region,
        "output": "json",
        "s3": {"addressing_style": "path"},
    }
    assert parsed["profiles"]["OTHER"] == {"region": "us-west-2"}
    assert parsed["plugins"] == {"custom": "kept"}
    assert body.replace(f"{indent}{addition}\n", "", 1) == existing
    assert build_aws_config(["PRIMARY"], body, region="us-east-1") == body

    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    session = boto3.Session(
        profile_name="PRIMARY",
        aws_access_key_id="SYNTHETIC_KEY",
        aws_secret_access_key="SYNTHETIC_SECRET",
    )
    client = session.client("ec2")
    try:
        assert client.meta.region_name == expected_region
    finally:
        client.close()


def test_build_aws_config_preserves_indented_header_after_empty_profile(tmp_path):
    existing = "[profile PRIMARY]\n  [profile OTHER]\n  region = eu-west-1\n"
    body = build_aws_config(["PRIMARY"], existing)
    parsed = _parsed_config(tmp_path, body)["profiles"]
    assert parsed == {
        "PRIMARY": {"credential_process": """sh -c 'cat "$HOME/.aws/creds/PRIMARY.json"'"""},
        "OTHER": {"region": "eu-west-1"},
    }
    assert build_aws_config(["PRIMARY"], body) == body


@pytest.mark.parametrize("existing", ["", "# comment", "[profile PRIMARY]", "[profile PRIMARY]\n"])
def test_build_aws_config_handles_missing_final_newline(tmp_path, existing):
    body = build_aws_config(["PRIMARY"], existing)
    assert _parsed_config(tmp_path, body)["profiles"]["PRIMARY"]["credential_process"]
    assert body.startswith(existing)


def test_build_aws_config_preserves_exact_existing_process(tmp_path):
    existing = build_aws_config(["PRIMARY"])
    assert build_aws_config(["PRIMARY"], existing) == existing
    body = build_aws_config(["PRIMARY"], existing, region="us-west-2")
    assert _parsed_config(tmp_path, body)["profiles"]["PRIMARY"]["region"] == "us-west-2"
    assert body.count("credential_process") == 1


@pytest.mark.parametrize("header", ["default", "profile default", 'profile "default"'])
def test_build_aws_config_supports_default_profile(tmp_path, header):
    body = build_aws_config(["default"], f"[{header}]\noutput=json\n")
    assert _parsed_config(tmp_path, body)["profiles"]["default"] == {
        "credential_process": """sh -c 'cat "$HOME/.aws/creds/default.json"'""",
        "output": "json",
    }


@pytest.mark.parametrize(
    "setting",
    [
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "aws_security_token",
        "aws_account_id",
        "role_arn",
        "source_profile",
        "credential_source",
        "web_identity_token_file",
        "role_session_name",
        "external_id",
        "mfa_serial",
        "duration_seconds",
        "sso_session",
        "sso_start_url",
        "sso_region",
        "sso_account_id",
        "sso_role_name",
        "login_session",
        "credential_process",
    ],
)
@pytest.mark.parametrize("value", ["DO_NOT_LOG_THIS_VALUE", ""])
def test_build_aws_config_rejects_selected_credential_settings_with_redacted_errors(setting, value):
    existing = f"[profile PRIMARY]\n{setting.upper()} = {value}\n"
    with pytest.raises(CredentialError, match="conflicting credential settings") as exc:
        build_aws_config(["PRIMARY"], existing)
    assert "DO_NOT_LOG_THIS_VALUE" not in "".join(traceback.format_exception(exc.value))


@pytest.mark.parametrize(
    "existing",
    [
        "[DEFAULT]\naws_secret_access_key=DO_NOT_LOG_THIS_VALUE\n",
        '[profile "PRIMARY"]\ncredential_process=DO_NOT_LOG_THIS_VALUE\n',
        "[profile PRIMARY]\n[profile 'PRIMARY']\n",
        "[default]\n[profile default]\n",
    ],
)
def test_build_aws_config_rejects_inherited_or_aliased_conflicts(existing):
    with pytest.raises(CredentialError) as exc:
        build_aws_config(["PRIMARY", "default"], existing)
    assert "DO_NOT_LOG_THIS_VALUE" not in "".join(traceback.format_exception(exc.value))


@pytest.mark.parametrize(
    "existing",
    [
        "DO_NOT_LOG_THIS_VALUE",
        "[broken\nDO_NOT_LOG_THIS_VALUE",
        "[profile PRIMARY]\nkey=DO_NOT_LOG_THIS_VALUE\nkey=duplicate\n",
        "[PRIMARY]\n[PRIMARY]\nsecret=DO_NOT_LOG_THIS_VALUE\n",
    ],
)
def test_ini_parse_errors_do_not_disclose_file_contents(existing):
    for operation in (build_aws_config, check_static_profiles):
        with pytest.raises(CredentialError, match="Cannot parse AWS") as exc:
            operation(["PRIMARY"], existing)
        assert "DO_NOT_LOG_THIS_VALUE" not in "".join(traceback.format_exception(exc.value))


def test_build_aws_config_refuses_ambiguous_multiline_header():
    existing = "[profile OTHER]\ns3=\n  [profile PRIMARY]\n[profile PRIMARY]\n"
    with pytest.raises(CredentialError, match="unambiguously"):
        build_aws_config(["PRIMARY"], existing)


@pytest.mark.parametrize("region", ["", "us-east-1\n", "x\naws_access_key_id=secret", "x y"])
def test_build_aws_config_rejects_region_injection(region):
    with pytest.raises(ValueError, match="Invalid AWS region"):
        build_aws_config(["PRIMARY"], region=region)


@pytest.mark.parametrize(
    "contents", ["", "region=us-east-1\n", "aws_access_key_id=STATIC_SECRET\n"]
)
@pytest.mark.parametrize("tag", ["PRIMARY", "default"])
def test_check_static_profiles_rejects_any_selected_section(tag, contents):
    with pytest.raises(CredentialError, match="selected profile") as exc:
        check_static_profiles([tag], f"[{tag}]\n{contents}")
    assert "STATIC_SECRET" not in str(exc.value)


def test_check_static_profiles_allows_unrelated_profiles():
    existing = "[OTHER]\naws_access_key_id=KEEP_ME\n[primary]\naws_secret_access_key=KEEP_ME\n"
    assert check_static_profiles(["PRIMARY"], existing) is None
    assert check_static_profiles(["PRIMARY"], "") is None
    assert check_static_profiles([], existing) is None


def test_process_snapshot_fetches_deferred_credentials_before_validating_expiry():
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    refresh = MagicMock(return_value=_metadata("FRESH", expiry))
    creds = DeferredRefreshableCredentials(refresh, "test")
    assert creds._expiry_time is None
    result = session_to_credential_process(_session_with_credentials(creds))
    assert result["AccessKeyId"] == "FRESH_KEY"
    assert result["Expiration"] == expiry.isoformat()
    refresh.assert_called_once()


def test_process_snapshot_reads_keys_and_expiry_after_a_forced_refresh(monkeypatch):
    old_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    new_expiry = old_expiry + timedelta(hours=1)
    creds = _refreshable_credentials("OLD", old_expiry, lambda: _metadata("NEW", new_expiry))
    session = _session_with_credentials(creds)
    original = creds.get_frozen_credentials

    def freeze_then_refresh():
        frozen = original()
        with patch.object(creds, "get_frozen_credentials", original):
            session_to_env_credentials(session)
        return frozen

    monkeypatch.setattr(creds, "get_frozen_credentials", freeze_then_refresh)
    assert session_to_credential_process(session) == {
        "Version": 1,
        "AccessKeyId": "NEW_KEY",
        "SecretAccessKey": "NEW_SECRET",
        "SessionToken": "NEW_TOKEN",
        "Expiration": new_expiry.isoformat(),
    }


def test_process_snapshot_waits_for_concurrent_forced_refresh(monkeypatch):
    old_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    new_expiry = old_expiry + timedelta(hours=1)
    creds = _refreshable_credentials("OLD", old_expiry, lambda: _metadata("NEW", new_expiry))
    session = _session_with_credentials(creds)
    updated = Event()
    release = Event()
    snapshot_started = Event()
    snapshot_finished = Event()
    original_set = creds._set_from_data
    original_freeze = creds.get_frozen_credentials

    def pause_before_frozen_credentials(data):
        original_set(data)
        updated.set()
        assert release.wait(5)

    def freeze():
        snapshot_started.set()
        return original_freeze()

    def snapshot():
        try:
            return session_to_credential_process(session)
        finally:
            snapshot_finished.set()

    monkeypatch.setattr(creds, "_set_from_data", pause_before_frozen_credentials)
    monkeypatch.setattr(creds, "get_frozen_credentials", freeze)
    with ThreadPoolExecutor(max_workers=2) as executor:
        env_result = executor.submit(session_to_env_credentials, session)
        try:
            assert updated.wait(5)
            process_result = executor.submit(snapshot)
            assert snapshot_started.wait(5)
            assert not snapshot_finished.wait(0.1)
        finally:
            release.set()
        assert env_result.result(timeout=5)["AWS_ACCESS_KEY_ID"] == "NEW_KEY"
        result = process_result.result(timeout=5)
    assert result["AccessKeyId"] == "NEW_KEY"
    assert result["SecretAccessKey"] == "NEW_SECRET"
    assert result["SessionToken"] == "NEW_TOKEN"
    assert result["Expiration"] == new_expiry.isoformat()


@pytest.mark.parametrize("kind", ["missing", "naive", "expired", "malformed"])
def test_process_snapshot_rejects_invalid_real_credentials(kind):
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    if kind == "missing":
        creds = Credentials("STATIC_KEY", "STATIC_SECRET", "STATIC_TOKEN")
    else:
        returned_expiry = {
            "naive": expiry.replace(tzinfo=None).isoformat(),
            "expired": (expiry - timedelta(hours=2)).isoformat(),
            "malformed": "not-a-date",
        }[kind]
        metadata = {**_metadata("BAD", expiry), "expiry_time": returned_expiry}
        creds = DeferredRefreshableCredentials(lambda: metadata, "test")
    with pytest.raises(CredentialError):
        session_to_credential_process(_session_with_credentials(creds))


@pytest.mark.parametrize("kind", ["naive", "expired", "missing"])
def test_process_snapshot_rejects_invalid_cached_expiry(kind):
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    if kind == "naive":
        expiry = expiry.replace(tzinfo=None)
    elif kind == "expired":
        expiry -= timedelta(hours=2)
    else:
        expiry = None
    creds = _refreshable_credentials("BAD", expiry)
    if kind == "expired":
        # Model a credential provider that returned an expired cached snapshot.
        creds._advisory_refresh_timeout = -7200
    with pytest.raises(CredentialError):
        session_to_credential_process(_session_with_credentials(creds))


@pytest.mark.parametrize(
    "error",
    [
        ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "AssumeRole"),
        NoCredentialsError(),
    ],
)
@pytest.mark.parametrize("failure_at", ["session", "refresh"])
def test_env_snapshot_preserves_native_retrieval_errors(error, failure_at):
    session = MagicMock()
    if failure_at == "session":
        session.get_credentials.side_effect = error
    else:
        session.get_credentials.return_value = DeferredRefreshableCredentials(
            MagicMock(side_effect=error), "test"
        )
    with pytest.raises(type(error)) as exc:
        session_to_env_credentials(session)
    assert exc.value is error


@pytest.mark.parametrize(
    "error",
    [
        ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "AssumeRole"),
        NoCredentialsError(),
    ],
)
def test_process_snapshot_preserves_retrieval_failure_before_expiry_validation(error):
    refresh = MagicMock(side_effect=error)
    creds = DeferredRefreshableCredentials(refresh, "test")
    with pytest.raises(CredentialError, match="Failed to retrieve") as exc:
        session_to_credential_process(_session_with_credentials(creds))
    assert exc.value.__cause__ is error
    assert creds._expiry_time is None


def test_regional_session_resolves_lazy_ambient_credentials_once():
    parent = boto3.Session(region_name="us-east-1")
    assert parent._session._credentials is None
    regional = create_regional_session(parent, "eu-west-1")
    assert regional.get_credentials() is parent.get_credentials()
    assert regional.get_credentials().get_frozen_credentials().access_key == "HOST_TEST_KEY"


@pytest.fixture
def fake_sts(monkeypatch):
    calls = []
    signers = []

    def call(client, operation, params):
        assert operation == "AssumeRole"
        parent = client._request_signer._credentials
        frozen = parent.get_frozen_credentials()
        calls.append((params, frozen.access_key))
        signers.append(parent)
        generation = len(calls)
        return {
            "Credentials": {
                "AccessKeyId": f"ROLE_{generation}_KEY",
                "SecretAccessKey": f"ROLE_{generation}_SECRET",
                "SessionToken": f"ROLE_{generation}_TOKEN",
                "Expiration": datetime.now(timezone.utc) + timedelta(hours=1),
            }
        }

    monkeypatch.setattr(BaseClient, "_make_api_call", call)
    return calls, signers


@pytest.mark.parametrize("role", [None, ORG_ACCESS_ROLE, "service/TaskRole"])
def test_refreshable_managed_chain_preserves_role_order_and_neutral_names(fake_sts, role):
    calls, _ = fake_sts
    provider = CredentialProvider()
    session = provider.get_chained_session_for_account(
        "111122223333", role, "app-session-opaque", "eu-west-1"
    )
    assert isinstance(session.get_credentials(), DeferredRefreshableCredentials)
    assert calls == []
    result = session_to_credential_process(session)
    assert session.region_name == "eu-west-1"
    expected_roles = [ORG_ACCESS_ROLE] + ([role] if role not in (None, ORG_ACCESS_ROLE) else [])
    assert [params["RoleArn"] for params, _ in calls] == [
        f"arn:aws:iam::111122223333:role/{name}" for name in expected_roles
    ]
    assert [parent for _, parent in calls] == ["HOST_TEST_KEY"] + (
        ["ROLE_1_KEY"] if len(expected_roles) == 2 else []
    )
    assert calls[-1][0]["RoleSessionName"] == "app-session-opaque"
    if len(expected_roles) == 2:
        assert calls[0][0]["RoleSessionName"] == "app-session-223333"
    assert result["AccessKeyId"] == f"ROLE_{len(expected_roles)}_KEY"


def test_direct_session_api_keeps_single_hop_for_managed_accounts(fake_sts):
    calls, _ = fake_sts
    session = CredentialProvider().get_session_for_account(
        "111122223333", "TaskRole", "app-session"
    )
    session_to_credential_process(session)
    assert len(calls) == 1
    assert calls[0] == (
        {
            "RoleArn": "arn:aws:iam::111122223333:role/TaskRole",
            "RoleSessionName": "app-session",
        },
        "HOST_TEST_KEY",
    )


def test_refreshable_managed_chain_renews_host_and_both_role_hops(fake_sts):
    calls, signers = fake_sts
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    host = _refreshable_credentials("HOST_OLD", expiry, lambda: _metadata("HOST_NEW", expiry))
    provider = CredentialProvider(_session_with_credentials(host))
    session = provider.get_chained_session_for_account("111122223333", "TaskRole", "app-session")
    first = session_to_credential_process(session)
    assert first["AccessKeyId"] == "ROLE_2_KEY"
    expired = datetime.now(timezone.utc) - timedelta(seconds=1)
    for creds in [*signers, session.get_credentials()]:
        creds._expiry_time = expired
    second = session_to_credential_process(session)
    assert second["AccessKeyId"] == "ROLE_4_KEY"
    assert [parent for _, parent in calls] == [
        "HOST_OLD_KEY",
        "ROLE_1_KEY",
        "HOST_NEW_KEY",
        "ROLE_3_KEY",
    ]
    assert [params["RoleArn"] for params, _ in calls[:2]] == [
        params["RoleArn"] for params, _ in calls[2:]
    ]


def test_mint_credentials_returns_only_json_files_and_earliest_expiry(monkeypatch, tmp_path):
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    sessions = [
        _session_with_credentials(_refreshable_credentials("PRIMARY", expiry + timedelta(hours=1))),
        _session_with_credentials(_refreshable_credentials("SECONDARY", expiry)),
    ]
    provider = CredentialProvider()
    get_session = MagicMock(side_effect=sessions)
    monkeypatch.setattr(provider, "get_chained_session_for_account", get_session)
    before = set(tmp_path.iterdir())
    files, expires_at = mint_credentials(
        provider,
        MappingProxyType({"PRIMARY": "111122223333", "SECONDARY": "444455556666"}),
        "TaskRole",
        "app-session",
    )
    assert set(files) == {"PRIMARY.json", "SECONDARY.json"}
    assert expires_at == expiry
    assert json.loads(files["PRIMARY.json"])["AccessKeyId"] == "PRIMARY_KEY"
    assert json.loads(files["SECONDARY.json"])["Expiration"] == expiry.isoformat()
    assert get_session.call_args_list[0].args == ("111122223333", "TaskRole", "app-session")
    assert get_session.call_args_list[1].args == ("444455556666", "TaskRole", "app-session")
    assert set(tmp_path.iterdir()) == before


def test_mint_credentials_uses_the_real_managed_chain(fake_sts):
    calls, _ = fake_sts
    files, expiry = mint_credentials(
        CredentialProvider(), {"PRIMARY": "111122223333"}, "TaskRole", "app-session"
    )
    assert len(calls) == 2
    assert json.loads(files["PRIMARY.json"])["AccessKeyId"] == "ROLE_2_KEY"
    assert datetime.fromisoformat(json.loads(files["PRIMARY.json"])["Expiration"]) == expiry


def test_mint_credentials_copies_mapping_before_cloud_calls(monkeypatch):
    accounts = {"PRIMARY": "111122223333"}
    provider = CredentialProvider()
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)

    def get_session(account_id, role_name, session_name):
        accounts["PRIMARY"] = "999988887777"
        accounts["SECONDARY"] = "444455556666"
        assert account_id == "111122223333"
        return _session_with_credentials(_refreshable_credentials("ORIGINAL", expiry))

    monkeypatch.setattr(provider, "get_chained_session_for_account", get_session)
    files, _ = mint_credentials(provider, accounts, "TaskRole", "app-session")
    assert set(files) == {"PRIMARY.json"}


@pytest.mark.parametrize(
    "accounts, session_name",
    [
        ({}, "app-session"),
        ({"PRIMARY": "111122223333", "../bad": "444455556666"}, "app-session"),
        ({"PRIMARY": "111122223333"}, "bad-name"),
    ],
)
def test_mint_credentials_validates_before_any_cloud_call(accounts, session_name):
    provider = MagicMock()
    with pytest.raises((CredentialError, ValueError)):
        mint_credentials(provider, accounts, "TaskRole", session_name)
    provider.get_chained_session_for_account.assert_not_called()


def test_mint_credentials_rejects_insufficient_lifetime_after_all_snapshots(monkeypatch):
    provider = MagicMock()
    expiry = datetime.now(timezone.utc) + timedelta(seconds=30)
    provider.get_chained_session_for_account.return_value = _session_with_credentials(
        _refreshable_credentials("SHORT", expiry)
    )
    # Keep the real frozen object without refreshing this deliberately short session.
    creds = provider.get_chained_session_for_account.return_value.get_credentials()
    assert isinstance(creds, RefreshableCredentials)
    creds._advisory_refresh_timeout = 0
    with pytest.raises(CredentialError, match="minimum refresh sleep"):
        mint_credentials(provider, {"PRIMARY": "111122223333"}, "TaskRole", "app-session")


def test_mint_credentials_does_not_return_partial_files_after_failure():
    provider = MagicMock()
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    provider.get_chained_session_for_account.side_effect = [
        _session_with_credentials(_refreshable_credentials("GOOD", expiry)),
        RuntimeError("second account unavailable"),
    ]
    with pytest.raises(RuntimeError, match="second account unavailable"):
        mint_credentials(
            provider,
            {"PRIMARY": "111122223333", "SECONDARY": "444455556666"},
            "TaskRole",
            "app-session",
        )
