"""Tests for aws_bench.utils.bedrock_credentials."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from aws_bench.utils.bedrock_credentials import (
    IAM_USER_NAME,
    POLICY_ARN,
    SSM_PARAMETER,
    BedrockCredentialError,
    _delete_all_credentials,
    _ensure_iam_user,
    _find_reusable_credential,
    _get_token_from_ssm,
    _store_token_in_ssm,
    _verify_token,
    generate_bearer_token,
)

# ===========================================================================
# _ensure_iam_user
# ===========================================================================


def test_ensure_iam_user_creates_user():
    """Test _ensure_iam_user creates user and attaches policy."""
    iam = MagicMock()
    iam.create_user.return_value = {}

    _ensure_iam_user(iam, IAM_USER_NAME)

    iam.create_user.assert_called_once_with(UserName=IAM_USER_NAME)
    iam.attach_user_policy.assert_called_once_with(UserName=IAM_USER_NAME, PolicyArn=POLICY_ARN)


def test_ensure_iam_user_already_exists():
    """Test _ensure_iam_user is idempotent when user already exists."""
    iam = MagicMock()
    iam.create_user.side_effect = ClientError(
        {"Error": {"Code": "EntityAlreadyExists", "Message": "exists"}},
        "CreateUser",
    )

    _ensure_iam_user(iam, IAM_USER_NAME)

    iam.attach_user_policy.assert_called_once()


def test_ensure_iam_user_raises_on_unexpected_error():
    """Test _ensure_iam_user raises on non-EntityAlreadyExists errors."""
    iam = MagicMock()
    iam.create_user.side_effect = ClientError(
        {"Error": {"Code": "ServiceFailure", "Message": "broken"}},
        "CreateUser",
    )

    with pytest.raises(ClientError):
        _ensure_iam_user(iam, IAM_USER_NAME)


# ===========================================================================
# _find_reusable_credential
# ===========================================================================


def test_find_reusable_credential_active_not_expiring():
    """Test finding a valid active credential."""
    future = datetime.now(timezone.utc) + timedelta(days=10)
    creds = [{"Status": "Active", "ExpirationDate": future, "ServiceSpecificCredentialId": "id1"}]

    result = _find_reusable_credential(creds, min_remaining_days=1)
    assert result is not None
    assert result["ServiceSpecificCredentialId"] == "id1"


def test_find_reusable_credential_expiring_soon():
    """Test that expiring credential is not reusable."""
    soon = datetime.now(timezone.utc) + timedelta(hours=12)
    creds = [{"Status": "Active", "ExpirationDate": soon, "ServiceSpecificCredentialId": "id1"}]

    result = _find_reusable_credential(creds, min_remaining_days=1)
    assert result is None


def test_find_reusable_credential_inactive():
    """Test that inactive credentials are skipped."""
    future = datetime.now(timezone.utc) + timedelta(days=10)
    creds = [{"Status": "Inactive", "ExpirationDate": future, "ServiceSpecificCredentialId": "id1"}]

    result = _find_reusable_credential(creds, min_remaining_days=1)
    assert result is None


def test_find_reusable_credential_empty_list():
    """Test with no credentials."""
    result = _find_reusable_credential([], min_remaining_days=1)
    assert result is None


# ===========================================================================
# _get_token_from_ssm / _store_token_in_ssm
# ===========================================================================


@mock_aws
def test_get_token_from_ssm_not_found():
    """Test SSM parameter not found returns None."""
    session = boto3.Session(region_name="us-east-1")
    ssm = session.client("ssm")

    result = _get_token_from_ssm(ssm, "/nonexistent/param")
    assert result is None


@mock_aws
def test_store_and_get_token_from_ssm():
    """Test storing and retrieving a token from SSM."""
    session = boto3.Session(region_name="us-east-1")
    ssm = session.client("ssm")

    _store_token_in_ssm(ssm, SSM_PARAMETER, "my-secret-token")
    result = _get_token_from_ssm(ssm, SSM_PARAMETER)

    assert result == "my-secret-token"


@mock_aws
def test_store_token_overwrites():
    """Test that storing a token overwrites existing value."""
    session = boto3.Session(region_name="us-east-1")
    ssm = session.client("ssm")

    _store_token_in_ssm(ssm, SSM_PARAMETER, "old-token")
    _store_token_in_ssm(ssm, SSM_PARAMETER, "new-token")

    result = _get_token_from_ssm(ssm, SSM_PARAMETER)
    assert result == "new-token"


# ===========================================================================
# _delete_all_credentials
# ===========================================================================


def test_delete_all_credentials():
    """Test deleting all service-specific credentials."""
    creds = [
        {"ServiceSpecificCredentialId": "cred-1"},
        {"ServiceSpecificCredentialId": "cred-2"},
    ]

    mock_iam = MagicMock()
    _delete_all_credentials(mock_iam, IAM_USER_NAME, creds)

    mock_iam.delete_service_specific_credential.assert_any_call(
        UserName=IAM_USER_NAME, ServiceSpecificCredentialId="cred-1"
    )
    mock_iam.delete_service_specific_credential.assert_any_call(
        UserName=IAM_USER_NAME, ServiceSpecificCredentialId="cred-2"
    )
    assert mock_iam.delete_service_specific_credential.call_count == 2


# ===========================================================================
# _verify_token
# ===========================================================================


@patch("aws_bench.utils.bedrock_credentials.urllib.request.urlopen")
def test_verify_token_success(mock_urlopen):
    """Test token verification succeeds on 200."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_response

    assert _verify_token("test-key", retries=1) is True


@patch("aws_bench.utils.bedrock_credentials.urllib.request.urlopen")
def test_verify_token_403_retries(mock_urlopen):
    """Test token verification retries on 403."""
    import urllib.error
    from http.client import HTTPMessage

    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="", code=403, msg="Forbidden", hdrs=HTTPMessage(), fp=None
    )

    with patch("aws_bench.utils.bedrock_credentials.time.sleep"):
        result = _verify_token("test-key", retries=2, delay=0)

    assert result is False
    assert mock_urlopen.call_count == 2


@patch("aws_bench.utils.bedrock_credentials.urllib.request.urlopen")
def test_verify_token_network_error(mock_urlopen):
    """Test token verification handles network errors."""
    mock_urlopen.side_effect = OSError("Connection refused")

    result = _verify_token("test-key", retries=1)
    assert result is False


# ===========================================================================
# generate_bearer_token
# ===========================================================================


@mock_aws
@patch("aws_bench.utils.bedrock_credentials._ensure_iam_user")
@patch("aws_bench.utils.bedrock_credentials._get_existing_credentials", return_value=[])
def test_generate_bearer_token_creates_new(mock_get_creds, mock_ensure):
    """Test generating a fresh bearer token when none exists."""
    session = boto3.Session(region_name="us-east-1")

    fake_cred_response = {
        "ServiceSpecificCredential": {
            "ServiceSpecificCredentialId": "cred-123",
            "ServiceCredentialSecret": "new-secret-key",
        }
    }

    original_client = session.client

    def _patched_client(svc, **kw):
        if svc == "iam":
            mock_iam = MagicMock()
            mock_iam.create_service_specific_credential.return_value = fake_cred_response
            return mock_iam
        return original_client(svc, **kw)

    with (
        patch("aws_bench.utils.bedrock_credentials.boto3.Session", return_value=session),
        patch.object(session, "client", side_effect=_patched_client),
    ):
        token = generate_bearer_token(no_verify=True)

    assert token == "new-secret-key"


@mock_aws
@patch("aws_bench.utils.bedrock_credentials._ensure_iam_user")
@patch("aws_bench.utils.bedrock_credentials._get_existing_credentials", return_value=[])
def test_generate_bearer_token_reuses_cached(mock_get_creds, mock_ensure):
    """Test that cached token from SSM is reused when valid."""
    session = boto3.Session(region_name="us-east-1")
    ssm = session.client("ssm")

    ssm.put_parameter(Name=SSM_PARAMETER, Value="cached-token", Type="SecureString", Overwrite=True)

    with patch("aws_bench.utils.bedrock_credentials.boto3.Session", return_value=session):
        token = generate_bearer_token(no_verify=True)

    assert token == "cached-token"


@mock_aws
@patch("aws_bench.utils.bedrock_credentials._get_existing_credentials")
@patch("aws_bench.utils.bedrock_credentials._ensure_iam_user")
@patch("aws_bench.utils.bedrock_credentials.urllib.request.urlopen")
@patch("aws_bench.utils.bedrock_credentials.time.sleep")
def test_generate_bearer_token_reuses_cached_token_that_403s_once(
    mock_sleep, mock_urlopen, mock_ensure, mock_get_creds
):
    """A cached token rejected once, then accepted, is reused without rotating."""
    import urllib.error
    from http.client import HTTPMessage

    # Arrange: a live credential on the account plus a cached token in SSM — the
    # state in which rotating would delete a credential that is still good.
    mock_get_creds.return_value = [
        {
            "Status": "Active",
            "ServiceSpecificCredentialId": "cred-live",
            "ExpirationDate": datetime.now(timezone.utc) + timedelta(days=29),
        }
    ]
    session = boto3.Session(region_name="us-east-1")
    ssm = session.client("ssm")
    ssm.put_parameter(Name=SSM_PARAMETER, Value="cached-token", Type="SecureString", Overwrite=True)

    accepted = MagicMock()
    accepted.status = 200
    accepted.__enter__ = lambda s: s
    accepted.__exit__ = MagicMock(return_value=False)
    mock_urlopen.side_effect = [
        urllib.error.HTTPError(url="", code=403, msg="Forbidden", hdrs=HTTPMessage(), fp=None),
        accepted,
    ]

    mock_iam = MagicMock()
    original_client = session.client

    def _patched_client(svc, **kw):
        return mock_iam if svc == "iam" else original_client(svc, **kw)

    # Act
    with (
        patch("aws_bench.utils.bedrock_credentials.boto3.Session", return_value=session),
        patch.object(session, "client", side_effect=_patched_client),
    ):
        token = generate_bearer_token()

    # Assert
    assert token == "cached-token"
    assert mock_urlopen.call_count == 2
    mock_iam.create_service_specific_credential.assert_not_called()
    mock_iam.delete_service_specific_credential.assert_not_called()


@mock_aws
@patch("aws_bench.utils.bedrock_credentials._ensure_iam_user")
@patch("aws_bench.utils.bedrock_credentials._get_existing_credentials", return_value=[])
@patch("aws_bench.utils.bedrock_credentials._verify_token", return_value=False)
def test_generate_bearer_token_rotates_when_verification_fails(
    mock_verify, mock_get_creds, mock_ensure
):
    """Test token rotation when cached token fails verification."""
    session = boto3.Session(region_name="us-east-1")
    ssm = session.client("ssm")

    ssm.put_parameter(Name=SSM_PARAMETER, Value="stale-token", Type="SecureString", Overwrite=True)

    fake_cred_response = {
        "ServiceSpecificCredential": {
            "ServiceSpecificCredentialId": "cred-new",
            "ServiceCredentialSecret": "new-key",
        }
    }
    original_client = session.client

    def _patched_client(svc, **kw):
        if svc == "iam":
            mock_iam = MagicMock()
            mock_iam.create_service_specific_credential.return_value = fake_cred_response
            return mock_iam
        return original_client(svc, **kw)

    with (
        patch("aws_bench.utils.bedrock_credentials.boto3.Session", return_value=session),
        patch.object(session, "client", side_effect=_patched_client),
    ):
        with pytest.raises(BedrockCredentialError, match="Token verification failed"):
            generate_bearer_token(force=False)


@mock_aws
@patch("aws_bench.utils.bedrock_credentials._ensure_iam_user")
@patch("aws_bench.utils.bedrock_credentials._get_existing_credentials", return_value=[])
@patch("aws_bench.utils.bedrock_credentials._delete_all_credentials")
def test_generate_bearer_token_force_recreates(mock_delete, mock_get_creds, mock_ensure):
    """Test --force deletes existing and creates fresh."""
    session = boto3.Session(region_name="us-east-1")
    ssm = session.client("ssm")

    ssm.put_parameter(Name=SSM_PARAMETER, Value="old-token", Type="SecureString", Overwrite=True)

    fake_cred_response = {
        "ServiceSpecificCredential": {
            "ServiceSpecificCredentialId": "cred-new",
            "ServiceCredentialSecret": "fresh-key",
        }
    }
    original_client = session.client

    def _patched_client(svc, **kw):
        if svc == "iam":
            mock_iam = MagicMock()
            mock_iam.create_service_specific_credential.return_value = fake_cred_response
            return mock_iam
        return original_client(svc, **kw)

    with (
        patch("aws_bench.utils.bedrock_credentials.boto3.Session", return_value=session),
        patch.object(session, "client", side_effect=_patched_client),
    ):
        token = generate_bearer_token(force=True, no_verify=True)

    assert token == "fresh-key"
