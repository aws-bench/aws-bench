"""Credential routing tests for externally owned accounts."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import boto3
import pytest
from botocore.client import BaseClient
from botocore.credentials import RefreshableCredentials
from botocore.exceptions import ClientError

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.account_management.exceptions import AccountResolutionError
from aws_bench.account_management.preexisting import ACCOUNT_CONFIG_ENV_VAR
from aws_bench.exceptions import CredentialError
from aws_bench.utils.credentials_provider import (
    CREDENTIAL_ENV_VARS,
    CredentialProvider,
    mint_credentials,
    session_to_credential_process,
)


@pytest.fixture(autouse=True)
def _isolate_aws(tmp_path, monkeypatch):
    """Prevent ambient config and unmocked AWS requests from entering these tests."""
    for key in (*CREDENTIAL_ENV_VARS, ACCOUNT_CONFIG_ENV_VAR):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "unused-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "unused-credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")

    def unexpected_call(*args, **kwargs):
        pytest.fail("Unmocked AWS call")

    monkeypatch.setattr(BaseClient, "_make_api_call", unexpected_call)


def _activate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner_role: str = "AWSBenchRunner"
) -> None:
    path = tmp_path / "accounts.yaml"
    path.write_text(
        "mode: preexisting\n"
        "name: aws-bench\n"
        f"runner_role: {runner_role}\n"
        "cfn_role: cfn-service-execution\n"
        "accounts:\n"
        "  scenario-a:\n"
        '    PRIMARY: "111122223333"\n'
    )
    monkeypatch.setenv(ACCOUNT_CONFIG_ENV_VAR, str(path))


def _provider(account_id: str = "111122223333", arn: str = "arn:aws:iam::111122223333:user/x"):
    session = MagicMock()
    sts = MagicMock()
    sts.get_caller_identity.return_value = {"Account": account_id, "Arn": arn}
    session.client.return_value = sts
    return CredentialProvider(session), session


def test_org_access_role_maps_directly_to_runner_role(tmp_path: Path, monkeypatch):
    _activate(tmp_path, monkeypatch, "AWSBenchRunner")
    provider, session = _provider()
    with patch(
        "aws_bench.utils.credentials_provider._create_refreshable_session"
    ) as create_session:
        provider.get_session_for_account(
            "111122223333", "OrganizationAccountAccessRole", "app-test"
        )
    create_session.assert_called_once_with(
        session,
        "arn:aws:iam::111122223333:role/AWSBenchRunner",
        "app-test",
        "us-east-1",
    )


def test_explicit_task_role_chains_from_configured_runner(tmp_path: Path, monkeypatch):
    _activate(tmp_path, monkeypatch, "AWSBenchRunner")
    provider, session = _provider()
    with patch(
        "aws_bench.utils.credentials_provider._create_refreshable_session"
    ) as create_session:
        provider.get_session_for_account("111122223333", "TaskRole", "app-task")
    assert create_session.call_count == 2
    runner_session = create_session.return_value
    assert create_session.call_args_list[0].args == (
        session,
        "arn:aws:iam::111122223333:role/AWSBenchRunner",
        "app-session-223333",
        "us-east-1",
    )
    assert create_session.call_args_list[1].args == (
        runner_session,
        "arn:aws:iam::111122223333:role/TaskRole",
        "app-task",
        "us-east-1",
    )


def test_already_active_runner_role_is_not_self_assumed(tmp_path: Path, monkeypatch):
    _activate(tmp_path, monkeypatch, "AWSBenchRunner")
    provider, session = _provider(
        arn="arn:aws:sts::111122223333:assumed-role/AWSBenchRunner/slurm-job"
    )
    with (
        patch("aws_bench.utils.credentials_provider._create_refreshable_session") as create,
        patch("aws_bench.utils.credentials_provider.create_regional_session") as regional,
    ):
        provider.get_session_for_account(
            "111122223333", "OrganizationAccountAccessRole", "app-test"
        )
    create.assert_not_called()
    regional.assert_called_once_with(session, "us-east-1")


def test_unnamed_role_assumes_runner_not_caller_credentials(tmp_path: Path, monkeypatch):
    """A task with no role_name gets the runner role, even when the caller sits in the account.

    The caller here is an admin identity inside the target account, so reusing the
    ambient session would hand the task the operator's own credentials.
    """
    _activate(tmp_path, monkeypatch)
    provider, _ = _provider(arn="arn:aws:sts::111122223333:assumed-role/Admin/operator")
    provider.assume_role = MagicMock(
        return_value={
            "AWS_ACCESS_KEY_ID": "runner-key",
            "AWS_SECRET_ACCESS_KEY": "runner-secret",
            "AWS_SESSION_TOKEN": "runner-token",
        }
    )

    credentials = provider.chain_assume_role("111122223333", "app-task", role_name=None)

    provider.assume_role.assert_called_once_with(
        "111122223333", "AWSBenchRunner", "app-task", duration_seconds=3600
    )
    assert credentials["AWS_ACCESS_KEY_ID"] == "runner-key"


def test_account_outside_allowlist_is_refused(tmp_path: Path, monkeypatch):
    _activate(tmp_path, monkeypatch)
    provider, _ = _provider()
    with pytest.raises(AccountResolutionError, match="not in the active pre-existing allowlist"):
        provider.chain_assume_role("999988887777", "app-test")


def test_static_task_credentials_chain_through_runner(tmp_path: Path, monkeypatch):
    _activate(tmp_path, monkeypatch, "AWSBenchRunner")
    provider, _ = _provider()
    runner_creds = {
        "AWS_ACCESS_KEY_ID": "runner-key",
        "AWS_SECRET_ACCESS_KEY": "runner-secret",
        "AWS_SESSION_TOKEN": "runner-token",
    }
    provider.assume_role = MagicMock(return_value=runner_creds)
    runner_session = MagicMock()
    runner_sts = MagicMock()
    runner_session.client.return_value = runner_sts
    runner_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "task-key",
            "SecretAccessKey": "task-secret",
            "SessionToken": "task-token",
        }
    }
    with patch(
        "aws_bench.utils.credentials_provider.env_credentials_dict_to_session",
        return_value=runner_session,
    ):
        credentials = provider.chain_assume_role("111122223333", "app-task", role_name="TaskRole")
    provider.assume_role.assert_called_once_with(
        "111122223333",
        "AWSBenchRunner",
        "app-session-223333",
        duration_seconds=3600,
    )
    runner_sts.assume_role.assert_called_once_with(
        RoleArn="arn:aws:iam::111122223333:role/TaskRole",
        RoleSessionName="app-task",
        DurationSeconds=3600,
    )
    assert credentials["AWS_ACCESS_KEY_ID"] == "task-key"


def test_ambient_runner_role_matches_configured_role_path(tmp_path: Path, monkeypatch):
    _activate(tmp_path, monkeypatch, "service/automation/AWSBenchRunner")
    provider, session = _provider(
        arn="arn:aws:sts::111122223333:assumed-role/AWSBenchRunner/runner-session"
    )
    with (
        patch("aws_bench.utils.credentials_provider._create_refreshable_session") as create,
        patch("aws_bench.utils.credentials_provider.create_regional_session") as regional,
    ):
        provider.get_session_for_account(
            "111122223333", "OrganizationAccountAccessRole", "app-session"
        )
    create.assert_not_called()
    regional.assert_called_once_with(session, "us-east-1")


@pytest.fixture
def real_provider(monkeypatch):
    """Use real sessions, refresh locks and signers with fake STS responses."""
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    refreshes = []

    def refresh_host():
        refreshes.append(True)
        return {
            "access_key": "AMBIENT_RENEWED_KEY",
            "secret_key": "AMBIENT_RENEWED_SECRET",
            "token": "AMBIENT_RENEWED_TOKEN",
            "expiry_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        }

    credentials = RefreshableCredentials(
        "AMBIENT_KEY", "AMBIENT_SECRET", "AMBIENT_TOKEN", expiry, refresh_host, "test"
    )
    session = boto3.Session(region_name="us-east-1")
    session._session._credentials = credentials
    identity = {"Account": "111122223333", "Arn": "arn:aws:iam::111122223333:user/operator"}
    calls = []
    signers = []

    def call(client, operation, params):
        assert operation in {"GetCallerIdentity", "AssumeRole"}
        signer = client._request_signer._credentials
        frozen = signer.get_frozen_credentials()
        if operation == "GetCallerIdentity":
            return dict(identity)
        calls.append((params, frozen.access_key))
        signers.append(signer)
        return {
            "Credentials": {
                "AccessKeyId": f"ROLE_{len(calls)}_KEY",
                "SecretAccessKey": f"ROLE_{len(calls)}_SECRET",
                "SessionToken": f"ROLE_{len(calls)}_TOKEN",
                "Expiration": datetime.now(timezone.utc) + timedelta(hours=1),
            }
        }

    monkeypatch.setattr(BaseClient, "_make_api_call", call)
    return CredentialProvider(session), identity, calls, signers, refreshes


@pytest.mark.parametrize(
    "target, ambient, expected",
    [
        (None, None, ["service/AWSBenchRunner"]),
        (ORG_ACCESS_ROLE, None, ["service/AWSBenchRunner"]),
        ("service/AWSBenchRunner", None, ["service/AWSBenchRunner"]),
        ("service/AWSBenchRunner", "AWSBenchRunner", []),
        ("service/TaskRole", "TaskRole", []),
        ("service/TaskRole", "AWSBenchRunner", ["service/TaskRole"]),
        ("service/TaskRole", None, ["service/AWSBenchRunner", "service/TaskRole"]),
        ("service/TaskRole", "TaskRoleExtra", ["service/AWSBenchRunner", "service/TaskRole"]),
    ],
)
def test_preexisting_refreshable_chain_respects_runner_and_ambient_identity(
    tmp_path, monkeypatch, real_provider, target, ambient, expected
):
    _activate(tmp_path, monkeypatch, "service/AWSBenchRunner")
    provider, identity, calls, _, _ = real_provider
    if ambient:
        identity["Arn"] = f"arn:aws:sts::111122223333:assumed-role/{ambient}/host-session"
    session = provider.get_chained_session_for_account(
        "111122223333", target, "app-session-opaque", "eu-west-1"
    )
    assert session.region_name == "eu-west-1"
    snapshot = session_to_credential_process(session)
    assert [params["RoleArn"] for params, _ in calls] == [
        f"arn:aws:iam::111122223333:role/{role}" for role in expected
    ]
    if expected:
        assert calls[-1][0]["RoleSessionName"] == "app-session-opaque"
        assert calls[0][1] == "AMBIENT_KEY"
        assert snapshot["AccessKeyId"] == f"ROLE_{len(expected)}_KEY"
    else:
        assert session.get_credentials() is provider.session.get_credentials()
        assert snapshot["AccessKeyId"] == "AMBIENT_KEY"
    if len(expected) == 2:
        assert calls[0][0]["RoleSessionName"] == "app-session-223333"
        assert calls[1][1] == "ROLE_1_KEY"


@pytest.mark.parametrize(
    "arn",
    [
        "arn:aws:sts::999988887777:assumed-role/AWSBenchRunner/session",
        "arn:aws:sts::111122223333:assumed-role/AWSBenchRunnerExtra/session",
        "arn:aws:iam::111122223333:role/service/AWSBenchRunner",
        "arn:aws:sts::111122223333:assumed-role/AWSBenchRunner/",
    ],
)
def test_preexisting_chain_does_not_reuse_a_different_ambient_identity(
    tmp_path, monkeypatch, real_provider, arn
):
    _activate(tmp_path, monkeypatch, "service/AWSBenchRunner")
    provider, identity, calls, _, _ = real_provider
    identity["Arn"] = arn
    session = provider.get_chained_session_for_account(
        "111122223333", ORG_ACCESS_ROLE, "app-session"
    )
    session_to_credential_process(session)
    assert len(calls) == 1
    assert calls[0][0]["RoleArn"] == "arn:aws:iam::111122223333:role/service/AWSBenchRunner"


@pytest.mark.parametrize("method", ["get_session_for_account", "get_chained_session_for_account"])
def test_preexisting_session_apis_enforce_allowlist_before_sts(
    tmp_path, monkeypatch, real_provider, method
):
    _activate(tmp_path, monkeypatch)
    provider, _, calls, _, _ = real_provider
    with (
        patch.object(BaseClient, "_make_api_call", side_effect=AssertionError("STS reached")),
        pytest.raises(AccountResolutionError, match="allowlist"),
    ):
        getattr(provider, method)("999988887777", "TaskRole", "app-session")
    assert calls == []


def test_static_api_reuses_ambient_role_with_a_configured_path(
    tmp_path, monkeypatch, real_provider
):
    _activate(tmp_path, monkeypatch, "service/AWSBenchRunner")
    provider, identity, calls, _, refreshes = real_provider
    identity["Arn"] = "arn:aws:sts::111122223333:assumed-role/AWSBenchRunner/host-session"
    result = provider.chain_assume_role("111122223333", "app-session")
    assert result == {
        "AWS_ACCESS_KEY_ID": "AMBIENT_RENEWED_KEY",
        "AWS_SECRET_ACCESS_KEY": "AMBIENT_RENEWED_SECRET",
        "AWS_SESSION_TOKEN": "AMBIENT_RENEWED_TOKEN",
    }
    assert calls == []
    assert refreshes == [True]


def test_preexisting_chain_retains_roles_and_renews_every_hop(tmp_path, monkeypatch, real_provider):
    _activate(tmp_path, monkeypatch, "service/AWSBenchRunner")
    provider, _, calls, signers, refreshes = real_provider
    session = provider.get_chained_session_for_account(
        "111122223333", "service/TaskRole", "app-session"
    )
    session_to_credential_process(session)
    _activate(tmp_path, monkeypatch, "DifferentRunner")
    for creds in [*signers, session.get_credentials()]:
        creds._expiry_time = datetime.now(timezone.utc) - timedelta(seconds=1)
    result = session_to_credential_process(session)
    assert result["AccessKeyId"] == "ROLE_4_KEY"
    assert refreshes == [True]
    assert [parent for _, parent in calls] == [
        "AMBIENT_KEY",
        "ROLE_1_KEY",
        "AMBIENT_RENEWED_KEY",
        "ROLE_3_KEY",
    ]
    assert [params["RoleArn"] for params, _ in calls[:2]] == [
        params["RoleArn"] for params, _ in calls[2:]
    ]


def test_mint_rejects_ambient_static_session_without_expiry(tmp_path, monkeypatch):
    _activate(tmp_path, monkeypatch)
    session = boto3.Session(
        aws_access_key_id="STATIC_KEY",
        aws_secret_access_key="STATIC_SECRET",
        aws_session_token="STATIC_TOKEN",
        region_name="us-east-1",
    )
    provider = CredentialProvider(session)

    def call(client, operation, params):
        assert operation == "GetCallerIdentity"
        return {
            "Account": "111122223333",
            "Arn": "arn:aws:sts::111122223333:assumed-role/AWSBenchRunner/host-session",
        }

    monkeypatch.setattr(BaseClient, "_make_api_call", call)
    with pytest.raises(CredentialError, match="no expiry"):
        mint_credentials(provider, {"PRIMARY": "111122223333"}, ORG_ACCESS_ROLE, "app-session")


def test_mint_preexisting_ambient_credentials_renews_same_identity(
    tmp_path, monkeypatch, real_provider
):
    _activate(tmp_path, monkeypatch, "service/AWSBenchRunner")
    provider, identity, calls, _, refreshes = real_provider
    identity["Arn"] = "arn:aws:sts::111122223333:assumed-role/AWSBenchRunner/host-session"
    first, _ = mint_credentials(
        provider, {"PRIMARY": "111122223333"}, ORG_ACCESS_ROLE, "app-session"
    )
    provider.session.get_credentials()._expiry_time = datetime.now(timezone.utc)
    second, _ = mint_credentials(
        provider, {"PRIMARY": "111122223333"}, ORG_ACCESS_ROLE, "app-session"
    )
    assert json.loads(first["PRIMARY.json"])["AccessKeyId"] == "AMBIENT_KEY"
    assert json.loads(second["PRIMARY.json"])["AccessKeyId"] == "AMBIENT_RENEWED_KEY"
    assert calls == []
    assert refreshes == [True]


@pytest.mark.parametrize("failed_role", ["service/AWSBenchRunner", "service/TaskRole"])
def test_preexisting_chain_failure_never_falls_back_to_ambient_keys(
    tmp_path, monkeypatch, real_provider, failed_role
):
    _activate(tmp_path, monkeypatch, "service/AWSBenchRunner")
    provider, _, calls, _, _ = real_provider
    original = BaseClient._make_api_call
    error = ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "AssumeRole")

    def call(client, operation, params):
        if params.get("RoleArn") == f"arn:aws:iam::111122223333:role/{failed_role}":
            # Resolve the parent exactly as the real STS signer does.
            client._request_signer._credentials.get_frozen_credentials()
            raise error
        return original(client, operation, params)

    monkeypatch.setattr(BaseClient, "_make_api_call", call)
    with pytest.raises(CredentialError):
        mint_credentials(provider, {"PRIMARY": "111122223333"}, "service/TaskRole", "app-session")
    assert len(calls) == (1 if failed_role == "service/TaskRole" else 0)
