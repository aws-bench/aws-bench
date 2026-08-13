"""Credential routing tests for externally owned accounts."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aws_bench.account_management.exceptions import AccountResolutionError
from aws_bench.account_management.preexisting import ACCOUNT_CONFIG_ENV_VAR
from aws_bench.utils.credentials_provider import CredentialProvider


def _activate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner_role: str = "AWSBenchRunner"
) -> None:
    path = tmp_path / "accounts.yaml"
    path.write_text(
        "mode: preexisting\n"
        "name: aws-bench\n"
        f"runner_role: {runner_role}\n"
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
            "111122223333", "OrganizationAccountAccessRole", "aws-bench-test"
        )
    create_session.assert_called_once_with(
        session,
        "arn:aws:iam::111122223333:role/AWSBenchRunner",
        "aws-bench-test",
        "us-east-1",
    )


def test_explicit_task_role_chains_from_configured_runner(tmp_path: Path, monkeypatch):
    _activate(tmp_path, monkeypatch, "AWSBenchRunner")
    provider, session = _provider()
    with patch(
        "aws_bench.utils.credentials_provider._create_refreshable_session"
    ) as create_session:
        provider.get_session_for_account("111122223333", "TaskRole", "aws-bench-task")
    assert create_session.call_count == 2
    runner_session = create_session.return_value
    assert create_session.call_args_list[0].args == (
        session,
        "arn:aws:iam::111122223333:role/AWSBenchRunner",
        "aws-bench-runner-223333",
        "us-east-1",
    )
    assert create_session.call_args_list[1].args == (
        runner_session,
        "arn:aws:iam::111122223333:role/TaskRole",
        "aws-bench-task",
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
            "111122223333", "OrganizationAccountAccessRole", "aws-bench-test"
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

    credentials = provider.chain_assume_role("111122223333", "aws-bench-task", role_name=None)

    provider.assume_role.assert_called_once_with(
        "111122223333", "AWSBenchRunner", "aws-bench-task", duration_seconds=3600
    )
    assert credentials["AWS_ACCESS_KEY_ID"] == "runner-key"


def test_account_outside_allowlist_is_refused(tmp_path: Path, monkeypatch):
    _activate(tmp_path, monkeypatch)
    provider, _ = _provider()
    with pytest.raises(AccountResolutionError, match="not in the active pre-existing allowlist"):
        provider.chain_assume_role("999988887777", "aws-bench-test")


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
        credentials = provider.chain_assume_role(
            "111122223333", "aws-bench-task", role_name="TaskRole"
        )
    provider.assume_role.assert_called_once_with(
        "111122223333",
        "AWSBenchRunner",
        "aws-bench-runner-223333",
        duration_seconds=3600,
    )
    runner_sts.assume_role.assert_called_once_with(
        RoleArn="arn:aws:iam::111122223333:role/TaskRole",
        RoleSessionName="aws-bench-task",
        DurationSeconds=3600,
    )
    assert credentials["AWS_ACCESS_KEY_ID"] == "task-key"
