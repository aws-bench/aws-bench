"""Tests for aws_bench.task.aws_creds — credential + env helpers for AWS trials."""

from __future__ import annotations

from uuid import UUID

from aws_bench.dataset.models import RoleType
from aws_bench.task import aws_creds


def test_session_name_includes_job_id_when_set():
    name = aws_creds.session_name(job_id=UUID(int=1))
    # Neutral app-session-<job>: no task name or role type leaks to CloudTrail.
    assert name == f"app-session-{UUID(int=1)}"


def test_session_name_omits_job_id_when_none():
    name = aws_creds.session_name(job_id=None)
    assert name == "app-session"


def test_resolve_env_with_creds_substitutes_then_appends_creds():
    env = aws_creds.resolve_env_with_creds(
        raw_env={"REGION": "us-east-1", "BUCKET": "{{BucketName}}"},
        placeholders={"PRIMARY": {"BucketName": "my-bucket"}},
        creds={"AWS_ACCESS_KEY_ID": "AKIA"},
    )
    assert env["REGION"] == "us-east-1"
    assert env["BUCKET"] == "my-bucket"
    assert env["AWS_ACCESS_KEY_ID"] == "AKIA"


def test_resolve_env_with_creds_creds_win_on_conflict():
    """Credentials are applied last, so they override conflicting keys."""
    env = aws_creds.resolve_env_with_creds(
        raw_env={"AWS_ACCESS_KEY_ID": "from-task"},
        placeholders={},
        creds={"AWS_ACCESS_KEY_ID": "from-creds"},
    )
    assert env["AWS_ACCESS_KEY_ID"] == "from-creds"


def test_assume_role_for_script_uses_named_role(mocker):
    cp = mocker.patch.object(aws_creds, "CredentialProvider", autospec=True)
    # assume_role_for_script goes through the CredentialProvider.get() singleton.
    chain = cp.get.return_value.chain_assume_role
    chain.return_value = {"AWS_ACCESS_KEY_ID": "AKIA"}

    creds = aws_creds.assume_role_for_script(
        account_id="123456789012",
        role_name="MyAgentRole",
        role_type=RoleType.AGENT,
        task_name="org/t",
        job_id=None,
    )
    assert creds == {"AWS_ACCESS_KEY_ID": "AKIA"}
    chain.assert_called_once()
    kwargs = chain.call_args.kwargs
    assert kwargs["account_id"] == "123456789012"
    assert kwargs["role_name"] == "MyAgentRole"


def test_assume_role_for_script_falls_back_to_org_access_role(mocker):
    """A missing role_name falls back to the org access role."""
    cp = mocker.patch.object(aws_creds, "CredentialProvider", autospec=True)
    chain = cp.get.return_value.chain_assume_role
    chain.return_value = {}
    from aws_bench.account_management.constants import ORG_ACCESS_ROLE

    aws_creds.assume_role_for_script(
        account_id="123456789012",
        role_name=None,
        role_type=RoleType.PRE_INVOKE,
        task_name="org/t",
        job_id=None,
    )
    assert chain.call_args.kwargs["role_name"] == ORG_ACCESS_ROLE


def test_session_name_is_neutral_and_within_sts_limit():
    """The name leaks no task/role identity and stays within STS's 64-char cap."""
    name = aws_creds.session_name(job_id=UUID(int=1))
    assert name == "app-session-00000000-0000-0000-0000-000000000001"
    assert len(name) <= 64
    # STS charset: [\w+=,.@-]. '/' must never appear.
    assert "/" not in name
