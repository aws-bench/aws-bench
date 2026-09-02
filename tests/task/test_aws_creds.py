"""Tests for aws_bench.task.aws_creds — credential + env helpers for AWS trials."""

from __future__ import annotations

from uuid import UUID

import pytest

from aws_bench.dataset.models import RoleType
from aws_bench.task import aws_creds


def test_session_name_includes_job_id_when_set():
    name = aws_creds.session_name(
        task_name="org/my-task", role_type=RoleType.AGENT, job_id=UUID(int=1)
    )
    # '/' is replaced with '-' (STS session names allow only [\w+=,.@-]).
    assert "org-my-task" in name
    # Ordered app-<role>-<task>-<job>: role leads so a trim drops the job tail.
    assert name.startswith(f"app-{RoleType.AGENT}-")
    assert str(UUID(int=1)) in name


def test_session_name_omits_job_id_when_none():
    name = aws_creds.session_name(task_name="org/my-task", role_type=RoleType.VERIFIER, job_id=None)
    assert name == "app-verifier-org-my-task"


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


@pytest.mark.parametrize("role_type", list(RoleType))
def test_session_name_role_types(role_type):
    name = aws_creds.session_name(task_name="t", role_type=role_type, job_id=None)
    assert name.startswith(f"app-{role_type}-")


def test_session_name_overflow_trims_to_64_keeping_role_and_task():
    """A long task name + UUID exceeds 64 chars; the trim drops the job-id tail."""
    long_task = "some-org/" + "a" * 80
    name = aws_creds.session_name(
        task_name=long_task, role_type=RoleType.VERIFIER, job_id=UUID(int=1)
    )
    assert len(name) == 64
    # Role and the (start of the) task survive the trim; the job-id tail is cut.
    assert name.startswith("app-verifier-some-org-aaa")
    # STS charset: [\w+=,.@-]. '/' must have been replaced.
    assert "/" not in name
