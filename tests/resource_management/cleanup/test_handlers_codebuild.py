"""Tests for the CodeBuild project cleanup handler.

CCAPI cannot delete AWS::CodeBuild::Project (UnsupportedActionException), so the
custom handler deletes via the CodeBuild service API. The identifier is the
project name.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import (
    CUSTOM_DELETION_REGISTRY,
    PREPARE_REGISTRY,
)
from aws_bench.resource_management.cleanup.handlers.codebuild import (
    _delete_codebuild_project,
    _prepare_codebuild_project,
)
from aws_bench.resource_management.cleanup.models import HandlerStatus

_TYPE = "AWS::CodeBuild::Project"


def _resource(name: str = "agent-project") -> Resource:
    return Resource(type=_TYPE, identifier=name)


def _session_with(client: MagicMock) -> MagicMock:
    session = MagicMock()
    session.client.return_value = client
    return session


def _not_found(op: str) -> ClientError:
    return ClientError({"Error": {"Code": "ResourceNotFoundException", "Message": "gone"}}, op)


def test_handler_is_registered():
    assert _TYPE in CUSTOM_DELETION_REGISTRY
    assert _TYPE in PREPARE_REGISTRY


def test_delete_calls_delete_project_by_name():
    client = MagicMock()
    result = _delete_codebuild_project(_resource(), _session_with(client))
    assert result.status is HandlerStatus.SUCCESS
    client.delete_project.assert_called_once_with(name="agent-project")


def test_delete_already_gone_is_success():
    client = MagicMock()
    client.delete_project.side_effect = _not_found("DeleteProject")
    result = _delete_codebuild_project(_resource(), _session_with(client))
    assert result.status is HandlerStatus.SUCCESS


def test_prepare_removes_resource_policy():
    client = MagicMock()
    client.batch_get_projects.return_value = {
        "projects": [{"arn": "arn:aws:codebuild:us-east-1:111122223333:project/agent-project"}]
    }
    result = _prepare_codebuild_project(_resource(), _session_with(client))
    assert result.status is HandlerStatus.SUCCESS
    client.delete_resource_policy.assert_called_once_with(
        resourceArn="arn:aws:codebuild:us-east-1:111122223333:project/agent-project"
    )


def test_prepare_no_policy_is_clean_noop():
    client = MagicMock()
    client.batch_get_projects.return_value = {
        "projects": [{"arn": "arn:aws:codebuild:us-east-1:111122223333:project/agent-project"}]
    }
    # No resource policy attached -> delete_resource_policy raises not-found, swallowed.
    client.delete_resource_policy.side_effect = ClientError(
        {"Error": {"Code": "ResourceNotFoundException", "Message": "no policy"}},
        "DeleteResourcePolicy",
    )
    result = _prepare_codebuild_project(_resource(), _session_with(client))
    assert result.status is HandlerStatus.SUCCESS


def test_prepare_missing_project_is_success():
    client = MagicMock()
    client.batch_get_projects.return_value = {"projects": []}
    result = _prepare_codebuild_project(_resource(), _session_with(client))
    assert result.status is HandlerStatus.SUCCESS
    client.delete_resource_policy.assert_not_called()


def test_prepare_unexpected_error_fails():
    client = MagicMock()
    client.batch_get_projects.return_value = {
        "projects": [{"arn": "arn:aws:codebuild:us-east-1:111122223333:project/agent-project"}]
    }
    client.delete_resource_policy.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
        "DeleteResourcePolicy",
    )
    result = _prepare_codebuild_project(_resource(), _session_with(client))
    assert result.status is HandlerStatus.FAILED


def test_full_delete_via_registry_runs_prepare_then_delete():
    """The registered delete chains prepare first (clear policy), then deletes."""
    client = MagicMock()
    client.batch_get_projects.return_value = {"projects": [{"arn": "arn:...:project/p"}]}
    handler = CUSTOM_DELETION_REGISTRY[_TYPE]
    result = handler(_resource("p"), _session_with(client))
    assert result.status is HandlerStatus.SUCCESS
    client.delete_project.assert_called_once_with(name="p")
