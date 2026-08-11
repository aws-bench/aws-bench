"""Tests for the IAM role cleanup (prepare) handler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handlers.iam import _delete, _prepare
from aws_bench.resource_management.cleanup.models import HandlerStatus


def _session_with_iam() -> tuple[MagicMock, MagicMock]:
    session = MagicMock()
    iam = MagicMock()
    # NoSuchEntityException must be a real exception class for except clauses.
    iam.exceptions.NoSuchEntityException = type("NoSuchEntityException", (Exception,), {})
    session.client.return_value = iam
    return session, iam


def _role(name: str) -> Resource:
    return Resource(type="AWS::IAM::Role", identifier=name)


def test_prepare_skips_service_linked_role():
    session, iam = _session_with_iam()
    result = _prepare(_role("AWSServiceRoleForAutoScaling"), session)
    assert result.status == HandlerStatus.SKIPPED
    session.client.assert_not_called()


def test_prepare_skips_protected_role():
    session, iam = _session_with_iam()
    result = _prepare(_role("OrganizationAccountAccessRole"), session)
    assert result.status == HandlerStatus.SKIPPED


def test_prepare_detaches_and_removes_from_instance_profiles():
    session, iam = _session_with_iam()

    def paginator(op: str) -> MagicMock:
        pages = {
            "list_attached_role_policies": [
                {"AttachedPolicies": [{"PolicyArn": "arn:aws:iam::aws:policy/Foo"}]}
            ],
            "list_role_policies": [{"PolicyNames": ["inline1"]}],
            "list_instance_profiles_for_role": [
                {"InstanceProfiles": [{"InstanceProfileName": "profile-1"}]}
            ],
        }
        p = MagicMock()
        p.paginate.return_value = pages[op]
        return p

    iam.get_paginator.side_effect = paginator
    result = _prepare(_role("EC2ImageBuilderRole"), session)

    iam.detach_role_policy.assert_called_once_with(
        RoleName="EC2ImageBuilderRole", PolicyArn="arn:aws:iam::aws:policy/Foo"
    )
    iam.delete_role_policy.assert_called_once_with(
        RoleName="EC2ImageBuilderRole", PolicyName="inline1"
    )
    iam.remove_role_from_instance_profile.assert_called_once_with(
        RoleName="EC2ImageBuilderRole", InstanceProfileName="profile-1"
    )
    assert result.status == HandlerStatus.SUCCESS


def test_prepare_skips_when_role_not_found():
    session, iam = _session_with_iam()
    iam.get_paginator.side_effect = iam.exceptions.NoSuchEntityException()
    result = _prepare(_role("gone-role"), session)
    assert result.status == HandlerStatus.SKIPPED


def test_prepare_fails_on_other_error():
    session, iam = _session_with_iam()
    iam.get_paginator.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied"}}, "ListAttachedRolePolicies"
    )
    result = _prepare(_role("EC2SSMRole"), session)
    assert result.status == HandlerStatus.FAILED


class TestDelete:
    def test_deletes_role_successfully(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        result = _delete(_role("my-role"), session)
        client.delete_role.assert_called_once_with(RoleName="my-role")
        assert result.status == HandlerStatus.SUCCESS

    def test_skips_when_role_not_found(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_role.side_effect = ClientError(
            {"Error": {"Code": "NoSuchEntity", "Message": "not found"}}, "DeleteRole"
        )
        result = _delete(_role("gone-role"), session)
        assert result.status == HandlerStatus.SKIPPED

    def test_retries_on_delete_conflict(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        # First call: DeleteConflict, second call: success
        client.delete_role.side_effect = [
            ClientError({"Error": {"Code": "DeleteConflict", "Message": "in use"}}, "DeleteRole"),
            None,  # success
        ]
        with patch("aws_bench.resource_management.cleanup.handlers.iam.time.sleep") as mock_sleep:
            result = _delete(_role("busy-role"), session)
        assert result.status == HandlerStatus.SUCCESS
        assert client.delete_role.call_count == 2
        mock_sleep.assert_called_once_with(15)

    def test_fails_after_max_retries_on_delete_conflict(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        # All 3 attempts fail with DeleteConflict
        client.delete_role.side_effect = ClientError(
            {"Error": {"Code": "DeleteConflict", "Message": "still in use"}}, "DeleteRole"
        )
        with patch("aws_bench.resource_management.cleanup.handlers.iam.time.sleep"):
            result = _delete(_role("stuck-role"), session)
        assert result.status == HandlerStatus.FAILED
        assert "DeleteConflict" in result.message or "Failed to delete" in result.message
        assert client.delete_role.call_count == 3

    def test_fails_on_access_denied(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_role.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "nope"}}, "DeleteRole"
        )
        result = _delete(_role("locked-role"), session)
        assert result.status == HandlerStatus.FAILED
        assert client.delete_role.call_count == 1  # no retry for non-conflict errors

    def test_skips_service_linked_role(self):
        result = _delete(_role("AWSServiceRoleForECS"), MagicMock())
        assert result.status == HandlerStatus.SKIPPED

    def test_skips_protected_role(self):
        result = _delete(_role("OrganizationAccountAccessRole"), MagicMock())
        assert result.status == HandlerStatus.SKIPPED

    def test_fails_on_connection_error(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.delete_role.side_effect = BotoCoreError()
        result = _delete(_role("broken-role"), session)
        assert result.status == HandlerStatus.FAILED
