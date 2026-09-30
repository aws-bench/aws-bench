"""Tests for aws_bench.account_management.organizations."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from tenacity import wait_none

from aws_bench.account_management.exceptions import (
    AccountCreationError,
    NotManagementAccountError,
    OrganizationNotReadyError,
)
from aws_bench.account_management.models import OrgInfo
from aws_bench.account_management.organizations import OrganizationsClient


@pytest.fixture()
def org_client():
    """Create an OrganizationsClient with mocked boto3 and credentials."""
    with patch("aws_bench.account_management.organizations.CredentialProvider") as mock_cred_cls:
        mock_cred = MagicMock()
        mock_cred_cls.get.return_value = mock_cred
        mock_boto = MagicMock()
        mock_cred.session.client.return_value = mock_boto
        mock_cred.get_caller_account_id.return_value = "111111111111"

        client = OrganizationsClient()
        yield client, mock_boto


# ── get_org_info ──


def test_get_org_info_returns_org_info(org_client):
    """Returns OrgInfo with correct fields from the API response."""
    client, mock_boto = org_client
    mock_boto.describe_organization.return_value = {
        "Organization": {
            "Id": "o-abc",
            "MasterAccountId": "111111111111",
            "MasterAccountEmail": "mgmt@example.com",
        }
    }
    mock_boto.list_roots.return_value = {"Roots": [{"Id": "r-root1"}]}

    info = client.get_org_info()
    assert isinstance(info, OrgInfo)
    assert info.org_id == "o-abc"
    assert info.root_id == "r-root1"


def test_get_org_info_raises_when_no_roots(org_client):
    """Raises OrganizationNotReadyError when no roots exist."""
    client, mock_boto = org_client
    mock_boto.describe_organization.return_value = {
        "Organization": {"Id": "o-abc", "MasterAccountId": "111", "MasterAccountEmail": "m@e"}
    }
    mock_boto.list_roots.return_value = {"Roots": []}

    with pytest.raises(OrganizationNotReadyError):
        client.get_org_info()


# ── create_organization ──


def test_create_organization_successfully(org_client):
    """Creates organization with ALL features."""
    client, mock_boto = org_client
    mock_boto.create_organization.return_value = {}
    client.create_organization()
    mock_boto.create_organization.assert_called_once_with(FeatureSet="ALL")


def test_create_organization_idempotent_when_already_exists(org_client):
    """Does not raise when organization already exists and caller is management account."""
    client, mock_boto = org_client
    error_response = {"Error": {"Code": "AlreadyInOrganizationException", "Message": ""}}
    mock_boto.create_organization.side_effect = ClientError(error_response, "CreateOrganization")
    mock_boto.describe_organization.return_value = {
        "Organization": {"Id": "o-abc", "MasterAccountId": "111111111111"}
    }

    client.create_organization()  # should not raise


def test_create_organization_raises_not_management_account(org_client):
    """Raises NotManagementAccountError when caller is not the management account."""
    client, mock_boto = org_client
    error_response = {"Error": {"Code": "AlreadyInOrganizationException", "Message": ""}}
    mock_boto.create_organization.side_effect = ClientError(error_response, "CreateOrganization")
    mock_boto.describe_organization.return_value = {
        "Organization": {"Id": "o-abc", "MasterAccountId": "999999999999"}
    }

    with pytest.raises(NotManagementAccountError):
        client.create_organization()


def test_create_organization_reraises_unexpected_client_error(org_client):
    """Re-raises unexpected ClientErrors."""
    client, mock_boto = org_client
    error_response = {"Error": {"Code": "ServiceException", "Message": "boom"}}
    mock_boto.create_organization.side_effect = ClientError(error_response, "CreateOrganization")

    with pytest.raises(ClientError):
        client.create_organization()


# ── find_ou_by_name ──


def test_find_ou_by_name_finds_existing(org_client):
    """Returns OU ID when a matching OU exists."""
    client, mock_boto = org_client
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"OrganizationalUnits": [{"Id": "ou-123", "Name": "test-env"}]}
    ]
    mock_boto.get_paginator.return_value = paginator

    assert client.find_ou_by_name("r-root1", "test-env") == "ou-123"


def test_find_ou_by_name_returns_none_when_not_found(org_client):
    """Returns None when no matching OU exists."""
    client, mock_boto = org_client
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"OrganizationalUnits": [{"Id": "ou-123", "Name": "other-env"}]}
    ]
    mock_boto.get_paginator.return_value = paginator

    assert client.find_ou_by_name("r-root1", "test-env") is None


# ── create_ou ──


def test_create_ou_returns_ou_id(org_client):
    """Creates an OU and returns its ID."""
    client, mock_boto = org_client
    mock_boto.create_organizational_unit.return_value = {"OrganizationalUnit": {"Id": "ou-new"}}

    assert client.create_ou("r-root1", "my-env") == "ou-new"
    mock_boto.create_organizational_unit.assert_called_once_with(ParentId="r-root1", Name="my-env")


# ── list_accounts_in_ou ──


def test_list_accounts_in_ou_returns_across_pages(org_client):
    """Aggregates accounts from multiple paginated responses."""
    client, mock_boto = org_client
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Accounts": [{"Id": "111", "Status": "ACTIVE"}]},
        {"Accounts": [{"Id": "222", "Status": "ACTIVE"}]},
    ]
    mock_boto.get_paginator.return_value = paginator

    assert len(client.list_accounts_in_ou("ou-123")) == 2


# ── get_tags ──


def test_get_tags_returns_dict(org_client):
    """Returns tags as a key-value dict."""
    client, mock_boto = org_client
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Tags": [{"Key": "Env", "Value": "test"}, {"Key": "Team", "Value": "bench"}]}
    ]
    mock_boto.get_paginator.return_value = paginator

    assert client.get_tags("acct-123") == {"Env": "test", "Team": "bench"}


def test_get_tags_returns_across_pages(org_client):
    """Aggregates tags from multiple paginated responses."""
    client, mock_boto = org_client
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Tags": [{"Key": "Env", "Value": "test"}]},
        {"Tags": [{"Key": "Team", "Value": "bench"}]},
    ]
    mock_boto.get_paginator.return_value = paginator

    assert client.get_tags("acct-123") == {"Env": "test", "Team": "bench"}


# ── create_account (async) ──


def test_create_account_successfully(org_client):
    """Creates an account and returns the account ID on success."""
    client, mock_boto = org_client
    mock_boto.create_account.return_value = {"CreateAccountStatus": {"Id": "req-1"}}
    mock_boto.describe_create_account_status.return_value = {
        "CreateAccountStatus": {"State": "SUCCEEDED", "AccountId": "222222222222"}
    }

    assert asyncio.run(client.create_account("agent", "agent@example.com")) == "222222222222"


def test_create_account_raises_on_failure(org_client):
    """Raises AccountCreationError when account creation fails."""
    client, mock_boto = org_client
    mock_boto.create_account.return_value = {"CreateAccountStatus": {"Id": "req-1"}}
    mock_boto.describe_create_account_status.return_value = {
        "CreateAccountStatus": {"State": "FAILED", "FailureReason": "EMAIL_ALREADY_EXISTS"}
    }

    with pytest.raises(AccountCreationError, match="EMAIL_ALREADY_EXISTS"):
        asyncio.run(client.create_account("agent", "agent@example.com"))


def test_create_account_retries_submission_on_concurrent_modification(org_client, monkeypatch):
    """Retries only the CreateAccount submission on ConcurrentModificationException."""
    client, mock_boto = org_client
    conflict = ClientError(
        {"Error": {"Code": "ConcurrentModificationException", "Message": "busy"}},
        "CreateAccount",
    )
    mock_boto.create_account.side_effect = [
        conflict,
        {"CreateAccountStatus": {"Id": "req-2"}},
    ]
    mock_boto.describe_create_account_status.return_value = {
        "CreateAccountStatus": {"State": "SUCCEEDED", "AccountId": "222222222222"}
    }
    monkeypatch.setattr(client._submit_create_account.retry, "wait", wait_none())

    assert asyncio.run(client.create_account("agent", "agent@example.com")) == "222222222222"

    assert mock_boto.create_account.call_count == 2
    mock_boto.describe_create_account_status.assert_called_once_with(CreateAccountRequestId="req-2")


def test_create_account_does_not_resubmit_when_polling_has_conflict(org_client):
    """Polling-side ClientErrors must not re-enter the CreateAccount submission."""
    client, mock_boto = org_client
    mock_boto.create_account.return_value = {"CreateAccountStatus": {"Id": "req-1"}}
    mock_boto.describe_create_account_status.side_effect = ClientError(
        {"Error": {"Code": "ConcurrentModificationException", "Message": "busy"}},
        "DescribeCreateAccountStatus",
    )

    with pytest.raises(ClientError):
        asyncio.run(client.create_account("agent", "agent@example.com"))

    assert mock_boto.create_account.call_count == 1
    mock_boto.describe_create_account_status.assert_called_once_with(CreateAccountRequestId="req-1")


# ── move_account_to_ou (async) ──


def test_move_account_to_ou_moves_account(org_client):
    """Moves account from current parent to target OU."""
    client, mock_boto = org_client
    mock_boto.list_parents.return_value = {"Parents": [{"Id": "r-root1"}]}

    asyncio.run(client.move_account_to_ou("222", "ou-target"))

    mock_boto.move_account.assert_called_once_with(
        AccountId="222", SourceParentId="r-root1", DestinationParentId="ou-target"
    )


def test_move_account_to_ou_skips_when_already_in_ou(org_client):
    """Skips move when account is already in the target OU."""
    client, mock_boto = org_client
    mock_boto.list_parents.return_value = {"Parents": [{"Id": "ou-target"}]}

    asyncio.run(client.move_account_to_ou("222", "ou-target"))

    mock_boto.move_account.assert_not_called()


def test_move_account_to_ou_retries_on_concurrent_modification(org_client, monkeypatch):
    """Retries MoveAccount when AWS Organizations raises ConcurrentModificationException."""
    client, mock_boto = org_client
    mock_boto.list_parents.return_value = {"Parents": [{"Id": "r-root1"}]}
    conflict = ClientError(
        {"Error": {"Code": "ConcurrentModificationException", "Message": "busy"}},
        "MoveAccount",
    )
    mock_boto.move_account.side_effect = [conflict, None]

    monkeypatch.setattr(client.move_account_to_ou.retry, "wait", wait_none())
    asyncio.run(client.move_account_to_ou("222", "ou-target"))

    assert mock_boto.move_account.call_count == 2


# ── tag_resource (async) ──


def test_tag_resource_tags_resource(org_client):
    """Tags a resource with the given key-value pair."""
    client, mock_boto = org_client
    asyncio.run(client.tag_resource("222", "Env", "test"))

    mock_boto.tag_resource.assert_called_once_with(
        ResourceId="222", Tags=[{"Key": "Env", "Value": "test"}]
    )


# ── SCP Protection Tests ──


def test_ensure_scp_creates_and_attaches(org_client):
    """Creates SCP and attaches to OU when neither exists."""
    client, mock_boto = org_client
    mock_boto.get_paginator.return_value.paginate.return_value = [{"Policies": []}]
    mock_boto.create_policy.return_value = {"Policy": {"PolicySummary": {"Id": "p-new123"}}}
    mock_boto.list_targets_for_policy.return_value = {"Targets": []}
    mock_boto.enable_policy_type.return_value = {}
    mock_boto.describe_organization.return_value = {
        "Organization": {
            "Id": "o-abc",
            "MasterAccountId": "111111111111",
            "MasterAccountEmail": "mgmt@example.com",
        }
    }
    mock_boto.list_roots.return_value = {
        "Roots": [
            {
                "Id": "r-root1",
                "PolicyTypes": [{"Type": "SERVICE_CONTROL_POLICY", "Status": "ENABLED"}],
            }
        ]
    }

    client.ensure_org_role_protection_scp("ou-test")

    mock_boto.create_policy.assert_called_once()
    mock_boto.attach_policy.assert_called_once_with(PolicyId="p-new123", TargetId="ou-test")


def test_ensure_scp_reuses_existing_policy(org_client):
    """Reuses existing SCP by name, reconciles content, and attaches to OU."""
    from aws_bench.account_management.organizations import OrganizationsClient

    client, mock_boto = org_client
    mock_boto.get_paginator.return_value.paginate.return_value = [
        {"Policies": [{"Name": "awsbench-protect-org-access-role", "Id": "p-exist"}]}
    ]
    # Return current content matching the desired state (no update needed)
    desired = OrganizationsClient._build_role_protection_policy("OrganizationAccountAccessRole")
    mock_boto.describe_policy.return_value = {"Policy": {"Content": desired}}
    mock_boto.list_targets_for_policy.return_value = {"Targets": []}
    mock_boto.enable_policy_type.side_effect = ClientError(
        {"Error": {"Code": "PolicyTypeAlreadyEnabledException", "Message": ""}}, "EnablePolicyType"
    )
    mock_boto.describe_organization.return_value = {
        "Organization": {
            "Id": "o-abc",
            "MasterAccountId": "111111111111",
            "MasterAccountEmail": "mgmt@example.com",
        }
    }
    mock_boto.list_roots.return_value = {
        "Roots": [
            {
                "Id": "r-root1",
                "PolicyTypes": [{"Type": "SERVICE_CONTROL_POLICY", "Status": "ENABLED"}],
            }
        ]
    }

    client.ensure_org_role_protection_scp("ou-test")

    mock_boto.create_policy.assert_not_called()
    mock_boto.update_policy.assert_not_called()
    mock_boto.attach_policy.assert_called_once_with(PolicyId="p-exist", TargetId="ou-test")


def test_ensure_scp_skips_when_already_attached(org_client):
    """No-op when SCP already attached to the OU."""
    from aws_bench.account_management.organizations import OrganizationsClient

    client, mock_boto = org_client
    mock_boto.get_paginator.return_value.paginate.return_value = [
        {"Policies": [{"Name": "awsbench-protect-org-access-role", "Id": "p-exist"}]}
    ]
    desired = OrganizationsClient._build_role_protection_policy("OrganizationAccountAccessRole")
    mock_boto.describe_policy.return_value = {"Policy": {"Content": desired}}
    mock_boto.list_targets_for_policy.return_value = {
        "Targets": [{"TargetId": "ou-test", "Type": "ORGANIZATIONAL_UNIT"}]
    }
    mock_boto.enable_policy_type.side_effect = ClientError(
        {"Error": {"Code": "PolicyTypeAlreadyEnabledException", "Message": ""}}, "EnablePolicyType"
    )
    mock_boto.describe_organization.return_value = {
        "Organization": {
            "Id": "o-abc",
            "MasterAccountId": "111111111111",
            "MasterAccountEmail": "mgmt@example.com",
        }
    }
    mock_boto.list_roots.return_value = {
        "Roots": [
            {
                "Id": "r-root1",
                "PolicyTypes": [{"Type": "SERVICE_CONTROL_POLICY", "Status": "ENABLED"}],
            }
        ]
    }

    client.ensure_org_role_protection_scp("ou-test")

    mock_boto.create_policy.assert_not_called()
    mock_boto.attach_policy.assert_not_called()


def test_ensure_scp_retries_on_policy_type_not_enabled(org_client):
    """Retries attach_policy when PolicyTypeNotEnabledException indicates propagation delay."""
    from aws_bench.account_management.organizations import OrganizationsClient

    client, mock_boto = org_client
    mock_boto.get_paginator.return_value.paginate.return_value = [
        {"Policies": [{"Name": "awsbench-protect-org-access-role", "Id": "p-exist"}]}
    ]
    desired = OrganizationsClient._build_role_protection_policy("OrganizationAccountAccessRole")
    mock_boto.describe_policy.return_value = {"Policy": {"Content": desired}}
    mock_boto.list_targets_for_policy.return_value = {"Targets": []}
    mock_boto.enable_policy_type.side_effect = ClientError(
        {"Error": {"Code": "PolicyTypeAlreadyEnabledException", "Message": ""}}, "EnablePolicyType"
    )
    mock_boto.describe_organization.return_value = {
        "Organization": {
            "Id": "o-abc",
            "MasterAccountId": "111111111111",
            "MasterAccountEmail": "mgmt@example.com",
        }
    }
    mock_boto.list_roots.return_value = {
        "Roots": [
            {
                "Id": "r-root1",
                "PolicyTypes": [{"Type": "SERVICE_CONTROL_POLICY", "Status": "ENABLED"}],
            }
        ]
    }

    not_enabled_err = ClientError(
        {"Error": {"Code": "PolicyTypeNotEnabledException", "Message": "Not enabled yet"}},
        "AttachPolicy",
    )
    mock_boto.attach_policy.side_effect = [not_enabled_err, not_enabled_err, None]

    client.ensure_org_role_protection_scp("ou-test")

    assert mock_boto.attach_policy.call_count == 3


# ── Region Restriction SCP Tests ──


def test_ensure_region_restriction_scp_creates_and_attaches_to_accounts(org_client):
    """Creates per-scenario SCP and attaches to each account."""
    client, mock_boto = org_client
    mock_boto.get_paginator.return_value.paginate.return_value = [{"Policies": []}]
    mock_boto.create_policy.return_value = {"Policy": {"PolicySummary": {"Id": "p-reg123"}}}
    mock_boto.list_targets_for_policy.return_value = {"Targets": []}
    mock_boto.enable_policy_type.return_value = {}
    mock_boto.describe_organization.return_value = {
        "Organization": {
            "Id": "o-abc",
            "MasterAccountId": "111111111111",
            "MasterAccountEmail": "mgmt@example.com",
        }
    }
    mock_boto.list_roots.return_value = {
        "Roots": [
            {
                "Id": "r-root1",
                "PolicyTypes": [{"Type": "SERVICE_CONTROL_POLICY", "Status": "ENABLED"}],
            }
        ]
    }

    client.ensure_region_restriction_scp("my-scenario", ["us-east-1"], ["111", "222"])

    mock_boto.create_policy.assert_called_once()
    call_args = mock_boto.create_policy.call_args
    assert call_args.kwargs["Name"] == "awsbench-region-restrict-my-scenario"
    assert mock_boto.attach_policy.call_count == 2
    mock_boto.attach_policy.assert_any_call(PolicyId="p-reg123", TargetId="111")
    mock_boto.attach_policy.assert_any_call(PolicyId="p-reg123", TargetId="222")
    mock_boto.detach_policy.assert_not_called()


def test_ensure_region_restriction_scp_reuses_existing_policy(org_client):
    """Reuses existing per-scenario SCP and attaches to accounts."""
    client, mock_boto = org_client
    mock_boto.get_paginator.return_value.paginate.return_value = [
        {"Policies": [{"Name": "awsbench-region-restrict-my-scenario", "Id": "p-exist"}]}
    ]
    mock_boto.list_targets_for_policy.return_value = {"Targets": []}
    mock_boto.describe_policy.return_value = {
        "Policy": {"Content": client._build_region_restriction_policy(["us-east-1"])}
    }
    mock_boto.enable_policy_type.side_effect = ClientError(
        {"Error": {"Code": "PolicyTypeAlreadyEnabledException", "Message": ""}}, "EnablePolicyType"
    )
    mock_boto.describe_organization.return_value = {
        "Organization": {
            "Id": "o-abc",
            "MasterAccountId": "111111111111",
            "MasterAccountEmail": "mgmt@example.com",
        }
    }
    mock_boto.list_roots.return_value = {
        "Roots": [
            {
                "Id": "r-root1",
                "PolicyTypes": [{"Type": "SERVICE_CONTROL_POLICY", "Status": "ENABLED"}],
            }
        ]
    }

    client.ensure_region_restriction_scp("my-scenario", ["us-east-1"], ["333"])

    mock_boto.create_policy.assert_not_called()
    mock_boto.attach_policy.assert_called_once_with(PolicyId="p-exist", TargetId="333")


def test_ensure_region_restriction_scp_skips_already_attached_accounts(org_client):
    """Skips accounts that already have the SCP attached."""
    client, mock_boto = org_client
    mock_boto.get_paginator.return_value.paginate.return_value = [
        {"Policies": [{"Name": "awsbench-region-restrict-my-scenario", "Id": "p-exist"}]}
    ]
    mock_boto.list_targets_for_policy.return_value = {
        "Targets": [{"TargetId": "111", "Type": "ACCOUNT"}]
    }
    mock_boto.describe_policy.return_value = {
        "Policy": {"Content": client._build_region_restriction_policy(["us-east-1"])}
    }
    mock_boto.enable_policy_type.side_effect = ClientError(
        {"Error": {"Code": "PolicyTypeAlreadyEnabledException", "Message": ""}}, "EnablePolicyType"
    )
    mock_boto.describe_organization.return_value = {
        "Organization": {
            "Id": "o-abc",
            "MasterAccountId": "111111111111",
            "MasterAccountEmail": "mgmt@example.com",
        }
    }
    mock_boto.list_roots.return_value = {
        "Roots": [
            {
                "Id": "r-root1",
                "PolicyTypes": [{"Type": "SERVICE_CONTROL_POLICY", "Status": "ENABLED"}],
            }
        ]
    }

    client.ensure_region_restriction_scp("my-scenario", ["us-east-1"], ["111"])

    mock_boto.attach_policy.assert_not_called()


def test_ensure_region_restriction_scp_updates_when_regions_change(org_client):
    """Updates SCP content when allowed regions have changed."""
    client, mock_boto = org_client
    mock_boto.get_paginator.return_value.paginate.return_value = [
        {"Policies": [{"Name": "awsbench-region-restrict-my-scenario", "Id": "p-exist"}]}
    ]
    mock_boto.list_targets_for_policy.return_value = {
        "Targets": [{"TargetId": "111", "Type": "ACCOUNT"}]
    }
    mock_boto.describe_policy.return_value = {"Policy": {"Content": '{"old": "content"}'}}
    mock_boto.enable_policy_type.side_effect = ClientError(
        {"Error": {"Code": "PolicyTypeAlreadyEnabledException", "Message": ""}}, "EnablePolicyType"
    )
    mock_boto.describe_organization.return_value = {
        "Organization": {
            "Id": "o-abc",
            "MasterAccountId": "111111111111",
            "MasterAccountEmail": "mgmt@example.com",
        }
    }
    mock_boto.list_roots.return_value = {
        "Roots": [
            {
                "Id": "r-root1",
                "PolicyTypes": [{"Type": "SERVICE_CONTROL_POLICY", "Status": "ENABLED"}],
            }
        ]
    }

    client.ensure_region_restriction_scp("my-scenario", ["us-east-1", "eu-west-1"], ["111"])

    expected_content = client._build_region_restriction_policy(["us-east-1", "eu-west-1"])
    mock_boto.update_policy.assert_called_once_with(PolicyId="p-exist", Content=expected_content)


@pytest.mark.parametrize("has_account_exceptions", [False, True])
def test_ensure_region_restriction_scp_upgrades_old_policy_with_same_regions(
    org_client: tuple[OrganizationsClient, MagicMock],
    has_account_exceptions: bool,
) -> None:
    """Upgrade attached legacy policies in place without relaxing allowed regions."""
    client, mock_boto = org_client
    desired_content = client._build_region_restriction_policy(["eu-south-2"])
    old_policy = json.loads(desired_content)
    old_policy["Statement"] = old_policy["Statement"][:1]
    old_statement = old_policy["Statement"][0]
    if not has_account_exceptions:
        old_statement["NotAction"] = [
            action for action in old_statement["NotAction"] if not action.startswith("account:")
        ]
    mock_boto.get_paginator.return_value.paginate.return_value = [
        {"Policies": [{"Name": "awsbench-region-restrict-my-scenario", "Id": "p-exist"}]}
    ]
    mock_boto.describe_policy.return_value = {"Policy": {"Content": json.dumps(old_policy)}}
    mock_boto.list_targets_for_policy.return_value = {
        "Targets": [{"TargetId": "111", "Type": "ACCOUNT"}]
    }

    with patch.object(client, "_enable_scp_policy_type"):
        client.ensure_region_restriction_scp("my-scenario", ["eu-south-2"], ["111"])

    mock_boto.update_policy.assert_called_once_with(PolicyId="p-exist", Content=desired_content)
    updated_policy = json.loads(mock_boto.update_policy.call_args.kwargs["Content"])
    updated_statement = updated_policy["Statement"][0]
    added_exceptions = set(updated_statement["NotAction"]) - set(old_statement["NotAction"])
    assert added_exceptions == (
        set() if has_account_exceptions else {"account:GetRegionOptStatus", "account:EnableRegion"}
    )
    assert updated_policy["Statement"][1] == {
        "Sid": "DenyOutOfScopeRegionOptIn",
        "Effect": "Deny",
        "Action": "account:EnableRegion",
        "Resource": "*",
        "Condition": {"StringNotEquals": {"account:TargetRegion": ["eu-south-2"]}},
    }
    updated_statement["NotAction"] = old_statement["NotAction"]
    updated_policy["Statement"] = updated_policy["Statement"][:1]
    assert updated_policy == old_policy
    assert updated_statement["Condition"] == {
        "StringNotEquals": {"aws:RequestedRegion": ["eu-south-2"]}
    }
    mock_boto.create_policy.assert_not_called()
    mock_boto.attach_policy.assert_not_called()
    mock_boto.detach_policy.assert_not_called()


def test_ensure_region_restriction_scp_no_update_when_regions_reordered(org_client):
    """Reordering the same region set must not trigger a redundant update_policy."""
    client, mock_boto = org_client
    mock_boto.get_paginator.return_value.paginate.return_value = [
        {"Policies": [{"Name": "awsbench-region-restrict-my-scenario", "Id": "p-exist"}]}
    ]
    mock_boto.list_targets_for_policy.return_value = {
        "Targets": [{"TargetId": "111", "Type": "ACCOUNT"}]
    }
    # Stored policy was built from one order; the rerun passes the reverse order.
    mock_boto.describe_policy.return_value = {
        "Policy": {"Content": client._build_region_restriction_policy(["eu-west-1", "us-east-1"])}
    }
    mock_boto.enable_policy_type.side_effect = ClientError(
        {"Error": {"Code": "PolicyTypeAlreadyEnabledException", "Message": ""}}, "EnablePolicyType"
    )
    mock_boto.describe_organization.return_value = {
        "Organization": {
            "Id": "o-abc",
            "MasterAccountId": "111111111111",
            "MasterAccountEmail": "mgmt@example.com",
        }
    }
    mock_boto.list_roots.return_value = {
        "Roots": [
            {
                "Id": "r-root1",
                "PolicyTypes": [{"Type": "SERVICE_CONTROL_POLICY", "Status": "ENABLED"}],
            }
        ]
    }

    client.ensure_region_restriction_scp("my-scenario", ["us-east-1", "eu-west-1"], ["111"])

    mock_boto.update_policy.assert_not_called()


def test_build_region_restriction_policy_exempts_only_required_account_actions(
    org_client: tuple[OrganizationsClient, MagicMock],
) -> None:
    """Allow global region initialization calls without opening other account APIs."""
    client, _ = org_client
    policy = json.loads(client._build_region_restriction_policy(["eu-south-2"]))

    assert len(policy["Statement"]) == 2
    statement = policy["Statement"][0]
    exceptions = statement["NotAction"]
    assert {action for action in exceptions if action.startswith("account:")} == {
        "account:GetRegionOptStatus",
        "account:EnableRegion",
    }
    assert "account:*" not in exceptions
    assert "account:DisableRegion" not in exceptions
    assert "*" not in exceptions
    assert statement["Effect"] == "Deny"
    assert statement["Resource"] == "*"
    assert statement["Condition"] == {"StringNotEquals": {"aws:RequestedRegion": ["eu-south-2"]}}


def test_build_region_restriction_policy_structure(org_client):
    """Produces valid JSON with correct policy structure."""
    client, _ = org_client
    regions = ["us-west-2", "us-east-1"]
    result = client._build_region_restriction_policy(regions)

    policy = json.loads(result)
    assert policy["Version"] == "2012-10-17"
    assert len(policy["Statement"]) == 2

    stmt = policy["Statement"][0]
    assert stmt["Effect"] == "Deny"
    assert "NotAction" in stmt
    assert isinstance(stmt["NotAction"], list)
    assert stmt["Resource"] == "*"
    # Regions are canonicalized (sorted) regardless of input order.
    assert stmt["Condition"]["StringNotEquals"]["aws:RequestedRegion"] == ["us-east-1", "us-west-2"]

    # Target-region guard is independent of the global Account endpoint region.
    assert policy["Statement"][1] == {
        "Sid": "DenyOutOfScopeRegionOptIn",
        "Effect": "Deny",
        "Action": "account:EnableRegion",
        "Resource": "*",
        "Condition": {"StringNotEquals": {"account:TargetRegion": ["us-east-1", "us-west-2"]}},
    }
    assert all("Principal" not in statement for statement in policy["Statement"])
