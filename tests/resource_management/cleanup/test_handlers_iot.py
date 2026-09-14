"""Tests for the IoT Policy cleanup handler.

moto enforces IoT's detach-before-delete contract: ``delete_policy`` raises
``DeleteConflictException`` while the policy is attached to a target or has a
non-default version. A SUCCESS result therefore proves the handler detached
every target and removed non-default versions before deleting the policy.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError
from moto import mock_aws

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import CUSTOM_DELETION_REGISTRY
from aws_bench.resource_management.cleanup.handlers.iot import _delete_policy
from aws_bench.resource_management.cleanup.models import HandlerStatus

_REGION = "us-east-1"
_POLICY_NAME = "bench-policy"
_POLICY_ARN = f"arn:aws:iot:{_REGION}:123456789012:policy/{_POLICY_NAME}"
_POLICY_DOC = json.dumps(
    {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": "iot:Connect", "Resource": "*"}],
    }
)


def _resource(identifier: str = _POLICY_ARN) -> Resource:
    return Resource(type="AWS::IoT::Policy", identifier=identifier)


def _policy_exists(client: object, name: str) -> bool:
    try:
        client.get_policy(policyName=name)  # type: ignore[attr-defined]
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
            return False
        raise


def _not_found(op: str) -> ClientError:
    return ClientError({"Error": {"Code": "ResourceNotFoundException"}}, op)


def _mock_session(client: MagicMock) -> MagicMock:
    """A session whose ``client("iot")`` returns the given mock (via build_client)."""
    session = MagicMock()
    session.client.return_value = client
    return session


def _quiet_client() -> MagicMock:
    """A mock IoT client with no targets and no non-default versions to clean up."""
    client = MagicMock()
    client.get_paginator.return_value.paginate.return_value = [{"targets": []}]
    client.list_policy_versions.return_value = {"policyVersions": []}
    return client


# -- registration --


def test_handler_registered_for_iot_policy_type():
    """The delete handler must be registered so the scan does not fall through to CCAPI.

    CCAPI cannot delete ``AWS::IoT::Policy``; without this registration an
    agent-created policy leaks and fails reset (the bug this handler fixes).
    """
    import aws_bench.resource_management.cleanup.handlers  # noqa: F401

    assert "AWS::IoT::Policy" in CUSTOM_DELETION_REGISTRY


# -- _delete_policy --


@mock_aws
def test_delete_detaches_targets_then_deletes():
    """An attached policy is detached from every target, then deleted."""
    client = boto3.client("iot", region_name=_REGION)
    client.create_policy(policyName=_POLICY_NAME, policyDocument=_POLICY_DOC)
    cert_arn = client.create_keys_and_certificate(setAsActive=True)["certificateArn"]
    client.attach_policy(policyName=_POLICY_NAME, target=cert_arn)

    result = _delete_policy(_resource(), boto3.Session(region_name=_REGION))

    assert result.status == HandlerStatus.SUCCESS
    assert not _policy_exists(client, _POLICY_NAME)
    # Detaching a policy must not delete the target it was attached to.
    assert (
        client.describe_certificate(certificateId=cert_arn.rsplit("/", 1)[-1])[
            "certificateDescription"
        ]["certificateArn"]
        == cert_arn
    )


@mock_aws
def test_delete_removes_non_default_versions_then_deletes():
    """A policy with a non-default version has that version dropped, then is deleted."""
    client = boto3.client("iot", region_name=_REGION)
    client.create_policy(policyName=_POLICY_NAME, policyDocument=_POLICY_DOC)
    # Second version is non-default; delete_policy fails until it is removed.
    client.create_policy_version(
        policyName=_POLICY_NAME, policyDocument=_POLICY_DOC, setAsDefault=False
    )

    result = _delete_policy(_resource(), boto3.Session(region_name=_REGION))

    assert result.status == HandlerStatus.SUCCESS
    assert not _policy_exists(client, _POLICY_NAME)


@mock_aws
def test_delete_already_gone_is_idempotent_success():
    """A policy that no longer exists yields SUCCESS (idempotent)."""
    result = _delete_policy(_resource(), boto3.Session(region_name=_REGION))

    assert result.status == HandlerStatus.SUCCESS


@mock_aws
def test_delete_accepts_bare_policy_name_identifier():
    """The identifier may be a bare policy name rather than an ARN."""
    client = boto3.client("iot", region_name=_REGION)
    client.create_policy(policyName=_POLICY_NAME, policyDocument=_POLICY_DOC)

    result = _delete_policy(_resource(identifier=_POLICY_NAME), boto3.Session(region_name=_REGION))

    assert result.status == HandlerStatus.SUCCESS
    assert not _policy_exists(client, _POLICY_NAME)


# -- failure paths (mocked clients) --


def test_delete_reports_failure_on_client_error():
    """A non-not-found ClientError from delete_policy maps to FAILED, not SUCCESS."""
    client = _quiet_client()
    client.delete_policy.side_effect = ClientError(
        {"Error": {"Code": "InvalidRequestException"}}, "DeletePolicy"
    )

    result = _delete_policy(_resource(), _mock_session(client))

    assert result.status == HandlerStatus.FAILED


def test_delete_reports_failure_on_botocore_error():
    """A connection-level BotoCoreError maps to FAILED."""
    client = _quiet_client()
    client.delete_policy.side_effect = EndpointConnectionError(endpoint_url="https://iot")

    result = _delete_policy(_resource(), _mock_session(client))

    assert result.status == HandlerStatus.FAILED


def test_detach_skips_targets_that_vanish_mid_iteration():
    """A target that disappears between listing and detach is skipped, then delete succeeds."""
    client = _quiet_client()
    client.get_paginator.return_value.paginate.return_value = [{"targets": ["cert-arn"]}]
    client.detach_policy.side_effect = _not_found("DetachPolicy")

    result = _delete_policy(_resource(), _mock_session(client))

    client.delete_policy.assert_called_once_with(policyName=_POLICY_NAME)
    assert result.status == HandlerStatus.SUCCESS


def test_version_prune_skips_versions_that_vanish_mid_iteration():
    """A non-default version that disappears before deletion is skipped, then delete succeeds."""
    client = _quiet_client()
    client.list_policy_versions.return_value = {
        "policyVersions": [
            {"versionId": "1", "isDefaultVersion": True},
            {"versionId": "2", "isDefaultVersion": False},
        ]
    }
    client.delete_policy_version.side_effect = _not_found("DeletePolicyVersion")

    result = _delete_policy(_resource(), _mock_session(client))

    client.delete_policy.assert_called_once_with(policyName=_POLICY_NAME)
    assert result.status == HandlerStatus.SUCCESS


def test_detach_failure_surfaces_as_failed():
    """A non-not-found error while detaching aborts before delete and maps to FAILED."""
    client = _quiet_client()
    client.get_paginator.return_value.paginate.return_value = [{"targets": ["cert-arn"]}]
    client.detach_policy.side_effect = ClientError(
        {"Error": {"Code": "ThrottlingException"}}, "DetachPolicy"
    )

    result = _delete_policy(_resource(), _mock_session(client))

    client.delete_policy.assert_not_called()
    assert result.status == HandlerStatus.FAILED


def test_version_prune_failure_surfaces_as_failed():
    """A non-not-found error while pruning a version aborts before delete and maps to FAILED."""
    client = _quiet_client()
    client.list_policy_versions.return_value = {
        "policyVersions": [{"versionId": "2", "isDefaultVersion": False}]
    }
    client.delete_policy_version.side_effect = ClientError(
        {"Error": {"Code": "ThrottlingException"}}, "DeletePolicyVersion"
    )

    result = _delete_policy(_resource(), _mock_session(client))

    client.delete_policy.assert_not_called()
    assert result.status == HandlerStatus.FAILED
