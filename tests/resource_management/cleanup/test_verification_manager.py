"""Tests for aws_bench.resource_management.cleanup.verification."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from aws_bench.resource_management.cleanup.models import ExistenceStatus, StackResource
from aws_bench.resource_management.cleanup.verification.manager import ResourceVerifier
from aws_bench.resource_management.cleanup.verification.registry import (
    _VERIFIER_REGISTRY,
    UNCHECKED_SUBRESOURCE_TYPES,
    verifies,
)


@pytest.fixture()
def mock_ccm():
    with patch(
        "aws_bench.resource_management.cleanup.verification.manager.CloudControlManager"
    ) as cls:
        ccm = MagicMock()
        cls.return_value = ccm
        yield ccm


# -- ResourceVerifier._should_skip --


def test_should_skip_empty_physical_id():
    assert ResourceVerifier._should_skip("AWS::S3::Bucket", "") is True


def test_should_skip_custom_resource():
    assert ResourceVerifier._should_skip("Custom::MyThing", "phys-1") is True


def test_should_skip_single_colon_type():
    assert ResourceVerifier._should_skip("AWS::S3", "phys-1") is True


def test_should_skip_account_singleton():
    assert ResourceVerifier._should_skip("AWS::ApiGateway::Account", "acct") is True


def test_should_not_skip_normal_resource():
    assert ResourceVerifier._should_skip("AWS::S3::Bucket", "my-bucket") is False


# -- verifies decorator --


def test_verifies_registers_function():
    @verifies("AWS::Test::VerifyDecoratorTest")
    def _check(session, physical_id):
        return True

    assert _VERIFIER_REGISTRY["AWS::Test::VerifyDecoratorTest"] is _check
    del _VERIFIER_REGISTRY["AWS::Test::VerifyDecoratorTest"]


def test_registered_verifiers_match_expected_types():
    for resource_type in _VERIFIER_REGISTRY:
        assert resource_type.startswith("AWS::"), f"Unexpected type: {resource_type}"
        assert resource_type.count("::") == 2, f"Malformed type: {resource_type}"


# -- UNCHECKED_SUBRESOURCE_TYPES --


def test_sub_resources_are_marked_unchecked():
    assert "AWS::Lambda::Permission" in UNCHECKED_SUBRESOURCE_TYPES
    assert "AWS::ApiGateway::Stage" in UNCHECKED_SUBRESOURCE_TYPES


def test_no_overlap_with_verifier_registry():
    overlap = UNCHECKED_SUBRESOURCE_TYPES & set(_VERIFIER_REGISTRY.keys())
    assert overlap == set(), f"Types in both UNCHECKED_SUBRESOURCE and registry: {overlap}"


# -- ResourceVerifier.verify_resources --


def test_verify_skips_resource_without_physical_id(mock_ccm):
    resource = StackResource(
        logical_id="L1",
        physical_id="",
        resource_type="AWS::S3::Bucket",
        status="CREATE_COMPLETE",
    )
    result = asyncio.run(ResourceVerifier(MagicMock()).verify_resources([resource]))
    assert result[0].existence_status == ExistenceStatus.SKIPPED


def test_verify_always_absent_type_returns_unchecked(mock_ccm):
    resource = StackResource(
        logical_id="L1",
        physical_id="some-id",
        resource_type="AWS::Lambda::Permission",
        status="CREATE_COMPLETE",
    )
    result = asyncio.run(ResourceVerifier(MagicMock()).verify_resources([resource]))
    assert result[0].existence_status == ExistenceStatus.UNCHECKED_SUBRESOURCE
    mock_ccm.resource_exists.assert_not_called()


def test_verify_resource_exists_via_ccapi(mock_ccm):
    """Test CCAPI-based verification for resources without custom verifiers."""
    mock_ccm.resource_exists.return_value = True
    resource = StackResource(
        logical_id="L1",
        physical_id="my-fn",
        resource_type="AWS::Lambda::Function",
        status="CREATE_COMPLETE",
    )
    result = asyncio.run(ResourceVerifier(MagicMock()).verify_resources([resource]))
    assert result[0].existence_status == ExistenceStatus.EXISTS
    # Verify CCAPI was actually called
    mock_ccm.resource_exists.assert_called_once()


def test_verify_resource_gone_via_ccapi(mock_ccm):
    """Test CCAPI-based verification confirms resource is gone."""
    mock_ccm.resource_exists.return_value = False
    resource = StackResource(
        logical_id="L1",
        physical_id="gone-fn",
        resource_type="AWS::Lambda::Function",
        status="DELETE_COMPLETE",
    )
    result = asyncio.run(ResourceVerifier(MagicMock()).verify_resources([resource]))
    assert result[0].existence_status == ExistenceStatus.ABSENT
    # Verify CCAPI was actually called
    mock_ccm.resource_exists.assert_called_once()


def test_verify_falls_back_to_registered_verifier(mock_ccm):
    exc = ClientError(
        {"Error": {"Code": "UnsupportedActionException", "Message": "unsupported"}},
        "GetResource",
    )
    mock_ccm.resource_exists.side_effect = exc

    resource = StackResource(
        logical_id="L1",
        physical_id="alloc-123",
        resource_type="AWS::EC2::EIP",
        status="CREATE_COMPLETE",
    )
    session = MagicMock()
    ec2_client = MagicMock()
    session.client.return_value = ec2_client
    ec2_client.describe_addresses.return_value = {"Addresses": [{"AllocationId": "alloc-123"}]}

    result = asyncio.run(ResourceVerifier(session).verify_resources([resource]))
    assert result[0].existence_status == ExistenceStatus.EXISTS


def test_verify_unknown_on_non_ccapi_error(mock_ccm):
    exc = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "throttled"}},
        "GetResource",
    )
    mock_ccm.resource_exists.side_effect = exc

    resource = StackResource(
        logical_id="L1",
        physical_id="my-fn",
        resource_type="AWS::Lambda::Function",
        status="CREATE_COMPLETE",
    )
    result = asyncio.run(ResourceVerifier(MagicMock()).verify_resources([resource]))
    assert result[0].existence_status == ExistenceStatus.UNKNOWN


def test_verify_throttled_returns_unknown_not_skipped(mock_ccm, caplog):
    """A throttled existence check maps to UNKNOWN, and says it was a throttle.

    SKIPPED is the 'genuinely unsupported' bucket; a throttled resource is merely
    unverified. Bucketing it as SKIPPED drops it from the orphan count, so a throttle can
    make a dirty account read as clean.
    """
    from aws_bench.resource_management.ccapi.exceptions import ResourceExistenceThrottledError

    mock_ccm.resource_exists.side_effect = ResourceExistenceThrottledError("throttled")
    resource = StackResource(
        logical_id="L1",
        physical_id="my-fn",
        resource_type="AWS::Lambda::Function",
        status="CREATE_COMPLETE",
    )
    with caplog.at_level(
        logging.DEBUG, logger="aws_bench.resource_management.cleanup.verification.manager"
    ):
        result = asyncio.run(ResourceVerifier(MagicMock()).verify_resources([resource]))
    assert result[0].existence_status == ExistenceStatus.UNKNOWN
    assert "throttled" in caplog.text.lower()


def test_verify_unsupported_ccapi_still_skipped(mock_ccm):
    """A genuinely unsupported CCAPI type still maps to SKIPPED.

    Guards against the throttle carve-out accidentally regressing the SKIPPED path.
    """
    from aws_bench.resource_management.ccapi.exceptions import ResourceExistenceUnsupportedError

    mock_ccm.resource_exists.side_effect = ResourceExistenceUnsupportedError(
        "CCAPI does not support X"
    )
    resource = StackResource(
        logical_id="L1",
        physical_id="x",
        resource_type="AWS::Unsupported::Type",
        status="CREATE_COMPLETE",
    )
    result = asyncio.run(ResourceVerifier(MagicMock()).verify_resources([resource]))
    assert result[0].existence_status == ExistenceStatus.SKIPPED


def test_verify_check_failure_returns_unknown_not_skipped(mock_ccm):
    """A failed existence check maps to UNKNOWN, not SKIPPED.

    SKIPPED asserts the type itself cannot be acted on, so the force-abandoned sweep
    drops it. A check that failed (e.g. AccessDenied on a governed resource) leaves
    existence unverified, not disproved, so the resource must stay a delete candidate.
    """
    from aws_bench.resource_management.ccapi.exceptions import ResourceExistenceCheckError

    mock_ccm.resource_exists.side_effect = ResourceExistenceCheckError(
        "Failed to check existence of AWS::Glue::Database 'db': AccessDeniedException"
    )
    resource = StackResource(
        logical_id="GlueDatabase",
        physical_id="analytics_db_111111111111_us-east-1",
        resource_type="AWS::Glue::Database",
        status="DELETE_FAILED",
    )
    result = asyncio.run(ResourceVerifier(MagicMock()).verify_resources([resource]))
    assert result[0].existence_status == ExistenceStatus.UNKNOWN


def test_verify_preserves_input_order(mock_ccm):
    mock_ccm.resource_exists.return_value = True
    resources = [
        StackResource(
            logical_id=f"L{idx}",
            physical_id=f"p{idx}",
            resource_type="AWS::Lambda::Function",
            status="CREATE_COMPLETE",
        )
        for idx in range(3)
    ]
    result = asyncio.run(ResourceVerifier(MagicMock()).verify_resources(resources))
    assert [entry.physical_id for entry in result] == ["p0", "p1", "p2"]


def test_verify_registered_verifier_takes_priority_over_ccapi(mock_ccm):
    session = MagicMock()
    s3_client = MagicMock()
    session.client.return_value = s3_client
    s3_client.exceptions.NoSuchBucket = type("NoSuchBucket", (Exception,), {})
    s3_client.list_objects_v2.side_effect = s3_client.exceptions.NoSuchBucket()

    resource = StackResource(
        logical_id="L1",
        physical_id="gone-bucket",
        resource_type="AWS::S3::Bucket",
        status="DELETE_COMPLETE",
    )
    result = asyncio.run(ResourceVerifier(session).verify_resources([resource]))
    assert result[0].existence_status == ExistenceStatus.ABSENT
    mock_ccm.resource_exists.assert_not_called()


def test_verify_resources_validates_max_concurrency(mock_ccm):
    """Test that verify_resources raises ValueError for invalid max_concurrency."""
    session = MagicMock()
    verifier = ResourceVerifier(session)
    resource = StackResource(
        logical_id="L1",
        physical_id="bucket-1",
        resource_type="AWS::S3::Bucket",
        status="DELETE_COMPLETE",
    )

    # Test with 0
    with pytest.raises(ValueError, match="max_concurrency must be > 0, got 0"):
        asyncio.run(verifier.verify_resources([resource], max_concurrency=0))

    # Test with negative
    with pytest.raises(ValueError, match="max_concurrency must be > 0, got -1"):
        asyncio.run(verifier.verify_resources([resource], max_concurrency=-1))


def test_verify_custom_verifier_botocore_error_returns_unknown(mock_ccm):
    """Test custom verifier ClientError/BotoCoreError returns unknown."""
    from botocore.exceptions import BotoCoreError

    session = MagicMock()
    verifier = ResourceVerifier(session)
    resource = StackResource("L1", "b1", "AWS::S3::Bucket", "CREATE_COMPLETE")
    verifier_fn = MagicMock(side_effect=BotoCoreError())
    with patch(
        "aws_bench.resource_management.cleanup.verification.registry._VERIFIER_REGISTRY",
        {"AWS::S3::Bucket": verifier_fn},
    ):
        result = asyncio.run(verifier.verify_resources([resource]))
    assert result[0].existence_status == ExistenceStatus.UNKNOWN


def test_verify_ccapi_unexpected_error_returns_unknown(mock_ccm):
    """Test unexpected CCAPI exception returns unknown."""
    session = MagicMock()
    verifier = ResourceVerifier(session)
    resource = StackResource("L1", "x1", "AWS::Custom::Thing", "CREATE_COMPLETE")
    with (
        patch("aws_bench.resource_management.cleanup.verification.registry._VERIFIER_REGISTRY", {}),
        patch.object(verifier._ccm, "resource_exists", side_effect=RuntimeError("boom")),
    ):
        result = asyncio.run(verifier.verify_resources([resource]))
    assert result[0].existence_status == ExistenceStatus.UNKNOWN
