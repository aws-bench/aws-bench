"""Tests for VPC and Internet Gateway pre-delete discovery hooks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aws_bench.resource_management.cleanup.handlers.vpc import (
    _clear_igw_public_address_wedge,
    _discover_vpc_dynamic_resources,
)
from aws_bench.resource_management.cleanup.models import StackResource


def test_discover_vpc_skips_empty_ids():
    session = MagicMock()
    resources = [StackResource("L1", "", "AWS::EC2::VPC", "CREATE_COMPLETE")]
    result = _discover_vpc_dynamic_resources(resources, session)
    assert result == []


def test_discover_vpc_calls_discover_function():
    from unittest.mock import patch

    session = MagicMock()
    resources = [StackResource("L1", "vpc-123", "AWS::EC2::VPC", "CREATE_COMPLETE")]
    with patch(
        "aws_bench.resource_management.cleanup.handlers.vpc.discover_vpc_dynamic_resources",
        return_value=[],
    ) as mock_discover:
        result = _discover_vpc_dynamic_resources(resources, session)
    mock_discover.assert_called_once_with(["vpc-123"], session)
    assert result == []


def test_discover_vpc_skips_non_vpc_resources():
    session = MagicMock()
    resources = [StackResource("L1", "bucket", "AWS::S3::Bucket", "CREATE_COMPLETE")]
    result = _discover_vpc_dynamic_resources(resources, session)
    assert result == []


# -- Internet Gateway wedge hook --

_VPC_MOD = "aws_bench.resource_management.cleanup.handlers.vpc"


def test_igw_hook_resolves_vpc_and_clears_wedge():
    session = MagicMock()
    resources = [StackResource("Igw", "igw-1", "AWS::EC2::InternetGateway", "CREATE_COMPLETE")]
    with patch(f"{_VPC_MOD}.clear_igw_public_address_wedge") as mock_clear:
        result = _clear_igw_public_address_wedge(resources, session)
    mock_clear.assert_called_once_with(session, ["igw-1"])
    # The hook discovers no new resources (the teardown deletes the IGW itself).
    assert result == []


def test_igw_hook_skips_when_no_igw():
    session = MagicMock()
    resources = [StackResource("L1", "vpc-1", "AWS::EC2::VPC", "CREATE_COMPLETE")]
    with patch(f"{_VPC_MOD}.clear_igw_public_address_wedge") as mock_clear:
        result = _clear_igw_public_address_wedge(resources, session)
    mock_clear.assert_not_called()
    assert result == []


def test_igw_hook_skips_igw_without_physical_id():
    session = MagicMock()
    resources = [StackResource("Igw", "", "AWS::EC2::InternetGateway", "CREATE_COMPLETE")]
    with patch(f"{_VPC_MOD}.clear_igw_public_address_wedge") as mock_clear:
        result = _clear_igw_public_address_wedge(resources, session)
    mock_clear.assert_not_called()
    assert result == []
