"""Tests for aws_bench.utils.regions."""

from unittest.mock import MagicMock

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from aws_bench.utils.regions import get_enabled_regions


@mock_aws
def test_get_enabled_regions_returns_list():
    """get_enabled_regions returns a list of region names."""
    session = boto3.Session(region_name="us-east-1")
    regions = get_enabled_regions(session)

    assert isinstance(regions, list)
    assert len(regions) > 0
    assert all(isinstance(r, str) for r in regions)
    assert "us-east-1" in regions


def test_get_enabled_regions_raises_on_failure():
    """get_enabled_regions raises RuntimeError when API call fails."""
    session = MagicMock()
    error = ClientError({"Error": {"Code": "UnauthorizedOperation"}}, "DescribeRegions")
    session.client.return_value.describe_regions.side_effect = error

    with pytest.raises(RuntimeError, match="Failed to list AWS regions"):
        get_enabled_regions(session)


def test_get_enabled_regions_passes_all_regions_false():
    """get_enabled_regions calls describe_regions with AllRegions=False."""
    session = MagicMock()
    session.client.return_value.describe_regions.return_value = {
        "Regions": [
            {"RegionName": "us-east-1", "OptInStatus": "opt-in-not-required"},
            {"RegionName": "eu-west-1", "OptInStatus": "opted-in"},
        ]
    }
    regions = get_enabled_regions(session)

    session.client.return_value.describe_regions.assert_called_once_with(AllRegions=False)
    assert regions == ["us-east-1", "eu-west-1"]
