"""Tests for aws_bench.utils.regions."""

from unittest.mock import MagicMock, patch

import boto3
import pytest
import tenacity
from botocore.exceptions import ClientError
from moto import mock_aws

from aws_bench.utils.regions import get_enabled_regions, wait_for_region_access
from aws_bench.utils.retry import retrying_region_read


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


@pytest.mark.parametrize("regions", [["eu-south-2", "ap-east-2", "eu-south-2"], []])
def test_wait_for_region_access_probes_declared_regions_once(regions: list[str]) -> None:
    session = MagicMock()
    clients = [MagicMock() for _ in dict.fromkeys(regions)]
    with patch("aws_bench.utils.regions.build_client", side_effect=clients) as build:
        wait_for_region_access(session, regions)
    assert build.call_count == len(clients)
    assert [call.kwargs["region_name"] for call in build.call_args_list] == list(
        dict.fromkeys(regions)
    )
    for call, client in zip(build.call_args_list, clients, strict=True):
        assert call.args == (session, "cloudformation")
        config = call.kwargs["config"]
        assert config.connect_timeout == 5
        assert config.read_timeout == 10
        assert config.retries == {"mode": "standard", "total_max_attempts": 2}
        client.list_stacks.assert_called_once_with()
        client.get_paginator.assert_not_called()


@pytest.mark.parametrize("scp", [True, False])
def test_wait_for_region_access_retries_only_scp_denial(
    scp: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = retrying_region_read.retry  # type: ignore[attr-defined]
    monkeypatch.setattr(controller, "wait", tenacity.wait_none())
    session = MagicMock()
    client = session.client.return_value
    error = ClientError(
        {
            "Error": {
                "Code": "AccessDenied",
                "Message": "service control policy" if scp else "IAM denial",
            }
        },
        "ListStacks",
    )
    client.list_stacks.side_effect = [error, {"StackSummaries": [], "NextToken": "ignored"}]
    if scp:
        wait_for_region_access(session, ["eu-south-2"])
        assert client.list_stacks.call_count == 2
    else:
        with pytest.raises(ClientError) as caught:
            wait_for_region_access(session, ["eu-south-2"])
        assert caught.value is error
        client.list_stacks.assert_called_once_with()
    client.get_paginator.assert_not_called()
