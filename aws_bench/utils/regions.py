"""AWS region utilities."""

from __future__ import annotations

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.constants import DEFAULT_REGION
from aws_bench.logging.logger import get_logger
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)


def get_enabled_regions(session: boto3.Session) -> list[str]:
    """List all enabled regions in the account.

    Uses EC2 describe_regions to discover which regions are enabled.
    Returns regions in the order provided by the API.

    Args:
        session: boto3 Session for AWS operations

    Returns:
        List of enabled region names

    Raises:
        RuntimeError: If unable to list regions
    """
    try:
        ec2 = build_client(session, "ec2", region_name=DEFAULT_REGION)
        regions = ec2.describe_regions(AllRegions=False)["Regions"]
        return [region["RegionName"] for region in regions]
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to list AWS regions: {exc}") from exc
