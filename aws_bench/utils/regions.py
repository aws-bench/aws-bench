"""AWS region utilities."""

from __future__ import annotations

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.constants import DEFAULT_REGION
from aws_bench.logging.logger import get_logger
from aws_bench.utils.concurrent import build_client
from aws_bench.utils.retry import retrying_region_read

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


def wait_for_region_access(session: boto3.Session, regions: list[str]) -> None:
    """Probe each declared region until CloudFormation access has converged.

    This synchronous, read-only readiness check belongs in ``asyncio.to_thread``
    when called from async orchestration. Only the first ListStacks page is read.
    """
    for region in dict.fromkeys(regions):
        client = build_client(
            session,
            "cloudformation",
            region_name=region,
            config=Config(
                connect_timeout=5,
                read_timeout=10,
                retries={"mode": "standard", "total_max_attempts": 2},
            ),
        )
        retrying_region_read(client.list_stacks)
