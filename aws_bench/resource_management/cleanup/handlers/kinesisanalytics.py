"""Kinesis Data Analytics cleanup handlers.

An agent that builds a Managed Service for Apache Flink "Studio" application
(a ``AWS::KinesisAnalyticsV2::Application`` notebook, often "deployed as an
application" into a nested ``environment-*-flink-studio`` CloudFormation stack)
leaves Kinesis Analytics applications that normal reset cannot remove:

- ``AWS::KinesisAnalytics::Application`` (v1) is **not supported by CCAPI**, so
  the CCAPI deleter skips it entirely and it survives every reset.
- The surviving v1 application also pins its ``environment-*-flink-studio`` CFN
  stack, whose CCAPI ``DeleteStack`` then fails ("Found N new resource(s)") and
  fails the reset closed.

Both application APIs delete the same way, and both require the application's
``CreateTimestamp`` (a conditional token that guards against a stale delete):

1. ``DescribeApplication`` to read ``ApplicationDetail.CreateTimestamp``.
2. ``DeleteApplication(ApplicationName=..., CreateTimestamp=...)``.

Custom delete handlers run before the CCAPI fallback in the reset pipeline, so
removing the applications here lets the subsequent CCAPI stack delete succeed.
"""

from __future__ import annotations

import boto3
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.models import HandlerResult, HandlerStatus
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)

_NOT_FOUND_CODES = ("ResourceNotFoundException",)


def _application_name_from_identifier(identifier: str) -> str:
    """Extract the application name from the scanned identifier.

    The v1 lister uses ``id_field="ApplicationARN"`` and the v2 lister
    ``id_field="ApplicationName"``, so the identifier may be either the full ARN
    (``arn:aws:kinesisanalytics:<region>:<account>:application/<name>``) or a bare
    name. ``DeleteApplication`` takes the name in both APIs; a bare name (no
    ``/``) is returned unchanged.
    """
    return identifier.rsplit("/", 1)[-1]


def _delete_application(
    resource: Resource, session: boto3.Session, *, service: str, label: str
) -> HandlerResult:
    """Delete a Kinesis Analytics (v1 or v2) application.

    ``DeleteApplication`` requires the ``CreateTimestamp`` from
    ``DescribeApplication``; both the v1 (``kinesisanalytics``) and v2
    (``kinesisanalyticsv2``) APIs share this shape.
    """
    name = _application_name_from_identifier(resource.identifier)
    client: BaseClient = build_client(session, service)
    try:
        detail = client.describe_application(ApplicationName=name)["ApplicationDetail"]
        client.delete_application(ApplicationName=name, CreateTimestamp=detail["CreateTimestamp"])
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in _NOT_FOUND_CODES:
            return HandlerResult(
                resource_id=resource.identifier,
                resource_type=resource.type,
                action="delete",
                status=HandlerStatus.SUCCESS,
                message=f"{label} application already gone",
            )
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Failed to delete {label} application '{name}': {e}",
        )
    except BotoCoreError as e:
        return HandlerResult(
            resource_id=resource.identifier,
            resource_type=resource.type,
            action="delete",
            status=HandlerStatus.FAILED,
            message=f"Connection error deleting {label} application '{name}': {e}",
        )
    logger.debug(f"Deleted {label} application '{name}'")
    return HandlerResult(
        resource_id=resource.identifier,
        resource_type=resource.type,
        action="delete",
        status=HandlerStatus.SUCCESS,
    )


@resource_handler("AWS::KinesisAnalyticsV2::Application", role="delete")
def _delete_v2(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete a Kinesis Data Analytics v2 (Managed Flink / Studio) application."""
    return _delete_application(
        resource, session, service="kinesisanalyticsv2", label="Kinesis Analytics v2"
    )


@resource_handler("AWS::KinesisAnalytics::Application", role="delete")
def _delete_v1(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete a Kinesis Data Analytics v1 application (not supported by CCAPI)."""
    return _delete_application(
        resource, session, service="kinesisanalytics", label="Kinesis Analytics v1"
    )
