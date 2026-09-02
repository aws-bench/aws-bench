"""AWS Lambda entry point: scan one (account, region) and return the raw result.

Deployed in the management account. Assumes the org access role into the target
account, runs the same FastResourceScanner the host path uses, and returns the
raw FastScanResult inline — projection onto CFN types happens on the caller.
Imports only the scan path; the CLI / snapshot / cleanup subsystems are not
pulled in.
"""

from __future__ import annotations

from typing import Any

import boto3

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.fastscan.constants import (
    SCAN_ASSUME_ROLE_DURATION_S,
    SCAN_SWEEP_TIMEOUT_S,
)
from aws_bench.resource_management.fastscan.engine import FastResourceScanner
from aws_bench.resource_management.fastscan.lambda_protocol import (
    error_envelope,
    ok_envelope,
    parse_event,
)
from aws_bench.resource_management.fastscan.models import FastScanResult

logger = get_logger(__name__)


def _session_for_account(account_id: str, region: str) -> boto3.Session:
    """Assume the org access role in ``account_id`` and return a region-bound session.

    Uses a raw STS call (only boto3) so the handler's import stays free of the
    credential-provider chain, which pulls heavy deps (harbor/pydantic/rich) absent in the
    Lambda runtime. The org role is assumed unscoped, matching the host scan path. A short
    duration bounds the credential lifetime. The ``app-`` session-name prefix (built
    inline; the shared builder is host-only) makes the assume attributable in CloudTrail
    while staying neutral (it must not reveal aws-bench to an evaluated agent).
    """
    sts = boto3.client("sts")
    resp = sts.assume_role(
        RoleArn=f"arn:aws:iam::{account_id}:role/{ORG_ACCESS_ROLE}",
        RoleSessionName=f"app-fastscan-{account_id[-6:]}",
        DurationSeconds=SCAN_ASSUME_ROLE_DURATION_S,
    )
    creds = resp["Credentials"]
    return boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name=region,
    )


def _scan(account_id: str, region: str) -> FastScanResult:
    """Assume the org role in ``account_id`` and fast-scan ``region``.

    Passes ``SCAN_SWEEP_TIMEOUT_S`` (< the Lambda's own timeout) so the sweep self-terminates
    and returns its partial result before the runtime hard-kills the process.
    """
    session = _session_for_account(account_id, region)
    return FastResourceScanner(session).scan(region, overall_timeout=SCAN_SWEEP_TIMEOUT_S)


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    """Lambda entry point: scan one (account, region), return an ok/error envelope."""
    _ = context
    try:
        account_id, region = parse_event(event)
    except ValueError as exc:
        return error_envelope(str(exc))
    try:
        result = _scan(account_id, region)
    except Exception as exc:  # noqa: BLE001 — envelope carries the failure to the caller
        logger.exception("fast-scan lambda failed")
        return error_envelope(f"{type(exc).__name__}: {exc}")
    return ok_envelope(result)
