"""Tests for the in-Lambda fast-scan handler."""

from __future__ import annotations

from unittest.mock import MagicMock

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.resource_management.fastscan import constants, lambda_handler
from aws_bench.resource_management.fastscan.constants import SCAN_SWEEP_TIMEOUT_S
from aws_bench.resource_management.fastscan.models import FastScanResult


def test_handler_returns_ok_envelope_with_scan_result(mocker):
    fake = FastScanResult(discovered={"ec2:DescribeVpcs": ["vpc-1"]}, failed={}, empty=set())
    # Stub the assume-role + session build so no AWS is touched.
    mocker.patch.object(lambda_handler, "_session_for_account", return_value=MagicMock())
    scanner = mocker.patch.object(lambda_handler, "FastResourceScanner")
    scanner.return_value.scan.return_value = fake

    out = lambda_handler.handler({"account_id": "111111111111", "region": "us-east-1"}, None)

    assert out["ok"] is True
    assert out["result"]["discovered"] == {"ec2:DescribeVpcs": ["vpc-1"]}
    # The handler must budget the sweep below the Lambda timeout so it returns a partial
    # result before the runtime kills the process.
    scanner.return_value.scan.assert_called_once_with(
        "us-east-1", overall_timeout=SCAN_SWEEP_TIMEOUT_S
    )


def test_handler_sweep_budget_is_below_lambda_timeout():
    assert constants.SCAN_SWEEP_TIMEOUT_S < constants.LAMBDA_FUNCTION_TIMEOUT_S


def test_handler_bad_event_returns_error_envelope(mocker):
    out = lambda_handler.handler({"account_id": "111111111111"}, None)  # missing region
    assert out["ok"] is False
    assert "region" in out["error"]


def test_handler_scan_exception_returns_error_envelope(mocker):
    mocker.patch.object(lambda_handler, "_session_for_account", return_value=MagicMock())
    scanner = mocker.patch.object(lambda_handler, "FastResourceScanner")
    scanner.return_value.scan.side_effect = RuntimeError("scan blew up")

    out = lambda_handler.handler({"account_id": "111111111111", "region": "us-east-1"}, None)
    assert out["ok"] is False
    assert "scan blew up" in out["error"]


def test_session_for_account_assumes_org_role_with_attribution(mocker):
    sts = MagicMock()
    sts.assume_role.return_value = {
        "Credentials": {"AccessKeyId": "AK", "SecretAccessKey": "SK", "SessionToken": "TK"}
    }
    mocker.patch.object(lambda_handler.boto3, "client", return_value=sts)

    lambda_handler._session_for_account("111111111111", "us-east-1")

    kwargs = sts.assume_role.call_args.kwargs
    assert kwargs["RoleArn"].endswith(f":role/{ORG_ACCESS_ROLE}")
    # Assumed unscoped, matching the host scan path — no session policy narrowing the creds.
    assert "PolicyArns" not in kwargs
    # Short-lived creds bound the credential lifetime.
    assert kwargs["DurationSeconds"] == constants.SCAN_ASSUME_ROLE_DURATION_S
    # Session name carries the app- prefix for CloudTrail attribution.
    assert kwargs["RoleSessionName"].startswith("app-")
