"""Scanner factory: select the resource-discovery backend per AWSBENCH_SCAN_METHOD.

Thin selection layer over three drop-in backends — the fast native-API scan
(:class:`~.fastscan.manager.FastScanManager`), that scan offloaded to a
management-account Lambda (:class:`~.fastscan.lambda_scanner.LambdaScanner`), and the
CloudControl scan (:class:`~.ccapi.manager.CloudControlManager`). Nothing imports back
into this module, so it may freely import from both ``fastscan`` and ``ccapi``.
"""

from __future__ import annotations

import os
from typing import Protocol

import boto3

from aws_bench.account_management.preexisting import active_account_config
from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.manager import CloudControlManager
from aws_bench.resource_management.ccapi.models import ScanResult
from aws_bench.resource_management.fastscan.lambda_scanner import LambdaScanner
from aws_bench.resource_management.fastscan.manager import FastScanManager

logger = get_logger(__name__)

SCAN_METHOD_ENV_VAR = "AWSBENCH_SCAN_METHOD"
_FASTSCAN = "fastscan"
_CCAPI = "ccapi"
_FASTSCAN_LAMBDA = "fastscan-lambda"
_VALID = (_FASTSCAN, _CCAPI, _FASTSCAN_LAMBDA)


class Scanner(Protocol):
    """Common discovery interface the factory returns (drop-in across backends)."""

    def scan_resources(
        self, resource_types: list[str] | None = None, region: str | None = None
    ) -> ScanResult:
        """Scan the account and return a CCAPI-shaped :class:`ScanResult`."""
        ...


def scan_method() -> str:
    """The configured scan backend. Unset → the Lambda scan (host scan only when no account_id)."""
    if active_account_config() is not None and SCAN_METHOD_ENV_VAR not in os.environ:
        # The shared scanner Lambda assumes OrganizationAccountAccessRole and is
        # deployed in the benchmark-created management account. Neither exists
        # in an externally owned account, so use the same scanner in-process.
        return _FASTSCAN
    method = os.environ.get(SCAN_METHOD_ENV_VAR, _FASTSCAN_LAMBDA).strip().lower()
    if method not in _VALID:
        logger.warning(f"Unknown {SCAN_METHOD_ENV_VAR}={method!r}; using {_FASTSCAN_LAMBDA!r}")
        return _FASTSCAN_LAMBDA
    return method


class _CcapiScanAdapter:
    """The ``ccapi`` backend: :class:`CloudControlManager` adapted to the ``Scanner`` interface."""

    def __init__(self, session: boto3.Session, region_name: str | None) -> None:
        self._mgr = CloudControlManager(session, region_name=region_name)

    def scan_resources(
        self, resource_types: list[str] | None = None, region: str | None = None
    ) -> ScanResult:
        _ = region  # region is fixed at construction for CCAPI
        return self._mgr.scan_resources(resource_types)


def make_scanner(
    session: boto3.Session,
    region_name: str | None = None,
    account_id: str | None = None,
) -> Scanner:
    """Return the configured discovery scanner (drop-in: ``.scan_resources(...) -> ScanResult``).

    Default (``account_id`` given, method unset) offloads the scan to the
    management-account discovery Lambda. That scan does NOT fall back per pair — it
    raises ``LambdaScanTransient`` if the Lambda cannot produce a result.
    ``AWSBENCH_SCAN_METHOD=fastscan`` forces host-only; ``=ccapi`` forces the
    CloudControl scan. Without ``account_id`` the Lambda cannot target an account,
    so the host scan runs (a construction-time choice, not a runtime fallback).
    """
    method = scan_method()
    if method == _CCAPI:
        logger.debug("Scan backend: ccapi (CloudControl)")
        return _CcapiScanAdapter(session, region_name)
    if method == _FASTSCAN_LAMBDA and account_id is not None:
        logger.debug(
            "Scan backend: fastscan-lambda (account %s, region %s)", account_id, region_name
        )
        return LambdaScanner(session, account_id=account_id, region_name=region_name)
    logger.debug("Scan backend: fastscan host (method=%s, account_id=%s)", method, account_id)
    return FastScanManager(session, region_name=region_name)
