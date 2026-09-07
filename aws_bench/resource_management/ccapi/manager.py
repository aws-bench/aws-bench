"""Cloud Control API manager — central entry point for all CCAPI operations."""

from __future__ import annotations

import time

import boto3
import tenacity
from botocore.exceptions import (
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.deleter import Deleter
from aws_bench.resource_management.ccapi.exceptions import (
    ResourceExistenceCheckError,
    ResourceExistenceHandlerFailureError,
    ResourceExistenceThrottledError,
    ResourceExistenceTransientError,
    ResourceExistenceUnsupportedError,
    is_not_found_error,
)
from aws_bench.resource_management.ccapi.models import (
    CCAPI_CLIENT_CONFIG,
    EXISTENCE_CHECK_CLIENT_CONFIG,
    EXISTENCE_CHECK_MAX_ATTEMPTS,
    HANDLER_FAILURE_ERROR_CODES,
    THROTTLE_ERROR_CODES,
    UNSUPPORTED_CCAPI_ERROR_CODES,
    DeletionFailureEvent,
    Resource,
    ScanResult,
)
from aws_bench.resource_management.ccapi.scanner import Scanner
from aws_bench.resource_management.ccapi.type_registry import TypeRegistry
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)

# Existence-check outcomes safe to retry: a throttle or a transient server/connection fault
# (as opposed to a broken-handler fault, which recurs identically and must not be retried).
_RECOVERABLE_EXISTENCE_ERRORS = (ResourceExistenceThrottledError, ResourceExistenceTransientError)

# botocore connection/timeout errors carry no error Code but are transient by nature.
_TRANSIENT_CONNECTION_ERRORS = (
    EndpointConnectionError,
    ConnectionClosedError,
    ConnectTimeoutError,
    ReadTimeoutError,
)


def _is_transient_existence_fault(exc: Exception) -> bool:
    """Whether a GetResource failure is a recoverable transient fault (retryable).

    True for a botocore connection/timeout error, or a server 5xx response. Handler-failure
    codes are classified before this is consulted, so a broken-handler 5xx never reaches here.
    """
    if isinstance(exc, _TRANSIENT_CONNECTION_ERRORS):
        return True
    status = getattr(exc, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
    return isinstance(status, int) and status >= 500


def _existence_check_sleep(seconds: float) -> None:
    """Backoff sleep for the existence-check retry.

    Indirected through this module-level function (rather than ``time.sleep`` directly) so tests
    can patch out the backoff without patching the stdlib.
    """
    time.sleep(seconds)


class CloudControlManager:
    """Central manager for AWS Cloud Control API operations."""

    def __init__(self, session: boto3.Session, region_name: str | None = None) -> None:
        """Initialize with a boto3 session and optional region.

        Args:
            session: boto3 Session for AWS operations
            region_name: AWS region name (optional, uses session's region if not provided)
        """
        self._session = session
        self._region_name = region_name or session.region_name
        self._client = build_client(
            session, "cloudcontrol", region_name=self._region_name, config=CCAPI_CLIENT_CONFIG
        )
        # Dedicated client for the fail-closed existence check (resource_exists) with retries
        # disabled — see EXISTENCE_CHECK_CLIENT_CONFIG. The scan/list/delete paths keep the
        # retrying self._client; only the per-resource GetResource check runs no-retry, so a
        # broken-handler type no longer burns 8 adaptive retries per check.
        self._existence_client = build_client(
            session,
            "cloudcontrol",
            region_name=self._region_name,
            config=EXISTENCE_CHECK_CLIENT_CONFIG,
        )
        self._scanner = Scanner(self._client, session=session, region_name=self._region_name)
        self._deleter = Deleter(self._client, resource_exists_fn=self.resource_exists)
        self._type_registry = TypeRegistry(
            session, scan_fn=self._scanner.scan_resources, region_name=self._region_name
        )

    def get_scannable_types(self) -> list[str]:
        """Get all CCAPI-supported resource types, excluding known problematic ones."""
        return self._scanner.get_scannable_types()

    def scan_resources(
        self,
        resource_types: list[str] | None = None,
    ) -> ScanResult:
        """Scan account for resources. If resource_types is None, scans all scannable types."""
        return self._scanner.scan_resources(resource_types)

    def generate_skip_types(self) -> set[str]:
        """Scan all AWS resource types and persist those that should be skipped."""
        return self._type_registry.generate_skip_types()

    def resource_exists(self, resource: Resource) -> bool:
        """Check if a resource exists via CCAPI get_resource.

        Uses the no-retry existence client (``EXISTENCE_CHECK_CLIENT_CONFIG``) so a broken-handler
        fault does not burn botocore's adaptive retries; retries are re-added HERE, at the
        application level, for the recoverable classes only. A throttle or a transient
        server/connection fault raises a recoverable error and is retried up to
        ``EXISTENCE_CHECK_MAX_ATTEMPTS`` times (keeping the reset/verification callers resilient to
        a momentary blip); a broken-handler or unsupported fault is non-recoverable and raised on
        the first attempt. Callers keep the resource on any raised error (fail-closed).
        """
        retryer = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(EXISTENCE_CHECK_MAX_ATTEMPTS),
            wait=tenacity.wait_exponential(multiplier=0.5, max=5) + tenacity.wait_random(0, 0.5),
            retry=tenacity.retry_if_exception_type(_RECOVERABLE_EXISTENCE_ERRORS),
            reraise=True,
            sleep=_existence_check_sleep,
        )
        return retryer(self._resource_exists_once, resource)

    def _resource_exists_once(self, resource: Resource) -> bool:
        """One CCAPI ``get_resource`` attempt, classifying the failure for the retry wrapper.

        Returns True/False on a definitive answer; otherwise raises a classified error — a
        recoverable one (throttle / transient fault) that ``resource_exists`` retries, or a
        non-recoverable one (handler failure / unsupported / other) that it re-raises at once.
        """
        try:
            self._existence_client.get_resource(
                TypeName=resource.type, Identifier=resource.identifier
            )
            return True
        except self._existence_client.exceptions.ResourceNotFoundException:
            return False
        except Exception as exc:
            error_code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if error_code in HANDLER_FAILURE_ERROR_CODES:
                # A type whose CCAPI handler is broken server-side (fails every check). Non-
                # recoverable — retrying only burns latency. High-volume during cleanup
                # verification — TRACE, like the unsupported case.
                logger.trace("CCAPI handler failed for %s: %s", resource.type, error_code)
                raise ResourceExistenceHandlerFailureError(
                    f"CCAPI handler failed for {resource.type} '{resource.identifier}': "
                    f"{error_code}"
                ) from exc
            if error_code in UNSUPPORTED_CCAPI_ERROR_CODES:
                # Expected for every CCAPI-unsupported type, high-volume — TRACE:
                # kept in the ledger run.log, filtered off the DEBUG job/trial sinks.
                logger.trace("CCAPI does not support %s: %s", resource.type, error_code)
                raise ResourceExistenceUnsupportedError(
                    f"CCAPI does not support {resource.type}: {error_code}"
                ) from exc
            if error_code in THROTTLE_ERROR_CODES:
                logger.debug("CCAPI throttled checking %s: %s", resource.type, error_code)
                raise ResourceExistenceThrottledError(
                    f"CCAPI throttled checking {resource.type} '{resource.identifier}': "
                    f"{error_code}"
                ) from exc
            if is_not_found_error(exc):
                logger.debug("Resource gone: %s '%s'", resource.type, resource.identifier)
                return False
            if _is_transient_existence_fault(exc):
                logger.debug(
                    "CCAPI transient fault checking %s '%s': %s",
                    resource.type,
                    resource.identifier,
                    error_code or type(exc).__name__,
                )
                raise ResourceExistenceTransientError(
                    f"CCAPI transient fault checking {resource.type} '{resource.identifier}': "
                    f"{error_code or type(exc).__name__}"
                ) from exc
            logger.debug(
                "Failed to check existence of %s '%s': %s",
                resource.type,
                resource.identifier,
                exc,
            )
            raise ResourceExistenceCheckError(
                f"Failed to check existence of {resource.type} '{resource.identifier}': {exc}"
            ) from exc

    def delete_resources(self, resources: list[Resource]) -> dict[Resource, DeletionFailureEvent]:
        """Delete resources via CCAPI with ordered batching and retry.

        Returns dict of Resource → failure for resources that couldn't be deleted.
        """
        return self._deleter.delete_resources(resources)
