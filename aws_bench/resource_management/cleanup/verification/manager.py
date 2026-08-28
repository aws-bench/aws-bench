"""ResourceVerifier class for checking post-deletion resource existence."""

from __future__ import annotations

import asyncio

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.exceptions import (
    ResourceExistenceCheckError,
    ResourceExistenceThrottledError,
    ResourceExistenceUnsupportedError,
)
from aws_bench.resource_management.ccapi.manager import CloudControlManager, Resource
from aws_bench.resource_management.ccapi.models import (
    CUSTOM_RESOURCE_PREFIX,
    MAX_WORKERS_LIGHT,
    UNSUPPORTED_CCAPI_ERROR_CODES,
)
from aws_bench.resource_management.cleanup.models import (
    ExistenceStatus,
    ResourceVerificationResult,
    StackResource,
)

# Import verifiers to trigger decorator registration
from aws_bench.resource_management.cleanup.verification import verifiers as _  # noqa: F401
from aws_bench.resource_management.cleanup.verification.registry import (
    SKIP_TYPES,
    UNCHECKED_SUBRESOURCE_TYPES,
    get_verifier,
)

logger = get_logger(__name__)


class ResourceVerifier:
    """Verifies resource existence via CCAPI + service API fallbacks."""

    def __init__(self, session: boto3.Session) -> None:
        """Initialize with a boto3 session."""
        self._session = session
        self._ccm = CloudControlManager(session)

    async def verify_resources(
        self, resources: list[StackResource], max_concurrency: int = MAX_WORKERS_LIGHT
    ) -> list[ResourceVerificationResult]:
        """Check which resources still exist.

        Returns one verification result per input resource (same order/length).
        Each result contains the resource info plus an existence_status:
            - EXISTS: Resource still exists
            - ABSENT: Resource confirmed gone
            - SKIPPED: Resource type cannot be verified (e.g., Custom::*)
            - UNKNOWN: Verification did not answer (throttled, denied, or errored)
            - UNCHECKED_SUBRESOURCE: Sub-resource that requires parent context

        Limits concurrency to avoid thundering herd of API calls.

        Args:
            resources: List of StackResource objects to verify
            max_concurrency: Maximum concurrent verification calls (must be > 0)

        Raises:
            ValueError: If max_concurrency <= 0
        """
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be > 0, got {max_concurrency}")
        sem = asyncio.Semaphore(max_concurrency)

        async def verify_with_limit(resource):
            async with sem:
                return await self._verify_single(resource)

        return await asyncio.gather(*[verify_with_limit(resource) for resource in resources])

    async def _verify_single(self, stack_resource: StackResource) -> ResourceVerificationResult:
        """Verify a single resource. Returns a ResourceVerificationResult."""
        rtype = stack_resource.resource_type
        pid = stack_resource.physical_id

        def _make_result(status: ExistenceStatus) -> ResourceVerificationResult:
            return ResourceVerificationResult(
                logical_id=stack_resource.logical_id,
                physical_id=stack_resource.physical_id,
                resource_type=stack_resource.resource_type,
                cfn_status=stack_resource.status,
                existence_status=status,
            )

        if self._should_skip(rtype, pid):
            return _make_result(ExistenceStatus.SKIPPED)

        # Sub-resources can't be checked without parent context (e.g., Lambda::Permission
        # requires the function name). Mark as unchecked rather than assuming they don't exist.
        if rtype in UNCHECKED_SUBRESOURCE_TYPES:
            return _make_result(ExistenceStatus.UNCHECKED_SUBRESOURCE)

        # Prefer registered service-API verifier (more accurate than CCAPI for some types)
        verifier = get_verifier(rtype)
        if verifier:
            try:
                exists = await asyncio.to_thread(verifier, self._session, pid)
                status = ExistenceStatus.EXISTS if exists else ExistenceStatus.ABSENT
                return _make_result(status)
            except (ClientError, BotoCoreError) as exc:
                logger.debug("Verifier failed for %s '%s': %s", rtype, pid, exc)
                return _make_result(ExistenceStatus.UNKNOWN)
            except Exception as exc:
                # Diagnostics-only: UNKNOWN never affects the cleanup verdict.
                logger.debug("Unexpected error in verifier for %s '%s': %s", rtype, pid, exc)
                return _make_result(ExistenceStatus.UNKNOWN)

        # Fall back to CCAPI
        resource = Resource(type=rtype, identifier=pid)
        try:
            exists = await asyncio.to_thread(self._ccm.resource_exists, resource)
            status = ExistenceStatus.EXISTS if exists else ExistenceStatus.ABSENT
            return _make_result(status)
        except ResourceExistenceThrottledError:
            # Throttled = unverified, not gone (SKIPPED below means "type unverifiable"). The
            # retry ceiling in CCAPI_CLIENT_CONFIG resolves most throttles before this; the
            # accurate label is defence-in-depth. Must precede the base-class catch.
            logger.debug("CCAPI throttled verifying %s '%s'; marking UNKNOWN", rtype, pid)
            return _make_result(ExistenceStatus.UNKNOWN)
        except ResourceExistenceUnsupportedError:
            # Must precede the base-class catch.
            logger.debug("CCAPI does not support %s '%s'", rtype, pid)
            return _make_result(ExistenceStatus.SKIPPED)
        except ResourceExistenceCheckError:
            logger.debug("Existence check failed for %s '%s'; marking UNKNOWN", rtype, pid)
            return _make_result(ExistenceStatus.UNKNOWN)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in UNSUPPORTED_CCAPI_ERROR_CODES:
                return _make_result(ExistenceStatus.SKIPPED)
            logger.debug("CCAPI error for %s '%s': %s", rtype, pid, code)
            return _make_result(ExistenceStatus.UNKNOWN)
        except Exception as exc:
            logger.debug("Unexpected CCAPI error for %s '%s': %s", rtype, pid, exc)
            return _make_result(ExistenceStatus.UNKNOWN)

    @staticmethod
    def _should_skip(resource_type: str, physical_id: str) -> bool:
        """Return True if this resource should be skipped during verification."""
        if not physical_id:
            return True
        if resource_type.count("::") != 2 or resource_type.startswith(CUSTOM_RESOURCE_PREFIX):
            return True
        if resource_type in SKIP_TYPES:
            return True
        return False
