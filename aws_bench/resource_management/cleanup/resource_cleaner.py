"""Resource cleanup orchestrator — prepare, custom delete, CCAPI fallback."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import as_completed
from typing import Any

import boto3

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.manager import CloudControlManager, Resource
from aws_bench.resource_management.ccapi.models import (
    LOG_TRUNCATE_LONG,
    LOG_TRUNCATE_SHORT,
    MAX_WORKERS_LIGHT,
    DeletionFailureEvent,
)
from aws_bench.resource_management.cleanup.handler_registry import (
    CUSTOM_DELETION_REGISTRY,
    FAILED_RESOURCE_HANDLERS,
    PRE_DELETE_HOOKS,
    PREPARE_REGISTRY,
)
from aws_bench.resource_management.cleanup.models import (
    DELETE_FAILED,
    CustomDeletionResult,
    HandlerResult,
    HandlerStatus,
    StackResource,
    to_ccapi_resources,
)
from aws_bench.utils.concurrent import interruptible_executor

logger = get_logger(__name__)

# Sample size for log messages
LOG_SAMPLE_SIZE = 3

# Resource types whose custom deletion must finish before any general prepare
# handler runs. An EKS-managed Auto Scaling group is prepared by suspending its
# ReplaceUnhealthy process; if that happens while the owning nodegroup is still
# being deleted, the nodegroup cannot recycle its instances and its deletion
# enters DELETE_FAILED. Deleting the nodegroup to terminal completion first
# removes that dependency before the ASG is ever touched.
DELETE_BEFORE_PREPARE_TYPES = frozenset({"AWS::EKS::Nodegroup"})


def partition_delete_before_prepare(
    resources: list[StackResource],
) -> tuple[list[StackResource], list[StackResource]]:
    """Split resources into ``(barrier, rest)`` by :data:`DELETE_BEFORE_PREPARE_TYPES`.

    ``barrier`` holds the direct stack resources that must be custom-deleted to
    completion before general preparation; ``rest`` is everything else and flows
    through the unchanged pipeline.
    """
    barrier: list[StackResource] = []
    rest: list[StackResource] = []
    for resource in resources:
        target = barrier if resource.resource_type in DELETE_BEFORE_PREPARE_TYPES else rest
        target.append(resource)
    return barrier, rest


def truncate_for_log(text: str, limit: int) -> str:
    """Truncate string for logging, adding ellipsis if truncated."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def format_sample_list(
    items: list, format_fn: Callable[[Any], str] = str, limit: int = LOG_SAMPLE_SIZE
) -> str:
    """Format list as 'item1, item2, item3 (+N more)'.

    Args:
        items: List of items to format
        format_fn: Function to convert each item to string (default: str)
        limit: Number of items to show before truncating

    Returns:
        Formatted string representation
    """
    if not items:
        return "(none)"
    sample = ", ".join(format_fn(item) for item in items[:limit])
    if len(items) > limit:
        sample += f" (+{len(items) - limit} more)"
    return sample


class ResourceCleaner:
    """Executes the cleanup pipeline: prepare, handle stuck, custom delete, CCAPI fallback."""

    def __init__(self, session: boto3.Session, region: str | None = None) -> None:
        """Initialize with a boto3 session and the region being cleaned.

        ``region`` scopes the failed-resource handlers to the stack's region so
        service-specific orphan cleanup (e.g. Redshift) targets the one region
        the stack lived in rather than scanning every enabled region.
        """
        self._session = session
        self._region = region

    async def cleanup(
        self,
        resources: list[StackResource],
        *,
        prepare: bool = False,
        handle_stuck: bool = False,
        custom_delete: bool = False,
        ccapi_fallback: bool = False,
    ) -> dict[Resource, DeletionFailureEvent]:
        """Run the requested cleanup steps on resources.

        Returns a dict mapping each failed resource to its failure reason.

        Raises:
            ValueError: If all operation flags are False
        """
        if not any([prepare, handle_stuck, custom_delete, ccapi_fallback]):
            raise ValueError("At least one cleanup operation must be enabled")

        if prepare and custom_delete:
            barrier_resources, resources = partition_delete_before_prepare(resources)
            barrier_failures = await self._delete_before_prepare(barrier_resources)
            if barrier_failures:
                self._log_failures(barrier_failures)
                return barrier_failures

        if prepare:
            ccapi_resources = to_ccapi_resources(resources)
            if ccapi_resources:
                prepare_results = await asyncio.to_thread(self._prepare_all, ccapi_resources)
                prepared = [r for r in prepare_results if r.status == HandlerStatus.SUCCESS]
                if prepared:
                    logger.debug(
                        "Prepared %d resource(s): %s",
                        len(prepared),
                        format_sample_list(prepared, lambda r: r.resource_type),
                    )

        if handle_stuck and FAILED_RESOURCE_HANDLERS:
            stuck_count = len([r for r in resources if r.status == DELETE_FAILED])
            if stuck_count > 0:
                logger.debug("Handling %d stuck (DELETE_FAILED) resource(s)", stuck_count)
            await asyncio.to_thread(self._handle_failed, resources)

        if not custom_delete and not ccapi_fallback:
            return {}

        resolved = await asyncio.to_thread(self._resolve_hooks, resources)
        if not resolved:
            return {}

        remaining, custom_failures = resolved, {}
        if custom_delete and CUSTOM_DELETION_REGISTRY:
            result = await asyncio.to_thread(self._custom_delete, resolved)
            remaining, custom_failures = result.skipped, result.failed
            if result.succeeded:
                logger.debug(
                    "Custom-deleted %d resource(s): %s",
                    len(result.succeeded),
                    format_sample_list(result.succeeded, lambda r: r.type),
                )

        if not ccapi_fallback:
            self._log_failures(custom_failures)
            return custom_failures

        ccm = CloudControlManager(self._session)
        ccapi_count = len(remaining)
        failures = await asyncio.to_thread(ccm.delete_resources, remaining)
        ccapi_succeeded = ccapi_count - len(failures)
        if ccapi_succeeded > 0:
            logger.debug("CCAPI-deleted %d resource(s)", ccapi_succeeded)
        failures.update(custom_failures)
        self._log_failures(failures)
        return failures

    async def _delete_before_prepare(
        self, resources: list[StackResource]
    ) -> dict[Resource, DeletionFailureEvent]:
        """Custom-delete barrier resources to completion before general preparation.

        Runs the registered custom handler (with its bounded execution and PR #66
        terminal waiter) for each barrier resource and fails closed:

        * A handler failure (or raised exception) is returned as-is.
        * A resource with no registered custom handler comes back in
          ``_custom_delete().skipped``; that means nothing would delete it, so it
          is converted into an explicit per-resource failure rather than silently
          falling through to prepare/CCAPI. This is distinct from a handler
          returning ``HandlerStatus.SKIPPED`` (already classified as succeeded by
          ``_custom_delete`` — safe when the resource is simply absent).

        Returns the failures mapping; empty means every barrier resource was
        handled terminally and the caller may proceed with the remaining pipeline.
        """
        ccapi_resources = to_ccapi_resources(resources)
        if not ccapi_resources:
            return {}

        result = await asyncio.to_thread(self._custom_delete, ccapi_resources)
        if result.succeeded:
            logger.debug(
                "Custom-deleted %d resource(s): %s",
                len(result.succeeded),
                format_sample_list(result.succeeded, lambda r: r.type),
            )

        failures: dict[Resource, DeletionFailureEvent] = dict(result.failed)
        for resource in result.skipped:
            failures[resource] = DeletionFailureEvent(
                f"No custom deletion handler registered for {resource.type}"
            )
        return failures

    def _prepare_all(
        self, resources: list[Resource], max_workers: int = MAX_WORKERS_LIGHT
    ) -> list[HandlerResult]:
        """Run registered prepare handlers to make resources deletable.

        Prepare handlers are independent per resource and make real, often slow
        AWS calls (S3 empties the whole bucket, RDS/DocDB toggle deletion
        protection), so run them in a bounded wave — same shape and cap as
        ``_custom_delete`` over the same resource set. Single account+region, so
        the bound protects one throttle bucket; do not use unbounded gather.
        """
        skipped, to_process = [], []
        for resource in resources:
            (to_process if resource.type in PREPARE_REGISTRY else skipped).append(resource)

        results: list[HandlerResult] = [
            HandlerResult(
                resource_id=resource.identifier,
                resource_type=resource.type,
                action="prepare",
                status=HandlerStatus.SKIPPED,
                message="No prepare handler registered",
            )
            for resource in skipped
        ]

        with interruptible_executor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(PREPARE_REGISTRY[resource.type], resource, self._session): resource
                for resource in to_process
            }
            for future in as_completed(futures):
                resource = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.warning(
                        "Prepare failed for %s '%s': %s", resource.type, resource.identifier, e
                    )
                    results.append(
                        HandlerResult(
                            resource_id=resource.identifier,
                            resource_type=resource.type,
                            action="prepare",
                            status=HandlerStatus.FAILED,
                            message=str(e),
                        )
                    )
        return results

    def _custom_delete(
        self, resources: list[Resource], max_workers: int = MAX_WORKERS_LIGHT
    ) -> CustomDeletionResult:
        """Delete resources via registered service-API handlers in parallel."""
        skipped, to_process = [], []
        for resource in resources:
            (to_process if resource.type in CUSTOM_DELETION_REGISTRY else skipped).append(resource)

        failures: dict[Resource, DeletionFailureEvent] = {}
        with interruptible_executor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    CUSTOM_DELETION_REGISTRY[resource.type], resource, self._session
                ): resource
                for resource in to_process
            }
            for future in as_completed(futures):
                resource = futures[future]
                try:
                    result = future.result()
                    if isinstance(result, HandlerResult) and result.status == HandlerStatus.FAILED:
                        logger.warning(
                            "Custom deletion failed for %s '%s': %s",
                            resource.type,
                            resource.identifier,
                            result.message,
                        )
                        failures[resource] = DeletionFailureEvent(result.message)
                except Exception as e:
                    logger.warning(
                        "Custom deletion failed for %s '%s': %s",
                        resource.type,
                        resource.identifier,
                        e,
                    )
                    failures[resource] = DeletionFailureEvent(str(e))

        return CustomDeletionResult(
            skipped=skipped,
            succeeded=[r for r in to_process if r not in failures],
            failed=failures,
        )

    def _resolve_hooks(self, resources: list[StackResource]) -> list[Resource]:
        """Convert StackResources and discover implicit dependencies via hooks."""
        ccapi_resources = to_ccapi_resources(resources)
        if not PRE_DELETE_HOOKS:
            return ccapi_resources
        discovered: list[Resource] = []
        types_present = {r.resource_type for r in resources}
        for hook_type, hook_fn in PRE_DELETE_HOOKS.items():
            if hook_type in types_present:
                try:
                    discovered.extend(hook_fn(resources, self._session))
                except Exception as e:
                    logger.warning("Pre-delete hook for %s failed: %s", hook_type, e)
        # Deduplicate discovered resources against known resources
        known = {(r.type, r.identifier) for r in ccapi_resources}
        new_resources = [r for r in discovered if (r.type, r.identifier) not in known]
        return ccapi_resources + new_resources

    def _handle_failed(self, resources: list[StackResource]) -> None:
        """Run handlers for resources stuck in DELETE_FAILED.

        Handlers execute in priority order (lowest priority first). Multiple handlers
        can match the same resource via prefix matching (intentional fall-through).
        """
        failed = [r for r in resources if r.status == DELETE_FAILED]
        if not failed:
            return
        # FAILED_RESOURCE_HANDLERS is already sorted by priority (maintained by decorator)
        for priority, type_prefix, handler_fn in FAILED_RESOURCE_HANDLERS:
            matching = [r for r in failed if r.resource_type.startswith(type_prefix)]
            if matching:
                try:
                    handler_fn(matching, resources, self._session, self._region)
                except Exception as e:
                    logger.warning(
                        "Failed resource handler for '%s' (priority=%d) raised: %s",
                        type_prefix,
                        priority,
                        e,
                    )

    @staticmethod
    def _log_failures(failures: dict[Resource, DeletionFailureEvent]) -> None:
        for resource, event in failures.items():
            logger.warning(
                "Delete failed: %s '%s': %s",
                resource.type,
                truncate_for_log(resource.identifier, LOG_TRUNCATE_SHORT),
                truncate_for_log(event.status_message, LOG_TRUNCATE_LONG),
            )
