"""CloudFormation stack cleanup strategy.

CloudFormation-native teardown: prepare hooks → DeleteStack → (reap blocking VPC ENIs
and re-drive) → FORCE_DELETE_STACK. Phase 2 never CCAPI-deletes the stack's *own*
resources, so CloudFormation runs every ``Custom::`` resource's Delete handler with its
provider Lambda + role still intact. Anything ``FORCE_DELETE_STACK`` abandons is removed
by the Phase-3 orphan sweep (``ResourceSweeper`` → CCAPI/custom handlers).

Deletes infrastructure stacks (CDKToolkit) last, runs stacks concurrently, and saves
per-region manifests.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from aws_bench.exceptions import OperationCancelled
from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.cleanup.handlers.cross_service import (
    reap_ipam_child_pools,
    reap_vpc_enis,
    reap_vpc_security_groups,
)
from aws_bench.resource_management.cleanup.models import (
    DELETE_COMPLETE,
    DELETE_FAILED,
    INFRA_PREFIXES,
    POLL_MAX_SEC,
    POLL_MIN_SEC,
    STACK_DELETE_CONCURRENCY,
    STACK_DELETE_MAX_TIMEOUT,
    STACK_DELETE_TIMEOUT,
    DeletionSummary,
    ExistenceStatus,
    StackDeletionResult,
    StackDeletionStatus,
    StackResource,
)
from aws_bench.resource_management.cleanup.resource_cleaner import ResourceCleaner
from aws_bench.resource_management.cleanup.verification.manager import ResourceVerifier
from aws_bench.resource_management.deferred import mark_deferred
from aws_bench.resource_management.utils.cloudformation import is_stack_not_found
from aws_bench.resource_management.utils.file_io import write_json
from aws_bench.utils.concurrent import build_client, reraise_if_cancelled

logger = get_logger(__name__)

# Polling and timeout constants
PROGRESS_LOG_INTERVAL_SEC = 60  # Log progress every 60 seconds during stack deletion

# Sentinel returned by _poll_deletion when a stack stays *_COMPLETE after DeleteStack,
# indicating CFN silently refused because another stack imports its exports.
_EXPORT_BLOCKED = "EXPORT_BLOCKED"
_EXPORT_BLOCKED_REASON = "Blocked by cross-stack export dependencies"
DEADLINE_EXTENSION_SEC = 600  # Extend deadline by 10 minutes when stack still in progress

# A subnet/VPC/security-group is the only thing a requester-managed ENI can pin in
# DELETE_FAILED. When a stack's outstanding failures are confined to these types and
# the reaper left only requester-managed ENIs, the stack is eventually-deletable (the
# owner releases the ENIs ~20-40 min after its own delete), not genuinely stuck — so
# it is deferred rather than failed. Any other DELETE_FAILED resource type signals a
# real, non-self-healing blocker and blocks the defer.
_DEFERRABLE_NETWORKING_TYPES = frozenset(
    {"AWS::EC2::Subnet", "AWS::EC2::VPC", "AWS::EC2::SecurityGroup"}
)


class StackDeleter:
    """CloudFormation-native stack teardown.

    ``delete_stack``: prepare hooks → ``DeleteStack`` → reap blocking VPC ENIs and
    re-drive → ``FORCE_DELETE_STACK``.
    """

    def __init__(
        self,
        session: boto3.Session,
        region: str | None = None,
        *,
        manifest_path: Path | str,
        cfn_role_arn: str | None = None,
    ) -> None:
        """Initialize with a session, optional region, and manifest output path.

        ``cfn_role_arn``: role CFN assumes for deletions (decouples from the
        stack's associated role). None = use the stack's own role (fallback).
        """
        self._session = session
        self._region = region
        self._cfn_role_arn = self._validate_cfn_role(session, cfn_role_arn)
        self._client = build_client(
            session,
            "cloudformation",
            region_name=region,
            config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
        )
        self._manifest_path = Path(manifest_path)
        self._manifest: dict[str, dict] = {}
        self._manifest_lock = asyncio.Lock()  # Async-safe manifest updates (not thread-safe)
        self._events_logged: set[str] = set()
        self._cleaner = ResourceCleaner(session, region)
        self._verifier = ResourceVerifier(session)

    @staticmethod
    def _validate_cfn_role(session: boto3.Session, role_arn: str | None) -> str | None:
        """Return role_arn if the role exists, None otherwise (pre-feature accounts)."""
        if not role_arn:
            return None
        role_name = role_arn.rsplit("/", 1)[-1]
        try:
            iam = build_client(session, "iam")
            iam.get_role(RoleName=role_name)
            return role_arn
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchEntity":
                logger.debug(
                    "CFN ops role '%s' not found; stack deletions will use "
                    "the stack's associated role",
                    role_name,
                )
                return None
            logger.debug("Could not verify CFN ops role '%s': %s; skipping", role_name, e)
            return None

    def _delete_stack_kwargs(self, stack_name: str, **extra: object) -> dict[str, object]:
        """Build kwargs for a ``delete_stack`` call, injecting the ops role if configured."""
        kwargs: dict[str, object] = {"StackName": stack_name, **extra}
        if self._cfn_role_arn:
            # CloudFormation's DeleteStack API uses "RoleARN" (all-caps ARN),
            # unlike most AWS APIs which use "RoleArn".
            kwargs["RoleARN"] = self._cfn_role_arn
        return kwargs

    # -- Stack queries -----------------------------------------------------

    def list_stacks(self) -> list[dict]:
        """Return all active, non-nested stacks in the region."""
        stacks: list[dict] = []
        for page in self._client.get_paginator("list_stacks").paginate():
            for stack_summary in page["StackSummaries"]:
                if (
                    stack_summary["StackStatus"] != DELETE_COMPLETE
                    and "ParentId" not in stack_summary
                ):
                    stacks.append(stack_summary)
        return stacks

    def get_stack_resources(self, stack_name: str) -> list[StackResource]:
        """Return all resources in a stack, or empty list if stack is gone."""
        resources: list[StackResource] = []
        try:
            for page in self._client.get_paginator("list_stack_resources").paginate(
                StackName=stack_name
            ):
                for resource_summary in page.get("StackResourceSummaries", []):
                    resources.append(
                        StackResource(
                            logical_id=resource_summary["LogicalResourceId"],
                            physical_id=resource_summary.get("PhysicalResourceId", ""),
                            resource_type=resource_summary["ResourceType"],
                            status=resource_summary["ResourceStatus"],
                        )
                    )
        except ClientError as e:
            if is_stack_not_found(e):
                logger.debug("Stack '%s' not found when listing resources", stack_name)
            else:
                logger.exception("Failed to list resources for stack '%s'", stack_name)
                raise
        return resources

    def _stack_exists(self, stack_name: str) -> bool:
        try:
            resp = self._client.describe_stacks(StackName=stack_name)
            stacks = resp.get("Stacks", [])
            return bool(stacks and stacks[0]["StackStatus"] != DELETE_COMPLETE)
        except ClientError as e:
            if is_stack_not_found(e):
                return False
            raise

    def _disable_termination_protection(self, stack_name: str) -> None:
        try:
            resp = self._client.describe_stacks(StackName=stack_name)
            if resp["Stacks"][0].get("EnableTerminationProtection", False):
                self._client.update_termination_protection(
                    EnableTerminationProtection=False, StackName=stack_name
                )
                logger.debug("Disabled termination protection on '%s'.", stack_name)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("AccessDenied", "UnauthorizedOperation", "InvalidClientTokenId"):
                logger.error(
                    "Permission denied when disabling termination protection on '%s': %s",
                    stack_name,
                    error_code,
                )
            elif is_stack_not_found(e):
                logger.debug(
                    "Stack '%s' not found when disabling termination protection", stack_name
                )
            else:
                logger.debug("Could not disable termination protection on '%s': %s", stack_name, e)

    def _get_failure_reason(self, stack_name: str) -> str:
        try:
            resp = self._client.describe_stacks(StackName=stack_name)
            return resp["Stacks"][0].get("StackStatusReason", "unknown")
        except ClientError as e:
            if is_stack_not_found(e):
                return "stack deleted"
            logger.debug("Could not get failure reason for '%s': %s", stack_name, e)
            return "unknown"
        except (IndexError, KeyError) as e:
            logger.debug(
                "Unexpected response structure getting failure reason for '%s': %s",
                stack_name,
                e,
            )
            return "unknown"

    # -- Polling -----------------------------------------------------------

    async def _check_final_deletion_status(self, stack_name: str, elapsed: float) -> str:
        """Check final status after deadline expires."""
        status, _ = await self._get_stack_status(stack_name)
        if status == DELETE_COMPLETE:
            logger.debug("Stack '%s' deleted (after %ds).", stack_name, int(elapsed))
            return DELETE_COMPLETE
        if status.endswith("_IN_PROGRESS"):
            logger.warning(
                "Stack '%s' still in progress after absolute maximum %ds; will force-delete.",
                stack_name,
                STACK_DELETE_MAX_TIMEOUT,
            )
            return DELETE_FAILED
        logger.warning(
            "Stack '%s' timed out after %ds with status: %s; will force-delete.",
            stack_name,
            int(elapsed),
            status,
        )
        return DELETE_FAILED

    async def _poll_deletion(self, stack_name: str, timeout: int) -> str:
        interval = POLL_MIN_SEC
        deadline = time.monotonic() + timeout
        absolute_deadline = time.monotonic() + STACK_DELETE_MAX_TIMEOUT
        start = time.monotonic()
        last_log_time = start

        while time.monotonic() <= min(deadline, absolute_deadline):
            status, reason = await self._get_stack_status(stack_name)

            # Check terminal states
            if status == DELETE_COMPLETE:
                logger.debug("Stack '%s' deleted.", stack_name)
                return DELETE_COMPLETE
            if status == DELETE_FAILED:
                logger.warning(
                    "Stack '%s' delete failed: %s; will force-delete.", stack_name, reason
                )
                return DELETE_FAILED
            if not status.endswith("_IN_PROGRESS"):
                # Stack stayed in a non-progress state after DeleteStack was issued.
                # This typically means CFN silently refused because exports are in use
                # by another stack (the API returns 200 but does nothing).
                if status.endswith("_COMPLETE"):
                    # delete_all_stacks retries this stack once its importer stacks are gone.
                    logger.debug(
                        "Stack '%s' still %s after DeleteStack — deferring; blocked by "
                        "cross-stack export dependencies, will retry after importers",
                        stack_name,
                        status,
                    )
                    return _EXPORT_BLOCKED
                logger.warning(
                    "Stack '%s' unexpected state: %s; will force-delete.", stack_name, status
                )
                return DELETE_FAILED

            # Log progress
            elapsed = time.monotonic() - start
            if time.monotonic() - last_log_time >= PROGRESS_LOG_INTERVAL_SEC:
                self._log_deletion_progress(stack_name, elapsed, status)
                last_log_time = time.monotonic()

            # status is known *_IN_PROGRESS here (terminal states returned above),
            # so pass it in rather than re-fetching it.
            if time.monotonic() > deadline and time.monotonic() <= absolute_deadline:
                deadline = self._maybe_extend_deadline(stack_name, deadline, elapsed, status)

            await asyncio.sleep(interval)
            interval = min(interval * 2, POLL_MAX_SEC)

        elapsed = time.monotonic() - start
        return await self._check_final_deletion_status(stack_name, elapsed)

    def _log_deletion_progress(self, stack_name: str, elapsed: float, status: str) -> None:
        """Log deletion progress periodically."""
        logger.debug(
            "Stack '%s' still deleting... (%ds elapsed, status: %s)",
            stack_name,
            int(elapsed),
            status,
        )

    def _maybe_extend_deadline(
        self, stack_name: str, deadline: float, elapsed: float, status: str
    ) -> float:
        """Extend deadline if stack is still in progress."""
        if status.endswith("_IN_PROGRESS"):
            logger.debug(
                "Stack '%s' still DELETE_IN_PROGRESS after %ds. "
                "Extending wait (CloudFormation needs more time).",
                stack_name,
                int(elapsed),
            )
            return time.monotonic() + DEADLINE_EXTENSION_SEC
        return deadline

    async def _get_stack_status(self, stack_name: str) -> tuple[str, str]:
        """Return (status, reason) for a stack. DELETE_COMPLETE if stack is gone."""
        try:
            resp = await asyncio.to_thread(self._client.describe_stacks, StackName=stack_name)
        except ClientError as e:
            if is_stack_not_found(e):
                return DELETE_COMPLETE, ""
            raise
        stacks = resp.get("Stacks", [])
        if not stacks:
            return DELETE_COMPLETE, ""
        return stacks[0]["StackStatus"], stacks[0].get("StackStatusReason", "")

    # -- Single stack deletion ---------------------------------------------

    async def delete_stack(self, stack_name: str) -> StackDeletionResult:
        """Delete one stack: prepare → DeleteStack → (ENI reap re-drive) → FORCE_DELETE_STACK.

        Phase 2 is intentionally CloudFormation-driven and does **no** direct deletion of
        the stack's own resources.
        Anything ``FORCE_DELETE_STACK`` abandons is removed by the Phase-3 sweep.
        """
        if not await asyncio.to_thread(self._stack_exists, stack_name):
            logger.debug("Stack '%s' does not exist, skipping.", stack_name)
            return StackDeletionResult(stack_name=stack_name, status=StackDeletionStatus.NOT_FOUND)

        resources = await self._prepare_stack_for_deletion(stack_name)
        status = await self._try_delete_stack(stack_name, resources)
        result = self._build_result(stack_name, status, resources)

        if result.status == StackDeletionStatus.FAILED:
            await self._recover_from_failed_delete(stack_name, result)

        async with self._manifest_lock:
            self._manifest.setdefault(stack_name, {})["status"] = result.status.value
        logger.debug("Stack '%s' final: %s", stack_name, result.status.value)
        return result

    @staticmethod
    def _still_stuck(result: StackDeletionResult) -> bool:
        """True if the stack still needs recovery: delete failed and not deferred.

        A deferred result is eventually-deletable (a later run reaps it), so it is
        not stuck and further escalation would be wrong.
        """
        return result.status == StackDeletionStatus.FAILED and not result.deferred

    async def _recover_from_failed_delete(
        self, stack_name: str, result: StackDeletionResult
    ) -> None:
        """Escalate recovery for a DELETE_FAILED stack, stopping once it clears or defers.

        Each step targets a blocker that is not a stack resource (so CloudFormation
        can't clear it itself), reaps it, and re-drives DeleteStack; force-delete is
        the last resort, then diagnostics.
        """
        # VPC delete blocked on service-managed ENIs that release only after their
        # owner is deleted mid-delete. May defer (eventually-deletable, not stuck).
        await self._reap_and_retry_networking(stack_name, result)
        if not self._still_stuck(result):
            return
        # Parent IPAM pool delete blocked by a leaked child pool's allocation.
        await self._reap_and_retry_ipam(stack_name, result)
        if not self._still_stuck(result):
            return
        await self._force_delete_failed_stack(stack_name, result)
        if not self._still_stuck(result):
            return
        # Still failed: record what survives and log the CFN events for diagnostics.
        await self._verify_deletion(result)
        await self._log_stack_failure_events(stack_name)

    async def _prepare_stack_for_deletion(self, stack_name: str) -> list[StackResource]:
        """Prepare stack for deletion: discover resources and disable protection."""
        resources = await asyncio.to_thread(self.get_stack_resources, stack_name)
        logger.debug("Stack '%s' has %d resources", stack_name, len(resources))
        async with self._manifest_lock:
            self._manifest.setdefault(stack_name, {})
            self._manifest[stack_name]["resources_before_delete"] = [
                asdict(resource) for resource in resources
            ]
        await asyncio.to_thread(self._disable_termination_protection, stack_name)
        return resources

    async def _try_delete_stack(self, stack_name: str, resources: list[StackResource]) -> str:
        """Run the prepare stage, then a standard DeleteStack, and poll to a terminal state.

        Returns the terminal stack status (``DELETE_COMPLETE`` / ``DELETE_FAILED``).
        """
        await self._cleaner.cleanup(resources, prepare=True)
        async with self._manifest_lock:
            self._manifest.setdefault(stack_name, {})["cleanup_stages"] = {"prepare": True}

        try:
            await asyncio.to_thread(
                lambda: self._client.delete_stack(**self._delete_stack_kwargs(stack_name))
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            logger.warning(
                "delete_stack failed for '%s' [%s]: %s; will force-delete.",
                stack_name,
                error_code,
                error_msg,
            )
            return DELETE_FAILED

        try:
            return await self._poll_deletion(stack_name, timeout=STACK_DELETE_TIMEOUT)
        except ClientError as e:
            logger.warning("Polling failed for '%s': %s; will force-delete.", stack_name, e)
            return DELETE_FAILED

    async def _force_delete_failed_stack(
        self, stack_name: str, result: StackDeletionResult
    ) -> None:
        """Last resort for a stack still ``DELETE_FAILED`` after all prior recovery.

        ``DeleteStack(DeletionMode="FORCE_DELETE_STACK")`` deletes the stack,
        abandoning any resource it cannot delete (with none left, it is a plain
        delete — the "state just lagging" case). On ``DELETE_COMPLETE`` the result
        flips to SUCCESS; abandoned logical IDs are recorded in the manifest
        (``force_deleted``), then :meth:`_sweep_force_abandoned` best-effort deletes
        whatever actually survived — so every caller (bulk, single-stack
        ``cleanup_stack``, reset) reaps the leftovers, not just the bulk path's
        Phase 3 scan. Resources the sweep could not delete are surfaced on
        ``result.abandoned_resources`` so the single-stack/reset path (which runs no
        orphan scan) can fail closed instead of absorbing them into a fresh baseline.
        Never raises: a ``ClientError`` is logged, result left FAILED.

        ``force_deleted`` keeps its DELETE_FAILED-only contract, but the sweep
        receives the full pre-force snapshot: a resource blocked *behind* a
        failed one (e.g. a bucket whose autoDeleteObjects custom resource
        failed) ends the force-delete as DELETE_SKIPPED while still showing its
        last healthy status in the snapshot — invisible to a DELETE_FAILED-only
        view, yet exactly what orphans fixed-name buckets.
        """
        try:
            current_status, _ = await self._get_stack_status(stack_name)
            if current_status != DELETE_FAILED:
                return

            snapshot = await asyncio.to_thread(self.get_stack_resources, stack_name)
            abandoned = [r.logical_id for r in snapshot if r.status == DELETE_FAILED]
            await asyncio.to_thread(
                lambda: self._client.delete_stack(
                    **self._delete_stack_kwargs(stack_name, DeletionMode="FORCE_DELETE_STACK")
                )
            )
            final_status = await self._poll_deletion(stack_name, timeout=STACK_DELETE_TIMEOUT)
        except ClientError as e:
            logger.error("Force delete of stack '%s' failed: %s", stack_name, e)
            return

        if final_status != DELETE_COMPLETE:
            logger.error(
                "Stack '%s' still not deleted after FORCE_DELETE_STACK (status: %s)",
                stack_name,
                final_status,
            )
            return

        result.status = StackDeletionStatus.SUCCESS
        result.reason = ""
        if not abandoned:
            logger.debug("Stack '%s': finalized (resources already removed)", stack_name)
            return
        async with self._manifest_lock:
            self._manifest.setdefault(stack_name, {})["force_deleted"] = abandoned
        logger.debug(
            "Stack '%s' force-deleted; abandoned %d resource(s) for downstream sweep: %s",
            stack_name,
            len(abandoned),
            ", ".join(abandoned),
        )
        # Surface only the survivors the sweep could NOT delete as orphans — the raw
        # DELETE_FAILED list would false-positive on everything the sweep reaped.
        result.abandoned_resources = await self._sweep_force_abandoned(stack_name, snapshot)

    async def _sweep_force_abandoned(
        self, stack_name: str, snapshot: list[StackResource]
    ) -> list[StackResource]:
        """Best-effort deletion of resources a ``FORCE_DELETE_STACK`` left behind.

        ``snapshot`` is the stack's resource list captured just before the
        force-delete. Everything not already ``DELETE_COMPLETE`` there is a
        candidate leftover; the verifier narrows that to resources that still
        EXIST, and the cleaner deletes them (``prepare`` empties buckets via the
        S3 handler, then custom handlers / CCAPI remove the resource itself).

        Orphaned fixed-name resources are the motivating case: a bucket whose
        name is baked into the template blocks every future deploy with
        "already exists" until someone reaps it, and the reset path has no
        other sweep.

        Never raises — the stack IS deleted at this point, so a sweep failure
        must not turn that success into a failure. Outcomes land in the
        manifest: ``force_abandoned_swept`` (deleted here) and
        ``force_abandoned_sweep_failures`` (still stuck; the next deploy's
        "already exists" failure is then the honest signal).

        Returns the still-stuck survivors so the single-stack/reset caller can
        surface them as orphans. A best-effort crash returns ``[]`` (never raises),
        so a sweep that dies under-reports rather than failing the delete.
        """
        try:
            candidates = [r for r in snapshot if r.status != DELETE_COMPLETE]
            if not candidates:
                return []
            verified = await self._verifier.verify_resources(candidates)
            by_key = {(r.logical_id, r.physical_id): r for r in candidates}
            survivors = [
                by_key[(v.logical_id, v.physical_id)]
                for v in verified
                if v.existence_status == ExistenceStatus.EXISTS
                and (v.logical_id, v.physical_id) in by_key
            ]
            if not survivors:
                return []
            logger.debug(
                "Stack '%s': sweeping %d resource(s) surviving force-delete: %s",
                stack_name,
                len(survivors),
                ", ".join(r.logical_id for r in survivors),
            )
            failures = await self._cleaner.cleanup(
                survivors,
                prepare=True,
                custom_delete=True,
                ccapi_fallback=True,
            )
            failed_ids = {f.identifier for f in failures}
            swept = [r.logical_id for r in survivors if r.physical_id not in failed_ids]
            stuck = [r for r in survivors if r.physical_id in failed_ids]
            async with self._manifest_lock:
                entry = self._manifest.setdefault(stack_name, {})
                if swept:
                    entry["force_abandoned_swept"] = swept
                if failures:
                    entry["force_abandoned_sweep_failures"] = sorted(failed_ids)
            if failures:
                logger.debug(
                    "Stack '%s': %d abandoned resource(s) could not be swept: %s",
                    stack_name,
                    len(failures),
                    ", ".join(sorted(failed_ids)),
                )
            return stuck
        except (asyncio.CancelledError, OperationCancelled):
            raise
        except Exception as e:  # noqa: BLE001 — sweep is best-effort by contract
            logger.warning(
                "Stack '%s': abandoned-resource sweep failed (best-effort): %s",
                stack_name,
                e,
            )
            return []

    def _build_result(
        self, stack_name: str, status: str, resources: list[StackResource]
    ) -> StackDeletionResult:
        if status == DELETE_COMPLETE:
            return StackDeletionResult(
                stack_name=stack_name, status=StackDeletionStatus.SUCCESS, resources=resources
            )
        reason = (
            _EXPORT_BLOCKED_REASON
            if status == _EXPORT_BLOCKED
            else self._get_failure_reason(stack_name)
        )
        return StackDeletionResult(
            stack_name=stack_name,
            status=StackDeletionStatus.FAILED,
            reason=reason,
            resources=resources,
        )

    # -- Networking reaper & failure diagnostics ---------------------------

    @staticmethod
    def _collect_resource_physical_ids(
        resources: list[StackResource], resource_type: str
    ) -> list[str]:
        """Physical IDs of a stack's resources of ``resource_type`` (deduped, order-preserving)."""
        seen: set[str] = set()
        physical_ids: list[str] = []
        for resource in resources:
            if (
                resource.resource_type == resource_type
                and resource.physical_id
                and resource.physical_id not in seen
            ):
                seen.add(resource.physical_id)
                physical_ids.append(resource.physical_id)
        return physical_ids

    async def _redrive_delete_stack(
        self, stack_name: str, result: StackDeletionResult, success_msg: str
    ) -> None:
        """Re-issue DeleteStack and poll once; flip ``result`` to SUCCESS on completion.

        Shared tail of the reaper re-drives. Never raises — anything but DELETE_COMPLETE
        leaves ``result`` FAILED to fall through to ``_force_delete_failed_stack``.
        """
        try:
            await asyncio.to_thread(
                lambda: self._client.delete_stack(**self._delete_stack_kwargs(stack_name))
            )
            retry_status = await self._poll_deletion(stack_name, timeout=STACK_DELETE_TIMEOUT)
        except ClientError as e:
            logger.debug(f"Retry DeleteStack for '{stack_name}' failed: {e}; will force-delete.")
            return
        if retry_status == DELETE_COMPLETE:
            result.status = StackDeletionStatus.SUCCESS
            result.reason = ""
            logger.debug(f"Stack '{stack_name}': {success_msg}")

    async def _reap_and_retry_networking(
        self, stack_name: str, result: StackDeletionResult
    ) -> None:
        """Reap leftover VPC ENIs blocking deletion, then retry DeleteStack once."""
        if result.status != StackDeletionStatus.FAILED:
            return
        vpc_ids = self._collect_resource_physical_ids(result.resources, "AWS::EC2::VPC")
        if not vpc_ids:
            return

        logger.debug(
            "Stack '%s' failed with %d VPC(s); reaping blocking ENIs before retry",
            stack_name,
            len(vpc_ids),
        )
        # Uses the reaper's default 5-min budget — just long enough to force-clear
        # customer/available ENIs and catch a fast requester-managed release. We no
        # longer wait out the full 20-40 min for requester-managed ENIs: if only
        # those remain, the stack is deferred (see _defer_if_requester_eni_only) and
        # a later run reaps them once the owner releases, rather than idling here.
        reap = await asyncio.to_thread(
            reap_vpc_enis,
            self._session,
            vpc_ids,
            region=self._region,
        )
        async with self._manifest_lock:
            self._manifest.setdefault(stack_name, {})["eni_reap"] = {
                "vpc_ids": vpc_ids,
                "deleted": reap.deleted,
                "detached": reap.detached,
                "remaining": reap.remaining,
            }
        if not reap.reaped_any and reap.remaining:
            # Nothing was clearable — only requester-managed ENIs remain. Their
            # owning service releases them asynchronously (~20-40 min after the
            # owner's own delete), well past this run's bounded wait, so retrying
            # DeleteStack now would just stall again. If the stack's only remaining
            # blockers are the subnet/VPC/SG those ENIs pin, it is eventually-
            # deletable, not stuck: defer it so the run isn't failed for a state
            # that self-heals. Otherwise leave it FAILED for force-delete.
            await self._defer_if_requester_eni_only(stack_name, result, reap.remaining)
            return

        # ENIs are freed, but a leftover NON-default security group (one a managed
        # service created in the VPC, or one a force-deleted stack abandoned) can
        # still block DeleteVpc with DependencyViolation. Delete those now so the
        # re-drive can remove the VPC in a single pass instead of falling through to
        # the slower FORCE_DELETE_STACK + orphan-sweep path.
        sg_remaining = await asyncio.to_thread(
            reap_vpc_security_groups, self._session, vpc_ids, region=self._region
        )
        if sg_remaining:
            async with self._manifest_lock:
                self._manifest.setdefault(stack_name, {})["sg_reap_remaining"] = sg_remaining

        await self._redrive_delete_stack(stack_name, result, "deleted after reaping blocking ENIs")

    async def _reap_and_retry_ipam(self, stack_name: str, result: StackDeletionResult) -> None:
        """Reap leaked child pools blocking a parent IPAM pool, then retry DeleteStack once.

        The IPAM analogue of :meth:`_reap_and_retry_networking`.
        """
        if result.status != StackDeletionStatus.FAILED:
            return
        pool_ids = self._collect_resource_physical_ids(result.resources, "AWS::EC2::IPAMPool")
        if not pool_ids:
            return

        logger.debug(
            f"Stack '{stack_name}' failed with {len(pool_ids)} IPAM pool(s); "
            f"reaping leaked child pools before retry"
        )
        # The stack's own pool ids are BOTH the parents to search under AND the CFN-owned
        # set to exclude: a sourced child NOT in the stack's resource list is a leak to
        # reap, a stack-owned child is left for CloudFormation.
        reap = await asyncio.to_thread(
            reap_ipam_child_pools,
            self._session,
            pool_ids,
            region=self._region,
            stack_owned_pool_ids=set(pool_ids),
        )
        async with self._manifest_lock:
            self._manifest.setdefault(stack_name, {})["ipam_child_pool_reap"] = {
                "parent_pool_ids": pool_ids,
                "deleted": reap.deleted,
                "remaining": reap.remaining,
            }
        if not reap.deleted:
            # No child was confirmed gone to unblock the parent; leave the stack FAILED
            # for the force-delete last resort.
            return

        await self._redrive_delete_stack(
            stack_name, result, "deleted after reaping leaked IPAM child pools"
        )

    async def _defer_if_requester_eni_only(
        self, stack_name: str, result: StackDeletionResult, remaining_enis: list[str]
    ) -> None:
        """Defer a stack blocked solely by requester-managed ENIs, else leave it FAILED.

        The reaper left ``remaining_enis`` (requester-managed interfaces it cannot
        force-release). If the stack's outstanding ``DELETE_FAILED`` resources are
        confined to the networking types those ENIs pin (subnet/VPC/security-group),
        the stack is eventually-deletable: record it — and the resources it left
        behind — in the run's deferred registry so the post-cleanup orphan scan
        excludes them, mark the result ``deferred`` (status stays ``FAILED`` for an
        honest record), and let a later run reap them once the owner releases the
        ENIs. Any non-networking blocker means a genuine, non-self-healing failure,
        so the result is left untouched for the force-delete path.
        """
        blockers = await asyncio.to_thread(self._describe_recent_delete_failures, stack_name)
        non_networking = [
            e
            for e in blockers
            if e["status"] == DELETE_FAILED
            and e["resource_type"] not in _DEFERRABLE_NETWORKING_TYPES
            and e["resource_type"] != "AWS::CloudFormation::Stack"
        ]
        if non_networking:
            logger.debug(
                "Stack '%s' has non-networking blocker(s); not deferring (leaving FAILED): %s",
                stack_name,
                ", ".join(e["logical_id"] for e in non_networking[:5]),
            )
            return

        stack_arn = await self._get_stack_arn(stack_name)
        mark_deferred("AWS::CloudFormation::Stack", stack_arn or stack_name)
        deferred_ids: list[tuple[str, str]] = [
            ("AWS::CloudFormation::Stack", stack_arn or stack_name)
        ]
        for resource in result.resources:
            if resource.physical_id and resource.resource_type in _DEFERRABLE_NETWORKING_TYPES:
                mark_deferred(resource.resource_type, resource.physical_id)
                deferred_ids.append((resource.resource_type, resource.physical_id))
        for eni_id in remaining_enis:
            mark_deferred("AWS::EC2::NetworkInterface", eni_id)
            deferred_ids.append(("AWS::EC2::NetworkInterface", eni_id))

        result.deferred = True
        result.reason = (
            "deferred: subnet/VPC pinned by requester-managed ENIs; AWS releases them "
            "~20-40 min after owner delete (excluded from orphan scan, reaped next run)"
        )
        async with self._manifest_lock:
            self._manifest.setdefault(stack_name, {})["deferred"] = {
                "reason": "requester-managed ENIs pin subnet/VPC/SG until owner releases them",
                "remaining_enis": list(remaining_enis),
                "deferred_ids": [{"type": t, "identifier": i} for t, i in deferred_ids],
            }
        logger.debug(
            "Stack '%s': deferred — %d requester-managed ENI(s) pin its subnet/VPC; "
            "excluded from the failure verdict and orphan scan, reaped by a later run.",
            stack_name,
            len(remaining_enis),
        )

    async def _get_stack_arn(self, stack_name: str) -> str | None:
        """Return the stack's ARN (``StackId``), or None if it cannot be resolved.

        The orphan scanner identifies a CloudFormation stack by its ARN, so the
        deferred registry must record that same form for ``exclude_deferred`` to
        match. Never raises — a missing ARN falls back to the stack name.
        """
        try:
            resp = await asyncio.to_thread(self._client.describe_stacks, StackName=stack_name)
        except ClientError as e:
            logger.debug("Could not resolve ARN for stack '%s': %s", stack_name, e)
            return None
        stacks = resp.get("Stacks", [])
        return stacks[0].get("StackId") if stacks else None

    async def _log_stack_failure_events(self, stack_name: str) -> None:
        """Log the CloudFormation events explaining why a stack failed or stuck.

        Captures the most recent DELETE_FAILED / DELETE_IN_PROGRESS event (with its
        ``ResourceStatusReason``) per logical resource, so a stuck teardown names
        the blocking resource (e.g. a subnet held by a lingering ENI) instead of
        the opaque "unknown". Persisted to the manifest, deduped per stack, and
        never raises — diagnostics must not mask the underlying failure.
        """
        if stack_name in self._events_logged:
            return
        self._events_logged.add(stack_name)
        try:
            events = await asyncio.to_thread(self._describe_recent_delete_failures, stack_name)
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not fetch failure events for '%s': %s", stack_name, e)
            return
        if not events:
            return
        logger.debug("Stack '%s' failure events (%d resource(s)):", stack_name, len(events))
        for event in events:
            logger.debug(
                "  %s (%s) [%s]: %s — %s",
                event["logical_id"],
                event["resource_type"],
                event["physical_id"] or "-",
                event["status"],
                event["reason"] or "(no reason reported)",
            )
        async with self._manifest_lock:
            self._manifest.setdefault(stack_name, {})["failure_events"] = events

    def _describe_recent_delete_failures(self, stack_name: str) -> list[dict]:
        """Return the newest DELETE_FAILED/DELETE_IN_PROGRESS event per logical resource.

        Reads the most recent page of stack events (newest first). Only resources
        whose *latest* event is a delete failure or an unfinished delete are
        returned, so a resource that ultimately deleted is not reported.
        """
        try:
            resp = self._client.describe_stack_events(StackName=stack_name)
        except ClientError as e:
            if is_stack_not_found(e):
                return []
            raise
        seen: set[str] = set()
        events: list[dict] = []
        for event in resp.get("StackEvents", []):  # newest first
            logical_id = event.get("LogicalResourceId", "")
            if logical_id in seen:
                continue
            seen.add(logical_id)  # only the latest event per resource matters
            status = event.get("ResourceStatus", "")
            if status in (DELETE_FAILED, "DELETE_IN_PROGRESS"):
                events.append(
                    {
                        "logical_id": logical_id,
                        "physical_id": event.get("PhysicalResourceId", ""),
                        "resource_type": event.get("ResourceType", ""),
                        "status": status,
                        "reason": event.get("ResourceStatusReason", ""),
                    }
                )
        return events

    # -- Bulk deletion -----------------------------------------------------

    async def delete_all_stacks(
        self, concurrency: int = STACK_DELETE_CONCURRENCY
    ) -> DeletionSummary:
        """Delete all stacks, infrastructure stacks last."""
        all_stacks = await asyncio.to_thread(self.list_stacks)
        env_stacks, infra_stacks = [], []
        for stack in all_stacks:
            (infra_stacks if stack["StackName"].startswith(INFRA_PREFIXES) else env_stacks).append(
                stack
            )
        sem = asyncio.Semaphore(concurrency)

        async def delete_with_limit(name: str) -> StackDeletionResult:
            async with sem:
                return await self.delete_stack(name)

        env_results = await self._delete_stacks_concurrently(
            "environment", env_stacks, delete_with_limit
        )

        # Retry stacks blocked by cross-stack export dependencies.
        # Now that their importing stacks have been deleted, exporters should succeed.
        # NOTE: Single retry handles the common case (A imports B, B deletes first).
        # Multi-level chains (A->B->C) would need iterative retries, but aws-bench
        # scenarios don't have 3+ levels of export dependencies in practice.
        blocked = [
            r
            for r in env_results
            if r.status == StackDeletionStatus.FAILED and r.reason == _EXPORT_BLOCKED_REASON
        ]
        if blocked:
            # Wait for non-blocked failed stacks to finish deleting before retrying.
            # An exporter can only be deleted once ALL its importers are fully gone,
            # but some importers may still be DELETE_IN_PROGRESS (e.g. LambdaMisc takes 1000s+).
            blocked_names = {r.stack_name for r in blocked}
            still_deleting = [
                r.stack_name
                for r in env_results
                if r.stack_name not in blocked_names and r.status == StackDeletionStatus.FAILED
            ]
            if still_deleting:
                logger.debug(
                    "Waiting for %d importer stack(s) to finish deleting before "
                    "retrying blocked exporters: %s",
                    len(still_deleting),
                    ", ".join(still_deleting),
                )
                wait_deadline = time.monotonic() + 300  # max 5 min
                while still_deleting and time.monotonic() < wait_deadline:
                    await asyncio.sleep(15)
                    checks = await asyncio.gather(
                        *[asyncio.to_thread(self._stack_exists, name) for name in still_deleting],
                        return_exceptions=True,
                    )
                    still_deleting = [
                        name
                        for name, exists in zip(still_deleting, checks)
                        if exists is True or isinstance(exists, BaseException)
                    ]
                if still_deleting:
                    logger.warning(
                        "%d importer stack(s) still exist after 5m wait: %s",
                        len(still_deleting),
                        ", ".join(still_deleting),
                    )

            logger.debug(
                "Retrying %d stack(s) that were blocked by export dependencies: %s",
                len(blocked),
                ", ".join(r.stack_name for r in blocked),
            )
            retry_results = await self._delete_stacks_concurrently(
                "export-blocked (retry)",
                [{"StackName": r.stack_name} for r in blocked],
                delete_with_limit,
            )
            # Replace blocked results with retry outcomes
            blocked_names = {r.stack_name for r in blocked}
            env_results = [r for r in env_results if r.stack_name not in blocked_names]
            env_results.extend(retry_results)

        infra_results = await self._maybe_delete_infra(infra_stacks, env_results, delete_with_limit)

        summary = DeletionSummary(results=list(env_results) + list(infra_results))
        try:
            write_json(self._manifest, self._manifest_path)
        except Exception as exc:
            logger.debug("Failed to save manifest to %s: %s", self._manifest_path, exc)
        return summary

    async def _delete_stacks_concurrently(
        self,
        label: str,
        stacks: list[dict],
        delete_fn,
    ) -> list[StackDeletionResult]:
        logger.debug("Deleting %d %s stacks...", len(stacks), label)
        results = await asyncio.gather(
            *[delete_fn(stack["StackName"]) for stack in stacks], return_exceptions=True
        )
        # Re-raise a captured shutdown rather than record the cancelled stack as a failure.
        reraise_if_cancelled(results)
        # Convert exceptions to failed results
        processed: list[StackDeletionResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                stack_name = stacks[i]["StackName"]
                logger.error("Stack deletion task failed for '%s': %s", stack_name, result)
                await self._log_stack_failure_events(stack_name)
                processed.append(
                    StackDeletionResult(
                        stack_name=stack_name,
                        status=StackDeletionStatus.FAILED,
                        reason=f"Unhandled exception: {result}",
                    )
                )
            else:
                # Type narrowing: result is StackDeletionResult after isinstance check
                processed.append(cast(StackDeletionResult, result))
        return processed

    async def _maybe_delete_infra(
        self,
        infra_stacks: list[dict],
        env_results: list[StackDeletionResult],
        delete_fn,
    ) -> list[StackDeletionResult]:
        # Check which failed stacks still exist (run in parallel). A DEFERRED stack
        # blocks infra deletion too: it is still present, and its eventual
        # DeleteStack retry (once the requester-managed ENIs release) needs the CDK
        # bootstrap cfn-exec-role that lives in the CDKToolkit stack. Deleting
        # CDKToolkit now would remove that role and permanently wedge the deferred
        # stack — CloudFormation could no longer assume the role to finish deleting
        # it. So keep the bootstrap alive until every FAILED-or-deferred env stack
        # is truly gone; a later run tears CDKToolkit down once they are.
        failed_results = [r for r in env_results if r.status == StackDeletionStatus.FAILED]
        existence_checks = await asyncio.gather(
            *[asyncio.to_thread(self._stack_exists, r.stack_name) for r in failed_results],
            return_exceptions=True,
        )
        env_failures = [
            result
            for result, exists in zip(failed_results, existence_checks)
            if not isinstance(exists, Exception) and exists
        ]
        if env_failures and infra_stacks:
            logger.debug(
                "Skipping %d infrastructure stack(s) — %d environment stack(s) still exist.",
                len(infra_stacks),
                len(env_failures),
            )
            return []
        return await self._delete_stacks_concurrently("infrastructure", infra_stacks, delete_fn)

    # -- Verification ------------------------------------------------------

    async def _verify_deletion(self, result: StackDeletionResult) -> None:
        """Record per-resource existence into the manifest (read-only diagnostics).

        Runs only for a still-FAILED stack, to capture which resources CloudFormation
        abandoned. Deletes nothing — the Phase-3 sweep owns removing leftovers.
        """
        if not result.resources:
            return

        verified = await self._verifier.verify_resources(result.resources)
        async with self._manifest_lock:
            verification_dicts = []
            for v in verified:
                v_dict = asdict(v)
                v_dict["existence_status"] = v.existence_status.value
                verification_dicts.append(v_dict)
            self._manifest.setdefault(result.stack_name, {})["verification"] = verification_dicts
