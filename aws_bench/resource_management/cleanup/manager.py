"""Cleanup orchestration manager for CloudFormation stacks and AWS resources.

Provides high-level cleanup operations by account ID, with support for
targeted stack deletion or full account cleanup across regions.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from aws_bench.account_management.constants import CFN_OPS_ROLE_NAME
from aws_bench.constants import OUTPUT_DIR
from aws_bench.exceptions import OperationCancelled
from aws_bench.logging.logger import get_logger, log_context
from aws_bench.resource_management.ccapi.models import DeletionFailureEvent, Resource, ScanResult
from aws_bench.resource_management.cleanup.account_scanner import AccountScanner
from aws_bench.resource_management.cleanup.models import (
    AccountScanResult,
    CleanupSummary,
    DeletionSummary,
    RegionResult,
    SnapshotResources,
    StackDeletionStatus,
    partition_by_scope,
)
from aws_bench.resource_management.cleanup.resource_sweeper import ResourceSweeper
from aws_bench.resource_management.cleanup.stack_deleter import StackDeleter
from aws_bench.resource_management.deferred import deferred_scope
from aws_bench.resource_management.exceptions import SnapshotNotFoundError
from aws_bench.resource_management.snapshot.manager import SnapshotManager
from aws_bench.resource_management.snapshot.models import SnapshotStage
from aws_bench.resource_management.utils.cloudformation import is_stack_not_found
from aws_bench.resource_management.verify.comparators import find_new_resources
from aws_bench.utils.concurrent import build_client, reraise_if_cancelled
from aws_bench.utils.credentials_provider import create_regional_session
from aws_bench.utils.regions import get_enabled_regions

logger = get_logger(__name__)


class CleanupManager:
    """Orchestrates cleanup operations across AWS accounts and regions.

    This manager coordinates stack deletion, verification, and post-cleanup scanning
    to ensure comprehensive resource cleanup with minimal orphans.
    """

    def __init__(
        self,
        session: boto3.Session,
        output_dir: Path | None = None,
        env_name: str | None = None,
        account_id: str | None = None,
    ) -> None:
        """Initialize with a boto3 session.

        Args:
            session: boto3 Session with credentials for the target account
            output_dir: Directory for cleanup output.
                Defaults to .aws-bench/cleanup/
            env_name: Environment name for loading baseline snapshots (optional)
            account_id: Target account id. Callers build ``session`` from it, so
                passing it avoids an STS ``get_caller_identity`` round-trip; falls
                back to that lookup only when a caller cannot supply it.
        """
        self._session = session
        self._account_id: str | None = account_id
        self._env_name = env_name
        if output_dir is None:
            output_dir = OUTPUT_DIR / "cleanup"
        self._output_dir = output_dir
        # Artifacts (summary, run metadata, per-region manifests) write directly here, and the
        # metadata/summary writers only warn on OSError — a missing dir would silently drop every
        # artifact. Create it once up front (the trial's phase dir isn't created by trial setup).
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if env_name:
            self._snapshot_mgr = SnapshotManager()
        else:
            self._snapshot_mgr = None

    def _get_account_id(self) -> str:
        """Get the AWS account ID from the session credentials."""
        if self._account_id is None:
            sts = build_client(self._session, "sts")
            account_id: str = sts.get_caller_identity()["Account"]
            self._account_id = account_id
        return self._account_id

    @property
    def _cfn_role_arn(self) -> str | None:
        """CFN ops role ARN derived from account ID, or None (fallback to stack's role)."""
        if self._account_id:
            return f"arn:aws:iam::{self._account_id}:role/{CFN_OPS_ROLE_NAME}"
        return None

    async def cleanup_all_stacks(
        self,
        regions: list[str] | None = None,
        *,
        all_regions: bool = False,
        sweep_post_setup: bool = True,
    ) -> CleanupSummary:
        """Delete all stacks then scan for orphans, under a per-run deferred-deletion scope.

        The scope lets a delete handler defer an eventually-consistent resource
        so the post-cleanup orphan scan does not report it and fail the run.
        """
        with deferred_scope():
            return await self._cleanup_all_stacks_impl(
                regions, all_regions=all_regions, sweep_post_setup=sweep_post_setup
            )

    async def sweep_post_setup_residuals(
        self,
        regions: list[str] | None = None,
        *,
        all_regions: bool = False,
    ) -> None:
        """Sweep run-created residuals (current - POST_SETUP) across regions.

        The in-cleanup Phase 1, exposed so the scenario path can run it BEFORE
        ``cleanup.sh``. Best-effort per region (see ``_run_sweep_phase``).
        """
        with deferred_scope():
            resolved = await self._resolve_regions(regions, all_regions=all_regions)
            setup = self._load_snapshot_resource_ids(SnapshotStage.POST_SETUP)
            await asyncio.gather(
                *[
                    self._run_sweep_phase(
                        region, self._delete_resources_created_after_setup(region, setup)
                    )
                    for region in resolved
                ]
            )

    async def _resolve_regions(self, regions: list[str] | None, *, all_regions: bool) -> list[str]:
        """Resolve the region list for a cleanup/sweep run.

        ``all_regions`` ignores the baseline and cleans every enabled region (or an
        explicit ``regions`` list); otherwise the baseline narrows the set, falling
        back to discovery. Validates a non-empty list of strings.
        """
        if all_regions:
            regions = regions if regions is not None else self._discover_regions()
        else:
            regions = await self._limit_to_baseline_regions(regions)
            if regions is None:
                regions = self._discover_regions()

        if not regions:
            raise ValueError("regions list cannot be empty")
        if not all(isinstance(r, str) for r in regions):
            invalid = [type(r).__name__ for r in regions if not isinstance(r, str)]
            raise ValueError(f"All regions must be strings, found: {', '.join(set(invalid))}")
        return regions

    async def _cleanup_all_stacks_impl(
        self,
        regions: list[str] | None = None,
        *,
        all_regions: bool = False,
        sweep_post_setup: bool = True,
    ) -> CleanupSummary:
        """Delete all CloudFormation stacks across regions.

        Args:
            regions: Regions to clean.
                If None and baseline exists, uses baseline regions.
                If None and no baseline, discovers all enabled regions.
                If specified and baseline exists, uses intersection of both.
            all_regions: If True, ignore the baseline snapshot and clean every
                enabled region. An explicit ``regions`` list still takes
                precedence; otherwise all enabled regions are discovered. Use
                this to reclaim resources that may have leaked outside the
                baseline regions.
            sweep_post_setup: If True (default), run the in-cleanup Phase 1
                post-setup residual sweep per region. Set False when the caller
                already swept residuals before cleanup (e.g. the scenario trial).

        Returns:
            CleanupSummary with per-region results and orphaned resources.

        Raises:
            ValueError: If regions contains non-string elements or resolves to empty
            RuntimeError: If region discovery or directory creation fails
        """
        regions = await self._resolve_regions(regions, all_regions=all_regions)

        run_dir = self._output_dir
        self._write_metadata(run_dir, {"regions": regions})

        raw_results = await asyncio.gather(
            *[
                self._cleanup_region_in_phases(region, run_dir, sweep_post_setup=sweep_post_setup)
                for region in regions
            ],
            return_exceptions=True,
        )

        # Re-raise a captured shutdown so cleanup unwinds instead of recording
        # the cancelled region as an error.
        reraise_if_cancelled(raw_results)
        region_results = []
        for i, result in enumerate(raw_results):
            if isinstance(result, Exception):
                logger.error("[%s] Region cleanup raised: %s", regions[i], result)
                region_results.append(RegionResult(region=regions[i], error=str(result)))
            else:
                region_results.append(result)

        # Global resources (IAM roles/policies, Route53, CloudFront) were withheld
        # from the concurrent per-region sweeps because they exist account-wide: a
        # region that finished could otherwise delete a role a still-deleting region
        # depends on. Sweep them once now, behind this barrier, after all regions
        # finish. Always run as last-resort cleanup regardless of phase-2 outcome:
        # orphaned global resources (agent-created IAM roles etc.) must be reaped
        # even when stacks survived. The sweep itself is best-effort (errors absorbed
        # by ``_run_sweep_phase``).
        #
        # Deferred stacks (still present, blocked only by requester-managed ENIs that
        # AWS releases later) do NOT count as survivors here, and that is safe: a
        # stack is only deferred when its sole remaining blockers are networking
        # (``_defer_if_requester_eni_only`` refuses to defer on any custom-resource
        # blocker), so its custom-resource provider roles have already done their job
        # and are no longer needed, and its cfn-exec-role is infra-protected
        # (never swept). The deferred stack's own resources are in the run's deferred
        # registry, which ``_scan_region`` excludes from the sweep scan.
        survivors = [r for r in region_results if r.stacks_failed or r.error]
        if survivors:
            logger.debug(
                "Phase 2 had failures: %d region(s) still have surviving stacks (%s); "
                "global-resource sweep proceeds anyway as last-resort cleanup",
                len(survivors),
                ", ".join(r.region for r in survivors),
            )
        await self._run_sweep_phase(regions[0], self._sweep_global_leftovers(regions))

        try:
            orphan_scan = await self._scan_orphaned_resources(run_dir, regions, include_infra=True)
            orphan_map = orphan_scan.orphaned_resources
            region_orphan_counts = orphan_scan.region_counts
            scan_incomplete = bool(orphan_scan.failed_regions)
        except Exception as exc:
            logger.error("Post-cleanup scan failed: %s", exc)
            orphan_map = {}
            region_orphan_counts = {}
            scan_incomplete = True

        # Update region results with orphan counts
        for region_result in region_results:
            region_result.orphan_count = region_orphan_counts.get(region_result.region, 0)

        summary = CleanupSummary(
            regions=region_results,
            orphaned_resources=orphan_map,
            run_dir=str(run_dir),
            scan_incomplete=scan_incomplete,
        )
        self._log_summary(summary)
        self._save_summary(summary, run_dir)
        return summary

    async def cleanup_stack(self, stack_name: str) -> CleanupSummary:
        """Delete a single CloudFormation stack.

        Args:
            stack_name: Name or ARN of the stack to delete

        Returns:
            CleanupSummary with the result of this single stack deletion

        Raises:
            ValueError: If stack_name is empty or if the stack is not found in any region
        """
        if not stack_name or not stack_name.strip():
            raise ValueError("stack_name cannot be empty")

        region = await self._find_stack_region(stack_name)
        if not region:
            raise ValueError(f"Stack '{stack_name}' not found in any region")

        run_dir = self._output_dir

        with log_context(region):
            self._write_metadata(run_dir, {"stack_name": stack_name})

            logger.debug("Deleting stack '%s'...", stack_name)
            deleter = StackDeleter(
                self._session,
                region=region,
                manifest_path=run_dir / f"{region}.json",
                cfn_role_arn=self._cfn_role_arn,
            )
            result = await deleter.delete_stack(stack_name)

            succeeded = [stack_name] if result.status == StackDeletionStatus.SUCCESS else []
            failed = [stack_name] if result.status == StackDeletionStatus.FAILED else []

            # This path runs no orphan scan, so a FORCE_DELETE-abandoned resource is
            # only knowable from the deletion result — surface it here so the caller
            # (reset) can fail closed instead of absorbing it into a fresh baseline.
            abandoned = result.abandoned_resources
            orphaned: dict[str, list[str]] = {}
            for r in abandoned:
                orphaned.setdefault(r.resource_type, []).append(r.physical_id or r.logical_id)

            region_result = RegionResult(
                region=region,
                stacks_found=1,
                stacks_deleted=len(succeeded),
                stacks_failed=failed,
                orphan_count=len(abandoned),
            )

            summary = CleanupSummary(
                regions=[region_result],
                orphaned_resources=orphaned,
                run_dir=str(run_dir),
            )
            self._log_summary(summary)
            self._save_summary(summary, run_dir)
            return summary

    def _write_metadata(self, run_dir: Path, context: dict) -> None:
        """Write run metadata to a JSON file."""
        try:
            metadata = {
                **context,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "account_id": self._get_account_id(),
            }
            (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
        except OSError as exc:
            logger.warning("Failed to write run metadata: %s", exc)

    def _discover_regions(self) -> list[str]:
        """List all enabled regions in the account."""
        return get_enabled_regions(self._session)

    async def _limit_to_baseline_regions(self, regions: list[str] | None) -> list[str] | None:
        """Limit regions to baseline snapshot regions if available.

        Args:
            regions: Requested regions to clean, or None to use all baseline regions

        Returns:
            - If baseline exists and regions=None: baseline regions
            - If baseline exists and regions specified: intersection of both
            - If no baseline: original regions (may be None)
        """
        if not self._snapshot_mgr or not self._env_name:
            return regions

        try:
            account_id = self._get_account_id()
            snapshot = await asyncio.to_thread(
                self._snapshot_mgr.load_snapshot, self._env_name, account_id
            )
            if snapshot.regions:
                baseline_regions = set(snapshot.regions)

                # If no regions specified, use all baseline regions
                if regions is None:
                    logger.debug(
                        f"Using {len(baseline_regions)} baseline region(s): "
                        f"{', '.join(sorted(baseline_regions))}"
                    )
                    return list(baseline_regions)

                # Otherwise, take intersection
                requested_regions = set(regions)
                limited_regions = list(baseline_regions & requested_regions)

                # Handle empty intersection
                if not limited_regions:
                    logger.debug(
                        f"No overlap between requested regions {sorted(requested_regions)} "
                        f"and baseline regions {sorted(baseline_regions)}. "
                        f"Falling back to requested regions."
                    )
                    return regions

                if set(limited_regions) != requested_regions:
                    logger.debug(
                        f"Limiting cleanup to {len(limited_regions)} baseline region(s): "
                        f"{', '.join(sorted(limited_regions))}"
                    )
                    logger.debug(
                        f"Baseline regions: {sorted(baseline_regions)}, "
                        f"requested regions: {sorted(requested_regions)}"
                    )

                return limited_regions
        except SnapshotNotFoundError:
            logger.debug("No baseline snapshot found, using requested regions")
        except Exception as exc:
            logger.warning(f"Failed to load baseline snapshot: {exc}")

        return regions

    async def _find_stack_region(self, stack_name: str) -> str | None:
        """Find which region contains the given stack.

        Returns None if stack not found.
        """
        regions = self._discover_regions()

        async def check_region(region: str) -> str | None:
            with log_context(region):
                try:
                    cfn = build_client(self._session, "cloudformation", region_name=region)
                    resp = await asyncio.to_thread(cfn.describe_stacks, StackName=stack_name)
                    if resp.get("Stacks"):
                        return region
                except ClientError as e:
                    if is_stack_not_found(e):
                        return None
                    # For other errors (permissions, throttling, etc.), log and re-raise
                    logger.warning("Error checking stack '%s': %s", stack_name, e)
                    raise
                return None

        results = await asyncio.gather(*[check_region(r) for r in regions], return_exceptions=True)
        for result in results:
            if isinstance(result, str):
                return result
        return None

    async def _cleanup_region_in_phases(
        self, region: str, run_dir: Path, *, sweep_post_setup: bool = True
    ) -> RegionResult:
        """Delete a region's scenario resources in up to three readable phases.

        The post-setup sweep (phase 1) is skipped when the caller ran it earlier.

        1. delete run-created residuals (current - setup) so stacks can delete cleanly;
        2. delete every scenario CloudFormation stack (CFN cascades child deletion);
        3. sweep whatever survived that is NOT in the init baseline (AWS defaults kept).

        The phase-2 stack result is what this region reports, so a failure in the
        best-effort sweep phases (1 or 3) must not discard it: those sweeps run
        inside ``_run_sweep_phase``, which absorbs a generic error but lets a
        cooperative-shutdown cancellation propagate to unwind the whole run.

        Phase 3 also sweeps CDK bootstrap/toolkit resources (``include_infra``) —
        chiefly the CDKToolkit stack's assets bucket, which CFN retains on stack
        deletion. They are absent from the init baseline (captured before ``cdk
        bootstrap``), so the ``current - init`` diff surfaces them for the sweep.
        """
        setup = self._load_snapshot_resource_ids(SnapshotStage.POST_SETUP)
        init = self._load_snapshot_resource_ids(SnapshotStage.PRE_SETUP)

        if sweep_post_setup:
            await self._run_sweep_phase(
                region, self._delete_resources_created_after_setup(region, setup)
            )
        region_result = await self._cleanup_single_region(region, run_dir)  # phase 2: stacks
        # Phase 3 sweeps this region's non-baseline (regional) leftovers, and
        # include_infra reclaims the CDKToolkit stack's retained regional assets.
        # Always run as last-resort cleanup regardless of phase-2 outcome: orphaned
        # resources (agent-created, not stack-managed) must be reaped even when a
        # stack is stuck in DELETE_FAILED, and the sweep itself is best-effort
        # (errors absorbed by ``_run_sweep_phase``).
        if region_result.stacks_failed or region_result.error:
            logger.debug(
                "[%s] Phase 2 had failures (%d stack(s) survived, error=%s); "
                "Phase 3 sweep proceeds anyway as last-resort cleanup",
                region,
                len(region_result.stacks_failed),
                region_result.error or "DELETE_FAILED",
            )
        await self._run_sweep_phase(
            region,
            self._delete_resources_not_in_init_snapshot(region, init, include_infra=True),
        )
        return region_result

    async def _run_sweep_phase(self, region: str, phase: Awaitable[None]) -> None:
        """Await a best-effort sweep phase; absorb errors, but propagate cancellation.

        The sweep phases (1 and 3) are cleanup-of-last-resort around the phase-2
        stack deletion whose result the region reports. A generic error here must
        not sink that result, so it is logged and swallowed. A cooperative shutdown
        (``OperationCancelled`` / ``asyncio.CancelledError``) is NOT an error — it
        must re-raise so the run unwinds instead of being recorded as a region
        outcome.
        """
        try:
            await phase
        except (OperationCancelled, asyncio.CancelledError):
            raise
        except Exception as exc:
            # Recovered: stacks already deleted in phase 2, orphan scan backstops leftovers.
            logger.warning(f"[{region}] Sweep phase failed (stacks already handled): {exc}")

    def _scan_region_resources(self, region: str, *, include_infra: bool = False) -> ScanResult:
        """Current per-region scan (detected resources + types that failed to enumerate).

        Returns the full ``ScanResult`` so the phases can pass ``failed`` into the
        diff and skip a type that failed to enumerate rather than deleting its live
        resources as if they were new. ``include_infra`` keeps CDK bootstrap/toolkit
        resources in the scan (filtered out by default).
        """
        region_session = create_regional_session(self._session, region)
        return AccountScanner(region_session, account_id=self._get_account_id()).scan_region(
            region, include_infra=include_infra
        )

    async def _delete_resources_created_after_setup(
        self, region: str, setup: SnapshotResources | None
    ) -> None:
        """Phase 1 — delete run-created residuals (current - setup) in ``region``.

        These are the resources a task created after deploy (e.g. task-made ENIs)
        that commonly wedge a CloudFormation stack in DELETE_FAILED. Deleting them
        first lets the stack delete cleanly in phase 2. Skipped when no setup
        baseline exists (nothing to diff against). A type that failed to enumerate
        in either the live scan or the baseline is excluded from the diff, so a
        transient enumeration failure never turns a baseline resource into a
        "residual" that gets deleted.
        """
        if setup is None:
            logger.debug("Phase 1: no setup baseline; skipping residual deletion")
            return
        current = await asyncio.to_thread(self._scan_region_resources, region)
        residuals = find_new_resources(
            current.detected, setup.resource_ids, current.failed, setup.failed_types
        )
        # Global resources (IAM etc.) surface in every region's scan and are swept
        # once after all regions finish (see ``_sweep_global_leftovers``), never
        # per-region: this phase runs concurrently across regions, so a global
        # role deleted here could be one another region's stack still needs.
        residuals, _global = partition_by_scope(residuals)
        if not residuals:
            logger.debug(f"Phase 1: no run-created regional residuals in {region}")
            return
        logger.debug(f"Phase 1: deleting run-created residuals in {region}")
        failures = await ResourceSweeper(create_regional_session(self._session, region)).delete(
            residuals
        )
        self._warn_on_sweep_failures(region, "Phase 1", failures)

    async def _delete_resources_not_in_init_snapshot(
        self, region: str, init: SnapshotResources | None, *, include_infra: bool = False
    ) -> None:
        """Phase 3 — sweep everything in ``region`` that is NOT in the init snapshot.

        Catches stack leftovers (from a DELETE_FAILED stack) and any residual the
        first phase missed. The init snapshot is the account's pristine pre-deploy
        state, so its resources (AWS defaults: default VPC family, default KMS keys)
        are kept.

        Fails closed when no init snapshot exists: without the pristine snapshot to
        diff against, a sweep could delete the account's own defaults (e.g. its
        default VPC), so the sweep is skipped entirely and a loud warning is logged.
        A type that failed to enumerate in either scan is excluded from the diff for
        the same reason as phase 1.

        ``include_infra`` keeps CDK bootstrap/toolkit resources in the scan so the
        CDKToolkit stack's retained assets bucket is reaped. Bootstrap resources are
        absent from the init snapshot (captured before ``cdk bootstrap``), so the
        same init diff that protects account defaults also treats them as leftovers.
        """
        if init is None:
            logger.warning(
                f"Phase 3: no init snapshot for {region}; skipping sweep (fail-closed) "
                "so the account's default resources are not deleted"
            )
            return
        current = await asyncio.to_thread(
            self._scan_region_resources, region, include_infra=include_infra
        )
        leftovers = find_new_resources(
            current.detected, init.resource_ids, current.failed, init.failed_types
        )
        # Withhold global resources (IAM etc.) — they are swept once, after all
        # regions finish, by ``_sweep_global_leftovers``. Sweeping them here would
        # race the other regions' concurrent phase 3 / still-running phase 2.
        leftovers, _global = partition_by_scope(leftovers)
        if not leftovers:
            logger.debug(f"Phase 3: nothing to sweep in {region}")
            return
        logger.debug(f"Phase 3: sweeping non-baseline leftovers in {region}")
        failures = await ResourceSweeper(create_regional_session(self._session, region)).delete(
            leftovers
        )
        self._warn_on_sweep_failures(region, "Phase 3", failures)

    async def _sweep_global_leftovers(self, regions: list[str]) -> None:
        """Sweep account-wide leftovers once, after every region finishes stack deletion.

        Global-typed resources are withheld from the concurrent per-region sweeps
        (phases 1 and 3) because sweeping them there races: a finished region could
        delete a role a still-deleting region's stack needs. This runs once, and the
        caller gates it on every region having deleted all its stacks, so nothing
        still references the resources it removes (notably app custom-resource
        provider roles; the CDK bootstrap roles are infra-protected and removed with
        the CDKToolkit stack in phase 2).

        ``GLOBAL_RESOURCE_TYPES`` mixes two shapes and this handles both:

        * **Truly global** (IAM, Route53, CloudFront) — CCAPI lists the same
          identifier in *every* region's scan. Deduped by identifier so each is
          deleted exactly once (in the first region that surfaces it; its endpoint
          is account-wide so the region is immaterial).
        * **Region-homed** (e.g. ``ResourceExplorer2::Index``) — only its own
          region's scan lists it, and its CCAPI delete must target that region's
          endpoint. Scanning a single region would miss those homed elsewhere, so
          every region is scanned and each region's leftovers are swept with that
          region's own session.

        Diffs against the pre-setup (init) snapshot so the account's original
        resources are kept, and fails closed (skips, warns) when no init snapshot
        exists — without the pristine baseline a sweep could delete the account's
        own roles.
        """
        init = self._load_snapshot_resource_ids(SnapshotStage.PRE_SETUP)
        if init is None:
            logger.warning(
                "Global-resource sweep skipped (fail-closed): no init snapshot, so the "
                "account's original global resources are not deleted"
            )
            return

        # Scan every region up front so a region-homed global (listed only in its
        # own region) is never missed, then process in order to dedup truly-global
        # resources that surface in multiple regions.
        scans = await asyncio.gather(
            *[asyncio.to_thread(self._scan_region_resources, region) for region in regions],
            return_exceptions=True,
        )
        reraise_if_cancelled(scans)

        swept_ids: set[str] = set()
        for region, scan in zip(regions, scans):
            if isinstance(scan, BaseException):
                # Cancellation was already re-raised above; anything left is a real
                # scan error for this region — skip it and sweep the others.
                logger.debug(
                    "Global sweep: scan of %s failed (%s); skipping this region", region, scan
                )
                continue
            leftovers = find_new_resources(
                scan.detected, init.resource_ids, scan.failed, init.failed_types
            )
            _regional, global_leftovers = partition_by_scope(leftovers)
            # Drop identifiers already swept from an earlier region — a truly-global
            # resource surfaces identically in every region and must be deleted once.
            deduped: dict[str, list[dict]] = {}
            for rtype, items in global_leftovers.items():
                fresh = [i for i in items if i.get("Identifier", "") not in swept_ids]
                if fresh:
                    deduped[rtype] = fresh
            if not deduped:
                continue
            for items in deduped.values():
                swept_ids.update(i.get("Identifier", "") for i in items if i.get("Identifier", ""))
            total = sum(len(items) for items in deduped.values())
            logger.debug(
                "Global-resource sweep: removing %d account-wide leftover(s) via %s now "
                "that all regions have cleared their stacks",
                total,
                region,
            )
            failures = await ResourceSweeper(create_regional_session(self._session, region)).delete(
                deduped
            )
            self._warn_on_sweep_failures(region, "Global sweep", failures)

    def _warn_on_sweep_failures(
        self, region: str, phase: str, failures: dict[Resource, DeletionFailureEvent]
    ) -> None:
        """Log a WARNING summarizing resources a sweep phase could not delete."""
        if not failures:
            return
        sample = ", ".join(f"{r.type} {r.identifier}" for r in list(failures)[:5])
        logger.warning(
            f"{phase}: could not sweep {len(failures)} resource(s) in {region} (sample: {sample})"
        )

    def _load_snapshot_resource_ids(self, stage: SnapshotStage) -> SnapshotResources | None:
        """Load a snapshot's resources and failed types, or None if absent.

        The failed types travel with the resource ids so the destructive diff can
        skip a type that failed to enumerate in the snapshot instead of treating it
        as having had zero resources.
        """
        if self._snapshot_mgr is None or self._env_name is None:
            return None
        try:
            snapshot = self._snapshot_mgr.load_snapshot(
                self._env_name, self._get_account_id(), stage=stage
            )
        except SnapshotNotFoundError:
            logger.debug(f"No {stage} snapshot for {self._env_name}/{self._get_account_id()}")
            return None
        return SnapshotResources(
            resource_ids=snapshot.resource_ids,
            failed_types=snapshot.failed_resource_types,
        )

    async def _cleanup_single_region(self, region: str, run_dir: Path) -> RegionResult:
        """Delete all stacks in a single region."""
        with log_context(region):
            result = RegionResult(region=region)
            try:
                deleter = StackDeleter(
                    self._session,
                    region=region,
                    manifest_path=run_dir / f"{region}.json",
                    cfn_role_arn=self._cfn_role_arn,
                )

                stacks = await asyncio.to_thread(deleter.list_stacks)
                if not stacks:
                    return result

                result.stacks_found = len(stacks)
                logger.debug("Cleaning up %d stack(s)...", len(stacks))
                summary: DeletionSummary = await deleter.delete_all_stacks()

                result.stacks_deleted = len(summary.succeeded)
                result.stacks_failed = [r.stack_name for r in summary.failed]

                for r in summary.succeeded:
                    logger.debug("✓ %s", r.stack_name)
                for r in summary.deferred:
                    logger.warning("⏳ %s (deferred): %s", r.stack_name, r.reason)
                for r in summary.failed:
                    logger.error("✗ %s: %s", r.stack_name, r.reason)
            except Exception as exc:
                logger.error("Cleanup failed: %s", exc)
                result.error = str(exc)

            return result

    async def _scan_orphaned_resources(
        self, run_dir: Path, regions: list[str], *, include_infra: bool = False
    ) -> AccountScanResult:
        """Scan for orphaned resources across regions after cleanup.

        Subtracts the pre-setup snapshot (the account's default resources
        captured at init) so only resources created after init are reported as
        orphans. Falls back to reporting all non-infra resources when no
        pre-setup snapshot exists.

        ``include_infra`` keeps CDK bootstrap/toolkit resources in the report so
        the CDKToolkit stack's retained leftovers surface as orphans.

        Returns:
            AccountScanResult with orphaned resources and region counts
        """
        snapshot = self._load_snapshot_resource_ids(SnapshotStage.PRE_SETUP)
        predeploy_snapshot = snapshot.resource_ids if snapshot else None
        return await asyncio.to_thread(
            AccountScanner(self._session, account_id=self._get_account_id()).run,
            run_dir,
            regions,
            predeploy_snapshot,
            include_infra=include_infra,
        )

    def _log_summary(self, summary: CleanupSummary) -> None:
        """Log the cleanup summary to file (verbose)."""
        logger.debug("=" * 60)
        logger.debug("DEEP CLEANUP SUMMARY")
        logger.debug("=" * 60)
        for r in summary.regions:
            if r.error:
                logger.error("  %s: ERROR — %s", r.region, r.error)
            elif r.stacks_found == 0:
                logger.debug("  %s: no stacks", r.region)
            else:
                status = "✓" if not r.stacks_failed else "✗"
                orphan_info = f", {r.orphan_count} orphans" if r.orphan_count > 0 else ""
                logger.debug(
                    "  %s: %s %d/%d deleted%s",
                    r.region,
                    status,
                    r.stacks_deleted,
                    r.stacks_found,
                    orphan_info,
                )
                for name in r.stacks_failed:
                    logger.error("    FAILED: %s", name)
        logger.debug("-" * 60)
        logger.info(
            "Total: %d stacks, %d deleted, %d failed",
            summary.total_stacks,
            summary.total_deleted,
            summary.total_failed,
        )
        logger.info("Orphaned resources: %d", summary.total_orphaned)
        if summary.scan_incomplete:
            status = "INCOMPLETE (orphan scan did not finish)"
        elif summary.all_stacks_succeeded:
            status = "PASS"
        else:
            status = "FAIL"
        logger.info("Stacks deletion status: %s", status)
        logger.debug("=" * 60)

    def _save_summary(self, summary: CleanupSummary, run_dir: Path) -> None:
        """Save the cleanup summary to a JSON file."""
        try:
            # Filter out regions with no stacks for cleaner JSON
            filtered_summary = CleanupSummary(
                regions=[r for r in summary.regions if r.stacks_found > 0 or r.error],
                orphaned_resources=summary.orphaned_resources,
                run_dir=summary.run_dir,
                scan_incomplete=summary.scan_incomplete,
            )

            (run_dir / "summary.json").write_text(
                json.dumps(asdict(filtered_summary), indent=2, default=str)
            )
        except OSError as exc:
            logger.warning("Failed to write summary: %s", exc)
