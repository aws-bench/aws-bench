"""Verification manager for account state validation."""

from __future__ import annotations

import dataclasses
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any

import boto3

from aws_bench.logging.logger import get_logger, log_context
from aws_bench.resource_management.ccapi.models import MAX_WORKERS_HEAVY, ScanResult
from aws_bench.resource_management.deferred import exclude_deferred
from aws_bench.resource_management.exceptions import SnapshotNotFoundError
from aws_bench.resource_management.scanner import make_scanner
from aws_bench.resource_management.snapshot.manager import SnapshotManager
from aws_bench.resource_management.snapshot.models import (
    Snapshot,
    SnapshotStage,
)
from aws_bench.resource_management.verify.comparators import count_resources, find_new_resources
from aws_bench.resource_management.verify.drift_detector import DriftDetector
from aws_bench.resource_management.verify.models import (
    AccountVerifyResult,
    RegionVerifyResult,
    VerifyResult,
)
from aws_bench.resource_management.verify.ownership import AwsManagedOwnershipProbe
from aws_bench.resource_management.verify.stack_inspector import StackInspector
from aws_bench.scenario.hashing import compute_scenario_hash
from aws_bench.utils.concurrent import interruptible_executor, raise_if_shutdown
from aws_bench.utils.credentials_provider import create_regional_session

logger = get_logger(__name__)


# Error signals that a baseline-tracked resource type cannot exist in the scanned
# region at all — the service or type is unavailable there — as opposed to an
# enumeration gap that could hide a leaked resource. A type that cannot exist in the
# region cannot hold an orphan, so a persistent scan failure carrying one of these
# signals is tolerated by ``_check_new_resources`` instead of failing verify closed.
#
# These are stable region/service-availability errors observed live across the CCAPI
# listers (e.g. GameLift → UnsupportedRegionException, EC2::CarrierGateway →
# UnsupportedOperation, IoTSiteWise → InvalidRequestException, Lightsail →
# InvalidInputException, Notifications/Bedrock prompt-routers → ValidationException,
# CUR → endpoint connect timeout). They are deliberately NOT the transient
# throttling/5xx codes the scanner already retries: a residual transient failure
# leaves a type genuinely un-enumerated and must still fail closed. AccessDenied is
# likewise excluded — with the broad scan role it means a real permission gap we
# cannot see past, not region unavailability.
_REGION_UNAVAILABLE_SIGNALS: frozenset[str] = frozenset(
    {
        "UnsupportedRegionException",
        "UnsupportedOperation",
        "UnknownOperationException",
        "OptInRequired",
        "InvalidRequestException",
        "InvalidInputException",
        "ValidationException",
        "UninitializedAccountException",
        # botocore endpoint-resolution/connection failures (no ``Error.Code``, so the
        # scanner records a truncated message): the service has no endpoint here.
        "Could not connect to the endpoint URL",
        "Connect timeout on endpoint URL",
    }
)


def _is_region_unavailable(error: str) -> bool:
    """True if ``error`` marks a resource type as unavailable in the scanned region.

    Substring match: the scanner records either a ``ClientError`` code or a truncated
    botocore message (e.g. an endpoint connection error), so match on signal fragments
    rather than requiring an exact code.

    Args:
        error: The error string recorded in ``scan_result.failed`` for a type.

    Returns:
        True if the failure indicates the type/service is unavailable in the region.
    """
    return any(signal in error for signal in _REGION_UNAVAILABLE_SIGNALS)


@dataclasses.dataclass
class _RegionScanContext:
    """Phase-1 scan state for one region in the multi-region verify.

    Carries the per-region manager, its region-filtered snapshot, and the Phase-1
    baseline scan (or the exception if that scan failed — which fails only this
    region, not the account).
    """

    mgr: VerifyManager
    snapshot: Snapshot
    scan: ScanResult | None
    scan_error: Exception | None


class VerifyManager:
    """Manages account state verification operations."""

    def __init__(
        self,
        session: boto3.Session,
        region_name: str | None = None,
        account_id: str | None = None,
    ):
        """Initialize verification manager.

        Args:
            session: Boto3 session for AWS API calls (scan account)
            region_name: Optional AWS region override for verification
            account_id: Target account for the scan. Routes the fast-scan to the
                management-account Lambda; without it the scan degrades to the
                throttled host path, where a failed lister is swallowed into
                ``scan_result.failed`` and skipped by ``find_new_resources`` —
                making verify falsely pass.
        """
        self._session = session
        self._region_name = region_name
        self._account_id = account_id

        self._snapshot_mgr = SnapshotManager()

        scan_session = create_regional_session(session, region_name) if region_name else session
        self._scan_mgr = make_scanner(scan_session, region_name=region_name, account_id=account_id)
        self._stack_inspector = StackInspector(scan_session)
        self._drift_detector = DriftDetector(session, region=region_name)

    @property
    def region_name(self) -> str | None:
        """Get the region name used for verification."""
        return self._region_name

    def load_snapshot(self, env_name: str, account_id: str):
        """Load snapshot for the given environment and account."""
        return self._snapshot_mgr.load_snapshot(env_name, account_id)

    def _check_region_compatibility(self, snapshot_regions: list[str]) -> VerifyResult | None:
        """Check if verification region is compatible with snapshot regions.

        Returns:
            VerifyResult with failure if incompatible, None if compatible
        """
        if self._region_name and snapshot_regions:
            if self._region_name not in snapshot_regions:
                logger.warning(
                    f"Region mismatch: verify using {self._region_name}, "
                    f"snapshot has {snapshot_regions}"
                )
                return VerifyResult(
                    success=False,
                    reason="Region mismatch",
                    details={
                        "verify_region": self._region_name,
                        "snapshot_regions": snapshot_regions,
                    },
                    suggestion=(
                        f"Verify against one of the snapshot regions: {', '.join(snapshot_regions)}"
                    ),
                )
        return None

    def _check_dataset_version(
        self, snapshot_hash: str, current_scenario_dir: Path | None = None
    ) -> VerifyResult | None:
        """Check if scenario version matches snapshot baseline.

        Args:
            snapshot_hash: Hash from snapshot
            current_scenario_dir: Path to current scenario directory (optional)

        Returns:
            VerifyResult with failure if mismatch, None if compatible
        """
        if current_scenario_dir is None:
            # No scenario path provided - skip version check
            logger.debug("No scenario path provided, skipping version check")
            return None

        try:
            current_hash = compute_scenario_hash(current_scenario_dir)
        except Exception as e:
            # Benign: the version check is skipped (not failed) and verify proceeds normally.
            logger.debug(f"Failed to compute scenario hash, skipping version check: {e}")
            return None

        if current_hash != snapshot_hash:
            return VerifyResult(
                success=False,
                reason="Scenario version mismatch",
                details={
                    "expected_hash": snapshot_hash,
                    "current_hash": current_hash,
                },
                suggestion=(
                    "Scenario folder has changed since setup. "
                    "Run 'aws-bench env cleanup' and 'aws-bench env setup' to reset."
                ),
                is_dataset_mismatch=True,
            )
        return None

    def _check_new_resources(
        self,
        baseline_resource_ids: dict[str, list[dict]],
        baseline_failed: dict[str, str],
        baseline_empty: set[str],
        *,
        precomputed_scan: ScanResult | None = None,
        enumerable_elsewhere: set[str] | None = None,
    ) -> VerifyResult | None:
        """Scan for new resources created after setup.

        Args:
            baseline_resource_ids: Resources from baseline snapshot
            baseline_failed: Resource types that failed in baseline scan
            baseline_empty: Resource types that were scanned but returned 0 resources
            precomputed_scan: A scan of ``baseline_types`` already run for this region
                (the multi-region path scans each region once, then reuses the result
                here). When None, this method runs the scan itself.
            enumerable_elsewhere: Baseline types that a region in the same account
                enumerated cleanly (detected or empty) this run. A type that enumerated
                in some region is region-available there, so a failure to enumerate it
                *here* is region-unavailability, not an orphan-hiding gap — it is
                tolerated regardless of error code. (The multi-region path passes the
                account-wide union, which includes this region; harmless, since a type
                that failed here is not in this region's own enumerated set.) Cross-region
                corroboration; None on the single-region / reset paths, where the
                error-code allow-list applies.

        Returns:
            VerifyResult with failure if new resources found, None if clean
        """
        # Scan resource types that were in baseline (had resources OR were empty)
        # This avoids false positives from:
        # - the scanner discovering new resource types since baseline
        # - AWS-managed resources appearing in newly-discovered types
        # - scanner inconsistencies between scans
        # But still catches new resources in types that existed during setup
        baseline_types = set(baseline_resource_ids.keys()) | baseline_empty
        if precomputed_scan is not None:
            scan_result = precomputed_scan
        else:
            logger.debug(
                f"Scanning for new resources via fast-scan "
                f"({len(baseline_types)} resource types from baseline)"
            )
            scan_result = self._scan_mgr.scan_resources(resource_types=list(baseline_types))

        # Fail closed on a persistent scan failure of a baseline-tracked type. The
        # scanner already retries transient errors internally (8 attempts w/ backoff),
        # so a type still in scan_result.failed is a non-transient failure.
        # find_new_resources deliberately SKIPS failed types to avoid false diffs — but
        # for a type that mattered at setup, silently skipping it would let a leaked
        # resource in that type hide forever. Scoped to baseline_types only.
        #
        # Exception: a type that cannot exist in this region at all (the service/type is
        # unavailable here) cannot hold an orphan, so it is tolerated rather than failing
        # verify closed. A type is proven region-unavailable-here two ways:
        #   1. Cross-region: it enumerated cleanly in ANOTHER region this run
        #      (enumerable_elsewhere) — the strongest signal, independent of error code.
        #   2. Error code: its failure matches a region/service-availability signal
        #      (_REGION_UNAVAILABLE_SIGNALS) — the single-region backstop.
        # A type un-enumerable in EVERY region (absent from enumerable_elsewhere) with a
        # non-region-unavailable code still fails closed — preserving the fail-closed
        # guarantee for a genuinely un-enumerable type that could hide a leak.
        #
        # Toleration is scoped to types that were EMPTY at baseline: a type that HAD
        # resources at setup must always fail closed when un-enumerable — we cannot confirm
        # those tracked resources are gone, whatever the error code. Only a type with
        # nothing to account for (in baseline_empty, not baseline_resource_ids) is tolerated.
        elsewhere = enumerable_elsewhere or set()
        failed_baseline = {t: err for t, err in scan_result.failed.items() if t in baseline_types}
        tolerated = sorted(
            t
            for t, err in failed_baseline.items()
            if t not in baseline_resource_ids and (t in elsewhere or _is_region_unavailable(err))
        )
        tolerated_set = set(tolerated)
        unenumerable = sorted(t for t in failed_baseline if t not in tolerated_set)
        if tolerated:
            logger.info(
                f"Tolerating {len(tolerated)} baseline resource type(s) unavailable in this "
                f"region (cannot hold an orphan): {tolerated}"
            )
        if unenumerable:
            logger.warning(f"Could not enumerate baseline resource type(s): {unenumerable}")
            return VerifyResult(
                success=False,
                reason=f"Could not enumerate {len(unenumerable)} baseline resource type(s)",
                details={"unenumerable_types": unenumerable},
                suggestion="Run 'aws-bench env cleanup' and 'aws-bench env setup' to reset",
            )

        new_resources = find_new_resources(
            scan_result.detected, baseline_resource_ids, scan_result.failed, baseline_failed
        )

        # Drop residuals the account can't delete, so reset doesn't loop on them.
        if new_resources:
            probe = AwsManagedOwnershipProbe(self._session, self._region_name)
            before = count_resources(new_resources)
            new_resources = probe.exclude_aws_managed(new_resources)
            excluded = before - count_resources(new_resources)
            if excluded:
                logger.debug(f"Excluded {excluded} AWS-managed resource(s) from new-resource set")

        # Drop resources whose deletion was deferred this run (e.g. a Lambda@Edge
        # function awaiting CloudFront replica teardown): a delete was issued, but the
        # resource lingers for hours, so counting it as a residual would fail reset
        # even though it will be reaped. See resource_management.deferred.
        if new_resources:
            before = count_resources(new_resources)
            new_resources = exclude_deferred(new_resources)
            deferred = before - count_resources(new_resources)
            if deferred:
                logger.debug(f"Excluded {deferred} deferred (eventually-consistent) resource(s)")

        if new_resources:
            resource_count = count_resources(new_resources)
            # Log details of new resources (logs only, not console output)
            for resource_type, resources in new_resources.items():
                logger.debug(
                    f"New resources detected - {resource_type}: {len(resources)} resource(s)"
                )
                for resource in resources:
                    logger.debug(f"  - {resource.get('Identifier', 'unknown')}")

            return VerifyResult(
                success=False,
                reason=f"Found {resource_count} new resource(s)",
                details=new_resources,
                suggestion="Run 'aws-bench env reset' to remove new resources",
                new_resources=new_resources,
            )
        return None

    def find_orphan_resources(
        self, snapshot: Snapshot, *, enumerable_elsewhere: set[str] | None = None
    ) -> VerifyResult | None:
        """Re-run only the new-resource + scan-health census (no stack/drift checks).

        Reset's fail-closed backstop after deleting stacks for re-setup: a survivor
        it could not delete or enumerate must fail the reset, not be absorbed into a
        fresh baseline.

        Args:
            snapshot: The (region-filtered) baseline snapshot.
            enumerable_elsewhere: Baseline types another region enumerated cleanly this
                run, for cross-region corroboration in ``_check_new_resources`` — a type
                proven region-available elsewhere cannot hold an orphan here, so failing
                to enumerate it here is region-unavailability, not an orphan-hiding gap.
                None falls back to the error-code allow-list only.
        """
        return self._check_new_resources(
            snapshot.resource_ids,
            snapshot.failed_resource_types,
            snapshot.empty_resource_types,
            enumerable_elsewhere=enumerable_elsewhere,
        )

    def scan_baseline_types(self, snapshot: Snapshot) -> ScanResult:
        """Fast-scan this region for the snapshot's baseline resource types.

        Phase 1 of the multi-region verify: the orchestrator runs this once per region,
        then reuses the result (via ``precomputed_scan``) and the union of every region's
        enumerated types (via ``enumerable_elsewhere``) when it calls
        ``verify_account_state`` for that region — so the scan is not repeated. Baseline
        types are the account-wide merged set (had resources or were empty at setup),
        matching ``_check_new_resources``.

        Args:
            snapshot: The (region-filtered) baseline snapshot.

        Returns:
            The region's ScanResult for the baseline types.
        """
        baseline_types = set(snapshot.resource_ids.keys()) | snapshot.empty_resource_types
        return self._scan_mgr.scan_resources(resource_types=list(baseline_types))

    def _check_stack_status(self, stack_metadata: dict) -> VerifyResult | None:
        """Check CloudFormation stack status.

        Returns:
            VerifyResult with failure if status invalid, None if OK
        """
        status_result = self._stack_inspector.check_stack_status(stack_metadata)
        if not status_result.success:
            # Check if this is a list API failure vs actual stack status mismatches
            is_api_error = (
                isinstance(status_result.error_details, dict)
                and "error" in status_result.error_details
            )
            if is_api_error:
                # Failed to list stacks - can't reset this
                return VerifyResult(
                    success=False,
                    reason=status_result.error_reason,
                    details=status_result.error_details,
                    suggestion="Check AWS permissions and connectivity",
                )
            # Stack status mismatches - reset can handle these
            return VerifyResult(
                success=False,
                reason=status_result.error_reason,
                details=status_result.error_details,
                suggestion="Run 'aws-bench env reset' to restore stacks",
                stack_status_failures=status_result.error_details,
            )
        return None

    def _check_template_hash(self, stack_metadata: dict) -> VerifyResult | None:
        """Check CloudFormation template hash matches baseline.

        Returns:
            VerifyResult with failure if hash mismatch, None if match
        """
        hash_result = self._stack_inspector.check_template_hash(stack_metadata)
        if not hash_result.success:
            # A template change is per-stack recoverable (reset delete+redeploys the named
            # stacks); a get_template read failure carries no stack list and stays non-remediable.
            details = hash_result.error_details
            mismatch_stacks = (
                details.get("template_mismatch_stacks") if isinstance(details, dict) else None
            )
            if mismatch_stacks:
                return VerifyResult(
                    success=False,
                    reason=hash_result.error_reason,
                    details=details,
                    suggestion="Run 'aws-bench env reset' to recreate the changed stack(s)",
                    is_template_mismatch=True,
                    template_mismatch_stacks=mismatch_stacks,
                )
            return VerifyResult(
                success=False,
                reason=hash_result.error_reason,
                details=details,
                suggestion="Run 'aws-bench env cleanup' and re-setup with matching dataset",
            )
        return None

    def _check_drift(self, drift_baseline: dict) -> VerifyResult | None:
        """Check stack drift against baseline.

        Returns:
            VerifyResult with failure if drift differs from baseline, None if matching
        """
        drift_result = self._drift_detector.detect_and_compare_drift(drift_baseline)
        if not drift_result.success:
            return VerifyResult(
                success=False,
                reason=drift_result.error_reason,
                details=drift_result.error_details,
                suggestion="Run 'aws-bench env reset' to restore baseline drift",
                drift_differences=drift_result.drift_differences,
                drift_undetectable=drift_result.drift_undetectable,
            )
        return None

    def verify_account_state(
        self,
        env_name: str,
        account_id: str,
        *,
        snapshot: Snapshot | None = None,
        scenario_dir: Path | None = None,
        skip_early: bool = True,
        precomputed_scan: ScanResult | None = None,
        enumerable_elsewhere: set[str] | None = None,
    ) -> VerifyResult:
        """Verify account matches post-setup baseline snapshot.

        Args:
            env_name: Environment name
            account_id: Account ID to verify
            snapshot: Pre-loaded snapshot (optional, loads if None)
            scenario_dir: Path to scenario directory for version check
            skip_early: When True (the ``env verify`` gate) return on the first
                failing check — one failure or many, the verdict is "not clean,
                run reset", so running the rest gains nothing. When False
                (reset's own diagnosis) run *every* remaining check and merge
                their failures into one ``VerifyResult`` so reset can remediate
                all categories (new resources, stack-status, drift) in a single
                pass. Otherwise a short-circuit on one category (e.g. a
                new-resource false positive) hides a DELETE_FAILED stack from
                reset entirely.
            precomputed_scan: A baseline scan already run for this region, passed by
                the multi-region orchestrator so the region is not re-scanned. None on
                the single-region / reset paths (this method scans itself).
            enumerable_elsewhere: Baseline types another region enumerated cleanly this
                run, for cross-region corroboration in ``_check_new_resources``. None on
                the single-region / reset paths.

        Returns:
            VerifyResult with success/failure details
        """
        logger.debug(f"Verifying account {account_id} state for environment {env_name}")

        # Fail-fast check: ensure snapshot exists before scanning
        if snapshot is None:
            if not self._snapshot_mgr.snapshot_exists(
                env_name, account_id, SnapshotStage.POST_SETUP
            ):
                logger.error(
                    f"Snapshot not found for environment '{env_name}', account {account_id}"
                )
                return VerifyResult(
                    success=False,
                    reason="Snapshot not found",
                    suggestion=(
                        f"No baseline snapshot found for environment '{env_name}', "
                        f"account {account_id}. Run 'aws-bench env setup {env_name}' first."
                    ),
                )
            snapshot = self._snapshot_mgr.load_snapshot(env_name, account_id)

        # Region compatibility check
        region_check = self._check_region_compatibility(snapshot.regions)
        if region_check is not None:
            return region_check

        # The dataset-version check is fail-fast in BOTH modes: a scenario-hash
        # mismatch means the baseline itself no longer describes this account, so
        # there is nothing coherent to aggregate the other checks against — reset
        # must bail to full cleanup regardless.
        dataset_result = self._check_dataset_version(snapshot.scenario_hash, scenario_dir)
        if dataset_result is not None:
            return dataset_result

        # Remaining checks each surface one remediable category.
        remediable_checks = [
            lambda: self._check_new_resources(
                snapshot.resource_ids,
                snapshot.failed_resource_types,
                snapshot.empty_resource_types,
                precomputed_scan=precomputed_scan,
                enumerable_elsewhere=enumerable_elsewhere,
            ),
            lambda: self._check_stack_status(snapshot.stack_metadata),
            lambda: self._check_template_hash(snapshot.stack_metadata),
            lambda: self._check_drift(snapshot.drift_baseline),
        ]

        failures: list[VerifyResult] = []
        for check in remediable_checks:
            result = check()
            if result is not None:  # Check failed
                if skip_early:
                    return result
                failures.append(result)

        if failures:
            return self._merge_failures(failures)

        # All checks passed
        logger.debug("All verification checks passed")
        return VerifyResult(success=True, reason="Account state matches post-setup baseline")

    @staticmethod
    def _merge_failures(failures: list[VerifyResult]) -> VerifyResult:
        """Combine per-check failures into one result carrying every category.

        Reset reads ``new_resources``, ``stack_status_failures``,
        ``drift_differences`` and ``drift_undetectable`` off a single
        ``VerifyResult`` to drive its remediation phases, so a non-short-circuiting
        diagnosis must fold all of them together rather than return just the first.
        """
        merged_details: dict[str, Any] = {}
        new_resources: dict[str, list[dict]] | None = None
        stack_status_failures: dict[str, dict[str, str]] | None = None
        drift_differences: dict[str, dict] | None = None
        drift_undetectable: list[str] | None = None
        is_template_mismatch = False
        template_mismatch_stacks: list[str] | None = None

        for failure in failures:
            if isinstance(failure.details, dict):
                merged_details.update(failure.details)
            if failure.new_resources:
                new_resources = failure.new_resources
            if failure.stack_status_failures:
                stack_status_failures = failure.stack_status_failures
            if failure.drift_differences:
                drift_differences = failure.drift_differences
            if failure.drift_undetectable:
                drift_undetectable = failure.drift_undetectable
            if failure.is_template_mismatch:
                is_template_mismatch = True
                template_mismatch_stacks = failure.template_mismatch_stacks

        return VerifyResult(
            success=False,
            reason="; ".join(failure.reason for failure in failures),
            details=merged_details or None,
            suggestion="Run 'aws-bench env reset' to restore baseline state",
            new_resources=new_resources,
            stack_status_failures=stack_status_failures,
            drift_differences=drift_differences,
            drift_undetectable=drift_undetectable,
            is_template_mismatch=is_template_mismatch,
            template_mismatch_stacks=template_mismatch_stacks,
        )

    @staticmethod
    def _filter_snapshot_to_region(snapshot: Snapshot, region: str) -> Snapshot:
        """Return a copy of the snapshot scoped to a single region's stacks.

        Stack-status / template-hash / drift checks scan only the verify region,
        so the baseline they compare against must be limited to that region's
        stacks — otherwise stacks in other regions look "missing". Resource ids
        are left intact (the fast-scan is region-filtered separately).

        Snapshots captured before stacks recorded a region (region == "") are
        returned unchanged, preserving legacy single-region behavior.
        """
        # Check for legacy snapshot FIRST before filtering
        if not any(meta.region for meta in snapshot.stack_metadata.values()):
            return snapshot

        region_stacks = {
            name: meta for name, meta in snapshot.stack_metadata.items() if meta.region == region
        }
        region_drift = {
            name: drift for name, drift in snapshot.drift_baseline.items() if name in region_stacks
        }
        return dataclasses.replace(
            snapshot, stack_metadata=region_stacks, drift_baseline=region_drift
        )

    def verify_account_multiregion(
        self,
        env_name: str,
        account_id: str,
        environment_id: str,
        regions: list[str] | None = None,
        scenario_dir: Path | None = None,
    ) -> AccountVerifyResult:
        """Verify account across multiple regions.

        Args:
            env_name: Testing environment name
            account_id: AWS account ID to verify
            environment_id: Environment ID for display
            regions: List of regions to verify, or None to use snapshot regions
            scenario_dir: Path to scenario directory for version (hash) check

        Returns:
            AccountVerifyResult with per-region results
        """
        try:
            # Load snapshot to get baseline regions if not specified
            snapshot = self._snapshot_mgr.load_snapshot(env_name, account_id)
            regions_to_verify = regions if regions else snapshot.regions

            if not regions_to_verify:
                # Benign legacy-snapshot compatibility default; verify continues normally.
                logger.debug("Snapshot has no region info, defaulting to [us-east-1]")
                regions_to_verify = ["us-east-1"]

            logger.info(
                f"Verifying account {account_id} across {len(regions_to_verify)} region(s): "
                f"{', '.join(regions_to_verify)}"
            )

            # Each region scans against the snapshot filtered to its own stacks,
            # so other regions' stacks aren't reported as "missing".
            #
            # Two-phase so verify can corroborate across regions. The baseline's
            # resource/empty/failed type sets are account-wide (merged over regions),
            # so a region checks types that may only be enumerable in another region —
            # a type absent from THIS region (service not offered here) would otherwise
            # false-fail verify. Phase 1 scans every region once and records which
            # baseline types each enumerated; Phase 2 verifies each region reusing its
            # Phase-1 scan, forgiving a failed type that enumerated in another region
            # (proven region-available there, so it cannot hold an orphan here). A type
            # un-enumerable in EVERY region still fails closed.
            def _scan_region(region: str) -> _RegionScanContext:
                with log_context(region):
                    raise_if_shutdown()
                    region_session = create_regional_session(self._session, region)
                    region_mgr = VerifyManager(
                        region_session, region_name=region, account_id=account_id
                    )
                    region_snapshot = self._filter_snapshot_to_region(snapshot, region)
                    try:
                        scan = region_mgr.scan_baseline_types(region_snapshot)
                        return _RegionScanContext(region_mgr, region_snapshot, scan, None)
                    except Exception as exc:  # noqa: BLE001 — a scan error fails only this region
                        return _RegionScanContext(region_mgr, region_snapshot, None, exc)

            # Phase 1: scan every region (parallel).
            contexts: dict[str, _RegionScanContext] = {}
            with interruptible_executor(max_workers=MAX_WORKERS_HEAVY) as executor:
                scan_futures = {
                    executor.submit(_scan_region, region): region for region in regions_to_verify
                }
                for future in as_completed(scan_futures):
                    contexts[scan_futures[future]] = future.result()

            # A type enumerable (detected or empty) in ANY region is region-available
            # there — so a failure to enumerate it in another region is region
            # unavailability, not an orphan-hiding gap.
            enumerable_anywhere: set[str] = set()
            for ctx in contexts.values():
                if ctx.scan is not None:
                    enumerable_anywhere |= set(ctx.scan.detected.keys()) | ctx.scan.empty

            # Phase 2: verify each region, reusing its Phase-1 scan + the cross-region set.
            def _verify_region(region: str) -> RegionVerifyResult:
                with log_context(region):
                    raise_if_shutdown()
                    ctx = contexts[region]
                    if ctx.scan_error is not None:
                        logger.error(
                            "Region %s scan failed: %s", region, ctx.scan_error, exc_info=True
                        )
                        return RegionVerifyResult(
                            region=region,
                            success=False,
                            error_message=f"Verification failed: {ctx.scan_error}",
                        )
                    try:
                        result = ctx.mgr.verify_account_state(
                            env_name,
                            account_id,
                            snapshot=ctx.snapshot,
                            scenario_dir=scenario_dir,
                            precomputed_scan=ctx.scan,
                            enumerable_elsewhere=enumerable_anywhere,
                        )
                        return RegionVerifyResult(
                            region=region,
                            success=result.success,
                            error_message=result.reason if not result.success else None,
                            suggestion=result.suggestion,
                        )
                    except Exception as exc:
                        logger.error(
                            "Region %s verification failed: %s", region, exc, exc_info=True
                        )
                        return RegionVerifyResult(
                            region=region,
                            success=False,
                            error_message=f"Verification failed: {exc}",
                        )

            results_by_region: dict[str, RegionVerifyResult] = {}
            with interruptible_executor(max_workers=MAX_WORKERS_HEAVY) as executor:
                future_to_region = {
                    executor.submit(_verify_region, region): region for region in regions_to_verify
                }
                for future in as_completed(future_to_region):
                    region = future_to_region[future]
                    results_by_region[region] = future.result()

            # Emit in input-region order (not completion order) so the persisted
            # result is stable across runs.
            region_results = [results_by_region[region] for region in regions_to_verify]
            all_passed = all(r.success for r in region_results)

            return AccountVerifyResult(
                account_id=account_id,
                environment_id=environment_id,
                success=all_passed,
                region_results=region_results,
            )

        except SnapshotNotFoundError as e:
            return AccountVerifyResult(
                account_id=account_id,
                environment_id=environment_id,
                success=False,
                region_results=[],
                error_message=str(e),
            )
        except Exception as e:
            return AccountVerifyResult(
                account_id=account_id,
                environment_id=environment_id,
                success=False,
                region_results=[],
                error_message=f"Verification failed: {str(e)}",
            )
