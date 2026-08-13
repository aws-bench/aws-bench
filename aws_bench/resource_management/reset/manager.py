"""Reset manager for restoring accounts to baseline state."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import boto3

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.account_management.models import ScenarioAccount
from aws_bench.logging.logger import get_logger, log_context
from aws_bench.resource_management.ccapi.manager import CloudControlManager, Resource
from aws_bench.resource_management.ccapi.models import GLOBAL_RESOURCE_TYPES, MAX_WORKERS_HEAVY
from aws_bench.resource_management.cleanup.manager import CleanupManager
from aws_bench.resource_management.cleanup.models import StackResource
from aws_bench.resource_management.cleanup.resource_cleaner import ResourceCleaner
from aws_bench.resource_management.constants import RESOURCE_MANAGEMENT_SESSION
from aws_bench.resource_management.deferred import deferred_scope, mark_deferred
from aws_bench.resource_management.reset.models import (
    ResetFailure,
    ResetResult,
    ResetupDeletion,
    RestoreOutcome,
    StackResetOutcome,
)
from aws_bench.resource_management.reset.stack_restorer import StackRestorer
from aws_bench.resource_management.snapshot.models import Snapshot
from aws_bench.resource_management.verify.manager import VerifyManager
from aws_bench.resource_management.verify.models import VerifyResult
from aws_bench.utils.account_utils import group_accounts_by_scenario
from aws_bench.utils.concurrent import raise_if_shutdown, reraise_if_cancelled
from aws_bench.utils.credentials_provider import (
    CredentialProvider,
    build_session_name,
    create_regional_session,
)

logger = get_logger(__name__)

# Max drifted stacks restored concurrently per region — small because each is a
# heavy CFN mutation, kept under drift detection's read-only in-flight cap.
_DRIFT_RESTORE_CONCURRENCY = 4


class ResetManager:
    """Manages account reset operations to restore baseline state."""

    def __init__(
        self,
        session: boto3.Session,
        output_dir: Path | None = None,
        account_id: str | None = None,
    ):
        """Initialize with boto3 session and optional output directory.

        Args:
            session: boto3 Session for AWS operations (scan account)
            output_dir: Directory for reset operation outputs
            account_id: Target account for scans. Threaded into VerifyManager so
                its fast-scan routes to the management-account Lambda rather than
                the throttled host path. The per-region VerifyManager built in
                ``_reset_region`` receives it from that method's own argument.
        """
        self._session = session
        self._output_dir = output_dir
        self._account_id = account_id
        self._verify_mgr = VerifyManager(session, account_id=account_id)
        self._cleanup_mgr = CleanupManager(session, output_dir=output_dir, account_id=account_id)
        self._stack_restorer = StackRestorer(session, self._cleanup_mgr)

    async def reset_account(
        self, env_name: str, account_id: str, scenario_dir: Path | None = None
    ) -> ResetResult:
        """Reset account to post-setup baseline state.

        Args:
            env_name: Environment (scenario) name
            account_id: AWS account ID to reset
            scenario_dir: Path to scenario directory for version (hash) check.
                A hash mismatch is treated as unrecoverable and routes the
                operator to full cleanup + setup.
        """
        try:
            logger.debug(f"Starting reset for account {account_id} in environment '{env_name}'")

            snapshot = await asyncio.to_thread(self._verify_mgr.load_snapshot, env_name, account_id)

            # Reset every region the baseline covers. Each region is verified
            # and fixed against its own slice of the snapshot, so stacks in
            # other regions aren't mistaken for "missing" (the baseline holds
            # all regions but each scan only sees one).
            regions = snapshot.regions or [self._verify_mgr.region_name or "us-east-1"]
            logger.debug(f"Resetting across {len(regions)} region(s): {', '.join(regions)}")

            # Regions are independent, so reset them concurrently under a bound.
            region_sem = asyncio.Semaphore(MAX_WORKERS_HEAVY)

            async def _reset_region_bounded(
                region: str,
            ) -> tuple[list[str], dict[str, list[dict]]]:
                async with region_sem:
                    return await self._reset_region(
                        env_name, account_id, snapshot, region, scenario_dir
                    )

            region_results = await asyncio.gather(
                *(_reset_region_bounded(region) for region in regions),
                return_exceptions=True,
            )
            reraise_if_cancelled(region_results)  # a shutdown must unwind, not become a failure

            # Log every region's failure before raising the first, so a second
            # broken region isn't hidden until the operator re-runs.
            deleted_stacks: list[str] = []
            global_resources: dict[str, list[dict]] = {}
            failures: list[tuple[str, BaseException]] = []
            for region, result in zip(regions, region_results):
                if isinstance(result, BaseException):
                    failures.append((region, result))
                else:
                    region_deleted, region_globals = result
                    deleted_stacks.extend(region_deleted)
                    for rtype, items in region_globals.items():
                        global_resources.setdefault(rtype, []).extend(items)
            for region, exc in failures:
                logger.error(f"Region '{region}' reset failed: {exc}")
            if failures:
                # Intentional: on a region failure we skip the global-delete pass and
                # raise. The failing region's dependents may not have finished tearing
                # down, so deleting the shared global (e.g. a KB's exec role) now could
                # re-trigger the very race this deferral prevents. The reset returns
                # failure -> the account is flagged contaminated (not reused) and the
                # operator runs `env cleanup`, which removes the deferred globals. So a
                # skipped global here is a contained partial-cleanup, never a false pass.
                raise failures[0][1]

            # All regional deletes converged; now delete account-global resources
            # (IAM roles/etc.) once, deferred out of the per-region passes so a
            # region without a dependent never race-deletes a global that another
            # region's dependent (e.g. a Bedrock KB) still needs mid-teardown.
            await self._delete_global_resources(global_resources, regions)

            # Stacks that couldn't be reverted were deleted; they are recreated
            # by re-running setup (idempotent deploy.sh recreates only the
            # missing stack). Skip final verify — the stack is intentionally
            # gone — and signal the CLI to redeploy.
            if deleted_stacks:
                logger.info(
                    f"{len(deleted_stacks)} stack(s) deleted for re-setup: "
                    f"{', '.join(deleted_stacks)}"
                )
                return ResetResult(
                    account_id=account_id,
                    scenario_name=env_name,
                    success=True,
                    reason="Drift reverted; deleted stacks will be recreated by setup",
                    details={"deleted_stacks": deleted_stacks},
                    needs_redeploy=True,
                )

            logger.info("Account successfully reset to baseline state")
            return ResetResult(
                account_id=account_id,
                scenario_name=env_name,
                success=True,
                reason="Account successfully reset to baseline state",
            )

        except ResetFailure as e:
            logger.error(f"Reset failed: {e.reason}")
            orphans = e.details.get("unresolved_orphans") if isinstance(e.details, dict) else None
            return ResetResult(
                account_id=account_id,
                scenario_name=env_name,
                success=False,
                reason=e.reason,
                details=e.details,
                suggestion=e.suggestion,
                unresolved_orphans=orphans,
            )

    async def _reset_region(
        self,
        env_name: str,
        account_id: str,
        snapshot: Snapshot,
        region: str,
        scenario_dir: Path | None,
    ) -> tuple[list[str], dict[str, list[dict]]]:
        """Reset one region against its slice of the baseline.

        Returns:
            ``(deleted_stacks, global_resources)`` — the names of stacks deleted
            for re-setup (revert impossible), and the global (account-wide) new
            resources this region saw but deferred to the account-level pass.

        Raises:
            ResetFailure: On unrecoverable mismatch, restore failure, or if the
                region still fails verification after the reset operations.
        """
        with log_context(region), deferred_scope():
            region_session = create_regional_session(self._session, region)
            region_verify = VerifyManager(region_session, region_name=region, account_id=account_id)
            region_snapshot = VerifyManager._filter_snapshot_to_region(snapshot, region)
            # Own CleanupManager/session per region: regions run concurrently and
            # a boto3 Session isn't thread-safe for client creation. account_id is
            # threaded in so it skips the STS get_caller_identity lookup.
            region_cleanup = CleanupManager(
                region_session, output_dir=self._output_dir, account_id=account_id
            )
            region_restorer = StackRestorer(region_session, region_cleanup)

            logger.debug("Phase 1: verifying to identify issues")
            # skip_early=False: reset fixes every category in one pass, so the
            # diagnosis must report ALL of them, not short-circuit on the first.
            verify_result = await asyncio.to_thread(
                region_verify.verify_account_state,
                env_name,
                account_id,
                snapshot=region_snapshot,
                scenario_dir=scenario_dir,
                skip_early=False,
            )

            if verify_result.success:
                logger.debug("Already in baseline state, nothing to reset")
                return [], {}

            self._check_recoverable(verify_result)
            global_resources = await self._delete_new_resources(
                verify_result, region_session, region
            )
            # Delete stacks unrecoverable in place (bad status / undetectable drift)
            # so setup recreates them.
            status_outcome = await self._delete_unrecoverable_stacks(verify_result, region_restorer)
            # A DELETE_FAILED stack is in both status_failures and drift_differences;
            # skip the already-deleted ones so restore_stack doesn't error on them.
            drift_outcome = await self._restore_drifted_stacks(
                verify_result,
                region_restorer,
                already_handled=set(status_outcome.deleted_stacks),
            )

            deleted_stacks = status_outcome.deleted_stacks + drift_outcome.deleted_stacks

            if deleted_stacks:
                # Deleted stacks are intentionally absent (setup recreates them), so skip the
                # stack/drift re-checks. But a survivor — reset failed to delete it, an
                # unenumerable baseline type, or one FORCE_DELETE_STACK abandoned — would be
                # absorbed into a fresh baseline, so re-run the census and fail closed.
                # This runs inside the region's deferred_scope(): the account-global resources
                # that _delete_new_resources marked deferred are excluded by exclude_deferred,
                # so the not-yet-deleted globals (deleted later by _delete_global_resources)
                # never false-positive here.
                orphan_result = await asyncio.to_thread(
                    region_verify.find_orphan_resources, region_snapshot
                )
                census = self._census_orphans(orphan_result) if orphan_result is not None else {}
                unresolved = self._merge_orphan_maps(
                    census, status_outcome.abandoned, drift_outcome.abandoned
                )
                if unresolved:
                    reason = "Stacks deleted for re-setup, but reset left the region unresolved"
                    if orphan_result is not None:
                        reason = f"{reason}: {orphan_result.reason}"
                    else:
                        n = sum(len(v) for v in unresolved.values())
                        reason = f"{reason}: {n} resource(s) abandoned by force-delete"
                    raise ResetFailure(
                        reason=reason,
                        details={"unresolved_orphans": unresolved},
                        suggestion=(orphan_result.suggestion if orphan_result else None)
                        or "Run 'aws-bench env cleanup' for full reset",
                    )
                return deleted_stacks, global_resources

            logger.debug("Phase 4: final verification")
            # Aggregate here too: if reset did not fully converge, report every
            # category still failing, not just the first.
            final_verify = await asyncio.to_thread(
                region_verify.verify_account_state,
                env_name,
                account_id,
                snapshot=region_snapshot,
                scenario_dir=scenario_dir,
                skip_early=False,
            )
            if not final_verify.success:
                raise ResetFailure(
                    reason=(
                        f"Reset operations completed but verification still failing: "
                        f"{final_verify.reason}"
                    ),
                    details=final_verify.details,
                    suggestion=(
                        final_verify.suggestion or "Run 'aws-bench env cleanup' for full reset"
                    ),
                )
            return [], global_resources

    def _check_recoverable(self, verify_result: VerifyResult) -> None:
        """Check if account state can be recovered via reset."""
        if verify_result.is_dataset_mismatch or verify_result.is_script_mismatch:
            logger.error(
                "Dataset or script version mismatch detected - requires full cleanup and redeploy"
            )
            raise ResetFailure(
                reason="Dataset/script version mismatch - requires full cleanup",
                details=verify_result.details,
                suggestion="Run 'aws-bench env cleanup' and 'aws-bench env setup'",
            )

    async def _delete_new_resources(
        self, verify_result: VerifyResult, session: boto3.Session, region: str
    ) -> dict[str, list[dict]]:
        """Delete a region's regional new resources; defer global ones.

        Global (account-wide) resource types — IAM roles/policies/etc. — surface
        in every region's scan with the same identifier, so deleting them here
        would let a region without a dependent race-delete a global resource that
        another region's dependent still needs (e.g. a Bedrock knowledge base's
        execution role, which ``delete_knowledge_base`` must assume to finish an
        async teardown). Such a delete wedges the dependent in ``DELETE_FAILED``
        and contaminates the account. So this method deletes only the region's
        *regional* new resources and returns the *global* ones for the caller to
        delete once, in a final account-level pass after every region has drained.

        The returned globals are also marked deferred for this region's scope so
        the region's own final verification does not fail on the not-yet-deleted
        global resource.

        Returns:
            The global new resources seen in this region (``{type: [items]}``),
            already excluded from this region's deletion and its final verify.
        """
        if not verify_result.new_resources:
            logger.debug("Phase 2: No new resources to delete, skipping")
            return {}

        regional: dict[str, list[dict]] = {}
        global_resources: dict[str, list[dict]] = {}
        for rtype, resources in verify_result.new_resources.items():
            if rtype in GLOBAL_RESOURCE_TYPES:
                global_resources[rtype] = resources
                # Suppress the region's final verify on globals we defer to the
                # account-level pass (deferred scope is per-region — see caller).
                for r in resources:
                    mark_deferred(rtype, r.get("Identifier", ""))
            else:
                regional[rtype] = resources

        await self._delete_resource_set(regional, session, region, phase_label="Phase 2")
        return global_resources

    async def _delete_resource_set(
        self,
        resources: dict[str, list[dict]],
        session: boto3.Session,
        region: str,
        *,
        phase_label: str,
    ) -> None:
        """Delete a ``{type: [items]}`` set via the shared cleanup pipeline.

        Best-effort: failures are logged, not raised. The caller's verification
        (per-region final verify for regional resources, the existence re-check in
        ``_delete_global_resources`` for globals) is what actually gates the reset.
        """
        if not resources:
            return
        count = sum(len(v) for v in resources.values())
        logger.debug(f"{phase_label}: Deleting {count} new resource(s)")

        # Delete via the shared cleanup pipeline (service-API custom handlers,
        # then CCAPI fallback). Using raw CCAPI alone cannot delete types CCAPI
        # doesn't support (e.g. S3 buckets), which would leave them behind and
        # fail final verification.
        stack_resources = [
            StackResource(
                logical_id=r["Identifier"],
                physical_id=r["Identifier"],
                resource_type=rtype,
                status="",
            )
            for rtype, items in resources.items()
            for r in items
        ]
        cleaner = ResourceCleaner(session, region)
        failures = await cleaner.cleanup(
            stack_resources,
            prepare=True,
            custom_delete=True,
            ccapi_fallback=True,
        )
        if failures:
            logger.warning(
                f"Failed to delete {len(failures)} resource(s); the fail-closed verify / "
                f"orphan re-check will fail the reset if they are still present."
            )

    async def _delete_global_resources(
        self, global_resources: dict[str, list[dict]], regions: list[str]
    ) -> None:
        """Delete account-global new resources once, after every region drained.

        Regional deletes (each region's dependents, incl. any KB whose async
        delete blocks on its execution role until terminal) have all completed by
        now, so deleting the shared global resources here can no longer race a
        still-in-flight dependent. Deduped across regions since the same global
        surfaces in each.

        Because these globals were deferred out of every per-region final verify,
        this pass is the ONLY thing guarding them — so it does not trust the
        cleanup pipeline's failure count alone (CCAPI can silently *skip* a
        resource whose existence pre-check throttles or errors, reporting zero
        failures while the resource survives). It re-checks each global's actual
        existence and raises ``ResetFailure`` if any remain, mirroring the live
        re-scan the per-region final verify would otherwise have done.
        """
        deduped: dict[str, list[dict]] = {}
        seen: set[tuple[str, str]] = set()
        for rtype, items in global_resources.items():
            for r in items:
                key = (rtype, r.get("Identifier", ""))
                if key not in seen:
                    seen.add(key)
                    deduped.setdefault(rtype, []).append(r)
        if not deduped:
            return

        # IAM and other globals are region-agnostic; use the primary region's
        # session. Runs outside any per-region deferred scope, at account level.
        primary_region = regions[0] if regions else (self._verify_mgr.region_name or "us-east-1")
        region_session = create_regional_session(self._session, primary_region)
        await self._delete_resource_set(
            deduped, region_session, primary_region, phase_label="Final pass (global resources)"
        )

        # Authoritative existence re-check — do NOT rely on the pipeline's failure
        # count (a silent skip reports success while the resource lives). A still
        # present global (or one we cannot confirm gone) fails the reset closed.
        survivors = await asyncio.to_thread(self._surviving_globals, region_session, deduped)
        if survivors:
            raise ResetFailure(
                reason=f"Global resource(s) still present after reset: {len(survivors)}",
                details={
                    rtype: [{"Identifier": i} for i in ids] for rtype, ids in survivors.items()
                },
                suggestion="Run 'aws-bench env cleanup' for full reset",
            )

    @staticmethod
    def _surviving_globals(
        session: boto3.Session, resources: dict[str, list[dict]]
    ) -> dict[str, list[str]]:
        """Return globals that still exist (or cannot be confirmed gone) via CCAPI.

        Fails closed: a throttled/erroring existence check counts the resource as
        surviving, since a false "gone" here would leak a global into the next
        task. IAM types are all CCAPI-supported, so ``resource_exists`` resolves.
        """
        ccm = CloudControlManager(session)
        survivors: dict[str, list[str]] = {}
        for rtype, items in resources.items():
            for r in items:
                identifier = r.get("Identifier", "")
                if not identifier:
                    continue
                try:
                    still_there = ccm.resource_exists(Resource(type=rtype, identifier=identifier))
                except Exception as exc:  # noqa: BLE001 — cannot confirm gone -> fail closed
                    logger.warning(
                        "Could not confirm deletion of %s '%s' (%s); treating as surviving",
                        rtype,
                        identifier,
                        exc,
                    )
                    still_there = True
                if still_there:
                    survivors.setdefault(rtype, []).append(identifier)
        return survivors

    @staticmethod
    def _merge_orphan_maps(*maps: dict[str, list[dict]] | None) -> dict[str, list[dict]]:
        """Union several {type: [id-dict, ...]} maps, concatenating ids per type."""
        merged: dict[str, list[dict]] = {}
        for m in maps:
            for rtype, ids in (m or {}).items():
                merged.setdefault(rtype, []).extend(ids)
        return merged

    @staticmethod
    def _census_orphans(result: VerifyResult) -> dict[str, list[dict]]:
        """Represent a failed orphan census as {type: [id-dict]} for unresolved_orphans.

        A found-new-resource failure carries ``new_resources`` directly; an
        unenumerable-baseline-type failure carries only type names, so mark each
        with a sentinel id-dict so the type still surfaces as unresolved.
        """
        if result.new_resources:
            return dict(result.new_resources)
        details = result.details if isinstance(result.details, dict) else {}
        unenumerable = details.get("unenumerable_types", [])
        return {t: [{"error": "could not enumerate"}] for t in unenumerable}

    async def _delete_unrecoverable_stacks(
        self, verify_result: VerifyResult, restorer: StackRestorer
    ) -> StackResetOutcome:
        """Delete stacks unfixable in place so ``env setup`` recreates them.

        Covers status mismatches (missing / DELETE_FAILED / ...), drift that regressed to
        undetectable, and template-hash mismatches (``UsePreviousTemplate`` drift-revert
        cannot undo a template change). A missing stack needs no delete call; the rest go
        through the shared cleanup path, deduped across every category.

        Raises:
            ResetFailure: If any stack could not be deleted.
        """
        missing = {
            name
            for name, info in (verify_result.stack_status_failures or {}).items()
            if info["actual"] == "MISSING"
        }
        to_delete: list[str] = []
        for name, info in (verify_result.stack_status_failures or {}).items():
            if name in missing:
                logger.debug(f"{name}: Stack missing, will be recreated by setup")
                continue
            logger.debug(
                f"{name}: Status mismatch (expected {info['expected']}, actual {info['actual']}), "
                f"deleting for re-setup"
            )
            to_delete.append(name)
        for name in verify_result.drift_undetectable or []:
            if name not in missing and name not in to_delete:
                logger.debug(f"{name}: Drift no longer detectable, deleting for re-setup")
                to_delete.append(name)
        for name in verify_result.template_mismatch_stacks or []:
            if name not in missing and name not in to_delete:
                logger.debug(f"{name}: Template changed, deleting for re-setup")
                to_delete.append(name)

        if not missing and not to_delete:
            logger.debug("Phase 2: No unrecoverable stacks to delete, skipping")
            return StackResetOutcome()

        failed_stacks: list[str] = []
        deleted_stacks: list[str] = sorted(missing)
        abandoned: dict[str, list[dict]] = {}
        for stack_name in to_delete:
            raise_if_shutdown()
            deletion = await restorer._delete_for_resetup(stack_name)
            if deletion.outcome is RestoreOutcome.DELETED_NEEDS_REDEPLOY:
                deleted_stacks.append(stack_name)
                abandoned = self._merge_orphan_maps(abandoned, deletion.abandoned)
                logger.debug(f"Deleted (will re-setup): {stack_name}")
            else:
                failed_stacks.append(stack_name)
                logger.error(f"Failed to delete: {stack_name}")

        if failed_stacks:
            raise ResetFailure(
                reason=f"Failed to delete {len(failed_stacks)} unrecoverable stack(s)",
                details={"failed_stacks": failed_stacks},
                suggestion="Run 'aws-bench env cleanup' for full reset",
            )

        logger.debug(f"Phase 2: {len(deleted_stacks)} stack(s) will be recreated by setup")
        return StackResetOutcome(deleted_stacks=deleted_stacks, abandoned=abandoned)

    async def _restore_drifted_stacks(
        self,
        verify_result: VerifyResult,
        restorer: StackRestorer,
        already_handled: set[str] | None = None,
    ) -> StackResetOutcome:
        """Restore stacks with drift differences to baseline state.

        Args:
            verify_result: The region's verify result carrying drift differences.
            restorer: Region-scoped StackRestorer to act in the right region.
            already_handled: Stack names already remediated by an earlier phase
                (e.g. status-failure deletion). These are skipped so we never
                act twice on the same stack — a DELETE_FAILED stack appears in
                both ``stack_status_failures`` and ``drift_differences``, and
                restoring an already-deleted stack errors.

        Returns:
            A StackResetOutcome naming the stacks that were deleted (revert
            impossible) and merging any resources their force-delete abandoned.
            Empty if every stack was reverted in place.

        Raises:
            ResetFailure: If any stack could neither be reverted nor deleted.
        """
        if not verify_result.drift_differences:
            logger.debug("Phase 3: No drift differences to fix, skipping")
            return StackResetOutcome()

        already_handled = already_handled or set()
        drift_stacks = {
            name: diff
            for name, diff in verify_result.drift_differences.items()
            if name not in already_handled
        }
        if not drift_stacks:
            logger.debug("Phase 3: All drifted stacks already handled by status fix, skipping")
            return StackResetOutcome()

        logger.debug(f"Phase 3: Fixing drift on {len(drift_stacks)} stack(s)")

        # Independent stacks, each a heavy CFN mutation — restore concurrently
        # under a small bound to protect the account+region throttle bucket.
        restore_sem = asyncio.Semaphore(_DRIFT_RESTORE_CONCURRENCY)

        async def _restore_one(stack_name: str, baseline: Any) -> tuple[str, ResetupDeletion]:
            async with restore_sem:
                raise_if_shutdown()  # don't start a new restore after a shutdown
                deletion = await restorer.restore_stack(
                    stack_name=stack_name,
                    baseline_drift=baseline,
                )
                return stack_name, deletion

        outcomes = await asyncio.gather(
            *(_restore_one(name, diff["baseline"]) for name, diff in drift_stacks.items()),
            return_exceptions=True,
        )
        reraise_if_cancelled(outcomes)

        failed_stacks: list[str] = []
        deleted_stacks: list[str] = []
        abandoned: dict[str, list[dict]] = {}
        for result in outcomes:
            # An unexpected error fails that stack, as the per-stack loop did.
            if isinstance(result, BaseException):
                raise result
            stack_name, deletion = result
            if deletion.outcome is RestoreOutcome.RESTORED:
                logger.debug(f"Restored: {stack_name}")
            elif deletion.outcome is RestoreOutcome.DELETED_NEEDS_REDEPLOY:
                deleted_stacks.append(stack_name)
                abandoned = self._merge_orphan_maps(abandoned, deletion.abandoned)
                logger.debug(f"Deleted (will re-setup): {stack_name}")
            else:
                failed_stacks.append(stack_name)
                logger.debug(f"Failed: {stack_name}")

        if failed_stacks:
            logger.debug(f"Restoring stacks: {len(failed_stacks)} stack(s) failed")
            raise ResetFailure(
                reason=f"Failed to restore {len(failed_stacks)} stack(s) to baseline",
                details={"failed_stacks": failed_stacks},
                suggestion="Run 'aws-bench env cleanup' for full reset",
            )

        if deleted_stacks:
            logger.debug(f"Restoring stacks: {len(deleted_stacks)} stack(s) deleted for re-setup")
        else:
            logger.debug("Restoring stacks: all stacks restored successfully")
        return StackResetOutcome(deleted_stacks=deleted_stacks, abandoned=abandoned)

    @staticmethod
    async def reset_scenarios(
        ou_name: str,
        scenario_accounts: list[ScenarioAccount],
        max_concurrent: int = 10,
        scenario_dirs: dict[str, Path] | None = None,
        cred_provider=None,  # CredentialProvider, avoid circular import
    ) -> list[ResetResult | Exception]:
        """Reset accounts across the given scenarios.

        If stacks need redeployment (marked with needs_redeploy), the scenario
        trial automatically handles this by running the DEPLOY phase after reset.

        Args:
            ou_name: Organizational unit name
            scenario_accounts: Accounts from all scenarios in the OU (1+ accounts per scenario)
            max_concurrent: Maximum concurrent resets
            scenario_dirs: Pre-resolved scenario directories (from CLI)
            cred_provider: Credential provider (unused, kept for API compatibility)

        Returns:
            Flattened list of ResetResult or Exception for all accounts
        """
        # Group accounts by scenario
        by_scenario = group_accounts_by_scenario(scenario_accounts)

        # Shared semaphore for concurrent reset across all scenarios
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _reset_scenario(
            scenario_name: str, accounts: list[ScenarioAccount]
        ) -> list[ResetResult | Exception]:
            """Reset all accounts in a scenario."""
            scenario_dir_path = scenario_dirs.get(scenario_name) if scenario_dirs else None

            async def reset_account(account: ScenarioAccount):
                async with semaphore:
                    with log_context(account.account_tag):
                        try:
                            session = CredentialProvider.get().get_session_for_account(
                                account.account_id,
                                ORG_ACCESS_ROLE,
                                build_session_name(RESOURCE_MANAGEMENT_SESSION, "reset-account"),
                            )
                            reset_mgr = ResetManager(session, account_id=account.account_id)
                            return await reset_mgr.reset_account(
                                scenario_name, account.account_id, scenario_dir_path
                            )
                        except Exception as e:
                            return e

            with log_context(scenario_name):
                return await asyncio.gather(*[reset_account(acc) for acc in accounts])

        # Process all scenarios concurrently
        scenario_results = await asyncio.gather(
            *[_reset_scenario(name, accs) for name, accs in by_scenario.items()]
        )

        # Flatten results
        return [result for scenario in scenario_results for result in scenario]
