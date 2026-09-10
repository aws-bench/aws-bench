"""High-level resource management orchestration.

Coordinates deployment, tracking, and cleanup of CloudFormation stacks
and AWS resources across accounts and regions.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.account_management.models import ScenarioAccount, TestEnvironment
from aws_bench.logging.logger import get_logger, log_context
from aws_bench.resource_management.cleanup.manager import CleanupManager
from aws_bench.resource_management.cleanup.models import (
    AccountCleanupResult,
    CleanupSummary,
    EnvironmentCleanupResult,
)
from aws_bench.resource_management.reset.manager import ResetManager
from aws_bench.resource_management.reset.models import ResetResult
from aws_bench.resource_management.snapshot.manager import SnapshotManager
from aws_bench.resource_management.snapshot.models import (
    SnapshotContext,
    SnapshotResult,
    SnapshotStage,
)
from aws_bench.resource_management.verify.manager import VerifyManager
from aws_bench.resource_management.verify.models import (
    AccountVerifyResult,
    VerificationReport,
)
from aws_bench.scenario.scenario import Scenario
from aws_bench.utils.concurrent import build_client
from aws_bench.utils.credentials_provider import CredentialProvider, build_session_name
from aws_bench.utils.regions import get_enabled_regions

logger = get_logger(__name__)

__all__ = ["ResourceManager", "CleanupSummary", "EnvironmentCleanupResult"]


class ResourceManager:
    """High-level API for AWS resource management operations.

    Provides stateless methods for deploying, tracking, and cleaning up
    CloudFormation stacks and AWS resources across accounts.

    Example:
        >>> # Delete all stacks in an account
        >>> summary = ResourceManager.cleanup_account("123456789012")
        >>> print(f"Deleted {summary.total_deleted}/{summary.total_stacks} stacks")
        >>>
        >>> # Delete all stacks in specific regions
        >>> summary = ResourceManager.cleanup_account(
        ...     "123456789012", regions=["us-east-1", "us-west-2"]
        ... )
        >>>
        >>> # Delete a specific stack
        >>> summary = ResourceManager.cleanup_stack("123456789012", "my-app-stack")
    """

    @staticmethod
    def _validate_account_id(account_id: str) -> None:
        """Validate AWS account ID format.

        Args:
            account_id: AWS account ID to validate

        Raises:
            ValueError: If account_id is invalid format
        """
        if not account_id or not isinstance(account_id, str):
            raise ValueError("account_id must be a non-empty string")
        if not account_id.isdigit() or len(account_id) != 12:
            raise ValueError(f"account_id must be a 12-digit string, got: {account_id}")

    @staticmethod
    def _validate_stack_name(stack_name: str) -> None:
        """Validate CloudFormation stack name.

        Args:
            stack_name: Stack name to validate

        Raises:
            ValueError: If stack_name is invalid
        """
        if not stack_name or not isinstance(stack_name, str):
            raise ValueError("stack_name must be a non-empty string")
        if stack_name.strip() != stack_name:
            raise ValueError(f"stack_name cannot have leading/trailing whitespace: '{stack_name}'")

    @staticmethod
    async def cleanup_account(
        account_id: str,
        regions: list[str] | None = None,
        env_name: str | None = None,
    ) -> CleanupSummary:
        """Delete all CloudFormation stacks in a member AWS account within the organization.

        Args:
            account_id: AWS account ID (12-digit string)
            regions: List of regions to clean. If None, discovers all enabled regions.
            env_name: Environment name for loading baseline snapshots (optional).
                If provided, limits orphan scanning to baseline regions only.

        Returns:
            CleanupSummary with per-region results and orphaned resources

        Raises:
            ValueError: If account_id is invalid format
            TypeError: If regions is not a list
            RuntimeError: If unable to discover regions
        """
        ResourceManager._validate_account_id(account_id)

        session = CredentialProvider.get().get_session_for_account(
            account_id,
            ORG_ACCESS_ROLE,
            build_session_name("session"),
        )
        manager = CleanupManager(session, env_name=env_name, account_id=account_id)
        return await manager.cleanup_all_stacks(regions=regions)

    @staticmethod
    async def cleanup_stack(
        account_id: str,
        stack_name: str,
    ) -> CleanupSummary:
        """Delete a specific CloudFormation stack.

        Args:
            account_id: AWS account ID (12-digit string)
            stack_name: Name or ARN of the stack to delete

        Returns:
            CleanupSummary with the result of this single stack deletion

        Raises:
            ValueError: If account_id or stack_name is invalid, or stack not found
            RuntimeError: If unable to discover regions
        """
        ResourceManager._validate_account_id(account_id)
        ResourceManager._validate_stack_name(stack_name)

        session = CredentialProvider.get().get_session_for_account(
            account_id,
            ORG_ACCESS_ROLE,
            build_session_name("session"),
        )
        manager = CleanupManager(session, account_id=account_id)
        return await manager.cleanup_stack(stack_name)

    @staticmethod
    async def snapshot_scenarios(
        contexts: list[SnapshotContext],
    ) -> dict[str, list[SnapshotResult]]:
        """Capture baseline snapshots for scenarios.

        Thin async wrapper over SnapshotManager.snapshot_scenarios.

        Args:
            contexts: List of SnapshotContext (one per scenario)

        Returns:
            Dict mapping scenario name to list of SnapshotResult per account
        """
        snapshot_mgr = SnapshotManager()
        return await asyncio.to_thread(snapshot_mgr.snapshot_scenarios, contexts)

    @staticmethod
    def capture_pre_setup_baseline(
        account_id: str,
        scenario_name: str,
        regions: list[str],
        *,
        cred_provider: CredentialProvider,
    ) -> SnapshotResult:
        """Capture the pristine PRE_SETUP baseline for one provisioned account.

        Fast-scans ``account_id`` across ``regions`` and stores the result in the
        PRE_SETUP S3 slot that ``env cleanup`` subtracts. Idempotent: an existing
        baseline is left untouched, so re-running ``env init`` retries only accounts
        still missing one. Capture failures are reported through
        ``SnapshotResult.success``, not raised.

        Runs synchronously; call it from a thread when overlapping accounts.
        """
        mgr = SnapshotManager()
        with log_context(account_id):
            if mgr.snapshot_exists(scenario_name, account_id, SnapshotStage.PRE_SETUP):
                logger.debug("PRE_SETUP baseline exists for %s — skipping", scenario_name)
                return SnapshotResult(account_id=account_id, success=True, regions_captured=regions)
            session = cred_provider.get_session_for_account(
                account_id,
                ORG_ACCESS_ROLE,
                build_session_name("session"),
            )
            ctx = SnapshotContext(
                scenario_id=scenario_name,
                scenario_hash="",
                regions=regions,
                stage=SnapshotStage.PRE_SETUP,
                account_ids=[account_id],
            )
            return mgr.snapshot_account(session, account_id, ctx)

    @staticmethod
    async def verify_scenario(
        scenario_name: str,
        scenario_dir: Path | None,
        account_mapping: dict[str, str],
        region: str | None,
        max_concurrent: int = 10,
    ) -> list[AccountVerifyResult]:
        """Verify accounts match post-setup baseline state.

        Args:
            scenario_name: Name of the scenario
            scenario_dir: Path to scenario directory for version check (optional)
            account_mapping: Map of account tags to account IDs
            region: Optional region to verify (None = all baseline regions)
            max_concurrent: Maximum concurrent verifications

        Returns:
            List of verification results per account
        """
        cred_provider = CredentialProvider.get()
        regions = [region] if region else None
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _verify_account(tag: str, account_id: str) -> AccountVerifyResult:
            async with semaphore:
                with log_context(f"{tag}/{account_id}"):
                    try:
                        session = cred_provider.get_session_for_account(
                            account_id,
                            ORG_ACCESS_ROLE,
                            build_session_name("session"),
                        )
                        verify_mgr = VerifyManager(
                            session, region_name=region, account_id=account_id
                        )
                        # The default to_thread executor (~min(32, cpu+4)) is the
                        # real ceiling on account concurrency, not max_concurrent.
                        return await asyncio.to_thread(
                            verify_mgr.verify_account_multiregion,
                            scenario_name,
                            account_id,
                            scenario_name,  # environment_id
                            regions,
                            scenario_dir,
                        )
                    except Exception as exc:
                        logger.error("Failed to verify account: %s", exc)
                        return AccountVerifyResult(
                            account_id=account_id,
                            environment_id=scenario_name,
                            success=False,
                            region_results=[],
                            error_message=str(exc),
                        )

        tasks = [_verify_account(tag, aid) for tag, aid in account_mapping.items()]
        return await asyncio.gather(*tasks)

    @staticmethod
    async def verify_environment(
        test_environment: TestEnvironment,
        scenarios: list[Scenario],
        *,
        n_concurrent: int,
    ) -> VerificationReport:
        """Verify each scenario's account against its post-setup baseline, in parallel.

        Scenarios run under an outer semaphore (bound ``n_concurrent``); each
        delegates to ``verify_scenario``, which fans out its accounts internally.
        """
        semaphore = asyncio.Semaphore(n_concurrent)

        async def _verify_one(scenario: Scenario) -> list[AccountVerifyResult]:
            async with semaphore:
                with log_context(scenario.name):
                    return await ResourceManager.verify_scenario(
                        scenario_name=scenario.name,
                        scenario_dir=scenario.scenario_dir,
                        account_mapping=test_environment.mapping_for(scenario.name),
                        region=None,
                    )

        per_scenario = await asyncio.gather(*(_verify_one(s) for s in scenarios))
        results = [result for group in per_scenario for result in group]
        return VerificationReport(
            passed=all(r.success for r in results),
            env_name=test_environment.ou_name,
            results=results,
        )

    @staticmethod
    async def reset_scenarios(
        scenario_name: str,
        scenario_dir: Path | None,
        account_mapping: dict[str, str],
        max_concurrent: int,
        output_dir: Path | None = None,
    ) -> list[ResetResult]:
        """Reset accounts to post-setup baseline state.

        Args:
            scenario_name: Name of the scenario
            scenario_dir: Path to scenario directory for version check (optional)
            account_mapping: Map of account tags to account IDs
            max_concurrent: Maximum concurrent resets
            output_dir: Directory for reset outputs (e.g. the trial's reset phase
                dir). When None the manager uses its default location.

        Returns:
            List of reset results per account
        """
        cred_provider = CredentialProvider.get()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _reset_account(tag: str, account_id: str) -> ResetResult:
            async with semaphore:
                with log_context(f"{tag}/{account_id}"):
                    try:
                        session = cred_provider.get_session_for_account(
                            account_id,
                            ORG_ACCESS_ROLE,
                            build_session_name("session"),
                        )
                        reset_mgr = ResetManager(
                            # Nest per account tag (the unique mapping key) so concurrent
                            # accounts don't clobber each other's reset artifacts.
                            session,
                            output_dir=output_dir / tag if output_dir else None,
                            account_id=account_id,
                        )
                        return await reset_mgr.reset_account(
                            scenario_name,
                            account_id,
                            scenario_dir=scenario_dir,
                        )
                    except Exception as exc:
                        logger.error("Failed to reset account: %s", exc)
                        return ResetResult(
                            account_id=account_id,
                            scenario_name=scenario_name,
                            success=False,
                            reason=f"Reset failed: {exc}",
                            details={"error": str(exc), "account_id": account_id},
                        )

        tasks = [_reset_account(tag, aid) for tag, aid in account_mapping.items()]
        return await asyncio.gather(*tasks)

    @staticmethod
    async def cleanup_scenarios_by_name(
        scenario_name: str,
        account_mapping: dict[str, str],
        max_concurrent: int,
        all_regions: bool,
        output_dir: Path | None = None,
        regions: list[str] | None = None,
        *,
        sweep_post_setup: bool = True,
    ) -> list[AccountCleanupResult]:
        """Clean up CloudFormation stacks for scenario accounts.

        Args:
            scenario_name: Name of the scenario
            account_mapping: Map of account tags to account IDs
            max_concurrent: Maximum concurrent cleanups
            all_regions: If True, scan all regions; if False, use baseline regions from snapshot
            output_dir: Directory for cleanup outputs (e.g. the trial's cleanup phase
                dir). When None the manager uses its default location.
            regions: The scenario's declared regions, used as the floor when no baseline snapshot
                exists to derive them from (else cleanup discovers every enabled region and is
                SCP-denied on the ones the scenario never used).
            sweep_post_setup: If True (default), the in-cleanup Phase 1 post-setup
                residual sweep runs per region. Set False when the caller already
                swept residuals beforehand (e.g. the scenario trial's pre-cleanup sweep).

        Returns:
            List of cleanup results per account
        """
        cred_provider = CredentialProvider.get()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _cleanup_account(tag: str, account_id: str) -> AccountCleanupResult:
            async with semaphore:
                with log_context(f"{tag}/{account_id}"):
                    try:
                        session = cred_provider.get_session_for_account(
                            account_id,
                            ORG_ACCESS_ROLE,
                            build_session_name("session"),
                        )
                        cleanup_mgr = CleanupManager(
                            session,
                            # Nest per account tag (the unique mapping key): accounts clean
                            # concurrently and would otherwise clobber each other's summary.json /
                            # run_metadata.json / per-region manifests in a shared phase dir.
                            output_dir=output_dir / tag if output_dir else None,
                            env_name=scenario_name,
                            account_id=account_id,
                        )
                        # all_regions=False (default) limits cleanup to the baseline
                        # snapshot's regions; all_regions=True cleans every enabled region.
                        summary = await cleanup_mgr.cleanup_all_stacks(
                            regions=regions,
                            all_regions=all_regions,
                            sweep_post_setup=sweep_post_setup,
                        )
                        return AccountCleanupResult(
                            account_id=account_id, summary=summary, error=None
                        )
                    except Exception as exc:
                        logger.error("Failed to cleanup account: %s", exc)
                        return AccountCleanupResult(
                            account_id=account_id, summary=None, error=str(exc)
                        )

        tasks = [_cleanup_account(tag, aid) for tag, aid in account_mapping.items()]
        return await asyncio.gather(*tasks)

    @staticmethod
    async def sweep_scenario_residuals_by_name(
        scenario_name: str,
        account_mapping: dict[str, str],
        max_concurrent: int,
        all_regions: bool,
        output_dir: Path | None = None,
        regions: list[str] | None = None,
    ) -> list[AccountCleanupResult]:
        """Sweep post-setup residuals for scenario accounts, before ``cleanup.sh``.

        Mirrors ``cleanup_scenarios_by_name``'s per-account fan-out but runs only the
        post-setup residual sweep. ``regions`` is the scenario's declared regions, the
        floor for the no-baseline case (else the sweep discovers every enabled region
        and is SCP-denied on the ones the scenario never used). Best-effort: a
        per-account failure is captured in ``AccountCleanupResult.error`` (``summary``
        always None) rather than raised.
        """
        cred_provider = CredentialProvider.get()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _sweep_account(tag: str, account_id: str) -> AccountCleanupResult:
            async with semaphore:
                with log_context(f"{tag}/{account_id}"):
                    try:
                        session = cred_provider.get_session_for_account(
                            account_id,
                            ORG_ACCESS_ROLE,
                            build_session_name("session"),
                        )
                        cleanup_mgr = CleanupManager(
                            session,
                            # Nest per tag: accounts sweep concurrently and would
                            # otherwise clobber each other's per-region manifests.
                            output_dir=output_dir / tag if output_dir else None,
                            env_name=scenario_name,
                            account_id=account_id,
                        )
                        await cleanup_mgr.sweep_post_setup_residuals(
                            regions=regions, all_regions=all_regions
                        )
                        return AccountCleanupResult(account_id=account_id, summary=None, error=None)
                    except Exception as exc:
                        logger.error("Failed to sweep residuals for account: %s", exc)
                        return AccountCleanupResult(
                            account_id=account_id, summary=None, error=str(exc)
                        )

        tasks = [_sweep_account(tag, aid) for tag, aid in account_mapping.items()]
        return await asyncio.gather(*tasks)

    @staticmethod
    def _validate_cleanup_params(
        scenario_accounts: list[ScenarioAccount], max_concurrent: int
    ) -> None:
        """Validate cleanup_ou parameters."""
        if not scenario_accounts:
            raise ValueError("scenario_accounts list cannot be empty")
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got: {max_concurrent}")

    @staticmethod
    async def _cleanup_account_task(
        account_id: str, semaphore: asyncio.Semaphore, env_name: str | None = None
    ) -> AccountCleanupResult:
        """Clean up a single account with semaphore control.

        Catches all Exception subclasses and returns them in AccountCleanupResult.
        BaseException subclasses (KeyboardInterrupt, SystemExit) are NOT caught
        and will propagate to the caller.
        """
        async with semaphore:
            with log_context(account_id):
                logger.info("Starting cleanup")
                try:
                    summary = await ResourceManager.cleanup_account(account_id, env_name=env_name)
                    logger.info("Cleanup completed successfully")
                    return AccountCleanupResult(
                        account_id=account_id,
                        summary=summary,
                        error=None,
                    )
                except Exception as e:
                    logger.error("Cleanup failed: %s", e)
                    return AccountCleanupResult(
                        account_id=account_id,
                        summary=None,
                        error=str(e),
                    )


# Stack statuses considered "active" (not deleted)
ACTIVE_STACK_STATUSES = [
    "CREATE_COMPLETE",
    "CREATE_FAILED",
    "CREATE_IN_PROGRESS",
    "DELETE_FAILED",
    "DELETE_IN_PROGRESS",
    "IMPORT_COMPLETE",
    "IMPORT_IN_PROGRESS",
    "IMPORT_ROLLBACK_COMPLETE",
    "IMPORT_ROLLBACK_FAILED",
    "IMPORT_ROLLBACK_IN_PROGRESS",
    "ROLLBACK_COMPLETE",
    "ROLLBACK_FAILED",
    "ROLLBACK_IN_PROGRESS",
    "UPDATE_COMPLETE",
    "UPDATE_FAILED",
    "UPDATE_IN_PROGRESS",
    "UPDATE_ROLLBACK_COMPLETE",
    "UPDATE_ROLLBACK_FAILED",
    "UPDATE_ROLLBACK_IN_PROGRESS",
]


def list_account_stacks(account_id: str, cred_provider: CredentialProvider) -> list[dict[str, str]]:
    """List non-deleted CFN stacks across all enabled regions for a member account.

    Returns a list of dicts with keys: name, status, region.
    """
    stacks: list[dict[str, str]] = []
    try:
        session = cred_provider.get_session_for_account(
            account_id, ORG_ACCESS_ROLE, build_session_name("session")
        )
        regions = get_enabled_regions(session)
    except Exception:
        return stacks

    for region in regions:
        try:
            cfn = build_client(session, "cloudformation", region_name=region)
            paginator = cfn.get_paginator("list_stacks")
            for page in paginator.paginate(StackStatusFilter=ACTIVE_STACK_STATUSES):
                for s in page.get("StackSummaries", []):
                    stacks.append(
                        {
                            "name": s["StackName"],
                            "status": s["StackStatus"],
                            "region": region,
                        }
                    )
        except Exception:
            pass
    return stacks
