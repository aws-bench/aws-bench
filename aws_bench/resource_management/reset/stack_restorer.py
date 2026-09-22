"""Stack restoration operations for reset."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import boto3
import botocore.exceptions

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.cleanup.manager import CleanupManager
from aws_bench.resource_management.reset.models import (
    ChangesetResult,
    ResetupDeletion,
    RestoreOutcome,
)
from aws_bench.resource_management.snapshot.drift import DRIFT_CLIENT_CONFIG, DRIFT_DETECTION_FAILED
from aws_bench.resource_management.utils.cloudformation import (
    get_stack_resource_drifts,
    wait_for_changeset_creation,
    wait_for_drift_detection,
    wait_for_stack_update,
)
from aws_bench.resource_management.verify.comparators import drifts_match
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)


class StackRestorer:
    """Handles stack restoration to baseline drift state."""

    def __init__(
        self,
        session: boto3.Session,
        cleanup_mgr: CleanupManager,
    ):
        """Initialize with boto3 session and dependencies.

        Args:
            session: boto3 Session for AWS operations
            cleanup_mgr: CleanupManager for stack deletion
        """
        self._session = session
        self._cleanup_mgr = cleanup_mgr
        # Adaptive retries: drifted stacks are restored concurrently in waves,
        # so the client self-throttles under CloudFormation rate limits.
        self._cfn_client = build_client(session, "cloudformation", config=DRIFT_CLIENT_CONFIG)

    async def restore_stack(
        self,
        stack_name: str,
        baseline_drift: list[dict[str, Any]],
    ) -> ResetupDeletion:
        """Restore stack to baseline drift state.

        Strategy:
        1. Try drift revert via REVERT_DRIFT changeset (in place).
        2. Fallback: delete the stack and signal that it must be recreated by
           re-running setup. The container's idempotent ``deploy.sh`` recreates
           only the missing stack, so the redeploy happens at the CLI layer.

        Args:
            stack_name: Name of stack to fix
            baseline_drift: Expected drift state from snapshot

        Returns:
            ResetupDeletion: RESTORED with no abandoned resources if reverted in
            place; otherwise the outcome of the delete-for-re-setup fallback
            (DELETED_NEEDS_REDEPLOY / FAILED plus any force-abandoned resources).
        """
        logger.debug(f"Attempting to fix drift for stack {stack_name}")

        # Attempt 1: Drift revert
        if await self._attempt_drift_revert(stack_name, baseline_drift):
            return ResetupDeletion(RestoreOutcome.RESTORED)

        # Attempt 2: Delete so setup can recreate it
        return await self._delete_for_resetup(stack_name)

    async def _attempt_drift_revert(
        self, stack_name: str, baseline_drift: list[dict[str, Any]]
    ) -> bool:
        """Attempt to revert drift via changeset.

        Args:
            stack_name: Name of stack to fix
            baseline_drift: Expected drift state

        Returns:
            True if revert succeeded, False otherwise
        """
        change_set_name = None
        try:
            logger.debug(f"Attempting drift revert for {stack_name}")

            # Create changeset for drift revert
            change_set_name = f"{stack_name}-revert-{int(datetime.now(timezone.utc).timestamp())}"

            try:
                await asyncio.to_thread(
                    self._cfn_client.create_change_set,
                    StackName=stack_name,
                    ChangeSetName=change_set_name,
                    ChangeSetType="UPDATE",
                    UsePreviousTemplate=True,
                    IncludeNestedStacks=False,
                    Capabilities=[
                        "CAPABILITY_IAM",
                        "CAPABILITY_NAMED_IAM",
                        "CAPABILITY_AUTO_EXPAND",
                    ],
                )
            except botocore.exceptions.ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ["ValidationError", "InvalidParameterValue"]:
                    logger.debug(f"{stack_name}: Cannot create changeset")
                    return False
                raise

            # Wait for changeset creation
            changeset_result = await self._wait_for_changeset(
                stack_name, change_set_name, baseline_drift
            )
            if changeset_result == ChangesetResult.ALREADY_BASELINE:
                logger.debug(f"{stack_name}: Already at baseline, no revert needed")
                return True
            elif changeset_result == ChangesetResult.FAILED:
                return False

            # Execute changeset
            await asyncio.to_thread(
                self._cfn_client.execute_change_set,
                StackName=stack_name,
                ChangeSetName=change_set_name,
            )

            # A False return is recovered: restore_stack falls through to delete+redeploy.
            status = await asyncio.to_thread(wait_for_stack_update, self._cfn_client, stack_name)
            if status != "UPDATE_COMPLETE":
                logger.warning(
                    f"{stack_name}: Drift revert ended in {status}, will delete for re-setup"
                )
                return False

            logger.debug(f"{stack_name}: Drift revert completed")

            if await self._verify_drift_matches_baseline(stack_name, baseline_drift, retries=3):
                logger.debug(f"{stack_name}: Successfully restored to baseline via drift revert")
                return True

            logger.warning(
                f"{stack_name}: Drift revert didn't restore baseline, will delete for re-setup"
            )
            return False

        except Exception as e:
            logger.warning(f"{stack_name}: Drift revert failed ({e}), will delete for re-setup")
            return False
        finally:
            if change_set_name:
                try:
                    await asyncio.to_thread(
                        self._cfn_client.delete_change_set,
                        StackName=stack_name,
                        ChangeSetName=change_set_name,
                    )
                except Exception:
                    pass

    async def _wait_for_changeset(
        self, stack_name: str, change_set_name: str, baseline_drift: list[dict[str, Any]]
    ) -> ChangesetResult:
        """Wait for changeset creation to complete.

        Args:
            stack_name: Name of stack
            change_set_name: Name of changeset
            baseline_drift: Expected drift state for verification

        Returns:
            ChangesetResult indicating outcome
        """
        try:
            change_set = await asyncio.to_thread(
                wait_for_changeset_creation, self._cfn_client, stack_name, change_set_name
            )
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            logger.warning(
                f"{stack_name}: Changeset creation failed ({e}), will delete for re-setup"
            )
            return ChangesetResult.FAILED

        if change_set["Status"] == "CREATE_COMPLETE":
            return ChangesetResult.READY_TO_EXECUTE

        # CloudFormation reports an empty changeset (no drift to revert) as FAILED
        # with a telltale StatusReason — distinguish it from a real failure so an
        # already-baseline stack isn't treated as an error.
        reason = change_set.get("StatusReason", "").lower()
        if "didn't contain changes" in reason or "no updates are to be performed" in reason:
            logger.debug(f"{stack_name}: No drift to revert")
            matches_baseline = await self._verify_drift_matches_baseline(
                stack_name, baseline_drift, retries=2
            )
            return ChangesetResult.ALREADY_BASELINE if matches_baseline else ChangesetResult.FAILED

        logger.warning(f"{stack_name}: Changeset not created ({reason}), will delete for re-setup")
        return ChangesetResult.FAILED

    async def _delete_for_resetup(self, stack_name: str) -> ResetupDeletion:
        """Delete a stack that couldn't be reverted so setup can recreate it.

        The stack is recreated by re-running ``aws-bench env setup`` (the
        container's idempotent ``deploy.sh`` recreates only the missing stack).
        This method only performs the deletion and signals that a redeploy is
        required.

        Args:
            stack_name: Name of stack to delete

        Returns:
            ResetupDeletion carrying DELETED_NEEDS_REDEPLOY on success (with any
            resources FORCE_DELETE_STACK abandoned, from ``cleanup_stack``'s
            ``orphaned_resources``) or FAILED otherwise. Already-absent counts as
            a successful delete with nothing abandoned.
        """
        try:
            logger.debug(f"{stack_name}: Drift revert not possible; deleting for re-setup")

            # Delete via CleanupManager
            deletion_result = await self._cleanup_mgr.cleanup_stack(stack_name)

            if not deletion_result.all_stacks_succeeded:
                logger.error(f"{stack_name}: Deletion failed")
                return ResetupDeletion(RestoreOutcome.FAILED)

            logger.debug(
                f"{stack_name}: Deletion succeeded; will be recreated by 'aws-bench env setup'"
            )
            abandoned = {
                rtype: [{"Identifier": i} for i in ids]
                for rtype, ids in deletion_result.orphaned_resources.items()
            }
            return ResetupDeletion(RestoreOutcome.DELETED_NEEDS_REDEPLOY, abandoned=abandoned)

        except ValueError as e:
            # cleanup_stack raises ValueError("... not found in any region") when
            # the stack is already gone. The goal here is for the stack to be
            # ABSENT so setup recreates it — already-absent satisfies that, so
            # treat it as a successful delete rather than a reset-failing error.
            if "not found in any region" in str(e):
                logger.debug(
                    f"{stack_name}: Already absent; will be recreated by 'aws-bench env setup'"
                )
                return ResetupDeletion(RestoreOutcome.DELETED_NEEDS_REDEPLOY)
            logger.error(f"{stack_name}: Deletion for re-setup failed: {e}")
            return ResetupDeletion(RestoreOutcome.FAILED)

        except Exception as e:
            logger.error(f"{stack_name}: Deletion for re-setup failed: {e}")
            return ResetupDeletion(RestoreOutcome.FAILED)

    async def _verify_drift_matches_baseline(
        self, stack_name: str, baseline_drift: list[dict[str, Any]], retries: int = 3
    ) -> bool:
        """Run drift detection and compare to baseline with retries.

        Args:
            stack_name: Name of stack to verify
            baseline_drift: Expected drift state
            retries: Number of attempts

        Returns:
            True if drift matches baseline, False otherwise
        """
        for attempt in range(retries):
            try:
                # Detect drift
                response = await asyncio.to_thread(
                    self._cfn_client.detect_stack_drift, StackName=stack_name
                )
                detection_id = response["StackDriftDetectionId"]

                status = await asyncio.to_thread(
                    wait_for_drift_detection, self._cfn_client, detection_id
                )

                if status == DRIFT_DETECTION_FAILED:
                    if attempt < retries - 1:
                        logger.debug(f"Drift detection failed, retrying... (attempt {attempt + 1})")
                        await asyncio.sleep(5)
                        continue
                    return False

                # Compare drift - use utility function to get all drifts
                current_drift = await asyncio.to_thread(
                    get_stack_resource_drifts, self._cfn_client, stack_name
                )

                return drifts_match(current_drift, baseline_drift)

            except Exception as e:
                logger.debug(f"Drift verification attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(5)
                else:
                    return False

        return False
