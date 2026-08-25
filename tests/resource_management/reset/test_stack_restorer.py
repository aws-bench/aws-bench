"""Tests for StackRestorer."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import boto3
import botocore.exceptions
import pytest
from moto import mock_aws

from aws_bench.resource_management.cleanup.manager import CleanupManager
from aws_bench.resource_management.reset.models import ResetupDeletion, RestoreOutcome
from aws_bench.resource_management.reset.stack_restorer import StackRestorer


@pytest.fixture
def session():
    """Create a boto3 session for testing."""
    return boto3.Session(region_name="us-east-1")


@pytest.fixture
def cleanup_mgr(session):
    """Create a CleanupManager for testing."""
    return CleanupManager(session)


@pytest.fixture
def restorer(session, cleanup_mgr):
    """Create a StackRestorer for testing."""
    return StackRestorer(session, cleanup_mgr)


# ===========================================================================
# restore_stack — main restoration workflow
# ===========================================================================


@mock_aws
def test_restore_stack_succeeds_with_drift_revert(restorer):
    """Restores stack successfully using drift revert."""
    baseline_drift = [{"LogicalResourceId": "MyRole", "StackResourceDriftStatus": "IN_SYNC"}]

    with patch.object(restorer, "_attempt_drift_revert", new_callable=AsyncMock) as mock_revert:
        mock_revert.return_value = True

        result = asyncio.run(
            restorer.restore_stack(
                stack_name="test-stack",
                baseline_drift=baseline_drift,
            )
        )

    assert result.outcome is RestoreOutcome.RESTORED
    assert result.abandoned == {}
    mock_revert.assert_called_once()


@mock_aws
def test_restore_stack_deletes_when_revert_fails(restorer):
    """Deletes the stack for re-setup when drift revert is not possible."""
    baseline_drift = [{"LogicalResourceId": "MyRole", "StackResourceDriftStatus": "IN_SYNC"}]

    with patch.object(restorer, "_attempt_drift_revert", new_callable=AsyncMock) as mock_revert:
        with patch.object(restorer, "_delete_for_resetup", new_callable=AsyncMock) as mock_delete:
            # Revert fails -> falls back to delete-for-resetup
            mock_revert.return_value = False
            mock_delete.return_value = ResetupDeletion(RestoreOutcome.DELETED_NEEDS_REDEPLOY)

            result = asyncio.run(
                restorer.restore_stack(
                    stack_name="test-stack",
                    baseline_drift=baseline_drift,
                )
            )

    assert result.outcome is RestoreOutcome.DELETED_NEEDS_REDEPLOY
    mock_revert.assert_called_once()
    mock_delete.assert_called_once_with("test-stack")


@mock_aws
def test_restore_stack_does_not_delete_on_shutdown(restorer):
    """A shutdown during revert propagates and must NOT fall through to deletion.

    A cancelled revert returning False would otherwise be read as "revert
    impossible" and delete the stack — a destructive action triggered by Ctrl+C.
    """
    from aws_bench.exceptions import OperationCancelled

    baseline_drift = [{"LogicalResourceId": "MyRole", "StackResourceDriftStatus": "IN_SYNC"}]

    with (
        patch.object(restorer, "_attempt_drift_revert", new_callable=AsyncMock) as mock_revert,
        patch.object(restorer, "_delete_for_resetup", new_callable=AsyncMock) as mock_delete,
    ):
        mock_revert.side_effect = OperationCancelled("stop")

        with pytest.raises(OperationCancelled):
            asyncio.run(restorer.restore_stack("test-stack", baseline_drift))

    mock_delete.assert_not_called()


# ===========================================================================
# _attempt_drift_revert — changeset creation errors
# ===========================================================================


@mock_aws
def test_attempt_drift_revert_handles_validation_error(restorer):
    """Handles ValidationError when creating changeset."""
    baseline_drift = []

    error_response = {"Error": {"Code": "ValidationError", "Message": "Invalid stack"}}
    error = botocore.exceptions.ClientError(error_response, "create_change_set")

    with patch.object(restorer._cfn_client, "create_change_set", side_effect=error):
        result = asyncio.run(restorer._attempt_drift_revert("test-stack", baseline_drift))

    assert result is False


@mock_aws
def test_attempt_drift_revert_handles_invalid_parameter_value(restorer):
    """Handles InvalidParameterValue when creating changeset."""
    baseline_drift = []

    error_response = {"Error": {"Code": "InvalidParameterValue", "Message": "Invalid parameter"}}
    error = botocore.exceptions.ClientError(error_response, "create_change_set")

    with patch.object(restorer._cfn_client, "create_change_set", side_effect=error):
        result = asyncio.run(restorer._attempt_drift_revert("test-stack", baseline_drift))

    assert result is False


@mock_aws
def test_attempt_drift_revert_raises_other_client_errors(restorer):
    """Raises other ClientErrors that aren't ValidationError."""
    baseline_drift = []

    error_response = {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}
    error = botocore.exceptions.ClientError(error_response, "create_change_set")

    with patch.object(restorer._cfn_client, "create_change_set", side_effect=error):
        result = asyncio.run(restorer._attempt_drift_revert("test-stack", baseline_drift))

    # Should return False due to outer try-except
    assert result is False


# ===========================================================================
# _attempt_drift_revert — changeset wait failures
# ===========================================================================


@mock_aws
def test_attempt_drift_revert_returns_false_when_changeset_wait_fails(restorer):
    """Returns False when changeset creation wait fails."""
    from aws_bench.resource_management.reset.models import ChangesetResult

    baseline_drift = []

    with patch.object(restorer._cfn_client, "create_change_set"):
        with patch.object(restorer, "_wait_for_changeset", new_callable=AsyncMock) as mock_wait:
            mock_wait.return_value = ChangesetResult.FAILED

            result = asyncio.run(restorer._attempt_drift_revert("test-stack", baseline_drift))

    assert result is False


# ===========================================================================
# _attempt_drift_revert — stack update failures
# ===========================================================================


@mock_aws
def test_attempt_drift_revert_handles_stack_update_failure(restorer):
    """Returns False when the stack update ends in a terminal non-success status."""
    from aws_bench.resource_management.reset.models import ChangesetResult

    baseline_drift = []

    with patch.object(restorer._cfn_client, "create_change_set"):
        with patch.object(restorer, "_wait_for_changeset", new_callable=AsyncMock) as mock_wait:
            with patch.object(restorer._cfn_client, "execute_change_set"):
                with patch.object(restorer._cfn_client, "describe_stacks") as mock_describe:
                    # Changeset creation succeeds
                    mock_wait.return_value = ChangesetResult.READY_TO_EXECUTE

                    # Stack update settles in a terminal non-success status
                    mock_describe.return_value = {
                        "Stacks": [{"StackStatus": "UPDATE_ROLLBACK_COMPLETE"}]
                    }

                    result = asyncio.run(
                        restorer._attempt_drift_revert("test-stack", baseline_drift)
                    )

    assert result is False


# ===========================================================================
# _attempt_drift_revert — success path
# ===========================================================================


@mock_aws
def test_attempt_drift_revert_succeeds_with_verification(restorer):
    """Successfully reverts drift and verifies baseline match."""
    from aws_bench.resource_management.reset.models import ChangesetResult

    baseline_drift = [{"LogicalResourceId": "MyRole", "StackResourceDriftStatus": "IN_SYNC"}]

    with patch.object(restorer._cfn_client, "create_change_set"):
        with patch.object(restorer, "_wait_for_changeset", new_callable=AsyncMock) as mock_wait:
            with patch.object(restorer._cfn_client, "execute_change_set"):
                with patch.object(restorer._cfn_client, "describe_stacks") as mock_describe:
                    with patch.object(
                        restorer, "_verify_drift_matches_baseline", new_callable=AsyncMock
                    ) as mock_verify:
                        # All operations succeed
                        mock_wait.return_value = ChangesetResult.READY_TO_EXECUTE

                        mock_describe.return_value = {
                            "Stacks": [{"StackStatus": "UPDATE_COMPLETE"}]
                        }

                        # Verification succeeds
                        mock_verify.return_value = True

                        result = asyncio.run(
                            restorer._attempt_drift_revert("test-stack", baseline_drift)
                        )

    assert result is True


# ===========================================================================
# _attempt_drift_revert — verification failures
# ===========================================================================


@mock_aws
def test_attempt_drift_revert_returns_false_when_verification_fails(restorer):
    """Returns False when drift verification doesn't match baseline."""
    from aws_bench.resource_management.reset.models import ChangesetResult

    baseline_drift = []

    with patch.object(restorer._cfn_client, "create_change_set"):
        with patch.object(restorer, "_wait_for_changeset", new_callable=AsyncMock) as mock_wait:
            with patch.object(restorer._cfn_client, "execute_change_set"):
                with patch.object(restorer._cfn_client, "describe_stacks") as mock_describe:
                    with patch.object(
                        restorer, "_verify_drift_matches_baseline", new_callable=AsyncMock
                    ) as mock_verify:
                        # All operations succeed
                        mock_wait.return_value = ChangesetResult.READY_TO_EXECUTE

                        mock_describe.return_value = {
                            "Stacks": [{"StackStatus": "UPDATE_COMPLETE"}]
                        }

                        # But verification fails
                        mock_verify.return_value = False

                        result = asyncio.run(
                            restorer._attempt_drift_revert("test-stack", baseline_drift)
                        )

    assert result is False


@mock_aws
def test_attempt_drift_revert_handles_exception(restorer):
    """Handles exception during drift revert."""
    baseline_drift = []

    with patch.object(
        restorer._cfn_client, "create_change_set", side_effect=Exception("Unexpected error")
    ):
        result = asyncio.run(restorer._attempt_drift_revert("test-stack", baseline_drift))

    assert result is False


# ===========================================================================
# _wait_for_changeset — success path
# ===========================================================================


@mock_aws
def test_wait_for_changeset_succeeds(restorer):
    """Successfully waits for changeset creation."""
    from aws_bench.resource_management.reset.models import ChangesetResult

    with patch.object(restorer._cfn_client, "describe_change_set") as mock_describe:
        mock_describe.return_value = {"Status": "CREATE_COMPLETE"}

        result = asyncio.run(restorer._wait_for_changeset("test-stack", "test-changeset", []))

    assert result == ChangesetResult.READY_TO_EXECUTE


# ===========================================================================
# _wait_for_changeset — no changes scenario
# ===========================================================================


@mock_aws
def test_wait_for_changeset_handles_no_changes(restorer):
    """Handles changeset with no changes."""
    from aws_bench.resource_management.reset.models import ChangesetResult

    with patch.object(restorer._cfn_client, "describe_change_set") as mock_describe:
        with patch.object(
            restorer, "_verify_drift_matches_baseline", new_callable=AsyncMock
        ) as mock_verify:
            # CloudFormation reports an empty changeset as FAILED with a telltale reason
            mock_describe.return_value = {
                "Status": "FAILED",
                "StatusReason": "The submitted information didn't contain changes.",
            }

            # Verification succeeds
            mock_verify.return_value = True

            result = asyncio.run(restorer._wait_for_changeset("test-stack", "test-changeset", []))

    assert result == ChangesetResult.ALREADY_BASELINE
    mock_verify.assert_called_once_with("test-stack", [], retries=2)


@mock_aws
def test_wait_for_changeset_no_changes_but_verification_fails(restorer):
    """Returns FAILED when no changes and verification fails."""
    from aws_bench.resource_management.reset.models import ChangesetResult

    with patch.object(restorer._cfn_client, "describe_change_set") as mock_describe:
        with patch.object(
            restorer, "_verify_drift_matches_baseline", new_callable=AsyncMock
        ) as mock_verify:
            # Empty changeset (no changes) reported as FAILED with the telltale reason
            mock_describe.return_value = {
                "Status": "FAILED",
                "StatusReason": "The submitted information didn't contain changes.",
            }

            # Verification fails
            mock_verify.return_value = False

            result = asyncio.run(restorer._wait_for_changeset("test-stack", "test-changeset", []))

    assert result == ChangesetResult.FAILED


# ===========================================================================
# _wait_for_changeset — invalid stack state
# ===========================================================================


@mock_aws
def test_wait_for_changeset_handles_invalid_stack_state(restorer):
    """Handles a FAILED changeset for an invalid stack state."""
    from aws_bench.resource_management.reset.models import ChangesetResult

    with patch.object(restorer._cfn_client, "describe_change_set") as mock_describe:
        mock_describe.return_value = {
            "Status": "FAILED",
            "StatusReason": "Stack is in UPDATE_ROLLBACK_COMPLETE state",
        }

        result = asyncio.run(restorer._wait_for_changeset("test-stack", "test-changeset", []))

    assert result == ChangesetResult.FAILED


@mock_aws
def test_wait_for_changeset_handles_cannot_be_updated(restorer):
    """Handles a FAILED changeset when the stack cannot be updated."""
    from aws_bench.resource_management.reset.models import ChangesetResult

    with patch.object(restorer._cfn_client, "describe_change_set") as mock_describe:
        mock_describe.return_value = {
            "Status": "FAILED",
            "StatusReason": "Stack cannot be updated in current state",
        }

        result = asyncio.run(restorer._wait_for_changeset("test-stack", "test-changeset", []))

    assert result == ChangesetResult.FAILED


@mock_aws
def test_wait_for_changeset_handles_invalid_error(restorer):
    """Handles a FAILED changeset with 'invalid' in the reason."""
    from aws_bench.resource_management.reset.models import ChangesetResult

    with patch.object(restorer._cfn_client, "describe_change_set") as mock_describe:
        mock_describe.return_value = {
            "Status": "FAILED",
            "StatusReason": "Invalid request parameters",
        }

        result = asyncio.run(restorer._wait_for_changeset("test-stack", "test-changeset", []))

    assert result == ChangesetResult.FAILED


# ===========================================================================
# _wait_for_changeset — other errors
# ===========================================================================


@mock_aws
def test_wait_for_changeset_handles_describe_client_error(restorer):
    """Handles a ClientError raised while describing the changeset."""
    from aws_bench.resource_management.reset.models import ChangesetResult

    error_response = {"Error": {"Code": "Throttling", "Message": "Some other error occurred"}}
    error = botocore.exceptions.ClientError(error_response, "describe_change_set")

    with patch.object(restorer._cfn_client, "describe_change_set", side_effect=error):
        result = asyncio.run(restorer._wait_for_changeset("test-stack", "test-changeset", []))

    assert result == ChangesetResult.FAILED


# ===========================================================================
# _delete_for_resetup — delete so setup can recreate the stack
# ===========================================================================


@mock_aws
def test_delete_for_resetup_succeeds(restorer):
    """Returns DELETED_NEEDS_REDEPLOY after a successful stack deletion."""
    with patch.object(
        restorer._cleanup_mgr, "cleanup_stack", new_callable=AsyncMock
    ) as mock_cleanup:
        cleanup_result = MagicMock()
        cleanup_result.all_stacks_succeeded = True
        cleanup_result.orphaned_resources = {}
        mock_cleanup.return_value = cleanup_result

        result = asyncio.run(restorer._delete_for_resetup("test-stack"))

    assert result.outcome is RestoreOutcome.DELETED_NEEDS_REDEPLOY
    assert result.abandoned == {}
    mock_cleanup.assert_called_once_with("test-stack")


@mock_aws
def test_delete_for_resetup_carries_abandoned_from_cleanup(restorer):
    """Force-abandoned resources on cleanup_stack surface as ResetupDeletion.abandoned.

    cleanup_stack reports FORCE_DELETE-abandoned resources in orphaned_resources
    (type -> [id]); _delete_for_resetup reshapes them to type -> [{"Identifier": id}]
    so reset can fold them into unresolved_orphans and fail closed.
    """
    with patch.object(
        restorer._cleanup_mgr, "cleanup_stack", new_callable=AsyncMock
    ) as mock_cleanup:
        cleanup_result = MagicMock()
        cleanup_result.all_stacks_succeeded = True
        cleanup_result.orphaned_resources = {
            "AWS::EC2::InternetGateway": ["igw-abandoned"],
        }
        mock_cleanup.return_value = cleanup_result

        result = asyncio.run(restorer._delete_for_resetup("test-stack"))

    assert result.outcome is RestoreOutcome.DELETED_NEEDS_REDEPLOY
    assert result.abandoned == {
        "AWS::EC2::InternetGateway": [{"Identifier": "igw-abandoned"}],
    }


@mock_aws
def test_delete_for_resetup_returns_failed_when_deletion_fails(restorer):
    """Returns FAILED when the stack deletion does not succeed."""
    with patch.object(
        restorer._cleanup_mgr, "cleanup_stack", new_callable=AsyncMock
    ) as mock_cleanup:
        cleanup_result = MagicMock()
        cleanup_result.all_stacks_succeeded = False
        mock_cleanup.return_value = cleanup_result

        result = asyncio.run(restorer._delete_for_resetup("test-stack"))

    assert result.outcome is RestoreOutcome.FAILED


@mock_aws
def test_delete_for_resetup_handles_exception(restorer):
    """Returns FAILED when an exception occurs during deletion."""
    with patch.object(
        restorer._cleanup_mgr,
        "cleanup_stack",
        new_callable=AsyncMock,
        side_effect=Exception("Unexpected error"),
    ):
        result = asyncio.run(restorer._delete_for_resetup("test-stack"))

    assert result.outcome is RestoreOutcome.FAILED


@mock_aws
def test_delete_for_resetup_already_gone_is_success(restorer):
    """A stack that is already gone counts as deleted-for-resetup, not a failure.

    cleanup_stack raises ValueError('... not found in any region') when the
    stack no longer exists (e.g. a ROLLBACK_COMPLETE/empty stack removed before
    or during reset). The goal of delete_for_resetup is for the stack to be
    absent so setup recreates it — already-absent satisfies that, so it must
    return DELETED_NEEDS_REDEPLOY (nothing abandoned) rather than FAILED.
    """
    with patch.object(
        restorer._cleanup_mgr,
        "cleanup_stack",
        new_callable=AsyncMock,
        side_effect=ValueError("Stack 'test-stack' not found in any region"),
    ):
        result = asyncio.run(restorer._delete_for_resetup("test-stack"))

    assert result.outcome is RestoreOutcome.DELETED_NEEDS_REDEPLOY
    assert result.abandoned == {}


# ===========================================================================
# _verify_drift_matches_baseline — detection failures
# ===========================================================================


@mock_aws
def test_verify_drift_matches_baseline_handles_detection_failed(restorer):
    """Retries when drift detection fails."""
    baseline_drift = []

    with (
        patch.object(restorer._cfn_client, "detect_stack_drift") as mock_detect,
        patch.object(restorer._cfn_client, "describe_stack_drift_detection_status") as mock_status,
    ):
        # Setup detection
        mock_detect.return_value = {"StackDriftDetectionId": "drift-123"}

        # But status shows DETECTION_FAILED
        mock_status.return_value = {"DetectionStatus": "DETECTION_FAILED"}

        result = asyncio.run(
            restorer._verify_drift_matches_baseline("test-stack", baseline_drift, retries=1)
        )

    assert result is False


@mock_aws
def test_verify_drift_matches_baseline_retries_on_detection_failure(restorer):
    """Retries multiple times on detection failure."""
    baseline_drift = []

    with patch.object(restorer._cfn_client, "detect_stack_drift") as mock_detect:
        with patch.object(
            restorer._cfn_client, "describe_stack_drift_detection_status"
        ) as mock_status:
            # Setup detection
            mock_detect.return_value = {"StackDriftDetectionId": "drift-123"}

            # Status always shows DETECTION_FAILED
            mock_status.return_value = {"DetectionStatus": "DETECTION_FAILED"}

            with patch("asyncio.sleep"):
                result = asyncio.run(
                    restorer._verify_drift_matches_baseline("test-stack", baseline_drift, retries=3)
                )

    # Should retry 3 times
    assert mock_detect.call_count == 3
    assert result is False


@mock_aws
def test_verify_drift_matches_baseline_succeeds_after_retry(restorer):
    """Succeeds after retrying on detection failure."""
    baseline_drift = []

    with patch.object(restorer._cfn_client, "detect_stack_drift") as mock_detect:
        with patch.object(
            restorer._cfn_client, "describe_stack_drift_detection_status"
        ) as mock_status:
            with patch.object(
                restorer._cfn_client, "describe_stack_resource_drifts"
            ) as mock_drifts:
                with patch(
                    "aws_bench.resource_management.reset.stack_restorer.drifts_match"
                ) as mock_match:
                    # Setup detection
                    mock_detect.return_value = {"StackDriftDetectionId": "drift-123"}

                    # First attempt fails, second succeeds
                    mock_status.side_effect = [
                        {"DetectionStatus": "DETECTION_FAILED"},
                        {"DetectionStatus": "DETECTION_COMPLETE"},
                    ]

                    mock_drifts.return_value = {"StackResourceDrifts": []}
                    mock_match.return_value = True

                    with patch("asyncio.sleep"):
                        result = asyncio.run(
                            restorer._verify_drift_matches_baseline(
                                "test-stack", baseline_drift, retries=3
                            )
                        )

    assert mock_detect.call_count == 2
    assert result is True


# ===========================================================================
# _verify_drift_matches_baseline — comparison logic
# ===========================================================================


@mock_aws
def test_verify_drift_matches_baseline_compares_drift(restorer):
    """Compares drift correctly using drifts_match."""
    baseline_drift = [{"LogicalResourceId": "MyRole", "StackResourceDriftStatus": "IN_SYNC"}]

    with patch.object(restorer._cfn_client, "detect_stack_drift") as mock_detect:
        with patch.object(
            restorer._cfn_client, "describe_stack_drift_detection_status"
        ) as mock_status:
            with patch.object(
                restorer._cfn_client, "describe_stack_resource_drifts"
            ) as mock_drifts:
                with patch(
                    "aws_bench.resource_management.reset.stack_restorer.drifts_match"
                ) as mock_match:
                    # Setup mocks
                    mock_detect.return_value = {"StackDriftDetectionId": "drift-123"}

                    mock_status.return_value = {"DetectionStatus": "DETECTION_COMPLETE"}

                    current_drift = [
                        {"LogicalResourceId": "MyRole", "StackResourceDriftStatus": "MODIFIED"}
                    ]
                    mock_drifts.return_value = {"StackResourceDrifts": current_drift}

                    mock_match.return_value = False

                    result = asyncio.run(
                        restorer._verify_drift_matches_baseline(
                            "test-stack", baseline_drift, retries=1
                        )
                    )

    assert result is False
    mock_match.assert_called_once_with(current_drift, baseline_drift)


# ===========================================================================
# _verify_drift_matches_baseline — exception handling
# ===========================================================================


@mock_aws
def test_verify_drift_matches_baseline_handles_exception(restorer):
    """Handles exception during drift verification."""
    baseline_drift = []

    with patch.object(
        restorer._cfn_client, "detect_stack_drift", side_effect=Exception("Drift failed")
    ):
        result = asyncio.run(
            restorer._verify_drift_matches_baseline("test-stack", baseline_drift, retries=1)
        )

    assert result is False


@mock_aws
def test_verify_drift_matches_baseline_retries_on_exception(restorer):
    """Retries on exception during drift verification."""
    baseline_drift = []

    with patch.object(
        restorer._cfn_client, "detect_stack_drift", side_effect=Exception("Drift failed")
    ):
        with patch("asyncio.sleep"):
            result = asyncio.run(
                restorer._verify_drift_matches_baseline("test-stack", baseline_drift, retries=3)
            )

    assert result is False


@mock_aws
def test_verify_drift_matches_baseline_sleeps_between_retries(restorer):
    """Sleeps between retries when exceptions occur."""
    baseline_drift = []

    with patch.object(
        restorer._cfn_client, "detect_stack_drift", side_effect=Exception("Drift failed")
    ):
        with patch("asyncio.sleep") as mock_sleep:
            result = asyncio.run(
                restorer._verify_drift_matches_baseline("test-stack", baseline_drift, retries=2)
            )

    # Should sleep once (after first failure, but not after last)
    assert mock_sleep.call_count == 1
    mock_sleep.assert_called_with(5)
    assert result is False


@mock_aws
def test_verify_drift_matches_baseline_succeeds_on_final_retry(restorer):
    """Succeeds on final retry attempt."""
    baseline_drift = []

    with patch.object(restorer._cfn_client, "detect_stack_drift") as mock_detect:
        with patch.object(
            restorer._cfn_client, "describe_stack_drift_detection_status"
        ) as mock_status:
            with patch.object(
                restorer._cfn_client, "describe_stack_resource_drifts"
            ) as mock_drifts:
                with patch(
                    "aws_bench.resource_management.reset.stack_restorer.drifts_match"
                ) as mock_match:
                    # First two attempts fail, third succeeds
                    mock_detect.side_effect = [
                        Exception("Failed"),
                        Exception("Failed"),
                        {"StackDriftDetectionId": "drift-123"},
                    ]

                    mock_status.return_value = {"DetectionStatus": "DETECTION_COMPLETE"}
                    mock_drifts.return_value = {"StackResourceDrifts": []}
                    mock_match.return_value = True

                    with patch("asyncio.sleep"):
                        result = asyncio.run(
                            restorer._verify_drift_matches_baseline(
                                "test-stack", baseline_drift, retries=3
                            )
                        )

    assert result is True


# ===========================================================================
# _verify_drift_matches_baseline — returns False after all retries
# ===========================================================================


@mock_aws
def test_verify_drift_matches_baseline_returns_false_after_all_retries_exhausted(restorer):
    """Returns False when all retry attempts are exhausted."""
    baseline_drift = []

    with patch.object(
        restorer._cfn_client, "detect_stack_drift", side_effect=Exception("Persistent failure")
    ):
        with patch("asyncio.sleep"):
            result = asyncio.run(
                restorer._verify_drift_matches_baseline("test-stack", baseline_drift, retries=5)
            )

    assert result is False
