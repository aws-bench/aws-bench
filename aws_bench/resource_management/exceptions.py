"""Exception hierarchy for the resource management module."""

from aws_bench.exceptions import AWSBenchError
from aws_bench.resource_management.snapshot.models import SnapshotStage


class PollTimeout(AWSBenchError):
    """A polled CloudFormation operation did not reach a terminal status in time."""


class DriftDetectionError(AWSBenchError):
    """Drift detection failed for one or more stacks after retries.

    Aborts snapshot capture so an unreliable baseline is never saved.
    """


class DeploymentError(AWSBenchError):
    """Any deployment-related failure."""


class ConfigurationError(DeploymentError):
    """Invalid configuration or missing prerequisites."""


class CleanupError(AWSBenchError):
    """Base exception for cleanup operations."""


class SnapshotNotFoundError(AWSBenchError):
    """Raised when snapshot cannot be found for an account."""

    def __init__(
        self, env_name: str, account_id: str, stage: SnapshotStage = SnapshotStage.POST_SETUP
    ):
        """Initialize snapshot not found error.

        Args:
            env_name: Environment name
            account_id: AWS account ID
            stage: Snapshot stage (default: POST_SETUP)
        """
        self.env_name = env_name
        self.account_id = account_id
        self.stage = stage
        remedy = "env init" if stage == SnapshotStage.PRE_SETUP else "env setup"
        super().__init__(
            f"No {stage} snapshot found for account {account_id} in environment {env_name}. "
            f"Run 'aws-bench {remedy}' to create the baseline."
        )


class SnapshotRegionMismatchError(AWSBenchError):
    """The PRE_SETUP snapshot records a different region set than scenario.toml."""

    def __init__(
        self,
        env_name: str,
        account_id: str,
        declared_regions: list[str],
        recorded_regions: list[str],
    ):
        """Keep the declared and recorded region sets on the error."""
        self.env_name = env_name
        self.account_id = account_id
        self.declared_regions = declared_regions
        self.recorded_regions = recorded_regions
        super().__init__(
            f"Scenario '{env_name}' declares regions {sorted(declared_regions)} but the PRE_SETUP "
            f"snapshot for account {account_id} records {sorted(recorded_regions)}. "
            "Restore the previous regions in scenario.toml, run 'aws-bench env cleanup', "
            "then set the new regions and rerun 'aws-bench env init'."
        )


class EnvironmentVerifyError(AWSBenchError):
    """Environment verification failed before running trials."""
