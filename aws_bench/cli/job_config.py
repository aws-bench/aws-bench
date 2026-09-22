"""``AwsBenchJobConfig`` — the ``run`` command's job config.

A ``JobConfig`` carrying a single scenario-aware ``AwsBenchDatasetConfig``, the
operator's environment name, and the resolved test environment. Rejects an
authored ``tasks`` / ``datasets`` key, any non-Docker environment, and
``install_only``.
"""

from typing import Any, override

from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig, JobConfig
from harbor.models.trial.config import TaskConfig
from pydantic import Field, model_validator

from aws_bench.account_management.models import TestEnvironment
from aws_bench.dataset.config import AwsBenchDatasetConfig


class AwsBenchJobConfig(JobConfig):
    """Job config for the ``run`` command.

    Persists as the job's ``config.json``. ``env_name`` and ``test_environment``
    are part of the run's identity (see ``__eq__``), so a changed environment
    refuses resume. Per-trial CFN/SSM exports are ``exclude=True`` on the trial
    config: re-resolved each run, never persisted.
    """

    dataset: AwsBenchDatasetConfig | None = None
    # A value above 1 is safe: the queue's per-scenario gate keeps same-scenario
    # mutating trials from overlapping on one account.
    n_concurrent_trials: int = 4
    env_name: str | None = None
    # Default-on pre-run AWS drift check; --no-verify-env opts out.
    verify: bool = True
    test_environment: TestEnvironment | None = None
    # Resolution is driven by ``dataset``; these inherited fields stay empty and
    # are excluded from the dump.
    tasks: list[TaskConfig] = Field(default_factory=list, exclude=True)
    datasets: list[DatasetConfig] = Field(default_factory=list, exclude=True)

    @override
    def __eq__(self, other: object) -> bool:
        # Like the base dump comparison but also excludes ``verify`` (persisted
        # for audit, not part of run identity); the base hardcodes its exclude set.
        if not isinstance(other, AwsBenchJobConfig):
            return NotImplemented
        exclude = {"job_name", "debug", "verify"}
        return self.model_dump(exclude=exclude) == other.model_dump(exclude=exclude)

    @model_validator(mode="before")
    @classmethod
    def _reject_authored_tasks_and_datasets(cls, data: Any) -> Any:
        """Reject the inherited ``tasks`` / ``datasets`` input keys."""
        if isinstance(data, dict):
            unsupported = [key for key in ("tasks", "datasets") if data.get(key)]
            if unsupported:
                raise ValueError(
                    f"aws-bench run config does not use {unsupported}; declare a "
                    f"single 'dataset' instead."
                )
        return data

    @model_validator(mode="after")
    def _reject_non_docker_environment(self) -> "AwsBenchJobConfig":
        """Require ``environment.type`` be Docker (or unset) with no import_path."""
        if self.environment.import_path is not None:
            raise ValueError(
                "A custom environment import_path is not supported; aws-bench "
                "runs trials in the built-in Docker environment."
            )
        if self.environment.type not in (None, EnvironmentType.DOCKER):
            raise ValueError(
                f"aws-bench only supports the Docker environment; got "
                f"environment.type={self.environment.type}."
            )
        return self

    @model_validator(mode="after")
    def _reject_install_only(self) -> "AwsBenchJobConfig":
        """Reject ``install_only``, which the aws-bench trial cannot honor.

        Harbor's ``JobConfig`` disables the verifier for it while
        ``AwsBenchJob._build_trial_config`` never forwards the flag, so the agent
        would run in full unverified.
        """
        if self.install_only:
            raise ValueError("install_only is not supported by aws-bench.")
        return self
