"""Scenario job/trial *input* models: retry policy, account mapping, configs.

Builds on the leaf modules ``locator`` (the source descriptor) and
``scenario.config`` (the manifest's ``TrialEnvironmentConfig``). Holds the
configuration a run is launched with; results live in ``scenario.results``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import shortuuid
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aws_bench.account_management.models import TestEnvironment
from aws_bench.dataset.config import AwsBenchDatasetConfig
from aws_bench.scenario.config import TrialEnvironmentConfig
from aws_bench.scenario.locator import ScenarioConfig

_TRIAL_NAME_MAX_PREFIX = 32
_TRIAL_NAME_SUFFIX_LEN = 7


def generate_trial_name(scenario_name: str) -> str:
    """Build a per-run unique trial name.

    Truncates the scenario name to 32 characters, trims trailing
    separators, and appends a 7-character shortuuid for uniqueness.
    """
    prefix = scenario_name[:_TRIAL_NAME_MAX_PREFIX].rstrip("_-")
    suffix = shortuuid.ShortUUID().random(length=_TRIAL_NAME_SUFFIX_LEN)
    return f"{prefix}__{suffix}"


class RetryConfig(BaseModel):
    """Per-trial retry policy for the queue.

    ``max_retries=0`` (default) means no retries — a single attempt per
    trial. Increase to retry transient failures (Docker daemon flakes,
    AWS throttling).

    ``include_exceptions`` is an allowlist of types to retry; if None,
    every exception class is eligible. ``exclude_exceptions`` is checked
    first and always wins. Exception names are matched by ``__name__``
    so subclasses are not auto-included — list them explicitly if needed.
    """

    max_retries: int = Field(default=0, ge=0, description="Maximum number of retry attempts")
    include_exceptions: set[str] | None = Field(
        default=None,
        description="Exception types to retry on. None = retry all.",
    )
    exclude_exceptions: set[str] | None = Field(
        default=None,
        description="Exception types to never retry. Takes precedence over include.",
    )
    wait_multiplier: float = Field(default=2.0, gt=0, description="Exponential backoff multiplier")
    min_wait_sec: float = Field(default=1.0, ge=0, description="Minimum wait between retries")
    max_wait_sec: float = Field(default=60.0, ge=0, description="Maximum wait between retries")

    @classmethod
    def from_cli_args(
        cls,
        max_retries: int | None = None,
        retry_include: list[str] | None = None,
        retry_exclude: list[str] | None = None,
    ) -> "RetryConfig":
        """Build a ``RetryConfig`` from CLI flags; only override defaults when explicitly set."""
        cfg = cls()
        if max_retries is not None:
            cfg.max_retries = max_retries
        if retry_include is not None:
            cfg.include_exceptions = set(retry_include)
        if retry_exclude is not None:
            cfg.exclude_exceptions = set(retry_exclude)
        return cfg


class ScenarioJobConfig(BaseModel):
    """Configuration for a scenario job.

    STS credentials are NOT stored — they are resolved on demand by an
    injected ``CredentialProvider`` (see ``ScenarioJob.create``).

    ``test_environment`` is part of the run's *identity*, not just audit
    metadata: a scenario run against a different OU / account set is a
    different run. ``ScenarioJob.create`` populates it from the OU on a
    deep copy of the operator's config, so the operator's input is not
    mutated. It mirrors ``AwsBenchJobConfig.test_environment`` on the
    task side, carrying full per-account detail (scenario, tag, email).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ou_name: str
    dataset: AwsBenchDatasetConfig
    jobs_dir: Path = Path("scenario-jobs")
    job_name: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S")
    )
    n_concurrent: int = 4
    timeout_multiplier: float = 1.0
    environment: TrialEnvironmentConfig = Field(default_factory=TrialEnvironmentConfig)
    # Per-trial retry policy, applied by the queue. Lives on the config;
    # ``ScenarioJob.create`` reads it rather than taking a separate argument.
    retry: RetryConfig = Field(default_factory=RetryConfig)
    # Resolved OU + per-scenario accounts — populated by ScenarioJob.create
    # from the OU; None when the operator constructs a config directly.
    test_environment: TestEnvironment | None = None

    @classmethod
    def from_cli_args(
        cls,
        ou_name: str,
        dataset: AwsBenchDatasetConfig,
        job_name: str | None = None,
        jobs_dir: Path | None = None,
        n_concurrent: int | None = None,
        timeout_multiplier: float | None = None,
        force_build: bool | None = None,
        delete: bool | None = None,
        override_cpus: int | None = None,
        override_memory_mb: int | None = None,
        override_build_timeout_sec: float | None = None,
        mounts: str | None = None,
        max_retries: int | None = None,
        retry_include: list[str] | None = None,
        retry_exclude: list[str] | None = None,
    ) -> "ScenarioJobConfig":
        """Build a ``ScenarioJobConfig`` from CLI flags.

        Only overrides model defaults when the corresponding CLI flag was passed.
        """
        env_overrides = TrialEnvironmentConfig(
            force_build=force_build if force_build is not None else False,
            delete=delete if delete is not None else True,
            override_cpus=override_cpus,
            override_memory_mb=override_memory_mb,
            override_build_timeout_sec=override_build_timeout_sec,
            mounts_json=json.loads(mounts) if mounts else None,
        )
        job_cfg = cls(
            ou_name=ou_name,
            dataset=dataset,
            environment=env_overrides,
            retry=RetryConfig.from_cli_args(max_retries, retry_include, retry_exclude),
        )
        if job_name is not None:
            job_cfg.job_name = job_name
        if jobs_dir is not None:
            job_cfg.jobs_dir = jobs_dir
        if n_concurrent is not None:
            job_cfg.n_concurrent = n_concurrent
        if timeout_multiplier is not None:
            job_cfg.timeout_multiplier = timeout_multiplier
        return job_cfg


class ScenarioTrialConfig(BaseModel):
    """Descriptor-only trial config.

    Carries a ``ScenarioConfig`` descriptor (path + optional git provenance)
    plus the operator's trial-level overrides. The parsed manifest is NOT
    re-bundled here — it lives on the materialized ``Scenario`` for the
    trial's lifetime and can be re-read from the descriptor's path.
    Persisted ``config.json`` contains the descriptor only.

    ``trial_name`` defaults to ``<scenario_name_truncated>__<shortuuid>``
    so re-running the same scenario in the same job produces a fresh
    trial directory and container name.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario: ScenarioConfig
    trial_name: str = ""
    output_dir: Path
    environment: TrialEnvironmentConfig = Field(default_factory=TrialEnvironmentConfig)
    account_mapping: dict[str, str]
    timeout_multiplier: float = 1.0
    ou_name: str = ""
    # Docker labels applied to this trial's scenario container. Operational only
    # (e.g. ``{"awsbench.role": "scenario-reset"}`` so tooling can match reset
    # containers by role, not name); excluded from the persisted config.json so
    # it stays off disk and does not perturb resume identity.
    labels: dict[str, str] = Field(default_factory=dict, exclude=True)
    # Phase-specific resource management parameters
    verify_region: str | None = None
    reset_max_concurrent: int = 10
    cleanup_max_concurrent: int = 10
    cleanup_all_regions: bool = False

    @model_validator(mode="after")
    def _set_default_trial_name(self) -> ScenarioTrialConfig:
        if not self.trial_name:
            self.trial_name = generate_trial_name(self.scenario.name)
        return self
