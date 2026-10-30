"""Typed aws-bench task model: task.toml parsing + construction-time validation.

``AwsBenchTaskConfig`` extends Harbor's task.toml model with the aws-bench
sections (``[scenario]``, ``[concurrency]``, ``[pre_invoke]``, ``[post_invoke]``)
so all of task.toml parses in one ``model_validate_toml`` pass. ``AwsBenchTask``
carries that config and validates the aws-bench task layout at construction time.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import cast

from harbor.environments.definition import has_agent_environment_definition
from harbor.models.task.config import TaskConfig as HarborTaskTomlConfig
from harbor.models.task.paths import TaskPaths
from harbor.models.task.task import Task
from harbor.models.task.verifier_mode import (
    VerifierEnvironmentMode,
    resolve_task_verifier_mode,
)
from harbor.models.trial.config import TaskConfig
from harbor.tasks.client import TaskClient
from harbor.utils.scripts import discover_script
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aws_bench.constants import TASK_CACHE_DIR
from aws_bench.dataset.models import RoleType, ScriptType


class ConcurrencyMode(StrEnum):
    """Author-declared trial behavior for within-scenario admission.

    ``MUTATING`` (default) holds the scenario account exclusively; ``READ_ONLY``
    trials of one scenario co-run. A pure scheduling declaration — NOT tied to the
    task's IAM role or any credential scope; the framework never derives or enforces it.
    """

    READ_ONLY = "read-only"
    MUTATING = "mutating"


class ConcurrencyConfig(BaseModel):
    """task.toml ``[concurrency]`` — the per-task concurrency declaration."""

    mode: ConcurrencyMode = ConcurrencyMode.MUTATING
    # Skip the framework reset (account scan, drift check, cleanup) after a trial whose
    # post_invoke ran and exited 0; the scenario's reset.sh still runs. post_invoke must
    # revert and verify the task's changes.
    skip_framework_reset: bool = False

    @model_validator(mode="after")
    def _reject_skip_on_read_only(self) -> ConcurrencyConfig:
        if self.skip_framework_reset and self.mode is ConcurrencyMode.READ_ONLY:
            raise ValueError("skip_framework_reset applies to mutating tasks only")
        return self


class ScenarioRef(BaseModel):
    """task.toml ``[scenario]`` — the scenario reference + per-phase IAM roles."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    agent_role_name: str | None = None
    agent_account_tags: list[str] | None = Field(default=None, min_length=1)
    verifier_role_name: str | None = None
    pre_invoke_role_name: str | None = None
    post_invoke_role_name: str | None = None

    def role_name(self, role_type: RoleType) -> str | None:
        """Return the IAM role name for the given phase, or None if unset."""
        return {
            RoleType.AGENT: self.agent_role_name,
            RoleType.VERIFIER: self.verifier_role_name,
            RoleType.PRE_INVOKE: self.pre_invoke_role_name,
            RoleType.POST_INVOKE: self.post_invoke_role_name,
        }[role_type]


class PhaseScript(BaseModel):
    """task.toml ``[pre_invoke]`` / ``[post_invoke]`` — timeout + env only.

    Whether the script RUNS is answered by ``AwsBenchTask.phase_script_path``
    (file on disk).
    """

    timeout_sec: float = 600.0
    env: dict[str, str] = Field(default_factory=dict)


class AwsBenchTaskConfig(HarborTaskTomlConfig):
    """Harbor task.toml model plus the aws-bench sections, parsed in one pass.

    ``model_validate_toml`` is cls-bound, so it returns this subtype with the
    ``[scenario]``, ``[concurrency]``, ``[pre_invoke]``, and ``[post_invoke]``
    sections populated.
    """

    scenario: ScenarioRef
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    pre_invoke: PhaseScript = Field(default_factory=PhaseScript)
    post_invoke: PhaseScript = Field(default_factory=PhaseScript)


class AwsBenchTask(Task):
    """Harbor Task carrying an AwsBenchTaskConfig plus aws-bench task-side behavior.

    Construction order: ``super().__init__`` runs Harbor's instruction and
    test-script validation, then ``self.config`` is re-parsed as the subtype so
    the aws-bench sections are populated, then ``_validate_layout`` runs the
    aws-bench-specific checks.
    """

    config: AwsBenchTaskConfig

    def __init__(
        self, task_dir: Path | str, extra_instruction_paths: list[Path] | None = None
    ) -> None:
        """Build the task, then validate its layout (raises on a malformed task)."""
        super().__init__(task_dir, extra_instruction_paths)
        # model_validate_toml is cls-bound (returns AwsBenchTaskConfig at runtime)
        # but its declared return type is the base TaskConfig; cast to the subtype.
        # This reassignment must stay the LAST config mutation in __init__: until it
        # runs, self.config carries only the base task.toml fields (the aws-bench
        # sections are absent), so no config-dependent logic may run before it.
        self.config = cast(
            AwsBenchTaskConfig,
            AwsBenchTaskConfig.model_validate_toml(self.paths.config_path.read_text()),
        )
        self._validate_layout()

    @classmethod
    async def from_config(
        cls, config: TaskConfig, extra_instruction_paths: list[Path] | None = None
    ) -> AwsBenchTask:
        """Construct an AwsBenchTask from a git or local task reference.

        A git task is downloaded under ``TASK_CACHE_DIR`` (or the config's
        ``download_dir`` override) and read from the resulting path; a local task
        is read from its literal path. ``download_tasks`` is idempotent, so a
        prior batched ``cache_tasks`` is a cache hit. Package references are
        refused (only git/local are supported).
        """
        if config.is_package_task():
            raise NotImplementedError(
                f"aws-bench does not support package task references; got {config.get_task_id()}."
            )
        if config.is_git_task():
            task_dir = (
                await TaskClient().download_tasks(
                    task_ids=[config.get_task_id()],
                    overwrite=config.overwrite,
                    output_dir=config.download_dir or TASK_CACHE_DIR,
                )
            ).paths[0]
            return cls(task_dir, extra_instruction_paths)
        if config.path is None:
            raise ValueError("Task path must be set for a local task.")
        return cls(config.path, extra_instruction_paths)

    @staticmethod
    def is_valid_dir(task_dir: Path | str, disable_verification: bool = False) -> bool:
        """Structural discovery filter for local-task listing: True iff a task.toml exists.

        Does not validate contents — full validation runs at ``AwsBenchTask``
        construction, so a malformed ``task.toml``-bearing dir fails loud and
        named during ``get_tasks``.

        ``disable_verification`` is accepted for signature parity with the
        overridden ``Task.is_valid_dir`` staticmethod; the structural check is
        unconditional and ignores it.
        """
        return TaskPaths(task_dir).config_path.is_file()

    def phase_script_path(self, script_type: ScriptType) -> Path | None:
        """OS-aware path to a phase script, or None if absent.

        Convention: <task_dir>/<type>/<type>.<ext>.
        """
        return discover_script(
            self.paths.task_dir / script_type,
            script_type,
            task_os=self.config.environment.os,
        )

    def has_phase_script(self, script_type: ScriptType) -> bool:
        """Return whether a script file exists on disk for the given phase."""
        return self.phase_script_path(script_type) is not None

    def _validate_layout(self) -> None:
        """Validate the aws-bench task layout at construction time.

        (1) The environment must be buildable now: a Dockerfile,
        docker-compose.yaml, or ``[environment].docker_image`` (verified via
        ``has_agent_environment_definition``).
        (2) The verifier must run in shared mode; separate verifier environments
        are not supported (phase scripts run in the agent container, which a
        separate verifier environment tears down before verification).
        """
        if not has_agent_environment_definition(
            self.paths.environment_dir,
            docker_image=self.config.environment.docker_image,
        ):
            raise FileNotFoundError(
                f"{self.paths.task_dir}: environment/ has no Dockerfile, "
                "docker-compose.yaml, or [environment].docker_image."
            )
        if resolve_task_verifier_mode(self.config) == VerifierEnvironmentMode.SEPARATE:
            raise ValueError(
                f"{self.paths.task_dir}: separate-environment verifiers are not "
                "supported; remove [verifier].environment / environment_mode."
            )
