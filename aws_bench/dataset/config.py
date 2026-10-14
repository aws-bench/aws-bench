"""AwsBenchDatasetConfig — dataset source config with two modes.

Local (``path`` / ``scenarios_path``) and JSON registry (``name@version``).
Resolves both tasks-side and scenarios-side descriptors.
"""

from __future__ import annotations

import subprocess
import tomllib
from fnmatch import fnmatch
from pathlib import Path

from harbor.models.job.config import DatasetConfig as HarborDatasetConfig
from harbor.models.metric.config import MetricConfig
from harbor.models.metric.type import MetricType
from harbor.models.task.id import GitTaskId, LocalTaskId
from harbor.models.trial.config import TaskConfig
from harbor.tasks.client import BatchDownloadResult, TaskClient, TaskIdType
from pydantic import ConfigDict, Field, PrivateAttr, ValidationError, model_validator

from aws_bench.constants import (
    DEFAULT_REGISTRY_URL,
    INSTRUCTION_CACHE_DIR,
    METRIC_CACHE_DIR,
    SCENARIO_CACHE_DIR,
    TASK_CACHE_DIR,
)
from aws_bench.dataset.exceptions import (
    ExtraInstructionFetchError,
    MetricFetchError,
    RegistryValidationError,
    ScenarioFetchError,
    TaskConfigInvalidError,
)
from aws_bench.dataset.registry import (
    AwsBenchDatasetSpec,
    AwsBenchRegistry,
    RegistryValidator,
)
from aws_bench.dataset.source_ids import GitUVScriptId, LocalScenarioId, ScenarioIdType
from aws_bench.dataset.task_config import AwsBenchTask
from aws_bench.logging.logger import get_logger
from aws_bench.scenario.exceptions import DuplicateScenarioNameError, ScenarioDiscoveryError
from aws_bench.scenario.locator import ScenarioConfig
from aws_bench.scenario.scenario import Scenario
from aws_bench.utils.retry import retrying_git_fetch

logger = get_logger(__name__)


class AwsBenchDatasetConfig(HarborDatasetConfig):
    """Dataset source config. Two modes: local + JSON registry.

    Provides scenarios-side resolution (``scenarios_path``,
    ``include_scenario_names`` / ``exclude_scenario_names``,
    ``get_scenario_configs``, ``cache_scenarios``, ``get_scenarios``)
    alongside the task resolution it inherits.
    """

    # extra="allow" lets tests replace instance methods via patch.object;
    # Pydantic v2 rejects setattr on names that are not declared fields.
    model_config = ConfigDict(extra="allow")

    scenarios_path: Path | None = None
    include_scenario_names: list[str] | None = Field(default=None)
    exclude_scenario_names: list[str] | None = None
    scenario_download_dir: Path | None = Field(
        default=None,
        description="Directory to cache remote scenarios to. Defaults to SCENARIO_CACHE_DIR.",
    )

    # Memoized, validated registry (registry mode only). Loaded + validated
    # once by ``_load_registry``; resolution then re-reads it for free.
    _registry_cache: "AwsBenchRegistry | None" = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_dataset_source(self) -> "AwsBenchDatasetConfig":  # type: ignore[override]
        """Validate the dataset source: local paths xor a registry name.

        Rejects package names (``org/name``), ``ref``, and registry fields
        paired with local paths.
        """
        local = self.path is not None or self.scenarios_path is not None
        registry = self.name is not None
        if local and registry:
            raise ValueError(
                "Specify either local paths (-p / --scenario-path) or a "
                "registry dataset (-d), not both."
            )
        if not local and not registry:
            raise ValueError(
                "No dataset source provided. Pass -p <tasks-dir>, "
                "--scenario-path <scenarios-dir>, or -d <name>[@<version>]."
            )
        if self.name is not None and "/" in self.name:
            raise ValueError(
                f"Package datasets ('org/name' form) are not supported; aws-bench "
                f"uses a JSON registry. Got dataset name {self.name!r}."
            )
        if self.ref is not None:
            raise ValueError(
                "'ref' selects a package-dataset version, which is not supported. "
                "Use a registry dataset with -d <name>@<version>."
            )
        if local and (self.registry_url is not None or self.registry_path is not None):
            raise ValueError(
                "registry_url / registry_path apply only to registry datasets "
                "(-d <name>); they are not valid with local paths "
                "(-p / --scenario-path)."
            )
        return self

    @classmethod
    def from_cli_args(
        cls,
        scenario_path: Path | None = None,
        dataset: str | None = None,
        registry_url: str | None = None,
        registry_path: Path | None = None,
        include_scenarios: list[str] | None = None,
        exclude_scenarios: list[str] | None = None,
        *,
        path: Path | None = None,
        include_task_names: list[str] | None = None,
        exclude_task_names: list[str] | None = None,
        n_tasks: int | None = None,
    ) -> "AwsBenchDatasetConfig":
        """Build a dataset from CLI flags (env and run share this constructor).

        ``dataset`` is the ``name[@version]`` string from ``-d``; ``scenario_path``
        is ``--scenario-path``. ``path`` and the ``*_task_names`` / ``n_tasks``
        kwargs are run-only. The source rule (local paths xor registry name) is
        enforced by the model validator; command-specific rules are
        ``validate_run()`` / ``validate_env()``.
        """
        dataset_name, dataset_version = (None, None)
        if dataset is not None:
            name, _, version = dataset.partition("@")
            dataset_name, dataset_version = name or None, version or None
        return cls(
            name=dataset_name,
            version=dataset_version,
            path=path,
            scenarios_path=scenario_path,
            registry_url=registry_url,
            registry_path=registry_path,
            include_scenario_names=include_scenarios,
            exclude_scenario_names=exclude_scenarios,
            task_names=include_task_names,
            exclude_task_names=exclude_task_names,
            n_tasks=n_tasks,
        )

    def is_local(self) -> bool:
        """``True`` when any local source (tasks ``path`` or ``scenarios_path``) is set.

        ``scenarios_path`` satisfies "local mode" in addition to ``path``.
        """
        return self.path is not None or self.scenarios_path is not None

    def validate_run(self) -> None:
        """Require a local run dataset to declare both a tasks dir and a scenarios dir.

        Raises ``ValueError`` if local with only one of ``path`` /
        ``scenarios_path``. Registry datasets pass unchanged.
        """
        if self.is_local() and (self.path is None or self.scenarios_path is None):
            raise ValueError(
                f"A local run dataset needs both a tasks directory and a "
                f"scenarios directory: pass -p <tasks-dir> and "
                f"--scenario-path <scenarios-dir> "
                f"(got path={self.path}, scenarios_path={self.scenarios_path})."
            )

    def validate_env(self) -> None:
        """Require an env dataset to be scenarios-only (no tasks dir).

        Raises ``ValueError`` if ``path`` is set. Registry and scenarios-only
        datasets pass unchanged.
        """
        if self.path is not None:
            raise ValueError(
                f"The env command takes scenarios only; a tasks directory "
                f"(-p {self.path}) is not valid here. Pass --scenario-path "
                f"<scenarios-dir> or -d <name>[@<version>]."
            )

    # ---- Descriptors ----
    def get_scenario_configs(self) -> list[ScenarioConfig]:
        """Return scenario descriptors for the configured source.

        Local mode reads ``scenarios_path``; registry mode loads the JSON
        registry and resolves the named dataset. Filters apply to both.
        """
        if self.is_local():
            return self._get_local_scenario_configs()
        return self._get_registry_scenario_configs()

    def _filter_scenario_ids(self, scenario_ids: list[ScenarioIdType]) -> list[ScenarioIdType]:
        """Apply include/exclude name globs to scenario ids (mirrors ``_filter_task_ids``)."""
        filtered_ids = scenario_ids
        if self.include_scenario_names:
            filtered_ids = [
                sid
                for sid in filtered_ids
                if any(fnmatch(sid.get_name(), pattern) for pattern in self.include_scenario_names)
            ]
            if not filtered_ids:
                available = sorted(sid.get_name() for sid in scenario_ids)
                raise ValueError(
                    f"No scenarios matched the filter(s) {self.include_scenario_names}. "
                    f"There are {len(available)} scenarios available in this dataset. "
                    f"Example scenario names: {available[:5]}"
                )
        if self.exclude_scenario_names:
            filtered_ids = [
                sid
                for sid in filtered_ids
                if not any(
                    fnmatch(sid.get_name(), pattern) for pattern in self.exclude_scenario_names
                )
            ]
        return filtered_ids

    def _get_local_scenario_configs(self) -> list[ScenarioConfig]:
        """Resolve local scenario directories into config objects."""
        # is_valid_dir is a structural filter only; each listed dir is fully
        # validated at Scenario construction in get_scenarios, so a malformed
        # scenario fails loud and named rather than being silently dropped.
        if self.scenarios_path is None:
            return []
        source = self.scenarios_path.expanduser().resolve().name
        scenario_ids: list[ScenarioIdType] = [
            LocalScenarioId(path=path)
            for path in sorted(self.scenarios_path.iterdir())
            if Scenario.is_valid_dir(path)
        ]
        return [
            ScenarioConfig(name=sid.path.name, path=sid.path, source=source)
            for sid in self._filter_scenario_ids(scenario_ids)
        ]

    def _registry_spec(self) -> "AwsBenchDatasetSpec":
        """Resolve the registry dataset spec. Registry mode only.

        Raises ``RegistryValidationError`` if called outside registry mode
        (callers must check ``is_local()`` first). Resolution is a dict lookup
        over the memoized, already-validated registry.
        """
        if self.name is None:
            raise RegistryValidationError(
                "Registry spec requested outside registry mode (no dataset name set)."
            )
        return self._load_registry().get_dataset_spec(self.name, self.version)

    def _get_registry_scenario_configs(self) -> list[ScenarioConfig]:
        """Build scenario configs from the registry dataset spec."""
        spec = self._registry_spec()
        configs = [
            ScenarioConfig(
                name=s.name,
                path=s.path,
                git_url=s.git_url,
                git_commit_id=s.git_commit_id,
                overwrite=self.overwrite,
                download_dir=self.scenario_download_dir,
                source=self.name,
            )
            for s in spec.scenarios
        ]
        kept = set(self._filter_scenario_ids([c.to_scenario_id() for c in configs]))
        return [c for c in configs if c.to_scenario_id() in kept]

    # ---- Cache (batched fetch) ----
    async def _fetch_remote_scenarios(
        self, remote: list[ScenarioConfig], output_dir: Path
    ) -> BatchDownloadResult:
        """Fetch remote scenarios with retry for transient git/CodeCommit failures."""
        return await retrying_git_fetch(
            lambda: TaskClient().download_tasks(
                task_ids=[c.to_scenario_id() for c in remote],
                overwrite=self.overwrite,
                output_dir=output_dir,
                export=False,
            )
        )

    async def cache_scenarios(
        self, scenario_configs: list[ScenarioConfig] | None = None
    ) -> dict[ScenarioIdType, Path]:
        """Materialize git scenario dirs on disk; return their actual on-disk paths.

        Batches git refs through ``TaskClient().download_tasks`` in a single
        call and returns ``{scenario_id: result.path}`` keyed off the download
        results, which honor ``scenario_download_dir``. Local refs are absent
        from the map; callers resolve them via their literal ``get_local_path()``.
        A fetch failure raises ``ScenarioFetchError`` so a missing scenario
        aborts the run rather than being silently dropped.
        """
        if scenario_configs is None:
            scenario_configs = self.get_scenario_configs()
        remote = [c for c in scenario_configs if c.is_git()]
        if not remote:
            return {}

        output_dir = self.scenario_download_dir or SCENARIO_CACHE_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            batch = await self._fetch_remote_scenarios(remote, output_dir)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            raise ScenarioFetchError(
                f"Failed to fetch scenarios via git. "
                f"Underlying error: {stderr.strip() or e}. "
                "Try --scenario-path <local-dir> to work offline."
            ) from e
        except (FileNotFoundError, ValueError, OSError) as e:
            raise ScenarioFetchError(
                f"Failed to materialize scenarios: {type(e).__name__}: {e}. "
                "Try --scenario-path <local-dir> to work offline."
            ) from e

        return {
            c.to_scenario_id(): result.path for c, result in zip(remote, batch.results, strict=True)
        }

    async def _fetch_one(self, source_id: GitTaskId | LocalTaskId, output_dir: Path) -> Path:
        """Fetch one git/local reference via ``TaskClient``; return its path.

        Uses the download machinery (git clone + sparse-checkout +
        commit-addressed cache + dir copy) for single-directory fetches
        (extra-instructions, git uv-scripts). Raises the underlying error;
        callers wrap it in a typed error.
        """
        batch = await retrying_git_fetch(
            lambda: TaskClient().download_tasks(
                task_ids=[source_id],
                overwrite=self.overwrite,
                output_dir=output_dir,
                export=False,
            )
        )
        return batch.results[0].path

    # ---- High-level: descriptors → materialized scenarios ----
    async def get_scenarios(
        self, scenario_configs: list[ScenarioConfig] | None = None
    ) -> dict[ScenarioConfig, Scenario]:
        """Build and validate every scenario, keyed by its descriptor.

        Single pass over one list of scenario configs, in descriptor order. When
        ``scenario_configs`` is supplied the result matches that exact list; when
        omitted, ``get_scenario_configs()`` resolves them. ``cache_scenarios``
        batches the git downloads first; each descriptor then materializes via
        ``Scenario.from_config`` (git scenarios from the cached download path,
        local from their literal path), which validates layout + manifest.
        Keying by the descriptor keeps each scenario's source/overwrite/git
        provenance attached for ``.values()`` / ``.keys()`` / ``.items()`` reads.
        Per-scenario failures are collected and raised once as
        ``ScenarioDiscoveryError``; a duplicate manifest name fails loud.
        """
        if scenario_configs is None:
            scenario_configs = self.get_scenario_configs()
        # Batch the git downloads first; from_config then hits the warm cache
        # (download_tasks is idempotent).
        await self.cache_scenarios(scenario_configs)
        scenarios: dict[ScenarioConfig, Scenario] = {}
        seen_names: dict[str, Path] = {}
        errors: list[tuple[Path, Exception]] = []
        for c in scenario_configs:
            p = c.get_local_path()
            try:
                scenario = await Scenario.from_config(c)
            except (
                FileNotFoundError,
                OSError,
                tomllib.TOMLDecodeError,
                ValidationError,
                KeyError,
                ValueError,
            ) as e:
                errors.append((p, e))
                continue
            # Distinct source dirs can still declare the same manifest
            # ``[scenario].name``. That name is the run-wide key for account
            # mapping, quota gating, and task->scenario resolution, so a
            # collision would silently bind a task to the wrong scenario and
            # merge their accounts. Reject it, naming both colliding dirs.
            first = seen_names.get(scenario.name)
            if first is not None:
                raise DuplicateScenarioNameError(
                    f"Duplicate scenario name {scenario.name!r}: declared by both "
                    f"{first} and {p}. Scenario names must be unique within a dataset."
                )
            seen_names[scenario.name] = p
            scenarios[c] = scenario
        if errors:
            raise ScenarioDiscoveryError(errors)
        return scenarios

    # ---- Registry-declared metrics & extra-instructions ----
    async def resolve_instruction_paths(self) -> list[Path]:
        """Resolve the dataset's registry-declared extra-instruction files.

        Registry mode only (local datasets have none — the job flow can call
        this unconditionally and get ``[]`` back). Fetches each git-referenced
        file (reusing TaskClient via _fetch_one) and passes local paths through
        unchanged. A fetch failure raises ``ExtraInstructionFetchError`` and
        aborts the run — a missing instruction changes eval semantics, so it
        fails loud rather than silently running without it. Warns on an
        empty/whitespace-only resolved file.
        """
        if self.is_local():
            return []
        spec = self._registry_spec()
        if not spec.extra_instruction_paths:
            return []
        INSTRUCTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        resolved: list[Path] = []
        for ref in spec.extra_instruction_paths:
            try:
                if ref.git_url is not None:
                    fetched_dir = await self._fetch_one(ref.to_source_id(), INSTRUCTION_CACHE_DIR)
                    path = fetched_dir / ref.path.name
                    if not path.is_file():
                        raise ExtraInstructionFetchError(
                            f"Instruction file {ref.path.name!r} not found in "
                            f"fetched {ref.git_url}@{ref.git_commit_id}"
                        )
                else:
                    path = ref.path.expanduser()
            except ExtraInstructionFetchError:
                # A missing/unfetchable instruction changes eval semantics,
                # so abort rather than silently run without it.
                raise
            except Exception as e:  # noqa: BLE001 — wrap unknown errors, then abort
                raise ExtraInstructionFetchError(
                    f"Failed to resolve instruction {ref.path}: {type(e).__name__}: {e}"
                ) from e
            if not path.read_text().strip():
                # An empty extra-instruction contributes nothing to the agent
                # prompt yet changes eval semantics, so surface it to the operator.
                logger.warning("Extra-instruction file %s is empty; appending nothing.", path)
            resolved.append(path)
        return resolved

    async def resolve_metric_configs(self) -> list[MetricConfig]:
        """Resolve the dataset's registry-declared metrics into MetricConfigs.

        Declarative metrics (anything not ``uv-script``) pass through verbatim;
        their kwargs are validated downstream at job construction. A uv-script
        may be either:

        - git: ``git_url`` + ``git_commit_id`` + ``script_path`` in kwargs. The
          ``script_path`` is repo-relative; we fetch via ``GitUVScriptId`` and
          rewrite it to the fetched local path, keeping the same kwarg key.
        - local: ``script_path`` only (no ``git_url``). The path is a literal
          local path read directly, so we pass the metric through verbatim.

        A git fetch failure raises ``MetricFetchError`` and aborts the run.
        Registry mode only; local datasets declare no registry metrics, so the
        job flow calls this unconditionally and gets ``[]`` back.
        """
        if self.is_local():
            return []
        spec = self._registry_spec()
        if not spec.metrics:
            return []
        resolved: list[MetricConfig] = []
        for m in spec.metrics:
            if m.type != MetricType.UV_SCRIPT:
                resolved.append(m)
                continue
            if "git_url" not in m.kwargs:
                resolved.append(m)
                continue
            # git uv-script: fetch the referenced dir and rewrite script_path
            # to the fetched local path, keeping the same kwarg key. A git
            # uv-script needs both git_commit_id and script_path alongside
            # git_url; check up front so a malformed declaration fails with an
            # actionable message rather than an opaque KeyError downstream.
            missing = [k for k in ("git_commit_id", "script_path") if k not in m.kwargs]
            if missing:
                raise MetricFetchError(
                    f"git uv-script metric is missing required kwarg(s) {missing}: "
                    f"a git_url uv-script must also declare git_commit_id and script_path."
                )
            try:
                METRIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                repo_rel = Path(m.kwargs["script_path"])
                source_id = GitUVScriptId(
                    git_url=m.kwargs["git_url"],
                    git_commit_id=m.kwargs["git_commit_id"],
                    path=repo_rel.parent,
                )
                fetched_dir = await self._fetch_one(source_id, METRIC_CACHE_DIR)
                local_script = fetched_dir / repo_rel.name
                if not local_script.is_file():
                    raise MetricFetchError(
                        f"metric script {repo_rel.name!r} not found in fetched dataset"
                    )
            except MetricFetchError:
                # The operator declared this metric; a fetch failure aborts
                # rather than silently scoring with a different/floor metric.
                raise
            except Exception as e:  # noqa: BLE001 — wrap unknown errors, then abort
                raise MetricFetchError(
                    f"Failed to resolve metric {m.kwargs.get('script_path')}: "
                    f"{type(e).__name__}: {e}"
                ) from e
            resolved.append(MetricConfig(type=m.type, kwargs={"script_path": str(local_script)}))
        return resolved

    # ---- Tasks-side registry override ----
    async def _get_registry_task_configs(self) -> list[TaskConfig]:
        """Build TaskConfigs from the aws-bench JSON registry.

        Uses ``AwsBenchRegistry``, which (1) resolves only against the configured
        ``registry_path`` / ``registry_url`` — never an external hosted registry —
        and (2) accepts the aws-bench-specific ``scenarios[]`` key on each dataset
        entry.
        """
        spec = self._registry_spec()
        # to_source_task_id() returns GitTaskId | LocalTaskId; the wider TaskIdType
        # annotation matches the list type _filter_task_ids accepts.
        task_ids: list[TaskIdType] = [t.to_source_task_id() for t in spec.tasks]
        result: list[TaskConfig] = []
        for tid in self._filter_task_ids(task_ids):
            if isinstance(tid, GitTaskId):
                result.append(
                    TaskConfig(
                        path=tid.path,
                        git_url=tid.git_url,
                        git_commit_id=tid.git_commit_id,
                        overwrite=self.overwrite,
                        download_dir=self.download_dir,
                        source=self.name,
                    )
                )
            elif isinstance(tid, LocalTaskId):
                result.append(
                    TaskConfig(
                        path=tid.path,
                        overwrite=self.overwrite,
                        download_dir=self.download_dir,
                        source=self.name,
                    )
                )
        return result

    # ---- Tasks-side: deterministic local listing ----
    @staticmethod
    def _list_local_task_dirs(tasks_path: Path) -> list[Path]:
        """List task dirs under ``tasks_path``, flat or grouped one level deep.

        Accepts both ``<tasks-path>/<task>`` and ``<tasks-path>/<scenario>/<task>``.
        A dir that is itself a task is never descended into; nesting stops at
        one level.
        """
        task_dirs: list[Path] = []
        for child in sorted(tasks_path.iterdir()):
            if not child.is_dir():
                continue
            if AwsBenchTask.is_valid_dir(child):
                task_dirs.append(child)
                continue
            task_dirs.extend(
                grandchild
                for grandchild in sorted(child.iterdir())
                if AwsBenchTask.is_valid_dir(grandchild)
            )
        return task_dirs

    async def _get_local_task_configs(self, disable_verification: bool) -> list[TaskConfig]:
        """Resolve local task directories into config objects."""
        # is_valid_dir is a structural filter only; each listed dir is fully
        # validated at AwsBenchTask construction in get_tasks, so a malformed
        # task fails loud and named. disable_verification is unused (validation
        # is unconditional); it is accepted for base-class signature parity.
        assert self.path is not None
        task_ids: list[TaskIdType] = [
            LocalTaskId(path=path) for path in self._list_local_task_dirs(self.path)
        ]
        return [
            TaskConfig(
                path=task_id.path,
                source=self.path.expanduser().resolve().name,
            )
            for task_id in self._filter_task_ids(task_ids)
            if isinstance(task_id, LocalTaskId)
        ]

    # ---- Tasks-side: cache + typed task model ----
    async def cache_tasks(self, task_configs: list[TaskConfig]) -> dict[TaskIdType, Path]:
        """Materialize git task dirs on disk; return their actual on-disk paths.

        Batches git refs through ``TaskClient().download_tasks`` (idempotent:
        skips existing when overwrite=False) and returns ``{task_id:
        result.path}`` keyed off the download results, which honor
        ``download_dir``. Local refs are absent from the map; callers resolve
        them via their literal ``get_local_path()``. A fetch failure raises
        ``TaskConfigInvalidError`` so a missing task aborts rather than being
        silently dropped.
        """
        remote = [tc for tc in task_configs if tc.is_git_task()]
        if not remote:
            return {}
        output_dir = self.download_dir or TASK_CACHE_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            batch = await retrying_git_fetch(
                lambda: TaskClient().download_tasks(
                    task_ids=[tc.get_task_id() for tc in remote],
                    overwrite=self.overwrite,
                    output_dir=output_dir,
                    export=False,
                )
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            raise TaskConfigInvalidError(
                f"Failed to fetch task(s) via git. "
                f"Underlying error: {stderr.strip() or e}. "
                "Try -p <local-dir> to work offline."
            ) from e
        except (FileNotFoundError, ValueError, OSError) as e:
            raise TaskConfigInvalidError(
                f"Failed to materialize task(s): {type(e).__name__}: {e}. "
                "Try -p <local-dir> to work offline."
            ) from e
        # batch.results is positionally aligned to the task_ids we passed: Harbor
        # builds it as [resolved[task_id] for task_id in task_ids], so results[i]
        # is the download for remote[i]. strict=True guards the length; this order
        # contract guards the pairing. Re-verify if Harbor's download_tasks changes.
        return {
            tc.get_task_id(): result.path for tc, result in zip(remote, batch.results, strict=True)
        }

    async def get_tasks(
        self, task_configs: list[TaskConfig] | None = None
    ) -> list[tuple[TaskConfig, AwsBenchTask]]:
        """Build and validate every task, returning ``(config, AwsBenchTask)`` pairs.

        Single pass over one list of task configs, in descriptor order. When
        ``task_configs`` is supplied the result matches that exact list; when
        omitted, ``get_task_configs()`` resolves them. ``cache_tasks`` batches the
        git downloads first; each descriptor then materializes via
        ``AwsBenchTask.from_config`` (git tasks from the cached download path,
        local from their literal path), which runs ``_validate_tests`` +
        ``_validate_layout``. All failures are collected and raised once. The
        config is returned alongside its task as a pair because ``TaskConfig``
        is not hashable and cannot key a dict.
        """
        if task_configs is None:
            task_configs = await self.get_task_configs()
        # Batch the git downloads first; from_config then hits the warm cache
        # (download_tasks is idempotent).
        await self.cache_tasks(task_configs)
        pairs: list[tuple[TaskConfig, AwsBenchTask]] = []
        invalid: list[str] = []
        for tc in task_configs:
            try:
                pairs.append((tc, await AwsBenchTask.from_config(tc)))
            except (
                ValidationError,
                FileNotFoundError,
                OSError,
                tomllib.TOMLDecodeError,
                KeyError,
                ValueError,
            ) as exc:
                invalid.append(f"  {tc.get_local_path()}: {exc}")
        if invalid:
            raise TaskConfigInvalidError("Invalid task(s):\n" + "\n".join(invalid))
        return pairs

    # ---- Helpers ----
    def _load_registry(self) -> AwsBenchRegistry:
        """Load, validate, and memoize the registry.

        Memoized; validation runs on first load.
        """
        if self._registry_cache is None:
            if self.registry_path is not None:
                registry = AwsBenchRegistry.from_path(self.registry_path)
            else:
                registry = AwsBenchRegistry.from_url(self.registry_url or DEFAULT_REGISTRY_URL)
            RegistryValidator(registry).validate()
            self._registry_cache = registry
        return self._registry_cache
