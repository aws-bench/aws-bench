"""AwsBenchJob — the run-side Job.

Resolves tasks, scenarios, accounts, and exports from the single
``AwsBenchDatasetConfig`` (``config.dataset``) inside ``create``, then constructs
the job in one step (resolve-then-construct).
"""

from __future__ import annotations

import logging
import shutil
from collections import defaultdict

from harbor.environments.factory import EnvironmentFactory
from harbor.job import Job
from harbor.metrics.base import BaseMetric
from harbor.metrics.factory import MetricFactory
from harbor.metrics.mean import Mean
from harbor.models.job.config import JobConfig
from harbor.models.job.result import EvalsRewardsMap, JobStats
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult

from aws_bench.account_management.manager import AccountManager
from aws_bench.cli.job_config import AwsBenchJobConfig
from aws_bench.dataset.exceptions import ScenarioReferenceError
from aws_bench.dataset.task_config import AwsBenchTask
from aws_bench.logging.logger import build_console_handler, get_logger
from aws_bench.resource_management.export_collector import collect_account_exports
from aws_bench.task.queue import AwsBenchTrialQueue
from aws_bench.task.trial_config import AwsBenchTrialConfig
from aws_bench.utils.retry import retrying_git_fetch

logger = get_logger(__name__)

# Reserved placeholder exposing the scenario's allowed regions to task
# instructions as ``{{aws-bench-scenario-allowed-regions}}``. The ``aws-bench-``
# prefix avoids collision with author CloudFormation/SSM exports. Hyphen (not
# colon) so the key is a legal env-var name when a task declares it in
# ``[verifier.env]`` to resolve the token in ground_truth.json too.
ALLOWED_REGIONS_PLACEHOLDER = "aws-bench-scenario-allowed-regions"

# Harbor's run-path modules (job, trial, queue, verifier) all log on this single
# shared logger. We bridge it onto the aws-bench console handler so run lines share
# one format; it does not propagate to the aws_bench tree.
_HARBOR_LOGGER_NAME = "harbor.utils.logger"


class AwsBenchJob(Job):
    """Run-side Job: resolves AWS context in ``create`` and builds AwsBench trials.

    ``create`` resolves the AWS context (tasks, scenarios, accounts, exports);
    the trial queue is the aws-bench one so every trial is credentialed and admitted
    through its scenario's readers-writer gate before taking a global slot.
    """

    config: AwsBenchJobConfig

    def __init__(
        self,
        config: AwsBenchJobConfig,
        *,
        _task_pairs,
        _account_exports,
        _run_pairs,
        **kwargs,
    ) -> None:
        """Build the job and install the aws-bench trial queue with the Job's hooks."""
        # (TaskConfig, AwsBenchTask) pairs in task-config order; each trial
        # config is built from its own paired task.
        self._task_pairs = _task_pairs
        # Account-keyed export ledger, sliced per trial in _init_trial_configs.
        self._account_exports = _account_exports
        # Set before super().__init__: the base init runs _init_trial_configs
        # -> _build_trial_config, which reads _run_pairs to stamp each trial's
        # scenario descriptor and derive its allowed regions.
        self._run_pairs = _run_pairs
        self.run_scenarios = [scenario for _, scenario in _run_pairs.values()]
        super().__init__(config, **kwargs)
        # Replace the base queue with the aws-bench one, preserving the hooks the
        # base wired during super().__init__. Each trial config carries its own
        # scenario_id + concurrency_mode, so the queue gates per scenario.
        self._trial_queue = AwsBenchTrialQueue(
            n_concurrent=self.config.n_concurrent_trials,
            retry_config=self.config.retry,
            hooks=self._trial_queue._hooks,
        )

    def _init_logger(self) -> None:
        """Bridge Harbor's run logger onto the aws-bench RichHandler for the console.

        The per-job ``job.log`` *file* is owned by the caller (``aws-bench run``
        wraps create+run in nested ``file_logging`` spans on the aws_bench tree
        and Harbor's run tree), so this override does NOT open a file handler — it
        only neutralizes the base's own console/file handlers and routes Harbor's
        shared run logger through the aws-bench ``RichHandler`` so console lines
        share one format. Console level is DEBUG iff ``config.debug``.
        """
        harbor_logger = logging.getLogger(_HARBOR_LOGGER_NAME)

        # Do NOT call super()._init_logger(): it opens a truncating mode="w"
        # job.log (clobbering the caller's file) and a plain StreamHandler
        # console. The caller's file_logging span already captures the file;
        # we just bridge the console.
        self._log_file_handler = None
        self._console_handler = None

        level = logging.DEBUG if self.config.debug else logging.INFO
        self._aws_console_handler = build_console_handler(level=level)
        harbor_logger.addHandler(self._aws_console_handler)

    def _close_logger_handlers(self) -> None:
        """Detach the bridged console handler from Harbor's tree; close cleanly."""
        harbor_logger = logging.getLogger(_HARBOR_LOGGER_NAME)
        aws_console = getattr(self, "_aws_console_handler", None)
        if aws_console is not None:
            harbor_logger.removeHandler(aws_console)
            aws_console.close()
            self._aws_console_handler = None

    @classmethod
    async def create(cls, config: AwsBenchJobConfig) -> "AwsBenchJob":  # type: ignore[override]
        """Resolve tasks/scenarios/accounts/exports from config.dataset; return a ready Job."""
        assert config.dataset is not None  # guarded by validate_run() before create
        # job.log is owned by the caller: `aws-bench run` wraps create+run in
        # file_logging spans (aws_bench tree + Harbor's run tree), so the
        # resolution lines below (Fetching exports, account mapping) are captured.
        task_configs = await cls._resolve_task_configs(config)
        EnvironmentFactory.validate_resource_policies(config.environment)
        metrics = await cls._resolve_metrics(config, task_configs)
        task_download_results = await cls._cache_tasks(task_configs)

        # --- Resolve AWS context (tasks, scenarios, accounts, exports) ---
        task_pairs = await config.dataset.get_tasks(task_configs)
        pairs_by_name = {
            scenario.name: (descriptor, scenario)
            for descriptor, scenario in (await config.dataset.get_scenarios()).items()
        }
        # CLI override wins; else fall back to the dataset's registry instructions.
        config.extra_instruction_paths = (
            config.extra_instruction_paths or await config.dataset.resolve_instruction_paths()
        )

        referenced = {task.config.scenario.scenario_id for _, task in task_pairs}
        unknown = referenced - set(pairs_by_name)
        if unknown:
            raise ScenarioReferenceError(
                f"Tasks reference scenarios not present in the selected dataset: "
                f"{sorted(unknown)}. Available scenarios: {sorted(pairs_by_name)}."
            )
        # Sorted so the downstream account-export fetch order and the persisted
        # config are stable across runs (set iteration order is not).
        run_pairs = {name: pairs_by_name[name] for name in sorted(referenced)}
        run_scenarios = [scenario for _, scenario in run_pairs.values()]

        assert config.env_name is not None  # presence enforced in start()
        test_env = AccountManager().resolve_test_environment(
            config.env_name,
            {s.name: set(s.manifest.scenario.account_tags) for s in run_scenarios},
        )
        config.test_environment = test_env  # part of the resume identity

        targets: dict[str, set[str]] = {}
        for s in run_scenarios:
            # Every account the scenario maps to, so each tag's export slice is populated.
            for account_id in test_env.mapping_for(s.name).values():
                targets.setdefault(account_id, set()).update(s.manifest.scenario.regions)
        logger.info(
            "Fetching CloudFormation exports across %d account(s) / %d pair(s)...",
            len(targets),
            sum(len(r) for r in targets.values()),
        )
        # {account_id: {EXPORT: value}}, sliced per trial in _init_trial_configs.
        account_exports = collect_account_exports(targets=targets)

        return cls(
            config,
            _task_configs=task_configs,
            _metrics=metrics,
            _task_download_results=task_download_results,
            _task_pairs=task_pairs,
            _account_exports=account_exports,
            _run_pairs=run_pairs,
        )

    @staticmethod
    async def _resolve_task_configs(config: JobConfig) -> list[TaskConfig]:
        """Resolve task configs from the single AwsBench dataset (``config.dataset``)."""
        assert isinstance(config, AwsBenchJobConfig) and config.dataset is not None
        return await config.dataset.get_task_configs()

    @staticmethod
    async def _cache_tasks(task_configs):  # type: ignore[override]
        """Retry the base task fetch, the first (un-retried) git fetch in ``create``.

        Wraps the base ``download_tasks`` in ``retrying_git_fetch``, preserving
        its return type for the job lock. Not ``super()``: the base is a
        ``@staticmethod``.
        """
        return await retrying_git_fetch(lambda: Job._cache_tasks(task_configs))

    @staticmethod
    async def _resolve_metrics(
        config: JobConfig, task_configs: list[TaskConfig]
    ) -> dict[str, list[BaseMetric]]:
        """Resolve metrics into the ``{source: [BaseMetric], "adhoc": [...]}`` shape Job expects.

        Builds the adhoc bucket from the job-level metrics plus a per-source bucket
        combining the dataset's registry metrics with the job metrics, then floors
        each empty bucket to ``[Mean()]``. Driven by the single ``config.dataset``.
        """
        assert isinstance(config, AwsBenchJobConfig) and config.dataset is not None
        metrics: dict[str, list[BaseMetric]] = defaultdict(list)

        job_metrics = [MetricFactory.create_metric(m.type, **m.kwargs) for m in config.metrics]
        metrics["adhoc"].extend(job_metrics)

        resolved = await config.dataset.resolve_metric_configs()
        source = next((tc.source for tc in task_configs if tc.source), None)
        if source is not None:
            metrics[source].extend(
                MetricFactory.create_metric(m.type, **m.kwargs) for m in resolved
            )
            metrics[source].extend(job_metrics)

        for bucket in metrics.values():
            if not bucket:
                bucket.append(Mean())
        return metrics

    def _build_trial_config(
        self, *, task_config: TaskConfig, task: AwsBenchTask, agent_config: AgentConfig
    ) -> AwsBenchTrialConfig:
        """Build one trial config, deriving its account mapping and export slice.

        ``scenario_id`` and ``concurrency_mode`` are read straight off the paired
        ``task``; the ``scenario`` descriptor comes from ``_run_pairs`` under that
        scenario_id; ``account_mapping`` ({account_tag: account_id}) comes from
        ``test_environment`` via that scenario_id; ``exports`` is tag-keyed
        (``tag -> {export name -> value}``), each tag carrying its account's export
        slice. ``verifier`` is deep-copied and each tag's ``exports`` slice is a
        fresh dict per trial, so a trial's runtime mutation (the transient
        verifier-cred overlay, the placeholder seed) can never reach into another
        trial's config.

        The scenario's ``scenario.toml`` ``regions`` are injected into every tag's
        ``exports`` slice under ``ALLOWED_REGIONS_PLACEHOLDER`` for the
        allowed-regions placeholder.
        """
        assert self.config.test_environment is not None
        scenario_id = task.config.scenario.scenario_id
        descriptor, scenario = self._run_pairs[scenario_id]
        mapping = self.config.test_environment.mapping_for(scenario_id)
        if not mapping:
            # Empty mapping (unknown scenario) would otherwise build a credential-less trial.
            raise ScenarioReferenceError(
                f"No account mapping for scenario '{scenario_id}' in the resolved test environment."
            )
        exports = {tag: dict(self._account_exports.get(acct, {})) for tag, acct in mapping.items()}
        # regions is validated non-empty, so the joined value is never blank.
        regions = scenario.manifest.scenario.regions
        allowed_regions = ", ".join(regions)
        for tag_exports in exports.values():
            if ALLOWED_REGIONS_PLACEHOLDER in tag_exports:
                # An author export/SSM param used the reserved key; fail loud at build
                # time rather than crash mid-trial or silently mis-resolve.
                raise ScenarioReferenceError(
                    f"Export key '{ALLOWED_REGIONS_PLACEHOLDER}' is reserved by aws-bench. "
                    f"Rename the conflicting export/SSM param."
                )
            tag_exports[ALLOWED_REGIONS_PLACEHOLDER] = allowed_regions
        return AwsBenchTrialConfig(
            task=task_config,
            trials_dir=self.job_dir,
            agent=agent_config,
            timeout_multiplier=self.config.timeout_multiplier,
            agent_timeout_multiplier=self.config.agent_timeout_multiplier,
            verifier_timeout_multiplier=self.config.verifier_timeout_multiplier,
            agent_setup_timeout_multiplier=self.config.agent_setup_timeout_multiplier,
            environment_build_timeout_multiplier=self.config.environment_build_timeout_multiplier,
            environment=self.config.environment,
            verifier=self.config.verifier.model_copy(deep=True),
            artifacts=self.config.artifacts,
            extra_instruction_paths=self.config.extra_instruction_paths,
            job_id=self._id,
            scenario=descriptor,
            scenario_id=scenario_id,
            concurrency_mode=task.config.concurrency.mode,
            account_mapping=mapping,
            regions=regions,
            exports=exports,
            verify_env=self.config.verify,
        )

    def _init_trial_configs(self) -> None:
        """Build an AwsBenchTrialConfig per (attempt x task x agent)."""
        self._trial_configs = [
            self._build_trial_config(task_config=tc, task=task, agent_config=agent_config)
            for _ in range(self.config.n_attempts)
            for tc, task in self._task_pairs
            for agent_config in self.config.agents
        ]

    def _maybe_init_existing_job(self) -> None:
        """Resume-init: parse the existing job and trial configs as their aws-bench subtypes.

        Parsing as the subclasses round-trips the aws-bench identity fields
        (``test_environment`` on the job; ``scenario_id`` / ``account_mapping`` on each
        trial) so an unchanged config compares equal and resumes, while a changed
        environment refuses cleanly. ``exports`` is ``exclude=True`` on the trial
        config, so it is absent from both the persisted JSON and the equality dump:
        an export-value change alone does not force a re-run.
        """
        self._existing_trial_configs: list[TrialConfig] = []
        self._existing_trial_results: list[TrialResult] = []
        self._previous_trial_results: dict[str, TrialResult] = {}
        self._existing_rewards: EvalsRewardsMap = defaultdict(dict)
        self._existing_stats = JobStats()

        if not self._job_config_path.exists():
            return

        existing_config = AwsBenchJobConfig.model_validate_json(self._job_config_path.read_text())
        if existing_config != self.config:
            raise FileExistsError(
                f"Job directory {self.job_dir} already exists and cannot be "
                "resumed with a different config."
            )

        for trial_dir in self.job_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            trial_paths = TrialPaths(trial_dir)
            if not trial_paths.result_path.exists():
                shutil.rmtree(trial_paths.trial_dir)
            else:
                self._existing_trial_configs.append(
                    AwsBenchTrialConfig.model_validate_json(trial_paths.config_path.read_text())
                )
                self._existing_trial_results.append(
                    TrialResult.model_validate_json(trial_paths.result_path.read_text())
                )

        for trial_result in self._existing_trial_results:
            agent_name = trial_result.agent_info.name
            model_name = (
                trial_result.agent_info.model_info.name
                if trial_result.agent_info.model_info
                else None
            )
            dataset_name = trial_result.source or "adhoc"
            evals_key = JobStats.format_agent_evals_key(agent_name, model_name, dataset_name)
            self._existing_rewards[evals_key][trial_result.trial_name] = (
                trial_result.verifier_result.rewards
                if trial_result.verifier_result is not None
                else None
            )
            self._previous_trial_results[trial_result.trial_name] = trial_result

        self._existing_stats = JobStats.from_trial_results(self._existing_trial_results)
