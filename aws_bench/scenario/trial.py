"""Single scenario trial: build → execute one phase → stop.

The trial doesn't know which phase (deploy / verify / cleanup) it runs —
that's set by the caller. The trial wraps the container layer with
lifecycle event emission, result persistence, and credential injection.

Callers register an async callback per lifecycle event via
:meth:`add_hook`. Multiple observers can register independently for
the same event.
"""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError, WaiterError
from harbor.models.trial.result import ExceptionInfo, TimingInfo

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.account_management.manager import AccountManager
from aws_bench.exceptions import AccountContaminatedError, OperationCancelled
from aws_bench.logging.logger import file_logging, get_logger
from aws_bench.resource_management.constants import RESOURCE_MANAGEMENT_SESSION
from aws_bench.resource_management.export_collector import collect_account_exports
from aws_bench.resource_management.manager import ResourceManager
from aws_bench.resource_management.reset.models import ResetResult
from aws_bench.resource_management.snapshot.manager import SnapshotManager
from aws_bench.resource_management.snapshot.models import (
    SnapshotContext,
    SnapshotStage,
)
from aws_bench.scenario.config import EnvironmentConfig
from aws_bench.scenario.container import (
    ScenarioContainer,
    sanitize_container_name,
    sanitize_image_tag,
)
from aws_bench.scenario.events import ScenarioEvent, ScenarioPhase
from aws_bench.scenario.exceptions import (
    CleanupFailedError,
    NonZeroExitCodeError,
    PhaseTimeoutError,
    ResetFailedError,
    SetupValidationError,
    SnapshotFailedError,
    VerifyFailedError,
)
from aws_bench.scenario.job_config import ScenarioTrialConfig
from aws_bench.scenario.results import ScenarioHookEvent, ScenarioTrialResult
from aws_bench.scenario.scenario import Scenario
from aws_bench.scenario.trial_paths import ScenarioJobPaths, ScenarioTrialPaths
from aws_bench.utils.concurrent import build_client
from aws_bench.utils.credentials_provider import (
    CredentialProvider,
    build_session_name,
)
from aws_bench.utils.placeholders import substitute_placeholders

logger = get_logger(__name__)

HookCallback = Callable[[ScenarioHookEvent], Awaitable[None]]

MANAGEMENT_ROLE_ENV_VAR = "MANAGEMENT_ROLE"

# Post-reset redeploy retry policy. The redeploy recreates the stack(s) reset
# deleted; a transient AWS failure (notably the S3 Tables bucket-name deletion
# cooldown returning a 409 right after the stack delete removed the bucket)
# clears on its own within a few minutes, so retry with growing backoff before
# treating the reset as incomplete.
_REDEPLOY_MAX_ATTEMPTS = 3
_REDEPLOY_BACKOFF_SEC = 120.0

# Phases whose env is resolved against the scenario account's CloudFormation
# exports + SSM /exports. DEPLOY is excluded: it creates the stacks that produce
# the exports, so they do not exist yet when deploy.sh runs.
_EXPORT_RESOLVED_PHASES = frozenset(
    {ScenarioPhase.VERIFY, ScenarioPhase.RESET, ScenarioPhase.CLEANUP}
)


class ScenarioTrial:
    """One scenario phase invocation against one account.

    Lifecycle:
      START -> ENVIRONMENT_START (build + container start)
      -> PHASE_START (run /<phase>/<phase>.sh) -> END
      with CANCEL on ``asyncio.CancelledError`` or ``OperationCancelled``.

    Credentials are resolved at script-invocation time via the injected
    ``CredentialProvider``; they are never stored on the config or result.
    """

    def __init__(
        self,
        config: ScenarioTrialConfig,
        cred_provider: CredentialProvider,
        *,
        scenario: Scenario,
        container: ScenarioContainer | None = None,
    ) -> None:
        """Initialize the trial.

        Args:
            config: Trial configuration; the caller has already merged
                operator overrides into ``config.environment``. Output paths are
                derived from ``config.output_dir`` and ``config.trial_name``.
            cred_provider: Source of management credentials. Asked at
                script-execution time, not at construction.
            scenario: Materialized scenario carrying the parsed manifest. The
                trial reads phase env / timeouts / per-phase config from
                ``scenario.manifest``; on-disk paths from ``scenario.scenario_dir``.
            container: Optional pre-built ``ScenarioContainer``. One is
                constructed by default; injection is for tests.
        """
        self._config = config
        self._scenario = scenario
        self._paths = ScenarioJobPaths(config.output_dir).trial_paths(config.trial_name)
        self._cred_provider = cred_provider
        self._account_manager = AccountManager()
        self._hooks: dict[ScenarioEvent, list[HookCallback]] = {e: [] for e in ScenarioEvent}

        self._paths.mkdir()
        merged_env = self._merge_environment_config()
        self._container = container or ScenarioContainer(
            scenario.paths,
            merged_env,
            image_tag=sanitize_image_tag(f"awsbench-{scenario.name}"),
            container_name=sanitize_container_name(f"awsbench-{config.trial_name}"),
            host_logs_dir=self._paths.trial_dir,
            cred_provider=cred_provider,
            account_mapping=config.account_mapping,
        )
        self._merged_env = merged_env
        self._result = ScenarioTrialResult(
            scenario_name=scenario.name,
            trial_name=config.trial_name,
            account_mapping=config.account_mapping,
            regions=scenario.manifest.scenario.regions,
        )

    @classmethod
    async def create(
        cls,
        config: ScenarioTrialConfig,
        cred_provider: CredentialProvider,
    ) -> ScenarioTrial:
        """Async factory: materialize the scenario from ``config.scenario``."""
        scenario = await Scenario.from_config(config.scenario)
        return cls(config, cred_provider, scenario=scenario)

    def _merge_environment_config(self) -> EnvironmentConfig:
        """Apply operator overrides to a copy of the author's env config.

        Returns a fresh, mutated copy so the persisted on-disk config
        keeps both sides distinct for audit.
        """
        author = self._scenario.manifest.environment.model_copy(deep=True)
        self._config.environment.apply_to(author)
        return author

    def add_hook(self, event: ScenarioEvent, hook: HookCallback) -> ScenarioTrial:
        """Register an async callback for one lifecycle event."""
        self._hooks[event].append(hook)
        return self

    @property
    def config(self) -> ScenarioTrialConfig:
        """The trial's configuration."""
        return self._config

    @property
    def paths(self) -> ScenarioTrialPaths:
        """The trial's output paths."""
        return self._paths

    @property
    def result(self) -> ScenarioTrialResult:
        """The trial's result so far."""
        return self._result

    async def run(self, phase: ScenarioPhase) -> ScenarioTrialResult:
        """Run one phase end-to-end: optional container script, then resource management.

        Returns the trial result on completion or failure; failures are recorded
        as ``exception_info``. On cancellation (``asyncio.CancelledError`` or
        ``OperationCancelled``) the error is recorded, cleanup runs, and it is
        re-raised.
        """
        self._result.phase = phase
        self._result.started_at = datetime.now(timezone.utc)
        self._persist_config()
        with file_logging(self._paths.log_path, scope=self._config.trial_name):
            logger.info(
                "%s trial %s",
                phase.gerund,
                self._config.trial_name,
            )
            try:
                # START is emitted inside the try so a raising START hook
                # still routes through except/finally and cleans up the
                # container rather than leaking it.
                await self._invoke_hooks(ScenarioEvent.START, phase=phase)

                # Sweep post-setup residuals BEFORE cleanup.sh so it and the stack
                # deletion run on a cleaner account. Best-effort: a failure here must
                # not skip cleanup.sh or the framework cleanup that follow, so catch
                # and log it; a cooperative shutdown still unwinds.
                if phase == ScenarioPhase.CLEANUP:
                    try:
                        await self._run_pre_cleanup_residual_sweep()
                    except (asyncio.CancelledError, OperationCancelled):
                        raise
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Pre-cleanup residual sweep failed (best-effort): %s", exc)

                # Every phase runs its <phase>/<phase>.sh if one exists on disk.
                # deploy.sh is required (validated at construction), so DEPLOY
                # always has one; verify/cleanup/reset are optional.
                script_error: Exception | None = None
                has_script = self._scenario.has_phase_script(phase)

                if not has_script:
                    logger.debug(
                        f"No {phase.value}.sh for {self._scenario.name}, skipping custom script"
                    )
                else:
                    try:
                        await self._run_phase_in_container(phase)
                    except Exception as exc:
                        script_error = exc

                # Recovery phases run regardless; the deploy snapshot is gated
                # on script success inside _run_resource_management.
                await self._run_resource_management(phase, script_failed=script_error is not None)

                # Re-raise script error after resource management completes
                if script_error:
                    raise script_error

            except (asyncio.CancelledError, OperationCancelled) as exc:
                logger.debug(
                    "Trial %s/%s %s cancelled",
                    self._scenario.name,
                    self._config.trial_name,
                    phase.value,
                )
                self._record_exception(exc)
                await self._invoke_hooks(ScenarioEvent.CANCEL, phase=phase)
                raise
            except Exception as exc:  # noqa: BLE001
                self._record_exception(exc)
                logger.error(
                    "Trial %s/%s %s failed: %s",
                    self._scenario.name,
                    self._config.trial_name,
                    phase,
                    exc,
                )
            finally:
                # Always attempt cleanup even if container didn't start - there might be
                # a leftover from a previous run, and _cleanup_container handles errors gracefully
                await self._cleanup_container()
                self._result.finished_at = datetime.now(timezone.utc)
                self._persist_result()
                await self._invoke_hooks(ScenarioEvent.END, phase=phase)
            return self._result

    def _persist_config(self) -> None:
        """Write the trial config to disk at the start of the run.

        The config is known up front, so persisting it before execution
        leaves a record even if the trial crashes mid-run.
        """
        self._paths.trial_dir.mkdir(parents=True, exist_ok=True)
        self._paths.config_path.write_text(self._config.model_dump_json(indent=2))

    def _persist_result(self) -> None:
        """Write the trial result to disk before the ``END`` event is emitted.

        Persisting ahead of the event keeps the on-disk record complete
        regardless of what any observer does.

        Best-effort: this runs in ``run``'s finally, which on a cancel executes
        while the cancellation is propagating. Any write/serialization failure is
        logged and swallowed so it cannot replace the in-flight cancel; the
        ``BaseException`` keeps unwinding.
        """
        try:
            self._paths.trial_dir.mkdir(parents=True, exist_ok=True)
            self._paths.result_path.write_text(self._result.model_dump_json(indent=2))
        except Exception as exc:  # noqa: BLE001 — best-effort artifact; must not mask a cancel
            logger.warning(
                "Could not persist result for %s/%s (in-memory result is unaffected): %s",
                self._scenario.name,
                self._config.trial_name,
                exc,
            )

    def _record_exception(self, exc: BaseException) -> None:
        """Record the first failure as ``exception_info`` and persist its traceback.

        The first exception to land wins: if ``exception_info`` is already
        set, a later failure (e.g. one raised during cleanup) is logged but
        does not overwrite the recorded root cause or the traceback file.
        """
        if self._result.exception_info is not None:
            logger.debug(
                "Trial %s/%s already has a recorded exception; not overwriting with %r",
                self._scenario.name,
                self._config.trial_name,
                exc,
            )
            return
        self._result.exception_info = ExceptionInfo.from_exception(exc)
        # Best-effort (see _persist_result): a write failure must not replace a
        # propagating cancel. exception_info above is already recorded.
        try:
            self._paths.trial_dir.mkdir(parents=True, exist_ok=True)
            self._paths.exception_path.write_text("".join(traceback.format_exception(exc)))
        except Exception as write_exc:  # noqa: BLE001 — best-effort artifact; must not mask a cancel
            logger.warning(
                "Could not persist exception traceback for %s/%s: %s",
                self._scenario.name,
                self._config.trial_name,
                write_exc,
            )

    async def _run_phase_in_container(
        self, phase: ScenarioPhase, *, check_contamination: bool = True
    ) -> None:
        """Build/start the container (if needed), apply guardrails, run the phase script.

        Idempotent on start: the post-reset deploy redeploy reuses this and may
        already have a running container (from a reset.sh) or none (a reset with no
        reset.sh) — start only when not already running, since ``container.start``
        refuses a double start.

        ``check_contamination`` gates the DEPLOY contamination refusal. The reset's
        own redeploy passes ``False``: it recovers a contaminated account, so it must
        not block on the flag it is about to clear.
        """
        await self._invoke_hooks(ScenarioEvent.ENVIRONMENT_START, phase=phase)
        if not self._container.is_started:
            await self._build_and_start()

        # Lock the account to its declared regions BEFORE deploy.sh can create
        # anything: an out-of-region action is denied at the source rather than
        # discovered (and orphaned) by the region-scoped snapshot/verify/cleanup
        # afterward. Fail-closed — if the guardrail can't be applied, deploy never runs.
        if phase == ScenarioPhase.DEPLOY:
            if check_contamination:
                await self._raise_if_contaminated()
            await self._validate_init_snapshot_exists()
            await self._apply_region_restriction_scp()
            await self._clean_stale_changesets()
            await self._delete_terminal_stacks()

        await self._invoke_hooks(ScenarioEvent.PHASE_START, phase=phase)
        await self._execute(phase)

    async def _build_and_start(self) -> None:
        timing = TimingInfo(started_at=datetime.now(timezone.utc))
        try:
            await self._container.build(
                force=self._config.environment.force_build,
                timeout_sec=self._merged_env.build_timeout_sec * self._config.timeout_multiplier,
            )
            # start() mints the per-account credential files and writes
            # ~/.aws/config (credential_process profiles) itself; scripts select an
            # account with AWS_PROFILE=<tag>.
            await self._container.start()
        finally:
            timing.finished_at = datetime.now(timezone.utc)
            self._result.environment_setup = timing

    async def _execute(self, phase: ScenarioPhase) -> None:
        timing = TimingInfo(started_at=datetime.now(timezone.utc))
        try:
            env = self._build_phase_env()
            phase_cfg = getattr(self._scenario.manifest, phase)
            resolved_phase_env = await self._resolve_phase_env(phase, phase_cfg.env)
            phase_env = {**env, **resolved_phase_env}
            timeout = phase_cfg.timeout_sec * self._config.timeout_multiplier
            # The script's stdout goes to a log file, so the console is silent
            # while it runs; log start/finish so a long phase isn't mistaken for a hang.
            logger.debug("Running %s script in container (timeout %.0fs)", phase.value, timeout)
            try:
                exec_result = await self._container.run_phase(
                    phase, env=phase_env, timeout_sec=timeout
                )
            except TimeoutError as exc:
                # Name the timeout so the recorded failure says which phase and budget.
                raise PhaseTimeoutError(phase=phase, timeout_sec=timeout) from exc
            self._result.exit_code = exec_result.exit_code
            logger.debug("%s script finished (exit=%d)", phase.value, exec_result.exit_code)
            if exec_result.exit_code != 0:
                raise NonZeroExitCodeError(
                    phase=phase,
                    exit_code=exec_result.exit_code,
                    stdout=exec_result.stdout,
                )
        finally:
            timing.finished_at = datetime.now(timezone.utc)
            self._result.execute = timing

    def _build_phase_env(self) -> dict[str, str]:
        """Build the phase's account-id env vars.

        The scenario container gets:
          - One ``<TAG>=<account_id>`` entry per scenario account-tag.
          - ``MANAGEMENT_ROLE`` set to the org-access role constant.

        AWS credentials are NOT injected as env vars: the container reads them via
        ``~/.aws/config`` credential_process profiles backed by files the host
        refresher keeps fresh (see ``ScenarioContainer``). Scripts select an
        account by exporting ``AWS_PROFILE=<TAG>``; multi-account scenarios may
        switch profiles for different commands.
        """
        env = {MANAGEMENT_ROLE_ENV_VAR: ORG_ACCESS_ROLE}
        for tag, account_id in self._config.account_mapping.items():
            env[tag] = account_id
        return env

    async def _resolve_phase_env(
        self, phase: ScenarioPhase, raw_env: dict[str, str]
    ) -> dict[str, str]:
        """Resolve ``{{export}}`` placeholders in a phase's declared env.

        For post-deploy phases (verify/reset/cleanup), collect the scenario
        account's CloudFormation exports + SSM ``/exports`` and substitute them
        into ``raw_env``. DEPLOY is passed through unresolved: it creates the
        stacks that produce the exports. An env value referencing an export that
        was not collected raises ``PlaceholderMissingError``.
        """
        if phase not in _EXPORT_RESOLVED_PHASES or not raw_env:
            return dict(raw_env)
        tag_map = await self._collect_scenario_exports()
        return {k: substitute_placeholders(v, tag_map) for k, v in raw_env.items()}

    async def _collect_scenario_exports(self) -> dict[str, dict[str, str]]:
        """Collect the scenario's CFN exports + SSM ``/exports`` params, keyed by tag.

        Returns ``{account_tag: {export name: value}}`` for every tag in the trial's
        account mapping. The collector is blocking (boto3 across regions), so it runs
        in a worker thread to keep the event loop free for concurrent trials.
        """
        if not self._config.account_mapping:
            return {}
        regions = self._scenario.manifest.scenario.regions
        targets = dict.fromkeys(self._config.account_mapping.values(), regions)
        per_account = await asyncio.to_thread(collect_account_exports, targets)
        return {
            tag: dict(per_account.get(account_id, {}))
            for tag, account_id in self._config.account_mapping.items()
        }

    async def _cleanup_container(self) -> None:
        # Stop under asyncio.shield so a single cancellation lets it finish. Runs
        # in run()'s finally, which must still persist the result and emit END, so
        # neither failure mode re-raises here — both are recorded and swallowed.
        try:
            await asyncio.shield(self._container.stop(delete=self._config.environment.delete))
        except (asyncio.CancelledError, OperationCancelled) as exc:
            logger.warning(
                "Container stop interrupted by cancellation for %s/%s; "
                "the container may be left running",
                self._scenario.name,
                self._config.trial_name,
            )
            self._record_exception(exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to stop container for %s/%s: %s",
                self._scenario.name,
                self._config.trial_name,
                exc,
            )
            # Record the cleanup failure unless an earlier lifecycle error
            # already claimed exception_info (the guard keeps that root cause).
            self._record_exception(exc)

    async def _invoke_hooks(self, event: ScenarioEvent, *, phase: ScenarioPhase) -> None:
        """Invoke every callback registered for ``event``.

        Hook errors are logged and swallowed so a faulty observer cannot break
        the trial. The trial config and result are written to disk before the
        ``END`` event is emitted, ensuring on-disk records remain complete
        regardless of hook failures.
        """
        callbacks = self._hooks[event]
        if not callbacks:
            return
        payload = ScenarioHookEvent(
            event=event,
            scenario_name=self._scenario.name,
            trial_name=self._config.trial_name,
            phase=phase,
            config=self._config,
            result=self._result if event is ScenarioEvent.END else None,
        )
        for hook in callbacks:
            try:
                await hook(payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "on_event(%s) for %s/%s raised: %s",
                    event.value,
                    self._scenario.name,
                    self._config.trial_name,
                    exc,
                )

    async def _raise_if_contaminated(self) -> None:
        """Raise AccountContaminatedError if any mapped account carries the flag.

        ``get_contaminated_accounts`` is a blocking per-account Organizations read;
        run it off the loop so concurrent sibling trials aren't stalled.
        """
        account_ids = list(self._config.account_mapping.values())
        contaminated = await asyncio.to_thread(
            self._account_manager.get_contaminated_accounts, account_ids
        )
        if contaminated:
            raise AccountContaminatedError(
                account_ids=contaminated,
                scenario_id=self._scenario.name,
            )

    async def _run_resource_management(
        self, phase: ScenarioPhase, *, script_failed: bool = False
    ) -> None:
        """Run the phase's resource management, attaching results to the trial.

        The deploy snapshot is skipped when the deploy script failed, so a
        half-deployed environment never becomes a baseline. Recovery phases
        (verify/reset/cleanup) run regardless of script outcome.
        """
        if phase == ScenarioPhase.DEPLOY:
            if script_failed:
                logger.debug("Deploy failed; skipping baseline snapshot")
                return
            await self._run_snapshot()
        elif phase == ScenarioPhase.VERIFY:
            await self._run_verify()
        elif phase == ScenarioPhase.RESET:
            await self._run_reset()
        elif phase == ScenarioPhase.CLEANUP:
            await self._run_cleanup(script_failed=script_failed)

    async def _run_snapshot(self, forbidden_identifiers: set[str] | None = None) -> None:
        """Capture the post-setup baseline snapshot for this scenario's accounts.

        Assumes a successful deploy; the caller gates on deploy outcome.
        ``forbidden_identifiers`` are orphan ids the snapshot layer refuses to
        absorb into a baseline (defense-in-depth for the recapture path).
        """
        ctx = SnapshotContext(
            scenario_id=self._scenario.name,
            scenario_hash=self._scenario.hash,
            regions=self._scenario.manifest.scenario.regions,
            stage=SnapshotStage.POST_SETUP,
            account_ids=list(self._config.account_mapping.values()),
            forbidden_identifiers=forbidden_identifiers or set(),
        )

        results_by_scenario = await ResourceManager.snapshot_scenarios([ctx])
        results = results_by_scenario.get(self._scenario.name, [])
        if self._result.resource_results is None:
            self._result.resource_results = {}
        self._result.resource_results["snapshot"] = results

        # trial.success ignores resource_results, so a failing result must raise or the
        # trial passes with a silently missing baseline (verify/reset can't diff without it).
        failed = [r for r in results if not r.success]
        if failed:
            raise SnapshotFailedError(
                failures=[
                    f"{r.account_id}: {r.error_message or 'snapshot failed'}" for r in failed
                ],
            )

    async def _validate_init_snapshot_exists(self) -> None:
        """Validate that the PRE_SETUP (init) snapshot exists for all accounts.

        The init snapshot is the pristine account baseline captured by ``env init``.
        Setup depends on it: without it, ``env cleanup`` cannot distinguish account
        defaults from deployed infrastructure, leading to incorrect cleanup behavior.
        """
        mgr = SnapshotManager()
        missing: list[str] = []
        for account_id in self._config.account_mapping.values():
            if not mgr.snapshot_exists(self._scenario.name, account_id, SnapshotStage.PRE_SETUP):
                missing.append(account_id)

        if missing:
            raise SetupValidationError(
                f"Init snapshot (PRE_SETUP) missing for account(s): {', '.join(missing)}. "
                f"Run 'aws-bench env init' first to capture the baseline snapshot."
            )

    async def _apply_region_restriction_scp(self) -> None:
        """Lock this scenario's accounts to its declared regions before deploy.

        Applied before deploy.sh runs so an out-of-region action is denied at
        the source rather than discovered (and orphaned) by the region-scoped
        snapshot/verify/cleanup afterward. Scoped to the accounts this trial
        uses, so a sibling scenario's failure never blocks a healthy scenario's
        restriction. The underlying Organizations calls are blocking and
        idempotent (reuse the policy by name, update content when regions
        change, skip already-attached accounts), so they run in a worker thread.
        """
        regions = self._scenario.manifest.scenario.regions
        account_ids = list(self._config.account_mapping.values())
        if not account_ids or not regions:
            logger.debug(
                "No accounts or regions for %s; skipping region-restriction SCP",
                self._scenario.name,
            )
            return
        await asyncio.to_thread(
            AccountManager().ensure_region_restriction_scp,
            self._scenario.name,
            regions,
            account_ids,
        )

    async def _clean_stale_changesets(self) -> None:
        """Delete stale CDK changesets and REVIEW_IN_PROGRESS stacks before deploy.

        A prior interrupted deploy can leave a ``cdk-deploy-change-set`` changeset
        that collides on ClientToken when CDK tries to create a new one. Stacks
        stuck in REVIEW_IN_PROGRESS (created by a changeset that was never
        executed) must be deleted entirely since they cannot accept new deploys.

        Best-effort: failures are logged and swallowed so a transient API error
        does not block the deploy from being attempted.
        """
        account_ids = list(self._config.account_mapping.values())
        regions = self._scenario.manifest.scenario.regions
        if not account_ids or not regions:
            return
        try:
            await asyncio.to_thread(self._delete_stale_changesets_sync, account_ids, regions)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stale changeset cleanup failed: %s", exc)

    def _delete_stale_changesets_sync(self, account_ids: list[str], regions: list[str]) -> None:
        """Synchronous implementation for _clean_stale_changesets."""
        for account_id in account_ids:
            try:
                session = self._cred_provider.get_session_for_account(
                    account_id,
                    ORG_ACCESS_ROLE,
                    build_session_name(RESOURCE_MANAGEMENT_SESSION, "changeset-cleanup"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to get session for account %s: %s", account_id, exc)
                continue
            for region in regions:
                try:
                    self._clean_region_changesets(session, region)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Stale changeset cleanup failed for account %s region %s: %s",
                        account_id,
                        region,
                        exc,
                    )

    def _clean_region_changesets(self, session: boto3.Session, region: str) -> None:
        """Delete cdk-deploy-change-set from all active stacks in a region."""
        cfn = build_client(session, "cloudformation", region_name=region)
        paginator = cfn.get_paginator("list_stacks")
        for page in paginator.paginate(
            StackStatusFilter=[
                "CREATE_COMPLETE",
                "UPDATE_COMPLETE",
                "UPDATE_ROLLBACK_COMPLETE",
                "REVIEW_IN_PROGRESS",
            ]
        ):
            for stack in page.get("StackSummaries", []):
                stack_name = stack["StackName"]
                status = stack["StackStatus"]

                if status == "REVIEW_IN_PROGRESS":
                    logger.info(
                        "Deleting REVIEW_IN_PROGRESS stack %s in %s",
                        stack_name,
                        region,
                    )
                    try:
                        cfn.delete_stack(StackName=stack_name)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Failed to delete stack %s: %s", stack_name, exc)
                    continue

                # For healthy stacks, delete stale cdk-deploy-change-set if present
                try:
                    cfn.delete_change_set(
                        ChangeSetName="cdk-deploy-change-set",
                        StackName=stack_name,
                    )
                    logger.debug(
                        "Deleted stale cdk-deploy-change-set from %s in %s",
                        stack_name,
                        region,
                    )
                except ClientError as exc:
                    # ChangeSetNotFound means nothing to clean — not an error
                    code = exc.response.get("Error", {}).get("Code", "")
                    if code != "ChangeSetNotFound":
                        logger.debug("Could not delete changeset from %s: %s", stack_name, exc)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not delete changeset from %s: %s", stack_name, exc)

    # -- Terminal stack cleanup (ROLLBACK_COMPLETE / DELETE_FAILED) ---------

    _TERMINAL_STACK_STATES = ("ROLLBACK_COMPLETE",)

    async def _delete_terminal_stacks(self) -> None:
        """Delete stacks in terminal states before deploy to free leaked resources.

        Stacks stuck in ROLLBACK_COMPLETE from prior failed deploys hold resources
        (VPCs, subnets, ENIs) that count against quotas. Stacks in DELETE_FAILED
        retain problematic resources so the stack record can be removed, unblocking
        redeployment.

        Best-effort: failures are logged and swallowed so a transient API error
        does not block the deploy from being attempted.
        """
        account_ids = list(self._config.account_mapping.values())
        regions = self._scenario.manifest.scenario.regions
        if not account_ids or not regions:
            return
        try:
            await asyncio.to_thread(self._delete_terminal_stacks_sync, account_ids, regions)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Terminal stack cleanup failed: %s", exc)

    def _delete_terminal_stacks_sync(self, account_ids: list[str], regions: list[str]) -> None:
        """Synchronous implementation for _delete_terminal_stacks."""
        scenario_prefix = self._scenario.name
        for account_id in account_ids:
            try:
                session = self._cred_provider.get_session_for_account(
                    account_id,
                    ORG_ACCESS_ROLE,
                    build_session_name(RESOURCE_MANAGEMENT_SESSION, "terminal-cleanup"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to get session for account %s: %s", account_id, exc)
                continue
            for region in regions:
                try:
                    self._clean_terminal_stacks_in_region(session, region, scenario_prefix)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Terminal stack cleanup failed for account %s region %s: %s",
                        account_id,
                        region,
                        exc,
                    )

    def _clean_terminal_stacks_in_region(
        self, session: boto3.Session, region: str, scenario_prefix: str
    ) -> None:
        """Delete terminal stacks matching the scenario prefix in one region.

        Issues all delete requests first, then does a single bounded wait pass
        so deletions proceed in parallel on the CloudFormation side.
        """
        cfn = build_client(session, "cloudformation", region_name=region)
        paginator = cfn.get_paginator("list_stacks")
        deleted_stacks: list[str] = []

        # Phase 1: Issue all delete requests
        for page in paginator.paginate(StackStatusFilter=list(self._TERMINAL_STACK_STATES)):
            for stack in page.get("StackSummaries", []):
                name = stack.get("StackName", "")
                if not name.startswith(scenario_prefix):
                    continue
                status = stack["StackStatus"]
                logger.info("Deleting terminal stack %s (%s) in %s", name, status, region)
                try:
                    cfn.delete_stack(StackName=name)
                    deleted_stacks.append(name)
                except ClientError as exc:
                    logger.warning("Failed to initiate delete for stack %s: %s", name, exc)

        # Phase 2: Single bounded wait for all deletions (parallel on CFN side)
        if deleted_stacks:
            waiter = cfn.get_waiter("stack_delete_complete")
            for name in deleted_stacks:
                try:
                    waiter.wait(
                        StackName=name,
                        WaiterConfig={"Delay": 10, "MaxAttempts": 18},
                    )
                    logger.info("Deleted terminal stack %s", name)
                except (ClientError, WaiterError) as exc:
                    logger.warning("Terminal stack %s did not finish deleting: %s", name, exc)

    async def _run_verify(self) -> None:
        """Verify this scenario's accounts match baseline state."""
        # Verify runs even if custom script failed - it's a recovery/diagnostic operation
        results = await ResourceManager.verify_scenario(
            scenario_name=self._scenario.name,
            scenario_dir=self._scenario.scenario_dir,
            account_mapping=self._config.account_mapping,
            region=self._config.verify_region,
        )
        if self._result.resource_results is None:
            self._result.resource_results = {}
        self._result.resource_results["verify"] = results

        # Raise on a failing result (incl. missing baseline): trial.success ignores
        # resource_results, so without this a failed verify counts as a passed trial.
        failed = [r for r in results if not r.success]
        if failed:
            raise VerifyFailedError(
                failures=[
                    f"{r.account_id}: {r.error_message or 'verification failed'}" for r in failed
                ]
            )

    async def _run_reset(self) -> None:
        """Reset this scenario's accounts to baseline state.

        If any stacks were deleted during reset (couldn't be reverted in place),
        automatically redeploy by running the DEPLOY phase and recapturing baseline.
        """
        # Reset runs even if custom script failed - it's a recovery operation
        results = await ResourceManager.reset_scenarios(
            scenario_name=self._scenario.name,
            scenario_dir=self._scenario.scenario_dir,
            account_mapping=self._config.account_mapping,
            max_concurrent=self._config.reset_max_concurrent,
            output_dir=self._paths.phase_dir(ScenarioPhase.RESET),
        )
        if self._result.resource_results is None:
            self._result.resource_results = {}
        self._result.resource_results["reset"] = results

        # Check if any accounts need redeploy (stacks were deleted)
        needs_redeploy = any(r.needs_redeploy for r in results if hasattr(r, "needs_redeploy"))

        if needs_redeploy:
            logger.info(
                f"Stacks were deleted for {self._scenario.name}, redeploying via DEPLOY phase..."
            )
            try:
                await self._redeploy_with_retry()

                # Recapture the baseline only when every account is clean — otherwise a
                # still-present orphan would be absorbed into the new baseline and hide
                # forever. The trial fails below on the unclean account, so the run is
                # discarded and the preserved baseline never goes stale.
                if self._baseline_recapture_allowed(results):
                    await self._run_snapshot(self._orphan_identifiers(results))
                else:
                    logger.error(
                        f"Skipping baseline recapture for {self._scenario.name}: an account's "
                        "reset left unresolved orphans or failed; the prior baseline is "
                        "preserved to avoid absorbing a live orphan."
                    )

                # Mark redeploy as successful in results
                for result in results:
                    if hasattr(result, "needs_redeploy") and result.needs_redeploy:
                        result.redeploy_succeeded = True
                        result.reason = "Reset complete; deleted stacks recreated via setup"

                logger.info(f"Successfully redeployed {self._scenario.name}")

            except Exception as exc:  # noqa: BLE001 — do not raise; tagging below must run first
                logger.error(f"Redeploy failed for {self._scenario.name}: {exc}")
                for result in results:
                    if hasattr(result, "needs_redeploy") and result.needs_redeploy:
                        result.redeploy_succeeded = False
                        result.success = False
                        result.reason = "Stacks deleted but redeploy via setup failed"

        clear_error = await self._apply_contamination_tags(results)

        # Severity order: reset failure > clear failure.
        failed = [r for r in results if not r.success]
        if failed:
            raise ResetFailedError(
                failures=[r.reason or "reset failed" for r in failed],
            )

        if clear_error is not None:
            raise clear_error

    @staticmethod
    def _baseline_recapture_allowed(results: list[ResetResult]) -> bool:
        """A failed or orphan-carrying reset must not refresh the baseline.

        Recapture is scenario-wide, so ANY unclean account would absorb its live
        orphan into a fresh POST_SETUP baseline (``current - snapshot`` subtracts
        it forever). When unclean the trial fails and the run is discarded, so the
        stale baseline is correct to keep.
        """
        return all(r.success and not r.unresolved_orphans for r in results)

    @staticmethod
    def _orphan_identifiers(results: list[ResetResult]) -> set[str]:
        """Union of identifiers every result flagged as an unresolved orphan."""
        idents: set[str] = set()
        for r in results:
            for ids in (r.unresolved_orphans or {}).values():
                idents.update(item["Identifier"] for item in ids if "Identifier" in item)
        return idents

    async def _apply_contamination_tags(self, results: list[ResetResult]) -> Exception | None:
        """SET the flag on accounts whose reset failed, CLEAR it on those that succeeded.

        The ``AccountManager`` tag calls retry transient throttling internally.

        Returns a CLEAR failure (a succeeded account whose flag could not be
        cleared) for the caller to raise in severity order, or ``None``. It is
        returned rather than raised because a still-failing reset or a failed
        redeploy is the more severe outcome and must take precedence.

        SET-failure is fail-dangerous but the reset already fails via
        ``ResetFailedError`` (or a redeploy error), so it is logged as an error, not
        raised. A succeeded account is never re-flagged contaminated.
        """
        clear_error: Exception | None = None
        for r in results:
            if not r.account_id:
                logger.warning("Reset result missing account_id; cannot update flag: %s", r.reason)
                continue
            if not r.success:
                try:
                    await self._account_manager.mark_contaminated(r.account_id)
                except Exception as exc:  # noqa: BLE001 — reset failure stays primary
                    logger.error(
                        "Failed to flag contaminated account %s after reset failure: %s. "
                        "The account may be reused while dirty.",
                        r.account_id,
                        exc,
                    )
            else:
                try:
                    await self._account_manager.clear_contaminated(r.account_id)
                except Exception as exc:  # noqa: BLE001 — returned for severity-ordered raise
                    logger.error(
                        "Account %s reset succeeded but the contamination flag could not "
                        "be cleared: %s. The account is CLEAN but remains blocked; retry "
                        "'aws-bench env reset' to clear.",
                        r.account_id,
                        exc,
                    )
                    clear_error = exc
        return clear_error

    async def _redeploy_with_retry(self) -> None:
        """Run the post-reset DEPLOY redeploy, retrying transient failures.

        Reset's contract is to restore the environment to baseline, which
        includes recreating the stack(s) it deleted. A transient deploy failure
        (e.g. the S3 Tables bucket-name deletion cooldown returning a 409 right
        after the stack delete removed the bucket) clears within minutes, so
        retry with backoff rather than leaving the environment broken. The final
        attempt's exception propagates so a genuinely-failing redeploy still
        fails loud. A cooperative shutdown (OperationCancelled) is never retried.

        Uses the full container path (not bare ``_execute``): a reset with no
        reset.sh never started a container, and ``_execute`` requires one.
        """
        for attempt in range(1, _REDEPLOY_MAX_ATTEMPTS + 1):
            try:
                # Recovery redeploy: skip the contamination gate — this reset is
                # clearing the flag, so it must not block on it.
                await self._run_phase_in_container(ScenarioPhase.DEPLOY, check_contamination=False)
                return
            except (asyncio.CancelledError, OperationCancelled):
                raise
            except Exception as exc:
                if attempt >= _REDEPLOY_MAX_ATTEMPTS:
                    raise
                delay = _REDEPLOY_BACKOFF_SEC * attempt
                logger.warning(
                    "Redeploy attempt %d/%d for %s failed (%s); retrying in %.0fs",
                    attempt,
                    _REDEPLOY_MAX_ATTEMPTS,
                    self._scenario.name,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

    async def _run_pre_cleanup_residual_sweep(self) -> None:
        """Sweep post-setup residuals before the cleanup.sh container script.

        The relocated framework Phase-1 sweep, run first so ``cleanup.sh`` and stack
        deletion operate on a cleaner account. Best-effort: recorded, not gated on
        (stack deletion + orphan scan remain the authoritative pass/fail).
        """
        results = await ResourceManager.sweep_scenario_residuals_by_name(
            scenario_name=self._scenario.name,
            account_mapping=self._config.account_mapping,
            max_concurrent=self._config.cleanup_max_concurrent,
            all_regions=self._config.cleanup_all_regions,
            # Scenario regions are the floor when no baseline exists (same as _run_cleanup) —
            # else the sweep discovers all enabled regions and is SCP-denied on unused ones.
            regions=self._scenario.manifest.scenario.regions,
            output_dir=self._paths.phase_dir(ScenarioPhase.CLEANUP) / "pre-sweep",
        )
        if self._result.resource_results is None:
            self._result.resource_results = {}
        self._result.resource_results["pre_cleanup_sweep"] = results

    async def _run_cleanup(self, *, script_failed: bool = False) -> None:
        """Clean up this scenario's CloudFormation stacks."""
        # Recovery op: runs even if the custom script failed. sweep_post_setup=False
        # because _run_pre_cleanup_residual_sweep already swept before cleanup.sh.
        results = await ResourceManager.cleanup_scenarios_by_name(
            scenario_name=self._scenario.name,
            account_mapping=self._config.account_mapping,
            max_concurrent=self._config.cleanup_max_concurrent,
            all_regions=self._config.cleanup_all_regions,
            # Scenario regions are the floor when no baseline exists to derive them from — else
            # cleanup discovers all enabled regions and is SCP-denied on the ones it never used.
            regions=self._scenario.manifest.scenario.regions,
            output_dir=self._paths.phase_dir(ScenarioPhase.CLEANUP),
            sweep_post_setup=False,
        )
        if self._result.resource_results is None:
            self._result.resource_results = {}
        self._result.resource_results["cleanup"] = results

        # Results are stored above before raising, so the artifact keeps the orphaned-resource
        # data. Gate on ``is_clean`` (every stack deleted, no region error, AND a finished scan):
        # an incomplete orphan scan leaves total_orphaned at 0 while regions went unscanned, so
        # gating on orphan count alone would pass an unverified account and delete its baselines.
        failures = []
        for r in results:
            if r.error is not None:
                failures.append(f"{r.account_id}: {r.error or 'cleanup failed'}")
            elif r.summary is None:
                failures.append(f"{r.account_id}: cleanup produced no summary")
            elif not r.summary.is_clean:
                failures.append(f"{r.account_id}: {r.summary.failure_reason}")
        if failures:
            raise CleanupFailedError(failures=failures)

        # is_clean covers only framework cleanup, not the cleanup.sh failure held by the caller.
        # Keep the baselines a re-run needs rather than delete them on a half-failed cleanup.
        if script_failed:
            logger.debug("cleanup script failed; keeping baselines for re-run")
            return

        await self._delete_snapshots_after_cleanup()

        await self._clear_contamination_after_cleanup()

    async def _clear_contamination_after_cleanup(self) -> None:
        """Remove the contamination flag from every cleaned account (best-effort)."""
        for account_id in self._config.account_mapping.values():
            try:
                await self._account_manager.clear_contaminated(account_id)
            except Exception as exc:  # noqa: BLE001 — cleanup succeeded; flag clear is best-effort
                logger.warning(
                    "Cleanup succeeded for %s/%s but clearing the contamination flag "
                    "failed: %s. Re-run 'aws-bench env cleanup' to clear.",
                    self._scenario.name,
                    account_id,
                    exc,
                )

    async def _delete_snapshots_after_cleanup(self) -> None:
        """Delete init and setup snapshots after successful cleanup.

        After a successful cleanup, the account is empty (except AWS defaults).
        Stale snapshots would confuse subsequent env init/setup cycles — init
        would skip PRE_SETUP capture (idempotent check), and verify/reset would
        compare against an outdated baseline.
        """
        mgr = SnapshotManager()
        for account_id in self._config.account_mapping.values():
            for stage in (SnapshotStage.PRE_SETUP, SnapshotStage.POST_SETUP):
                try:
                    if mgr.snapshot_exists(self._scenario.name, account_id, stage):
                        mgr.delete_snapshot(self._scenario.name, account_id, stage)
                except Exception as e:
                    # Non-fatal: cleanup succeeded, snapshot deletion is best-effort
                    logger.warning(
                        "Failed to delete %s snapshot for %s/%s: %s",
                        stage.value,
                        self._scenario.name,
                        account_id,
                        e,
                    )
