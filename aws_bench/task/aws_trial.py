"""AwsBenchTrial — AWS credential injection + placeholder substitution.

``AwsBenchTrial.create`` is the factory; ``AwsBenchSingleStepTrial`` is the
concrete trial, carrying the AWS behavior as lifecycle overrides. Multi-step AWS
tasks are not supported (per-step pre/post-invoke credentialing is undefined).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator

from harbor.agents.oracle import OracleAgent
from harbor.models.task.task import Task
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.paths import TrialPaths
from harbor.trial.single_step import SingleStepTrial

from aws_bench.account_management.manager import AccountManager
from aws_bench.dataset.models import RoleType, ScriptType
from aws_bench.dataset.task_config import AwsBenchTask, ConcurrencyMode, PhaseScript
from aws_bench.exceptions import AccountContaminatedError, OperationCancelled
from aws_bench.logging.logger import (
    FILE_FORMAT,
    DefaultPrefixFilter,
    ShortNameFormatter,
    log_context,
)
from aws_bench.scenario.events import ScenarioPhase
from aws_bench.scenario.job_config import ScenarioTrialConfig
from aws_bench.scenario.trial import ScenarioTrial
from aws_bench.task.aws_creds import assume_role_for_script, resolve_env_with_creds
from aws_bench.task.script_runner import ScriptRunner
from aws_bench.task.trial_config import AwsBenchTrialConfig
from aws_bench.utils.credentials_provider import CredentialProvider, build_aws_credentials_file
from aws_bench.utils.placeholders import substitute_placeholders, update_placeholder_values

PLACEHOLDER_OUTPUT_FILE_NAME = "placeholder.json"

# In-container AWS credentials file at the SDK's default location, so tools
# resolve it with no extra env. ``$HOME`` is expanded by the in-container shell,
# which runs as the stage's own user, so the file lands in that user's home.
_CREDS_FILE_PATH = "$HOME/.aws/credentials"

# Heredoc terminator for the credentials-file write. A body containing this
# line could close the heredoc early and inject shell, so such a body is
# rejected before the write.
_CREDS_HEREDOC_SENTINEL = "AWSBENCH_CREDS_EOF"

# AWS env vars emptied in the stage env so neither a host-forwarded credential
# set nor a stray default-profile selector can outrank the credentials file and
# the AWS_PROFILE the stage sets to its tag.
_RAW_CRED_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_PROFILE",
)


class AwsBenchSingleStepTrial(SingleStepTrial):
    """Single-step trial with AWS credential injection + placeholder substitution.

    Built by ``AwsBenchTrial.create``. Not instantiated directly.
    """

    config: AwsBenchTrialConfig
    task: AwsBenchTask

    def __init__(self, config: TrialConfig, *, _task: Task | None = None) -> None:
        """Initialize state before the base init so teardown paths can read it.

        Teardown can run before ``_prepare`` (failure/cancel during setup), so
        these must exist at construction.
        """
        self._aws_placeholders: dict[str, dict[str, str]] = {}
        self._aws_post_invoke_done = False
        # Gates post-invoke: skipped if setup never produced a running container.
        self._agent_container_started = False
        self._account_manager = AccountManager()
        super().__init__(config, _task=_task)

    def _init_logger(self) -> None:
        """Give trial.log the aws-bench file format, replacing Harbor's bare handler.

        ``super()`` is required for its durable effect — creating ``self.logger``,
        the per-trial logger every component shares — so we keep it and only swap
        the handler it attaches.
        """
        super()._init_logger()
        if self._log_handler is not None:
            self.logger.removeHandler(self._log_handler)
            self._log_handler.close()
        handler = logging.FileHandler(self.paths.log_path)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(ShortNameFormatter(FILE_FORMAT))
        handler.addFilter(DefaultPrefixFilter())
        self._log_handler = handler
        self.logger.addHandler(self._log_handler)

    async def run(self):  # type: ignore[override]
        """Run the trial under its log context; reset the account after a mutating run.

        ``super().run()`` re-raises cancellation and lets ``OperationCancelled``
        propagate, so it only returns on a settled run — a cancelled or shutdown
        trial never reaches the reset line. The queue holds the scenario's
        exclusive gate across this method, so the next mutating trial on the
        account waits for the reset.
        """
        with log_context(self.config.trial_name):
            result = await super().run()
            if self.config.concurrency_mode is ConcurrencyMode.MUTATING:
                await self._reset_scenario_account()
            return result

    async def _reset_scenario_account(self) -> None:
        """Restore the scenario account to baseline via ``ScenarioTrial(RESET)``.

        Runs the env-side recovery flow (reset.sh, infra diff/restore, redeploy +
        re-snapshot on un-revertable stacks), writing under
        ``<trial_dir>/scenario-reset/``. A reset failure is logged, never raised:
        it must not fail a finished benchmark. Only cancellation propagates.
        """
        reset_config = ScenarioTrialConfig(
            scenario=self.config.scenario,
            output_dir=self.paths.trial_dir,
            trial_name="scenario-reset",
            account_mapping=self.config.account_mapping,
            timeout_multiplier=self.config.timeout_multiplier,
        )
        try:
            trial = await ScenarioTrial.create(reset_config, CredentialProvider.get())
            reset_result = await trial.run(ScenarioPhase.RESET)
            if not reset_result.success:
                self.logger.error(
                    "Post-trial reset did not restore %s; the account is flagged "
                    "contaminated and later trials will be refused. Run "
                    "'aws-bench env cleanup' to clean it and clear the flag.",
                    self.config.scenario_id,
                )
        except (asyncio.CancelledError, OperationCancelled):
            raise
        except Exception as exc:  # noqa: BLE001 — reset must not fail a finished benchmark
            self.logger.error("Post-trial reset raised for %s: %s", self.config.scenario_id, exc)

    def _assume_all_tags(self, role_type: RoleType) -> dict[str, dict[str, str]]:
        """Assume ``role_type``'s role in every account tag of this trial.

        Returns ``{account_tag: creds}``. Each tag resolves the scoped role when
        the task sets one for this phase, else the org-access role.
        """
        role_name = self.task.config.scenario.role_name(role_type)
        return {
            tag: assume_role_for_script(
                account_id=account_id,
                role_name=role_name,
                role_type=role_type,
                task_name=self.task.name,
                job_id=self.config.job_id,
            )
            for tag, account_id in self.config.account_mapping.items()
        }

    @contextlib.asynccontextmanager
    async def _staged_credentials(
        self, role_type: RoleType
    ) -> AsyncGenerator[dict[str, str], None]:
        """Write the per-tag credentials file into the container; remove it on exit.

        Writes ``~/.aws/credentials`` as the stage's own user (so ``$HOME`` and
        file ownership match the consuming process) with one static-credential
        profile per account tag. Yields the credential-chain env vars emptied to
        ``""`` so a host-forwarded credential set cannot outrank the file, plus
        ``AWS_PROFILE`` set to the first tag. For the agent role only,
        ``AWS_REGION``/``AWS_DEFAULT_REGION`` are pinned to the scenario's first
        declared region. Removed on exit so a later stage in the same container
        never inherits these credentials.

        Raises:
            RuntimeError: if the account mapping is empty, the credentials body
                could break out of the heredoc, or the in-container write fails.
        """
        if not self.config.account_mapping:
            raise RuntimeError(
                f"trial {self.config.trial_name}: empty account_mapping; cannot stage credentials"
            )

        per_tag = self._assume_all_tags(role_type)
        body = build_aws_credentials_file(per_tag)
        if _CREDS_HEREDOC_SENTINEL in body:
            raise RuntimeError("credentials body contains the heredoc sentinel")

        # Run as the stage's user so $HOME resolves to that user's home and the
        # file is owned by the process that reads it. Quoted-sentinel heredoc:
        # the body (STS secrets) is written literally, no shell expansion.
        user = self.task.config.agent.user
        await self._exec_checked(
            command=(
                f"mkdir -p $(dirname {_CREDS_FILE_PATH}) && "
                f"cat > {_CREDS_FILE_PATH} <<'{_CREDS_HEREDOC_SENTINEL}'\n"
                f"{body}\n"
                f"{_CREDS_HEREDOC_SENTINEL}\n"
                f"chmod 600 {_CREDS_FILE_PATH}"
            ),
            user=user,
            action="write credentials file",
        )
        cred_env: dict[str, str] = dict.fromkeys(_RAW_CRED_VARS, "")
        # Default AWS_PROFILE to the first tag so a single-account task gets
        # ambient credentials by default; multi-account tasks override per tag.
        tag = next(iter(self.config.account_mapping))
        cred_env["AWS_PROFILE"] = tag
        cred_env["AWS_DEFAULT_PROFILE"] = tag
        # Agent only: pre/post-invoke and verifier scripts declare their own
        # regions in task.toml [*.env], which this would otherwise override.
        if role_type is RoleType.AGENT:
            cred_env["AWS_REGION"] = self.config.regions[0]
            cred_env["AWS_DEFAULT_REGION"] = self.config.regions[0]

        try:
            yield cred_env
        finally:
            # Best-effort removal: never let a cleanup failure mask the body's
            # exception. A surviving file is bounded to this trial's container.
            try:
                await self.agent_environment.exec(
                    command=f"rm -f {_CREDS_FILE_PATH}", user=self.task.config.agent.user
                )
            except Exception as exc:  # noqa: BLE001 — cleanup must not raise
                self.logger.warning("Failed to remove credentials file: %s", exc)

    async def _exec_checked(self, *, command: str, user, action: str):
        """Exec in the agent environment; raise ``RuntimeError`` on non-zero exit.

        ``environment.exec`` reports failures through the result object only, so
        call this when a non-zero exit must abort. ``action`` names the step in
        the error message.
        """
        result = await self.agent_environment.exec(command=command, user=user)
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to {action} for {self.config.trial_name} "
                f"(exit {result.return_code}): {result.stderr or result.stdout}"
            )
        return result

    async def _run_phase_script(
        self,
        *,
        script_type: ScriptType,
        role_type: RoleType,
        phase: PhaseScript,
        output_file_name: str | None = None,
    ) -> dict[str, str]:
        """Stage the phase's per-tag credentials file, resolve env, run the script."""
        self.logger.info("Running %s script", script_type)
        async with self._staged_credentials(role_type) as cred_env:
            override_env = resolve_env_with_creds(
                raw_env=phase.env, placeholders=self._aws_placeholders, creds=cred_env
            )
            runner = ScriptRunner(
                script_type=script_type,
                task_dir=self.task.paths.task_dir,
                trial_paths=TrialPaths(trial_dir=self.paths.trial_dir),
                environment=self.agent_environment,
                override_env=override_env,
                timeout_sec=phase.timeout_sec,
                script_logger=self.logger,  # route lines into the trial's own trial.log
            )
            return await runner.run(output_file_name=output_file_name)

    async def _setup_agent_environment(self) -> None:
        """Record that the agent container reached a running state."""
        await super()._setup_agent_environment()
        self._agent_container_started = True

    async def _raise_if_contaminated(self) -> None:
        """Raise AccountContaminatedError if any of this trial's accounts is flagged.

        ``get_contaminated_accounts`` is a blocking per-account Organizations read;
        run it off the loop so concurrent sibling trials aren't stalled.
        """
        account_ids = list(self.config.account_mapping.values())
        contaminated = await asyncio.to_thread(
            self._account_manager.get_contaminated_accounts, account_ids
        )
        if contaminated:
            raise AccountContaminatedError(
                account_ids=contaminated,
                scenario_id=self.config.scenario_id,
            )

    async def _prepare(self) -> None:
        """Seed placeholders, start the agent environment, run pre-invoke.

        Pre-invoke needs the running container; if setup fails before the
        container starts, pre-invoke is skipped.
        """
        if self.config.verify_env:
            await self._raise_if_contaminated()

        # Fresh inner dict per tag so the pre-invoke merge below can't mutate the
        # shared self.config.exports.
        self._aws_placeholders = {tag: dict(v) for tag, v in self.config.exports.items()}

        await super()._prepare()

        if self.task.has_phase_script(ScriptType.PRE_INVOKE):
            output = await self._run_phase_script(
                script_type=ScriptType.PRE_INVOKE,
                role_type=RoleType.PRE_INVOKE,
                phase=self.task.config.pre_invoke,
                output_file_name=PLACEHOLDER_OUTPUT_FILE_NAME,
            )
            if output:
                # Pre-invoke placeholders are deployed-resource identifiers, never
                # credentials; logging them makes an unresolved {{...}} visible here
                # rather than as an opaque downstream error.
                self.logger.debug(
                    "Got %d placeholder(s) from pre-invoke: %s",
                    len(output),
                    ", ".join(f"{{{{{k}}}}}={v}" for k, v in sorted(output.items())),
                )
                self._aws_placeholders = update_placeholder_values(self._aws_placeholders, output)
            else:
                self.logger.debug("No placeholders produced from pre-invoke script.")

    async def _run_agent_phase(self, *, instruction: str, **kwargs) -> None:
        """Substitute placeholders, then run the agent phase under scoped creds."""
        if self._aws_placeholders:
            instruction = substitute_placeholders(instruction, self._aws_placeholders)

        # Refuse agents without _extra_env: they would silently use host creds.
        if not hasattr(self.agent, "_extra_env"):
            raise RuntimeError(
                f"Agent {type(self.agent).__name__} does not support AWS credential "
                "injection (no _extra_env); cannot run an aws-bench trial with it."
            )

        async with self._staged_credentials(RoleType.AGENT) as cred_env:
            extra_env = self.agent._extra_env  # type: ignore[attr-defined]
            saved = dict(extra_env)
            # Oracle only: resolve [solution.env] placeholders into its env; real
            # agents discover resources from the instruction and never see it.
            # Harbor's OracleAgent re-parses task.toml into its OWN config and, in
            # run(), re-applies the RAW solution.env via a ${VAR}-only resolver that
            # mangles our {{...}} tokens. So resolve from and blank THAT object (the
            # trial's copy wouldn't reach harbor); restore after, config outlives us.
            oracle_config = (
                self.agent._task.config
                if isinstance(self.agent, OracleAgent)
                and getattr(self.agent, "_task", None) is not None
                else None
            )
            solution_env = oracle_config.solution.env if oracle_config is not None else {}
            extra_env.update(
                resolve_env_with_creds(
                    raw_env=solution_env, placeholders=self._aws_placeholders, creds=cred_env
                )
            )
            if oracle_config is not None:
                oracle_config.solution.env = {}
            try:
                await super()._run_agent_phase(instruction=instruction, **kwargs)
            finally:
                if oracle_config is not None:
                    oracle_config.solution.env = solution_env
                extra_env.clear()
                extra_env.update(saved)

    @contextlib.asynccontextmanager
    async def _verifier_creds(self) -> AsyncGenerator[None, None]:
        """Stage the verifier's creds file and transiently overlay ``verifier.env``.

        The env overlay (placeholders + emptied raw-credential vars) is restored
        on exit: the config is persisted and reused across retries, so a permanent
        mutation would leak creds to disk and break resume equality. The creds
        file is removed by the staging context.
        """
        async with self._staged_credentials(RoleType.VERIFIER) as cred_env:
            original_env = self.task.config.verifier.env
            self.task.config.verifier.env = resolve_env_with_creds(
                raw_env=original_env, placeholders=self._aws_placeholders, creds=cred_env
            )

            try:
                yield
            finally:
                self.task.config.verifier.env = original_env

    async def _run_shared_verifier(self, **kwargs):
        async with self._verifier_creds():
            return await super()._run_shared_verifier(**kwargs)

    async def _recover_outputs(self) -> None:
        """Salvage agent outputs without stopping the env.

        The env stop is deferred to ``_finalize`` (which runs after
        ``_emit(CANCEL)``) so the long post-invoke reset cannot strand the
        cancellation signal behind it.
        """
        await self._sync_agent_output(self.result)
        await self._collect_artifacts()

    async def _stop_agent_environment(self) -> None:
        """Run post-invoke (the account reset) once, then the base teardown.

        A cancelled post-invoke is recorded but NOT re-raised: this runs inside
        Harbor's ``_finalize``, which must still persist the result and emit END.
        Swallowing here doesn't strand the cancel — ``_finalize`` runs in
        ``Trial.run``'s ``finally``, so the originating cancel resumes unwinding
        after it. A dirty account left by an interrupted reset is corrected by
        the scenario reset/cleanup phase.
        """
        try:
            run_post_invoke = (
                self._agent_container_started
                and not self._aws_post_invoke_done
                and self.task.has_phase_script(ScriptType.POST_INVOKE)
            )
            if run_post_invoke:
                self._aws_post_invoke_done = True
                try:
                    await self._run_phase_script(
                        script_type=ScriptType.POST_INVOKE,
                        role_type=RoleType.POST_INVOKE,
                        phase=self.task.config.post_invoke,
                    )
                except (asyncio.CancelledError, OperationCancelled) as e:
                    self.logger.warning(
                        "Post-invoke interrupted by cancellation; the "
                        "scenario account may be left dirty"
                    )
                    self._record_exception(e)
                except Exception as e:  # noqa: BLE001 — recorded; must not block teardown
                    self.logger.exception("Post-invoke script failed")
                    self._record_exception(e)
        finally:
            await super()._stop_agent_environment()


class AwsBenchTrial:
    """Factory: ``create`` builds an ``AwsBenchSingleStepTrial`` (refusing multi-step)."""

    @classmethod
    async def create(cls, config: TrialConfig) -> AwsBenchSingleStepTrial:
        """Build the concrete single-step trial, refusing multi-step AWS tasks."""
        task = await AwsBenchTask.from_config(config.task, config.extra_instruction_paths)
        if task.has_steps:
            raise NotImplementedError(
                "multi-step AWS tasks are not yet supported (per-step pre/post-invoke "
                "credentialing is undefined)."
            )
        return AwsBenchSingleStepTrial(config, _task=task)
