"""AwsBenchTrial — AWS credential injection + placeholder substitution.

``AwsBenchTrial.create`` is the factory; ``AwsBenchSingleStepTrial`` is the
concrete trial, carrying the AWS behavior as lifecycle overrides. Multi-step AWS
tasks are not supported (per-step pre/post-invoke credentialing is undefined).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import shlex
import tempfile
from collections.abc import AsyncGenerator, Coroutine, Iterator
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.agents.oracle import OracleAgent
from harbor.environments.base import ExecResult
from harbor.models.task.task import Task
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.paths import TrialPaths
from harbor.trial.errors import AgentTimeoutError
from harbor.trial.single_step import SingleStepTrial
from harbor.utils.env import resolve_env_vars

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
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
from aws_bench.task.aws_creds import resolve_env_with_creds, session_name
from aws_bench.task.script_runner import ScriptRunner
from aws_bench.task.trial_config import AwsBenchTrialConfig
from aws_bench.utils import credentials_provider
from aws_bench.utils.credentials_provider import (
    CREDS_DIR,
    CredentialProvider,
    build_aws_config,
    check_static_profiles,
    credential_command,
    credential_env,
    credential_refresh_delay,
    mint_credentials,
    refresh_credentials_loop,
)
from aws_bench.utils.placeholders import substitute_placeholders, update_placeholder_values

PLACEHOLDER_OUTPUT_FILE_NAME = "placeholder.json"

# Trial-name prefix and Docker-label value for the post-trial account reset. The
# invoking task trial name (unique per attempt) is appended to the prefix so the
# derived scenario container name is unique per reset: overlapping resets of
# different scenarios on one Docker daemon no longer share the fixed
# ``awsbench-scenario-reset`` name and force-remove each other mid-reset. The
# ``awsbench.role`` label lets operational tooling match reset containers by role
# rather than by name.
_SCENARIO_RESET_ROLE = "scenario-reset"
_ROLE_LABEL_KEY = "awsbench.role"

_CREDENTIAL_OPERATION_TIMEOUT_SEC = 30


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
        self._credential_dir: PurePosixPath | None = None
        self._credential_operation_task: asyncio.Task | None = None
        self._credential_failed = False
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
        ``<trial_dir>/scenario-reset-<trial_name>/``. The reset trial name (and
        thus the scenario container name) is suffixed with the invoking trial name
        so concurrent resets of different scenarios never collide on one Docker
        daemon. A reset failure is logged, never raised: it must not fail a
        finished benchmark. Only cancellation propagates.
        """
        reset_config = ScenarioTrialConfig(
            scenario=self.config.scenario,
            output_dir=self.paths.trial_dir,
            trial_name=f"{_SCENARIO_RESET_ROLE}-{self.config.trial_name}",
            account_mapping=self.config.account_mapping,
            timeout_multiplier=self.config.timeout_multiplier,
            labels={_ROLE_LABEL_KEY: _SCENARIO_RESET_ROLE},
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

    @contextlib.contextmanager
    def _credential_commands(
        self,
        profile: str,
        user: str | int | None,
        env: dict[str, str] | None = None,
    ) -> Iterator[None]:
        """Remove credential sources after Harbor and the image supply their env."""
        environment = self.agent_environment
        original_exec = environment.exec
        had_override = "exec" in vars(environment)
        phase_env = dict(env or {})

        async def scoped_exec(
            command: str,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_sec: int | None = None,
            user: str | int | None = None,
        ) -> ExecResult:
            return await original_exec(
                command=credential_command(command, profile),
                cwd=cwd,
                env=credential_env(profile, {**phase_env, **(env or {})}),
                timeout_sec=timeout_sec,
                user=user,
            )

        with environment.with_default_user(user):
            environment.exec = scoped_exec
            try:
                yield
            finally:
                if had_override:
                    environment.exec = original_exec
                else:
                    del environment.exec

    async def _credential_operation[T](
        self, operation: Coroutine[Any, Any, T], *, publication: bool = False
    ) -> T:
        """Settle a file operation before propagating cancellation to its owner."""
        previous = self._credential_operation_task
        if previous is not None and not previous.done():
            operation.close()
            raise RuntimeError("A previous credential file operation is still active")
        task = asyncio.create_task(operation)
        self._credential_operation_task = task
        deadline = asyncio.get_running_loop().time() + _CREDENTIAL_OPERATION_TIMEOUT_SEC
        cancellation: asyncio.CancelledError | None = None
        try:
            while not task.done():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    self._credential_failed |= publication
                    task.add_done_callback(
                        lambda done: None if done.cancelled() else done.exception()
                    )
                    if cancellation is not None:
                        raise cancellation
                    raise TimeoutError("Credential file operation did not finish before timeout")
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
                except asyncio.CancelledError as exc:
                    cancellation = cancellation or exc
                except Exception:
                    if task.done():
                        break
            if task.cancelled():
                self._credential_failed |= publication
            if cancellation is not None:
                if not task.cancelled():
                    task.exception()
                raise cancellation
            return task.result()
        finally:
            if task.done():
                self._credential_operation_task = None

    async def _credential_home(self) -> tuple[PurePosixPath, str]:
        """Resolve the same home and owner that the phase's commands use."""
        result = await self._exec_checked(
            command='test -n "$HOME" && cd -- "$HOME" && pwd -P && id -u && id -g',
            user=None,
            action="resolve the credential home",
        )
        parts = (result.stdout or "").splitlines()
        if (
            len(parts) != 3
            or not PurePosixPath(parts[0]).is_absolute()
            or ".." in PurePosixPath(parts[0]).parts
            or "\0" in parts[0]
            or not all(re.fullmatch(r"[0-9]+", value) for value in parts[1:])
        ):
            raise RuntimeError("Cannot determine the phase user's home and file ownership")
        return PurePosixPath(parts[0]), f"{parts[1]}:{parts[2]}"

    async def _check_credential_mounts(self, directory: PurePosixPath) -> None:
        """Refuse mounts that would place credential cleanup outside the container."""
        result = await self._exec_checked(
            command="cat /proc/self/mountinfo", user=None, action="check credential mounts"
        )
        lines = (result.stdout or "").splitlines()
        if not lines:
            raise RuntimeError("Cannot inspect the container's credential mounts")
        targets = (directory, directory.parent / "config")
        for line in lines:
            fields = line.split()
            if len(fields) < 7 or "-" not in fields:
                raise RuntimeError("Invalid container mount information")
            destination = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[4])
            mounted = PurePosixPath(destination)
            if mounted == PurePosixPath("/"):
                continue
            if not mounted.is_absolute() or ".." in mounted.parts:
                raise RuntimeError("Invalid container mount destination")
            if any(
                target.is_relative_to(mounted) or mounted.is_relative_to(target)
                for target in targets
            ):
                raise RuntimeError(f"Container mount overlaps the credential path: {destination}")

    async def _read_credential_config(self, path: PurePosixPath) -> str:
        """Read a regular config file without putting its contents in error messages."""
        quoted = shlex.quote(str(path))
        result = await self._exec_checked(
            command=(
                f"test ! -L {quoted} && "
                f"if [ -e {quoted} ]; then test -f {quoted} && cat {quoted}; fi"
            ),
            user=None,
            action="read AWS credential configuration",
        )
        return result.stdout or ""

    async def _clear_credentials(self, directory: PurePosixPath) -> None:
        """Clear every credential file, including partial uploads, through Harbor."""
        await self._exec_checked(
            command=f"test ! -L {shlex.quote(str(directory.parent))}",
            user="root",
            action="check the AWS directory",
        )
        result = await self.agent_environment.empty_dirs([directory], chmod=False)
        if result is None or result.return_code != 0:
            raise RuntimeError("Failed to clear the credential directory")
        if self._credential_dir == directory:
            self._credential_dir = None

    async def _publish_credential_file(
        self, path: PurePosixPath, body: str, owner: str, expires_at: datetime | None = None
    ) -> None:
        """Upload privately, then replace the destination in one filesystem rename."""
        assert self._credential_dir is not None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as temporary:
                source = Path(temporary.name)
                temporary.write(body)
                temporary.flush()
                destination = self._credential_dir / f".{path.name}.{source.name}.tmp"
                quoted = shlex.quote(str(destination))
                await self.agent_environment.upload_file(str(source), str(destination))
                freshness = ""
                if expires_at is not None:
                    credential_refresh_delay(expires_at)
                    latest = int(
                        expires_at.timestamp() - credentials_provider.CRED_REFRESH_MIN_SLEEP_SEC
                    )
                    freshness = f'test "$(date +%s)" -lt {latest} && '
                await self._exec_checked(
                    command=(
                        f"test -f {quoted} && test ! -L {quoted} && "
                        f"chmod 600 {quoted} && chown {owner} {quoted} && "
                        f"{freshness}mv -f -- {quoted} {shlex.quote(str(path))}"
                    ),
                    user="root",
                    action="publish credential file",
                )
        except Exception:
            # Transport errors can include remote output. Never copy it into trial logs.
            raise RuntimeError(f"Failed to publish credential file {path.name}") from None

    @contextlib.asynccontextmanager
    async def _staged_credentials(
        self,
        role_type: RoleType,
        *,
        user: str | int | None = None,
        env: dict[str, str] | None = None,
    ) -> AsyncGenerator[dict[str, str], None]:
        """Publish renewable phase credentials and clear the directory on exit."""
        accounts = dict(self.config.account_mapping)
        if not accounts:
            raise RuntimeError(
                f"trial {self.config.trial_name}: empty account_mapping; cannot stage credentials"
            )
        if self._credential_failed:
            raise RuntimeError("Previous credential phase failed; cannot start another phase")
        profile = next(iter(accounts))
        cred_env = credential_env(profile)
        if role_type is RoleType.AGENT:
            cred_env["AWS_REGION"] = self.config.regions[0]
            cred_env["AWS_DEFAULT_REGION"] = self.config.regions[0]
        role_name = self.task.config.scenario.role_name(role_type) or ORG_ACCESS_ROLE
        label = session_name(job_id=self.config.job_id)
        provider = CredentialProvider.get()
        refresher: asyncio.Task[None] | None = None
        abort_phase: asyncio.Timeout | None = None
        consumer_started = False
        primary: BaseException | None = None
        with self._credential_commands(profile, user, {**(env or {}), **cred_env}):
            try:
                if self._credential_dir is not None:
                    await self._credential_operation(self._clear_credentials(self._credential_dir))
                home, owner = await self._credential_home()
                directory = home / CREDS_DIR
                await self._check_credential_mounts(directory)
                await self._exec_checked(
                    command=(
                        f"test ! -L {shlex.quote(str(directory.parent))} && "
                        f"mkdir -p {shlex.quote(str(directory.parent))} && "
                        f"chown {owner} {shlex.quote(str(directory.parent))}"
                    ),
                    user="root",
                    action="prepare the AWS directory",
                )
                self._credential_dir = directory
                await self._credential_operation(self._clear_credentials(directory))
                self._credential_dir = directory
                await self._exec_checked(
                    command=(
                        f"chmod 700 {shlex.quote(str(directory))} && "
                        f"chown {owner} {shlex.quote(str(directory))}"
                    ),
                    user="root",
                    action="set credential directory permissions",
                )
                config_path = directory.parent / "config"
                quoted_config = shlex.quote(str(config_path))
                await self._exec_checked(
                    command=(
                        f"test ! -L {quoted_config} && "
                        f"if [ -e {quoted_config} ]; then test -f {quoted_config} && "
                        f"chmod 600 {quoted_config} && chown {owner} {quoted_config}; fi"
                    ),
                    user="root",
                    action="set AWS config permissions",
                )
                original_config = await self._read_credential_config(config_path)
                static_config = await self._read_credential_config(directory.parent / "credentials")
                check_static_profiles(accounts, static_config)
                config = build_aws_config(accounts, original_config)
                files, expires_at = await asyncio.to_thread(
                    mint_credentials, provider, accounts, role_name, label
                )

                async def publish(changed: dict[str, str], expiration: datetime) -> None:
                    for name, body in changed.items():
                        await self._publish_credential_file(
                            directory / name, body, owner, expiration
                        )

                if config != original_config:
                    await self._credential_operation(
                        self._publish_credential_file(config_path, config, owner),
                        publication=True,
                    )
                await self._credential_operation(publish(files, expires_at), publication=True)
                await self._exec_checked(
                    command=" && ".join(
                        f"test -r {shlex.quote(str(path))}"
                        for path in (config_path, *(directory / name for name in files))
                    ),
                    user=None,
                    action="check phase credential access",
                )
                credential_refresh_delay(expires_at)

                async def refresh_once() -> datetime:
                    nonlocal files
                    fresh, expiration = await asyncio.to_thread(
                        mint_credentials, provider, accounts, role_name, label
                    )
                    if fresh.keys() != files.keys():
                        raise RuntimeError("Credential refresh changed the account profiles")
                    changed = {name: body for name, body in fresh.items() if files[name] != body}
                    try:
                        await self._credential_operation(
                            publish(changed, expiration), publication=True
                        )
                    finally:
                        if self._credential_failed and abort_phase is not None:
                            abort_phase.reschedule(asyncio.get_running_loop().time())
                    credential_refresh_delay(expiration)
                    files = fresh
                    return expiration

                refresher = asyncio.create_task(
                    refresh_credentials_loop(expires_at, refresh_once, self.logger)
                )
                guard = asyncio.timeout(None)
                try:
                    async with guard:
                        abort_phase = guard
                        consumer_started = True
                        yield cred_env
                except TimeoutError:
                    if guard.expired():
                        raise RuntimeError(
                            "Credential publication did not settle; ending the trial"
                        ) from None
                    raise
                finally:
                    abort_phase = None
                if self._credential_failed:
                    raise RuntimeError("Credential publication failed during the phase")
            except BaseException as exc:
                # Harbor cancellation can leave the container command alive.
                # Never give that command credentials for a later role.
                # Docker command timeouts arrive wrapped in RuntimeError.
                cause = exc.__cause__ if exc.__cause__ is not None else exc.__context__
                if consumer_started and (
                    isinstance(
                        exc,
                        (
                            asyncio.CancelledError,
                            OperationCancelled,
                            AgentTimeoutError,
                            TimeoutError,
                        ),
                    )
                    or isinstance(cause, TimeoutError)
                ):
                    self._credential_failed = True
                primary = exc
                raise
            finally:
                exit_error: BaseException | None = None
                current = asyncio.current_task()
                cancellations = current.cancelling() if current is not None else 0
                if refresher is not None:
                    refresher.cancel()
                    try:
                        await refresher
                    except asyncio.CancelledError:
                        if current is not None and current.cancelling() > cancellations:
                            exit_error = asyncio.CancelledError()
                    except BaseException as exc:
                        exit_error = exc
                operation = self._credential_operation_task
                if self._credential_dir is not None and (operation is None or operation.done()):
                    try:
                        await self._credential_operation(
                            self._clear_credentials(self._credential_dir)
                        )
                    except (asyncio.CancelledError, OperationCancelled) as exc:
                        exit_error = exit_error or exc
                        if self._credential_dir is not None:
                            self.logger.warning("Credential cleanup was interrupted")
                    except Exception:
                        self.logger.warning("Credential cleanup failed; teardown will retry it")
                elif self._credential_dir is not None:
                    self.logger.warning(
                        "Credential file operation is still active; container teardown will proceed"
                    )
                if exit_error is not None and primary is None:
                    raise exit_error
                if self._credential_failed and primary is None:
                    raise RuntimeError("Credential publication did not settle; ending the trial")

    async def _exec_checked(
        self, *, command: str, user: str | int | None, action: str
    ) -> ExecResult:
        """Check credential I/O without including possibly sensitive command output."""
        result = await self.agent_environment.exec(command=command, user=user)
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to {action} for {self.config.trial_name} (exit {result.return_code})"
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
        """Resolve the script's environment and run with renewable credentials."""
        self.logger.info("Running %s script", script_type)
        phase_env = resolve_env_with_creds(
            raw_env=phase.env,
            placeholders=self._aws_placeholders,
            creds=credential_env(next(iter(self.config.account_mapping))),
        )
        async with self._staged_credentials(
            role_type, user=self.agent_environment.default_user, env=phase_env
        ) as cred_env:
            override_env = {**phase_env, **cred_env}
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

    async def _run_agent_phase(self, *, instruction: str, user: str | int | None, **kwargs) -> None:
        """Substitute placeholders, then run the agent phase under scoped creds."""
        if self._aws_placeholders:
            instruction = substitute_placeholders(instruction, self._aws_placeholders)

        # Refuse agents without _extra_env: they would silently use host creds.
        if not hasattr(self.agent, "_extra_env"):
            raise RuntimeError(
                f"Agent {type(self.agent).__name__} does not support AWS credential "
                "injection (no _extra_env); cannot run an aws-bench trial with it."
            )

        extra_env = self.agent._extra_env  # type: ignore[attr-defined]
        saved = dict(extra_env)
        # Oracle reparses task.toml into its own config and reapplies solution.env.
        # Resolve that copy here; real agents must never receive solution settings.
        oracle_config = (
            self.agent._task.config
            if isinstance(self.agent, OracleAgent)
            and getattr(self.agent, "_task", None) is not None
            else None
        )
        solution_env = oracle_config.solution.env if oracle_config is not None else {}
        phase_env = {
            **extra_env,
            **resolve_env_vars(
                resolve_env_with_creds(
                    raw_env=solution_env,
                    placeholders=self._aws_placeholders,
                    creds=credential_env(next(iter(self.config.account_mapping))),
                )
            ),
        }
        async with self._staged_credentials(RoleType.AGENT, user=user, env=phase_env) as cred_env:
            extra_env.update({**phase_env, **cred_env})
            if oracle_config is not None:
                oracle_config.solution.env = {}
            try:
                await super()._run_agent_phase(instruction=instruction, user=user, **kwargs)
            finally:
                if oracle_config is not None:
                    oracle_config.solution.env = solution_env
                extra_env.clear()
                extra_env.update(saved)

    @contextlib.asynccontextmanager
    async def _verifier_creds(
        self, *, user: str | int | None, env: dict[str, str] | None = None
    ) -> AsyncGenerator[None, None]:
        """Stage the verifier's credentials and overlay both configuration layers.

        The env overlay (placeholders + emptied raw-credential vars) is restored
        on exit: the config is persisted and reused across retries, so a permanent
        mutation would leak creds to disk and break resume equality. The creds
        directory is cleared by the staging context.
        """
        original_task_env = self.task.config.verifier.env
        original_trial_env = self.config.verifier.env
        merged_env = resolve_env_with_creds(
            raw_env={**original_task_env, **(env or {}), **original_trial_env},
            placeholders=self._aws_placeholders,
            creds=credential_env(next(iter(self.config.account_mapping))),
        )
        async with self._staged_credentials(
            RoleType.VERIFIER, user=user, env=resolve_env_vars(merged_env)
        ) as cred_env:
            self.task.config.verifier.env = {**merged_env, **cred_env}
            self.config.verifier.env = {**merged_env, **cred_env}
            try:
                yield
            finally:
                self.task.config.verifier.env = original_task_env
                self.config.verifier.env = original_trial_env

    async def _run_shared_verifier(
        self, *, user: str | int | None, env: dict[str, str] | None = None, **kwargs
    ):
        async with self._verifier_creds(user=user, env=env):
            return await super()._run_shared_verifier(user=user, env=env, **kwargs)

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
                and not self._credential_failed
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
            try:
                if self._credential_dir is not None:
                    operation = self._credential_operation_task
                    if operation is None or operation.done():
                        profile = next(iter(self.config.account_mapping))
                        with self._credential_commands(profile, None):
                            await self._credential_operation(
                                self._clear_credentials(self._credential_dir)
                            )
                    else:
                        self.logger.warning("Credential file operation remains active at teardown")
            except (Exception, asyncio.CancelledError, OperationCancelled):
                self.logger.warning(
                    "Final credential cleanup failed; continuing container teardown"
                )
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
