"""Pure helpers for the integration suite, unit-tested in ``tests/test_integration_helpers.py``.

They live outside ``conftest.py`` because everything collected under
``tests/integration`` is marked ``integration`` and deselected by default; the
unit tests that exercise these helpers must run in ``make test``.

- Opt-in settings and CLI source arguments (``IntegSettings``, ``load_settings``).
- Running the CLI without leaking secrets (``run_aws_bench``, ``Secret``,
  ``SecretEnv``, ``redact_args``, ``scrub``).
- Deploy with guaranteed cleanup (``deployed``).
- Subprocess timeouts derived from the scenario and task budgets
  (``step_timeouts_for``, ``run_timeout_for``) and the expected trial count
  (``resolve_run_tasks``, ``expected_trial_count``).
- Parsing and checking `env show` / `env list` output (``parse_env_show``,
  ``env_show_problems``, ``parse_env_list``, ``env_list_problems``).
"""

from __future__ import annotations

import asyncio
import re
import signal
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest

from aws_bench.account_management.preexisting import load_account_config
from aws_bench.cli.job_config import AwsBenchJobConfig
from aws_bench.dataset.config import AwsBenchDatasetConfig
from aws_bench.dataset.models import ScriptType
from aws_bench.dataset.task_config import AwsBenchTaskConfig
from aws_bench.scenario.config import ScenarioManifest

REDACTED = "***"

# --------------------------------------------------------------------------- #
# Opt-in settings
# --------------------------------------------------------------------------- #

OPT_IN_VAR = "AWS_BENCH_INTEG"
ACCOUNT_CONFIG_VAR = "AWS_BENCH_INTEG_ACCOUNT_CONFIG"
ENV_NAME_VAR = "AWS_BENCH_INTEG_ENV_NAME"
DATASET_VAR = "AWS_BENCH_INTEG_DATASET"
# Optional comma-separated `run -i` globs, e.g. "*list-ec-instances-all-regions".
# Harbor matches the task directory name (local/git) or "org/name" (registry);
# a leading "*" covers both.
TASKS_VAR = "AWS_BENCH_INTEG_TASKS"
# Optional local aws-bench-datasets checkout (has scenarios/ and tasks/). When
# set, it replaces `-d <dataset>`, so unreleased dataset changes can be tested.
DATASET_PATH_VAR = "AWS_BENCH_INTEG_DATASET_PATH"
# Scenario to use with DATASET_PATH_VAR (ignored otherwise).
SCENARIO_VAR = "AWS_BENCH_INTEG_SCENARIO"
# Extra paths (comma-separated, relative to HOME) to symlink into the scratch
# HOME, e.g. config/cache directories a credential_process helper needs.
HOME_LINKS_VAR = "AWS_BENCH_INTEG_HOME_LINKS"
# Optional positive integer: the number of trials the run must plan. It is an
# expectation independent of the selection, so it catches a selection that
# resolves fewer (or more) tasks than intended.
EXPECTED_TRIALS_VAR = "AWS_BENCH_INTEG_EXPECTED_TRIALS"
DEFAULT_DATASET = "aws-bench-quickstart"
DEFAULT_SCENARIO = "ec2-multiregion"


@dataclass(frozen=True)
class IntegSettings:
    """Validated opt-in settings for one integration session."""

    run_id: str
    env_name: str
    dataset: str
    account_config: Path
    task_globs: tuple[str, ...] = ()
    dataset_path: Path | None = None
    scenario: str = DEFAULT_SCENARIO
    expected_trials: int | None = None

    def env_source_args(self) -> list[str]:
        """Scenario-source arguments for `env init|setup|cleanup`.

        Local mode: `--scenario-path` must be a directory *containing* scenario
        directories (each immediate subdirectory with a scenario.toml counts), so
        it points at `<checkout>/scenarios`, narrowed by `--include-scenarios`.
        """
        if self.dataset_path is None:
            return ["-d", self.dataset]
        return [
            "--scenario-path", str(self.dataset_path / "scenarios"),
            "--include-scenarios", self.scenario,
        ]  # fmt: skip

    def run_source_args(self) -> list[str]:
        """Dataset-source arguments for `run`.

        Local mode needs both a tasks dir (`-p`) and a scenarios dir. `run` loads
        every scenario under `--scenario-path` but only uses those its tasks
        reference, so `-p <checkout>/tasks/<scenario>` limits it to one.
        """
        if self.dataset_path is None:
            return ["-d", self.dataset]
        return [
            "--scenario-path", str(self.dataset_path / "scenarios"),
            "-p", str(self.dataset_path / "tasks" / self.scenario),
        ]  # fmt: skip

    def task_filter_args(self) -> list[str]:
        """`-i <glob>` for each task glob (empty: every task of the dataset)."""
        return [arg for g in self.task_globs for arg in ("-i", g)]

    def run_dataset_config(self, *, download_dir: Path | None = None) -> AwsBenchDatasetConfig:
        """The dataset config that `run` builds from ``run_source_args`` + ``task_filter_args``.

        Built with the CLI's own constructor (``AwsBenchDatasetConfig.from_cli_args``),
        so task selection matches what `run` resolves. ``download_dir`` sets
        where registry tasks are fetched to (the CLI's cache under its ``HOME``).
        """
        include = list(self.task_globs) or None
        if self.dataset_path is None:
            cfg = AwsBenchDatasetConfig.from_cli_args(
                dataset=self.dataset, include_task_names=include
            )
        else:
            cfg = AwsBenchDatasetConfig.from_cli_args(
                scenario_path=self.dataset_path / "scenarios",
                path=self.dataset_path / "tasks" / self.scenario,
                include_task_names=include,
            )
        if download_dir is not None:
            cfg.download_dir = download_dir
        return cfg


def load_settings(environ: Mapping[str, str], run_id: str | None = None) -> IntegSettings:
    """Validate the opt-in variables in ``environ`` and return the settings.

    Skips (``pytest.skip``) when not opted in or the account config is missing;
    fails (``pytest.fail``) on contradictory or broken configuration.
    """
    if environ.get(OPT_IN_VAR) != "1":
        pytest.skip(f"integration tests are opt-in: set {OPT_IN_VAR}=1 (see `make integration`)")
    raw = environ.get(ACCOUNT_CONFIG_VAR)
    if not raw:
        pytest.skip(f"{ACCOUNT_CONFIG_VAR} must point to a pre-existing account config YAML")
    config = Path(raw).expanduser().resolve()
    if not config.is_file():
        pytest.skip(f"{ACCOUNT_CONFIG_VAR}={raw} is not a file")
    # Pre-existing mode requires --env-name to equal the config's `name`
    # (TestEnvironmentNotFoundError otherwise), so it cannot carry a per-run
    # suffix. Isolation between runs comes from the redirected HOME and unique
    # job names; AWS-level isolation needs one account per concurrent runner.
    env_name = load_account_config(config).name
    override = environ.get(ENV_NAME_VAR)
    if override and override != env_name:
        pytest.fail(
            f"{ENV_NAME_VAR}={override!r} does not match the account config's "
            f"name {env_name!r}; pre-existing mode requires them to be equal. "
            f"Unset {ENV_NAME_VAR} or change `name` in {config}.",
            pytrace=False,
        )
    dataset_path: Path | None = None
    scenario = environ.get(SCENARIO_VAR) or DEFAULT_SCENARIO
    raw_dataset_path = environ.get(DATASET_PATH_VAR)
    if raw_dataset_path:
        if environ.get(DATASET_VAR):
            pytest.fail(f"Set either {DATASET_VAR} or {DATASET_PATH_VAR}, not both.", pytrace=False)
        dataset_path = Path(raw_dataset_path).expanduser().resolve()
        for required in (
            dataset_path / "scenarios" / scenario / "scenario.toml",
            dataset_path / "tasks" / scenario,
        ):
            if not required.exists():
                pytest.fail(
                    f"{DATASET_PATH_VAR}={raw_dataset_path}: missing {required}. Point it at "
                    f"an aws-bench-datasets checkout and set {SCENARIO_VAR} if needed.",
                    pytrace=False,
                )
    expected_trials: int | None = None
    raw_expected = environ.get(EXPECTED_TRIALS_VAR, "").strip()
    if raw_expected:
        try:
            expected_trials = int(raw_expected)
        except ValueError:
            expected_trials = 0
        if expected_trials < 1:
            pytest.fail(
                f"{EXPECTED_TRIALS_VAR}={raw_expected!r} must be a positive integer "
                "(the number of trials the run must plan), or unset.",
                pytrace=False,
            )
    return IntegSettings(
        run_id=run_id or uuid.uuid4().hex[:8],
        env_name=env_name,
        dataset=environ.get(DATASET_VAR, DEFAULT_DATASET),
        account_config=config,
        task_globs=tuple(g.strip() for g in environ.get(TASKS_VAR, "").split(",") if g.strip()),
        dataset_path=dataset_path,
        scenario=scenario,
        expected_trials=expected_trials,
    )


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #

# Flags whose value is an env assignment KEY=VALUE (`run --ve/--ae`).
_ENV_FLAGS = frozenset({"--ve", "--verifier-env", "--ae", "--agent-env"})
# Key-name fragments that mark a value as secret (case-insensitive).
_SECRET_KEY_RE = re.compile(r"TOKEN|SECRET|PASSWORD|KEY|BEARER|CREDENTIAL", re.IGNORECASE)
# Values of secret-named environment variables shorter than this are not
# scrubbed, so a short flag value ("1", "true") does not mangle ordinary output.
# A secret that short would appear unmasked if the CLI printed it.
MIN_ENV_SECRET_LEN = 8


class Secret:
    """A secret string whose ``repr``/``str`` never reveal the value.

    pytest prints the arguments of every frame in a failure traceback
    (``tok = '...'``); wrapping a secret in ``Secret`` keeps it out of that
    output. ``run_aws_bench`` accepts ``Secret`` arguments, so the raw value
    is never an argument of any frame between the test and the subprocess.
    Use ``.value`` where the real value is needed.
    """

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        """Wrap ``value``."""
        self.value = value

    def __repr__(self) -> str:
        """Return a masked representation."""
        return f"Secret({REDACTED!r})"

    __str__ = __repr__


class SecretEnv(Mapping[str, str]):
    """Read-only environment mapping whose ``repr`` shows no values.

    The CLI's environment (``cli_env`` fixture) carries the caller's AWS
    credentials and the Bedrock token. pytest renders fixture arguments with
    ``repr`` when a fixture fails, so the mapping masks itself. Values whose
    names look secret are also scrubbed from CLI output (``env_secret_values``).
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, str]) -> None:
        """Copy ``data``."""
        self._data = dict(data)

    def __getitem__(self, key: str) -> str:
        """Return the value of ``key``."""
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the variable names."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of variables."""
        return len(self._data)

    def __repr__(self) -> str:
        """Return a representation without values."""
        return f"SecretEnv(<{len(self._data)} variables, values hidden>)"

    __str__ = __repr__


@dataclass(frozen=True)
class CliResult:
    """Completed ``aws-bench`` invocation. ``args`` are redacted, output is scrubbed."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


def _mask_assignment(arg: str) -> str:
    key, sep, _ = arg.partition("=")
    return f"{key}={REDACTED}" if sep else REDACTED


def _is_secret_assignment(arg: str) -> bool:
    key, sep, value = arg.partition("=")
    return bool(sep and value and _SECRET_KEY_RE.search(key))


def redact_args(args: Sequence[str]) -> list[str]:
    """Return ``args`` with secret values replaced by ``***``; keys stay visible.

    Masked: the value of every ``--ve``/``--ae`` (and long-form) assignment,
    as ``KEY=VALUE`` or ``--ve=KEY=VALUE``; and any ``KEY=VALUE`` whose key
    contains TOKEN, SECRET, PASSWORD, KEY, BEARER or CREDENTIAL.
    """
    out: list[str] = []
    after_env_flag = False
    for arg in args:
        if after_env_flag:
            out.append(_mask_assignment(arg))
            after_env_flag = False
            continue
        flag, sep, rest = arg.partition("=")
        if flag in _ENV_FLAGS:
            if sep:
                out.append(f"{flag}={_mask_assignment(rest)}")
            else:
                out.append(arg)
                after_env_flag = True
            continue
        out.append(_mask_assignment(arg) if _is_secret_assignment(arg) else arg)
    return out


def secret_values(args: Sequence[str]) -> list[str]:
    """Return the raw values that ``redact_args`` masks (non-empty, longest first)."""
    values: set[str] = set()
    for raw, masked in zip(args, redact_args(args), strict=True):
        if raw == masked:
            continue
        # The masked form keeps everything up to the last "=" before "***".
        prefix = masked[: -len(REDACTED)]
        value = raw[len(prefix) :]
        if value:
            values.add(value)
    return sorted(values, key=len, reverse=True)


def env_secret_values(env: Mapping[str, str]) -> list[str]:
    """Values of secret-named ``env`` variables at least ``MIN_ENV_SECRET_LEN`` (8) long.

    Values shorter than 8 characters are deliberately left out, so they are not
    scrubbed from CLI output: replacing short strings such as ``1`` or ``true``
    would mangle ordinary output. A secret-named variable holding a real secret
    shorter than 8 characters is therefore not masked.
    """
    values = {
        value
        for name, value in env.items()
        if _SECRET_KEY_RE.search(name) and len(value) >= MIN_ENV_SECRET_LEN
    }
    return sorted(values, key=len, reverse=True)


def scrub(text: str, secrets: Sequence[str]) -> str:
    """Replace every occurrence of each secret in ``text`` with ``***``."""
    for secret in sorted(secrets, key=len, reverse=True):
        if secret:
            text = text.replace(secret, REDACTED)
    return text


def _to_text(data: str | bytes | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


# --------------------------------------------------------------------------- #
# Running the CLI
# --------------------------------------------------------------------------- #

# Program prefix for every call; the args follow it.
CLI_PROGRAM: tuple[str, ...] = (sys.executable, "-m", "aws_bench.cli.main")
# After a timeout the CLI gets SIGTERM (its cooperative shutdown stops
# containers and persists results) and this long to exit before SIGKILL.
TERM_GRACE_SEC = 60.0
_TAIL = 4000


class _Command:
    """argv, environment and scrub list of one CLI call; ``repr`` hides all of them.

    Built once from the caller's args, so the raw command line exists only
    inside this object and never as an argument or local of a frame that
    pytest could render.
    """

    __slots__ = ("_argv", "_env", "_secrets", "shown")

    def __init__(self, args: Sequence[str | Secret], env: Mapping[str, str]) -> None:
        raw = [a.value if isinstance(a, Secret) else a for a in args]
        shown = redact_args(raw)
        secrets = set(secret_values(raw))
        for i, arg in enumerate(args):
            if isinstance(arg, Secret):
                shown[i] = _mask_assignment(arg.value)
                _, sep, value = arg.value.partition("=")
                secrets.update(v for v in (arg.value, value if sep else "") if v)
        secrets.update(env_secret_values(env))
        self._argv = [*CLI_PROGRAM, *raw]
        self._env = dict(env)
        self._secrets = sorted(secrets, key=len, reverse=True)
        self.shown = tuple(shown)

    def __repr__(self) -> str:
        return "_Command(<hidden>)"

    def scrub(self, data: str | bytes | None) -> str:
        return scrub(_to_text(data), self._secrets)

    def start(self, cwd: Path) -> subprocess.Popen[str]:
        return subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            self._argv,
            cwd=cwd,
            env=self._env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )


@dataclass(frozen=True)
class _Outcome:
    """What happened to one call. Output is already scrubbed."""

    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    problem: str | None = None


def _stop(proc: subprocess.Popen[str], grace: float) -> tuple[str, str | None, str | None]:
    """SIGTERM ``proc``, wait up to ``grace`` seconds, then SIGKILL; return (how, out, err)."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            out, err = proc.communicate(timeout=grace)
            return "the CLI exited after SIGTERM", out, err
        except subprocess.TimeoutExpired:
            how = f"SIGKILL sent after a {grace:g}s SIGTERM grace period"
        except Exception:  # noqa: BLE001 - output is unreadable; still stop the child
            how = "SIGKILL sent"
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        proc.kill()
    else:
        how = "the CLI had already exited"
    try:
        out, err = proc.communicate(timeout=grace)
    except Exception:  # noqa: BLE001 - e.g. a grandchild keeps a pipe open
        proc.wait()
        out = err = None
    return how, out, err


def _wait(cmd: _Command, cwd: Path, timeout: float | None, grace: float) -> _Outcome:
    try:
        proc = cmd.start(cwd)
    except Exception as e:  # noqa: BLE001 - reported without its (argv-bearing) message
        return _Outcome(problem=f"could not start: {type(e).__name__}")
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        how, out, err = _stop(proc, grace)
        return _Outcome(
            proc.returncode,
            cmd.scrub(out),
            cmd.scrub(err),
            problem=f"timed out after {timeout:g}s ({how})",
        )
    except Exception as e:  # noqa: BLE001 - reported without its message
        how, _, _ = _stop(proc, grace)
        return _Outcome(problem=f"failed while reading its output: {type(e).__name__} ({how})")
    except BaseException:
        _stop(proc, grace)
        raise
    return _Outcome(proc.returncode, cmd.scrub(out), cmd.scrub(err))


def _execute(
    cmd: _Command, cwd: Path, timeout: float | None, grace: float
) -> _Outcome | BaseException:
    """Run ``cmd``; an interrupt is returned (child already stopped), never raised here."""
    try:
        return _wait(cmd, cwd, timeout, grace)
    except BaseException as e:  # KeyboardInterrupt, SystemExit: re-raised by the caller
        return e


def _tails(outcome: _Outcome) -> str:
    return f"--- stdout ---\n{outcome.stdout[-_TAIL:]}\n--- stderr ---\n{outcome.stderr[-_TAIL:]}"


def run_aws_bench(
    args: Sequence[str | Secret],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float | None,
    check: bool = True,
    term_grace: float = TERM_GRACE_SEC,
) -> CliResult:
    """Run ``aws-bench <args>`` as a subprocess; fail the test without leaking secrets.

    Secret values are those ``redact_args`` masks, every ``Secret`` argument,
    and every ``env`` value whose name looks secret (``env_secret_values``).
    They are masked in the returned ``CliResult`` and in every failure message.

    - Non-zero exit (with ``check``), timeout, a start failure or any other
      ``Exception`` from the subprocess layer: ``pytest.fail(..., pytrace=False)``
      with redacted args and scrubbed output. ``pytrace=False`` keeps pytest
      from printing traceback frames, and the failure is raised outside any
      ``except`` block, so no exception carrying the raw argv is chained.
    - Timeout: the CLI gets SIGTERM, ``term_grace`` seconds to shut down, then
      SIGKILL. ``timeout=None`` sets no deadline: the call waits for the CLI.
    - ``KeyboardInterrupt``/``SystemExit``: the child is stopped the same way and
      the interrupt is re-raised with its traceback and context cleared.

    ``args`` and ``env`` are deleted from this frame once wrapped, so even a
    ``--full-trace``/``-l`` report of this frame shows no raw values. Frames
    of callers still show their own arguments: pass secrets as ``Secret``.
    """
    cmd = _Command(args, env)
    # pytest renders a frame's arguments (and with -l its locals) for
    # --full-trace or unexpected errors; it skips names missing from the frame.
    del args, env
    outcome = _execute(cmd, cwd, timeout, term_grace)
    shown = " ".join(cmd.shown)
    if isinstance(outcome, BaseException):
        outcome.__context__ = None
        outcome.__cause__ = None
        raise outcome.with_traceback(None) from None
    if outcome.problem is not None:
        pytest.fail(f"`aws-bench {shown}` {outcome.problem}\n{_tails(outcome)}", pytrace=False)
    assert outcome.returncode is not None
    result = CliResult(cmd.shown, outcome.returncode, outcome.stdout, outcome.stderr)
    if check and result.returncode != 0:
        pytest.fail(
            f"`aws-bench {shown}` exited {result.returncode}\n{_tails(outcome)}", pytrace=False
        )
    return result


# --------------------------------------------------------------------------- #
# Deploy with guaranteed cleanup
# --------------------------------------------------------------------------- #

RunCli = Callable[..., CliResult]
_LEAK_NOTICE = "env cleanup failed (resources may be left behind)"


def _attempt_cleanup(
    run_cli: RunCli,
    cleanup_args: Sequence[str | Secret],
    cleanup_timeout: Callable[[], float | None],
) -> str | None:
    """Run cleanup once; return a description of its failure, or None if it succeeded."""
    try:
        result = run_cli(*cleanup_args, timeout=cleanup_timeout(), check=False)
    except (Exception, pytest.fail.Exception) as e:  # noqa: BLE001 - reported, not raised here
        # run_aws_bench failures are already redacted; anything else is named only.
        detail = str(e) if isinstance(e, pytest.fail.Exception) else type(e).__name__
        return detail
    if result.returncode != 0:
        return f"exit {result.returncode}\n{result.output[-_TAIL:]}"
    return None


@contextmanager
def deployed(
    run_cli: RunCli,
    *,
    init_args: Sequence[str | Secret],
    setup_args: Sequence[str | Secret],
    cleanup_args: Sequence[str | Secret],
    init_timeout: float,
    setup_timeout: Callable[[], float],
    cleanup_timeout: Callable[[], float | None],
) -> Iterator[None]:
    """Run `env init` and `env setup`, yield, then always attempt `env cleanup` once.

    The timeouts are callables because they are read from manifests that
    `env init` caches. Outcomes:

    - Everything succeeds: returns normally.
    - Only cleanup fails: ``pytest.fail`` with the leak notice.
    - Init, setup or the body raised (failure, error, timeout, interrupt):
      cleanup is still attempted and the original exception is re-raised. A
      cleanup failure is appended to its message (``pytest.fail`` failures) or
      attached as a note (other exceptions) instead of replacing it.

    Used by the ``deployed_env`` fixture around its ``yield``. pytest does not
    throw a test's own failure into a fixture, so for a failing test the
    fixture exits normally and a cleanup failure is reported as a separate
    teardown error.
    """
    primary: BaseException | None = None
    try:
        run_cli(*init_args, timeout=init_timeout)
        run_cli(*setup_args, timeout=setup_timeout())
        yield
    except BaseException as e:  # re-raised below, after cleanup
        primary = e
    problem = _attempt_cleanup(run_cli, cleanup_args, cleanup_timeout)
    if primary is None:
        if problem is not None:
            pytest.fail(f"{_LEAK_NOTICE}:\n{problem}", pytrace=False)
        return
    if problem is not None:
        notice = f"{_LEAK_NOTICE}:\n{problem}"
        if isinstance(primary, pytest.fail.Exception):
            # A pytrace=False failure is rendered from its message only.
            primary.msg = f"{primary.msg}\n\nAlso, {notice}"
        else:
            primary.add_note(notice)
    raise primary


# --------------------------------------------------------------------------- #
# Timeouts derived from scenario and task budgets
# --------------------------------------------------------------------------- #

# Seconds added to the scenario's own budgets for framework work outside them.
# Setup: CDK bootstrap and the post-deploy baseline snapshot. Cleanup: the
# framework sweep (CloudFormation deletes, Cloud Control deletes, drift scan).
SETUP_MARGIN_SEC = 1800.0
CLEANUP_MARGIN_SEC = 1800.0
# Setup timeout used when no manifest can be read. At least the ec2-multiregion
# budgets (build 600 + deploy 1800) plus the margin above. Cleanup has no such
# fallback: without a manifest it runs with no deadline from the test (any
# guess could stop it inside a scenario's own budget and leave resources).
FALLBACK_SETUP_TIMEOUT_SEC = 4200.0
# `run`: framework work outside the per-trial budgets (job start, agent setup,
# image pulls, pre-run environment verification, per-trial resets).
RUN_MARGIN_SEC = 1800.0
# Budget assumed for one task whose task.toml is unreadable or sets no agent
# timeout. At least one quickstart task (build 1200 + agent 600 + verifier 240
# + pre_invoke 240).
FALLBACK_TASK_BUDGET_SEC = 3600.0
# Used when no task.toml can be read. At least the nine quickstart tasks
# (9 x 2040 + 240 = 18600) plus RUN_MARGIN_SEC.
FALLBACK_RUN_TIMEOUT_SEC = 21600.0


@dataclass(frozen=True)
class StepTimeouts:
    """Subprocess timeouts (seconds) for ``env setup`` and ``env cleanup``.

    ``cleanup`` is None when no scenario manifest could be read: cleanup then
    runs without a deadline from the test. The CLI bounds only its container
    build and cleanup-phase script; the rest of ``env cleanup`` has no declared
    budget, so a hung cleanup needs a manual interrupt.
    """

    setup: float
    cleanup: float | None
    source: str

    def describe(self) -> str:
        if self.cleanup is None:
            cleanup = "env cleanup without a deadline (interrupt manually if it hangs)"
        else:
            cleanup = f"env cleanup {self.cleanup:g}s"
        return f"env setup {self.setup:g}s, {cleanup} ({self.source})"


def find_scenario_manifests(*, dataset_path: Path | None, scenario: str, home: Path) -> list[Path]:
    """Return the scenario.toml files whose budgets apply to this run.

    Local mode (``dataset_path`` set): ``<dataset_path>/scenarios/<scenario>``.
    Registry mode: every scenario aws-bench cached under the CLI's ``HOME``
    (``~/.aws-bench/cache/scenarios/<id>/<name>/``), which `env init` populates.
    The scratch HOME is per session, so the cache holds only this run's scenarios.
    """
    if dataset_path is not None:
        candidates = [dataset_path / "scenarios" / scenario / "scenario.toml"]
    else:
        cache = home / ".aws-bench" / "cache" / "scenarios"
        candidates = sorted(cache.glob("*/*/scenario.toml"))
    return [p for p in candidates if p.is_file()]


def step_timeouts_for(manifest_paths: Sequence[Path]) -> StepTimeouts:
    """Derive step timeouts from the scenario manifests' own budgets.

    Per scenario:
    setup = environment.build_timeout_sec + deploy.timeout_sec
    cleanup = environment.build_timeout_sec + cleanup.timeout_sec (the cleanup
    command builds the scenario container before running the cleanup script).
    With several scenarios the per-scenario budgets are summed, a safe upper
    bound whatever the scenario concurrency. SETUP_MARGIN_SEC /
    CLEANUP_MARGIN_SEC are added once. The test passes no
    ``--timeout-multiplier``, so the budgets are used unscaled. Unreadable
    manifests are skipped. With none readable, setup uses
    FALLBACK_SETUP_TIMEOUT_SEC and cleanup gets no deadline (``None``).
    """
    manifests: list[tuple[Path, ScenarioManifest]] = []
    skipped: list[str] = []
    for path in manifest_paths:
        try:
            manifests.append((path, ScenarioManifest.model_validate_toml(path.read_bytes())))
        except (OSError, ValueError) as e:
            skipped.append(f"{path}: {type(e).__name__}")
    note = f"; skipped unreadable {', '.join(skipped)}" if skipped else ""
    if not manifests:
        return StepTimeouts(
            setup=FALLBACK_SETUP_TIMEOUT_SEC,
            cleanup=None,
            source=f"fallback: no readable scenario.toml{note}",
        )
    setup = sum(m.environment.build_timeout_sec + m.deploy.timeout_sec for _, m in manifests)
    cleanup = sum(m.environment.build_timeout_sec + m.cleanup.timeout_sec for _, m in manifests)
    return StepTimeouts(
        setup=setup + SETUP_MARGIN_SEC,
        cleanup=cleanup + CLEANUP_MARGIN_SEC,
        source="from " + ", ".join(str(p) for p, _ in manifests) + note,
    )


@dataclass(frozen=True)
class RunTimeout:
    """Subprocess timeout (seconds) for ``run``."""

    seconds: float
    source: str

    def describe(self) -> str:
        return f"run {self.seconds:g}s ({self.source})"


def task_budget_sec(cfg: AwsBenchTaskConfig, task_dir: Path) -> float | None:
    """One trial's authored worst case, or None if the agent timeout is unset (unbounded).

    environment.build_timeout_sec + agent.timeout_sec + verifier.timeout_sec,
    plus pre_invoke / post_invoke timeout_sec for each phase the task declares
    in task.toml or ships a script directory for.
    """
    if cfg.agent.timeout_sec is None:
        return None
    total = cfg.environment.build_timeout_sec + cfg.agent.timeout_sec + cfg.verifier.timeout_sec
    phases = ((ScriptType.PRE_INVOKE, cfg.pre_invoke), (ScriptType.POST_INVOKE, cfg.post_invoke))
    for phase, budget in phases:
        if phase.value in cfg.model_fields_set or (task_dir / phase.value).is_dir():
            total += budget.timeout_sec
    return total


def run_timeout_for(task_dirs: Sequence[Path]) -> RunTimeout:
    """Derive the ``run`` timeout from the selected tasks' own budgets.

    Sum of ``task_budget_sec`` over all tasks (as if trials ran one after
    another, a safe upper bound whatever the trial concurrency) plus
    RUN_MARGIN_SEC. A task whose task.toml is unreadable or has no agent
    timeout counts FALLBACK_TASK_BUDGET_SEC. With no task at all,
    FALLBACK_RUN_TIMEOUT_SEC is used. Budgets are unscaled: the test passes
    no timeout multipliers.
    """
    total = 0.0
    fallback: list[str] = []
    for task_dir in task_dirs:
        try:
            cfg = cast(
                AwsBenchTaskConfig,
                AwsBenchTaskConfig.model_validate_toml((task_dir / "task.toml").read_text()),
            )
            budget = task_budget_sec(cfg, task_dir)
        except (OSError, ValueError) as e:
            fallback.append(f"{task_dir.name}: {type(e).__name__}")
            budget = None
        else:
            if budget is None:
                fallback.append(f"{task_dir.name}: no agent timeout")
        total += FALLBACK_TASK_BUDGET_SEC if budget is None else budget
    note = f"; {FALLBACK_TASK_BUDGET_SEC:g}s assumed for {', '.join(fallback)}" if fallback else ""
    if not task_dirs:
        return RunTimeout(FALLBACK_RUN_TIMEOUT_SEC, "fallback: no task.toml to read")
    return RunTimeout(
        total + RUN_MARGIN_SEC,
        f"sum of {len(task_dirs)} task budget(s) + {RUN_MARGIN_SEC:g}s margin{note}",
    )


@dataclass(frozen=True)
class RunTasks:
    """Tasks `run` will select: the expected trial count and their directories."""

    expected_trials: int
    task_dirs: tuple[Path, ...]
    note: str = ""


def resolve_run_tasks(cfg: AwsBenchDatasetConfig) -> RunTasks:
    """Resolve the tasks of ``cfg`` with aws-bench's own resolver.

    ``expected_trials`` = number of selected tasks x the `run` default
    ``n_attempts`` (the test passes no ``-k``). Resolution errors propagate:
    without an expected count the reward check cannot be trusted. Task
    directories (registry tasks are fetched to ``cfg.download_dir``) are only
    used for the timeout; if they cannot be materialized they are left empty
    and ``note`` says why.
    """
    n_attempts = int(AwsBenchJobConfig.model_fields["n_attempts"].default)

    async def _resolve() -> RunTasks:
        configs = await cfg.get_task_configs()
        expected = len(configs) * n_attempts
        try:
            pairs = await cfg.get_tasks(configs)
        except Exception as e:  # noqa: BLE001 - the timeout falls back instead
            return RunTasks(expected, (), f"task directories unavailable: {type(e).__name__}")
        return RunTasks(expected, tuple(task.paths.task_dir for _, task in pairs))

    return asyncio.run(_resolve())


def expected_trial_count(settings: IntegSettings, *, resolved: int) -> int:
    """Return the trial count the job must plan.

    Without ``AWS_BENCH_INTEG_EXPECTED_TRIALS`` this is ``resolved`` (from
    ``resolve_run_tasks``): the check then confirms the job ran every task the
    selection resolves, but cannot detect a selection that is itself too small.
    With the variable set, ``resolved`` must equal it (``pytest.fail``
    otherwise, before anything runs) and the variable's value is returned.
    """
    expected = settings.expected_trials
    if expected is None:
        return resolved
    if resolved != expected:
        globs = list(settings.task_globs) or "all tasks"
        pytest.fail(
            f"{EXPECTED_TRIALS_VAR}={expected}, but the selection resolves {resolved} "
            f"trial(s) (task globs: {globs}). Fix the selection or the expectation.",
            pytrace=False,
        )
    return expected


# --------------------------------------------------------------------------- #
# `env show` / `env list` output
# --------------------------------------------------------------------------- #

# Console width the CLI subprocess gets (``COLUMNS``, which Rich reads). Rich
# assumes 80 columns on a pipe and then truncates long panel titles and table
# cells, which would hide the fields the tests parse. 200 columns fit the
# `ec2-multiregion` panels the unit tests render; a very long env or scenario
# name can still be truncated, and the parsers then fail (the header or account
# panel is not recognised) rather than pass.
CLI_COLUMNS = 200
# Removed from the CLI's environment. ``FORCE_COLOR`` and ``TTY_COMPATIBLE``
# make Rich treat a pipe as a terminal; with ``TERM`` dumb or unknown, Rich then
# ignores ``COLUMNS`` and uses 80 columns. Rich does not read ``PY_COLORS`` or
# ``CLICOLOR_FORCE``; they are removed defensively.
COLOR_FORCING_VARS = ("FORCE_COLOR", "TTY_COMPATIBLE", "PY_COLORS", "CLICOLOR_FORCE")
# OU id printed in pre-existing mode: `env show` takes it from
# ``PreexistingEnvironmentConfig.to_test_environment``, `env list` hard-codes
# it in ``list_envs``. Only ``env_list_problems`` checks it; the OU row of the
# `env show` header is deliberately not checked.
PREEXISTING_OU_ID = "preexisting"
# Stack statuses that mean a deploy finished successfully.
HEALTHY_STACK_STATUSES = frozenset({"CREATE_COMPLETE", "UPDATE_COMPLETE", "IMPORT_COMPLETE"})

# CSI sequences (colours, styles) and OSC sequences (hyperlinks). Rich emits
# them on a pipe too when FORCE_COLOR is set, as it is in CI.
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_PANEL_TITLE_RE = re.compile(r"^╭─ (?P<title>.+?) ─*╮$")
_PANEL_BODY_RE = re.compile(r"^│ (?P<body>.*?)\s*│$")
# `render_account_state` title: "<scenario> / <tag>  ·  <account id>  ·  <status>".
_ACCOUNT_TITLE_RE = re.compile(
    r"^(?P<scenario>\S+) / (?P<tag>\S+)\s+·\s+(?P<account_id>\d{12})\s+·\s+(?P<status>\S+)$"
)
_ACCOUNT_COUNT_RE = re.compile(r"^Scenario accounts\s+(?P<count>\d+)$")
_STACK_STATUS_RE = re.compile(r"^[A-Z]+(?:_[A-Z]+)+$")
_REGION_RE = re.compile(r"^[a-z]{2}(?:-[a-z]+)+-\d+$")


def plain(text: str) -> str:
    """Return ``text`` without ANSI escape sequences."""
    return _ANSI_RE.sub("", text)


def cli_display_env(env: Mapping[str, str]) -> dict[str, str]:
    """Return ``env`` with ``COLUMNS=CLI_COLUMNS`` and no ``COLOR_FORCING_VARS``.

    Without colour forcing, Rich sees a pipe as a non-terminal and honours
    ``COLUMNS`` whatever ``TERM`` is.
    """
    out = {k: v for k, v in env.items() if k not in COLOR_FORCING_VARS}
    out["COLUMNS"] = str(CLI_COLUMNS)
    return out


def build_cli_env(
    environ: Mapping[str, str], *, account_config: Path, scratch_home: Path
) -> dict[str, str]:
    """Return the CLI subprocess environment built from ``environ`` (see ``cli_env``).

    Symlinks ``.aws``, ``.docker`` and the ``HOME_LINKS_VAR`` paths of the real
    ``HOME`` into ``scratch_home``, points ``HOME`` at ``scratch_home``, sets
    ``AWSBENCH_ACCOUNT_CONFIG`` and applies ``cli_display_env``.
    """
    env = dict(environ)
    real_home = Path(env.get("HOME", str(Path.home())))
    extra = [p.strip() for p in env.get(HOME_LINKS_VAR, "").split(",") if p.strip()]
    for name in (".aws", ".docker", *extra):
        if (real_home / name).exists():
            link = scratch_home / name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(real_home / name, target_is_directory=True)
    env["HOME"] = str(scratch_home)
    env["AWSBENCH_ACCOUNT_CONFIG"] = str(account_config)
    return cli_display_env(env)


@dataclass(frozen=True)
class ShownStack:
    """One row of an account's stack table in `env show`."""

    name: str
    status: str
    region: str


@dataclass(frozen=True)
class ShownAccount:
    """One account panel of `env show`."""

    scenario: str
    tag: str
    account_id: str
    status: str
    stacks: tuple[ShownStack, ...] = ()


@dataclass(frozen=True)
class EnvShow:
    """What `env show` printed: header panel title and count, then one panel per account."""

    env_name: str | None
    scenario_accounts: int | None
    accounts: tuple[ShownAccount, ...]


def parse_env_show(output: str) -> EnvShow:
    """Parse `env show` output (``render_env_header`` + ``render_account_state`` panels).

    The first panel whose title is not an account title is the header: its
    title is the env name, and its "Scenario accounts" row the account count.
    Each account panel contributes the stack rows (name, status, region) of
    its stack table. Lines outside panels (e.g. log lines) are ignored.
    """
    env_name: str | None = None
    count: int | None = None
    accounts: list[ShownAccount] = []
    stacks: list[ShownStack] = []
    panel: str | None = None  # "header", "account" or None (outside / unknown panel)

    def close_account() -> None:
        if panel == "account":
            accounts[-1] = replace(accounts[-1], stacks=tuple(stacks))

    for line in plain(output).splitlines():
        line = line.rstrip()
        title = _PANEL_TITLE_RE.match(line)
        if title:
            close_account()
            stacks = []
            account = _ACCOUNT_TITLE_RE.match(title["title"])
            if account:
                fields = account["scenario"], account["tag"], account["account_id"]
                accounts.append(ShownAccount(*fields, account["status"]))
                panel = "account"
            elif env_name is None and not accounts:
                env_name = title["title"]
                panel = "header"
            else:
                panel = None
            continue
        if line.startswith("╰"):
            close_account()
            panel = None
            continue
        body = _PANEL_BODY_RE.match(line)
        if body is None or panel is None:
            continue
        text = body["body"].strip()
        if panel == "header":
            found = _ACCOUNT_COUNT_RE.match(text)
            if found:
                count = int(found["count"])
            continue
        cells = re.split(r"\s{2,}", text)
        if len(cells) == 3 and _STACK_STATUS_RE.match(cells[1]) and _REGION_RE.match(cells[2]):
            stacks.append(ShownStack(*cells))
    close_account()
    return EnvShow(env_name, count, tuple(accounts))


def _configured(accounts: Mapping[str, Mapping[str, str]]) -> list[tuple[str, str, str]]:
    return [(s, t, a) for s, tags in accounts.items() for t, a in tags.items()]


def env_show_problems(
    shown: EnvShow, *, env_name: str, accounts: Mapping[str, Mapping[str, str]]
) -> list[str]:
    """Return what is wrong with `env show` output for a deployed pre-existing env.

    ``accounts`` is the account config's ``{scenario: {tag: account_id}}``.
    Expected: a header titled ``env_name`` counting every configured account;
    exactly one panel per configured account, each ``ACTIVE``; and in every
    configured (scenario, tag, account) panel, at least one stack named
    ``<scenario>-*`` in a ``HEALTHY_STACK_STATUSES`` state. Other stacks are
    ignored: failed or rolled-back stacks are not checked (drift is left to
    the `env verify` test), nor are the regions stacks are in, nor the
    header's OU row. The env name, count and account triples come from the
    same account config the CLI reads, so they verify rendering and plumbing.

    The stack check assumes ``env setup`` deployed the scenario's stacks into
    every configured account. aws-bench does not enforce that (each scenario's
    own deploy script decides), but it holds when the account config maps only
    scenarios this run sets up: every scenario in aws-bench-datasets has one
    account tag, and its CDK app names its stacks ``<scenario>-<id>-<region>``.
    ``env show`` lists stacks per account ID, so if two tags shared an account
    ID the check would reduce to per-account evidence (pre-existing configs
    reject that mapping).
    """
    problems: list[str] = []
    if shown.env_name is None:
        problems.append("no environment header panel")
    elif shown.env_name != env_name:
        problems.append(f"header shows environment {shown.env_name!r}, expected {env_name!r}")
    expected = _configured(accounts)
    if shown.scenario_accounts != len(expected):
        problems.append(
            f"header shows {shown.scenario_accounts!r} scenario accounts, expected {len(expected)}"
        )
    seen = [(a.scenario, a.tag, a.account_id) for a in shown.accounts]
    duplicates = sorted({k for k in seen if seen.count(k) > 1})
    if duplicates:
        problems.append(f"accounts shown more than once: {duplicates}")
    missing = sorted(set(expected) - set(seen))
    if missing:
        problems.append(f"configured accounts not shown: {missing}")
    unexpected = sorted(set(seen) - set(expected))
    if unexpected:
        problems.append(f"accounts shown that are not configured: {unexpected}")
    inactive = sorted((a.scenario, a.tag, a.status) for a in shown.accounts if a.status != "ACTIVE")
    if inactive:
        problems.append(f"accounts not ACTIVE: {inactive}")
    without_stack = sorted(
        (a.scenario, a.tag, a.account_id)
        for a in shown.accounts
        if (a.scenario, a.tag, a.account_id) in expected
        and not any(
            s.name.startswith(f"{a.scenario}-") and s.status in HEALTHY_STACK_STATUSES
            for s in a.stacks
        )
    )
    if without_stack:
        problems.append(
            f"configured accounts without a healthy <scenario>-* stack "
            f"({sorted(HEALTHY_STACK_STATUSES)}): {without_stack}"
        )
    return problems


@dataclass(frozen=True)
class ListedEnv:
    """One row of the `env list` table (``render_env_list``); ``accounts`` None if "?"."""

    name: str
    ou_id: str
    accounts: int | None


def parse_env_list(output: str) -> tuple[ListedEnv, ...]:
    """Parse the body rows (``│ name │ ou id │ accounts │``) of the `env list` table."""
    rows: list[ListedEnv] = []
    for line in plain(output).splitlines():
        line = line.strip()
        if not (line.startswith("│") and line.endswith("│")):
            continue
        cells = [c.strip() for c in line[1:-1].split("│")]
        if len(cells) != 3:
            continue
        name, ou_id, count = cells
        rows.append(ListedEnv(name, ou_id, int(count) if count.isdigit() else None))
    return tuple(rows)


def env_list_problems(
    rows: Sequence[ListedEnv], *, env_name: str, accounts: Mapping[str, Mapping[str, str]]
) -> list[str]:
    """Return what is wrong with `env list` output in pre-existing mode.

    Pre-existing mode lists exactly the configured environment: one row named
    ``env_name``, OU id ``PREEXISTING_OU_ID``, counting every configured account.
    """
    if len(rows) != 1:
        return [f"expected exactly one environment row, got {len(rows)}: {list(rows)}"]
    row, problems = rows[0], []
    if row.name != env_name:
        problems.append(f"environment {row.name!r} listed, expected {env_name!r}")
    if row.ou_id != PREEXISTING_OU_ID:
        problems.append(f"OU id {row.ou_id!r}, expected {PREEXISTING_OU_ID!r}")
    expected = len(_configured(accounts))
    if row.accounts != expected:
        problems.append(f"{row.accounts!r} accounts listed, expected {expected}")
    return problems
