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
- Masking account IDs in everything that can reach a failure message, a report
  or a log (``mask_account_ids``, ``fail_masked``, ``mask_problems``,
  ``MaskedDict``, ``MaskedList``, ``CliResult.masked_output``).
- Parsing and checking `env show` / `env list` output (``parse_env_show``,
  ``env_show_problems``, ``parse_env_list``, ``env_list_problems``).
- The independent post-cleanup check: read-only CloudFormation stack listing and
  its verdict (``collect_account_stacks``, ``stack_problems``,
  ``cleanup_output_problems``).
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
from typing import Any, NoReturn, TypeVar, cast

import pytest
from botocore.exceptions import ClientError

from aws_bench.account_management.preexisting import (
    PreexistingEnvironmentConfig,
    load_account_config,
)
from aws_bench.cli.job_config import AwsBenchJobConfig
from aws_bench.dataset.config import AwsBenchDatasetConfig
from aws_bench.dataset.models import ScriptType
from aws_bench.dataset.task_config import AwsBenchTaskConfig
from aws_bench.resource_management.cleanup.models import is_infra_identifier
from aws_bench.scenario.config import ScenarioManifest

REDACTED = "***"
_T = TypeVar("_T")
_V = TypeVar("_V")


# --------------------------------------------------------------------------- #
# Account IDs
# --------------------------------------------------------------------------- #
# Text from AWS, the CLI or the config (names, paths, ARNs, error messages,
# command output) can carry a 12-digit account ID. The suite masks the strings
# it builds for a failure message, a problem list or a status print where it
# builds them. pytest then has little to repeat: its assertion explanation
# prints the operands of the failed ``assert`` (including call arguments), and
# with ``--tb=long``/``-vvv``/``-l`` it also prints frame arguments and locals
# with ``repr``. Objects that hold raw IDs therefore mask their own ``repr``
# (``_masked_repr``, ``MaskedDict``, ``MaskedList``, ``MaskedStr``). The raw
# text stays in the fields, because the parsers need it.
#
# What the suite does not build (exceptions raised by called code, frame
# arguments and locals, captured logs, skip reasons) is masked when pytest
# builds its report: see ``pytest_runtest_makereport`` in ``conftest.py``.
def mask_account_id(account_id: str) -> str:
    """Show only the last four digits of a 12-digit account ID (``********3333``)."""
    if re.fullmatch(r"\d{12}", account_id):
        return "*" * 8 + account_id[-4:]
    return "****"


def mask_account_ids(text: str) -> str:
    """Mask every 12-digit run in ``text`` that looks like an account ID.

    The result has the same length and layout as ``text``, so line numbers and
    columns stay usable. Digit runs of any other length are left alone.
    """
    return re.sub(r"(?<!\d)\d{12}(?!\d)", lambda m: mask_account_id(m.group()), text)


def mask_problems(problems: Sequence[str]) -> list[str]:
    """Mask account IDs in every problem string.

    Stack, scenario, environment and trial names can carry an account ID. The
    strings are masked where they are built, so every place that prints them (an
    assertion message and pytest's own ``assert ... == []`` explanation) shows
    only masked IDs.
    """
    return [mask_account_ids(p) for p in problems]


def account_label(scenario: str, tag: str, account_id: str) -> str:
    """Account description for failure messages: scenario/tag plus the masked ID."""
    return f"{scenario}/{tag} ({mask_account_id(account_id)})"


def fail_masked(message: str) -> NoReturn:
    """``pytest.fail`` with account IDs masked and no traceback."""
    message = mask_account_ids(message)  # also replaces the raw text in this frame's locals
    pytest.fail(message, pytrace=False)


def _masked_repr(cls: type[_T]) -> type[_T]:
    """Class decorator: mask account IDs in the generated ``repr``.

    pytest renders fixture values, frame arguments and (with ``-l``) locals with
    ``repr``, so an object that holds raw names or output must not print them.
    Apply it above ``@dataclass``, which must generate the ``repr`` first::

        @_masked_repr
        @dataclass(frozen=True)
        class Thing: ...
    """
    plain_repr = cls.__repr__

    def __repr__(self: Any) -> str:
        return mask_account_ids(plain_repr(self))

    cls.__repr__ = __repr__  # type: ignore[method-assign,assignment]
    return cls


class MaskedStr(str):
    """A ``str`` whose ``repr`` masks account IDs; the text itself stays raw for parsing.

    pytest renders a frame argument or local with ``repr`` and then cuts it in the
    middle when it is longer than 240 characters, which can leave a few digits of
    an ID that no mask matches any more. Masking inside ``repr`` happens before
    that cut. Slices, ``split`` and ``+`` return a plain ``str``: wrap those that
    carry raw output and can reach a traceback.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return the quoted text with account IDs masked."""
        return mask_account_ids(super().__repr__())


class MaskedDict(dict[str, _V]):
    """A ``dict`` whose ``repr`` masks account IDs (the account config's mapping)."""

    __slots__ = ()

    def __repr__(self) -> str:
        """Return the ``dict``/``list`` representation with account IDs masked."""
        return mask_account_ids(super().__repr__())

    __str__ = __repr__


class MaskedList(list[_T]):
    """A ``list`` whose ``repr`` masks account IDs (CLI arguments, names)."""

    __slots__ = ()

    def __repr__(self) -> str:
        """Return the ``dict``/``list`` representation with account IDs masked."""
        return mask_account_ids(super().__repr__())

    __str__ = __repr__


def _load_config(path: Path) -> PreexistingEnvironmentConfig:
    """Load the account config; fail the test with a masked message if that is not possible.

    A validation error quotes the value it rejected, and a missing-file error
    the path, so neither may reach pytest's report as it is.
    """
    problem = None
    try:
        return load_account_config(path)
    except Exception as e:  # noqa: BLE001 - reported masked, outside this block
        problem = f"cannot load the account config {path}: {type(e).__name__}: {e}"
    fail_masked(problem)


def load_accounts(path: Path) -> MaskedDict[MaskedDict[str]]:
    """Return the account config's ``{scenario: {tag: account_id}}`` mapping."""
    accounts = _load_config(path).accounts
    return MaskedDict({scenario: MaskedDict(tags) for scenario, tags in accounts.items()})


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
# `env init` validates the account and records the baseline; it has no
# manifest budget of its own.
INIT_TIMEOUT_SEC = 1200.0


@_masked_repr
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

    def env_command(self, step: str, *extra: str) -> MaskedList[str]:
        """The arguments of `env <step>` for this environment: name, source, then ``extra``."""
        return MaskedList(
            ["env", step, "--env-name", self.env_name, *self.env_source_args(), *extra]
        )

    def env_source_args(self) -> MaskedList[str]:
        """Scenario-source arguments for `env init|setup|cleanup`.

        Local mode: `--scenario-path` must be a directory *containing* scenario
        directories (each immediate subdirectory with a scenario.toml counts), so
        it points at `<checkout>/scenarios`, narrowed by `--include-scenarios`.
        """
        if self.dataset_path is None:
            return MaskedList(["-d", self.dataset])
        return MaskedList([
            "--scenario-path", str(self.dataset_path / "scenarios"),
            "--include-scenarios", self.scenario,
        ])  # fmt: skip

    def run_source_args(self) -> MaskedList[str]:
        """Dataset-source arguments for `run`.

        Local mode needs both a tasks dir (`-p`) and a scenarios dir. `run` loads
        every scenario under `--scenario-path` but only uses those its tasks
        reference, so `-p <checkout>/tasks/<scenario>` limits it to one.
        """
        if self.dataset_path is None:
            return MaskedList(["-d", self.dataset])
        return MaskedList([
            "--scenario-path", str(self.dataset_path / "scenarios"),
            "-p", str(self.dataset_path / "tasks" / self.scenario),
        ])  # fmt: skip

    def task_filter_args(self) -> MaskedList[str]:
        """`-i <glob>` for each task glob (empty: every task of the dataset)."""
        return MaskedList(arg for g in self.task_globs for arg in ("-i", g))

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
        pytest.skip(mask_account_ids(f"{ACCOUNT_CONFIG_VAR}={raw} is not a file"))
    # Pre-existing mode requires --env-name to equal the config's `name`
    # (TestEnvironmentNotFoundError otherwise), so it cannot carry a per-run
    # suffix. Isolation between runs comes from the redirected HOME and unique
    # job names; AWS-level isolation needs one account per concurrent runner.
    env_name = _load_config(config).name
    override = environ.get(ENV_NAME_VAR)
    if override and override != env_name:
        fail_masked(
            f"{ENV_NAME_VAR}={override!r} does not match the account config's "
            f"name {env_name!r}; pre-existing mode requires them to be equal. "
            f"Unset {ENV_NAME_VAR} or change `name` in {config}."
        )
    dataset_path: Path | None = None
    scenario = environ.get(SCENARIO_VAR) or DEFAULT_SCENARIO
    raw_dataset_path = environ.get(DATASET_PATH_VAR)
    if raw_dataset_path:
        if environ.get(DATASET_VAR):
            fail_masked(f"Set either {DATASET_VAR} or {DATASET_PATH_VAR}, not both.")
        dataset_path = Path(raw_dataset_path).expanduser().resolve()
        for required in (
            dataset_path / "scenarios" / scenario / "scenario.toml",
            dataset_path / "tasks" / scenario,
        ):
            if not required.exists():
                fail_masked(
                    f"{DATASET_PATH_VAR}={raw_dataset_path}: missing {required}. Point it at "
                    f"an aws-bench-datasets checkout and set {SCENARIO_VAR} if needed."
                )
    expected_trials: int | None = None
    raw_expected = environ.get(EXPECTED_TRIALS_VAR, "").strip()
    if raw_expected:
        try:
            expected_trials = int(raw_expected)
        except ValueError:
            expected_trials = 0
        if expected_trials < 1:
            fail_masked(
                f"{EXPECTED_TRIALS_VAR}={raw_expected!r} must be a positive integer "
                "(the number of trials the run must plan), or unset."
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


class SecretDict(dict[str, str]):
    """A mutable environment whose ``repr`` shows no values (see ``SecretEnv``)."""

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a representation without values."""
        return f"SecretDict(<{len(self)} variables, values hidden>)"

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


@_masked_repr
@dataclass(frozen=True)
class CliResult:
    """Completed ``aws-bench`` invocation. ``args`` are redacted, output is scrubbed.

    ``stdout`` and ``stderr`` keep account IDs, because the parsers need them;
    use ``masked_output`` in a message. ``repr`` masks them.
    """

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        """Scrubbed stdout and stderr, raw: for parsing only, never for a message."""
        return MaskedStr(self.stdout + self.stderr)

    def masked_output(self, tail: int | None = None) -> str:
        """``output`` with account IDs masked; this is the view for messages.

        The whole text is masked first and cut afterwards, so an ID that the
        cut would split cannot leave its last digits visible. Secrets are already
        scrubbed from ``stdout`` and ``stderr``.
        """
        text = mask_account_ids(self.output)
        return text if tail is None else text[-tail:]


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


@_masked_repr
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
    """The last ``_TAIL`` characters of each stream, account IDs masked before the cut."""
    out, err = mask_account_ids(outcome.stdout), mask_account_ids(outcome.stderr)
    return f"--- stdout ---\n{out[-_TAIL:]}\n--- stderr ---\n{err[-_TAIL:]}"


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
    Account IDs are masked in every failure message too, after the secrets: the
    command line and the tail of each stream (``_tails``). The returned
    ``CliResult`` keeps them in ``stdout``/``stderr`` for the parsers and masks
    them in its ``repr``.

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
    shown = mask_account_ids(" ".join(cmd.shown))
    if isinstance(outcome, BaseException):
        outcome.__context__ = None
        outcome.__cause__ = None
        raise outcome.with_traceback(None) from None
    if outcome.problem is not None:
        fail_masked(f"`aws-bench {shown}` {outcome.problem}\n{_tails(outcome)}")
    assert outcome.returncode is not None
    result = CliResult(
        cmd.shown, outcome.returncode, MaskedStr(outcome.stdout), MaskedStr(outcome.stderr)
    )
    if check and result.returncode != 0:
        fail_masked(f"`aws-bench {shown}` exited {result.returncode}\n{_tails(outcome)}")
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
        # run_aws_bench failures are already redacted and masked; anything else is named only.
        detail = str(e) if isinstance(e, pytest.fail.Exception) else type(e).__name__
        return detail
    if result.returncode != 0:
        return f"exit {result.returncode}\n{result.masked_output(_TAIL)}"
    return None


class Deployment:
    """Handle yielded by ``deployed``; lets a test that performs the cleanup itself say so."""

    def __init__(self) -> None:
        self.cleaned = False

    def mark_clean(self) -> None:
        """Record that the body ran `env cleanup` and confirmed nothing was left.

        Call it only after both the cleanup step and every check on its result
        passed. ``deployed`` then skips its own cleanup attempt, but only if the
        body then exits normally: a body that calls this and then raises still
        gets the attempt. Without this call, the attempt always runs.
        """
        self.cleaned = True


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
) -> Iterator[Deployment]:
    """Run `env init` and `env setup`, yield, then always attempt `env cleanup` once.

    The timeouts are callables because they are read from manifests that
    `env init` caches. Outcomes:

    - Everything succeeds: returns normally.
    - Only cleanup fails: ``pytest.fail`` with the leak notice.
    - Init, setup or the body raised (failure, error, timeout, interrupt):
      cleanup is still attempted and the original exception is re-raised. A
      cleanup failure is appended to its message (``pytest.fail`` failures) or
      attached as a note (other exceptions) instead of replacing it.
    - The body called ``Deployment.mark_clean()`` (it ran `env cleanup` itself
      and checked the result) and then exited normally: the attempt is skipped,
      since there is nothing left to clean. A body that fails before that call,
      or raises after it, still gets the attempt.

    Used by the ``deployed_env`` fixture around its ``yield``. pytest does not
    throw a test's own failure into a fixture, so for a failing test the
    fixture exits normally and a cleanup failure is reported as a separate
    teardown error.
    """
    primary: BaseException | None = None
    deployment = Deployment()
    try:
        run_cli(*init_args, timeout=init_timeout)
        run_cli(*setup_args, timeout=setup_timeout())
        yield deployment
    except BaseException as e:  # re-raised below, after cleanup
        primary = e
    problem = (
        None
        if deployment.cleaned and primary is None
        else _attempt_cleanup(run_cli, cleanup_args, cleanup_timeout)
    )
    if primary is None:
        if problem is not None:
            fail_masked(f"{_LEAK_NOTICE}:\n{problem}")
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


@_masked_repr
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
        return mask_account_ids(f"env setup {self.setup:g}s, {cleanup} ({self.source})")


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


@_masked_repr
@dataclass(frozen=True)
class RunTimeout:
    """Subprocess timeout (seconds) for ``run``."""

    seconds: float
    source: str

    def describe(self) -> str:
        return mask_account_ids(f"run {self.seconds:g}s ({self.source})")


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


@_masked_repr
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
        fail_masked(
            f"{EXPECTED_TRIALS_VAR}={expected}, but the selection resolves {resolved} "
            f"trial(s) (task globs: {globs}). Fix the selection or the expectation."
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


def plain(text: str) -> MaskedStr:
    """Return ``text`` without ANSI escape sequences, with a masked ``repr``."""
    return MaskedStr(_ANSI_RE.sub("", text))


def cli_display_env(env: Mapping[str, str]) -> SecretDict:
    """Return ``env`` with ``COLUMNS=CLI_COLUMNS`` and no ``COLOR_FORCING_VARS``.

    Without colour forcing, Rich sees a pipe as a non-terminal and honours
    ``COLUMNS`` whatever ``TERM`` is. The result is a ``SecretDict``: a plain
    ``dict`` for ``subprocess``, with a ``repr`` that hides the values.
    """
    out = SecretDict((k, v) for k, v in env.items() if k not in COLOR_FORCING_VARS)
    out["COLUMNS"] = str(CLI_COLUMNS)
    return out


def build_cli_env(
    environ: Mapping[str, str], *, account_config: Path, scratch_home: Path
) -> SecretDict:
    """Return the CLI subprocess environment built from ``environ`` (see ``cli_env``).

    Symlinks ``.aws``, ``.docker`` and the ``HOME_LINKS_VAR`` paths of the real
    ``HOME`` into ``scratch_home``, points ``HOME`` at ``scratch_home``, sets
    ``AWSBENCH_ACCOUNT_CONFIG`` and applies ``cli_display_env``. The environment
    is held as a ``SecretDict`` throughout, so ``-l`` shows no credential values.
    """
    env = SecretDict(environ)
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


@_masked_repr
@dataclass(frozen=True)
class ShownStack:
    """One row of an account's stack table in `env show`."""

    name: str
    status: str
    region: str


@_masked_repr
@dataclass(frozen=True)
class ShownAccount:
    """One account panel of `env show`."""

    scenario: str
    tag: str
    account_id: str
    status: str
    stacks: tuple[ShownStack, ...] = ()


@_masked_repr
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
    output = MaskedStr(output)  # pytest prints this frame's arguments and locals with repr
    panel: str | None = None  # "header", "account" or None (outside / unknown panel)

    def close_account() -> None:
        if panel == "account":
            accounts[-1] = replace(accounts[-1], stacks=tuple(stacks))

    for line in [MaskedStr(raw.rstrip()) for raw in plain(output).splitlines()]:
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
        text = MaskedStr(body["body"].strip())
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
    Account IDs in the returned messages (the triples, and names read from the
    output) are masked. Accounts that share their last four digits render
    identically; compare the raw values when two IDs matter.

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
    return mask_problems(problems)


@_masked_repr
@dataclass(frozen=True)
class ListedEnv:
    """One row of the `env list` table (``render_env_list``); ``accounts`` None if "?"."""

    name: str
    ou_id: str
    accounts: int | None


def parse_env_list(output: str) -> tuple[ListedEnv, ...]:
    """Parse the body rows (``│ name │ ou id │ accounts │``) of the `env list` table."""
    rows: list[ListedEnv] = []
    output = MaskedStr(output)  # pytest prints this frame's arguments and locals with repr
    for line in [MaskedStr(raw.strip()) for raw in plain(output).splitlines()]:
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
    Account IDs in the returned messages (names read from the output) are masked.
    """
    if len(rows) != 1:
        return mask_problems(
            [f"expected exactly one environment row, got {len(rows)}: {list(rows)}"]
        )
    row, problems = rows[0], []
    if row.name != env_name:
        problems.append(f"environment {row.name!r} listed, expected {env_name!r}")
    if row.ou_id != PREEXISTING_OU_ID:
        problems.append(f"OU id {row.ou_id!r}, expected {PREEXISTING_OU_ID!r}")
    expected = len(_configured(accounts))
    if row.accounts != expected:
        problems.append(f"{row.accounts!r} accounts listed, expected {expected}")
    return mask_problems(problems)


# --------------------------------------------------------------------------- #
# Post-cleanup check: what is left in the accounts
# --------------------------------------------------------------------------- #

# CloudFormation status of a stack that is gone. `list_stacks` keeps reporting
# deleted stacks (for about 90 days) with this status.
STACK_GONE_STATUS = "DELETE_COMPLETE"

SESSION_FOR_ACCOUNT = Callable[[str], Any]
"""Builds a boto3 session for one account ID (the CLI's own credential path)."""


@_masked_repr
@dataclass(frozen=True)
class StackRecord:
    """One CloudFormation stack as listed by `list_stacks`."""

    region: str
    name: str
    status: str


@_masked_repr
@dataclass(frozen=True)
class AccountStacks:
    """The stacks found in one configured account, region by region.

    ``regions`` are the regions that were queried; ``errors`` describe any
    query that failed (error codes only, never the service's message).
    """

    scenario: str
    tag: str
    account_id: str
    regions: tuple[str, ...]
    stacks: tuple[StackRecord, ...] = ()
    errors: tuple[str, ...] = ()


def _error_label(error: BaseException) -> str:
    """Name an error without its message, which may carry ARNs or account IDs."""
    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code")
        return f"ClientError {code}" if isinstance(code, str) and code else "ClientError"
    return type(error).__name__


def scenario_regions(manifest_paths: Sequence[Path]) -> MaskedDict[list[str]]:
    """Map each readable manifest's scenario name to the regions it declares.

    These are the regions the cleanup contract covers and the region
    restriction allows. Unreadable manifests are skipped, so their scenarios
    are missing from the result (``stack_problems`` reports that).
    """
    regions: MaskedDict[list[str]] = MaskedDict()
    for path in manifest_paths:
        try:
            manifest = ScenarioManifest.model_validate_toml(path.read_bytes())
        except (OSError, ValueError):
            continue
        regions[manifest.scenario.name] = list(manifest.scenario.regions)
    return regions


def collect_account_stacks(
    *,
    scenario: str,
    tag: str,
    account_id: str,
    regions: Sequence[str],
    session_for: SESSION_FOR_ACCOUNT,
) -> AccountStacks:
    """List every stack in ``regions`` of one account, using only read-only calls.

    Calls ``cloudformation:ListStacks`` (no status filter, all pages) once per
    region. A failure to get a session or to list a region is recorded in
    ``errors`` instead of being skipped, so it cannot pass as "no stacks".
    """
    errors: list[str] = []
    stacks: list[StackRecord] = []
    try:
        session = session_for(account_id)
    except Exception as e:  # noqa: BLE001 - reported by type, never by message
        return AccountStacks(
            scenario, tag, account_id, tuple(regions), errors=(f"credentials: {_error_label(e)}",)
        )
    for region in regions:
        try:
            cfn = session.client("cloudformation", region_name=region)
            for page in cfn.get_paginator("list_stacks").paginate():
                for summary in page.get("StackSummaries", []):
                    stacks.append(StackRecord(region, summary["StackName"], summary["StackStatus"]))
        except Exception as e:  # noqa: BLE001 - reported by type, never by message
            errors.append(f"{region}: {_error_label(e)}")
    return AccountStacks(
        scenario, tag, account_id, tuple(regions), stacks=tuple(stacks), errors=tuple(errors)
    )


def _check_coverage(
    found: Sequence[AccountStacks],
    *,
    accounts: Mapping[str, Mapping[str, str]],
    regions_by_scenario: Mapping[str, Sequence[str]],
) -> tuple[list[str], dict[tuple[str, str, str], AccountStacks]]:
    """Validate that ``found`` covers the configured accounts, once each, cleanly.

    Shared by ``stack_problems`` and ``deployed_stack_problems`` so that both
    report the same problems in the same words. Returns the problems and the
    records that passed every check (configured, seen once, queried over exactly
    the regions the scenario declares, no failed query), keyed by
    ``(scenario, tag, account_id)``. A problem is reported for:

    - an empty account config, a configured account that was not checked,
      checked twice, or an unconfigured one;
    - a scenario with no known regions (an unreadable manifest), or an account
      queried over other regions than its scenario declares;
    - any failed query (credentials or listing a region): an unreadable region
      is not an empty one.
    """
    problems: list[str] = []
    expected = _configured(accounts)
    if not expected:
        problems.append("the account config maps no accounts: nothing was checked")
    seen = [(a.scenario, a.tag, a.account_id) for a in found]
    duplicates = sorted({k for k in seen if seen.count(k) > 1})
    if duplicates:
        problems.append(
            "accounts checked more than once: " + ", ".join(account_label(*k) for k in duplicates)
        )
    missing = sorted(set(expected) - set(seen))
    if missing:
        problems.append(
            "configured accounts not checked: " + ", ".join(account_label(*k) for k in missing)
        )
    unexpected = sorted(set(seen) - set(expected))
    if unexpected:
        problems.append(
            "accounts checked that are not configured: "
            + ", ".join(account_label(*k) for k in unexpected)
        )
    clean: dict[tuple[str, str, str], AccountStacks] = {}
    for a in found:
        key = (a.scenario, a.tag, a.account_id)
        label = account_label(*key)
        account_problems: list[str] = []
        declared = regions_by_scenario.get(a.scenario)
        if not declared:
            account_problems.append(
                f"{label}: no declared regions for scenario {a.scenario!r} (manifest?)"
            )
        elif sorted(a.regions) != sorted(declared):
            account_problems.append(
                f"{label}: queried regions {sorted(a.regions)}, "
                f"scenario declares {sorted(declared)}"
            )
        account_problems.extend(f"{label}: could not list stacks: {err}" for err in a.errors)
        problems.extend(account_problems)
        if not account_problems and seen.count(key) == 1 and key in expected:
            clean[key] = a
    return problems, clean


def stack_problems(
    found: Sequence[AccountStacks],
    *,
    accounts: Mapping[str, Mapping[str, str]],
    regions_by_scenario: Mapping[str, Sequence[str]],
) -> list[str]:
    """Return what `env cleanup` left behind, or could not be shown to have removed.

    ``accounts`` is the account config's ``{scenario: {tag: account_id}}`` and
    ``regions_by_scenario`` the regions each scenario manifest declares. The
    coverage problems listed in ``_check_coverage`` are reported, and so is any
    stack whose status is not ``DELETE_COMPLETE``. That includes
    ``DELETE_IN_PROGRESS`` (not gone yet) and ``DELETE_FAILED``.

    A CDK bootstrap stack (a name the product treats as CDK bootstrap
    infrastructure, ``is_infra_identifier``) is not tolerated in any state other
    than ``DELETE_COMPLETE``. On the path the test asserts (exit 0 and the
    success summary) the CLI deletes the bootstrap stacks too, so one that
    remains is reported, labelled as a bootstrap stack. A surviving environment
    stack that the product did not defer makes the command exit non-zero (the
    test then fails at the exit-code step) and skips the infrastructure
    deletion. A deferred environment stack keeps its FAILED status internally
    but is left out of the failed list, so the command exits 0 with the success
    summary while ``CDKToolkit`` is deliberately retained. That is the case
    where the asserted summary coexists with a retained bootstrap stack, and the
    deferred environment stack reported next to it is the real cause of the
    failure. A bug in the product's infrastructure-name list would also change
    which stacks get the bootstrap label, but not whether they are reported.

    Nested stacks (summaries with ``ParentId``) are listed and counted on
    purpose. The product's ``StackDeleter.list_stacks`` never deletes them, so a
    surviving nested stack needs manual deletion. Other resource types are not
    checked. Account IDs inside stack or scenario names are masked in the result.
    """
    problems, _ = _check_coverage(found, accounts=accounts, regions_by_scenario=regions_by_scenario)
    for a in found:
        label = account_label(a.scenario, a.tag, a.account_id)
        for stack in sorted(a.stacks, key=lambda s: (s.region, s.name)):
            if stack.status == STACK_GONE_STATUS:
                continue
            kind = "CDK bootstrap stack" if is_infra_identifier(stack.name) else "stack"
            problems.append(f"{label}: {stack.region}: {kind} {stack.name} is {stack.status}")
    return mask_problems(problems)


def deployed_stack_problems(
    found: Sequence[AccountStacks],
    *,
    accounts: Mapping[str, Mapping[str, str]],
    regions_by_scenario: Mapping[str, Sequence[str]],
) -> list[str]:
    """Return why ``found`` does not show a deployment in every configured account.

    Run between `env setup` and `env cleanup`. The coverage problems listed in
    ``_check_coverage`` are reported, and every configured account that passed
    them must hold at least one stack named ``<scenario>-*`` in a
    ``HEALTHY_STACK_STATUSES`` state. Without this, a cleanup that had nothing
    to remove would pass the post-cleanup check. Other stacks (``CDKToolkit``,
    unrelated ones) do not count. The stack only has to exist at that moment;
    it does not show that `env setup` created it.
    """
    problems, clean = _check_coverage(
        found, accounts=accounts, regions_by_scenario=regions_by_scenario
    )
    for (scenario, tag, account_id), entry in clean.items():
        if not any(
            s.name.startswith(f"{scenario}-") and s.status in HEALTHY_STACK_STATUSES
            for s in entry.stacks
        ):
            problems.append(
                f"{account_label(scenario, tag, account_id)}: no healthy stack named "
                f"'{scenario}-*' before cleanup, so no deployment was found in this account "
                "(or it was not listed)"
            )
    return mask_problems(problems)


def cleanup_output_problems(output: str, *, accounts: int) -> list[str]:
    """Check the `env cleanup` summary the CLI prints on stdout (``display_cleanup_results``).

    On success it prints ``Cleanup complete: N account(s) cleaned successfully.``
    with one account per configured account; any failed row makes it print
    ``Cleanup completed with errors`` and exit non-zero instead. The caller has
    already required exit code 0; this guards the printed summary. Pass the
    masked stdout (``mask_account_ids(result.stdout)``); the checks do not need
    the IDs.
    """
    text = plain(output)
    problems: list[str] = []
    if "Cleanup completed with errors" in text:
        problems.append("the cleanup summary reports errors")
    expected = f"Cleanup complete: {accounts} account(s) cleaned successfully."
    if expected not in text:
        problems.append(f"the cleanup summary lacks {expected!r}")
    return problems
