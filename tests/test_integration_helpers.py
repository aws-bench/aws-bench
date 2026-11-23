"""Unit tests for ``tests/integration/helpers.py``.

Kept outside ``tests/integration`` on purpose: everything there is marked
``integration`` and deselected by default, but these must run in ``make test``.
Secret-handling tests check the full report of an inner pytest session (run as
a subprocess), because that is what a developer or CI log actually shows.
"""

import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import replace
from pathlib import Path

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws
from rich.console import Console

from aws_bench.account_management.models import ScenarioAccount
from aws_bench.cli.display import render_account_state, render_env_header, render_env_list
from aws_bench.resource_management.models import QuotaEntry
from tests.integration import helpers
from tests.integration.helpers import (
    CLEANUP_MARGIN_SEC,
    CLI_COLUMNS,
    COLOR_FORCING_VARS,
    FALLBACK_RUN_TIMEOUT_SEC,
    FALLBACK_SETUP_TIMEOUT_SEC,
    FALLBACK_TASK_BUDGET_SEC,
    HOME_LINKS_VAR,
    PREEXISTING_OU_ID,
    RUN_MARGIN_SEC,
    SETUP_MARGIN_SEC,
    AccountStacks,
    CliResult,
    EnvShow,
    IntegSettings,
    ListedEnv,
    Secret,
    SecretEnv,
    ShownAccount,
    ShownStack,
    StackRecord,
    _error_label,
    build_cli_env,
    cleanup_output_problems,
    cli_display_env,
    collect_account_stacks,
    deployed,
    deployed_stack_problems,
    env_list_problems,
    env_secret_values,
    env_show_problems,
    expected_trial_count,
    find_scenario_manifests,
    load_settings,
    mask_account_id,
    mask_account_ids,
    parse_env_list,
    parse_env_show,
    plain,
    redact_args,
    resolve_run_tasks,
    run_aws_bench,
    run_timeout_for,
    scenario_regions,
    scrub,
    secret_values,
    stack_problems,
    step_timeouts_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKEN = "CANARY-TOKEN-0123456789"
ENV_TOKEN = "CANARY-ENV-SECRET-9876543210"


# --------------------------------------------------------------------------- #
# redact_args / secret_values / scrub
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("flag", ["--ve", "--verifier-env", "--ae", "--agent-env"])
def test_env_flag_value_is_masked(flag):
    args = ["run", flag, f"AWS_BEARER_TOKEN_BEDROCK={TOKEN}", "--yes"]
    assert redact_args(args) == ["run", flag, "AWS_BEARER_TOKEN_BEDROCK=***", "--yes"]


def test_env_flag_masks_non_secret_named_keys_too():
    assert redact_args(["--ae", "FOO=bar"]) == ["--ae", "FOO=***"]


def test_env_flag_equals_form_is_masked():
    assert redact_args([f"--ve=MY_TOKEN={TOKEN}"]) == ["--ve=MY_TOKEN=***"]


def test_env_flag_value_without_equals_is_fully_masked():
    assert redact_args(["--ve", TOKEN]) == ["--ve", "***"]


@pytest.mark.parametrize(
    "key",
    ["API_KEY", "db_password", "MY_SECRET", "Bearer", "AWS_SESSION_TOKEN", "credential_file"],
)
def test_secret_named_assignment_is_masked(key):
    assert redact_args(["-k", f"{key}={TOKEN}"]) == ["-k", f"{key}=***"]


def test_non_secret_args_unchanged():
    args = [
        "run", "-i", "*list-ec-instances-all-regions", "--env-name", "env-a",
        "-d", "aws-bench-quickstart", "--job-name", "integ-1", "--agent-kwarg", "x=1",
        "--quiet",
    ]  # fmt: skip
    assert redact_args(args) == args


def test_trailing_env_flag_without_value():
    assert redact_args(["run", "--ve"]) == ["run", "--ve"]


def test_secret_values_returns_raw_values():
    args = ["--ve", f"A={TOKEN}", "--ae=B=other-secret", "C_PASSWORD=pw123", "--quiet"]
    assert set(secret_values(args)) == {TOKEN, "other-secret", "pw123"}


def test_secret_values_handles_value_containing_equals():
    assert secret_values(["--ve", "A=x=y"]) == ["x=y"]


def test_scrub_replaces_every_occurrence():
    text = f"token {TOKEN} and again {TOKEN}"
    assert scrub(text, [TOKEN]) == "token *** and again ***"


def test_scrub_ignores_empty_secret():
    assert scrub("abc", [""]) == "abc"


def test_secret_repr_and_str_are_masked():
    s = Secret(TOKEN)
    assert TOKEN not in repr(s)
    assert TOKEN not in str(s)
    assert TOKEN not in f"{s}"
    assert s.value == TOKEN


def test_env_secret_values_by_name_and_length():
    env = {
        "AWS_SECRET_ACCESS_KEY": ENV_TOKEN,
        "AWS_SESSION_TOKEN": TOKEN,
        "SOME_KEY": "short",  # below MIN_ENV_SECRET_LEN
        "MY_PASSWORD": "",
        "HOME": "/home/user-with-long-path",  # not secret-named
    }
    assert env_secret_values(env) == [ENV_TOKEN, TOKEN]


def test_secret_env_hides_values_but_behaves_as_mapping():
    env = SecretEnv({"AWS_SECRET_ACCESS_KEY": ENV_TOKEN, "HOME": "/h"})
    assert ENV_TOKEN not in repr(env) and ENV_TOKEN not in str(env)
    assert env["HOME"] == "/h"
    assert dict(env) == {"AWS_SECRET_ACCESS_KEY": ENV_TOKEN, "HOME": "/h"}
    assert len(env) == 2


def test_command_repr_hides_argv_and_env():
    cmd = helpers._Command(  # pyright: ignore[reportPrivateUsage]
        ["run", "--ve", Secret(f"A={TOKEN}"), f"B_TOKEN={TOKEN}"], {"X_SECRET": ENV_TOKEN}
    )
    assert TOKEN not in repr(cmd) and ENV_TOKEN not in repr(cmd)
    assert cmd.shown == ("run", "--ve", "A=***", "B_TOKEN=***")
    assert cmd.scrub(f"{TOKEN} {ENV_TOKEN}") == "*** ***"


def test_secret_argument_without_flag_is_masked():
    cmd = helpers._Command(["x", Secret(TOKEN)], {})  # pyright: ignore[reportPrivateUsage]
    assert cmd.shown == ("x", "***")
    assert cmd.scrub(TOKEN) == "***"


# --------------------------------------------------------------------------- #
# run_aws_bench with a real child process (CLI_PROGRAM replaced by a script)
# --------------------------------------------------------------------------- #


def _use_program(monkeypatch: pytest.MonkeyPatch, script: str) -> None:
    """Run ``python -c script <args>`` instead of the CLI."""
    monkeypatch.setattr(helpers, "CLI_PROGRAM", (sys.executable, "-c", textwrap.dedent(script)))


def _run_args() -> tuple[str, ...]:
    return ("run", "--ve", f"AWS_BEARER_TOKEN_BEDROCK={TOKEN}", "--yes")


def _failed_report(excinfo: pytest.ExceptionInfo[BaseException]) -> str:
    """What pytest prints for a ``pytrace=False`` failure: the message, no frames, no chain."""
    assert excinfo.value.pytrace is False  # type: ignore[attr-defined]
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    return str(excinfo.value)


ECHO_AND_EXIT = """
import sys
print("argv:", " ".join(sys.argv[1:]))
print("reversed:", sys.argv[3][::-1])
print("err", sys.argv[3], file=sys.stderr)
sys.exit(int(sys.argv[-1]) if sys.argv[-1].isdigit() else 0)
"""


def test_nonzero_exit_message_has_no_secret_even_if_output_echoes_it(monkeypatch, tmp_path):
    _use_program(monkeypatch, ECHO_AND_EXIT)
    with pytest.raises(pytest.fail.Exception) as excinfo:
        run_aws_bench((*_run_args(), "2"), cwd=tmp_path, env=dict(os.environ), timeout=30)
    report = _failed_report(excinfo)
    assert TOKEN not in report
    assert "exited 2" in report
    assert "AWS_BEARER_TOKEN_BEDROCK=***" in report


def test_check_false_returns_redacted_result_and_passes_real_value(monkeypatch, tmp_path):
    _use_program(monkeypatch, ECHO_AND_EXIT)
    result = run_aws_bench(
        (*_run_args(), "1"), cwd=tmp_path, env=dict(os.environ), timeout=30, check=False
    )
    assert result.returncode == 1
    assert TOKEN not in repr(result)
    # The child received the real value (its reversal is not a scrubbed string).
    assert f"AWS_BEARER_TOKEN_BEDROCK={TOKEN}"[::-1] in result.stdout
    assert "AWS_BEARER_TOKEN_BEDROCK=***" in result.stdout


def test_success_returns_output(monkeypatch, tmp_path):
    _use_program(monkeypatch, "print('ok', end='')")
    result = run_aws_bench(("env", "list"), cwd=tmp_path, env=dict(os.environ), timeout=30)
    assert (result.returncode, result.output, result.args) == (0, "ok", ("env", "list"))


def test_env_only_secret_is_scrubbed_from_result(monkeypatch, tmp_path):
    _use_program(monkeypatch, "import os; print(os.environ['AWS_SECRET_ACCESS_KEY'])")
    env = SecretEnv({**os.environ, "AWS_SECRET_ACCESS_KEY": ENV_TOKEN})
    result = run_aws_bench(("env", "list"), cwd=tmp_path, env=env, timeout=30)
    assert ENV_TOKEN not in result.output
    assert "***" in result.output


def test_invalid_utf8_output_is_replaced_not_raised(monkeypatch, tmp_path):
    _use_program(
        monkeypatch,
        "import sys; sys.stdout.buffer.write(b'\\xff\\xfe ' + sys.argv[3].encode()); sys.exit(1)",
    )
    with pytest.raises(pytest.fail.Exception) as excinfo:
        run_aws_bench(_run_args(), cwd=tmp_path, env=dict(os.environ), timeout=30)
    report = _failed_report(excinfo)
    assert "exited 1" in report and "\ufffd" in report
    assert TOKEN not in report


def test_start_failure_message_has_no_secret(monkeypatch, tmp_path):
    monkeypatch.setattr(helpers, "CLI_PROGRAM", (str(tmp_path / f"missing-{TOKEN}"),))
    with pytest.raises(pytest.fail.Exception) as excinfo:
        run_aws_bench(_run_args(), cwd=tmp_path, env={}, timeout=5)
    report = _failed_report(excinfo)
    assert "could not start: FileNotFoundError" in report
    assert TOKEN not in report


IGNORES_SIGTERM = """
import signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("started", sys.argv[3], flush=True)
time.sleep(30)
"""

HANDLES_SIGTERM = """
import signal, sys, time
def _bye(*_):
    print("graceful shutdown", sys.argv[3], flush=True)
    sys.exit(0)
signal.signal(signal.SIGTERM, _bye)
print("started", flush=True)
time.sleep(30)
"""


# Long enough for the child to install its SIGTERM handler before the timeout fires.
CHILD_TIMEOUT = 2.0


def test_timeout_sends_sigterm_then_sigkill_to_a_child_that_ignores_it(monkeypatch, tmp_path):
    _use_program(monkeypatch, IGNORES_SIGTERM)
    start = time.monotonic()
    with pytest.raises(pytest.fail.Exception) as excinfo:
        run_aws_bench(
            _run_args(), cwd=tmp_path, env=dict(os.environ), timeout=CHILD_TIMEOUT, term_grace=0.5
        )
    assert time.monotonic() - start < 15
    report = _failed_report(excinfo)
    assert f"timed out after {CHILD_TIMEOUT:g}s" in report
    assert "SIGKILL sent after a 0.5s SIGTERM grace period" in report
    assert "started AWS_BEARER_TOKEN_BEDROCK=***" in report
    assert TOKEN not in report


def test_timeout_lets_a_cooperative_child_shut_down(monkeypatch, tmp_path):
    _use_program(monkeypatch, HANDLES_SIGTERM)
    with pytest.raises(pytest.fail.Exception) as excinfo:
        run_aws_bench(
            _run_args(), cwd=tmp_path, env=dict(os.environ), timeout=CHILD_TIMEOUT, term_grace=10
        )
    report = _failed_report(excinfo)
    assert "the CLI exited after SIGTERM" in report
    assert "graceful shutdown AWS_BEARER_TOKEN_BEDROCK=***" in report
    assert TOKEN not in report


# --------------------------------------------------------------------------- #
# run_aws_bench with a fake Popen: unexpected exceptions and interrupts
# --------------------------------------------------------------------------- #


class _FakePopen:
    """Stands in for ``subprocess.Popen``; ``communicate`` raises ``raises``."""

    raises: BaseException = ValueError(f"boom {TOKEN}")
    instances: list["_FakePopen"] = []

    def __init__(self, argv, **kwargs):
        self.returncode: int | None = None
        self.signals: list[int] = []
        self.kwargs = kwargs
        _FakePopen.instances.append(self)

    def communicate(self, timeout=None):
        raise type(self).raises

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)
        self.returncode = -sig

    def kill(self):
        self.signals.append(signal.SIGKILL)
        self.returncode = -signal.SIGKILL

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def fake_popen(monkeypatch):
    _FakePopen.instances = []
    monkeypatch.setattr(helpers.subprocess, "Popen", _FakePopen)
    return _FakePopen


def _unicode_error() -> UnicodeDecodeError:
    return UnicodeDecodeError("utf-8", TOKEN.encode(), 0, 1, f"invalid {TOKEN}")


@pytest.mark.parametrize(
    "exc", [ValueError(f"boom {TOKEN}"), _unicode_error()], ids=["ValueError", "UnicodeDecodeError"]
)
def test_unexpected_exception_fails_redacted_and_stops_child(
    fake_popen, monkeypatch, tmp_path, exc
):
    monkeypatch.setattr(fake_popen, "raises", exc)
    with pytest.raises(pytest.fail.Exception) as excinfo:
        run_aws_bench(_run_args(), cwd=tmp_path, env={}, timeout=5, term_grace=0)
    report = _failed_report(excinfo)
    assert f"failed while reading its output: {type(exc).__name__}" in report
    assert TOKEN not in report
    assert fake_popen.instances[0].signals == [signal.SIGTERM, signal.SIGKILL]


def test_popen_uses_replacement_decoding(fake_popen, tmp_path):
    with pytest.raises(pytest.fail.Exception):
        run_aws_bench(("x",), cwd=tmp_path, env={}, timeout=5, term_grace=0)
    assert fake_popen.instances[0].kwargs["errors"] == "replace"
    assert fake_popen.instances[0].kwargs["text"] is True


def _frames_text(exc: BaseException) -> str:
    """Repr of every local in every frame of ``exc``'s traceback (what ``-l`` would show)."""
    parts = []
    tb = exc.__traceback__
    while tb is not None:
        parts.extend(f"{k}={v!r}" for k, v in tb.tb_frame.f_locals.items())
        tb = tb.tb_next
    return "\n".join(parts)


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(3)])
def test_interrupt_stops_child_and_reraises_without_argv(
    fake_popen, monkeypatch, tmp_path, interrupt
):
    monkeypatch.setattr(fake_popen, "raises", interrupt)
    env = SecretEnv({"AWS_SECRET_ACCESS_KEY": ENV_TOKEN})
    with pytest.raises(type(interrupt)) as excinfo:
        run_aws_bench(
            ("run", "--ve", Secret(f"AWS_BEARER_TOKEN_BEDROCK={TOKEN}")),
            cwd=tmp_path,
            env=env,
            timeout=5,
            term_grace=0,
        )
    assert excinfo.value is interrupt
    assert excinfo.value.__context__ is None and excinfo.value.__cause__ is None
    assert fake_popen.instances[0].signals[0] == signal.SIGTERM
    frames = _frames_text(excinfo.value)
    assert "run_aws_bench" in "".join(e.name for e in excinfo.traceback)
    assert TOKEN not in frames and ENV_TOKEN not in frames
    # Only the run_aws_bench frame (and pytest.raises' caller) remain: no subprocess frames.
    assert all("subprocess" not in str(e.path) for e in excinfo.traceback)


# --------------------------------------------------------------------------- #
# The full pytest report never contains a canary
# --------------------------------------------------------------------------- #

INNER_TESTS = f"""
import os, sys, textwrap
import pytest
from tests.integration import helpers
from tests.integration.helpers import Secret, SecretEnv, deployed, run_aws_bench

TOKEN = {TOKEN!r}
ENV_TOKEN = {ENV_TOKEN!r}
ARGS = ("run", "--ve", f"AWS_BEARER_TOKEN_BEDROCK={{TOKEN}}", "--yes")


class FakePopen:
    kind = None
    def __init__(self, argv, **kw): self.returncode = None
    def communicate(self, timeout=None):
        if type(self).kind == "ValueError":
            raise ValueError(f"boom {{TOKEN}}")
        raise UnicodeDecodeError("utf-8", TOKEN.encode(), 0, 1, f"invalid {{TOKEN}}")
    def poll(self): return self.returncode
    def send_signal(self, sig): self.returncode = -sig
    def kill(self): self.returncode = -9
    def wait(self, timeout=None): return self.returncode


def _program(monkeypatch, script):
    monkeypatch.setattr(helpers, "CLI_PROGRAM", (sys.executable, "-c", textwrap.dedent(script)))


@pytest.mark.parametrize("kind", ["ValueError", "UnicodeDecodeError"])
def test_subprocess_layer_raises(monkeypatch, tmp_path, kind):
    FakePopen.kind = kind
    monkeypatch.setattr(helpers.subprocess, "Popen", FakePopen)
    run_aws_bench(ARGS, cwd=tmp_path, env={{}}, timeout=5, term_grace=0)


def test_nonzero_exit(monkeypatch, tmp_path):
    _program(
        monkeypatch,
        "import sys; print(sys.argv); sys.stdout.buffer.write(b'\\\\xff'); sys.exit(4)",
    )
    run_aws_bench(ARGS, cwd=tmp_path, env=dict(os.environ), timeout=30)


def test_timeout(monkeypatch, tmp_path):
    _program(monkeypatch, "import sys, time; print(sys.argv, flush=True); time.sleep(30)")
    run_aws_bench(ARGS, cwd=tmp_path, env=dict(os.environ), timeout=1.5, term_grace=0.5)


@pytest.fixture
def cli_env():
    return SecretEnv({{**os.environ, "AWS_SECRET_ACCESS_KEY": ENV_TOKEN}})


def test_env_only_secret(monkeypatch, tmp_path, cli_env):
    _program(monkeypatch, "import os, sys; print(os.environ['AWS_SECRET_ACCESS_KEY']); sys.exit(5)")
    run_aws_bench(("env", "list"), cwd=tmp_path, env=cli_env, timeout=30)


@pytest.fixture
def broken_fixture(cli_env):
    raise RuntimeError("fixture failed")


def test_fixture_arguments_are_rendered(broken_fixture):
    pass


def test_setup_and_cleanup_both_fail(monkeypatch, tmp_path, cli_env):
    _program(monkeypatch, "import sys; print('step', sys.argv); sys.exit(7)")

    def run_cli(*args, timeout, check=True):
        return run_aws_bench(args, cwd=tmp_path, env=cli_env, timeout=timeout, check=check)

    secret = Secret(f"AWS_BEARER_TOKEN_BEDROCK={{TOKEN}}")
    with deployed(
        run_cli,
        init_args=["env", "init", "--ve", secret],
        setup_args=["env", "setup"],
        cleanup_args=["env", "cleanup", "--ve", secret],
        init_timeout=30,
        setup_timeout=lambda: 30,
        cleanup_timeout=lambda: 30,
    ):
        pass


def test_unexpected_error_and_cleanup_failure(monkeypatch, tmp_path, cli_env):
    _program(monkeypatch, "import sys; print('cleanup', sys.argv); sys.exit(8)")
    secret = Secret(f"AWS_BEARER_TOKEN_BEDROCK={{TOKEN}}")

    def run_cli(*args, timeout, check=True):
        if args[1] == "init":
            raise RuntimeError("init exploded")
        return run_aws_bench(args, cwd=tmp_path, env=cli_env, timeout=timeout, check=check)

    with deployed(
        run_cli,
        init_args=["env", "init", "--ve", secret],
        setup_args=["env", "setup"],
        cleanup_args=["env", "cleanup", "--ve", secret],
        init_timeout=30,
        setup_timeout=lambda: 30,
        cleanup_timeout=lambda: 30,
    ):
        pass
"""


@pytest.fixture(scope="module")
def inner_test_file(tmp_path_factory):
    root = tmp_path_factory.mktemp("inner")
    (root / "pytest.ini").write_text("[pytest]\n")
    path = root / "test_inner_canary.py"
    path.write_text(INNER_TESTS)
    return path


@pytest.mark.parametrize(
    "flags",
    [[], ["-l", "--tb=long", "--full-trace", "-rA", "-vv"]],
    ids=["default", "showlocals-long"],
)
def test_rendered_report_never_contains_canary(inner_test_file, flags):
    # Drop color-forcing variables (CI sets FORCE_COLOR) so ANSI escapes cannot
    # split the summary line or a secret and make the string checks unreliable.
    color_vars = {"FORCE_COLOR", "PY_COLORS", "CLICOLOR_FORCE"}
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("COV_CORE", "PYTEST_")) and k not in color_vars
    }  # PYTEST_* (PYTEST_ADDOPTS, ...) would change the inner session
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(  # noqa: S603 - fixed argv
        [
            sys.executable,
            "-m",
            "pytest",
            str(inner_test_file),
            "-c",
            str(inner_test_file.parent / "pytest.ini"),
            "--rootdir",
            str(inner_test_file.parent),
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "--color=no",
            *flags,
        ],  # fmt: skip
        cwd=inner_test_file.parent,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=25,
    )
    report = proc.stdout + proc.stderr
    assert "7 failed, 1 error" in report, report[-3000:]
    assert TOKEN not in report
    assert ENV_TOKEN not in report
    # Each failure path was reached and reported.
    for expected in (
        "failed while reading its output: ValueError",
        "failed while reading its output: UnicodeDecodeError",
        "exited 4",
        "timed out after 1.5s",
        "exited 5",
        "SecretEnv(<",
        "env cleanup failed (resources may be left behind)",
        "`aws-bench env init --ve AWS_BEARER_TOKEN_BEDROCK=***` exited 7",
        "Also, env cleanup failed (resources may be left behind)",
        "init exploded",
        "exit 8",
    ):
        assert expected in report, expected


# --------------------------------------------------------------------------- #
# deployed(): cleanup on every exit path
# --------------------------------------------------------------------------- #


class _StubCli:
    """Records calls; ``fail`` maps a step name (args[1]) to what that call does."""

    def __init__(self, fail: dict | None = None):
        self.calls: list[tuple[str, float, bool]] = []
        self.fail = fail or {}

    def __call__(self, *args, timeout: float, check: bool = True) -> CliResult:
        step = str(args[1])
        self.calls.append((step, timeout, check))
        action = self.fail.get(step)
        if isinstance(action, BaseException):
            raise action
        if callable(action):
            result = action(*args, timeout=timeout, check=check)
            assert isinstance(result, CliResult)
            return result
        code = action if isinstance(action, int) else 0
        return CliResult(tuple(map(str, args)), code, f"{step} out", "")

    def steps(self) -> list[str]:
        return [c[0] for c in self.calls]


def _deployed(cli: _StubCli):
    return deployed(
        cli,
        init_args=["env", "init"],
        setup_args=["env", "setup"],
        cleanup_args=["env", "cleanup"],
        init_timeout=11,
        setup_timeout=lambda: 22,
        cleanup_timeout=lambda: 33,
    )


def test_deployed_success_runs_cleanup_once_without_check():
    cli = _StubCli()
    with _deployed(cli):
        pass
    assert cli.calls == [("init", 11, True), ("setup", 22, True), ("cleanup", 33, False)]


def test_deployed_cleanup_failure_alone_fails_with_leak_notice():
    cli = _StubCli({"cleanup": 3})
    with pytest.raises(pytest.fail.Exception) as excinfo:
        with _deployed(cli):
            pass
    assert "env cleanup failed (resources may be left behind)" in str(excinfo.value)
    assert "exit 3" in str(excinfo.value)
    assert cli.steps() == ["init", "setup", "cleanup"]


def test_deployed_setup_failure_reports_original_and_cleans_up_once():
    cli = _StubCli({"setup": pytest.fail.Exception("SETUP BROKE", pytrace=False)})
    with pytest.raises(pytest.fail.Exception, match="SETUP BROKE") as excinfo:
        with _deployed(cli):
            pytest.fail("body must not run")
    assert cli.steps() == ["init", "setup", "cleanup"]
    assert not getattr(excinfo.value, "__notes__", [])


def test_deployed_init_failure_still_cleans_up():
    cli = _StubCli({"init": pytest.fail.Exception("INIT BROKE", pytrace=False)})
    with pytest.raises(pytest.fail.Exception, match="INIT BROKE"):
        with _deployed(cli):
            pass
    assert cli.steps() == ["init", "cleanup"]


@pytest.mark.parametrize(
    "exc", [AssertionError("body assert"), RuntimeError("body boom"), KeyboardInterrupt()]
)
def test_deployed_body_exception_cleans_up_and_reraises(exc):
    cli = _StubCli()
    with pytest.raises(type(exc)) as excinfo:
        with _deployed(cli):
            raise exc
    assert excinfo.value is exc
    assert cli.steps() == ["init", "setup", "cleanup"]


def test_deployed_interrupt_during_setup_still_attempts_cleanup():
    cli = _StubCli({"setup": KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        with _deployed(cli):
            pass
    assert cli.steps() == ["init", "setup", "cleanup"]


def test_deployed_setup_timeout_cleans_up(monkeypatch, tmp_path):
    _use_program(monkeypatch, "import time; time.sleep(30)")

    def real_setup(*args, timeout, check):
        return run_aws_bench(args, cwd=tmp_path, env=dict(os.environ), timeout=1, term_grace=0.5)

    cli = _StubCli({"setup": real_setup})
    with pytest.raises(pytest.fail.Exception, match="timed out after 1s"):
        with _deployed(cli):
            pass
    assert cli.steps() == ["init", "setup", "cleanup"]


@pytest.mark.parametrize(
    "cleanup_action",
    [4, pytest.fail.Exception("CLEANUP TIMED OUT", pytrace=False), RuntimeError("cleanup boom")],
    ids=["nonzero", "failed", "exception"],
)
def test_deployed_setup_and_cleanup_failures_are_both_visible(cleanup_action):
    cli = _StubCli(
        {"setup": pytest.fail.Exception("SETUP BROKE", pytrace=False), "cleanup": cleanup_action}
    )
    with pytest.raises(pytest.fail.Exception) as excinfo:
        with _deployed(cli):
            pass
    message = str(excinfo.value)
    assert message.startswith("SETUP BROKE\n")  # the original, not replaced
    assert "Also, env cleanup failed (resources may be left behind)" in message
    expected = {"nonzero": "exit 4", "failed": "CLEANUP TIMED OUT", "exception": "RuntimeError"}
    assert any(v in message for v in expected.values())
    assert cli.steps() == ["init", "setup", "cleanup"]


def test_deployed_body_error_and_cleanup_failure_adds_note():
    cli = _StubCli({"cleanup": 2})
    with pytest.raises(RuntimeError, match="body boom") as excinfo:
        with _deployed(cli):
            raise RuntimeError("body boom")
    note = excinfo.value.__notes__[0]
    assert "env cleanup failed (resources may be left behind):\nexit 2" in note


def test_deployed_mark_clean_skips_the_safety_net_cleanup():
    cli = _StubCli()
    with _deployed(cli) as deployment:
        cli("env", "cleanup", timeout=1)  # the body's own cleanup step
        deployment.mark_clean()
    assert cli.steps() == ["init", "setup", "cleanup"]


def test_deployed_mark_clean_then_raise_still_cleans_up_and_reraises():
    cli = _StubCli()
    boom = RuntimeError("after mark_clean")
    with pytest.raises(RuntimeError) as excinfo:
        with _deployed(cli) as deployment:
            deployment.mark_clean()
            raise boom
    assert excinfo.value is boom
    assert cli.steps() == ["init", "setup", "cleanup"]


def test_deployed_without_mark_clean_still_cleans_up_after_a_body_cleanup():
    """A body that ran cleanup but did not confirm the result still gets the safety net."""
    cli = _StubCli()
    with _deployed(cli):
        cli("env", "cleanup", timeout=1)
    assert cli.steps() == ["init", "setup", "cleanup", "cleanup"]


def test_deployed_body_failure_before_mark_clean_cleans_up_and_reraises():
    cli = _StubCli()
    with pytest.raises(AssertionError, match="leftover"):
        with _deployed(cli) as deployment:
            assert False, "leftover"  # noqa: B011
            deployment.mark_clean()
    assert cli.steps() == ["init", "setup", "cleanup"]


def test_deployed_reads_cleanup_timeout_lazily():
    order: list[str] = []
    cli = _StubCli()
    with deployed(
        cli,
        init_args=["env", "init"],
        setup_args=["env", "setup"],
        cleanup_args=["env", "cleanup"],
        init_timeout=1,
        setup_timeout=lambda: order.append("setup-timeout") or 2,
        cleanup_timeout=lambda: order.append("cleanup-timeout") or 3,
    ):
        order.append("body")
    assert order == ["setup-timeout", "body", "cleanup-timeout"]


def test_deployed_cleanup_without_deadline_passes_none():
    """With no readable manifest the cleanup timeout is None and reaches the CLI call as is."""
    cli = _StubCli()
    with deployed(
        cli,
        init_args=["env", "init"],
        setup_args=["env", "setup"],
        cleanup_args=["env", "cleanup"],
        init_timeout=1,
        setup_timeout=lambda: 2,
        cleanup_timeout=lambda: step_timeouts_for([]).cleanup,
    ):
        pass
    assert cli.calls[-1] == ("cleanup", None, False)


def test_run_without_timeout_waits_for_the_cli(monkeypatch, tmp_path):
    _use_program(monkeypatch, "import time; time.sleep(0.2); print('done')")
    result = run_aws_bench(["env", "cleanup"], cwd=tmp_path, env=dict(os.environ), timeout=None)
    assert (result.returncode, result.stdout) == (0, "done\n")


# --------------------------------------------------------------------------- #
# settings and source arguments
# --------------------------------------------------------------------------- #

ACCOUNT_CONFIG = """\
schema_version: "1.0"
mode: preexisting
name: n
runner_role: R
cfn_role: C
accounts:
  ec2-multiregion:
    PRIMARY: "111122223333"
"""


@pytest.fixture
def account_config(tmp_path) -> Path:
    path = tmp_path / "accounts.yaml"
    path.write_text(ACCOUNT_CONFIG)
    return path


def _make_dataset(root: Path, scenario: str = "ec2-multiregion") -> Path:
    (root / "scenarios" / scenario).mkdir(parents=True)
    (root / "scenarios" / scenario / "scenario.toml").write_text("")
    (root / "tasks" / scenario).mkdir(parents=True)
    return root


@pytest.mark.parametrize(
    ("environ", "reason"),
    [
        ({}, "opt-in"),
        ({"AWS_BENCH_INTEG": "0"}, "opt-in"),
        ({"AWS_BENCH_INTEG": "1"}, "must point to"),
        (
            {"AWS_BENCH_INTEG": "1", "AWS_BENCH_INTEG_ACCOUNT_CONFIG": "/nonexistent.yaml"},
            "not a file",
        ),
    ],
)
def test_settings_skip_without_opt_in(environ, reason):
    with pytest.raises(pytest.skip.Exception, match=reason):
        load_settings(environ, "r1")


def test_settings_registry_mode(account_config):
    s = load_settings(
        {
            "AWS_BENCH_INTEG": "1",
            "AWS_BENCH_INTEG_ACCOUNT_CONFIG": str(account_config),
            "AWS_BENCH_INTEG_TASKS": " *a , ,*b ",
            "AWS_BENCH_INTEG_ENV_NAME": "n",
        },
        "r1",
    )
    assert (s.run_id, s.env_name, s.dataset) == ("r1", "n", "aws-bench-quickstart")
    assert s.dataset_path is None
    assert s.account_config == account_config.resolve()
    assert s.task_globs == ("*a", "*b")
    assert s.task_filter_args() == ["-i", "*a", "-i", "*b"]
    assert s.env_source_args() == ["-d", "aws-bench-quickstart"]
    assert s.run_source_args() == ["-d", "aws-bench-quickstart"]


@pytest.mark.parametrize(
    ("scenario_var", "scenario"), [(None, "ec2-multiregion"), ("other-scn", "other-scn")]
)
def test_settings_local_mode(account_config, tmp_path, scenario_var, scenario):
    ds = _make_dataset(tmp_path / "ds", scenario)
    environ = {
        "AWS_BENCH_INTEG": "1",
        "AWS_BENCH_INTEG_ACCOUNT_CONFIG": str(account_config),
        "AWS_BENCH_INTEG_DATASET_PATH": str(ds),
    }
    if scenario_var:
        environ["AWS_BENCH_INTEG_SCENARIO"] = scenario_var
    s = load_settings(environ, "r1")
    ds = ds.resolve()
    assert (s.dataset_path, s.scenario, s.task_globs) == (ds, scenario, ())
    assert s.task_filter_args() == []
    assert s.env_source_args() == [
        "--scenario-path", str(ds / "scenarios"), "--include-scenarios", scenario,
    ]  # fmt: skip
    assert s.run_source_args() == [
        "--scenario-path", str(ds / "scenarios"), "-p", str(ds / "tasks" / scenario),
    ]  # fmt: skip


def test_settings_custom_registry_dataset(account_config):
    s = load_settings(
        {
            "AWS_BENCH_INTEG": "1",
            "AWS_BENCH_INTEG_ACCOUNT_CONFIG": str(account_config),
            "AWS_BENCH_INTEG_DATASET": "other-ds@1.0",
        }
    )
    assert s.env_source_args() == ["-d", "other-ds@1.0"]
    assert len(s.run_id) == 8


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"AWS_BENCH_INTEG_ENV_NAME": "other"}, "does not match the account config's name 'n'"),
        (
            {"AWS_BENCH_INTEG_DATASET": "x", "AWS_BENCH_INTEG_DATASET_PATH": "{ds}"},
            "Set either AWS_BENCH_INTEG_DATASET or AWS_BENCH_INTEG_DATASET_PATH",
        ),
        (
            {"AWS_BENCH_INTEG_DATASET_PATH": "{ds}", "AWS_BENCH_INTEG_SCENARIO": "missing"},
            "missing .*scenarios/missing/scenario.toml",
        ),
        ({"AWS_BENCH_INTEG_DATASET_PATH": "{empty}"}, "missing .*scenario.toml"),
        ({"AWS_BENCH_INTEG_DATASET_PATH": "{no_tasks}"}, "missing .*tasks/ec2-multiregion"),
    ],
    ids=["env-name-mismatch", "both-dataset", "missing-scenario", "not-a-checkout", "no-tasks"],
)
def test_settings_guards_fail(account_config, tmp_path, extra, message):
    ds = _make_dataset(tmp_path / "ds")
    no_tasks = tmp_path / "no_tasks"
    (no_tasks / "scenarios" / "ec2-multiregion").mkdir(parents=True)
    (no_tasks / "scenarios" / "ec2-multiregion" / "scenario.toml").write_text("")
    paths = {"ds": ds, "empty": tmp_path, "no_tasks": no_tasks}
    environ = {
        "AWS_BENCH_INTEG": "1",
        "AWS_BENCH_INTEG_ACCOUNT_CONFIG": str(account_config),
        **{k: v.format(**paths) for k, v in extra.items()},
    }
    with pytest.raises(pytest.fail.Exception, match=message):
        load_settings(environ, "r1")


@pytest.mark.parametrize(
    "extra",
    [
        {"AWS_BENCH_INTEG_ENV_NAME": "other"},
        {"AWS_BENCH_INTEG_DATASET": "x", "AWS_BENCH_INTEG_DATASET_PATH": "{ds}"},
        {"AWS_BENCH_INTEG_DATASET_PATH": "{empty}"},
        {"AWS_BENCH_INTEG_EXPECTED_TRIALS": "0"},
    ],
    ids=["env-name-mismatch", "both-dataset", "not-a-checkout", "expected-trials-invalid"],
)
def test_settings_failures_never_render_the_environment(account_config, tmp_path, extra):
    """Every settings failure is reported without a traceback.

    pytest renders the arguments of traceback frames, and ``load_settings``
    receives the whole environment, so a traceback could show secret values.
    """
    ds = _make_dataset(tmp_path / "ds")
    environ = {
        "AWS_BENCH_INTEG": "1",
        "AWS_BENCH_INTEG_ACCOUNT_CONFIG": str(account_config),
        "AWS_BEARER_TOKEN_BEDROCK": TOKEN,
        "MY_SERVICE_API_KEY": ENV_TOKEN,
        **{k: v.format(ds=ds, empty=tmp_path) for k, v in extra.items()},
    }
    with pytest.raises(pytest.fail.Exception) as excinfo:
        load_settings(environ, "r1")
    report = _failed_report(excinfo)
    assert TOKEN not in report
    assert ENV_TOKEN not in report


def _opted_in(account_config: Path, **extra: str) -> dict[str, str]:
    return {
        "AWS_BENCH_INTEG": "1",
        "AWS_BENCH_INTEG_ACCOUNT_CONFIG": str(account_config),
        **extra,
    }


@pytest.mark.parametrize(("raw", "expected"), [(None, None), ("", None), ("9", 9), (" 1 ", 1)])
def test_settings_expected_trials(account_config, raw, expected):
    extra = {} if raw is None else {"AWS_BENCH_INTEG_EXPECTED_TRIALS": raw}
    assert load_settings(_opted_in(account_config, **extra), "r1").expected_trials == expected


@pytest.mark.parametrize("raw", ["0", "-1", "abc", "1.5", "9x"])
def test_settings_expected_trials_invalid_fails(account_config, raw):
    environ = _opted_in(account_config, AWS_BENCH_INTEG_EXPECTED_TRIALS=raw)
    with pytest.raises(pytest.fail.Exception, match="must be a positive integer"):
        load_settings(environ, "r1")


def _settings(expected: int | None, globs: tuple[str, ...] = ()) -> IntegSettings:
    return IntegSettings(
        run_id="r", env_name="n", dataset="d", account_config=Path("a"),
        task_globs=globs, expected_trials=expected,
    )  # fmt: skip


def test_expected_trial_count_without_variable_uses_resolver():
    assert expected_trial_count(_settings(None), resolved=3) == 3


def test_expected_trial_count_with_variable_matching_resolver():
    assert expected_trial_count(_settings(9), resolved=9) == 9


@pytest.mark.parametrize("resolved", [1, 10])
def test_expected_trial_count_with_variable_rejects_other_selection(resolved):
    with pytest.raises(pytest.fail.Exception) as excinfo:
        expected_trial_count(_settings(9, ("*list-a",)), resolved=resolved)
    message = str(excinfo.value)
    assert f"AWS_BENCH_INTEG_EXPECTED_TRIALS=9, but the selection resolves {resolved}" in message
    assert "['*list-a']" in message
    assert excinfo.value.pytrace is False  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# expected trials (aws-bench's own dataset resolver)
# --------------------------------------------------------------------------- #

TASK_TOML = """\
[scenario]
scenario_id = "ec2-multiregion"
[agent]
timeout_sec = 600.0
[verifier]
timeout_sec = 240.0
[environment]
build_timeout_sec = 1200.0
"""


def _write_task(directory: Path, toml: str = TASK_TOML, *, valid: bool = True) -> Path:
    directory.mkdir(parents=True)
    (directory / "task.toml").write_text(toml)
    if valid:
        (directory / "instruction.md").write_text("Do it.\n")
        (directory / "environment").mkdir()
        (directory / "environment" / "Dockerfile").write_text("FROM scratch\n")
        (directory / "tests").mkdir()
        (directory / "tests" / "test.sh").write_text("#!/bin/sh\n")
    return directory


def _local_settings(ds: Path, globs: tuple[str, ...] = ()) -> IntegSettings:
    return IntegSettings(
        run_id="r", env_name="n", dataset="unused", account_config=ds,
        task_globs=globs, dataset_path=ds,
    )  # fmt: skip


@pytest.mark.parametrize(
    ("globs", "expected"),
    [((), 3), (("*list-a",), 1), (("*list-*",), 2), (("*list-a", "*other"), 2)],
)
def test_resolve_run_tasks_counts_selected_tasks(tmp_path, globs, expected):
    ds = _make_dataset(tmp_path / "ds")
    for name in ("list-a", "list-b", "other"):
        _write_task(ds / "tasks" / "ec2-multiregion" / name)
    tasks = resolve_run_tasks(_local_settings(ds, globs).run_dataset_config())
    assert tasks.expected_trials == expected
    assert len(tasks.task_dirs) == expected, tasks.note


def test_resolve_run_tasks_glob_matching_nothing_raises(tmp_path):
    ds = _make_dataset(tmp_path / "ds")
    _write_task(ds / "tasks" / "ec2-multiregion" / "list-a")
    with pytest.raises(ValueError, match="No tasks matched"):
        resolve_run_tasks(_local_settings(ds, ("*nope",)).run_dataset_config())


def test_resolve_run_tasks_keeps_count_when_tasks_are_invalid(tmp_path):
    ds = _make_dataset(tmp_path / "ds")
    _write_task(ds / "tasks" / "ec2-multiregion" / "broken", valid=False)
    tasks = resolve_run_tasks(_local_settings(ds).run_dataset_config())
    assert tasks.expected_trials == 1
    assert tasks.task_dirs == ()
    assert "task directories unavailable" in tasks.note


def test_run_dataset_config_registry_mode(tmp_path):
    s = IntegSettings(
        run_id="r", env_name="n", dataset="aws-bench-quickstart@1.2", account_config=tmp_path,
        task_globs=("*a",),
    )  # fmt: skip
    cfg = s.run_dataset_config(download_dir=tmp_path / "cache")
    assert (cfg.name, cfg.version, cfg.task_names) == ("aws-bench-quickstart", "1.2", ["*a"])
    assert cfg.download_dir == tmp_path / "cache"
    assert not cfg.is_local()


# --------------------------------------------------------------------------- #
# run timeout
# --------------------------------------------------------------------------- #

QUICKSTART_TASK_BUDGET = 1200.0 + 600.0 + 240.0


def test_run_timeout_sums_task_budgets_serially(tmp_path):
    dirs = [_write_task(tmp_path / f"t{i}", valid=False) for i in range(3)]
    t = run_timeout_for(dirs)
    assert t.seconds == 3 * QUICKSTART_TASK_BUDGET + RUN_MARGIN_SEC
    assert "sum of 3 task budget(s)" in t.source


def test_run_timeout_for_quickstart_shape(tmp_path):
    """Nine tasks, one with a 240s pre_invoke: the ec2-multiregion quickstart budgets."""
    dirs = [_write_task(tmp_path / f"t{i}", valid=False) for i in range(8)]
    pre = TASK_TOML + "[pre_invoke]\ntimeout_sec = 240.0\n"
    dirs.append(_write_task(tmp_path / "with-pre", pre, valid=False))
    (tmp_path / "with-pre" / "pre_invoke").mkdir()
    t = run_timeout_for(dirs)
    assert t.seconds == 9 * QUICKSTART_TASK_BUDGET + 240.0 + RUN_MARGIN_SEC == 20400.0
    assert FALLBACK_RUN_TIMEOUT_SEC >= t.seconds


def test_run_timeout_counts_phase_dirs_without_section_at_default(tmp_path):
    d = _write_task(tmp_path / "t", valid=False)
    (d / "post_invoke").mkdir()
    # PhaseScript default timeout_sec is 600.
    assert run_timeout_for([d]).seconds == QUICKSTART_TASK_BUDGET + 600.0 + RUN_MARGIN_SEC


def test_run_timeout_counts_declared_post_invoke(tmp_path):
    toml = TASK_TOML + "[post_invoke]\ntimeout_sec = 100.0\n"
    d = _write_task(tmp_path / "t", toml, valid=False)
    assert run_timeout_for([d]).seconds == QUICKSTART_TASK_BUDGET + 100.0 + RUN_MARGIN_SEC


@pytest.mark.parametrize(
    ("toml", "reason"),
    [("not = [toml", "TOMLDecodeError"), ("[agent]\ntimeout_sec = 1.0\n", "ValidationError"),
     (TASK_TOML.replace("timeout_sec = 600.0\n", ""), "no agent timeout")],
    ids=["bad-toml", "invalid-model", "no-agent-timeout"],
)  # fmt: skip
def test_run_timeout_assumes_fallback_budget_for_unusable_task(tmp_path, toml, reason):
    good = _write_task(tmp_path / "good", valid=False)
    bad = _write_task(tmp_path / "bad", toml, valid=False)
    t = run_timeout_for([good, bad])
    assert t.seconds == QUICKSTART_TASK_BUDGET + FALLBACK_TASK_BUDGET_SEC + RUN_MARGIN_SEC
    assert f"bad: {reason}" in t.source


def test_run_timeout_missing_task_toml_uses_fallback_budget(tmp_path):
    t = run_timeout_for([tmp_path / "missing"])
    assert t.seconds == FALLBACK_TASK_BUDGET_SEC + RUN_MARGIN_SEC
    assert "FileNotFoundError" in t.source


def test_run_timeout_fallback_without_tasks():
    t = run_timeout_for([])
    assert t.seconds == FALLBACK_RUN_TIMEOUT_SEC
    assert "fallback" in t.source


def test_fallback_task_budget_covers_quickstart_task():
    assert FALLBACK_TASK_BUDGET_SEC >= QUICKSTART_TASK_BUDGET + 240.0


# --------------------------------------------------------------------------- #
# step timeouts
# --------------------------------------------------------------------------- #

EC2_MULTIREGION_TOML = """\
schema_version = "1.0"
[scenario]
name = "ec2-multiregion"
description = "d"
account_tags = ["PRIMARY"]
regions = ["us-east-1", "us-west-1", "us-west-2"]
[environment]
build_timeout_sec = 600.0
[deploy]
timeout_sec = 1800.0
[verify]
timeout_sec = 120.0
[cleanup]
timeout_sec = 5400.0
"""


def _write_manifest(directory: Path, text: str) -> Path:
    directory.mkdir(parents=True)
    path = directory / "scenario.toml"
    path.write_text(text)
    return path


def test_timeouts_from_ec2_multiregion_budgets(tmp_path):
    path = _write_manifest(tmp_path / "ec2-multiregion", EC2_MULTIREGION_TOML)
    t = step_timeouts_for([path])
    assert t.setup == 600.0 + 1800.0 + SETUP_MARGIN_SEC == 4200.0
    assert t.cleanup == 600.0 + 5400.0 + CLEANUP_MARGIN_SEC == 7800.0
    assert str(path) in t.source


def test_timeouts_sum_budgets_across_scenarios(tmp_path):
    a = _write_manifest(tmp_path / "a", EC2_MULTIREGION_TOML)
    big = EC2_MULTIREGION_TOML.replace("5400.0", "9000.0").replace("1800.0", "3000.0")
    b = _write_manifest(tmp_path / "b", big)
    t = step_timeouts_for([a, b])
    assert t.setup == (600.0 + 1800.0) + (600.0 + 3000.0) + SETUP_MARGIN_SEC
    assert t.cleanup == (600.0 + 5400.0) + (600.0 + 9000.0) + CLEANUP_MARGIN_SEC


def test_timeouts_use_manifest_defaults_for_missing_tables(tmp_path):
    minimal = EC2_MULTIREGION_TOML.split("[environment]")[0]
    t = step_timeouts_for([_write_manifest(tmp_path / "m", minimal)])
    # ScenarioManifest defaults: build 600, deploy 600, cleanup 300.
    assert t.setup == 1200.0 + SETUP_MARGIN_SEC
    assert t.cleanup == 900.0 + CLEANUP_MARGIN_SEC


def test_timeouts_fallback_without_manifests():
    """No manifest: setup keeps its fallback, cleanup gets no outer deadline at all."""
    t = step_timeouts_for([])
    assert (t.setup, t.cleanup) == (FALLBACK_SETUP_TIMEOUT_SEC, None)
    assert "fallback" in t.source
    assert "env cleanup without a deadline" in t.describe()


def test_fallback_covers_known_ec2_multiregion_budgets():
    assert FALLBACK_SETUP_TIMEOUT_SEC >= 600.0 + 1800.0 + SETUP_MARGIN_SEC


def test_describe_with_manifest_shows_both_deadlines(tmp_path):
    path = _write_manifest(tmp_path / "ec2-multiregion", EC2_MULTIREGION_TOML)
    assert step_timeouts_for([path]).describe().startswith("env setup 4200s, env cleanup 7800s (")


@pytest.mark.parametrize("bad", ["not = [valid toml", "schema_version = '1.0'\n"])
def test_timeouts_fallback_when_manifest_unreadable(tmp_path, bad):
    path = _write_manifest(tmp_path / "bad", bad)
    t = step_timeouts_for([path])
    assert (t.setup, t.cleanup) == (FALLBACK_SETUP_TIMEOUT_SEC, None)
    assert "skipped unreadable" in t.source


def test_timeouts_skip_unreadable_but_use_readable(tmp_path):
    good = _write_manifest(tmp_path / "good", EC2_MULTIREGION_TOML)
    bad = _write_manifest(tmp_path / "bad", "oops = [")
    t = step_timeouts_for([bad, good, tmp_path / "missing.toml"])
    assert t.cleanup == 7800.0
    assert "skipped unreadable" in t.source


def test_find_manifests_local_mode(tmp_path):
    path = _write_manifest(tmp_path / "ds" / "scenarios" / "ec2-multiregion", EC2_MULTIREGION_TOML)
    _write_manifest(tmp_path / "ds" / "scenarios" / "other", EC2_MULTIREGION_TOML)
    found = find_scenario_manifests(
        dataset_path=tmp_path / "ds", scenario="ec2-multiregion", home=tmp_path / "home"
    )
    assert found == [path]


def test_find_manifests_local_mode_missing(tmp_path):
    found = find_scenario_manifests(dataset_path=tmp_path, scenario="nope", home=tmp_path)
    assert found == []


def test_find_manifests_registry_mode_reads_cli_cache(tmp_path):
    cache = tmp_path / "home" / ".aws-bench" / "cache" / "scenarios"
    path = _write_manifest(cache / "abc123" / "ec2-multiregion", EC2_MULTIREGION_TOML)
    found = find_scenario_manifests(
        dataset_path=None, scenario="ec2-multiregion", home=tmp_path / "home"
    )
    assert found == [path]


def test_find_manifests_registry_mode_empty_cache(tmp_path):
    assert find_scenario_manifests(dataset_path=None, scenario="x", home=tmp_path) == []


# --------------------------------------------------------------------------- #
# env show / env list output: rendered with the CLI's own display functions
# --------------------------------------------------------------------------- #

CONFIGURED = {"ec2-multiregion": {"PRIMARY": "111122223333"}, "other": {"A": "444455556666"}}
BOOTSTRAP = {"name": "CDKToolkit", "status": "CREATE_COMPLETE", "region": "us-east-1"}
HEALTHY = [
    BOOTSTRAP,
    {
        "name": "ec2-multiregion-EC2-ls9fuhb522-us-west-2",
        "status": "UPDATE_COMPLETE",
        "region": "us-west-2",
    },
]
OTHER_HEALTHY = [{"name": "other-stack", "status": "CREATE_COMPLETE", "region": "us-east-1"}]


def _console(width: int = CLI_COLUMNS, *, color: bool = False):
    return Console(file=io.StringIO(), width=width, force_terminal=color, highlight=False)


def _account(scenario: str, tag: str, account_id: str, status: str = "ACTIVE"):
    return ScenarioAccount(
        account_id=account_id, email="", scenario_name=scenario, account_tag=tag, status=status
    )


def _render_show(
    *,
    name: str = "integ-env",
    count: int | None = 2,
    panels: list[tuple[tuple[str, str, str, str], list[dict[str, str]]]] | None = None,
    width: int = CLI_COLUMNS,
    color: bool = False,
    quota_error: str | None = None,
    quotas: list[QuotaEntry] | None = None,
) -> str:
    """Mirror `env show` in pre-existing mode: header panel, then one panel per account."""
    if panels is None:
        panels = [
            (("ec2-multiregion", "PRIMARY", "111122223333", "ACTIVE"), HEALTHY),
            (("other", "A", "444455556666", "ACTIVE"), OTHER_HEALTHY),
        ]
    console = _console(width, color=color)
    render_env_header(
        console, "000000000000", "preexisting", name, "preexisting", count,
        account_quota="[dim]externally managed[/dim]",
    )  # fmt: skip
    for (scenario, tag, account_id, status), stacks in panels:
        console.print(
            render_account_state(
                _account(scenario, tag, account_id, status), quotas or [], quota_error, stacks
            )
        )
    return console.file.getvalue()  # type: ignore[attr-defined]


def _render_list(rows: list[dict], *, color: bool = False) -> str:
    console = _console(color=color)
    render_env_list(console, "preexisting", "111122223333", rows)
    return console.file.getvalue()  # type: ignore[attr-defined]


@pytest.mark.parametrize("color", [False, True])
def test_parse_env_show_reads_header_accounts_and_stacks(color):
    shown = parse_env_show("INFO some log line\n" + _render_show(color=color))
    assert shown == EnvShow(
        env_name="integ-env",
        scenario_accounts=2,
        accounts=(
            ShownAccount(
                "ec2-multiregion", "PRIMARY", "111122223333", "ACTIVE",
                stacks=(
                    ShownStack("CDKToolkit", "CREATE_COMPLETE", "us-east-1"),
                    ShownStack(
                        "ec2-multiregion-EC2-ls9fuhb522-us-west-2", "UPDATE_COMPLETE", "us-west-2"
                    ),
                ),
            ),
            ShownAccount(
                "other", "A", "444455556666", "ACTIVE",
                stacks=(ShownStack("other-stack", "CREATE_COMPLETE", "us-east-1"),),
            ),
        ),
    )  # fmt: skip
    assert env_show_problems(shown, env_name="integ-env", accounts=CONFIGURED) == []


def test_plain_strips_ansi_and_hyperlinks():
    text = "\x1b[1;36mbold\x1b[0m \x1b]8;id=1;https://example.com\x1b\\link\x1b]8;;\x1b\\"
    assert plain(text) == "bold link"


def test_parse_env_show_with_quota_error_still_reads_stacks():
    shown = parse_env_show(_render_show(quota_error="AccessDenied"))
    assert [len(a.stacks) for a in shown.accounts] == [2, 1]


@pytest.mark.parametrize("color", [False, True])
def test_parse_env_show_with_quota_rows_still_reads_stacks(color):
    """Live panels carry a quota table; its rows must not be read as stacks."""
    quotas = [
        QuotaEntry("us-east-1", "L-1216C47A", "Running On-Demand instances", 8, 32, True),
        QuotaEntry("us-west-2", "L-0263D0A3", "EC2-VPC Elastic IPs", 10, 5, False),
    ]
    output = _render_show(quotas=quotas, color=color)
    assert plain(output).count("L-1216C47A") == 2  # the quota table is in both panels
    shown = parse_env_show(output)
    assert [a.stacks for a in shown.accounts] == [
        tuple(ShownStack(**s) for s in HEALTHY),
        tuple(ShownStack(**s) for s in OTHER_HEALTHY),
    ]
    assert env_show_problems(shown, env_name="integ-env", accounts=CONFIGURED) == []


def test_parse_env_show_without_accounts():
    console = _console()
    render_env_header(console, "000000000000", "preexisting", "integ-env", "preexisting")
    console.print("  No scenario accounts found.")
    shown = parse_env_show(console.file.getvalue())  # type: ignore[attr-defined]
    assert shown == EnvShow("integ-env", None, ())
    problems = env_show_problems(shown, env_name="integ-env", accounts=CONFIGURED)
    assert any("scenario accounts, expected 2" in p for p in problems)
    assert any("not shown" in p for p in problems)


@pytest.mark.parametrize("output", ["", "Error: something failed\n"])
def test_env_show_problems_on_empty_output(output):
    problems = env_show_problems(parse_env_show(output), env_name="e", accounts=CONFIGURED)
    assert problems[0] == "no environment header panel"
    assert len(problems) == 3  # header, count, missing accounts


def test_env_show_problems_wrong_name_and_count():
    shown = parse_env_show(_render_show(name="someone-else", count=3))
    problems = env_show_problems(shown, env_name="integ-env", accounts=CONFIGURED)
    assert problems == [
        "header shows environment 'someone-else', expected 'integ-env'",
        "header shows 3 scenario accounts, expected 2",
    ]


def test_env_show_problems_missing_unexpected_and_duplicate_accounts():
    panels = [
        (("ec2-multiregion", "PRIMARY", "111122223333", "ACTIVE"), HEALTHY),
        (("ec2-multiregion", "PRIMARY", "111122223333", "ACTIVE"), HEALTHY),
        (("stray", "B", "111122223333", "ACTIVE"), []),
    ]
    shown = parse_env_show(_render_show(panels=panels))
    problems = env_show_problems(shown, env_name="integ-env", accounts=CONFIGURED)
    assert problems == [
        "accounts shown more than once: [('ec2-multiregion', 'PRIMARY', '********3333')]",
        "configured accounts not shown: [('other', 'A', '********6666')]",
        "accounts shown that are not configured: [('stray', 'B', '********3333')]",
    ]


def test_env_show_problems_inactive_account():
    panels = [
        (("ec2-multiregion", "PRIMARY", "111122223333", "SUSPENDED"), HEALTHY),
        (("other", "A", "444455556666", "ACTIVE"), OTHER_HEALTHY),
    ]
    problems = env_show_problems(
        parse_env_show(_render_show(panels=panels)), env_name="integ-env", accounts=CONFIGURED
    )
    assert problems == ["accounts not ACTIVE: [('ec2-multiregion', 'PRIMARY', 'SUSPENDED')]"]


def _show_problems(primary: list[dict[str, str]], other: list[dict[str, str]]) -> list[str]:
    """Problems for a rendered two-account `env show` with the given stack rows."""
    panels = [
        (("ec2-multiregion", "PRIMARY", "111122223333", "ACTIVE"), primary),
        (("other", "A", "444455556666", "ACTIVE"), other),
    ]
    return env_show_problems(
        parse_env_show(_render_show(panels=panels)), env_name="integ-env", accounts=CONFIGURED
    )


NO_SCENARIO_STACK = (
    "configured accounts without a healthy <scenario>-* stack "
    "(['CREATE_COMPLETE', 'IMPORT_COMPLETE', 'UPDATE_COMPLETE']): "
)
PRIMARY_KEY = "('ec2-multiregion', 'PRIMARY', '********3333')"
OTHER_KEY = "('other', 'A', '********6666')"


def _stack(name: str, status: str = "CREATE_COMPLETE", region: str = "us-west-1"):
    return {"name": name, "status": status, "region": region}


def test_env_show_problems_bootstrap_stack_only():
    """CDKToolkit alone is not evidence that the scenario deployed."""
    assert _show_problems([BOOTSTRAP], [BOOTSTRAP]) == [
        NO_SCENARIO_STACK + f"[{PRIMARY_KEY}, {OTHER_KEY}]"
    ]


@pytest.mark.parametrize(
    "unrelated",
    ["unrelated-personal-stack", "ec2-multiregionx-AppRoles-us-east-1", "other-AppRoles-us-east-1"],
    ids=["unrelated", "prefix-without-dash", "other-scenario"],
)
def test_env_show_problems_only_unrelated_healthy_stack(unrelated):
    """A healthy stack not named `<scenario>-*` does not count for that scenario's account."""
    assert _show_problems([BOOTSTRAP, _stack(unrelated)], OTHER_HEALTHY) == [
        NO_SCENARIO_STACK + f"[{PRIMARY_KEY}]"
    ]


@pytest.mark.parametrize(
    "other",
    [[], [_stack("other-stack", "DELETE_IN_PROGRESS")]],
    ids=["no-stacks", "in-progress-only"],
)
def test_env_show_problems_one_account_without_scenario_stack(other):
    """A healthy stack in one account does not cover another configured account."""
    assert _show_problems(HEALTHY, other) == [NO_SCENARIO_STACK + f"[{OTHER_KEY}]"]


@pytest.mark.parametrize(
    "status",
    [
        "ROLLBACK_COMPLETE",
        "UPDATE_ROLLBACK_COMPLETE",
        "UPDATE_FAILED",
        "CREATE_FAILED",
        "CREATE_IN_PROGRESS",
        "UPDATE_IN_PROGRESS",
        "DELETE_IN_PROGRESS",
    ],
)
def test_env_show_problems_scenario_stack_not_healthy(status):
    """A `<scenario>-*` stack that did not finish deploying is not evidence."""
    primary = [BOOTSTRAP, _stack("ec2-multiregion-EC2-ls9fuhb522-us-west-1", status)]
    assert _show_problems(primary, OTHER_HEALTHY) == [NO_SCENARIO_STACK + f"[{PRIMARY_KEY}]"]


@pytest.mark.parametrize(
    "failed",
    [
        _stack("unrelated-personal-stack", "UPDATE_FAILED"),
        _stack("ec2-multiregion-cloudformation-x-us-west-1", "UPDATE_FAILED"),
        _stack("ec2-multiregion-TigrisService-x-us-west-1", "ROLLBACK_COMPLETE"),
        _stack("danube-eks-cluster-x", "DELETE_FAILED"),
        _stack("CDKToolkit", "UPDATE_ROLLBACK_COMPLETE"),
    ],
    ids=lambda s: f"{s['name']}-{s['status']}",
)
def test_env_show_problems_ignores_failed_stacks_next_to_healthy(failed):
    """Failed or rolled-back stacks are left to `env verify`; some scenarios create them."""
    assert _show_problems([*HEALTHY, failed], OTHER_HEALTHY) == []


def test_cli_display_env_sets_columns_and_drops_color_forcing():
    env = {"HOME": "/h", "TERM": "dumb", "COLUMNS": "80", **dict.fromkeys(COLOR_FORCING_VARS, "1")}
    assert cli_display_env(env) == {"HOME": "/h", "TERM": "dumb", "COLUMNS": str(CLI_COLUMNS)}
    assert set(COLOR_FORCING_VARS) == {
        "FORCE_COLOR",
        "TTY_COMPATIBLE",
        "PY_COLORS",
        "CLICOLOR_FORCE",
    }
    assert env["COLUMNS"] == "80"  # the input is not modified


def test_build_cli_env_applies_display_env_and_scratch_home(tmp_path):
    """The `cli_env` fixture's environment: fixed width, no colour forcing, scratch HOME."""
    real_home, scratch = tmp_path / "home", tmp_path / "scratch"
    (real_home / ".aws").mkdir(parents=True)
    (real_home / ".config" / "tool").mkdir(parents=True)
    scratch.mkdir()
    environ = {
        "HOME": str(real_home),
        "TERM": "dumb",
        "COLUMNS": "80",
        HOME_LINKS_VAR: " .config/tool , ",
        **dict.fromkeys(COLOR_FORCING_VARS, "1"),
    }
    env = build_cli_env(environ, account_config=tmp_path / "accounts.yaml", scratch_home=scratch)
    assert env["COLUMNS"] == str(CLI_COLUMNS)
    assert not set(COLOR_FORCING_VARS) & env.keys()
    assert env["HOME"] == str(scratch)
    assert env["AWSBENCH_ACCOUNT_CONFIG"] == str(tmp_path / "accounts.yaml")
    assert (scratch / ".aws").resolve() == (real_home / ".aws").resolve()
    assert (scratch / ".config" / "tool").resolve() == (real_home / ".config" / "tool").resolve()
    assert not (scratch / ".docker").exists()  # absent in the real HOME, so not linked
    assert environ["HOME"] == str(real_home)  # the input is not modified


def test_env_show_truncated_at_narrow_width_is_not_accepted():
    """At Rich's 80-column pipe default long titles get cut; the check must not pass."""
    panels = [
        (("a-very-long-scenario-name-x" * 2, "PRIMARY", "111122223333", "ACTIVE"), HEALTHY),
    ]
    shown = parse_env_show(_render_show(panels=panels, count=1, width=60))
    accounts = {"a-very-long-scenario-name-x" * 2: {"PRIMARY": "111122223333"}}
    problems = env_show_problems(shown, env_name="integ-env", accounts=accounts)
    assert any("configured accounts not shown" in p for p in problems)


@pytest.mark.parametrize("color", [False, True])
def test_parse_env_list_reads_rows(color):
    out = _render_list(
        [{"name": "integ-env", "ou_id": "preexisting", "account_count": 2}], color=color
    )
    rows = parse_env_list(out)
    assert rows == (ListedEnv("integ-env", "preexisting", 2),)
    assert env_list_problems(rows, env_name="integ-env", accounts=CONFIGURED) == []


def test_parse_env_list_unknown_count_is_none():
    rows = parse_env_list(_render_list([{"name": "e", "ou_id": "ou-1", "account_count": None}]))
    assert rows == (ListedEnv("e", "ou-1", None),)


@pytest.mark.parametrize("output", ["", "No test environments found.\n"])
def test_env_list_problems_without_rows(output):
    assert env_list_problems(parse_env_list(output), env_name="e", accounts=CONFIGURED) == [
        "expected exactly one environment row, got 0: []"
    ]


def test_env_list_problems_more_than_one_row():
    out = _render_list(
        [
            {"name": "integ-env", "ou_id": "preexisting", "account_count": 2},
            {"name": "integ-env", "ou_id": "preexisting", "account_count": 2},
        ]
    )
    problems = env_list_problems(parse_env_list(out), env_name="integ-env", accounts=CONFIGURED)
    assert len(problems) == 1 and problems[0].startswith(
        "expected exactly one environment row, got 2"
    )


def test_env_list_problems_wrong_fields():
    out = _render_list([{"name": "other-env", "ou_id": "ou-abcd", "account_count": None}])
    problems = env_list_problems(parse_env_list(out), env_name="integ-env", accounts=CONFIGURED)
    assert problems == [
        "environment 'other-env' listed, expected 'integ-env'",
        f"OU id 'ou-abcd', expected {PREEXISTING_OU_ID!r}",
        "None accounts listed, expected 2",
    ]


# --------------------------------------------------------------------------- #
# Fixture wiring of the integration suite (run in a nested pytest)
# --------------------------------------------------------------------------- #


def _nested_pytest(args: list[str], env_extra: dict[str, str], *, timeout: float = 60) -> str:
    """Run pytest on the integration tests in a subprocess and return its output.

    Colour-forcing variables, coverage hooks and every inherited
    ``AWS_BENCH_INTEG*`` and ``PYTEST_*`` variable (``PYTEST_ADDOPTS``,
    ``PYTEST_TIMEOUT``, ``PYTEST_PLUGINS``, ...) are dropped, so the session
    depends only on what the arguments and ``env_extra`` set and the output
    can be searched as plain text.
    ``-o addopts=`` lifts the default deselection of the ``integration`` marker.
    """
    drop = {"FORCE_COLOR", "TTY_COMPATIBLE", "PY_COLORS", "CLICOLOR_FORCE"}
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("COV_CORE", "AWS_BENCH_INTEG", "PYTEST_")) and k not in drop
    }
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.update(env_extra)
    proc = subprocess.run(  # noqa: S603 - fixed argv
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )
    return proc.stdout + proc.stderr


def test_a_test_sharing_the_deployment_checks_the_bedrock_token_first():
    """``env show`` only shares the deployment, yet the token check must precede it.

    ``bedrock_token`` is session-scoped but is set up only for a test that
    requests it, directly or through ``deployed_env``. If ``deployed_env`` did
    not, this test would deploy in a run whose quickstart test already failed
    for lack of a token.
    """
    plan = _nested_pytest(
        [
            "tests/integration/test_acceptance.py",
            "-m",
            "integration",
            "-k",
            "env_show",
            "--setup-plan",
            "-q",
        ],
        {},
    )
    assert "deployed_env" in plan, plan[-2000:]
    assert "bedrock_token" in plan, plan[-2000:]
    assert plan.index("bedrock_token") < plan.index("deployed_env")


def test_cli_env_failure_never_renders_the_environment(account_config, tmp_path):
    """A failure while building the CLI environment shows no credentials, even at -vvv.

    pytest prints the arguments of traceback frames, and ``build_cli_env``
    receives the whole environment. An extra link that overlaps ``.aws`` makes
    it raise ``FileExistsError``.
    """
    home = tmp_path / "home"
    (home / ".aws").mkdir(parents=True)
    out = _nested_pytest(
        [
            "tests/integration/test_acceptance.py",
            "-m",
            "integration",
            "-k",
            "env_list",
            "--tb=long",
            "-vvv",
        ],
        {
            "HOME": str(home),
            "AWS_BENCH_INTEG": "1",
            "AWS_BENCH_INTEG_ACCOUNT_CONFIG": str(account_config),
            "AWS_BENCH_INTEG_HOME_LINKS": ".aws",
            "AWS_SECRET_ACCESS_KEY": TOKEN,
            "AWS_SESSION_TOKEN": ENV_TOKEN,
        },
    )
    assert "FileExistsError" in out, out[-3000:]  # the failure path was reached
    assert TOKEN not in out
    assert ENV_TOKEN not in out


# --------------------------------------------------------------------------- #
# Post-cleanup stack check
# --------------------------------------------------------------------------- #
REGIONS = {"ec2-multiregion": ["us-east-1", "us-west-2"], "other": ["us-east-1"]}


def _acct(scenario, tag, account_id, stacks=(), errors=(), regions=None):
    return AccountStacks(
        scenario,
        tag,
        account_id,
        tuple(regions if regions is not None else REGIONS[scenario]),
        stacks=tuple(StackRecord(*x) for x in stacks),
        errors=tuple(errors),
    )


def _both(primary_stacks=(), other_stacks=(), **kw):
    return [
        _acct("ec2-multiregion", "PRIMARY", "111122223333", primary_stacks, **kw),
        _acct("other", "A", "444455556666", other_stacks),
    ]


def _problems(found):
    return stack_problems(found, accounts=CONFIGURED, regions_by_scenario=REGIONS)


def test_stack_problems_empty_accounts_pass():
    assert _problems(_both()) == []


def test_stack_problems_deleted_stacks_pass():
    found = _both([("us-east-1", "ec2-multiregion-EC2-a-us-east-1", "DELETE_COMPLETE")])
    assert _problems(found) == []


@pytest.mark.parametrize("name", ["CDKToolkit", "cdk-hnb659fds-assets-x"])
def test_stack_problems_healthy_bootstrap_stack_fails(name):
    found = _both([("us-west-2", name, "CREATE_COMPLETE")])
    assert _problems(found) == [
        f"ec2-multiregion/PRIMARY (********3333): us-west-2: CDK bootstrap stack {name} "
        "is CREATE_COMPLETE"
    ]


def test_stack_problems_leftover_scenario_stack_fails():
    found = _both([("us-east-1", "ec2-multiregion-EC2-a-us-east-1", "CREATE_COMPLETE")])
    assert _problems(found) == [
        "ec2-multiregion/PRIMARY (********3333): us-east-1: "
        "stack ec2-multiregion-EC2-a-us-east-1 is CREATE_COMPLETE"
    ]


def test_stack_problems_masks_account_ids_in_stack_names():
    """A stack name can carry an account ID; failure messages must not repeat it."""
    found = _both([("us-east-1", "ec2-multiregion-123456789012-us-east-1", "CREATE_COMPLETE")])
    (problem,) = _problems(found)
    assert "123456789012" not in problem
    assert "ec2-multiregion-********9012-us-east-1" in problem


def test_deployed_stack_problems_masks_account_ids_in_scenario_names():
    scenario = "s-123456789012"
    found = [AccountStacks(scenario, "A", "111122223333", ("us-east-1",))]
    problems = deployed_stack_problems(
        found,
        accounts={scenario: {"A": "111122223333"}},
        regions_by_scenario={scenario: ["us-east-1"]},
    )
    assert problems, "an account without a deployment must be reported"
    assert all("123456789012" not in p for p in problems)


def test_stack_problems_leftover_in_non_primary_region_fails():
    found = _both([("us-west-2", "ec2-multiregion-EC2-b-us-west-2", "UPDATE_COMPLETE")])
    (problem,) = _problems(found)
    assert "us-west-2" in problem and "UPDATE_COMPLETE" in problem


@pytest.mark.parametrize("status", ["DELETE_FAILED", "DELETE_IN_PROGRESS"])
def test_stack_problems_stack_not_yet_gone_fails(status):
    found = _both([("us-east-1", "ec2-multiregion-EC2-a-us-east-1", status)])
    (problem,) = _problems(found)
    assert problem.endswith(f"is {status}")


@pytest.mark.parametrize(
    "status", ["DELETE_FAILED", "DELETE_IN_PROGRESS", "CREATE_FAILED", "ROLLBACK_COMPLETE"]
)
def test_stack_problems_bootstrap_in_a_bad_state_fails(status):
    found = _both([("us-east-1", "CDKToolkit", status)])
    assert len(_problems(found)) == 1


def test_stack_problems_deleted_bootstrap_stack_passes():
    assert _problems(_both([("us-east-1", "CDKToolkit", "DELETE_COMPLETE")])) == []


def test_stack_problems_one_account_with_leftover_among_clean_ones():
    found = _both(other_stacks=[("us-east-1", "other-stack", "CREATE_COMPLETE")])
    (problem,) = _problems(found)
    assert problem.startswith("other/A (********6666)") and "other-stack" in problem


def test_stack_problems_listing_error_is_a_problem_not_an_empty_region():
    found = _both(errors=["us-west-2: ClientError AccessDenied"])
    assert _problems(found) == [
        "ec2-multiregion/PRIMARY (********3333): could not list stacks: "
        "us-west-2: ClientError AccessDenied"
    ]


def test_stack_problems_credentials_error_is_a_problem():
    found = _both(errors=["credentials: ClientError AccessDenied"])
    assert len(_problems(found)) == 1


def test_stack_problems_no_accounts_checked_fails():
    problems = _problems([])
    assert len(problems) == 1 and problems[0].startswith("configured accounts not checked")


def test_stack_problems_empty_config_fails():
    assert stack_problems([], accounts={}, regions_by_scenario={}) == [
        "the account config maps no accounts: nothing was checked"
    ]


def test_stack_problems_unchecked_duplicate_and_unconfigured_accounts_fail():
    first = _acct("ec2-multiregion", "PRIMARY", "111122223333")
    extra = _acct("other", "B", "000000000000")
    problems = _problems([first, first, extra])
    assert any(p.startswith("accounts checked more than once") for p in problems)
    assert any(p.startswith("configured accounts not checked: other/A") for p in problems)
    assert any(p.startswith("accounts checked that are not configured") for p in problems)


def test_stack_problems_fewer_regions_than_declared_fails():
    found = [
        _acct("ec2-multiregion", "PRIMARY", "111122223333", regions=["us-east-1"]),
        _acct("other", "A", "444455556666"),
    ]
    (problem,) = _problems(found)
    assert "queried regions ['us-east-1'], scenario declares ['us-east-1', 'us-west-2']" in problem


def test_stack_problems_scenario_without_declared_regions_fails():
    found = _both()
    problems = stack_problems(
        found, accounts=CONFIGURED, regions_by_scenario={"other": REGIONS["other"]}
    )
    assert len(problems) == 1 and "no declared regions" in problems[0]


def test_stack_problems_never_prints_a_full_account_id():
    found = _both([("us-east-1", "x", "CREATE_COMPLETE")], errors=["us-west-2: boom"])
    text = "\n".join(_problems(found))
    assert "111122223333" not in text and "444455556666" not in text


_EXPECTED = {"ec2-multiregion": {"PRIMARY": "111122223333"}}
_HEALTHY = ("us-east-1", "ec2-multiregion-EC2-a-us-east-1", "CREATE_COMPLETE")


_DECLARED = {"ec2-multiregion": ["us-east-1", "us-west-2"]}


def _before(*stacks, errors=(), found=None, regions=None):
    entry = AccountStacks(
        "ec2-multiregion",
        "PRIMARY",
        "111122223333",
        ("us-east-1", "us-west-2"),
        stacks=tuple(StackRecord(*x) for x in stacks),
        errors=tuple(errors),
    )
    return deployed_stack_problems(
        [entry] if found is None else found,
        accounts=_EXPECTED,
        regions_by_scenario=_DECLARED if regions is None else regions,
    )


@pytest.mark.parametrize("status", ["CREATE_COMPLETE", "UPDATE_COMPLETE", "IMPORT_COMPLETE"])
def test_deployed_stack_problems_healthy_scenario_stack_passes(status):
    assert _before(("us-west-2", "ec2-multiregion-EC2-b-us-west-2", status)) == []


def test_deployed_stack_problems_empty_account_fails():
    (problem,) = _before()
    assert problem.startswith("ec2-multiregion/PRIMARY (********3333): no healthy stack named")


@pytest.mark.parametrize(
    "stacks",
    [
        [("us-east-1", "unrelated-stack", "CREATE_COMPLETE")],
        [("us-east-1", "CDKToolkit", "CREATE_COMPLETE")],
        [("us-east-1", "ec2-multiregion-x", "CREATE_IN_PROGRESS")],
        [("us-east-1", "ec2-multiregion-x", "ROLLBACK_COMPLETE")],
        [("us-east-1", "ec2-multiregion-x", "DELETE_COMPLETE")],
        [("us-east-1", "ec2-multiregionx", "CREATE_COMPLETE")],
    ],
    ids=["unrelated", "bootstrap-only", "in-progress", "rolled-back", "deleted", "prefix-only"],
)
def test_deployed_stack_problems_no_healthy_scenario_stack_fails(stacks):
    (problem,) = _before(*stacks)
    assert "no healthy stack named 'ec2-multiregion-*'" in problem


def test_deployed_stack_problems_listing_error_fails_even_with_a_healthy_stack():
    problems = _before(_HEALTHY, errors=["us-west-2: ClientError AccessDenied"])
    assert problems == [
        "ec2-multiregion/PRIMARY (********3333): could not list stacks: "
        "us-west-2: ClientError AccessDenied"
    ]


def test_deployed_stack_problems_unchecked_account_and_empty_config_fail():
    assert _before(found=[]) == [
        "configured accounts not checked: ec2-multiregion/PRIMARY (********3333)"
    ]
    assert deployed_stack_problems([], accounts={}, regions_by_scenario={}) == [
        "the account config maps no accounts: nothing was checked"
    ]


def _record(
    *stacks, errors=(), account_id="111122223333", scenario="ec2-multiregion", tag="PRIMARY"
):
    return AccountStacks(
        scenario,
        tag,
        account_id,
        ("us-east-1", "us-west-2"),
        stacks=tuple(StackRecord(*x) for x in stacks),
        errors=tuple(errors),
    )


def test_deployed_stack_problems_duplicate_record_fails():
    record = _record(_HEALTHY)
    (problem,) = _before(found=[record, record])
    assert problem == "accounts checked more than once: ec2-multiregion/PRIMARY (********3333)"


def test_deployed_stack_problems_conflicting_duplicate_does_not_hide_a_listing_error():
    broken = _record(errors=["us-west-2: ClientError AccessDenied"])
    healthy = _record(_HEALTHY)
    for found in ([broken, healthy], [healthy, broken]):
        problems = _before(found=found)
        assert any(p.startswith("accounts checked more than once") for p in problems)
        assert any(
            "could not list stacks: us-west-2: ClientError AccessDenied" in p for p in problems
        )


def test_deployed_stack_problems_unconfigured_record_fails():
    other = _record(_HEALTHY, account_id="000000000000", tag="OTHER")
    assert _before(found=[_record(_HEALTHY), other]) == [
        "accounts checked that are not configured: ec2-multiregion/OTHER (********0000)"
    ]


def test_deployed_stack_problems_unreadable_manifest_is_reported_as_a_manifest_problem():
    problems = _before(_HEALTHY, regions={})
    assert len(problems) == 1
    assert "no declared regions for scenario 'ec2-multiregion' (manifest?)" in problems[0]
    assert "did not deploy" not in problems[0]


def test_deployed_stack_problems_fewer_regions_than_declared_fails():
    entry = AccountStacks(
        "ec2-multiregion", "PRIMARY", "111122223333", ("us-east-1",), (StackRecord(*_HEALTHY),)
    )
    (problem,) = _before(found=[entry])
    assert "queried regions ['us-east-1'], scenario declares ['us-east-1', 'us-west-2']" in problem


def test_deployed_stack_problems_never_prints_a_full_account_id():
    assert "111122223333" not in "\n".join(_before(errors=["us-west-2: boom"]))


def test_error_label_without_an_error_code():
    assert _error_label(ClientError({}, "ListStacks")) == "ClientError"
    assert _error_label(ClientError({"Error": {"Message": "m"}}, "ListStacks")) == "ClientError"
    assert _error_label(ClientError({"Error": {"Code": ""}}, "ListStacks")) == "ClientError"


def test_mask_account_id_keeps_only_the_last_four_digits():
    assert mask_account_id("111122223333") == "********3333"
    assert mask_account_id("not-an-id") == "****"
    assert mask_account_ids("a 111122223333 b 1234 c 1111222233334") == (
        "a ********3333 b 1234 c 1111222233334"
    )


class _FakeCfn:
    def __init__(self, pages, error=None):
        self._pages, self._error = pages, error

    def get_paginator(self, name):
        assert name == "list_stacks"
        return self

    def paginate(self, **kwargs):
        assert not kwargs, "no status filter: DELETE_IN_PROGRESS must be visible"
        if self._error:
            raise self._error
        return iter(self._pages)


class _FakeSession:
    def __init__(self, by_region):
        self._by_region = by_region
        self.services = []

    def client(self, service, region_name):
        self.services.append(service)
        return self._by_region[region_name]


def _page(*stacks):
    return {"StackSummaries": [{"StackName": n, "StackStatus": s} for n, s in stacks]}


def test_collect_reads_every_page_of_every_region():
    session = _FakeSession(
        {
            "us-east-1": _FakeCfn([_page(("a", "CREATE_COMPLETE")), _page(("b", "DELETE_FAILED"))]),
            "us-west-2": _FakeCfn([_page()]),
        }
    )
    got = collect_account_stacks(
        scenario="s", tag="T", account_id="111122223333",
        regions=["us-east-1", "us-west-2"], session_for=lambda _id: session,
    )  # fmt: skip
    assert got.stacks == (
        StackRecord("us-east-1", "a", "CREATE_COMPLETE"),
        StackRecord("us-east-1", "b", "DELETE_FAILED"),
    )
    assert got.errors == () and set(session.services) == {"cloudformation"}


def test_collect_records_a_region_error_by_code_only():
    denied = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "arn:aws:iam::111122223333:role/x"}},
        "ListStacks",
    )
    session = _FakeSession(
        {"us-east-1": _FakeCfn([_page()]), "us-west-2": _FakeCfn([], error=denied)}
    )
    got = collect_account_stacks(
        scenario="s", tag="T", account_id="111122223333",
        regions=["us-east-1", "us-west-2"], session_for=lambda _id: session,
    )  # fmt: skip
    assert got.errors == ("us-west-2: ClientError AccessDenied",)
    assert "111122223333" not in repr(got.errors)


def test_collect_records_a_credentials_error_without_its_message():
    def session_for(_id):
        raise RuntimeError("token=CANARY-TOKEN-0123456789 arn:aws:iam::111122223333:role/x")

    got = collect_account_stacks(
        scenario="s", tag="T", account_id="111122223333",
        regions=["us-east-1"], session_for=session_for,
    )  # fmt: skip
    assert got.errors == ("credentials: RuntimeError",) and got.stacks == ()


@pytest.fixture
def moto_cfn(monkeypatch):
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(var, "testing")
    with mock_aws():
        yield boto3.Session(region_name="us-east-1")


_TEMPLATE = '{"Resources": {"Topic": {"Type": "AWS::SNS::Topic"}}}'  # one trivial resource


def test_collect_against_moto_sees_a_stack_until_it_is_deleted(moto_cfn):
    cfn = moto_cfn.client("cloudformation", region_name="us-west-2")
    cfn.create_stack(StackName="ec2-multiregion-x", TemplateBody=_TEMPLATE)

    def check():
        got = collect_account_stacks(
            scenario="ec2-multiregion", tag="PRIMARY", account_id="111122223333",
            regions=["us-east-1", "us-west-2"], session_for=lambda _id: moto_cfn,
        )  # fmt: skip
        return stack_problems(
            [got],
            accounts={"ec2-multiregion": {"PRIMARY": "111122223333"}},
            regions_by_scenario=REGIONS,
        )

    (problem,) = check()
    assert "us-west-2: stack ec2-multiregion-x is CREATE_COMPLETE" in problem
    cfn.delete_stack(StackName="ec2-multiregion-x")
    assert check() == []


def test_scenario_regions_reads_declared_regions_and_skips_unreadable(tmp_path):
    good = _write_manifest(tmp_path / "good", EC2_MULTIREGION_TOML)
    bad = _write_manifest(tmp_path / "bad", "not = [toml")
    assert scenario_regions([good, bad, tmp_path / "missing.toml"]) == {
        "ec2-multiregion": ["us-east-1", "us-west-1", "us-west-2"]
    }


def test_cleanup_output_problems_accepts_the_success_summary():
    out = "Cleanup summary\nCleanup complete: 2 account(s) cleaned successfully.\n"
    assert cleanup_output_problems(out, accounts=2) == []


@pytest.mark.parametrize(
    "out",
    [
        "",
        "Cleanup complete: 1 account(s) cleaned successfully.\n",
        "Cleanup completed with errors: 1 succeeded, 1 failed.\n",
        "Cleanup cancelled.\n",
    ],
)
def test_cleanup_output_problems_rejects_other_output(out):
    assert cleanup_output_problems(out, accounts=2) != []


def test_cleanup_output_problems_flags_errors_even_beside_a_success_line():
    out = (
        "Cleanup complete: 2 account(s) cleaned successfully.\n"
        "Cleanup completed with errors: 1 succeeded, 1 failed.\n"
    )
    assert cleanup_output_problems(out, accounts=2) == ["the cleanup summary reports errors"]


# --------------------------------------------------------------------------- #
# Account IDs never reach a failure message, a report or a log
# --------------------------------------------------------------------------- #
# A canary that is not a placeholder, put where text from AWS, the CLI or the
# config can carry an ID: names, paths, ARNs, CLI output, an account config.
CANARY = "987654321098"
MASKED = "********1098"


# Every run of six digits taken from the canary: a cut-off ID leaves one of these.
CANARY_FRAGMENTS = tuple(sorted({CANARY[i : i + 6] for i in range(len(CANARY) - 5)}))


def _leaks(text: str) -> list[str]:
    """Lines of ``text`` that show the canary or any run of six of its digits."""
    return [
        line[:200]
        for line in text.splitlines()
        if CANARY in line or any(fragment in line for fragment in CANARY_FRAGMENTS)
    ]


def _clean(text: str) -> None:
    """Fail if ``text`` shows the canary ID or any six consecutive digits of it (a cut-off ID)."""
    assert CANARY not in text
    leaks = _leaks(text)
    assert not leaks, "\n".join(leaks)


CANARY_PRINTER = """
import sys
print("arn:aws:iam::987654321098:role/x stack s-987654321098")
print("err 987654321098", file=sys.stderr)
sys.exit(2)
"""


def test_cli_failure_message_masks_the_whole_output_tail(monkeypatch, tmp_path):
    _use_program(monkeypatch, CANARY_PRINTER)
    with pytest.raises(pytest.fail.Exception) as excinfo:
        run_aws_bench(
            ("env", "show", "--env-name", f"integ-{CANARY}", "--ve", f"K_TOKEN={TOKEN}"),
            cwd=tmp_path,
            env=dict(os.environ),
            timeout=30,
        )
    report = _failed_report(excinfo)
    _clean(report)
    assert TOKEN not in report
    assert "exited 2" in report
    assert f"arn:aws:iam::{MASKED}:role/x stack s-{MASKED}" in report
    assert f"err {MASKED}" in report
    assert f"integ-{MASKED}" in report  # a name in the command line


def test_timeout_message_masks_the_output_tail(monkeypatch, tmp_path):
    _use_program(
        monkeypatch, "import time; print('stack 987654321098', flush=True); time.sleep(30)"
    )
    with pytest.raises(pytest.fail.Exception) as excinfo:
        run_aws_bench(("env", "setup"), cwd=tmp_path, env=dict(os.environ), timeout=1.5)
    report = _failed_report(excinfo)
    _clean(report)
    assert f"stack {MASKED}" in report


def test_masking_happens_before_the_tail_is_cut(monkeypatch, tmp_path):
    """An ID cut by the 4000-character tail must not leave its last digits visible."""
    _use_program(monkeypatch, "import sys; print('987654321098' + 'y' * 3994); sys.exit(1)")
    with pytest.raises(pytest.fail.Exception) as excinfo:
        run_aws_bench(("env", "setup"), cwd=tmp_path, env=dict(os.environ), timeout=30)
    report = _failed_report(excinfo)
    assert "21098" not in report
    assert "*1098yyy" in report


def test_secret_is_scrubbed_before_ids_are_masked(monkeypatch, tmp_path):
    """A secret that holds an ID-like run is replaced as a whole, not left in pieces."""
    secret = f"CANARY-{CANARY}-SECRET"
    _use_program(monkeypatch, f"import sys; print({secret!r}, 'id 987654321098'); sys.exit(3)")
    with pytest.raises(pytest.fail.Exception) as excinfo:
        run_aws_bench(
            ("run", "--ve", f"AWS_BEARER_TOKEN_BEDROCK={secret}"),
            cwd=tmp_path,
            env=dict(os.environ),
            timeout=30,
        )
    report = _failed_report(excinfo)
    _clean(report)
    assert secret not in report
    assert f"*** id {MASKED}" in report


def test_cli_result_keeps_raw_output_for_parsing_but_never_prints_it(monkeypatch, tmp_path):
    _use_program(monkeypatch, CANARY_PRINTER)
    result = run_aws_bench(
        ("env", "show", "--env-name", f"integ-{CANARY}"),
        cwd=tmp_path,
        env=dict(os.environ),
        timeout=30,
        check=False,
    )
    assert CANARY in result.stdout and CANARY in result.stderr  # the parsers need it
    for text in (repr(result), str(result), f"{result}", result.masked_output()):  # type: ignore[attr-defined]
        _clean(text)
        assert MASKED in text
    assert result.masked_output() == result.output.replace(CANARY, MASKED)  # type: ignore[attr-defined]
    assert len(result.masked_output(tail=10)) == 10  # type: ignore[attr-defined]


def test_leak_notice_masks_the_cleanup_output():
    def cleanup(*args, timeout, check):
        return CliResult(("env", "cleanup"), 3, f"arn:aws:iam::{CANARY}:role/x\n", "")

    cli = _StubCli({"cleanup": cleanup})
    with pytest.raises(pytest.fail.Exception) as excinfo:
        with _deployed(cli):
            pass
    text = str(excinfo.value)
    _clean(text)
    assert f"exit 3\narn:aws:iam::{MASKED}:role/x" in text


def test_leak_notice_attached_to_another_failure_is_masked():
    def cleanup(*args, timeout, check):
        return CliResult(("env", "cleanup"), 3, f"stack {CANARY}\n", "")

    cli = _StubCli({"setup": RuntimeError("setup exploded"), "cleanup": cleanup})
    with pytest.raises(RuntimeError) as excinfo:
        with _deployed(cli):
            pass
    notes = "\n".join(excinfo.value.__notes__)
    _clean(notes)
    assert f"stack {MASKED}" in notes


CANARY_ACCOUNTS = {f"scn-{CANARY}": {"PRIMARY": CANARY}}


def _show_with(env_name: str, scenario: str, tag: str, account_id: str, stacks: list[dict]):
    return parse_env_show(
        _render_show(
            name=env_name, count=1, panels=[((scenario, tag, account_id, "ACTIVE"), stacks)]
        )
    )


def test_env_show_problems_never_print_an_account_id():
    stack = {
        "name": f"scn-{CANARY}-EC2-{CANARY}",
        "status": "ROLLBACK_COMPLETE",
        "region": "us-east-1",
    }
    shown = _show_with(f"integ-{CANARY}", f"scn-{CANARY}", "OTHER", CANARY, [stack])
    problems = env_show_problems(shown, env_name="expected-name", accounts=CANARY_ACCOUNTS)
    text = "\n".join(problems)
    _clean(text)
    assert f"header shows environment 'integ-{MASKED}'" in text
    assert f"('scn-{MASKED}', 'PRIMARY', '{MASKED}')" in text  # configured, not shown
    assert f"('scn-{MASKED}', 'OTHER', '{MASKED}')" in text  # shown, not configured


def test_env_show_problems_without_a_healthy_stack_never_print_an_account_id():
    stack = {"name": f"scn-{CANARY}-EC2", "status": "ROLLBACK_COMPLETE", "region": "us-east-1"}
    shown = _show_with(f"integ-{CANARY}", f"scn-{CANARY}", "PRIMARY", CANARY, [stack])
    problems = env_show_problems(shown, env_name=f"integ-{CANARY}", accounts=CANARY_ACCOUNTS)
    assert len(problems) == 1
    _clean(problems[0])
    assert MASKED in problems[0]


def test_env_list_problems_never_print_an_account_id():
    out = _render_list([{"name": f"env-{CANARY}", "ou_id": f"ou-{CANARY}", "account_count": 5}])
    rows = parse_env_list(out)
    text = "\n".join(env_list_problems(rows, env_name="x", accounts=CANARY_ACCOUNTS))
    _clean(text)
    assert f"environment 'env-{MASKED}' listed" in text
    assert f"OU id 'ou-{MASKED}'" in text
    two = parse_env_list(out + out)
    (many,) = env_list_problems(two, env_name="x", accounts=CANARY_ACCOUNTS)
    _clean(many)
    assert MASKED in many


def _canary_objects() -> dict[str, object]:
    stack = StackRecord("us-east-1", f"s-{CANARY}", "CREATE_COMPLETE")
    account = ShownAccount(
        f"scn-{CANARY}", "A", CANARY, "ACTIVE", (ShownStack(f"s-{CANARY}", "S", "r"),)
    )
    path = Path(f"/tmp/ds-{CANARY}/scenario.toml")
    return {
        "StackRecord": stack,
        "AccountStacks": AccountStacks(f"scn-{CANARY}", "A", CANARY, ("us-east-1",), (stack,), ()),
        "ShownStack": ShownStack(f"s-{CANARY}", "CREATE_COMPLETE", "us-east-1"),
        "ShownAccount": account,
        "EnvShow": EnvShow(f"env-{CANARY}", 1, (account,)),
        "ListedEnv": ListedEnv(f"env-{CANARY}", f"ou-{CANARY}", 1),
        "CliResult": CliResult(("--env-name", f"e-{CANARY}"), 1, f"out {CANARY}", f"err {CANARY}"),
        "IntegSettings": IntegSettings(
            run_id="r",
            env_name=f"env-{CANARY}",
            dataset="d",
            account_config=Path(f"/tmp/{CANARY}/accounts.yaml"),
            task_globs=(f"*{CANARY}",),
            dataset_path=Path(f"/tmp/ds-{CANARY}"),
            scenario=f"scn-{CANARY}",
        ),
        "StepTimeouts": step_timeouts_for([path]),
        "RunTimeout": run_timeout_for([Path(f"/tmp/task-{CANARY}")]),
        "RunTasks": helpers.RunTasks(1, (Path(f"/tmp/task-{CANARY}"),), f"note {CANARY}"),
        "_Outcome": helpers._Outcome(0, f"out {CANARY}", f"err {CANARY}", f"problem {CANARY}"),
    }


@pytest.mark.parametrize("name", list(_canary_objects()))
def test_dataclass_reprs_mask_account_ids(name):
    """Pytest prints locals (``-l``) and frame arguments with ``repr``."""
    obj = _canary_objects()[name]
    for text in (repr(obj), str(obj), f"{[obj]}", f"{ {'k': obj} }"):
        _clean(text)
        assert MASKED in text


def test_dataclass_masking_does_not_change_equality_or_fields():
    obj = ListedEnv(f"env-{CANARY}", "ou", 1)
    assert obj == ListedEnv(f"env-{CANARY}", "ou", 1)
    assert obj.name == f"env-{CANARY}"


def test_step_timeouts_and_run_timeout_descriptions_mask_paths_and_names(tmp_path):
    manifest = _write_manifest(tmp_path / f"ds-{CANARY}" / "scenario", EC2_MULTIREGION_TOML)
    text = step_timeouts_for([manifest, tmp_path / f"missing-{CANARY}.toml"]).describe()
    _clean(text)
    assert f"ds-{MASKED}" in text
    run = run_timeout_for([tmp_path / f"task-{CANARY}"]).describe()
    _clean(run)
    assert f"task-{MASKED}" in run


def _opted(config: Path, **extra: str) -> dict[str, str]:
    return {"AWS_BENCH_INTEG": "1", "AWS_BENCH_INTEG_ACCOUNT_CONFIG": str(config), **extra}


def _outcome_text(excinfo) -> str:
    assert excinfo.value.pytrace is False
    return str(excinfo.value)


def test_settings_skip_reasons_mask_the_config_path(tmp_path):
    missing = tmp_path / f"{CANARY}" / "accounts.yaml"
    with pytest.raises(pytest.skip.Exception) as excinfo:
        load_settings(_opted(missing), "r1")
    _clean(str(excinfo.value))
    assert f"{MASKED}/accounts.yaml" in str(excinfo.value)


def test_settings_fail_messages_mask_names_and_paths(account_config, tmp_path):
    ds = tmp_path / f"ds-{CANARY}"
    cases = [
        _opted(account_config, AWS_BENCH_INTEG_ENV_NAME=f"other-{CANARY}"),
        _opted(account_config, AWS_BENCH_INTEG_DATASET_PATH=str(ds)),
        _opted(account_config, AWS_BENCH_INTEG_EXPECTED_TRIALS=f"{CANARY}x"),
    ]
    for environ in cases:
        with pytest.raises(pytest.fail.Exception) as excinfo:
            load_settings(environ, "r1")
        text = _outcome_text(excinfo)
        _clean(text)
        assert MASKED in text


def test_settings_mismatch_with_an_env_name_from_the_config_is_masked(tmp_path):
    config = tmp_path / "accounts.yaml"
    config.write_text(ACCOUNT_CONFIG.replace("name: n", f"name: env-{CANARY}"))
    with pytest.raises(pytest.fail.Exception) as excinfo:
        load_settings(_opted(config, AWS_BENCH_INTEG_ENV_NAME="x"), "r1")
    text = _outcome_text(excinfo)
    _clean(text)
    assert f"env-{MASKED}" in text


def test_an_invalid_account_config_fails_without_the_value_it_rejects(tmp_path):
    """The validation error quotes the offending value; it must not reach the report."""
    config = tmp_path / "accounts.yaml"
    config.write_text(ACCOUNT_CONFIG.replace("name: n", f"name: 'bad name {CANARY}'"))
    with pytest.raises(pytest.fail.Exception) as excinfo:
        load_settings(_opted(config), "r1")
    text = _outcome_text(excinfo)
    _clean(text)
    assert "ValidationError" in text and MASKED in text


def test_load_accounts_returns_the_mapping_with_a_masked_repr(tmp_path):
    config = tmp_path / "accounts.yaml"
    config.write_text(ACCOUNT_CONFIG.replace("111122223333", CANARY))
    accounts = helpers.load_accounts(config)  # type: ignore[attr-defined]
    assert accounts == {"ec2-multiregion": {"PRIMARY": CANARY}}
    for text in (repr(accounts), str(accounts)):
        _clean(text)
        assert MASKED in text
    missing = tmp_path / f"{CANARY}.yaml"
    with pytest.raises(pytest.fail.Exception) as excinfo:
        helpers.load_accounts(missing)  # type: ignore[attr-defined]
    _clean(_outcome_text(excinfo))


def test_expected_trial_count_failure_masks_the_globs():
    with pytest.raises(pytest.fail.Exception) as excinfo:
        expected_trial_count(_settings(5, (f"*task-{CANARY}",)), resolved=2)
    text = _outcome_text(excinfo)
    _clean(text)
    assert f"task-{MASKED}" in text


def test_mask_problems_masks_reward_checker_messages():
    """``find_problems`` quotes trial names, which come from task names."""
    spec = importlib.util.spec_from_file_location(
        "check_job_rewards", REPO_ROOT / "scripts" / "check_job_rewards.py"
    )
    assert spec and spec.loader
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    result = {
        "n_total_trials": 1,
        "stats": {
            "n_completed_trials": 1,
            "n_errored_trials": 0,
            "evals": {
                f"oracle__{CANARY}": {
                    "n_trials": 1,
                    "reward_stats": {"reward": {"0.0": [f"task-{CANARY}__abc"]}},
                }
            },
        },
    }
    raw = checker.find_problems(result, expected_trials=1)
    assert CANARY in "\n".join(raw)  # the checker itself reports names as they are
    masked = helpers.mask_problems(raw)  # type: ignore[attr-defined]
    _clean("\n".join(masked))
    assert f"trial task-{MASKED}__abc: reward=0.0" in masked[0]


def test_masked_dict_repr_hides_ids_and_keeps_the_contents():
    data = helpers.MaskedDict({"k": {"t": CANARY}})  # type: ignore[attr-defined]
    assert data == {"k": {"t": CANARY}}
    _clean(repr(data))
    assert MASKED in repr(data)


def test_masked_dict_keeps_inner_mappings_masked_too(tmp_path):
    config = tmp_path / "accounts.yaml"
    config.write_text(ACCOUNT_CONFIG.replace("111122223333", CANARY))
    accounts = helpers.load_accounts(config)  # type: ignore[attr-defined]
    inner = accounts["ec2-multiregion"]
    assert isinstance(inner, helpers.MaskedDict)  # type: ignore[attr-defined]
    _clean(repr(inner))
    assert inner == {"PRIMARY": CANARY}


def test_masked_list_masks_its_text_and_keeps_the_values():
    items = helpers.MaskedList(["env", f"--env-name=n-{CANARY}", "-d"])  # type: ignore[attr-defined]
    assert items == ["env", f"--env-name=n-{CANARY}", "-d"]
    assert list(items) == ["env", f"--env-name=n-{CANARY}", "-d"]
    assert items[1] == f"--env-name=n-{CANARY}"
    assert json.loads(json.dumps(items)) == ["env", f"--env-name=n-{CANARY}", "-d"]
    for text in (repr(items), str(items), f"{items}"):
        _clean(text)
        assert f"n-{MASKED}" in text


def test_masked_str_masks_only_its_repr():
    text = helpers.MaskedStr(f"stack s-{CANARY}\n")
    assert text == f"stack s-{CANARY}\n"  # the parsers read the raw text
    assert str(text) == f"stack s-{CANARY}\n"
    assert repr(text) == repr(f"stack s-{MASKED}\n")
    assert helpers.parse_env_list(text) == ()


def test_cli_env_is_a_dict_whose_repr_hides_the_values(tmp_path):
    environ = {"HOME": str(tmp_path), "AWS_SECRET_ACCESS_KEY": TOKEN, "FORCE_COLOR": "1"}
    env = build_cli_env(environ, account_config=tmp_path / "a.yaml", scratch_home=tmp_path / "h")
    assert type(env) is helpers.SecretDict and isinstance(env, dict)  # usable as subprocess env
    assert env["AWS_SECRET_ACCESS_KEY"] == TOKEN and "FORCE_COLOR" not in env
    for text in (repr(env), str(env), f"{env}"):
        assert TOKEN not in text and str(tmp_path) not in text


def test_env_command_is_the_env_step_with_the_name_then_the_source():
    settings = IntegSettings(
        run_id="r",
        env_name="integ-env",
        dataset="ds",
        account_config=Path("a"),
        dataset_path=Path("/data/checkout"),
        scenario="scn",
    )
    assert settings.env_command("cleanup", "--yes", "--quiet") == [
        "env", "cleanup", "--env-name", "integ-env",
        "--scenario-path", "/data/checkout/scenarios", "--include-scenarios", "scn",
        "--yes", "--quiet",
    ]  # fmt: skip
    registry = replace(settings, dataset_path=None)
    assert registry.env_command("init") == ["env", "init", "--env-name", "integ-env", "-d", "ds"]


# --- the real test modules, run against a fake CLI, in a nested pytest ---------------

FAKE_CLI = """
import os, sys
argv = sys.argv[1:]
key = "_".join(argv[:2]).upper() if argv[0] == "env" else argv[0].upper()
sys.stdout.write(os.environ.get("FAKE_OUT_" + key, ""))
sys.stderr.write(os.environ.get("FAKE_ERR_" + key, ""))
if argv[0] == "run":
    folder = os.path.join(argv[argv.index("--jobs-dir") + 1], argv[argv.index("--job-name") + 1])
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "result.json"), "w") as f:
        f.write(os.environ["FAKE_RESULT_JSON"])
sys.exit(int(os.environ.get("FAKE_RC_" + key, "0")))
"""

FAKE_PLUGIN = f"""
import logging, os, sys
import pytest
from tests.integration import helpers, test_acceptance, test_env_cleanup
helpers.CLI_PROGRAM = (sys.executable, "-c", {FAKE_CLI!r})
CANARY = {CANARY!r}
MODES = os.environ.get("FAKE_MODE", "").split(",")
class _Provider:
    def __init__(self):
        if "provider-error" in MODES:
            raise RuntimeError(f"provider arn:aws:iam::{{CANARY}}:role/x")
        if "provider-log" in MODES:
            logging.getLogger("fake").warning("credential lookup for %s failed", CANARY)
    def get_session_for_account(self, *args):
        return object()
def _collect(*, scenario, tag, account_id, regions, session_for):
    stack = helpers.StackRecord(
        "us-east-1", f"{{scenario}}-EC2-{{account_id}}-us-east-1", "CREATE_COMPLETE"
    )
    return helpers.AccountStacks(scenario, tag, account_id, tuple(regions), stacks=(stack,))
def _chained(cfg):
    try:
        raise KeyError(f"inner {{CANARY}}")
    except KeyError as cause:
        error = RuntimeError(f"resolver arn:aws:iam::{{CANARY}}:role/x")
        error.add_note(f"note {{CANARY}}")
        raise error from cause
if "parser-raises" in MODES:
    helpers.plain = lambda text: None  # parse_env_show then fails in its own frame
def pytest_runtest_setup(item):
    if "skip-raw" in MODES and "env_list" in item.name:
        pytest.skip(f"skip reason {{CANARY}}")
test_env_cleanup.CredentialProvider = _Provider
test_env_cleanup.collect_account_stacks = _collect
if "resolver-chain" in MODES:
    test_acceptance.resolve_run_tasks = _chained
elif "resolver-real" not in MODES:
    test_acceptance.resolve_run_tasks = lambda cfg: helpers.RunTasks(1, ())
"""
NESTED_CONFIG = f"""\
schema_version: "1.0"
mode: preexisting
name: integ-{CANARY}
runner_role: R
cfn_role: C
accounts:
  scn-{CANARY}:
    PRIMARY: "{CANARY}"
"""

NESTED_MANIFEST = EC2_MULTIREGION_TOML.replace(
    'name = "ec2-multiregion"', f'name = "scn-{CANARY}"'
).replace('regions = ["us-east-1", "us-west-1", "us-west-2"]', 'regions = ["us-east-1"]')

NESTED_RESULT = {
    "n_total_trials": 1,
    "stats": {
        "n_completed_trials": 1,
        "n_errored_trials": 0,
        "evals": {
            f"oracle__scn-{CANARY}": {
                "n_trials": 1,
                "n_errors": 0,
                "reward_stats": {"reward": {"0.0": [f"task-{CANARY}__abc"]}},
            }
        },
    },
}
CLEANUP_SUMMARY = "Cleanup complete: 1 account(s) cleaned successfully.\n"


def _nested_env(root: Path, extra: dict[str, str]) -> dict[str, str]:
    scenario = f"scn-{CANARY}"
    dataset = root / f"ds-{CANARY}"
    _write_manifest(dataset / "scenarios" / scenario, NESTED_MANIFEST)
    (dataset / "tasks" / scenario).mkdir(parents=True)
    config = root / f"cfg-{CANARY}" / "accounts.yaml"
    config.parent.mkdir()
    config.write_text(NESTED_CONFIG)
    (root / "plugin").mkdir()
    (root / "plugin" / "fake_cli_plugin.py").write_text(FAKE_PLUGIN)
    (root / "home").mkdir()
    stack = {"name": f"{scenario}-EC2-{CANARY}", "status": "CREATE_COMPLETE", "region": "us-east-1"}
    shown = _render_show(
        name=f"integ-{CANARY}", count=1, panels=[((scenario, "OTHER", CANARY, "ACTIVE"), [stack])]
    )
    listed = _render_list(
        [{"name": f"integ-{CANARY}", "ou_id": f"ou-{CANARY}", "account_count": 1}]
    )
    return {
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(root / "plugin")]),
        "HOME": str(root / "home"),
        "AWS_BENCH_INTEG": "1",
        "AWS_BENCH_INTEG_ACCOUNT_CONFIG": str(config),
        "AWS_BENCH_INTEG_DATASET_PATH": str(dataset),
        "AWS_BENCH_INTEG_SCENARIO": scenario,
        "AWS_BENCH_INTEG_TASKS": f"*task-{CANARY}",
        "AWS_BEARER_TOKEN_BEDROCK": "dummy-bedrock-value-0001",
        "FAKE_OUT_ENV_SHOW": shown,
        "FAKE_OUT_ENV_LIST": listed,
        "FAKE_ERR_ENV_SHOW": f"warning for {CANARY}\n",
        "FAKE_RESULT_JSON": json.dumps(NESTED_RESULT),
        **extra,
    }


# name -> (extra environment, strings the report must show masked)
NESTED_SCENARIOS = {
    # The CLI succeeds; every check fails. The cleanup output has no summary.
    "checks-fail": (
        {"FAKE_OUT_ENV_CLEANUP": f"stack {CANARY} not removed\n"},
        [
            f"accounts shown that are not configured: [('scn-{MASKED}', 'OTHER', '{MASKED}')]",
            f"OU id 'ou-{MASKED}'",
            f"trial task-{MASKED}__abc: reward=0.0",
            f"stack {MASKED} not removed",
            f"ds-{MASKED}",  # the `step timeouts:` line
        ],
    ),
    # The cleanup summary is fine; a stack is left behind.
    "stacks-left": (
        {"FAKE_OUT_ENV_CLEANUP": CLEANUP_SUMMARY},
        [f"scn-{MASKED}/PRIMARY ({MASKED}): us-east-1: stack scn-{MASKED}-EC2-{MASKED}-us-east-1"],
    ),
    # `env setup` and `env cleanup` fail; their output carries the ID.
    "cli-fails": (
        {
            "FAKE_RC_ENV_SETUP": "4",
            "FAKE_OUT_ENV_SETUP": f"setup arn:aws:iam::{CANARY}:role/x\n",
            "FAKE_ERR_ENV_SETUP": f"setup err {CANARY}\n",
            "FAKE_RC_ENV_CLEANUP": "3",
            "FAKE_OUT_ENV_CLEANUP": f"cleanup arn:aws:iam::{CANARY}:role/x\n",
        },
        [
            "exited 4",
            f"setup arn:aws:iam::{MASKED}:role/x",
            f"setup err {MASKED}",
            "Also, env cleanup failed",
            f"exit 3\ncleanup arn:aws:iam::{MASKED}:role/x",
        ],
    ),
}
NESTED_FLAGS = {
    "no-locals": ["--tb=long", "-vvv", "-rA"],
    "locals": ["--tb=long", "-vvv", "-rA", "-l", "--full-trace"],
}


@pytest.mark.timeout(240)
@pytest.mark.parametrize("flags", NESTED_FLAGS.values(), ids=NESTED_FLAGS.keys())
@pytest.mark.parametrize("scenario", NESTED_SCENARIOS)
def test_integration_modules_never_print_an_account_id(scenario, flags, tmp_path):
    """The real acceptance tests, against a fake CLI, print the canary nowhere.

    ``-rA`` adds the captured output of passing tests, ``-vvv`` the full
    assertion diffs, ``--tb=long`` the frame arguments and ``-l`` the locals.
    """
    extra, expected = NESTED_SCENARIOS[scenario]
    out = _nested_pytest(
        [
            "tests/integration/test_acceptance.py",
            "tests/integration/test_env_cleanup.py",
            "-m",
            "integration",
            "-p",
            "fake_cli_plugin",
            "--color=no",
            *flags,
        ],  # fmt: skip
        _nested_env(tmp_path, extra),
        timeout=200,
    )
    assert " failed" in out or " error" in out, out[-3000:]  # the failure paths were reached
    for text in expected:
        assert text in out, text
    leaks = _leaks(out)
    assert not leaks, "\n".join(leaks)


# --- the pytest report: exceptions, logs, skips and fixture failures from called code ---
# The checks above stub the called code. These cases make it misbehave, so the
# report hook in ``tests/integration/conftest.py`` is what keeps the ID out.
ACCEPTANCE = "tests/integration/test_acceptance.py"
CLEANUP = "tests/integration/test_env_cleanup.py"
LONG = ["--tb=long", "-vvv", "-rA"]
LOCALS = ["--tb=long", "-vvv", "-rA", "-l", "--full-trace"]
TB_STYLES = [[f"--tb={style}", "-rA"] for style in ("short", "native", "line", "no")]
_MASKED_ARN = f"arn:aws:iam::{MASKED}:role/x"


def _link_aws_dir(root: Path) -> dict[str, str]:
    """An extra home link that overlaps ``.aws``: ``cli_env`` then raises ``FileExistsError``."""
    (root / "home" / ".aws").mkdir()
    return {"AWS_BENCH_INTEG_HOME_LINKS": ".aws"}


def _bad_config(root: Path) -> dict[str, str]:
    config = root / f"bad-{CANARY}.yaml"
    config.write_text(NESTED_CONFIG.replace(f"name: integ-{CANARY}", f"name: 'bad name {CANARY}'"))
    return {"AWS_BENCH_INTEG_ACCOUNT_CONFIG": str(config)}


# 500 characters of raw CLI output with the canary every 13 characters, so pytest's
# 240-character cut of a frame argument splits an ID wherever it falls.
LONG_STDOUT = (CANARY + "-") * 38 + "\n"
assert len(LONG_STDOUT) > 400

# name -> (files and selection, environment, flag sets, strings the report must show)
CHANNEL_SCENARIOS = {
    # The suite's own long raw string (the CLI's stdout) is a frame argument of the
    # parser that fails. pytest cuts a repr longer than 240 characters unless -vv.
    "long-raw-stdout": (
        [ACCEPTANCE, "-k", "env_show"],
        lambda root: {"FAKE_MODE": "parser-raises", "FAKE_OUT_ENV_SHOW": LONG_STDOUT},
        [["-rA"], ["-v", "-rA"], ["-l", "-rA"]],
        ["AttributeError", f"{MASKED}-{MASKED}"],
    ),
    # The real resolver raises "No tasks matched the filter(s) ['*task-<id>']".
    "resolver-real": (
        [ACCEPTANCE, "-k", "quickstart"],
        lambda root: {"FAKE_MODE": "resolver-real"},
        [LONG, LOCALS],
        ["No tasks matched", "task-********1098"],
    ),
    # An exception with a cause and a note, raised by the resolver.
    "resolver-chain": (
        [ACCEPTANCE, "-k", "quickstart"],
        lambda root: {"FAKE_MODE": "resolver-chain"},
        [LONG, LOCALS],
        [f"RuntimeError: resolver {_MASKED_ARN}", f"inner {MASKED}", f"note {MASKED}"],
    ),
    # ``CredentialProvider()`` raises; every traceback style and the summary show it masked.
    "provider-error": (
        [CLEANUP],
        lambda root: {"FAKE_MODE": "provider-error"},
        [LONG, LOCALS, *TB_STYLES],
        [f"RuntimeError: provider {_MASKED_ARN}"],
    ),
    # An ID in a log record; the failure itself is ID-free (a stack is left behind).
    "provider-log": (
        [CLEANUP],
        lambda root: {"FAKE_MODE": "provider-log", "FAKE_OUT_ENV_CLEANUP": CLEANUP_SUMMARY},
        [LONG, LOCALS],
        [f"credential lookup for {MASKED} failed"],
    ),
    # A fixture failure: the account config sits in a directory named after the account.
    "cli-env-failure": (
        [ACCEPTANCE, "-k", "env_list"],
        _link_aws_dir,
        [LONG, ["--tb=long", "-vvv", "-rA", "-l"], LOCALS],
        ["FileExistsError", f"cfg-{MASKED}"],
    ),
    "env-name-mismatch": (
        [ACCEPTANCE, "-k", "env_list"],
        lambda root: {"AWS_BENCH_INTEG_ENV_NAME": f"other-{CANARY}"},
        [LOCALS],
        [f"other-{MASKED}", f"integ-{MASKED}"],
    ),
    "invalid-config": (
        [ACCEPTANCE, "-k", "env_list"],
        _bad_config,
        [LOCALS],
        ["cannot load the account config", "ValidationError", f"bad name {MASKED}"],
    ),
    "missing-dataset": (
        [ACCEPTANCE, "-k", "env_list"],
        lambda root: {"AWS_BENCH_INTEG_DATASET_PATH": str(root / f"missing-{CANARY}")},
        [LOCALS],
        ["missing", f"missing-{MASKED}"],
    ),
    "trial-count": (
        [ACCEPTANCE, "-k", "quickstart"],
        lambda root: {"AWS_BENCH_INTEG_EXPECTED_TRIALS": "7"},
        [LOCALS],
        [f"task-{MASKED}"],
    ),
    # A skip whose reason carries an ID that did not come through the suite's masking.
    "skip-reason": (
        [ACCEPTANCE, "-k", "env_list"],
        lambda root: {"FAKE_MODE": "skip-raw"},
        [["-rs"]],
        [f"skip reason {MASKED}"],
    ),
    "skip-config-path": (
        [ACCEPTANCE, "-k", "env_list"],
        lambda root: {"AWS_BENCH_INTEG_ACCOUNT_CONFIG": str(root / f"none-{CANARY}.yaml")},
        [["-rs"]],
        [f"none-{MASKED}.yaml is not a file"],
    ),
}
CHANNEL_CASES = [
    pytest.param(name, tuple(flags), id=f"{name}-{'+'.join(flags)}")
    for name, (_, _, flag_sets, _) in CHANNEL_SCENARIOS.items()
    for flags in flag_sets
]


@pytest.mark.timeout(240)
@pytest.mark.parametrize(("name", "flags"), CHANNEL_CASES)
def test_pytest_report_masks_ids_from_called_code(name, flags, tmp_path):
    """Exceptions, log records, skips and fixture failures never show the canary.

    Covers the terminal report (traceback, frame arguments, locals, captured
    sections, the short summary, skip reasons) and the ``--junitxml`` file, and
    that the credentials in the session's environment appear in neither.
    The called code is made to misbehave, so the report hook in
    ``tests/integration/conftest.py`` is what masks these cases, except
    ``skip-config-path`` (``load_settings`` masks that reason itself) and
    ``long-raw-stdout`` (the suite's own string: ``MaskedStr`` masks its repr
    before pytest cuts it, which the hook cannot do afterwards).
    """
    selection, make_env, _, expected = CHANNEL_SCENARIOS[name]
    env = _nested_env(
        tmp_path,
        {"COLUMNS": "300", "AWS_SECRET_ACCESS_KEY": TOKEN, "AWS_SESSION_TOKEN": ENV_TOKEN},
    )  # COLUMNS: the short summary is cut to the width
    env.update(make_env(tmp_path))
    report = tmp_path / "report.xml"
    out = _nested_pytest(
        [*selection, "-m", "integration", "-p", "fake_cli_plugin", "--color=no",
         f"--junitxml={report}", *flags],
        env,
        timeout=200,
    )  # fmt: skip
    for text in expected:
        assert text in out, f"{text}\n{out[-3000:]}"
    for secret in (TOKEN, ENV_TOKEN):  # the credentials in the environment of the session
        assert secret not in out, secret
        assert secret not in report.read_text(), secret
    leaks = _leaks(out)
    assert not leaks, "\n".join(leaks)
    _clean(report.read_text())


@pytest.mark.timeout(240)  # above the 200 s of the nested session, which is the one that fires
def test_live_output_shows_no_account_id(tmp_path):
    """``-s`` adds no exposure: the suite masks what it prints, and the CLI's output is piped."""
    out = _nested_pytest(
        [ACCEPTANCE, CLEANUP, "-m", "integration", "-p", "fake_cli_plugin", "--color=no",
         "--tb=short", "-s", "-rA"],
        _nested_env(tmp_path, NESTED_SCENARIOS["checks-fail"][0]),
        timeout=200,
    )  # fmt: skip
    assert f"stack {MASKED} not removed" in out
    _clean(out)


REPORT_PROBE = f"""
import logging, sys, pytest
ID = {CANARY!r}

def test_call_failure():
    print("out", ID)
    print("err", ID, file=sys.stderr)
    logging.getLogger("probe").warning("log %s", ID)
    assert False, f"boom {{ID}}"

@pytest.fixture
def broken():
    raise ValueError(f"fixture {{ID}}")

def test_setup_error(broken):
    pass

@pytest.fixture
def late():
    yield
    raise ValueError(f"teardown {{ID}}")

def test_teardown_error(late):
    pass

def test_skipped():
    pytest.skip(f"skip {{ID}}")

@pytest.mark.xfail(reason=f"xfail {{ID}}")
def test_xfailed():
    assert False

@pytest.mark.xfail(reason=f"xpass {{ID}}")
def test_xpassed():
    pass

@pytest.mark.xfail(strict=True, reason=f"strict {{ID}}")
def test_strict_xpassed():
    pass
"""


@pytest.mark.timeout(90)  # above the 60 s of the nested session, which is the one that fires
def test_report_hook_masks_every_report_field(tmp_path):
    """The hook masks the setup, call and teardown reports of every outcome it lists.

    Outcomes: failure, setup error, teardown error, skip, xfail, XPASS and strict
    XPASS; fields: traceback, short summary, captured stdout, stderr and log
    sections, skip and xfail reasons, junit XML. The hook is loaded as a plugin
    so it applies to a probe file outside ``tests/integration``. The masked
    markers must be present (so the hook ran and the short summary kept its
    message), the outcome counts must be those of the probe, and the canary must
    be absent from the terminal output and the junit XML.
    """
    (tmp_path / "test_probe.py").write_text(REPORT_PROBE)
    report = tmp_path / "report.xml"
    out = _nested_pytest(
        [str(tmp_path / "test_probe.py"), "--rootdir", str(tmp_path),
         "-p", "tests.integration.conftest",
         "--color=no", "-rA", "--tb=long", "-l", f"--junitxml={report}"],
        {"COLUMNS": "300"},
        timeout=60,
    )  # fmt: skip
    for marker in (
        f"AssertionError: boom {MASKED}",  # call failure: traceback and short summary
        f"ValueError: fixture {MASKED}",  # setup error
        f"ValueError: teardown {MASKED}",  # teardown error
        f"out {MASKED}",  # captured stdout section
        f"err {MASKED}",  # captured stderr section
        f"log {MASKED}",  # captured log section
        f"skip {MASKED}",  # skip reason
        f"xfail {MASKED}",  # xfail reason
        f"xpass {MASKED}",  # XPASS reason
        f"[XPASS(strict)] strict {MASKED}",  # strict XPASS failure
    ):
        assert marker in out, f"{marker}\n{out[-3000:]}"
    assert "Captured stderr call" in out
    assert f"test_call_failure - AssertionError: boom {MASKED}" in out  # the short summary
    # the outcomes are those of the probe: the hook changed none of them
    assert "2 failed, 1 passed, 1 skipped, 1 xfailed, 1 xpassed, 2 errors" in out, out[-600:]
    _clean(out)
    xml = report.read_text()
    _clean(xml)
    assert f"boom {MASKED}" in xml


class _Boom(Exception):
    pass


def _hook_report(request, error: BaseException):
    """A failed ``call`` report for ``request.node``, run through the real report hook.

    Returns the report before the hook (its plain-text rendering) and after it.
    """
    from _pytest.reports import TestReport
    from pluggy import Result

    from tests.integration import conftest as integration_conftest

    def raise_error():
        raise error

    call = pytest.CallInfo.from_call(raise_error, "call")
    report = TestReport.from_item_and_call(request.node, call)
    before = _render(report.longrepr)
    hook = integration_conftest.pytest_runtest_makereport(request.node, call)
    next(hook)
    with pytest.raises(StopIteration):
        hook.send(Result.from_call(lambda: report))
    return before, report


def _render(longrepr) -> str:
    """What the terminal prints for ``longrepr``, without colour and without any stripping."""
    from _pytest._io import TerminalWriter

    out = io.StringIO()
    writer = TerminalWriter(out)
    writer.hasmarkup = False
    longrepr.toterminal(writer)
    return out.getvalue()


def test_report_without_an_id_is_unchanged_by_the_hook(request):
    before, report = _hook_report(request, _Boom("no identifier here"))
    assert before.strip()
    assert _render(report.longrepr) == before  # the same text, whitespace included
    assert str(report.longrepr) == before.strip()  # junit XML gets what pytest would give
    crash = getattr(report.longrepr, "reprcrash", None)
    assert crash is not None
    assert crash.message == "tests.test_integration_helpers._Boom: no identifier here"


def test_report_with_an_id_is_masked_in_its_text_str_and_reprcrash(request):
    _, report = _hook_report(request, _Boom(f"boom {CANARY}"))
    crash = getattr(report.longrepr, "reprcrash", None)
    assert crash is not None
    for text in (_render(report.longrepr), str(report.longrepr), crash.message):
        assert f"boom {MASKED}" in text
        _clean(text)


def test_report_round_tripped_through_serialization_is_masked_but_loses_its_summary_line(request):
    """A report that is serialized (pytest-xdist, report-log) keeps the masking, not ``reprcrash``.

    The restored ``longrepr`` is the masked text as a plain ``str``; pytest then
    has no one-line message for the short summary. Pinned so a change is noticed.
    """
    from _pytest.reports import TestReport

    _, report = _hook_report(request, _Boom(f"boom {CANARY}"))
    restored = TestReport._from_json(report._to_json())
    assert isinstance(restored.longrepr, str)
    assert f"boom {MASKED}" in restored.longrepr
    _clean(restored.longrepr)
    assert getattr(restored.longrepr, "reprcrash", None) is None  # no short-summary message


UNIT_PROBE = f"""
def test_fails():
    raise AssertionError("unit {CANARY}")
"""


def test_the_hook_applies_only_below_the_directory_of_its_conftest(tmp_path):
    """A test outside ``tests/integration`` keeps its report; one below it is masked."""
    (tmp_path / "integ").mkdir()
    (tmp_path / "integ" / "conftest.py").write_text(
        "from tests.integration.conftest import pytest_runtest_makereport  # noqa: F401\n"
    )
    (tmp_path / "integ" / "test_in.py").write_text(UNIT_PROBE.replace("unit", "integ"))
    (tmp_path / "unit").mkdir()
    (tmp_path / "unit" / "test_out.py").write_text(UNIT_PROBE)
    out = _nested_pytest(
        [str(tmp_path), "--rootdir", str(tmp_path), "--color=no", "-rf", "--tb=short"],
        {"COLUMNS": "300"},
        timeout=60,
    )
    assert f"AssertionError: integ {MASKED}" in out, out[-1500:]
    assert f"AssertionError: unit {CANARY}" in out, out[-1500:]  # untouched: still the raw ID
    assert f"integ {CANARY}" not in out
