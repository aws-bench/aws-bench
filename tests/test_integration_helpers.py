"""Unit tests for ``tests/integration/helpers.py``.

Kept outside ``tests/integration`` on purpose: everything there is marked
``integration`` and deselected by default, but these must run in ``make test``.
Secret-handling tests check the full report of an inner pytest session (run as
a subprocess), because that is what a developer or CI log actually shows.
"""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tests.integration import helpers
from tests.integration.helpers import (
    CLEANUP_MARGIN_SEC,
    FALLBACK_RUN_TIMEOUT_SEC,
    FALLBACK_SETUP_TIMEOUT_SEC,
    FALLBACK_TASK_BUDGET_SEC,
    RUN_MARGIN_SEC,
    SETUP_MARGIN_SEC,
    CliResult,
    IntegSettings,
    Secret,
    SecretEnv,
    deployed,
    env_secret_values,
    expected_trial_count,
    find_scenario_manifests,
    load_settings,
    redact_args,
    resolve_run_tasks,
    run_aws_bench,
    run_timeout_for,
    scrub,
    secret_values,
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
    env = {k: v for k, v in os.environ.items() if not k.startswith("COV_CORE")}
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
