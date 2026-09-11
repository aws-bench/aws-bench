"""Tests for aws_bench.scenario.container.

Mocks the docker CLI by patching asyncio.create_subprocess_exec so the
lifecycle (build dedup, start with bind-mount, run_phase, stop) can be
exercised without a real Docker daemon. Phase outputs land on the host
through the bind mount, so tests pre-seed the host_logs_dir to mimic
what the in-container script would have written.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aws_bench.scenario.config import EnvironmentConfig
from aws_bench.scenario.container import (
    DockerCLIError,
    ScenarioContainer,
    sanitize_container_name,
    sanitize_image_tag,
)
from aws_bench.scenario.paths import ScenarioPaths

VALID_TOML = """\
schema_version = "1.0"

[scenario]
name = "{name}"
account_tags = ["PRIMARY"]
regions = ["us-east-1"]
"""


def _make_scenario_dir(root: Path, *, with_verify: bool = False) -> Path:
    sd = root / "sc"
    sd.mkdir()
    (sd / "scenario.toml").write_text(VALID_TOML.format(name="sc"))
    (sd / "scenario").mkdir()
    (sd / "scenario" / "Dockerfile").write_text("FROM alpine\n")
    (sd / "deploy").mkdir()
    (sd / "deploy" / "deploy.sh").write_text("#!/bin/sh\necho deploy\n")
    if with_verify:
        (sd / "verify").mkdir()
        (sd / "verify" / "verify.sh").write_text("#!/bin/sh\nexit 0\n")
    return sd


Responder = Callable[[list[str], "bytes | None"], "tuple[int, bytes, bytes]"]


class FakeDocker:
    """Patches asyncio.create_subprocess_exec to simulate the docker CLI.

    Tests register matchers like ``fake.when("exec", rc=0)`` to declare
    "any docker exec ... call returns rc=0". Calls are recorded for
    assertion.
    """

    def __init__(self) -> None:
        self._matchers: list[tuple[Callable[[list[str]], bool], Responder]] = []
        self.calls: list[tuple[list[str], bytes | None]] = []
        self._patch = None

    def when(
        self,
        *prefix: str,
        rc: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        def matches(args: list[str]) -> bool:
            return tuple(args[: len(prefix)]) == tuple(prefix)

        def respond(_a: list[str], _s: bytes | None) -> tuple[int, bytes, bytes]:
            return rc, stdout, stderr

        self._matchers.append((matches, respond))

    def when_each(
        self,
        *prefix: str,
        responses: Iterable[tuple[int, bytes, bytes]],
    ) -> None:
        it = iter(responses)

        def matches(args: list[str]) -> bool:
            return tuple(args[: len(prefix)]) == tuple(prefix)

        def respond(_a: list[str], _s: bytes | None) -> tuple[int, bytes, bytes]:
            return next(it)

        self._matchers.append((matches, respond))

    def when_callable(self, *prefix: str, responder: Responder) -> None:
        def matches(args: list[str]) -> bool:
            return tuple(args[: len(prefix)]) == tuple(prefix)

        self._matchers.append((matches, responder))

    def __enter__(self) -> "FakeDocker":  # noqa: D105
        self._patch = patch(
            "aws_bench.scenario.container.asyncio.create_subprocess_exec",
            new=self._fake_exec,
        )
        self._patch.start()
        return self

    def __exit__(self, *exc) -> None:  # noqa: D105
        if self._patch is not None:
            self._patch.stop()

    async def _fake_exec(self, *cmd: str, stdin=None, stdout=None, stderr=None):
        args = list(cmd[1:])  # strip "docker"
        rc, out_bytes, err_bytes = 0, b"", b""
        for matches, respond in self._matchers:
            if matches(args):
                rc, out_bytes, err_bytes = respond(args, None)
                break
        self.calls.append((args, None))

        async def communicate(input=None):
            self.calls[-1] = (args, input)
            return out_bytes, err_bytes

        proc = MagicMock()
        proc.returncode = rc
        proc.communicate = communicate
        return proc

    def calls_with_prefix(self, *prefix: str) -> list[list[str]]:
        return [args for args, _ in self.calls if tuple(args[: len(prefix)]) == tuple(prefix)]


@pytest.fixture(autouse=True)
def reset_locks():
    """Class-level build locks are shared across tests; reset between cases."""
    ScenarioContainer._image_build_locks.clear()
    yield
    ScenarioContainer._image_build_locks.clear()


@pytest.fixture
def env_config():
    return EnvironmentConfig(cpus=1, memory_mb=512, build_timeout_sec=60)


def _fake_cred_provider():
    """A CredentialProvider stand-in whose session snapshots to valid credential_process JSON.

    get_session_for_account returns a session whose credentials expose the frozen
    keys plus an ``_expiry_time`` — the shape session_to_credential_process reads.
    """
    from datetime import datetime, timezone

    frozen = MagicMock(access_key="AKIATEST", secret_key="secret", token="token")
    creds = MagicMock()
    creds.get_frozen_credentials.return_value = frozen
    creds._expiry_time = datetime(2099, 1, 1, tzinfo=timezone.utc)
    session = MagicMock()
    session.get_credentials.return_value = creds

    cp = MagicMock()
    cp.get_session_for_account.return_value = session
    return cp


@pytest.fixture
def sc(tmp_path, env_config):
    sd = _make_scenario_dir(tmp_path)
    paths = ScenarioPaths(sd)
    return ScenarioContainer(
        paths,
        env_config,
        image_tag="awsbench-sc",
        container_name="awsbench-sc-trial-0",
        host_logs_dir=tmp_path / "trial-logs",
        cred_provider=_fake_cred_provider(),
        account_mapping={"PRIMARY": "111111111111"},
    )


# -- build ----------------------------------------------------------------


def test_build_invokes_docker_build_every_time(sc):
    """Docker's layer cache decides what to rebuild — we never short-circuit."""
    with FakeDocker() as fake:
        fake.when("build", rc=0)
        asyncio.run(sc.build())
    builds = fake.calls_with_prefix("build")
    assert len(builds) == 1
    assert "--no-cache" not in builds[0]
    assert "-t" in builds[0]
    assert "awsbench-sc" in builds[0]


def test_build_force_passes_no_cache(sc):
    with FakeDocker() as fake:
        fake.when("build", rc=0)
        asyncio.run(sc.build(force=True))
    builds = fake.calls_with_prefix("build")
    assert len(builds) == 1
    assert "--no-cache" in builds[0]


def test_build_raises_on_daemon_error(sc):
    with FakeDocker() as fake:
        fake.when("build", rc=1, stderr=b"syntax error in Dockerfile")
        with pytest.raises(DockerCLIError) as exc:
            asyncio.run(sc.build())
    assert "syntax error" in str(exc.value)


def test_build_lock_serializes_concurrent_builds_for_same_tag(sc):
    """Two concurrent builds of the same tag must serialize through one lock.

    Each concurrent caller still invokes ``docker build`` (the daemon's
    layer cache makes the second invocation a cache-hit no-op). The lock
    only enforces ordering — at most one build runs at any moment.
    """
    in_flight = 0
    peak = 0

    def slow_build(_args, _stdin):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        in_flight -= 1
        return 0, b"", b""

    with FakeDocker() as fake:
        fake.when_callable("build", responder=slow_build)

        async def run_two():
            await asyncio.gather(sc.build(), sc.build())

        asyncio.run(run_two())

    assert peak == 1
    assert len(fake.calls_with_prefix("build")) == 2


# -- start ----------------------------------------------------------------


def test_start_runs_container_with_resource_limits(sc, env_config):
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container: awsbench-sc-trial-0")
        fake.when("run", rc=0, stdout=b"abc123\n")
        asyncio.run(sc.start())

    runs = fake.calls_with_prefix("run")
    assert len(runs) == 1
    flat = " ".join(runs[0])
    assert "--detach" in flat
    assert "--name awsbench-sc-trial-0" in flat
    assert f"--cpus {env_config.cpus}" in flat
    assert f"--memory {env_config.memory_mb}m" in flat
    assert "sleep infinity" in flat


def test_start_emits_labels(tmp_path, env_config):
    """Labels become --label k=v args in the docker run command."""
    sd = _make_scenario_dir(tmp_path)
    paths = ScenarioPaths(sd)
    container = ScenarioContainer(
        paths,
        env_config,
        image_tag="awsbench-sc",
        container_name="awsbench-sc-trial-0",
        host_logs_dir=tmp_path / "trial-logs",
        cred_provider=_fake_cred_provider(),
        account_mapping={"PRIMARY": "111111111111"},
        labels={"awsbench.role": "scenario-reset"},
    )
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when("exec", rc=0)
        asyncio.run(container.start())

    runs = fake.calls_with_prefix("run")
    label_args = [a for i, a in enumerate(runs[0]) if i > 0 and runs[0][i - 1] == "--label"]
    assert label_args == ["awsbench.role=scenario-reset"]


def test_start_no_labels_emits_no_label_flag(sc):
    """With no labels (the default), start() emits no --label flag."""
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when("exec", rc=0)
        asyncio.run(sc.start())
    runs = fake.calls_with_prefix("run")
    assert "--label" not in runs[0]


def test_start_removes_stale_container_with_same_name(sc):
    with FakeDocker() as fake:
        fake.when("rm", rc=0)  # stale was present, removed
        fake.when("run", rc=0)
        asyncio.run(sc.start())
    rms = fake.calls_with_prefix("rm")
    assert len(rms) == 1
    assert "-f" in rms[0]
    assert "awsbench-sc-trial-0" in rms[0]


def test_start_twice_raises(sc):
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        asyncio.run(sc.start())
        with pytest.raises(RuntimeError, match="already started"):
            asyncio.run(sc.start())


# -- run_phase ------------------------------------------------------------


def _seed_phase_stdout(sc, phase: str, *, stdout: bytes) -> None:
    """Populate the bind-mounted host_logs_dir as the script would in-container.

    Only ``stdout.txt`` is written; the phase exit code is driven by the
    FakeDocker ``exec`` response for the script invocation, not a file.
    """
    phase_dir = sc._host_logs_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / "stdout.txt").write_bytes(stdout)


def _script_exec_responder(rc: int) -> Responder:
    """Respond rc for the script invocation; 0 for helper exec calls.

    The script invocation is the only ``exec`` whose command contains
    ``stdout.txt`` (the redirect target); helpers (upload mkdir, mkdir+chmod)
    must return 0 so run_phase reaches the script.
    """

    def respond(args: list[str], _stdin: bytes | None) -> tuple[int, bytes, bytes]:
        if any("stdout.txt" in seg for seg in args):
            return rc, b"", b""
        return 0, b"", b""

    return respond


def test_run_phase_uploads_runs_and_reads_back(sc):
    """Bind mount lets the host see stdout.txt directly — no docker cp out."""
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when_callable("exec", responder=_script_exec_responder(0))
        fake.when("cp", "-", rc=0)  # upload-tar still uses cp via stdin

        asyncio.run(sc.start())
        # Simulate the script having written stdout.txt during exec.
        _seed_phase_stdout(sc, "deploy", stdout=b"ok\n")
        result = asyncio.run(sc.run_phase("deploy", env={"K": "V"}, timeout_sec=30))

    assert result.exit_code == 0
    assert result.stdout == "ok\n"
    # Only the upload cp remains; we no longer cp the outputs back out.
    assert len(fake.calls_with_prefix("cp")) == 1


def test_run_phase_passes_env_to_script_invocation(sc):
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when("exec", rc=0)
        fake.when("cp", "-", rc=0)

        asyncio.run(sc.start())
        _seed_phase_stdout(sc, "deploy", stdout=b"")
        asyncio.run(sc.run_phase("deploy", env={"AWS_X": "1"}, timeout_sec=30))

    execs = fake.calls_with_prefix("exec")
    script_runs = [a for a in execs if any("stdout.txt" in seg for seg in a)]
    assert script_runs, "expected one exec call for the script invocation"
    assert "--env" in script_runs[-1]
    assert "AWS_X=1" in script_runs[-1]


def test_run_phase_returns_nonzero_exit_code(sc):
    """A non-zero script exit surfaces as ExecResult.exit_code; run_phase does not raise."""
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when_callable("exec", responder=_script_exec_responder(7))
        fake.when("cp", "-", rc=0)

        asyncio.run(sc.start())
        _seed_phase_stdout(sc, "deploy", stdout=b"boom\n")
        result = asyncio.run(sc.run_phase("deploy", env={}, timeout_sec=30))

    assert result.exit_code == 7
    assert result.stdout == "boom\n"


def test_run_phase_missing_stdout_returns_empty(sc):
    """Absent stdout.txt yields empty stdout rather than crashing.

    The exit code comes from the script's docker-exec return code; here the
    script exits 0 but writes nothing, so stdout is empty.
    """
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when_callable("exec", responder=_script_exec_responder(0))
        fake.when("cp", "-", rc=0)

        asyncio.run(sc.start())
        result = asyncio.run(sc.run_phase("deploy", env={}, timeout_sec=30))

    assert result.exit_code == 0
    assert result.stdout == ""


def test_run_phase_missing_phase_dir_raises(sc):
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        asyncio.run(sc.start())
        with pytest.raises(FileNotFoundError, match="Phase directory"):
            asyncio.run(sc.run_phase("verify", env={}, timeout_sec=10))


def test_run_phase_before_start_raises(sc):
    with pytest.raises(RuntimeError, match="not been started"):
        asyncio.run(sc.run_phase("deploy", env={}, timeout_sec=10))


def test_run_phase_helper_command_failure_raises(sc):
    """A non-zero exit from mkdir/chmod must surface as a RuntimeError."""
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        # Everything succeeds except the phase's helper mkdir+chmod (the "chmod +x"
        # exec), so start()'s config-write and the upload mkdir are unaffected.
        fake.when_callable(
            "exec",
            responder=lambda args, _s: (
                (2, b"", b"mkdir: permission denied\n") if "chmod +x" in args[-1] else (0, b"", b"")
            ),
        )

        asyncio.run(sc.start())
        with pytest.raises(RuntimeError, match="Container setup failed"):
            asyncio.run(sc.run_phase("deploy", env={}, timeout_sec=10))


# -- bind mount -----------------------------------------------------------


def test_start_bind_mounts_host_logs_dir(sc):
    """`/logs` must be bind-mounted from host_logs_dir so output is visible live."""
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        asyncio.run(sc.start())

    runs = fake.calls_with_prefix("run")
    flat = " ".join(runs[0])
    # k=v --mount form is colon-safe for operator-supplied paths.
    assert "--mount" in flat
    expected = f"type=bind,source={sc._host_logs_dir.resolve()},target=/logs"
    assert expected in flat


def test_start_bind_mount_handles_colon_in_host_path(tmp_path, env_config):
    """A host path with ':' must not break docker-arg parsing (use --mount, not -v)."""
    sd = _make_scenario_dir(tmp_path)
    weird_dir = tmp_path / "with:colon" / "trial-0"
    container = ScenarioContainer(
        ScenarioPaths(sd),
        env_config,
        image_tag="awsbench-sc",
        container_name="awsbench-sc-trial-x",
        host_logs_dir=weird_dir,
        cred_provider=_fake_cred_provider(),
        account_mapping={"PRIMARY": "111111111111"},
    )
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        asyncio.run(container.start())

    runs = fake.calls_with_prefix("run")
    flat = " ".join(runs[0])
    assert f"source={weird_dir.resolve()}" in flat
    assert "target=/logs" in flat


def test_start_creates_host_logs_dir_if_missing(sc):
    """Bind-mount target must exist pre-run; Docker would otherwise create it as root."""
    assert not sc._host_logs_dir.exists()
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        asyncio.run(sc.start())
    assert sc._host_logs_dir.is_dir()


# -- write_file -----------------------------------------------------------


def test_write_file_emits_mkdir_and_heredoc(sc):
    captured: list[str] = []

    def capture(args: list[str], stdin: bytes | None) -> tuple[int, bytes, bytes]:
        for seg in args:
            if "AWSBENCH_EOF" in seg:
                captured.append(seg)
        return 0, b"", b""

    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when_callable("exec", responder=capture)

        asyncio.run(sc.start())
        asyncio.run(sc.write_file("~/.aws/config", "[profile X]\nrole_arn = a\n"))

    assert captured, "expected docker exec to receive the heredoc"
    body = captured[-1]
    assert "mkdir -p ~/.aws" in body
    assert "cat > ~/.aws/config" in body
    assert "[profile X]" in body
    assert "role_arn = a" in body


def test_write_file_raises_on_failure(sc):
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        # start() writes ~/.aws/config via an exec; that must succeed. The write we
        # assert on targets /etc/foo, so match the failure to that command only.
        fake.when_callable(
            "exec",
            responder=lambda args, _s: (
                (2, b"", b"permission denied") if "/etc/foo" in args[-1] else (0, b"", b"")
            ),
        )

        asyncio.run(sc.start())
        with pytest.raises(RuntimeError, match="Failed to write"):
            asyncio.run(sc.write_file("/etc/foo", "bar"))


def test_write_file_requires_started(sc):
    with FakeDocker():
        with pytest.raises(RuntimeError, match="not been started"):
            asyncio.run(sc.write_file("/tmp/x", "y"))


def test_write_file_rejects_unsafe_path(sc):
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        asyncio.run(sc.start())
        with pytest.raises(ValueError, match="path must match"):
            asyncio.run(sc.write_file("/tmp/x;rm -rf /", "y"))


def test_write_file_uses_unique_sentinel_when_content_collides(sc):
    """If content contains a literal AWSBENCH_EOF line, sentinel must change."""
    captured: list[str] = []

    def capture(args: list[str], stdin: bytes | None) -> tuple[int, bytes, bytes]:
        for seg in args:
            if "EOF" in seg and "cat >" in seg:
                captured.append(seg)
        return 0, b"", b""

    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when_callable("exec", responder=capture)

        asyncio.run(sc.start())
        # Content includes the default sentinel as a line by itself.
        asyncio.run(sc.write_file("/tmp/x", "ok\nAWSBENCH_EOF\nstill ok"))

    assert captured
    cmd = captured[-1]
    # The default sentinel must NOT be the heredoc terminator (otherwise
    # the heredoc would close on the literal line in the body).
    assert "<<'AWSBENCH_EOF'\n" not in cmd
    assert "AWSBENCH_EOF_" in cmd


# -- symlink rejection ----------------------------------------------------


def test_upload_dir_rejects_top_level_symlink(sc, tmp_path):
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when("exec", rc=0)

        sd = sc._paths.scenario_dir
        target = tmp_path / "outside-secret"
        target.write_text("secret\n")
        (sd / "deploy" / "leak").symlink_to(target)

        asyncio.run(sc.start())
        with pytest.raises(ValueError, match="symlinks are not allowed"):
            asyncio.run(sc.run_phase("deploy", env={}, timeout_sec=10))


def test_upload_dir_rejects_nested_symlink(sc):
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when("exec", rc=0)

        sd = sc._paths.scenario_dir
        sub = sd / "deploy" / "lib"
        sub.mkdir()
        (sub / "self").symlink_to(sd / "deploy" / "deploy.sh")

        asyncio.run(sc.start())
        with pytest.raises(ValueError, match="symlinks are not allowed"):
            asyncio.run(sc.run_phase("deploy", env={}, timeout_sec=10))


# -- stop ----------------------------------------------------------------


def test_stop_no_op_before_start(sc):
    with FakeDocker() as fake:
        asyncio.run(sc.stop(delete=True))
    assert fake.calls == []


def test_stop_delete_runs_stop_then_rm(sc):
    with FakeDocker() as fake:
        fake.when("run", rc=0)
        fake.when("stop", rc=0)
        # Pre-start `rm -f <stale>` (rc=1 = no stale) and post-stop `rm -f` (rc=0).
        fake.when_each(
            "rm",
            responses=[(1, b"", b"No such container"), (0, b"", b"")],
        )

        asyncio.run(sc.start())
        asyncio.run(sc.stop(delete=True))

    assert fake.calls_with_prefix("stop")
    assert len(fake.calls_with_prefix("rm")) == 2


# -- name sanitizers ------------------------------------------------------


def test_sanitize_image_tag_examples():
    assert sanitize_image_tag("awsbench-Lambda Broken") == "awsbench-lambda-broken"
    assert sanitize_image_tag("__weird__") == "0__weird__"


def test_sanitize_container_name_examples():
    assert sanitize_container_name("awsbench-Lambda Broken") == "awsbench-lambda-broken"
    assert sanitize_container_name("9-leading-digit") == "9-leading-digit"
    assert sanitize_container_name("_leading-underscore") == "0_leading-underscore"


# -- mounts_json ----------------------------------------------------------


def test_start_emits_configured_bind_mount(tmp_path):
    """mounts_json entries become --mount args in the docker run command."""
    sd = _make_scenario_dir(tmp_path)
    paths = ScenarioPaths(sd)
    env = EnvironmentConfig(
        cpus=1,
        memory_mb=512,
        mounts_json=[
            {
                "type": "bind",
                "source": "/var/run/docker.sock",
                "target": "/var/run/docker.sock",
            }
        ],
    )
    container = ScenarioContainer(
        paths,
        env,
        image_tag="t",
        container_name="c",
        host_logs_dir=tmp_path / "logs",
        cred_provider=_fake_cred_provider(),
        account_mapping={"PRIMARY": "111111111111"},
    )
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        asyncio.run(container.start())

    runs = fake.calls_with_prefix("run")
    flat = " ".join(runs[0])
    assert "type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock" in flat


def test_start_emits_readonly_mount(tmp_path):
    """read_only=True appends ,readonly to the mount spec."""
    sd = _make_scenario_dir(tmp_path)
    paths = ScenarioPaths(sd)
    env = EnvironmentConfig(
        cpus=1,
        memory_mb=512,
        mounts_json=[{"type": "bind", "source": "/tmp/x", "target": "/mnt/x", "read_only": True}],
    )
    container = ScenarioContainer(
        paths,
        env,
        image_tag="t",
        container_name="c",
        host_logs_dir=tmp_path / "logs",
        cred_provider=_fake_cred_provider(),
        account_mapping={"PRIMARY": "111111111111"},
    )
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        asyncio.run(container.start())

    runs = fake.calls_with_prefix("run")
    flat = " ".join(runs[0])
    assert "type=bind,source=/tmp/x,target=/mnt/x,readonly" in flat


def test_start_emits_multiple_mounts(tmp_path):
    """Multiple mounts_json entries each get their own --mount flag."""
    sd = _make_scenario_dir(tmp_path)
    paths = ScenarioPaths(sd)
    env = EnvironmentConfig(
        cpus=1,
        memory_mb=512,
        mounts_json=[
            {"type": "bind", "source": "/a", "target": "/b"},
            {"type": "volume", "source": "vol1", "target": "/data"},
        ],
    )
    container = ScenarioContainer(
        paths,
        env,
        image_tag="t",
        container_name="c",
        host_logs_dir=tmp_path / "logs",
        cred_provider=_fake_cred_provider(),
        account_mapping={"PRIMARY": "111111111111"},
    )
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        asyncio.run(container.start())

    runs = fake.calls_with_prefix("run")
    flat = " ".join(runs[0])
    assert "type=bind,source=/a,target=/b" in flat
    assert "type=volume,source=vol1,target=/data" in flat


def test_start_no_mounts_json_unchanged(sc):
    """With no author mounts, start() emits only the framework mounts: /logs + creds."""
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when("exec", rc=0)  # start() writes ~/.aws/config via exec
        asyncio.run(sc.start())

    runs = fake.calls_with_prefix("run")
    mount_args = [a for i, a in enumerate(runs[0]) if i > 0 and runs[0][i - 1] == "--mount"]
    # The /logs bind mount plus the read-only credential mount — no author mounts.
    assert len(mount_args) == 2
    assert any("target=/logs" in m for m in mount_args)
    assert any("target=/awsbench-creds" in m and "readonly" in m for m in mount_args)


# -- credential refresher -------------------------------------------------


def test_start_writes_credential_process_config_and_creds_file(sc):
    """start() mints the initial per-tag creds file and writes a credential_process config."""
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when("exec", rc=0)
        asyncio.run(sc.start())
        try:
            # Initial creds minted on the host before any phase runs.
            sc._cred_provider.get_session_for_account.assert_called()
            creds_path = sc._creds_dir / "PRIMARY.json"
            assert creds_path.exists()
            assert json.loads(creds_path.read_text())["AccessKeyId"] == "AKIATEST"
            # ~/.aws/config uses credential_process (not credential_source=Environment).
            config_execs = [
                c for c, _ in fake.calls if c[:1] == ["exec"] and "/.aws/config" in c[-1]
            ]
            assert config_execs, "expected an exec writing ~/.aws/config"
            body = config_execs[0][-1]
            assert "[profile PRIMARY]" in body
            assert "credential_process" in body
            assert "credential_source = Environment" not in body
        finally:
            asyncio.run(sc.stop(delete=True))


def test_stop_cancels_refresher_and_removes_creds_dir(sc):
    """stop() cancels the refresh task and deletes the host credential dir (no secrets left)."""
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when("exec", rc=0)
        fake.when("stop", rc=0)
        asyncio.run(sc.start())
        creds_dir = sc._creds_dir
        assert creds_dir is not None and creds_dir.exists()
        assert sc._refresh_task is not None
        asyncio.run(sc.stop(delete=True))
        assert sc._refresh_task is None
        assert sc._creds_dir is None
        assert not creds_dir.exists()


# -- logs ownership handback (Harbor parity) ------------------------------


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX-only ownership handback")
def test_run_phase_chowns_logs_to_host_user(sc):
    """Phase output in /logs is chowned back to the host user on rootful Docker."""
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when("info", rc=0, stdout=b"name=seccomp,profile=default|")  # not rootless
        fake.when("exec", rc=0)
        fake.when("cp", "-", rc=0)

        asyncio.run(sc.start())
        _seed_phase_stdout(sc, "deploy", stdout=b"")
        asyncio.run(sc.run_phase("deploy", env={}, timeout_sec=30))

    uid, gid = os.getuid(), os.getgid()
    chowns = [
        a
        for a in fake.calls_with_prefix("exec")
        if any(f"chown -R {uid}:{gid} /logs" in seg for seg in a)
    ]
    assert chowns, "expected a `chown -R <host-uid>:<gid> /logs` exec after the phase"


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX-only ownership handback")
def test_chown_uses_uid_0_under_rootless_docker(sc):
    """Rootless Docker maps container UID 0 to the host user, so chown targets 0:0."""
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when("info", rc=0, stdout=b"name=rootless|")
        fake.when("exec", rc=0)
        fake.when("cp", "-", rc=0)

        asyncio.run(sc.start())
        _seed_phase_stdout(sc, "deploy", stdout=b"")
        asyncio.run(sc.run_phase("deploy", env={}, timeout_sec=30))

    chowns = [
        a for a in fake.calls_with_prefix("exec") if any("chown -R 0:0 /logs" in seg for seg in a)
    ]
    assert chowns, "rootless Docker: chown should target uid:gid 0:0"
