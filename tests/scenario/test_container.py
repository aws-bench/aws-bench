"""Tests for aws_bench.scenario.container.

Mocks the docker CLI by patching asyncio.create_subprocess_exec so the
lifecycle (build dedup, start with bind-mount, run_phase, stop) can be
exercised without a real Docker daemon. Phase outputs land on the host
through the bind mount, so tests pre-seed the host_logs_dir to mimic
what the in-container script would have written.
"""

from __future__ import annotations

import asyncio
import csv
import inspect
import io
import json
import os
import posixpath
import secrets
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import threading
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import aws_bench.scenario.container as container_module
import aws_bench.utils.credentials_provider as credentials_module
from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.exceptions import CredentialError, OperationCancelled
from aws_bench.scenario.config import EnvironmentConfig
from aws_bench.scenario.container import (
    DockerCLIError,
    ScenarioContainer,
    sanitize_container_name,
    sanitize_image_tag,
)
from aws_bench.scenario.paths import ScenarioPaths
from aws_bench.utils.credentials_provider import CREDENTIAL_ENV_VARS, build_aws_config

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


DockerResponse = tuple[int, bytes, bytes]
Responder = Callable[[list[str], bytes | None], DockerResponse | Awaitable[DockerResponse]]


class FakeDocker:
    """Patches asyncio.create_subprocess_exec to simulate the docker CLI.

    Tests register matchers like ``fake.when("exec", rc=0)`` to declare
    "any docker exec ... call returns rc=0". Calls are recorded for
    assertion.
    """

    def __init__(
        self,
        *,
        home: str = "/root",
        uid: int = 0,
        gid: int = 0,
        config: str = "",
        credentials: str = "",
        resolved_targets: dict[str, str] | None = None,
        mountinfo: str | None = None,
    ) -> None:
        self._matchers: list[tuple[Callable[[list[str]], bool], Responder]] = []
        self.calls: list[tuple[list[str], bytes | None]] = []
        self._patch = None
        self.home, self.uid, self.gid = home, uid, gid
        self.config, self.credentials = config, credentials
        self.resolved_targets = resolved_targets or {}
        mount_home = home.replace("\\", r"\134").replace(" ", r"\040")
        self.mountinfo = (
            mountinfo
            if mountinfo is not None
            else (
                "1 0 0:1 / / rw - overlay overlay rw\n"
                f"2 1 0:2 / {mount_home}/.aws/creds ro - tmpfs tmpfs rw\n"
            )
        )

    @staticmethod
    def _matches(args: list[str], prefix: tuple[str, ...]) -> bool:
        if prefix == ("run",) and args[:2] == ["run", "--rm"]:
            return False
        return tuple(args[: len(prefix)]) == prefix

    def when(
        self,
        *prefix: str,
        rc: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        def matches(args: list[str]) -> bool:
            return self._matches(args, prefix)

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
            return self._matches(args, prefix)

        def respond(_a: list[str], _s: bytes | None) -> tuple[int, bytes, bytes]:
            return next(it)

        self._matchers.append((matches, respond))

    def when_callable(self, *prefix: str, responder: Responder) -> None:
        def matches(args: list[str]) -> bool:
            return self._matches(args, prefix)

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
        index = len(self.calls)
        self.calls.append((args, None))
        proc = MagicMock()

        async def communicate(input=None):
            self.calls[index] = (args, input)
            rc, out_bytes, err_bytes = 0, b"", b""
            if args[:2] == ["run", "--rm"]:
                targets = args[args.index("awsbench-home") + 1 :]
                paths = [self.resolved_targets.get(t, posixpath.normpath(t)) for t in targets]
                out_bytes = (
                    "\n".join([self.home, str(self.uid), str(self.gid), *paths]) + "\n"
                ).encode()
            elif args[0] == "exec":
                if "cat /proc/self/mountinfo" in args[-1]:
                    out_bytes = self.mountinfo.encode()
                for name in ("config", "credentials"):
                    path = shlex.quote(f"{self.home}/.aws/{name}")
                    if f"cat {path}; fi;" in args[-1]:
                        out_bytes = getattr(self, name).encode()
            for matches, respond in self._matchers:
                if matches(args):
                    response = respond(args, input)
                    if inspect.isawaitable(response):
                        response = await response
                    rc, response_body, err_bytes = response
                    if response_body or rc != 0 or "cat /proc/self/mountinfo" not in args[-1]:
                        out_bytes = response_body
                    break
            proc.returncode = rc
            return out_bytes, err_bytes

        proc.communicate = communicate
        return proc

    def calls_with_prefix(self, *prefix: str) -> list[list[str]]:
        return [args for args, _ in self.calls if self._matches(args, prefix)]


@pytest.fixture(autouse=True)
def reset_locks(tmp_path, monkeypatch):
    """Reset build locks and keep synthetic credential directories inside the test."""
    ScenarioContainer._image_build_locks.clear()
    monkeypatch.setattr(tempfile, "mkdtemp", partial(tempfile.mkdtemp, dir=tmp_path))
    yield
    ScenarioContainer._image_build_locks.clear()


@pytest.fixture
def env_config():
    return EnvironmentConfig(cpus=1, memory_mb=512, build_timeout_sec=60)


def _fake_cred_provider():
    """A CredentialProvider stand-in whose session snapshots to valid credential_process JSON.

    get_chained_session_for_account returns a session whose credentials expose the frozen
    keys plus an ``_expiry_time`` — the shape session_to_credential_process reads.
    """
    frozen = MagicMock(access_key="AKIATEST", secret_key="secret", token="token")
    creds = MagicMock()
    creds.get_frozen_credentials.return_value = frozen
    creds._expiry_time = datetime(2099, 1, 1, tzinfo=timezone.utc)
    session = MagicMock()
    session.get_credentials.return_value = creds

    cp = MagicMock()
    cp.get_chained_session_for_account.return_value = session
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


def _uploaded_config(fake: FakeDocker) -> str:
    uploads = [
        body for args, body in fake.calls if args[:2] == ["cp", "-"] and args[-1].endswith("/.aws")
    ]
    assert len(uploads) == 1 and uploads[0] is not None
    with tarfile.open(fileobj=io.BytesIO(uploads[0])) as tar:
        entries = tar.getmembers()
        assert len(entries) == 1
        assert entries[0].name.startswith(".config-")
        assert entries[0].mode == 0o600
        body = tar.extractfile(entries[0])
        assert body is not None
        return body.read().decode()


def _generation(key: str, *, expires_at: datetime | None = None) -> tuple[dict[str, str], datetime]:
    expiry = expires_at or datetime.now(timezone.utc) + timedelta(hours=1)
    return {
        "PRIMARY.json": json.dumps(
            {
                "Version": 1,
                "AccessKeyId": key,
                "SecretAccessKey": f"{key}-secret",
                "SessionToken": f"{key}-token",
                "Expiration": expiry.isoformat(),
            }
        )
    }, expiry


@pytest.fixture
def refresh_clock(monkeypatch):
    """Advance the actual shared loop one sleep at a time without wall-clock delays."""
    delays: asyncio.Queue[float] = asyncio.Queue()
    ticks: asyncio.Queue[None] = asyncio.Queue()

    async def sleep(delay: float) -> None:
        delays.put_nowait(delay)
        await ticks.get()

    monkeypatch.setattr(
        credentials_module,
        "asyncio",
        SimpleNamespace(
            sleep=sleep,
            get_running_loop=asyncio.get_running_loop,
            timeout_at=asyncio.timeout_at,
        ),
    )
    return delays, ticks


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


@pytest.mark.asyncio
async def test_run_phase_uploads_runs_and_reads_back(sc):
    """Bind mount lets the host see stdout.txt directly — no docker cp out."""
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when_callable("exec", responder=_script_exec_responder(0))
        fake.when("cp", "-", rc=0)  # upload-tar still uses cp via stdin

        await sc.start()
        # Simulate the script having written stdout.txt during exec.
        _seed_phase_stdout(sc, "deploy", stdout=b"ok\n")
        result = await sc.run_phase("deploy", env={"K": "V"}, timeout_sec=30)
        await sc.stop(delete=True)

    assert result.exit_code == 0
    assert result.stdout == "ok\n"
    # The config and script travel into the container; no outputs are copied out.
    assert len(fake.calls_with_prefix("cp")) == 2
    assert all("-" in call for call in fake.calls_with_prefix("cp"))


@pytest.mark.asyncio
async def test_run_phase_passes_env_to_script_invocation(sc):
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when("exec", rc=0)
        fake.when("cp", "-", rc=0)

        await sc.start()
        _seed_phase_stdout(sc, "deploy", stdout=b"")
        await sc.run_phase("deploy", env={"AWS_X": "1"}, timeout_sec=30)
        await sc.stop(delete=True)

    execs = fake.calls_with_prefix("exec")
    script_runs = [a for a in execs if any("stdout.txt" in seg for seg in a)]
    assert script_runs, "expected one exec call for the script invocation"
    assert "--env" in script_runs[-1]
    assert "AWS_X=1" in script_runs[-1]


@pytest.mark.asyncio
async def test_run_phase_returns_nonzero_exit_code(sc):
    """A non-zero script exit surfaces as ExecResult.exit_code; run_phase does not raise."""
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when_callable("exec", responder=_script_exec_responder(7))
        fake.when("cp", "-", rc=0)

        await sc.start()
        _seed_phase_stdout(sc, "deploy", stdout=b"boom\n")
        result = await sc.run_phase("deploy", env={}, timeout_sec=30)
        await sc.stop(delete=True)

    assert result.exit_code == 7
    assert result.stdout == "boom\n"


@pytest.mark.asyncio
async def test_run_phase_missing_stdout_returns_empty(sc):
    """Absent stdout.txt yields empty stdout rather than crashing.

    The exit code comes from the script's docker-exec return code; here the
    script exits 0 but writes nothing, so stdout is empty.
    """
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when_callable("exec", responder=_script_exec_responder(0))
        fake.when("cp", "-", rc=0)

        await sc.start()
        result = await sc.run_phase("deploy", env={}, timeout_sec=30)
        await sc.stop(delete=True)

    assert result.exit_code == 0
    assert result.stdout == ""


@pytest.mark.asyncio
async def test_run_phase_missing_phase_dir_raises(sc):
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        await sc.start()
        with pytest.raises(FileNotFoundError, match="Phase directory"):
            await sc.run_phase("verify", env={}, timeout_sec=10)
        await sc.stop(delete=True)


def test_run_phase_before_start_raises(sc):
    with pytest.raises(RuntimeError, match="not been started"):
        asyncio.run(sc.run_phase("deploy", env={}, timeout_sec=10))


@pytest.mark.asyncio
async def test_run_phase_helper_command_failure_raises(sc):
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

        await sc.start()
        with pytest.raises(RuntimeError, match="Container setup failed"):
            await sc.run_phase("deploy", env={}, timeout_sec=10)
        await sc.stop(delete=True)


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


# -- symlink rejection ----------------------------------------------------


@pytest.mark.asyncio
async def test_upload_dir_rejects_top_level_symlink(sc, tmp_path):
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when("exec", rc=0)

        sd = sc._paths.scenario_dir
        target = tmp_path / "outside-secret"
        target.write_text("secret\n")
        (sd / "deploy" / "leak").symlink_to(target)

        await sc.start()
        with pytest.raises(ValueError, match="symlinks are not allowed"):
            await sc.run_phase("deploy", env={}, timeout_sec=10)
        await sc.stop(delete=True)


@pytest.mark.asyncio
async def test_upload_dir_rejects_nested_symlink(sc):
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when("exec", rc=0)

        sd = sc._paths.scenario_dir
        sub = sd / "deploy" / "lib"
        sub.mkdir()
        (sub / "self").symlink_to(sd / "deploy" / "deploy.sh")

        await sc.start()
        with pytest.raises(ValueError, match="symlinks are not allowed"):
            await sc.run_phase("deploy", env={}, timeout_sec=10)
        await sc.stop(delete=True)


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
    assert any("target=/root/.aws/creds" in m and "readonly" in m for m in mount_args)


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
            sc._cred_provider.get_chained_session_for_account.assert_called_once_with(
                "111111111111", ORG_ACCESS_ROLE, "app-session-PRIMARY"
            )
            creds_path = sc._creds_dir / "PRIMARY.json"
            assert creds_path.exists()
            assert json.loads(creds_path.read_text())["AccessKeyId"] == "AKIATEST"
            # ~/.aws/config uses credential_process (not credential_source=Environment).
            body = _uploaded_config(fake)
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
@pytest.mark.asyncio
async def test_run_phase_chowns_logs_to_host_user(sc):
    """Phase output in /logs is chowned back to the host user on rootful Docker."""
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when("info", rc=0, stdout=b"name=seccomp,profile=default|")  # not rootless
        fake.when("exec", rc=0)
        fake.when("cp", "-", rc=0)

        await sc.start()
        _seed_phase_stdout(sc, "deploy", stdout=b"")
        await sc.run_phase("deploy", env={}, timeout_sec=30)
        await sc.stop(delete=True)

    uid, gid = os.getuid(), os.getgid()
    chowns = [
        a
        for a in fake.calls_with_prefix("exec")
        if any(f"chown -R {uid}:{gid} /logs" in seg for seg in a)
    ]
    assert chowns, "expected a `chown -R <host-uid>:<gid> /logs` exec after the phase"


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX-only ownership handback")
@pytest.mark.asyncio
async def test_chown_uses_uid_0_under_rootless_docker(sc):
    """Rootless Docker maps container UID 0 to the host user, so chown targets 0:0."""
    with FakeDocker() as fake:
        fake.when("rm", rc=1, stderr=b"No such container")
        fake.when("run", rc=0)
        fake.when("info", rc=0, stdout=b"name=rootless|")
        fake.when("exec", rc=0)
        fake.when("cp", "-", rc=0)

        await sc.start()
        _seed_phase_stdout(sc, "deploy", stdout=b"")
        await sc.run_phase("deploy", env={}, timeout_sec=30)
        await sc.stop(delete=True)

    chowns = [
        a for a in fake.calls_with_prefix("exec") if any("chown -R 0:0 /logs" in seg for seg in a)
    ]
    assert chowns, "rootless Docker: chown should target uid:gid 0:0"


# -- unified credential lifecycle ----------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("home", "uid", "gid"),
    [("/root", 0, 0), ("/home/runner", 1001, 1002), ("/home/runner space,one", 1001, 1001)],
)
async def test_image_user_gets_readonly_inner_mount_and_private_host_root(sc, home, uid, gid):
    with FakeDocker(home=home, uid=uid, gid=gid) as fake:
        await sc.start()
        root, inner = sc._creds_root, sc._creds_dir
        assert root is not None and inner is not None
        assert inner.parent == root
        assert root.stat().st_mode & 0o777 == 0o700
        assert inner.stat().st_mode & 0o777 == 0o755
        assert (inner / "PRIMARY.json").stat().st_mode & 0o777 == 0o644
        assert sc._container_home == PurePosixPath(home)
        assert (sc._container_uid, sc._container_gid) == (uid, gid)
        args = fake.calls_with_prefix("run")[0]
        mounts = [args[i + 1] for i, value in enumerate(args) if value == "--mount"]
        assert ["type=bind", f"source={inner}", f"target={home}/.aws/creds", "readonly"] in [
            next(csv.reader([mount])) for mount in mounts
        ]
        assert all(f"source={root}" not in next(csv.reader([mount])) for mount in mounts)
        probe = fake.calls_with_prefix("run", "--rm")[0]
        assert probe[probe.index("--network") + 1] == "none"
        assert probe[probe.index("--entrypoint") + 1] == "sh"
        assert "--mount" not in probe
        assert '"$HOME/.aws/creds/PRIMARY.json"' in _uploaded_config(fake)
        await sc.stop(delete=True)
        assert not root.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    [
        "/",
        "/root",
        "/root/.aws",
        "/root/.aws/creds",
        "/root/.aws/creds/.hidden",
        "/root/.aws/creds/PRIMARY.json",
        "/root/.aws/config",
        "/root/.aws/config/child",
        "/root/.aws/unused/../creds",
    ],
)
async def test_rejects_reserved_mounts_before_mint(sc, target):
    sc._env_config.mounts_json = [{"type": "bind", "source": "/tmp/source", "target": target}]
    with FakeDocker() as fake:
        with pytest.raises(ValueError, match="overlaps reserved"):
            await sc.start()
    sc._cred_provider.get_chained_session_for_account.assert_not_called()
    assert not fake.calls_with_prefix("run")
    assert sc._creds_root is None


@pytest.mark.asyncio
async def test_rejects_mount_alias_resolved_in_image(sc):
    sc._env_config.mounts_json = [
        {"type": "bind", "source": "/tmp/source", "target": "/alias/config"}
    ]
    with FakeDocker(resolved_targets={"/alias/config": "/root/.aws/config"}):
        with pytest.raises(ValueError, match="overlaps reserved"):
            await sc.start()
    sc._cred_provider.get_chained_session_for_account.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    ["/root/.aws/config", "/root/.aws/creds/PRIMARY.json", "/root"],
)
async def test_rejects_actual_mount_overlaps_before_config_publication(sc, target):
    with FakeDocker() as fake:
        fake.mountinfo += f"3 1 0:3 / {target} rw - tmpfs tmpfs rw\n"
        with pytest.raises(ValueError, match="overlaps reserved"):
            await sc.start()
        assert not fake.calls_with_prefix("cp")
        assert sc._creds_root is None
        assert fake.calls_with_prefix("stop")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mountinfo",
    [
        "",
        "1 0 0:1 / / rw - overlay overlay rw\n",
        "2 1 0:2 / /root/.aws/creds rw - tmpfs tmpfs rw\n",
        "2 1 0:2 / /root/.aws/creds ro - tmpfs tmpfs rw\n"
        "3 1 0:3 / /root/.aws/creds ro - tmpfs tmpfs rw\n",
    ],
)
async def test_missing_writable_or_duplicate_credential_mount_fails(sc, mountinfo):
    with FakeDocker(mountinfo=mountinfo):
        with pytest.raises(ValueError, match="mount"):
            await sc.start()
    assert sc._creds_root is None


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["OTHER", "PRIMARY"])
async def test_readonly_shared_credentials_mount_is_checked_without_writes(sc, profile):
    target = "/root/.aws/credentials"
    original = f"[{profile}]\naws_access_key_id = MOUNTED_KEY\n"
    sc._env_config.mounts_json = [
        {"type": "bind", "source": "/tmp/static-credentials", "target": target, "read_only": True}
    ]
    with FakeDocker(credentials=original) as fake:
        fake.mountinfo += f"3 1 0:3 / {target} ro - tmpfs tmpfs rw\n"
        if profile == "PRIMARY":
            with pytest.raises(CredentialError, match="contains selected profile"):
                await sc.start()
            assert not fake.calls_with_prefix("cp")
        else:
            await sc.start()
            assert fake.calls_with_prefix("cp")
        reads = [args[-1] for args in fake.calls_with_prefix("exec") if target in args[-1]]
        assert reads
        assert all(f"test -f {target} && cat {target}" in command for command in reads)
        assert fake.credentials == original
        await sc.stop(delete=True)
        assert sc._creds_root is None


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["relative/creds", "~/.aws/config", "/mnt/x\n", "/mnt/x\0"])
async def test_rejects_invalid_mount_destinations(sc, target):
    sc._env_config.mounts_json = [{"type": "bind", "source": "/tmp/source", "target": target}]
    with FakeDocker() as fake:
        with pytest.raises(ValueError, match="absolute container paths"):
            await sc.start()
        assert not fake.calls_with_prefix("run", "--rm")


@pytest.mark.asyncio
async def test_mount_csv_cannot_inject_a_reserved_target(sc):
    target = "/mnt/data,target=/root/.aws/config"
    sc._env_config.mounts_json = [{"type": "bind", "source": "/tmp/a,b", "target": target}]
    with FakeDocker() as fake:
        await sc.start()
        args = fake.calls_with_prefix("run")[0]
        mounts = [args[i + 1] for i, value in enumerate(args) if value == "--mount"]
        assert next(csv.reader([mounts[-1]])) == [
            "type=bind",
            "source=/tmp/a,b",
            f"target={target}",
        ]
        await sc.stop(delete=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe",
    [
        b"",
        b"relative\n0\n0\n/logs\n",
        b"/root\nroot\n0\n/logs\n",
        b"/root\n0\n-1\n/logs\n",
        b"/root\n0\n0\n",
    ],
)
async def test_invalid_image_probe_fails_before_mint(sc, probe):
    with FakeDocker() as fake:
        fake.when("run", "--rm", stdout=probe)
        with pytest.raises(RuntimeError, match="resolve.*HOME and user"):
            await sc.start()
    sc._cred_provider.get_chained_session_for_account.assert_not_called()
    assert sc._creds_dir is None


@pytest.mark.asyncio
async def test_config_preserves_unrelated_secrets_without_arguments_or_logs(sc, caplog):
    original = (
        "# retained comment\n"
        "[profile unrelated]\n"
        "aws_access_key_id = UNRELATED_KEY\n"
        "aws_secret_access_key = UNRELATED_SECRET\n"
        "[profile PRIMARY]\n"
        "region = eu-west-1\n"
        "output = json\n"
    )
    static = "[OTHER]\naws_access_key_id = OTHER_KEY\naws_secret_access_key = OTHER_SECRET\n"
    with FakeDocker(config=original, credentials=static) as fake:
        await sc.start()
        config = _uploaded_config(fake)
        assert config == build_aws_config(["PRIMARY"], original, region="us-east-1")
        assert "# retained comment\n[profile unrelated]\n" in config
        assert "aws_secret_access_key = UNRELATED_SECRET" in config
        assert "region = eu-west-1\noutput = json\n" in config
        args = "\n".join(" ".join(args) for args, _ in fake.calls)
        for secret in ("AKIATEST", "UNRELATED_KEY", "UNRELATED_SECRET", "OTHER_SECRET"):
            assert secret not in args
            assert secret not in caplog.text
        assert '"Version":' not in args
        assert fake.credentials == static
        await sc.stop(delete=True)


@pytest.mark.asyncio
async def test_matching_process_config_needs_no_upload(sc):
    config = build_aws_config(["PRIMARY"], region="us-east-1")
    with FakeDocker(config=config) as fake:
        await sc.start()
        assert not fake.calls_with_prefix("cp")
        await sc.stop(delete=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "static"),
    [
        ("[profile PRIMARY]\ncredential_source = Environment\n", ""),
        ("[profile PRIMARY]\ncredential_process = cat /other/file\n", ""),
        ("", "[PRIMARY]\naws_access_key_id = OLD_KEY\naws_secret_access_key = OLD_SECRET\n"),
    ],
)
async def test_conflicting_config_cleans_up_failed_start(sc, tmp_path, config, static):
    with FakeDocker(config=config, credentials=static) as fake:
        with pytest.raises(CredentialError):
            await sc.start()
        assert not fake.calls_with_prefix("cp")
        assert fake.calls_with_prefix("stop")
        assert len(fake.calls_with_prefix("rm")) == 2
    assert sc._creds_dir is None and sc._creds_root is None
    assert not list(tmp_path.glob("awsbench-creds-*"))
    assert not sc.is_started


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["config", "credentials"])
@pytest.mark.parametrize("kind", ["fifo", "directory"])
async def test_nonregular_aws_file_is_rejected_before_cat(sc, tmp_path, name, kind):
    home = tmp_path / "image-home"
    aws_dir = home / ".aws"
    aws_dir.mkdir(parents=True)
    path = aws_dir / name
    if kind == "fifo":
        os.mkfifo(path)
    else:
        path.mkdir()
    read_commands = []

    def execute_read(args, stdin):
        command = args[-1]
        if f"cat {shlex.quote(str(path))}" in command:
            read_commands.append(command)
            result = subprocess.run(
                ["sh", "-c", command], capture_output=True, timeout=1, check=False
            )
            return result.returncode, result.stdout, result.stderr
        return 0, b"", b""

    with FakeDocker(home=str(home)) as fake:
        fake.when_callable("exec", responder=execute_read)
        with pytest.raises(RuntimeError, match=f"Could not read.*{name} file"):
            await sc.start()
        assert read_commands
        assert not fake.calls_with_prefix("cp")
        assert path.exists()
        assert sc._creds_root is None
        assert fake.calls_with_prefix("stop")


@pytest.mark.asyncio
async def test_exec_unsets_image_and_override_sources_in_the_actual_shell(sc):
    overrides = {
        **dict.fromkeys(CREDENTIAL_ENV_VARS, "OVERRIDE_SECRET"),
        "AWS_REGION": "eu-west-1",
        "AWS_DEFAULT_REGION": "ap-south-1",
        "AWS_BEARER_TOKEN_BEDROCK": "MODEL_TOKEN",
        "HOME": "/wrong/home",
        "OTHER_SETTING": "retained",
    }
    before = dict(overrides)
    host_env = dict(os.environ)
    with FakeDocker(home="/home/runner", uid=1001, gid=1001) as fake:
        await sc.start()
        await sc._exec_in_container("env -0", env=overrides, timeout_sec=10)
        args = fake.calls_with_prefix("exec")[-1]
        overlay = dict(args[i + 1].split("=", 1) for i, arg in enumerate(args) if arg == "--env")
        assert "OVERRIDE_SECRET" not in " ".join(args)
        image_env = {**host_env, **dict.fromkeys(CREDENTIAL_ENV_VARS, "IMAGE_SECRET")}
        result = subprocess.run(
            ["sh", "-c", args[-1]], env={**image_env, **overlay}, capture_output=True, check=True
        )
        actual = dict(part.decode().split("=", 1) for part in result.stdout.split(b"\0") if part)
        assert actual["AWS_PROFILE"] == "PRIMARY"
        assert actual["AWS_EC2_METADATA_DISABLED"] == "true"
        assert all(key not in actual for key in CREDENTIAL_ENV_VARS if key != "AWS_PROFILE")
        for key in (
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
            "AWS_BEARER_TOKEN_BEDROCK",
            "OTHER_SETTING",
        ):
            assert actual[key] == before[key]
        assert actual["HOME"] == "/home/runner"
        assert overrides == before and dict(os.environ) == host_env
        await sc.stop(delete=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["raise", "stall"])
async def test_expired_credentials_abort_phase_and_refuse_later_phases(sc, monkeypatch, failure):
    monkeypatch.setattr(credentials_module, "CRED_REFRESH_MIN_SLEEP_SEC", 0.005)
    phase_started = asyncio.Event()
    phase_cancelled = asyncio.Event()
    worker_release = threading.Event()
    minted = 0

    def mint(*args):
        nonlocal minted
        minted += 1
        if minted == 1:
            return _generation(
                "OLD", expires_at=datetime.now(timezone.utc) + timedelta(seconds=0.2)
            )
        if failure == "raise":
            raise RuntimeError("synthetic mint failure")
        assert worker_release.wait(3), "synthetic mint was not released"
        return _generation("LATE")

    async def execute(args, stdin):
        if "stdout.txt" in args[-1]:
            phase_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                phase_cancelled.set()
        return 0, b"", b""

    monkeypatch.setattr(container_module, "mint_credentials", mint)
    with FakeDocker() as fake:
        fake.when_callable("exec", responder=execute)
        await sc.start()
        phase = asyncio.create_task(sc.run_phase("deploy", env={}, timeout_sec=10))
        try:
            await asyncio.wait_for(phase_started.wait(), timeout=1)
            done, _ = await asyncio.wait({phase}, timeout=1)
            assert phase in done, "the scenario consumer outlived its credentials"
            with pytest.raises(CredentialError, match="credential"):
                await phase
            assert phase_cancelled.is_set()
            with pytest.raises(CredentialError, match="credential"):
                await sc.run_phase("deploy", env={}, timeout_sec=10)
        finally:
            phase.cancel()
            await asyncio.gather(phase, return_exceptions=True)
            await sc.stop(delete=True)
            worker_release.set()
        assert not sc.is_started
        assert sc._creds_root is None


@pytest.mark.asyncio
async def test_completed_phase_rejects_pending_credential_cancellation(sc, monkeypatch):
    """A completed phase must not outrun the refresher's expiry cancellation."""
    import time

    monkeypatch.setattr(credentials_module, "CRED_REFRESH_MIN_SLEEP_SEC", 0.005)
    attempted = asyncio.Event()
    loop = asyncio.get_running_loop()
    minted = False
    states = []
    original_wait = asyncio.wait

    def mint(*args):
        nonlocal minted
        if not minted:
            minted = True
            return _generation(
                "OLD", expires_at=datetime.now(timezone.utc) + timedelta(seconds=0.2)
            )
        loop.call_soon_threadsafe(attempted.set)
        raise RuntimeError("synthetic mint failure")

    async def execute(args, stdin):
        if "stdout.txt" in args[-1]:
            await attempted.wait()
            # Finish after expiry before the event loop can run its timer callback.
            time.sleep(0.25)
        return 0, b"", b""

    async def observe_wait(*args, **kwargs):
        result = await original_wait(*args, **kwargs)
        states.append((sc._refresh_task.done(), sc._refresh_task.cancelling()))
        return result

    monkeypatch.setattr(container_module, "mint_credentials", mint)
    monkeypatch.setattr(asyncio, "wait", observe_wait)
    with FakeDocker() as fake:
        fake.when_callable("exec", responder=execute)
        await sc.start()
        try:
            with pytest.raises(CredentialError, match="credential refresh stopped"):
                await asyncio.wait_for(sc.run_phase("deploy", env={}, timeout_sec=10), timeout=2)
            assert states == [(False, 1)]
        finally:
            await sc.stop(delete=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["raise", "cancel", "return"])
async def test_terminated_refresher_refuses_phase(sc, monkeypatch, termination):
    async def stopped(*args):
        if termination == "raise":
            raise CredentialError("synthetic credential failure")
        if termination == "cancel":
            raise asyncio.CancelledError

    monkeypatch.setattr(container_module, "refresh_credentials_loop", stopped)
    with FakeDocker():
        await sc.start()
        assert sc._refresh_task is not None
        await asyncio.gather(sc._refresh_task, return_exceptions=True)
        try:
            with pytest.raises(CredentialError, match="credential"):
                await sc.run_phase("deploy", env={}, timeout_sec=10)
        finally:
            await sc.stop(delete=True)


@pytest.mark.asyncio
async def test_refresh_generations_retry_without_extra_delay(
    sc, monkeypatch, refresh_clock, caplog
):
    delays, ticks = refresh_clock
    first = _generation("FIRST")
    second = _generation("SECOND", expires_at=first[1] + timedelta(hours=1))
    results = iter([first, RuntimeError("DO_NOT_LOG_SECRET"), second])
    main_thread = threading.get_ident()
    mint_calls = []

    def mint(*args):
        assert threading.get_ident() != main_thread
        mint_calls.append(args)
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(container_module, "mint_credentials", mint)
    with FakeDocker() as fake:
        await sc.start()
        path = sc._creds_dir / "PRIMARY.json"
        assert await delays.get() == pytest.approx(2700, abs=5)
        assert len(mint_calls) == 1
        assert json.loads(path.read_text())["AccessKeyId"] == "FIRST"
        ticks.put_nowait(None)
        assert await delays.get() == 60
        assert json.loads(path.read_text())["AccessKeyId"] == "FIRST"
        assert "Credential refresh failed" in caplog.text
        assert "DO_NOT_LOG_SECRET" not in caplog.text
        ticks.put_nowait(None)
        assert await delays.get() == pytest.approx(6300, abs=5)
        assert json.loads(path.read_text())["AccessKeyId"] == "SECOND"
        assert len(mint_calls) == 3
        assert all(call[1:] == mint_calls[0][1:] for call in mint_calls)
        assert len(fake.calls_with_prefix("cp")) == 1
        await sc.stop(delete=True)


@pytest.mark.asyncio
async def test_failed_publication_keeps_previous_generation(sc, monkeypatch):
    with FakeDocker():
        await sc.start()
        path = sc._creds_dir / "PRIMARY.json"
        before = path.read_text()
        mint = MagicMock(return_value=_generation("NEW"))
        monkeypatch.setattr(container_module, "mint_credentials", mint)
        with patch.object(container_module.os, "replace", side_effect=OSError("replace failed")):
            with pytest.raises(OSError, match="replace failed"):
                await sc._refresh_credentials()
        assert path.read_text() == before
        assert list(sc._creds_dir.iterdir()) == [path]
        await sc._refresh_credentials()
        assert json.loads(path.read_text())["AccessKeyId"] == "NEW"
        await sc.stop(delete=True)


@pytest.mark.asyncio
async def test_atomic_publication_with_continuous_readers(sc, monkeypatch):
    done = threading.Event()
    seen = threading.Event()
    errors: list[Exception] = []
    read_keys: set[str] = set()
    counter = 0
    main_thread = threading.get_ident()
    replace = os.replace

    def mint(*args):
        nonlocal counter
        assert threading.get_ident() != main_thread
        counter += 1
        return _generation(f"GEN_{counter}")

    def publish(src, dst):
        assert threading.get_ident() == main_thread
        assert Path(src).stat().st_mode & 0o777 == 0o644
        replace(src, dst)

    monkeypatch.setattr(container_module, "mint_credentials", mint)
    monkeypatch.setattr(container_module.os, "replace", publish)
    with FakeDocker():
        await sc.start()
        path = sc._creds_dir / "PRIMARY.json"

        def read():
            while not done.is_set():
                try:
                    body = json.loads(path.read_text())
                    assert body["SecretAccessKey"] == body["AccessKeyId"] + "-secret"
                    read_keys.add(body["AccessKeyId"])
                    seen.set()
                except Exception as exc:
                    errors.append(exc)
                    break

        reader = threading.Thread(target=read)
        reader.start()
        try:
            assert await asyncio.to_thread(seen.wait, 5)
            for _ in range(30):
                await sc._refresh_credentials()
        finally:
            done.set()
            reader.join(timeout=5)
            await sc.stop(delete=True)
        assert not reader.is_alive()
        assert not errors
        assert len(read_keys) > 1


@pytest.mark.asyncio
@pytest.mark.parametrize("during_refresh", [False, True])
async def test_cancel_mint_returns_without_waiting_for_data_only_worker(
    sc, monkeypatch, refresh_clock, during_refresh
):
    delays, ticks = refresh_clock
    entered = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    loop = asyncio.get_running_loop()
    calls = 0

    def mint(*args):
        nonlocal calls
        calls += 1
        if during_refresh and calls == 1:
            return _generation("OLD")
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), "test did not release the mint"
        finished.set()
        return _generation("CANCELLED")

    monkeypatch.setattr(container_module, "mint_credentials", mint)
    with FakeDocker() as fake:
        starter = asyncio.create_task(sc.start())
        if during_refresh:
            await starter
            await delays.get()
            ticks.put_nowait(None)
        await entered.wait()
        root = sc._creds_root
        assert root is not None
        if during_refresh:
            operation = asyncio.create_task(sc.stop(delete=True))
        else:
            starter.cancel()
            operation = starter
        try:
            if during_refresh:
                await asyncio.wait_for(operation, timeout=2)
            else:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(operation, timeout=2)
                assert not fake.calls_with_prefix("run")
            assert not finished.is_set()
            assert not root.exists()
            assert sc._creds_dir is None
            assert sc._refresh_task is None
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 5)
        await asyncio.sleep(0)
        assert not root.exists()
        await sc.stop(delete=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["mint", "config"])
async def test_explicit_start_shutdown_cleans_up_and_preserves_error(
    sc, monkeypatch, tmp_path, stage
):
    shutdown = OperationCancelled("shutdown requested")
    if stage == "mint":
        monkeypatch.setattr(container_module, "mint_credentials", MagicMock(side_effect=shutdown))
    else:

        async def prepare():
            raise shutdown

        monkeypatch.setattr(sc, "_prepare_aws_config", prepare)
    with FakeDocker() as fake:
        with pytest.raises(OperationCancelled) as error:
            await sc.start()
        assert error.value is shutdown
        assert sc._creds_root is None and sc._creds_dir is None
        assert sc._startup_task is None
        assert not list(tmp_path.glob("awsbench-creds-*"))
        if stage == "config":
            assert fake.calls_with_prefix("stop")
            assert len(fake.calls_with_prefix("rm")) == 2


@pytest.mark.asyncio
async def test_stop_handles_startup_shutdown_during_cancellation(sc, monkeypatch):
    start = sc._start
    entered = asyncio.Event()
    shutdown = OperationCancelled("shutdown while startup unwinds")

    async def bootstrap():
        await start()
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            raise shutdown

    monkeypatch.setattr(sc, "_start", bootstrap)
    with FakeDocker() as fake:
        starter = asyncio.create_task(sc.start())
        await entered.wait()
        root = sc._creds_root
        await sc.stop(delete=True)
        with pytest.raises(OperationCancelled) as error:
            await starter
        assert error.value is shutdown
        assert not root.exists()
        assert sc._creds_root is None
        assert fake.calls_with_prefix("stop")


@pytest.mark.asyncio
async def test_stop_cleans_up_after_completed_refresher_shutdown(sc, monkeypatch, refresh_clock):
    delays, ticks = refresh_clock
    shutdown = OperationCancelled("refresher shutdown")
    monkeypatch.setattr(
        container_module,
        "mint_credentials",
        MagicMock(side_effect=[_generation("OLD"), shutdown]),
    )
    with FakeDocker() as fake:
        await sc.start()
        root = sc._creds_root
        await delays.get()
        ticks.put_nowait(None)
        with pytest.raises(OperationCancelled) as error:
            await sc._refresh_task
        assert error.value is shutdown
        assert root.exists()
        await sc.stop(delete=True)
        assert not root.exists()
        assert sc._refresh_task is None
        assert fake.calls_with_prefix("stop")
        assert len(fake.calls_with_prefix("rm")) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("during_refresh", [False, True])
async def test_same_loop_cancel_before_atomic_replace(sc, monkeypatch, during_refresh):
    write_text = Path.write_text
    root = None

    def cancel_write(path, *args, **kwargs):
        nonlocal root
        result = write_text(path, *args, **kwargs)
        if path.name == "PRIMARY.json.tmp":
            root = path.parent.parent
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
        return result

    with FakeDocker():
        if during_refresh:
            await sc.start()
        monkeypatch.setattr(Path, "write_text", cancel_write)
        monkeypatch.setattr(container_module, "mint_credentials", lambda *args: _generation("NEW"))
        operation = asyncio.create_task(sc._refresh_credentials() if during_refresh else sc.start())
        with pytest.raises(asyncio.CancelledError):
            await operation
        if during_refresh:
            assert (
                json.loads((sc._creds_dir / "PRIMARY.json").read_text())["AccessKeyId"]
                == "AKIATEST"
            )
            assert not (sc._creds_dir / "PRIMARY.json.tmp").exists()
            await sc.stop(delete=True)
        assert root is not None and not root.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("upload_fails", [False, True])
async def test_cancel_config_upload_settles_before_cleanup(sc, monkeypatch, upload_fails):
    entered = asyncio.Event()
    release = asyncio.Event()
    uploaded = False
    rmtree = shutil.rmtree

    async def upload(args, stdin):
        nonlocal uploaded
        assert stdin is not None
        entered.set()
        await release.wait()
        uploaded = True
        return int(upload_fails), b"", b"upload failed" if upload_fails else b""

    def remove(path, *args, **kwargs):
        assert uploaded, "credential disposal raced the config upload"
        return rmtree(path, *args, **kwargs)

    with FakeDocker() as fake:
        fake.when_callable("cp", "-", responder=upload)
        monkeypatch.setattr(container_module.shutil, "rmtree", remove)
        starter = asyncio.create_task(sc.start())
        await entered.wait()
        root = sc._creds_root
        starter.cancel()
        await asyncio.sleep(0)
        starter.cancel()  # A repeated cancellation must not abandon Docker stdin.
        await asyncio.sleep(0)
        assert not starter.done() and root.exists()
        assert not fake.calls_with_prefix("stop")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await starter
        assert uploaded and not root.exists()
        assert not any("mv -f " in args[-1] for args in fake.calls_with_prefix("exec"))
        assert any(
            "rm -f /root/.aws/.config-" in args[-1] for args in fake.calls_with_prefix("exec")
        )
        assert fake.calls_with_prefix("stop")


@pytest.mark.asyncio
async def test_cancel_docker_creation_waits_then_removes_partial_container(sc):
    entered = asyncio.Event()
    release = asyncio.Event()
    with FakeDocker() as fake:

        async def spawn(*cmd, **kwargs):
            if cmd[1:3] == ("run", "--detach"):
                entered.set()
                await release.wait()
            return await fake._fake_exec(*cmd, **kwargs)

        with patch.object(container_module.asyncio, "create_subprocess_exec", new=spawn):
            starter = asyncio.create_task(sc.start())
            await entered.wait()
            root = sc._creds_root
            assert not sc.is_started
            starter.cancel()
            await asyncio.sleep(0)
            assert root.exists() and not starter.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await starter
        assert not root.exists()
        assert fake.calls_with_prefix("stop")
        assert len(fake.calls_with_prefix("rm")) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["mint", "run", "read", "upload", "move", "access"])
async def test_failed_start_disposes_credentials_and_stops_container(
    sc, monkeypatch, tmp_path, failure
):
    with FakeDocker() as fake:
        if failure == "mint":
            monkeypatch.setattr(
                container_module,
                "mint_credentials",
                MagicMock(side_effect=RuntimeError("mint failed")),
            )
        elif failure == "run":
            fake.when("run", rc=1, stderr=b"run failed")
        elif failure == "upload":
            fake.when("cp", "-", rc=1, stderr=b"upload failed")
        else:
            marker = {"read": "cat /root/.aws/config", "move": "mv -f ", "access": "test -r "}[
                failure
            ]
            fake.when_callable(
                "exec",
                responder=lambda args, _stdin: (1 if marker in args[-1] else 0, b"", b""),
            )
        with pytest.raises(RuntimeError):
            await sc.start()
        assert sc._creds_root is None and sc._creds_dir is None
        assert sc._refresh_task is None
        assert not sc.is_started
        assert not list(tmp_path.glob("awsbench-creds-*"))
        if failure != "mint":
            assert fake.calls_with_prefix("stop")
            assert len(fake.calls_with_prefix("rm")) == 2
        await sc.stop(delete=True)


@pytest.mark.asyncio
async def test_final_cleanup_warns_and_continues_stop_then_retries(sc, caplog):
    with FakeDocker() as fake:
        await sc.start()
        root, inner = sc._creds_root, sc._creds_dir
        (inner / ".hidden.tmp").write_text("pending")
        (inner / "nested").mkdir()
        (inner / "nested" / "leftover").write_text("pending")
        with patch.object(container_module.shutil, "rmtree", side_effect=PermissionError("denied")):
            await sc.stop(delete=False)
        assert fake.calls_with_prefix("stop")
        assert "Failed to remove scenario credentials" in caplog.text
        assert "denied" in caplog.text
        assert sc._creds_root == root and sc._creds_dir == inner
        assert root.exists() and not sc.is_started
        await sc.stop(delete=True)
        assert sc._creds_root is None and sc._creds_dir is None
        assert not root.exists()


@pytest.mark.asyncio
async def test_stop_removes_host_files_even_when_started_flag_is_false(sc):
    with FakeDocker():
        await sc.start()
        root = sc._creds_root
        sc._started = False
        await sc.stop(delete=True)
        assert not root.exists()
        assert sc._creds_root is None


@pytest.mark.asyncio
async def test_lifetime_rechecked_after_bootstrap(sc, monkeypatch):
    validate = MagicMock(
        side_effect=[100, 100, 100, CredentialError("bootstrap consumed lifetime")]
    )
    monkeypatch.setattr(container_module, "credential_refresh_delay", validate)
    with FakeDocker() as fake:
        with pytest.raises(CredentialError, match="bootstrap consumed lifetime"):
            await sc.start()
        assert sc._refresh_task is None
        assert sc._creds_root is None
        assert fake.calls_with_prefix("stop")


@pytest.mark.asyncio
async def test_account_mapping_is_captured(sc):
    original = {"PRIMARY": "111122223333"}
    provider = _fake_cred_provider()
    sc = ScenarioContainer(
        sc._paths,
        sc._env_config,
        image_tag="test",
        container_name="captured-scope",
        host_logs_dir=sc._host_logs_dir,
        cred_provider=provider,
        account_mapping=original,
    )
    original["PRIMARY"] = "999999999999"
    with FakeDocker():
        await sc.start()
        await sc._refresh_credentials()
        assert all(
            call.args == ("111122223333", ORG_ACCESS_ROLE, "app-session-PRIMARY")
            for call in provider.get_chained_session_for_account.call_args_list
        )
        await sc.stop(delete=True)


@pytest.mark.asyncio
async def test_teardown_errors_do_not_replace_start_failure(sc, caplog):
    with FakeDocker() as fake:
        fake.when("run", rc=1, stderr=b"original start failure")
        fake.when("info", rc=1, stderr=b"daemon unavailable")

        async def stop_fails(*args, **kwargs):
            raise OSError("stop transport unavailable")

        fake.when_callable("stop", responder=stop_fails)
        with pytest.raises(DockerCLIError, match="original start failure"):
            await sc.start()
        assert len(fake.calls_with_prefix("rm")) == 2
        assert "stop transport unavailable" in caplog.text
        assert sc._creds_root is None


@pytest.mark.asyncio
async def test_phase_exec_keeps_its_timeout(sc):
    with FakeDocker() as fake:
        await sc.start()

        async def hang(args, stdin):
            await asyncio.Event().wait()
            return 0, b"", b""

        fake.when_callable("exec", responder=hang)
        with pytest.raises(TimeoutError):
            await sc._exec_in_container("sleep infinity", env=None, timeout_sec=0.01)
        fake._matchers.clear()
        await sc.stop(delete=True)


@pytest.mark.asyncio
async def test_stop_during_start_cannot_publish_after_cleanup(sc, monkeypatch):
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def mint(*args):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return _generation("NEVER_PUBLISHED")

    monkeypatch.setattr(container_module, "mint_credentials", mint)
    with FakeDocker() as fake:
        starter = asyncio.create_task(sc.start())
        await entered.wait()
        root = sc._creds_root
        stopper = asyncio.create_task(sc.stop(delete=True))
        await asyncio.sleep(0)
        assert not stopper.done()
        release.set()
        await stopper
        with pytest.raises(asyncio.CancelledError):
            await starter
        assert not root.exists()
        assert not fake.calls_with_prefix("run")
        assert sc._startup_task is None


@pytest.mark.asyncio
async def test_cancel_stop_still_waits_for_disposal(sc, monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    with FakeDocker() as fake:
        await sc.start()
        root = sc._creds_root
        cleanup = sc._stop_credential_refresh

        async def delayed_cleanup():
            entered.set()
            await release.wait()
            await cleanup()

        monkeypatch.setattr(sc, "_stop_credential_refresh", delayed_cleanup)
        stopper = asyncio.create_task(sc.stop(delete=True))
        await entered.wait()
        stopper.cancel()
        await asyncio.sleep(0)
        assert not stopper.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await stopper
        assert not root.exists()
        assert fake.calls_with_prefix("stop")
        assert len(fake.calls_with_prefix("rm")) == 2


@pytest.mark.asyncio
async def test_rejected_initial_expiry_never_starts_consumer(sc, monkeypatch):
    expiry = datetime.now(timezone.utc) + timedelta(seconds=10)
    monkeypatch.setattr(
        container_module,
        "mint_credentials",
        MagicMock(return_value=_generation("SHORT", expires_at=expiry)),
    )
    with FakeDocker() as fake:
        with pytest.raises(CredentialError, match="minimum refresh sleep"):
            await sc.start()
        assert not fake.calls_with_prefix("run")
        assert sc._creds_root is None


@pytest.mark.asyncio
async def test_mint_cannot_change_reserved_filenames(sc, monkeypatch):
    with FakeDocker():
        await sc.start()
        expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        monkeypatch.setattr(
            container_module,
            "mint_credentials",
            MagicMock(return_value=({"../escape": "{}"}, expiry)),
        )
        with pytest.raises(ValueError, match="unexpected scenario filenames"):
            await sc._refresh_credentials()
        assert not (sc._creds_root / "escape").exists()
        await sc.stop(delete=True)


@pytest.mark.asyncio
async def test_snapshot_expiring_during_mint_does_not_replace_good_file(sc, monkeypatch):
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(credentials_module, "CRED_REFRESH_MIN_SLEEP_SEC", 0.001)

    def mint(*args):
        snapshot = _generation(
            "EXPIRED", expires_at=datetime.now(timezone.utc) + timedelta(milliseconds=30)
        )
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return snapshot

    with FakeDocker():
        await sc.start()
        path = sc._creds_dir / "PRIMARY.json"
        before = path.read_bytes()
        monkeypatch.setattr(container_module, "mint_credentials", mint)
        refresh = asyncio.create_task(sc._refresh_credentials())
        await entered.wait()
        await asyncio.sleep(0.05)
        release.set()
        with pytest.raises(CredentialError, match="future"):
            await refresh
        assert path.read_bytes() == before
        assert list(sc._creds_dir.iterdir()) == [path]
        await sc.stop(delete=True)


@pytest.mark.asyncio
async def test_expiry_rechecked_immediately_before_atomic_replace(sc, monkeypatch):
    with FakeDocker():
        await sc.start()
        path = sc._creds_dir / "PRIMARY.json"
        before = path.read_bytes()
        monkeypatch.setattr(container_module, "mint_credentials", lambda *args: _generation("NEW"))
        validate = MagicMock(
            side_effect=[100, CredentialError("snapshot expired during file preparation")]
        )
        monkeypatch.setattr(container_module, "credential_refresh_delay", validate)
        with pytest.raises(CredentialError, match="expired during file preparation"):
            await sc._refresh_credentials()
        assert path.read_bytes() == before
        assert not (sc._creds_dir / "PRIMARY.json.tmp").exists()
        await sc.stop(delete=True)


@pytest.fixture
def real_docker(sc, monkeypatch):
    """Use isolated local containers with synthetic credentials and no network."""
    sc._image_tag = f"awsbench-creds-check-{secrets.token_hex(6)}"
    sc._container_name = f"{sc._image_tag}-container"
    capture = sc._run_docker_capture

    async def isolated(args, **kwargs):
        args = list(args)
        if args[:2] == ["run", "--detach"]:
            args[2:2] = ["--network", "none"]
        return await capture(args, **kwargs)

    monkeypatch.setattr(sc, "_run_docker_capture", isolated)
    try:
        yield sc
    finally:
        for args in (
            ["docker", "rm", "-f", sc.container_name],
            ["docker", "image", "rm", sc.image_tag],
        ):
            subprocess.run(args, capture_output=True, timeout=30, check=False)


@pytest.mark.skipif(
    os.environ.get("AWS_BENCH_SCENARIO_DOCKER_TESTS") != "1",
    reason="set AWS_BENCH_SCENARIO_DOCKER_TESTS=1 for local Docker checks",
)
@pytest.mark.timeout(120)
@pytest.mark.asyncio
@pytest.mark.parametrize("uid", [0, 1001])
@pytest.mark.parametrize("existing_config", [False, True])
async def test_docker_user_home_readonly_refresh_and_stopped_cleanup(
    real_docker, monkeypatch, refresh_clock, uid, existing_config
):
    sc = real_docker
    delays, ticks = refresh_clock
    home = "/root" if uid == 0 else "/home/runner"
    context = sc._paths.build_context_dir
    original = (
        "# preserve this exact text\n"
        "[profile unrelated]\n"
        "aws_secret_access_key = UNRELATED_SECRET\n"
        "[profile PRIMARY]\n"
        "region = eu-west-1\n"
    )
    static = "[OTHER]\naws_access_key_id = OTHER_KEY\n"
    (context / "aws-config").write_text(original)
    (context / "aws-credentials").write_text(static)
    mounted_credentials = None
    if existing_config and uid == 1001:
        mounted_credentials = context / "aws-credentials"
        sc._env_config.mounts_json = [
            {
                "type": "bind",
                "source": str(mounted_credentials),
                "target": f"{home}/.aws/credentials",
                "read_only": True,
            }
        ]
    setup_config = ""
    if existing_config:
        setup_config = (
            "COPY aws-config /tmp/aws-config\n"
            "COPY aws-credentials /tmp/aws-credentials\n"
            'RUN mkdir -p "$HOME/.aws" && cp /tmp/aws-config "$HOME/.aws/config" && '
            'cp /tmp/aws-credentials "$HOME/.aws/credentials" && '
            f'chown -R {uid}:{uid} "$HOME/.aws" && chmod 700 "$HOME/.aws"\n'
        )
    (context / "Dockerfile").write_text(
        "FROM alpine:3.20\n"
        "RUN addgroup -g 1001 runner && adduser -D -u 1001 -G runner runner\n"
        f"ENV HOME={home}\n"
        "ENV AWS_ACCESS_KEY_ID=IMAGE_KEY AWS_SECRET_ACCESS_KEY=IMAGE_SECRET "
        "AWS_CONFIG_FILE=/tmp/wrong AWS_DEFAULT_PROFILE=OLD "
        "AWS_CONTAINER_CREDENTIALS_FULL_URI=http://127.0.0.1:9/creds\n"
        f"{setup_config}"
        f"USER {uid}:{uid}\n"
    )
    checks = "\n".join(
        f'test "${{{key}+set}}" != set' for key in CREDENTIAL_ENV_VARS if key != "AWS_PROFILE"
    )
    sc._paths.phase_script_path("deploy").write_text(
        "#!/bin/sh\nset -eu\n"
        f'test "$HOME" = {shlex.quote(home)}\n'
        f'test "$(id -u)" = {uid}\n'
        f'test "$(stat -c %a:%u:%g "$HOME/.aws/config")" = 600:{uid}:{uid}\n'
        'test "$AWS_PROFILE" = PRIMARY\n'
        'test "$AWS_EC2_METADATA_DISABLED" = true\n'
        'test "$AWS_BEARER_TOKEN_BEDROCK" = MODEL_TOKEN\n'
        'test "$AWS_REGION" = ap-south-1\n'
        f"{checks}\n"
        "if command -v python3 >/dev/null 2>&1; then exit 1; fi\n"
        'path="$HOME/.aws/creds/PRIMARY.json"\n'
        'test -r "$path"\n'
        'if (printf broken > "$path") 2>/dev/null; then exit 1; fi\n'
        'if touch "$HOME/.aws/creds/.new" 2>/dev/null; then exit 1; fi\n'
        'if rm "$path" 2>/dev/null; then exit 1; fi\n'
        'grep -q \'"AccessKeyId": "FIRST"\' "$path"\n'
        "touch /logs/deploy/ready\n"
        'until grep -q \'"AccessKeyId": "SECOND"\' "$path"; do sleep 0.01; done\n'
        "echo phase-ok\n"
    )
    generations = iter([_generation("FIRST"), _generation("SECOND")])
    monkeypatch.setattr(container_module, "mint_credentials", lambda *args: next(generations))
    root = None
    try:
        await sc.build(timeout_sec=60)
        await sc.start()
        root = sc._creds_root
        inner = sc._creds_dir
        assert root.stat().st_mode & 0o777 == 0o700
        assert inner.stat().st_mode & 0o777 == 0o755
        assert sc._container_home == PurePosixPath(home)
        await delays.get()
        phase = asyncio.create_task(
            sc.run_phase(
                "deploy",
                env={
                    **dict.fromkeys(CREDENTIAL_ENV_VARS, "PHASE_SECRET"),
                    "HOME": "/wrong",
                    "AWS_BEARER_TOKEN_BEDROCK": "MODEL_TOKEN",
                    "AWS_REGION": "ap-south-1",
                },
                timeout_sec=30,
            )
        )
        ready = sc._host_logs_dir / "deploy" / "ready"
        async with asyncio.timeout(15):
            while not ready.exists():
                if phase.done():
                    result = phase.result()
                    pytest.fail(f"phase exited before refresh: {result}")
                await asyncio.sleep(0.01)
        ticks.put_nowait(None)
        await delays.get()
        result = await phase
        assert result.exit_code == 0 and result.stdout == "phase-ok\n"
        assert json.loads((inner / "PRIMARY.json").read_text())["AccessKeyId"] == "SECOND"
        rc = await sc._exec_in_container(
            'touch "$HOME/.aws/creds/.root-write"', env=None, timeout_sec=10, user="0"
        )
        assert rc != 0, "even container root must see a read-only mount"
        await sc.stop(delete=False)
        assert not root.exists()
        if mounted_credentials is not None:
            assert mounted_credentials.read_text() == static
        # A stopped container still has its config. Host cleanup removed the keys.
        # Export reads the stopped filesystem without remounting the deleted bind source.
        _, archive, _ = await sc._run_docker_capture(["export", sc.container_name])
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            config_file = tar.extractfile(f"{home.lstrip('/')}/.aws/config")
            assert config_file is not None
            config = config_file.read().decode()
            if existing_config:
                credentials_file = tar.extractfile(f"{home.lstrip('/')}/.aws/credentials")
                assert credentials_file is not None and credentials_file.read().decode() == static
        assert config == build_aws_config(
            ["PRIMARY"], original if existing_config else "", region="us-east-1"
        )
    finally:
        await sc.stop(delete=True)
        if root is not None:
            assert not root.exists()


@pytest.mark.skipif(
    os.environ.get("AWS_BENCH_SCENARIO_DOCKER_TESTS") != "1",
    reason="set AWS_BENCH_SCENARIO_DOCKER_TESTS=1 for local Docker checks",
)
@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_docker_readonly_credentials_mount_rejects_matching_profile(real_docker, tmp_path):
    sc = real_docker
    (sc._paths.build_context_dir / "Dockerfile").write_text("FROM alpine:3.20\n")
    source = tmp_path / "shared-credentials"
    original = "[PRIMARY]\naws_access_key_id = MOUNTED_KEY\n"
    source.write_text(original)
    sc._env_config.mounts_json = [
        {
            "type": "bind",
            "source": str(source),
            "target": "/root/.aws/credentials",
            "read_only": True,
        }
    ]
    try:
        await sc.build(timeout_sec=60)
        with pytest.raises(CredentialError, match="contains selected profile"):
            await sc.start()
        assert source.read_text() == original
        assert sc._creds_root is None and sc._creds_dir is None
        assert not sc.is_started
    finally:
        await sc.stop(delete=True)


@pytest.mark.skipif(
    os.environ.get("AWS_BENCH_SCENARIO_DOCKER_TESTS") != "1",
    reason="set AWS_BENCH_SCENARIO_DOCKER_TESTS=1 for local Docker checks",
)
@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_docker_native_cli_and_persistent_sdk_across_expiration(
    real_docker, monkeypatch, refresh_clock, tmp_path, caplog
):
    from botocore.configloader import load_config
    from botocore.credentials import ProcessProvider

    sc = real_docker
    delays, ticks = refresh_clock
    (sc._paths.build_context_dir / "Dockerfile").write_text(
        "FROM public.ecr.aws/aws-cli/aws-cli:latest\n"
        "ENTRYPOINT []\n"
        "ENV AWS_CONFIG_FILE=/tmp/wrong AWS_DEFAULT_PROFILE=OLD "
        "AWS_ACCESS_KEY_ID=IMAGE_KEY AWS_SECRET_ACCESS_KEY=IMAGE_SECRET\n"
    )
    monkeypatch.setattr(credentials_module, "CRED_REFRESH_MIN_SLEEP_SEC", 0.01)
    monkeypatch.setattr(credentials_module, "_CRED_REFRESH_SKEW_SEC", 1)
    minted = []

    def mint(*args):
        expiry = datetime.now(timezone.utc) + timedelta(seconds=8 if not minted else 3600)
        result = _generation("NATIVE_OLD" if not minted else "NATIVE_NEW", expires_at=expiry)
        minted.append(result)
        return result

    monkeypatch.setattr(container_module, "mint_credentials", mint)
    try:
        await sc.build(timeout_sec=60)
        await sc.start()
        native_home = tmp_path / "native-home"
        aws_dir = native_home / ".aws"
        aws_dir.mkdir(parents=True)
        (aws_dir / "creds").symlink_to(sc._creds_dir, target_is_directory=True)
        (aws_dir / "config").write_text(build_aws_config(["PRIMARY"]))
        provider = ProcessProvider(
            profile_name="PRIMARY",
            load_config=lambda: load_config(str(aws_dir / "config")),
            # Botocore accepts a factory; its inferred annotation requires a class.
            popen=partial(  # type: ignore[arg-type]
                subprocess.Popen, env={**os.environ, "HOME": str(native_home)}
            ),
        )
        sdk_credentials = provider.load()
        assert sdk_credentials is not None
        assert sdk_credentials.get_frozen_credentials().access_key == "NATIVE_OLD"
        command = "aws configure export-credentials --format process"
        rc, output, stderr = await sc._exec_in_container_capture(
            command, env={"AWS_PROFILE": "WRONG"}, timeout_sec=10
        )
        assert rc == 0, stderr
        assert json.loads(output)["AccessKeyId"] == "NATIVE_OLD"
        await delays.get()
        ticks.put_nowait(None)
        await delays.get()
        assert len(minted) == 2
        await asyncio.sleep(
            max(0, (minted[0][1] - datetime.now(timezone.utc)).total_seconds()) + 0.05
        )
        assert datetime.now(timezone.utc) > minted[0][1]
        assert sdk_credentials.get_frozen_credentials().access_key == "NATIVE_NEW"
        rc, output, stderr = await sc._exec_in_container_capture(command, env=None, timeout_sec=10)
        assert rc == 0, stderr
        assert json.loads(output)["AccessKeyId"] == "NATIVE_NEW"
        assert "NATIVE_OLD" not in caplog.text and "NATIVE_NEW" not in caplog.text
    finally:
        await sc.stop(delete=True)
