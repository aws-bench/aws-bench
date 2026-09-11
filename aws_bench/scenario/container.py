"""Subprocess wrapper around the ``docker`` CLI for scenario script execution.

Exposes the surface needed by ``ScenarioTrial``: build, start, run a
phase script, stop. Each ``ScenarioContainer`` instance owns one image
build + one running container for one scenario trial.

We shell out to the ``docker`` binary (assumed on ``$PATH``) instead of
using the docker-py SDK. Reasons:

  * Reproducible failures: an operator can copy a logged ``docker``
    invocation directly into their terminal.
  * Setup variance: ``DOCKER_HOST`` / ``DOCKER_CONTEXT`` / rootless /
    podman-as-docker / Docker Desktop on macOS all behave consistently
    via the CLI; docker-py historically does not.
  * Lighter footprint: no transitive dep tree (requests, urllib3,
    websocket-client, paramiko).

The container is the *scenario* container — the deploy/verify/cleanup
tooling box. It is not the agent's environment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import shlex
import shutil
import tarfile
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from tempfile import SpooledTemporaryFile

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.constants import DEFAULT_REGION
from aws_bench.logging.logger import get_logger
from aws_bench.scenario.config import EnvironmentConfig
from aws_bench.scenario.events import ScenarioPhase
from aws_bench.scenario.paths import ScenarioPaths
from aws_bench.utils.credentials_provider import (
    CredentialProvider,
    build_session_name,
    session_to_credential_process,
)

logger = get_logger(__name__)

# Host bind-mounts the per-tag credential_process files here.
CREDS_DIR = PurePosixPath("/awsbench-creds")
# Re-mint this long before Expiration, so the file is fresh before the token dies.
_CRED_REFRESH_SKEW_SEC = 15 * 60
# Sleep floor, so a credential already within the skew window can't spin the loop.
_CRED_REFRESH_MIN_SLEEP_SEC = 30
# Backoff after a failed assume before retrying.
_CRED_REFRESH_RETRY_SEC = 60

# write_file path constraint. Tilde lets the in-container shell expand
# $HOME; everything else is conservative ASCII so the unquoted path
# interpolation in `cat > <path>` cannot inject shell metacharacters.
_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_./~-]+$")


class DockerCLIError(RuntimeError):
    """Raised when a ``docker`` CLI invocation exits non-zero.

    Carries the failed command, exit code, and captured stderr so call
    sites can decide how to surface the failure.
    """

    def __init__(self, command: list[str], returncode: int, stderr: str) -> None:
        """Initialize with the failed command, its exit code, and captured stderr."""
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"docker command failed (exit {returncode}): {' '.join(command)}\nstderr:\n{stderr}"
        )


@dataclass
class ExecResult:
    """Outcome of one in-container script invocation."""

    exit_code: int
    stdout: str


class ScenarioContainer:
    """One scenario trial's container lifecycle.

    Single-container, no compose. Authors ship a Dockerfile; we build it,
    keep one container running for the lifetime of the trial, and run
    each phase script (deploy.sh / verify.sh / cleanup.sh) inside it.

    Container path layout:
      ``/<phase>/`` — script directory uploaded from
        ``<scenario_dir>/<phase>/``
      ``/logs/<phase>/`` — combined stdout+stderr written by the script.
        ``/logs`` is a bind mount of ``host_logs_dir`` so this appears on
        the host as it is written; no download step. The phase exit code
        comes from the ``docker exec`` return code, not a file.

    Concurrency: builds for the same image tag are deduplicated via a
    class-level lock map so parallel trials of the same scenario don't
    race the Docker daemon.
    """

    LOGS_DIR = PurePosixPath("/logs")
    KEEPALIVE_CMD = ["sleep", "infinity"]
    DOCKER_BIN = "docker"

    # Locks are keyed by (image_tag, loop_id) because asyncio.Lock binds to
    # the loop it is first awaited on; sharing across loops raises at runtime.
    _image_build_locks: dict[tuple[str, int], asyncio.Lock] = {}

    def __init__(
        self,
        paths: ScenarioPaths,
        env_config: EnvironmentConfig,
        *,
        image_tag: str,
        container_name: str,
        host_logs_dir: Path,
        cred_provider: CredentialProvider,
        account_mapping: dict[str, str],
        labels: dict[str, str] | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        """Initialize the container wrapper.

        Args:
            paths: Discovered scenario paths.
            env_config: Author-side resource limits (after operator
                overrides have been applied).
            image_tag: Tag to build the Docker image under. Must satisfy
                Docker image name rules.
            container_name: Name to assign the running container. Must
                satisfy Docker container name rules.
            host_logs_dir: Host directory bind-mounted at ``/logs`` so
                each phase's stdout/exit-code is visible to the host as
                it is written. The trial reads phase outputs from here.
            cred_provider: Source of management credentials. The container's
                per-account credentials are minted and refreshed from this by
                the host-side refresher (see ``_refresh_credentials_loop``).
            account_mapping: ``{account_tag: account_id}`` for the scenario.
                Each tag becomes an ``AWS_PROFILE`` the container's scripts can
                select; the refresher writes one credential file per tag.
            labels: Optional Docker labels applied to the container at ``run``
                time (``--label k=v``). Operational metadata for tooling to match
                containers by; empty by default.
            log: Optional logger override.
        """
        self._paths = paths
        self._env_config = env_config
        self._image_tag = image_tag
        self._container_name = container_name
        self._host_logs_dir = host_logs_dir
        self._cred_provider = cred_provider
        self._account_mapping = account_mapping
        self._labels = dict(labels or {})
        self._log = (log or logger).getChild(container_name)
        self._started = False
        # Cache for rootless-daemon detection (see _is_rootless_docker).
        self._rootless_docker: bool | None = None
        # Host dir bind-mounted at CREDS_DIR; holds one credential_process JSON
        # per account tag. Created in start(), removed in stop().
        self._creds_dir: Path | None = None
        # Background task that re-mints the credential files before they expire.
        self._refresh_task: asyncio.Task[None] | None = None

    @property
    def image_tag(self) -> str:
        """Resolved Docker image tag."""
        return self._image_tag

    @property
    def container_name(self) -> str:
        """Resolved Docker container name."""
        return self._container_name

    @property
    def is_started(self) -> bool:
        """Whether ``start`` has been called and not yet torn down."""
        return self._started

    # ── lifecycle ────────────────────────────────────────────────────────

    async def build(self, *, force: bool = False, timeout_sec: float | None = None) -> None:
        """Build the scenario image, deduplicating concurrent builds.

        Always invokes ``docker build``; the daemon's layer cache decides
        what to rebuild based on hashed Dockerfile + build-context content.
        Unchanged inputs make the build a sub-second cache-hit no-op;
        edits to the Dockerfile or build context invalidate the right
        layers automatically. We never short-circuit on tag-existence —
        an existing tag whose source has changed silently shipped stale
        bits to operators.

        ``force=True`` adds ``--no-cache`` to bust the daemon cache
        completely. Builds use the scenario's ``scenario/`` directory as
        the build context.

        ``timeout_sec`` bounds the build's total wall time; on timeout the
        underlying daemon call is cancelled and ``asyncio.TimeoutError``
        is raised.
        """
        loop_key = (self._image_tag, id(asyncio.get_running_loop()))
        lock = self._image_build_locks.setdefault(loop_key, asyncio.Lock())
        async with lock:
            self._log.info("Building image %s ...", self._image_tag)
            args = ["build", "--rm", "--force-rm", "--pull=false"]
            if force:
                args.append("--no-cache")
            args.extend(["-t", self._image_tag, str(self._paths.build_context_dir)])
            await asyncio.wait_for(self._run_docker(args), timeout=timeout_sec)
            self._log.info("Built image %s.", self._image_tag)

    async def start(self) -> None:
        """Start a detached, keepalive container ready to run scripts.

        Uses ``sleep infinity`` as the entrypoint so the container stays up
        for the trial's duration regardless of the image's ``CMD``. Resource
        limits come from the merged ``EnvironmentConfig``.
        """
        if self._started:
            raise RuntimeError("Container already started.")

        await self._remove_existing()
        self._host_logs_dir.mkdir(parents=True, exist_ok=True)
        # Mint the initial credential files before the container starts, so a
        # script's first AWS call resolves. Bind-mounted read-only; the refresher
        # rewrites them in place as they near expiry.
        self._creds_dir = Path(tempfile.mkdtemp(prefix=f"awsbench-creds-{self._container_name}-"))
        for tag in self._account_mapping:
            await asyncio.to_thread(self._write_creds_file, tag)
        # --mount k=v form over --volume so a host path containing ':' can't
        # be misparsed into the target or options field; operator-supplied
        # output dirs flow into self._host_logs_dir.
        args = [
            "run",
            "--detach",
            "--name",
            self._container_name,
            "--cpus",
            str(self._env_config.cpus),
            "--memory",
            f"{self._env_config.memory_mb}m",
            "--mount",
            f"type=bind,source={self._host_logs_dir.resolve()},target={self.LOGS_DIR}",
            "--mount",
            f"type=bind,source={self._creds_dir},target={CREDS_DIR},readonly",
        ]
        # Operational labels (sorted for a deterministic command) so tooling can
        # match containers by role rather than by name.
        for key, value in sorted(self._labels.items()):
            args.extend(["--label", f"{key}={value}"])
        for m in self._env_config.mounts_json:
            mount_str = f"type={m['type']},source={m['source']},target={m['target']}"
            if m.get("read_only"):
                mount_str += ",readonly"
            args.extend(["--mount", mount_str])
        args.extend([self._image_tag, *self.KEEPALIVE_CMD])
        await self._run_docker(args)
        self._started = True
        # Scripts select an account with AWS_PROFILE=<tag>; each profile's
        # credential_process reads the file the refresher keeps fresh.
        await self.write_file("~/.aws/config", self._build_aws_config())
        self._refresh_task = asyncio.create_task(self._refresh_credentials_loop())
        self._log.debug("Started container %s.", self._container_name)

    def _creds_file(self, tag: str) -> Path:
        """Host path of the credential_process JSON for account ``tag``."""
        assert self._creds_dir is not None
        return self._creds_dir / f"{tag}.json"

    def _build_aws_config(self) -> str:
        """Render ``~/.aws/config`` — one credential_process profile per account tag.

        The host writes pre-assumed member-account creds to the bind-mounted file;
        credential_process is the one source the SDK re-invokes on expiry, which
        keeps long scripts alive.
        """
        parts: list[str] = []
        for tag in self._account_mapping:
            container_path = CREDS_DIR / f"{tag}.json"
            parts.append(
                f"[profile {tag}]\n"
                f'credential_process = sh -c "cat {container_path}"\n'
                f"region = {DEFAULT_REGION}\n"
            )
        return "\n".join(parts)

    def _write_creds_file(self, tag: str) -> datetime:
        """Mint fresh creds for ``tag``, write its credential_process file, return expiry.

        Atomic (temp + ``os.replace``) so a reader never sees a partial file; mode
        0644 so the container (root) can read the bind-mounted file.
        """
        session = self._cred_provider.get_session_for_account(
            self._account_mapping[tag],
            ORG_ACCESS_ROLE,
            build_session_name("session", tag[-8:]),
        )
        creds = session_to_credential_process(session)
        path = self._creds_file(tag)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(creds))
        tmp.chmod(0o644)
        os.replace(tmp, path)
        return datetime.fromisoformat(str(creds["Expiration"]))

    async def _refresh_credentials_loop(self) -> None:
        """Re-mint every tag's credential file before the earliest expiry, until cancelled.

        A transient assume failure is retried after a backoff rather than killing
        the loop (which would strand the container credential-less).
        """
        while True:
            try:
                expiries = [
                    await asyncio.to_thread(self._write_creds_file, tag)
                    for tag in self._account_mapping
                ]
                now = datetime.now(timezone.utc)
                soonest = min((e - now).total_seconds() for e in expiries)
                sleep_for = max(soonest - _CRED_REFRESH_SKEW_SEC, _CRED_REFRESH_MIN_SLEEP_SEC)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — refresher must not die on a transient error
                self._log.warning("Credential refresh failed; retrying: %s", exc)
                sleep_for = _CRED_REFRESH_RETRY_SEC
            await asyncio.sleep(sleep_for)

    async def _remove_existing(self) -> None:
        """Remove a stale container with the same name, if any."""
        rc, _, stderr = await self._run_docker_capture(
            ["rm", "-f", self._container_name], check=False
        )
        if rc == 0:
            self._log.debug("Removed stale container %s.", self._container_name)
        elif "no such container" not in stderr.lower():
            # `docker rm -f <missing>` exits 1 with "no such container" — that's
            # the not-stale case. Anything else is a real failure worth logging.
            self._log.warning(
                "Failed to remove stale container %s: %s", self._container_name, stderr.strip()
            )

    async def stop(self, *, delete: bool) -> None:
        """Stop the container; remove it when ``delete`` is True."""
        if not self._started:
            return
        # Stop the refresher (and drop its credential dir) before tearing down.
        await self._stop_credential_refresh()
        # Fix ownership of bind-mounted /logs so the host user can read/write/
        # delete phase outputs after the (root) container is gone. Must run
        # while the container is still up (uses docker exec). See
        # _chown_logs_to_host_user for the cross-platform rationale.
        await self._chown_logs_to_host_user()
        rc, _, stderr = await self._run_docker_capture(
            ["stop", "-t", "10", self._container_name], check=False
        )
        if rc != 0:
            self._log.warning(
                "Failed to stop container %s: %s", self._container_name, stderr.strip()
            )
        if delete:
            rc, _, stderr = await self._run_docker_capture(
                ["rm", "-f", self._container_name], check=False
            )
            if rc != 0:
                self._log.warning(
                    "Failed to remove container %s: %s", self._container_name, stderr.strip()
                )
        self._started = False

    async def _stop_credential_refresh(self) -> None:
        """Cancel the refresh task and remove the host credential dir.

        Always removes the dir (it holds live member-account creds — a leak leaves
        a secret on disk), even if cancellation or rmtree hiccups.
        """
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 — teardown, never raise
                pass
            self._refresh_task = None
        if self._creds_dir is not None:
            shutil.rmtree(self._creds_dir, ignore_errors=True)
            self._creds_dir = None

    async def _is_rootless_docker(self) -> bool:
        """Return True if the Docker daemon runs in rootless mode (cached).

        In rootless Docker the user-namespace mapping makes container UID 0 own
        bind-mounted files on the host as the daemon (host) user, so chowning to
        UID 0 inside the container is what makes files host-accessible.
        """
        if self._rootless_docker is not None:
            return self._rootless_docker
        rc, stdout, _ = await self._run_docker_capture(
            ["info", "--format", "{{range .SecurityOptions}}{{.}}|{{end}}"], check=False
        )
        self._rootless_docker = rc == 0 and b"rootless" in stdout
        return self._rootless_docker

    async def _chown_logs_to_host_user(self) -> None:
        """Best-effort: chown the ``/logs`` bind mount to the host user.

        Parity with Harbor's ``DockerEnvironment.prepare_logs_for_host``. The
        scenario container runs as root, so everything it writes into the
        bind-mounted logs dir is ``root``-owned; the trial's host-side
        resource-management (reset/verify/cleanup) then can't write sibling
        paths (e.g. ``reset/<account_tag>``) and fails with EACCES on rootful
        Docker. Cross-platform:

          * Windows: no-op (``os.getuid`` is unavailable).
          * macOS/Windows Docker Desktop: effectively a no-op — the VM file-
            sharing layer maps ownership to the host user transparently.
          * Rootful Linux: chown to ``os.getuid():os.getgid()``.
          * Rootless Linux: chown to ``0:0`` — container root maps to the host
            user via the user namespace (``os.getuid()`` would select a subUID
            mapping to a different host UID and leave files inaccessible).

        Never raises: ownership correction is best-effort and must not fail a
        phase or teardown.
        """
        if not self._started or not hasattr(os, "getuid"):
            return
        if await self._is_rootless_docker():
            uid, gid = 0, 0
        else:
            uid, gid = os.getuid(), os.getgid()
        try:
            rc = await self._exec_in_container(
                f"chown -R {uid}:{gid} {shlex.quote(str(self.LOGS_DIR))}",
                env=None,
                timeout_sec=120,
            )
            if rc != 0:
                self._log.warning("chown -R %s:%s %s exited %s", uid, gid, self.LOGS_DIR, rc)
        except Exception as e:  # noqa: BLE001 — best-effort ownership fixup
            self._log.warning("Failed to chown %s to host user: %s", self.LOGS_DIR, e)

    async def write_file(self, container_path: str, content: str) -> None:
        """Write ``content`` to ``container_path`` inside the running container.

        Creates the parent directory and writes via a quoted-sentinel
        heredoc, so the body is treated as literal text (no shell
        expansion of ``$VAR`` or backticks). Path is interpolated
        unquoted so a leading ``~`` is expanded by the in-container
        shell; callers must pass paths matching ``[A-Za-z0-9_./~-]``.

        The heredoc sentinel is randomized when ``content`` contains
        the default token, so a payload cannot terminate the heredoc
        early.
        """
        self._require_started()
        if not _SAFE_PATH_RE.match(container_path):
            raise ValueError(
                f"Refusing to write {container_path!r}: path must match [A-Za-z0-9_./~-]"
            )
        parent = str(PurePosixPath(container_path).parent)
        sentinel = self._unique_heredoc_sentinel(content)
        cmd = f"mkdir -p {parent} && cat > {container_path} <<'{sentinel}'\n{content}\n{sentinel}"
        rc = await self._exec_in_container(cmd, env=None, timeout_sec=10)
        if rc != 0:
            raise RuntimeError(
                f"Failed to write {container_path} in {self._container_name} (exit {rc})"
            )

    @staticmethod
    def _unique_heredoc_sentinel(content: str) -> str:
        """Return a sentinel guaranteed not to appear as a line in ``content``."""
        base = "AWSBENCH_EOF"
        candidate = base
        while f"\n{candidate}\n" in f"\n{content}\n":
            candidate = f"{base}_{secrets.token_hex(4)}"
        return candidate

    # ── phase execution ─────────────────────────────────────────────────

    async def run_phase(
        self,
        phase: ScenarioPhase,
        *,
        env: dict[str, str],
        timeout_sec: float,
    ) -> ExecResult:
        """Run one phase script inside the running container.

        Steps:
          1. Upload ``<scenario_dir>/<phase>/`` -> ``/<phase>/``.
          2. Create ``/logs/<phase>/`` and chmod the entry script.
             ``/logs/`` is a bind mount of ``host_logs_dir``, so anything
             the script writes there appears on the host immediately.
          3. Run ``/<phase>/<phase>.sh`` with ``env``, redirecting combined
             stdout+stderr to ``/logs/<phase>/stdout.txt``.
          4. Return the ``docker exec`` exit code together with the captured
             stdout read from the host side of the bind mount — no
             docker-cp round-trip.
        """
        self._require_started()
        host_dir = self._paths.phase_dir(phase)
        if not host_dir.is_dir():
            raise FileNotFoundError(f"Phase directory not found: {host_dir}")
        entry = self._paths.phase_script_path(phase)
        if not entry.is_file():
            raise FileNotFoundError(f"Phase script not found: {entry}")

        container_phase_dir = PurePosixPath("/") / phase
        container_logs_dir = self.LOGS_DIR / phase
        container_entry = container_phase_dir / f"{phase}.sh"
        container_stdout = container_logs_dir / "stdout.txt"
        host_phase_dir = self._host_logs_dir / phase

        await self._upload_dir(host_dir, container_phase_dir)
        # Helper commands must succeed; treat nonzero as a hard failure.
        helper_rc = await self._exec_in_container(
            f"mkdir -p {container_logs_dir} && chmod +x {container_entry}",
            env=None,
            timeout_sec=10,
        )
        if helper_rc != 0:
            raise RuntimeError(
                f"Container setup failed (mkdir/chmod) for phase {phase!r}: exit_code={helper_rc}"
            )

        # Combine stdout + stderr into a single file via shell redirection.
        # The redirect changes only where output goes, not the script's
        # return code, so docker-exec reports the script's real exit status.
        cmd = f"{container_entry} > {container_stdout} 2>&1"
        exit_code = await self._exec_in_container(cmd, env=env, timeout_sec=timeout_sec)

        # Read the captured stdout from the host bind mount.
        stdout_path = host_phase_dir / "stdout.txt"
        stdout = stdout_path.read_text(errors="replace") if stdout_path.is_file() else ""
        # The (root) container just wrote /logs; hand ownership back to the
        # host user so the trial's subsequent host-side resource-management
        # writes (e.g. reset/<account_tag>) don't hit EACCES on rootful Docker.
        await self._chown_logs_to_host_user()
        return ExecResult(exit_code=exit_code, stdout=stdout)

    # ── internal: container I/O via docker CLI ──────────────────────────

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("Container has not been started.")

    async def _exec_in_container(
        self,
        command: str,
        *,
        env: dict[str, str] | None,
        timeout_sec: float,
    ) -> int:
        """``docker exec`` a shell command and return its exit code.

        Raises ``asyncio.TimeoutError`` on timeout; the in-container
        process may still be running until the daemon reaps it.
        """
        self._require_started()
        args = ["exec"]
        for k, v in (env or {}).items():
            args.extend(["--env", f"{k}={v}"])
        args.extend([self._container_name, "sh", "-c", command])
        rc, _, _ = await asyncio.wait_for(
            self._run_docker_capture(args, check=False), timeout=timeout_sec
        )
        return rc

    async def _upload_dir(self, host_dir: Path, container_dir: PurePosixPath) -> None:
        """Tar ``host_dir`` (sync) and stream it into the container via ``docker cp``.

        Symlinks anywhere in the tree are rejected outright. They are not
        worth supporting: cross-phase sharing can be done by copying files
        into the build context, and rejecting them removes a class of
        bugs (escape paths, cycles, dangling targets in the container).
        """
        # Make sure the parent exists; `docker cp` requires it.
        await self._exec_in_container(f"mkdir -p {container_dir}", env=None, timeout_sec=10)
        tar_bytes = await asyncio.to_thread(self._tar_dir_bytes, host_dir)
        await self._cp_to_container(tar_bytes, container_dir)

    def _tar_dir_bytes(self, host_dir: Path) -> bytes:
        with SpooledTemporaryFile(max_size=64 << 20) as buf:
            with tarfile.open(fileobj=buf, mode="w") as tar:  # type: ignore[arg-type]
                for child in host_dir.iterdir():
                    self._reject_any_symlink(child)
                    tar.add(child, arcname=child.name, recursive=True)
            buf.seek(0)
            return buf.read()

    @staticmethod
    def _reject_any_symlink(path: Path) -> None:
        """Raise if ``path`` is or contains a symlink at any depth."""
        if path.is_symlink():
            raise ValueError(
                f"Refusing to upload {path}: symlinks are not allowed "
                f"inside scenario phase directories."
            )
        if path.is_dir():
            for child in path.iterdir():
                ScenarioContainer._reject_any_symlink(child)

    async def _cp_to_container(self, tar_bytes: bytes, container_dir: PurePosixPath) -> None:
        """``docker cp - <name>:<dst>`` — stream tar bytes into the container."""
        target = f"{self._container_name}:{container_dir}"
        await self._run_docker(["cp", "-", target], stdin=tar_bytes)

    # ── internal: docker CLI invocation ─────────────────────────────────

    async def _run_docker(self, args: Iterable[str], *, stdin: bytes | None = None) -> None:
        """Run ``docker <args>``; raise ``DockerCLIError`` on nonzero exit.

        Captured stdout is discarded; use :meth:`_run_docker_capture` when
        you need the bytes back.
        """
        rc, _, stderr = await self._run_docker_capture(args, stdin=stdin, check=False)
        if rc != 0:
            cmd = self._build_command(args)
            raise DockerCLIError(cmd, rc, stderr)

    async def _run_docker_capture(
        self,
        args: Iterable[str],
        *,
        stdin: bytes | None = None,
        check: bool = True,
    ) -> tuple[int, bytes, str]:
        """Run ``docker <args>`` and return (returncode, stdout, stderr)."""
        cmd = self._build_command(args)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate(input=stdin)
        rc = proc.returncode if proc.returncode is not None else -1
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        if check and rc != 0:
            raise DockerCLIError(cmd, rc, stderr_text)
        return rc, stdout_bytes, stderr_text

    def _build_command(self, args: Iterable[str]) -> list[str]:
        return [self.DOCKER_BIN, *args]


def sanitize_image_tag(name: str) -> str:
    """Produce a Docker-image-name-safe slug from any string.

    Lowercases, replaces invalid chars with ``-``, ensures the first char
    is alphanumeric.
    """
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    return re.sub(r"[^a-z0-9._-]", "-", name)


def sanitize_container_name(name: str) -> str:
    """Produce a Docker-container-name-safe slug.

    Container names must be ``[a-zA-Z0-9][a-zA-Z0-9_.-]+``; we lowercase
    for consistency with the image tag.
    """
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    return re.sub(r"[^a-z0-9_.-]", "-", name)


def docker_cli_available() -> bool:
    """True when the ``docker`` binary is on ``$PATH``.

    Provided for callers that want to fail fast at startup with a clear
    message rather than crashing inside a trial.
    """
    return shutil.which(ScenarioContainer.DOCKER_BIN) is not None
