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
import csv
import io
import logging
import os
import posixpath
import re
import secrets
import shlex
import shutil
import tarfile
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from tempfile import SpooledTemporaryFile

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.constants import DEFAULT_REGION
from aws_bench.exceptions import OperationCancelled
from aws_bench.logging.logger import get_logger
from aws_bench.scenario.config import EnvironmentConfig
from aws_bench.scenario.events import ScenarioPhase
from aws_bench.scenario.paths import ScenarioPaths
from aws_bench.utils.credentials_provider import (
    CREDS_DIR,
    CredentialProvider,
    build_aws_config,
    build_session_name,
    check_static_profiles,
    credential_command,
    credential_env,
    credential_refresh_delay,
    mint_credentials,
    refresh_credentials_loop,
)

logger = get_logger(__name__)


async def _await_owned[T](task: asyncio.Task[T]) -> T:
    """Finish an owned operation before propagating its caller's cancellation."""
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
        except (OperationCancelled, Exception):
            if cancelled:
                raise asyncio.CancelledError from None
            raise
    if cancelled:
        raise asyncio.CancelledError
    return result


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
                the host-side refresher.
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
        self._account_mapping = dict(account_mapping)
        self._profile = next(iter(self._account_mapping))
        self._labels = dict(labels or {})
        self._log = (log or logger).getChild(container_name)
        self._started = False
        self._stopping = False
        self._run_attempted = False
        self._startup_task: asyncio.Task[None] | None = None
        self._stop_task: asyncio.Task[None] | None = None
        self._container_home: PurePosixPath | None = None
        self._container_uid = 0
        self._container_gid = 0
        # Cache for rootless-daemon detection (see _is_rootless_docker).
        self._rootless_docker: bool | None = None
        # The private outer directory protects host access. Only its readable
        # inner directory is mounted, so non-root container users can read it.
        self._creds_root: Path | None = None
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
        if self._started or self._startup_task is not None:
            raise RuntimeError("Container already started.")
        if self._stop_task is not None and not self._stop_task.done():
            raise RuntimeError("Container is still stopping.")
        if self._creds_root is not None:
            await self._stop_credential_refresh()
            if self._creds_root is not None:
                raise RuntimeError("Previous scenario credentials could not be removed.")

        self._stopping = False
        self._startup_task = asyncio.create_task(self._start())
        try:
            await asyncio.shield(self._startup_task)
        except BaseException:
            await self.stop(delete=True)
            raise
        finally:
            self._startup_task = None

    async def _start(self) -> None:
        await self._remove_existing()
        await self._resolve_container_user()
        assert self._container_home is not None
        self._host_logs_dir.mkdir(parents=True, exist_ok=True)
        self._creds_root = Path(tempfile.mkdtemp(prefix=f"awsbench-creds-{self._container_name}-"))
        self._creds_root.chmod(0o700)
        self._creds_dir = self._creds_root / "creds"
        self._creds_dir.mkdir()
        self._creds_dir.chmod(0o755)
        expires_at = await self._refresh_credentials()
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
            self._mount_arg("bind", str(self._host_logs_dir.resolve()), str(self.LOGS_DIR)),
            "--mount",
            self._mount_arg(
                "bind", str(self._creds_dir), str(self._container_home / CREDS_DIR), readonly=True
            ),
        ]
        # Operational labels (sorted for a deterministic command) so tooling can
        # match containers by role rather than by name.
        for key, value in sorted(self._labels.items()):
            args.extend(["--label", f"{key}={value}"])
        for m in self._env_config.mounts_json:
            mount_str = self._mount_arg(
                m["type"], m["source"], m["target"], readonly=bool(m.get("read_only"))
            )
            args.extend(["--mount", mount_str])
        for key, value in credential_env(self._profile).items():
            args.extend(["--env", f"{key}={value}"])
        args.extend(["--env", f"HOME={self._container_home}"])
        args.extend([self._image_tag, *self.KEEPALIVE_CMD])
        self._run_attempted = True
        await _await_owned(asyncio.create_task(self._run_docker(args)))
        self._started = True
        await self._validate_credential_mount()
        await self._prepare_aws_config()
        self._check_publication_active()
        credential_refresh_delay(expires_at)
        self._refresh_task = asyncio.create_task(
            refresh_credentials_loop(expires_at, self._refresh_credentials, self._log)
        )
        self._log.debug("Started container %s.", self._container_name)

    @staticmethod
    def _mount_arg(kind: str, source: str, target: str, *, readonly: bool = False) -> str:
        """Quote mount fields as CSV, including paths containing commas."""
        buf = io.StringIO()
        fields = [f"type={kind}", f"source={source}", f"target={target}"]
        if readonly:
            fields.append("readonly")
        csv.writer(buf, lineterminator="").writerow(fields)
        return buf.getvalue()

    async def _resolve_container_user(self) -> None:
        """Probe the built image without credentials, mounts, or network access."""
        targets = [str(self.LOGS_DIR), *(m["target"] for m in self._env_config.mounts_json)]
        for target in targets:
            if not PurePosixPath(target).is_absolute() or any(c in target for c in "\0\r\n"):
                raise ValueError("Mount targets must be absolute container paths.")
        # Resolve existing parents so an image symlink cannot hide a mount overlap.
        probe = """
set -eu
cd "$HOME"
pwd -P
id -u
id -g
for path in "$HOME/.aws" "$HOME/.aws/config" "$HOME/.aws/credentials" "$HOME/.aws/creds"; do
    if [ -L "$path" ]; then
        echo "AWS credential paths must not be symlinks" >&2
        exit 1
    fi
done
for target do
    suffix=
    while [ ! -d "$target" ]; do
        if [ -L "$target" ]; then
            echo "Mount target must not be a file symlink" >&2
            exit 1
        fi
        suffix="/${target##*/}$suffix"
        target="${target%/*}"
        target="${target:-/}"
    done
    printf '%s%s\\n' "$(cd "$target" && pwd -P)" "$suffix"
done
"""
        args = ["run", "--rm", "--network", "none", "--entrypoint", "sh"]
        for key, value in credential_env(self._profile).items():
            args.extend(["--env", f"{key}={value}"])
        args.extend(
            [
                self._image_tag,
                "-c",
                credential_command(probe, self._profile),
                "awsbench-home",
                *(posixpath.normpath(t) for t in targets),
            ]
        )
        _, stdout, _ = await self._run_docker_capture(args, settle_on_cancel=True)
        parts = stdout.decode("utf-8").splitlines()
        if (
            len(parts) != 3 + len(targets)
            or not PurePosixPath(parts[0]).is_absolute()
            or not parts[1].isdigit()
            or not parts[2].isdigit()
        ):
            raise RuntimeError("Could not resolve the scenario image's HOME and user.")
        self._container_home = PurePosixPath(parts[0])
        self._container_uid, self._container_gid = int(parts[1]), int(parts[2])
        reserved = [
            self._container_home / CREDS_DIR,
            self._container_home / CREDS_DIR.parent / "config",
        ]
        for original, resolved in zip(targets, parts[3:], strict=True):
            target = PurePosixPath("/" + posixpath.normpath(resolved).lstrip("/"))
            if any(target.is_relative_to(path) or path.is_relative_to(target) for path in reserved):
                raise ValueError(
                    f"Mount target {original!r} overlaps reserved AWS credential paths."
                )

    async def _validate_credential_mount(self) -> None:
        """Check actual mounts, including aliases supplied inside another author mount."""
        assert self._container_home is not None
        creds = self._container_home / CREDS_DIR
        reserved = (creds, creds.parent / "config")
        rc, body, _ = await self._exec_in_container_capture(
            "cat /proc/self/mountinfo", env=None, timeout_sec=10
        )
        if rc != 0:
            raise RuntimeError("Could not read scenario container mounts.")
        found = False
        for line in body.decode("utf-8").splitlines():
            fields = line.split()
            if len(fields) < 6:
                raise RuntimeError("Invalid scenario container mount information.")
            target = PurePosixPath(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4]))
            if target == creds:
                if found or "ro" not in fields[5].split(","):
                    raise ValueError("Scenario credentials require one read-only directory mount.")
                found = True
            elif target != PurePosixPath("/") and any(
                target.is_relative_to(path) or path.is_relative_to(target) for path in reserved
            ):
                raise ValueError(f"Container mount at {target} overlaps reserved AWS paths.")
        if not found:
            raise ValueError("Scenario credential directory is not mounted.")

    async def _prepare_aws_config(self) -> None:
        """Preserve image config and stream its merged contents through Docker stdin."""
        assert self._container_home is not None
        aws_dir = self._container_home / CREDS_DIR.parent
        quoted_dir = shlex.quote(str(aws_dir))
        # Docker can create a missing mount parent as root. Give that directory
        # to the image user without changing any unrelated files beneath it.
        rc = await self._exec_in_container(
            f"test ! -L {quoted_dir} && mkdir -p {quoted_dir} && "
            f"chown {self._container_uid}:{self._container_gid} {quoted_dir} && "
            f"chmod 700 {quoted_dir}",
            env=None,
            timeout_sec=10,
            user="0",
        )
        if rc != 0:
            raise RuntimeError("Could not prepare the scenario user's AWS directory.")
        existing: dict[str, str] = {}
        for name in ("config", "credentials"):
            path = shlex.quote(str(aws_dir / name))
            rc, body, _ = await self._exec_in_container_capture(
                f"test ! -L {path} && "
                f"{{ if [ -e {path} ]; then test -f {path} && cat {path}; fi; }}",
                env=None,
                timeout_sec=10,
            )
            if rc != 0:
                raise RuntimeError(f"Could not read the scenario user's AWS {name} file.")
            existing[name] = body.decode("utf-8")
        check_static_profiles(self._account_mapping, existing["credentials"])
        config = build_aws_config(
            self._account_mapping, existing=existing["config"], region=DEFAULT_REGION
        )
        if config != existing["config"]:
            filename = f".config-{secrets.token_hex(8)}"
            tmp_path = shlex.quote(str(aws_dir / filename))
            body = config.encode("utf-8")
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                entry = tarfile.TarInfo(filename)
                entry.size = len(body)
                entry.mode = 0o600
                tar.addfile(entry, io.BytesIO(body))
            try:
                await self._cp_to_container(buf.getvalue(), aws_dir)
                self._check_publication_active()
                rc = await self._exec_in_container(
                    f"chown {self._container_uid}:{self._container_gid} {tmp_path} && "
                    f"mv -f {tmp_path} {shlex.quote(str(aws_dir / 'config'))}",
                    env=None,
                    timeout_sec=10,
                    user="0",
                )
                if rc != 0:
                    raise RuntimeError("Could not publish the scenario user's AWS config.")
            finally:
                try:
                    rc = await self._exec_in_container(
                        f"rm -f {tmp_path}", env=None, timeout_sec=10
                    )
                    if rc != 0:
                        self._log.warning("Could not remove temporary AWS config %s", tmp_path)
                except Exception as exc:
                    self._log.warning("Could not remove temporary AWS config: %s", exc)
        paths = [aws_dir / "config", self._container_home / CREDS_DIR / f"{self._profile}.json"]
        rc = await self._exec_in_container(
            " && ".join(
                f"test -f {shlex.quote(str(path))} && test -r {shlex.quote(str(path))}"
                for path in paths
            ),
            env=None,
            timeout_sec=10,
        )
        if rc != 0:
            raise RuntimeError("Scenario credentials are not readable by the image user.")

    def _check_publication_active(self) -> None:
        task = asyncio.current_task()
        if self._stopping or self._creds_dir is None or (task is not None and task.cancelling()):
            raise asyncio.CancelledError

    async def _refresh_credentials(self) -> datetime:
        """Mint in a worker; publish atomic replacements only on the owning event loop."""
        self._check_publication_active()
        files, expires_at = await asyncio.to_thread(
            mint_credentials,
            self._cred_provider,
            dict(self._account_mapping),
            ORG_ACCESS_ROLE,
            build_session_name("session", self._profile[-8:]),
        )
        self._check_publication_active()
        assert self._creds_dir is not None
        credential_refresh_delay(expires_at)
        if set(files) != {f"{self._profile}.json"}:
            raise ValueError("Credential mint returned unexpected scenario filenames.")
        for filename, body in files.items():
            path = self._creds_dir / filename
            tmp = path.with_suffix(".json.tmp")
            try:
                tmp.write_text(body, encoding="utf-8")
                tmp.chmod(0o644)
                self._check_publication_active()
                credential_refresh_delay(expires_at)
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        credential_refresh_delay(expires_at)
        return expires_at

    async def _remove_existing(self) -> None:
        """Remove a stale container with the same name, if any."""
        rc, _, stderr = await self._run_docker_capture(
            ["rm", "-f", self._container_name], check=False, settle_on_cancel=True
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
        self._stopping = True
        if self._stop_task is None or self._stop_task.done():
            self._stop_task = asyncio.create_task(self._stop(delete=delete))
        await _await_owned(self._stop_task)

    async def _stop(self, *, delete: bool) -> None:
        if self._startup_task is not None:
            if not self._startup_task.done():
                self._startup_task.cancel()
            try:
                await self._startup_task
            except (asyncio.CancelledError, OperationCancelled, Exception):
                pass
        await self._stop_credential_refresh()
        if not self._started and not self._run_attempted:
            return
        # Fix ownership of bind-mounted /logs so the host user can read/write/
        # delete phase outputs after the (root) container is gone. Must run
        # while the container is still up (uses docker exec). See
        # _chown_logs_to_host_user for the cross-platform rationale.
        await self._chown_logs_to_host_user()
        commands = [["stop", "-t", "10", self._container_name]]
        if delete:
            commands.append(["rm", "-f", self._container_name])
        for args in commands:
            try:
                rc, _, stderr = await self._run_docker_capture(args, check=False)
                if rc != 0:
                    self._log.warning(
                        "Failed to %s container %s: %s",
                        args[0],
                        self._container_name,
                        stderr.strip(),
                    )
                elif args[0] == "rm":
                    self._run_attempted = False
            except Exception as exc:
                self._log.warning(
                    "Failed to %s container %s: %s", args[0], self._container_name, exc
                )
        self._started = False

    async def _stop_credential_refresh(self) -> None:
        """Settle refresh and remove the whole private root, retaining failures for retry."""
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except (asyncio.CancelledError, OperationCancelled, Exception):
                pass
            self._refresh_task = None
        if self._creds_root is not None:
            try:
                shutil.rmtree(self._creds_root)
            except FileNotFoundError:
                pass
            except OSError as exc:
                self._log.warning(
                    "Failed to remove scenario credentials at %s: %s", self._creds_root, exc
                )
                return
            self._creds_root = None
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
        try:
            if await self._is_rootless_docker():
                uid, gid = 0, 0
            else:
                uid, gid = os.getuid(), os.getgid()
            rc = await self._exec_in_container(
                f"chown -R {uid}:{gid} {shlex.quote(str(self.LOGS_DIR))}",
                env=None,
                timeout_sec=120,
                user="0",
            )
            if rc != 0:
                self._log.warning("chown -R %s:%s %s exited %s", uid, gid, self.LOGS_DIR, rc)
        except Exception as e:  # noqa: BLE001 — best-effort ownership fixup
            self._log.warning("Failed to chown %s to host user: %s", self.LOGS_DIR, e)

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
            f"mkdir -p {container_logs_dir} && "
            f"chown {self._container_uid}:{self._container_gid} {container_logs_dir} && "
            f"chmod +x {container_entry}",
            env=None,
            timeout_sec=10,
            user="0",
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
        user: str | None = None,
    ) -> int:
        """``docker exec`` a shell command and return its exit code.

        Raises ``asyncio.TimeoutError`` on timeout; the in-container
        process may still be running until the daemon reaps it.
        """
        rc, _, _ = await self._exec_in_container_capture(
            command, env=env, timeout_sec=timeout_sec, user=user
        )
        return rc

    async def _exec_in_container_capture(
        self,
        command: str,
        *,
        env: dict[str, str] | None,
        timeout_sec: float,
        user: str | None = None,
    ) -> tuple[int, bytes, str]:
        self._require_started()
        args = ["exec"]
        if user is not None:
            args.extend(["--user", user])
        overlay = credential_env(self._profile, env)
        if self._container_home is not None:
            overlay["HOME"] = str(self._container_home)
        for k, v in overlay.items():
            args.extend(["--env", f"{k}={v}"])
        args.extend([self._container_name, "sh", "-c", credential_command(command, self._profile)])
        return await asyncio.wait_for(
            self._run_docker_capture(
                args, check=False, settle_on_cancel=self._startup_task is not None
            ),
            timeout=timeout_sec,
        )

    async def _upload_dir(self, host_dir: Path, container_dir: PurePosixPath) -> None:
        """Tar ``host_dir`` (sync) and stream it into the container via ``docker cp``.

        Symlinks anywhere in the tree are rejected outright. They are not
        worth supporting: cross-phase sharing can be done by copying files
        into the build context, and rejecting them removes a class of
        bugs (escape paths, cycles, dangling targets in the container).
        """
        # Make sure the parent exists; `docker cp` requires it.
        rc = await self._exec_in_container(
            f"mkdir -p {container_dir}", env=None, timeout_sec=10, user="0"
        )
        if rc != 0:
            raise RuntimeError(f"Could not create scenario script directory {container_dir}.")
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
        args = ["cp", "-", target]
        await _await_owned(asyncio.create_task(self._run_docker(args, stdin=tar_bytes)))

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
        settle_on_cancel: bool = False,
    ) -> tuple[int, bytes, str]:
        """Run ``docker <args>`` and return (returncode, stdout, stderr)."""
        cmd = self._build_command(args)

        async def communicate() -> tuple[int, bytes, str]:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await proc.communicate(input=stdin)
            rc = proc.returncode if proc.returncode is not None else -1
            return rc, stdout_bytes, stderr_bytes.decode("utf-8", errors="replace")

        # Bootstrap must settle Docker writes before disposing of credentials.
        # Phase commands retain their normal timeout and container-stop behavior.
        if settle_on_cancel:
            rc, stdout_bytes, stderr_text = await _await_owned(asyncio.create_task(communicate()))
        else:
            rc, stdout_bytes, stderr_text = await communicate()
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
