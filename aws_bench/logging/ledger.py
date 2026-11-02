"""Per-invocation command ledger.

One entry directory per ``aws-bench`` CLI call under ``LOGS_DIR``, holding:

- ``command.json`` — the invocation record, written in three beats (open /
  resolved / finalize) and mutated in place.
- ``run.log`` — the full ``aws_bench``-tree DEBUG capture for the invocation.
- ``job-snapshot.tar.gz`` — a frozen copy of the job directory, for commands
  that produce one.

Nothing here may fail the user's command: ``open_entry`` returns a no-op entry
on any setup failure, and every method swallows its own errors.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import shortuuid

from aws_bench import __version__
from aws_bench.constants import LOGS_DIR
from aws_bench.logging.logger import (
    FILE_FORMAT,
    NAMESPACE,
    TRACE,
    DefaultPrefixFilter,
    ShortNameFormatter,
    get_logger,
)
from aws_bench.logging.redaction import record_env, redact_argv, redact_config

logger = get_logger(__name__)

_SCHEMA_VERSION = 1
_ID_LEN = 7
_TS_FORMAT = "%Y-%m-%d__%H-%M-%S"

# Owner-only modes for the ledger tree. The record holds the resolved config,
# argv, cwd, account ids, and the job-dir snapshot — redacted but still
# sensitive — so it is kept off other local users' view. The 0o700 dir mode is
# the load-bearing control (it blocks traversal regardless of file modes); the
# 0o600 file mode is defense in depth, matching `env creds`' dotenv output.
_DIR_MODE = 0o700
_FILE_MODE = 0o600


def _chmod_best_effort(path: Path, mode: int) -> None:
    """Tighten ``path`` to ``mode``; ignore platforms/filesystems that can't."""
    try:
        os.chmod(path, mode)
    except OSError:
        logger.debug("ledger: chmod %o on %s skipped", mode, path, exc_info=True)


def _git_sha() -> str | None:
    """Best-effort short commit SHA; ``None`` outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
        logger.debug("git_sha lookup skipped: not a repo")
    except (OSError, subprocess.SubprocessError):
        logger.debug("git_sha lookup skipped: git unavailable")
    return None


def _docker_version() -> str | None:
    """Best-effort docker client version; ``None`` if docker is absent."""
    try:
        out = subprocess.run(
            ["docker", "version", "--format", "{{.Client.Version}}"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        logger.debug("docker version lookup skipped: docker unavailable")
    return None


def _context() -> dict[str, str | None]:
    return {
        "aws_bench_version": __version__,
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "docker": _docker_version(),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
    }


@dataclass
class LedgerEntry:
    """A single ledger entry; see module docstring for the on-disk shape."""

    invocation_id: str
    entry_dir: Path
    _record: dict = field(default_factory=dict)
    _handler: logging.Handler | None = None
    _job_dir: Path | None = None
    _active: bool = True

    @property
    def command_json_path(self) -> Path:
        """Path to this entry's ``command.json`` record."""
        return self.entry_dir / "command.json"

    @property
    def run_log_path(self) -> Path | None:
        """This entry's ``run.log`` path, or ``None`` for a disabled no-op entry.

        ``None`` (not the no-op entry's meaningless ``./run.log``) lets a caller
        skip teeing extra streams into a ledger that never opened.
        """
        return self.entry_dir / "run.log" if self._active else None

    def _flush(self) -> None:
        if not self._active:
            return
        try:
            self.command_json_path.write_text(json.dumps(self._record, indent=2, default=str))
        except OSError:
            logger.debug("ledger: failed to write command.json", exc_info=True)

    def set_resolved(
        self,
        *,
        job_id: str | None,
        job_dir: Path | None,
        is_resuming: bool,
        resolved_config: dict | None,
    ) -> None:
        """Beat 2: patch resolved job identity + config into the record."""
        if not self._active:
            return
        self._record["job_id"] = job_id
        self._record["job_dir"] = str(job_dir) if job_dir is not None else None
        self._record["is_resuming"] = is_resuming
        self._record["resolved_config"] = redact_config(resolved_config)
        self._flush()

    def record_failures(self, failures: list[dict[str, str | None]]) -> None:
        """Record per-unit failures (scenario trials, provisioned accounts) of the run."""
        if not self._active:
            return
        self._record["failures"] = failures
        self._flush()

    def register_job_dir(self, job_dir: Path) -> None:
        """Mark ``job_dir`` to be tarred into the entry at finalize."""
        self._job_dir = job_dir

    def _write_snapshot(self) -> None:
        if self._job_dir is None:
            return
        try:
            snapshot = self.entry_dir / "job-snapshot.tar.gz"
            with tarfile.open(snapshot, "w:gz") as tf:
                tf.add(self._job_dir, arcname=self._job_dir.name)
            _chmod_best_effort(snapshot, _FILE_MODE)
            self._record["snapshot_status"] = "ok"
        except (OSError, tarfile.TarError):
            logger.debug("ledger: job snapshot failed", exc_info=True)
            self._record["snapshot_status"] = "failed"

    def finalize(self, exc: BaseException | None) -> None:
        """Beat 3: stamp outcome, write snapshot, close the run.log handler."""
        if not self._active:
            return
        if exc is None:
            self._record["exit_status"] = "ok"
            self._record["error_type"] = None
        elif isinstance(exc, KeyboardInterrupt):
            self._record["exit_status"] = "interrupted"
            self._record["error_type"] = type(exc).__name__
        else:
            self._record["exit_status"] = "error"
            self._record["error_type"] = type(exc).__name__
        self._record["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._write_snapshot()
        self._flush()
        if self._handler is not None:
            logging.getLogger(NAMESPACE).removeHandler(self._handler)
            self._handler.close()
            self._handler = None
        self._active = False


def _noop_entry() -> LedgerEntry:
    entry = LedgerEntry(invocation_id="", entry_dir=Path())
    entry._active = False
    return entry


# Key under which the root callback stashes the open entry in ``ctx.meta``.
LEDGER_META_KEY = "ledger"


def current_ledger(meta: dict) -> LedgerEntry:
    """Return the open ledger entry from a Typer ``ctx.meta``, or a no-op entry.

    The single accessor for command bodies recording beat 2. Returning the
    inert no-op entry (never ``None``) when the key is absent means call sites
    record unconditionally — ``current_ledger(ctx.meta).set_resolved(...)`` —
    with no ``is not None`` guard to forget. A miss is the root callback not
    having run (a test driving a command in isolation, or a wiring regression),
    so it is logged at DEBUG rather than passing silently.
    """
    entry = meta.get(LEDGER_META_KEY)
    if entry is None:
        logger.debug("ledger: no entry in ctx.meta; recording into a no-op entry")
        return _noop_entry()
    return entry


def open_entry(command: str, argv: list[str], *, now: datetime) -> LedgerEntry:
    """Beat 1: create the entry dir, write command.json, open run.log.

    Never raises — returns a disabled no-op entry on any setup failure so the
    ledger can never fail the user's command.
    """
    try:
        invocation_id = shortuuid.ShortUUID().random(length=_ID_LEN)
        slug = command.replace(" ", "-")
        dir_name = f"{now.strftime(_TS_FORMAT)}__{slug}__{invocation_id}"
        # Lock LOGS_DIR too, not just the entry dir: a 0o700 parent stops another
        # user listing entry names and makes the files' create→chmod window
        # unreachable. mode= sets the entry dir's bits in the syscall (no window);
        # the chmods still run for exist_ok and a hostile umask.
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        _chmod_best_effort(LOGS_DIR, _DIR_MODE)
        entry_dir = LOGS_DIR / dir_name
        entry_dir.mkdir(mode=_DIR_MODE, exist_ok=True)
        _chmod_best_effort(entry_dir, _DIR_MODE)

        # Build the full record before attaching the handler, so a failure here
        # returns a clean no-op entry with no orphaned handler on the logger.
        record = {
            "invocation_id": invocation_id,
            "schema_version": _SCHEMA_VERSION,
            "started_at": now.isoformat(),
            "command": command,
            "argv": redact_argv(argv),
            "context": _context(),
            "env": record_env(dict(os.environ)),
        }

        run_log = entry_dir / "run.log"
        handler = logging.FileHandler(run_log)
        _chmod_best_effort(run_log, _FILE_MODE)
        # TRACE (below DEBUG): run.log is the full-fidelity capture, so it keeps
        # the high-volume TRACE lines that the DEBUG job/trial sinks filter out.
        handler.setLevel(TRACE)
        handler.setFormatter(ShortNameFormatter(FILE_FORMAT))
        handler.addFilter(DefaultPrefixFilter())
        logging.getLogger(NAMESPACE).addHandler(handler)

        entry = LedgerEntry(invocation_id=invocation_id, entry_dir=entry_dir, _record=record)
        entry._handler = handler
        entry._flush()
        _chmod_best_effort(entry.command_json_path, _FILE_MODE)
        return entry
    except Exception:
        logger.debug("ledger: open_entry failed; continuing without a ledger", exc_info=True)
        return _noop_entry()
