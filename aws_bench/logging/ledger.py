"""Per-invocation command ledger.

One entry directory per ``aws-bench`` CLI call under ``LOGS_DIR``, holding:

- ``command.json`` — the invocation record, written in three beats (open /
  resolved / finalize) and mutated in place.
- ``run.log`` — the full ``aws_bench``-tree DEBUG capture for the invocation.
- ``job-snapshot.tar.gz`` — a forensic copy of the job directory, for
  commands that produce one; not a replayable job directory (replay, resume or
  re-scoring needs the original job dir). It is archived with plain
  ``tarfile.add`` semantics (symlinks stored as links, never followed), except
  that a non-directory member is left out when its basename is
  ``config.json`` or ``lock.json`` (case-insensitive, any depth) or when it is
  a second name for one: a hard link sharing such a file's ``(st_dev,
  st_ino)``, found up front by an ``os.walk``/``os.lstat`` pass that never
  reads file contents, or a hard-link member pointing at an excluded member.
  A directory with either name is archived with its subtree. For ``aws-bench
  run`` / ``job`` those are the files Harbor serializes ``JobConfig`` /
  ``JobLock`` / ``TrialConfig`` into, so they hold agent and verifier kwargs,
  env and MCP server configs (``lock.json`` also records the CLI invocation,
  kwarg values as typed). Scenario job dirs (``aws-bench env``) lose their
  ``config.json`` as well, although that file is aws-bench's own scenario
  config and holds no credentials. ``command.json`` ``snapshot_excluded``
  lists the excluded arcnames; it is present once a job dir is registered.

  The snapshot is all-or-nothing. It is written to an owner-only (``0o600``)
  ``job-snapshot.tar.gz.partial`` (replacing any stale one), flushed and
  ``fsync``-ed (best effort), and renamed onto the final name only after a
  clean close. Any failure (an unreadable or vanished member, or a listing or
  ``lstat`` error in the pre-scan, included) records ``snapshot_status:
  "failed"``, resets ``snapshot_excluded`` to ``[]``, logs one WARNING line
  and removes the partial archive, and the final one if it exists. If that
  cleanup itself fails (e.g. an interrupt right after the rename, then a
  failed removal), ``snapshot_file_left`` names the file left behind, which
  may be the finished archive. ``KeyboardInterrupt`` / ``SystemExit`` are
  recorded the same way, then propagate; one that lands in ``finalize``
  before the snapshot starts may leave no ``snapshot_status`` at all.

  Known gaps: a plain copy of a config file under another name (a different
  inode, e.g. ``config.json.bak``) is archived as-is. A symlink under another
  name pointing at an excluded config is archived as a (dangling) symlink
  member and is not listed in ``snapshot_excluded``; it carries no bytes of
  the config. Every other member is
  archived byte-for-byte too. Harbor's trial ``result.json`` embeds the full
  ``TrialConfig`` (``TrialResult.config``), and the job-level ``result.json``
  can nest every ``TrialResult``, so both can hold agent/verifier kwargs and
  MCP server configs in cleartext. Harbor's own ``templatize_sensitive_env``
  serializer covers env values under keys it deems sensitive: a value equal
  to the host env var is written as ``${KEY}``, a ``${VAR}`` /
  ``${VAR:-default}`` template is kept as written (default included), and any
  other is written partially masked (its first 4 and last 3 characters kept;
  ``****`` if 8 characters or fewer); other env values are kept.
  Trajectories, ``*.jsonl``, logs, ``exception.txt`` and test output are
  archived as-is too. The job directory on disk is never modified: it is
  written by Harbor, outside aws-bench's control, and holds all of the above
  in cleartext.

Nothing here may fail the user's command: ``open_entry`` returns a no-op entry
on any setup failure, every method swallows its own errors, and ``finalize``
always attempts to flush, detach and close the ``run.log`` handler (each step
guarded on its own, interrupts included) and always deactivates the entry, so
a second call is a no-op. The only exceptions it lets out are interrupts
(``KeyboardInterrupt`` / ``SystemExit``) raised while it runs, and only after
that teardown.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import stat
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

# Basenames (matched case-insensitively, at any depth, non-directories only)
# that the snapshot leaves out: the files Harbor serializes JobConfig / JobLock /
# TrialConfig into, i.e. agent and verifier kwargs, env and MCP server configs.
# See the module docstring.
_SNAPSHOT_EXCLUDED_NAMES = frozenset({"config.json", "lock.json"})
_SNAPSHOT_NAME = "job-snapshot.tar.gz"
_SNAPSHOT_PARTIAL_NAME = f"{_SNAPSHOT_NAME}.partial"


def _is_excluded_name(arcname: str) -> bool:
    return arcname.rsplit("/", 1)[-1].lower() in _SNAPSHOT_EXCLUDED_NAMES


def _raise(err: OSError) -> None:
    raise err


def _excluded_inodes(root: Path) -> set[tuple[int, int]]:
    """``(st_dev, st_ino)`` of every excluded-name regular file under ``root``.

    Metadata only (``os.walk`` without following links, ``os.lstat``); file
    contents are never read. Any listing or ``lstat`` error propagates, failing
    the snapshot closed: a skipped file's hard-link aliases could not be matched.
    """
    found: set[tuple[int, int]] = set()
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False, onerror=_raise):
        for name in filenames:
            if name.lower() not in _SNAPSHOT_EXCLUDED_NAMES:
                continue
            st = os.lstat(os.path.join(dirpath, name))
            if stat.S_ISREG(st.st_mode):
                found.add((st.st_dev, st.st_ino))
    return found


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


def _remove_best_effort(path: Path) -> bool:
    """Delete ``path`` if present; ``False`` (logged, not raised) if it remains."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug("ledger: could not remove %s", path, exc_info=True)
        return False
    return True


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
        job_dir = self._job_dir
        excluded: list[str] = []
        excluded_set: set[str] = set()
        self._record["snapshot_excluded"] = excluded
        root_arc = job_dir.name
        final = self.entry_dir / _SNAPSHOT_NAME
        partial = self.entry_dir / _SNAPSHOT_PARTIAL_NAME

        def _inode(info: tarfile.TarInfo) -> tuple[int, int]:
            # Fail closed, like the pre-scan: an lstat error (e.g. a member that
            # vanished since tarfile's own lstat) fails the whole snapshot, since
            # the member's inode cannot be checked. Stricter than tarfile alone,
            # which never reopens a hard-link member.
            rel = info.name[len(root_arc) + 1 :] if info.name != root_arc else ""
            st = os.lstat(job_dir / rel if rel else job_dir)
            return (st.st_dev, st.st_ino)

        def _exclude_configs(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
            # Directories are always kept (returning None would drop the subtree).
            if info.isdir():
                return info
            drop = _is_excluded_name(info.name)
            if not drop and info.islnk():
                # tarfile maps a second name of an already-seen inode to a
                # hard-link member; never keep one pointing at an excluded member.
                drop = info.linkname in excluded_set or _is_excluded_name(info.linkname)
            if not drop and inodes and (info.isreg() or info.islnk()):
                drop = _inode(info) in inodes
            if drop:
                excluded.append(info.name)
                excluded_set.add(info.name)
                return None
            return info

        try:
            inodes = _excluded_inodes(job_dir)
            if os.path.lexists(partial):
                partial.unlink()  # a stale partial is replaced, not fatal
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
            fd = os.open(partial, flags, _FILE_MODE)
            try:
                fh = os.fdopen(fd, "wb")
            except BaseException:
                os.close(fd)
                raise
            with fh:
                if hasattr(os, "fchmod"):
                    os.fchmod(fh.fileno(), _FILE_MODE)  # in case the umask is unusual
                with tarfile.open(fileobj=fh, mode="w:gz") as tf:
                    tf.add(job_dir, arcname=root_arc, filter=_exclude_configs)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    logger.debug("ledger: fsync of the snapshot skipped", exc_info=True)
            os.replace(partial, final)
            self._record["snapshot_status"] = "ok"
        except BaseException as exc:
            # All-or-nothing: no archive, so nothing is reported as excluded.
            self._record["snapshot_status"] = "failed"
            self._record["snapshot_excluded"] = []
            logger.debug("ledger: job snapshot failed", exc_info=True)
            # ``final`` only exists if an interrupt landed right after the rename.
            left = None
            for path in (partial, final):
                if os.path.lexists(path) and not _remove_best_effort(path):
                    left = path.name
                    self._record["snapshot_file_left"] = left
            if left is None:
                logger.warning("ledger: job snapshot failed (%s); none written", type(exc).__name__)
            else:
                logger.warning(
                    "ledger: job snapshot failed (%s); could not remove %s",
                    type(exc).__name__,
                    left,
                )
            if not isinstance(exc, Exception):
                raise

    def finalize(self, exc: BaseException | None) -> None:
        """Beat 3: stamp outcome, write snapshot, close the run.log handler."""
        if not self._active:
            return
        try:
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
            try:
                self._write_snapshot()
            except Exception:
                logger.debug("ledger: job snapshot raised", exc_info=True)
                self._record["snapshot_status"] = "failed"
                self._record["snapshot_excluded"] = []
        finally:
            self._teardown()

    def _teardown(self) -> None:
        """Flush, detach and close; each step guarded, deactivation unconditional.

        An interrupt raised by a step does not skip the later ones; the first
        one is re-raised after all steps have run.
        """
        handler, self._handler = self._handler, None
        interrupt: BaseException | None = None
        try:
            try:
                self._flush()
            except BaseException as exc:
                interrupt = interrupt or (None if isinstance(exc, Exception) else exc)
                logger.debug("ledger: final flush raised", exc_info=True)
            if handler is not None:
                try:
                    logging.getLogger(NAMESPACE).removeHandler(handler)
                except BaseException as exc:
                    # The handler may be the broken part: do not log through it.
                    interrupt = interrupt or (None if isinstance(exc, Exception) else exc)
                try:
                    handler.close()
                except BaseException as exc:
                    interrupt = interrupt or (None if isinstance(exc, Exception) else exc)
                    logger.debug("ledger: closing run.log handler raised", exc_info=True)
        finally:
            self._active = False
        if interrupt is not None:
            raise interrupt


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
