"""Tests for the command ledger."""

from __future__ import annotations

import json
import logging
import stat
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

import aws_bench.logging.logger as _logger_mod
from aws_bench.logging.ledger import LedgerEntry, current_ledger, open_entry


def test_logs_dir_is_under_output_dir():
    from aws_bench.constants import LOGS_DIR, OUTPUT_DIR

    assert LOGS_DIR == OUTPUT_DIR / "logs"
    assert LOGS_DIR.name == "logs"


@pytest.fixture(autouse=True)
def _reset_console():
    root = logging.getLogger("aws_bench")
    saved = root.handlers.copy()
    saved_level = root.level
    root.handlers.clear()
    _logger_mod._configured = False
    yield
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            h.close()
    root.handlers.clear()
    root.handlers.extend(saved)
    root.setLevel(saved_level)
    _logger_mod._configured = False


_NOW = datetime(2026, 6, 23, 10, 15, 2, tzinfo=timezone.utc)


def _open(tmp_path, monkeypatch, command="run", argv=None):
    monkeypatch.setattr("aws_bench.logging.ledger.LOGS_DIR", tmp_path / "logs")
    return open_entry(command, argv or ["aws-bench", command], now=_NOW)


def test_open_entry_creates_named_dir(tmp_path, monkeypatch):
    entry = _open(tmp_path, monkeypatch)
    assert entry.entry_dir.parent == tmp_path / "logs"
    assert entry.entry_dir.name.startswith("2026-06-23__10-15-02__run__")
    assert entry.entry_dir.is_dir()


def test_entry_dir_name_timestamp_is_fixed_prefix(tmp_path, monkeypatch):
    entry = _open(tmp_path, monkeypatch)
    # Retention parses age from the fixed 20-char prefix, not a __ split.
    prefix = entry.entry_dir.name[:19]
    assert datetime.strptime(prefix, "%Y-%m-%d__%H-%M-%S")


def test_beat1_command_json_has_core_fields(tmp_path, monkeypatch):
    entry = _open(tmp_path, monkeypatch, argv=["aws-bench", "run", "-d", "awsbench@1.0.0"])
    data = json.loads(entry.command_json_path.read_text())
    assert data["schema_version"] == 1
    assert data["command"] == "run"
    assert data["invocation_id"] == entry.invocation_id
    assert data["started_at"] == "2026-06-23T10:15:02+00:00"
    assert data["argv"] == ["aws-bench", "run", "-d", "awsbench@1.0.0"]
    assert "aws_bench_version" in data["context"]
    assert "python" in data["context"]
    assert "hostname" in data["context"]
    assert isinstance(data["env"], dict)


def test_run_log_captures_aws_bench_debug(tmp_path, monkeypatch):
    entry = _open(tmp_path, monkeypatch)
    logging.getLogger("aws_bench.demo").debug("forensic-line")
    entry.finalize(None)
    assert "forensic-line" in (entry.entry_dir / "run.log").read_text()


def test_set_resolved_patches_and_redacts(tmp_path, monkeypatch):
    entry = _open(tmp_path, monkeypatch)
    entry.set_resolved(
        job_id="f47ac10b",
        job_dir=tmp_path / "jobs" / "j1",
        is_resuming=False,
        resolved_config={
            "registry_url": "https://github.com/org/repo",
            "api_token": "s3cr3t",
            "n": 4,
        },
    )
    data = json.loads(entry.command_json_path.read_text())
    assert data["job_id"] == "f47ac10b"
    # Plain URL field passes through; secret-named field is redacted by name.
    assert data["resolved_config"]["registry_url"] == "https://github.com/org/repo"
    assert data["resolved_config"]["api_token"] == "***"
    assert data["resolved_config"]["n"] == 4


def test_finalize_ok_status(tmp_path, monkeypatch):
    entry = _open(tmp_path, monkeypatch)
    entry.finalize(None)
    data = json.loads(entry.command_json_path.read_text())
    assert data["exit_status"] == "ok"
    assert data["error_type"] is None
    assert "finished_at" in data


def test_finalize_error_status(tmp_path, monkeypatch):
    entry = _open(tmp_path, monkeypatch)
    entry.finalize(ValueError("boom"))
    data = json.loads(entry.command_json_path.read_text())
    assert data["exit_status"] == "error"
    assert data["error_type"] == "ValueError"


def test_record_failures_written_to_command_json(tmp_path, monkeypatch):
    entry = _open(tmp_path, monkeypatch)
    failures: list[dict[str, str | None]] = [
        {"scenario": "s1", "unit": "t1", "error_type": "CleanupFailedError", "message": "orphans"}
    ]
    entry.record_failures(failures)
    data = json.loads(entry.command_json_path.read_text())
    assert data["failures"] == failures


def test_finalize_interrupted_status(tmp_path, monkeypatch):
    entry = _open(tmp_path, monkeypatch)
    entry.finalize(KeyboardInterrupt())
    data = json.loads(entry.command_json_path.read_text())
    assert data["exit_status"] == "interrupted"


def test_snapshot_tars_registered_job_dir(tmp_path, monkeypatch):
    entry = _open(tmp_path, monkeypatch)
    job_dir = tmp_path / "jobs" / "2026-06-23__run"
    job_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text('{"ok": true}')
    entry.register_job_dir(job_dir)
    entry.finalize(None)
    snap = entry.entry_dir / "job-snapshot.tar.gz"
    assert snap.is_file()
    with tarfile.open(snap, "r:gz") as tf:
        names = tf.getnames()
    assert any(n.endswith("result.json") for n in names)
    data = json.loads(entry.command_json_path.read_text())
    assert data["snapshot_status"] == "ok"


def test_snapshot_failure_does_not_raise(tmp_path, monkeypatch):
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(tmp_path / "does" / "not" / "exist")
    entry.finalize(None)  # must not raise
    data = json.loads(entry.command_json_path.read_text())
    assert data["snapshot_status"] == "failed"
    assert data["exit_status"] == "ok"


def test_open_entry_never_raises_on_bad_logs_dir(tmp_path, monkeypatch):
    # Point LOGS_DIR at a path that cannot be created (a file, not a dir).
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setattr("aws_bench.logging.ledger.LOGS_DIR", blocker / "logs")
    entry = open_entry("run", ["aws-bench", "run"], now=_NOW)
    # No-op entry: these must all be safe.
    entry.set_resolved(job_id=None, job_dir=None, is_resuming=False, resolved_config=None)
    entry.finalize(None)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_ledger_tree_is_owner_only(tmp_path, monkeypatch):
    """The entry dir and every file in it are created owner-only (0o700/0o600).

    The record holds the resolved config, argv, cwd, and a job-dir snapshot —
    redacted but still sensitive — so other local users must not be able to read
    it. The dir mode is the load-bearing control; the file modes are belt-and-braces.
    """
    entry = _open(tmp_path, monkeypatch)
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text("{}")
    entry.register_job_dir(job_dir)
    entry.finalize(None)

    # Both the logs root and the entry dir are owner-only, so no other local user
    # can even list entry names, let alone read contents.
    assert stat.S_IMODE(entry.entry_dir.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(entry.entry_dir.stat().st_mode) == 0o700
    for name in ("command.json", "run.log", "job-snapshot.tar.gz"):
        path = entry.entry_dir / name
        assert path.is_file(), name
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, name


def test_current_ledger_returns_stashed_entry():
    entry = LedgerEntry(invocation_id="x", entry_dir=Path())
    assert current_ledger({"ledger": entry}) is entry


def test_current_ledger_returns_noop_when_absent():
    """A missing key yields the inert no-op entry, never None."""
    entry = current_ledger({})
    assert isinstance(entry, LedgerEntry)
    # Every recording method must be a safe no-op and finalize must not raise.
    entry.set_resolved(job_id=None, job_dir=None, is_resuming=False, resolved_config=None)
    entry.register_job_dir(Path("/nope"))
    entry.record_failures([])
    entry.finalize(None)
    assert entry.run_log_path is None
