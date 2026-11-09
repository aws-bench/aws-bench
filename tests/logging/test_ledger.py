"""Tests for the command ledger."""

from __future__ import annotations

import json
import logging
import os
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


def test_secret_env_assignments_never_reach_the_entry(tmp_path, monkeypatch):
    """``--ve``/``--ae`` secret values are masked in ``command.json``'s argv and config.

    Drives the real writer through beats 1-3 with canary secrets in argv and in
    the resolved config, with no job dir registered (so no snapshot is written),
    then byte-scans the files the entry writes. The keys must stay visible.
    ``run.log`` is in the scan but proves nothing: no code path here logs the
    canaries, and ``run.log`` is outside the redaction scope by design (verbatim
    DEBUG). The snapshot is covered by
    ``test_snapshot_excludes_config_and_lock_files_at_every_depth``.
    """
    argv = [
        "aws-bench",
        "run",
        "--ve",
        "AWS_BEARER_TOKEN_BEDROCK=CANARY-LEDGER",
        "--ae",
        "SOME_API_KEY=CANARY-AE",
        "--ve=OTHER_TOKEN=CANARY-INLINE",
        "--ve",
        "FOO=bar",
    ]
    entry = _open(tmp_path, monkeypatch, argv=argv)
    entry.set_resolved(
        job_id="f47ac10b",
        job_dir=None,
        is_resuming=False,
        resolved_config={
            "verifier": {"env": {"AWS_BEARER_TOKEN_BEDROCK": "CANARY-LEDGER"}},
            "agents": [{"env": {"SOME_API_KEY": "CANARY-AE"}}],
        },
    )
    entry.finalize(None)

    files = [p for p in entry.entry_dir.rglob("*") if p.is_file()]
    assert entry.command_json_path in files
    for path in files:
        blob = path.read_bytes()
        for canary in (b"CANARY-LEDGER", b"CANARY-AE", b"CANARY-INLINE"):
            assert canary not in blob, f"{canary!r} leaked into {path.name}"

    data = json.loads(entry.command_json_path.read_text())
    assert data["argv"] == [
        "aws-bench",
        "run",
        "--ve",
        "AWS_BEARER_TOKEN_BEDROCK=***",
        "--ae",
        "SOME_API_KEY=***",
        "--ve=OTHER_TOKEN=***",
        "--ve",
        "FOO=bar",
    ]


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


def _snapshot_members(entry: LedgerEntry) -> dict[str, bytes]:
    """Decompressed content of every regular-file member of the entry's snapshot."""
    with tarfile.open(entry.entry_dir / "job-snapshot.tar.gz", "r:gz") as tf:
        out: dict[str, bytes] = {}
        for member in tf.getmembers():
            if member.isfile():
                fh = tf.extractfile(member)
                assert fh is not None
                out[member.name] = fh.read()
        return out


def _canary_job_dir(tmp_path: Path) -> Path:
    """A Harbor-shaped job dir: config/lock files carry canaries, the rest do not."""
    job_dir = tmp_path / "jobs" / "j-canary"
    trial = job_dir / "trials" / "t1"
    trial.mkdir(parents=True)
    config = {
        "agents": [
            {
                "kwargs": {"api_key": "CANARY-JOB-CONFIG"},
                "mcp_servers": [{"args": ["srv", "api_key=CANARY-JOB-MCP"]}],
            }
        ],
        "verifier": {"kwargs": {"api_token": "CANARY-JOB-VK"}},
    }
    (job_dir / "config.json").write_text(json.dumps(config))
    lock = {"invocation": ["harbor", "--ak", "api_key=CANARY-LOCK"], "trials": []}
    (job_dir / "lock.json").write_text(json.dumps(lock))
    (trial / "config.json").write_text(json.dumps({"agent": {"kwargs": {"k": "CANARY-TRIAL"}}}))
    (trial / "result.json").write_text(json.dumps({"trial_name": "t1", "reward": 1.0}))
    (trial / "trial.log").write_text("plain log line\n")
    return job_dir


_CANARY_EXCLUDED = [
    "j-canary/config.json",
    "j-canary/lock.json",
    "j-canary/trials/t1/config.json",
]


def test_snapshot_excludes_config_and_lock_files_at_every_depth(tmp_path, monkeypatch):
    """Harbor's config-serializing files are left out of the snapshot entirely.

    Opens the tarball and checks each decompressed member, since gzip would hide
    a canary from a byte scan of the archive itself.
    """
    job_dir = _canary_job_dir(tmp_path)
    on_disk = {p: p.read_bytes() for p in job_dir.rglob("*") if p.is_file()}
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(job_dir)
    entry.finalize(None)

    members = _snapshot_members(entry)
    assert set(members) == {"j-canary/trials/t1/result.json", "j-canary/trials/t1/trial.log"}
    for name, blob in members.items():
        assert b"CANARY" not in blob, f"canary leaked into {name}"
    # Kept members are archived byte-for-byte.
    assert (
        members["j-canary/trials/t1/result.json"]
        == (job_dir / "trials" / "t1" / "result.json").read_bytes()
    )
    with tarfile.open(entry.entry_dir / "job-snapshot.tar.gz", "r:gz") as tf:
        names = tf.getnames()
    assert not [n for n in names if n.rsplit("/", 1)[-1] in {"config.json", "lock.json"}]
    # The harbor-owned job dir on disk is left untouched.
    assert {p: p.read_bytes() for p in job_dir.rglob("*") if p.is_file()} == on_disk
    data = json.loads(entry.command_json_path.read_text())
    assert data["snapshot_status"] == "ok"
    assert sorted(data["snapshot_excluded"]) == _CANARY_EXCLUDED


@pytest.mark.parametrize("name", ["Config.JSON", "CONFIG.json", "Lock.Json"])
def test_snapshot_exclusion_is_case_insensitive(tmp_path, monkeypatch, name):
    job_dir = tmp_path / "jobs" / "j1"
    (job_dir / "sub").mkdir(parents=True)
    (job_dir / "sub" / name).write_text('{"api_key": "CANARY-CASE"}')
    (job_dir / "keep.txt").write_text("ok")
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(job_dir)
    entry.finalize(None)

    assert set(_snapshot_members(entry)) == {"j1/keep.txt"}
    data = json.loads(entry.command_json_path.read_text())
    assert data["snapshot_excluded"] == [f"j1/sub/{name}"]


def test_snapshot_does_not_exclude_lookalike_names(tmp_path, monkeypatch):
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    for name in ("my-config.json", "config.json.bak", "lock.jsonl", "result.json"):
        (job_dir / name).write_text("{}")
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(job_dir)
    entry.finalize(None)

    assert set(_snapshot_members(entry)) == {
        "j1/my-config.json",
        "j1/config.json.bak",
        "j1/lock.jsonl",
        "j1/result.json",
    }
    data = json.loads(entry.command_json_path.read_text())
    assert data["snapshot_excluded"] == []


def _hardlinked_config_job_dir(tmp_path: Path, alias: str) -> Path:
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    (job_dir / "config.json").write_text('{"api_key": "CANARY-HARDLINK"}')
    os.link(job_dir / "config.json", job_dir / alias)
    (job_dir / "result.json").write_text('{"ok": true}')
    return job_dir


@pytest.mark.skipif(not hasattr(os, "link"), reason="hard links")
@pytest.mark.parametrize("alias", ["aaa.json", "z.txt"], ids=["visited-first", "visited-after"])
def test_snapshot_excludes_hard_link_aliases_of_config(tmp_path, monkeypatch, alias):
    """A second name for an excluded file is excluded too, in either visit order.

    ``aaa.json`` sorts before ``config.json`` (tarfile would archive it as a
    regular file with the config bytes); ``z.txt`` sorts after (tarfile would
    archive a hard-link member pointing at the absent ``config.json``).
    """
    job_dir = _hardlinked_config_job_dir(tmp_path, alias)
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(job_dir)
    entry.finalize(None)
    snap = entry.entry_dir / "job-snapshot.tar.gz"
    with tarfile.open(snap, "r:gz") as tf:
        names = tf.getnames()
        assert not [m.name for m in tf.getmembers() if m.islnk()]
        # Every member is extractable: no dangling hard link is left behind.
        dest = tmp_path / "extract"
        tf.extractall(dest, filter="data")
    assert sorted(names) == ["j1", "j1/result.json"]
    for blob in _snapshot_members(entry).values():
        assert b"CANARY" not in blob
    assert (dest / "j1" / "result.json").read_text() == '{"ok": true}'
    data = json.loads(entry.command_json_path.read_text())
    assert data["snapshot_status"] == "ok"
    assert sorted(data["snapshot_excluded"]) == sorted(["j1/config.json", f"j1/{alias}"])


@pytest.mark.skipif(not hasattr(os, "link"), reason="hard links")
def test_snapshot_excludes_hard_link_to_alias_of_config(tmp_path, monkeypatch):
    """A hard-link member pointing at an excluded alias (not a config name) is dropped."""
    job_dir = _hardlinked_config_job_dir(tmp_path, "aaa.json")
    os.link(job_dir / "config.json", job_dir / "zzz.log")
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(job_dir)
    entry.finalize(None)
    with tarfile.open(entry.entry_dir / "job-snapshot.tar.gz", "r:gz") as tf:
        assert sorted(tf.getnames()) == ["j1", "j1/result.json"]
    data = json.loads(entry.command_json_path.read_text())
    assert sorted(data["snapshot_excluded"]) == ["j1/aaa.json", "j1/config.json", "j1/zzz.log"]


def test_excluded_inodes_reads_metadata_only(tmp_path, monkeypatch):
    from aws_bench.logging import ledger as ledger_mod

    job_dir = tmp_path / "j1"
    (job_dir / "sub").mkdir(parents=True)
    (job_dir / "config.json").write_text("{}")
    (job_dir / "sub" / "LOCK.json").write_text("{}")
    (job_dir / "sub" / "config.json.bak").write_text("{}")
    (job_dir / "lock.json").mkdir()  # a directory: not collected
    expected = {
        (st.st_dev, st.st_ino)
        for st in (os.lstat(job_dir / "config.json"), os.lstat(job_dir / "sub" / "LOCK.json"))
    }

    def no_open(*args, **kwargs):
        raise AssertionError("file contents must never be read")

    monkeypatch.setattr("builtins.open", no_open)
    assert ledger_mod._excluded_inodes(job_dir) == expected


def test_excluded_inodes_propagates_scan_errors(tmp_path, monkeypatch):
    """An lstat or listing error is never skipped: the caller fails the snapshot closed."""
    from aws_bench.logging import ledger as ledger_mod

    job_dir = tmp_path / "j1"
    (job_dir / "sub").mkdir(parents=True)
    (job_dir / "config.json").write_text("{}")

    with pytest.raises(OSError):
        ledger_mod._excluded_inodes(tmp_path / "missing")

    real_lstat, real_scandir = os.lstat, os.scandir

    def lstat_fails_for_config(path, *args, **kwargs):
        if str(path).endswith("config.json"):
            raise FileNotFoundError(path)
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", lstat_fails_for_config)
    with pytest.raises(FileNotFoundError):
        ledger_mod._excluded_inodes(job_dir)
    monkeypatch.setattr(os, "lstat", real_lstat)

    def scandir_fails_for_sub(path="."):
        if str(path).endswith("sub"):
            raise PermissionError(path)
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir_fails_for_sub)
    with pytest.raises(PermissionError):
        ledger_mod._excluded_inodes(job_dir)


@pytest.mark.skipif(not hasattr(os, "link"), reason="hard links")
def test_snapshot_transient_prescan_lstat_error_fails_closed(tmp_path, monkeypatch):
    """A pre-scan lstat error that tarfile would not hit must not let an alias through."""
    job_dir = _hardlinked_config_job_dir(tmp_path, "aaa.txt")
    real_lstat = os.lstat
    failed = {"n": 0}

    def lstat_fails_once_for_config(path, *args, **kwargs):
        if str(path).endswith("config.json") and not failed["n"]:
            failed["n"] += 1
            raise FileNotFoundError(path)
        return real_lstat(path, *args, **kwargs)

    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(job_dir)
    monkeypatch.setattr(os, "lstat", lstat_fails_once_for_config)
    entry.finalize(None)
    assert failed["n"] == 1
    _assert_finalized_after_snapshot_failure(entry)


def test_snapshot_transient_prescan_listing_error_fails_closed(tmp_path, monkeypatch):
    """A directory the pre-scan cannot list fails the snapshot, even if tarfile could."""
    job_dir = _canary_job_dir(tmp_path)
    real_scandir = os.scandir
    failed = {"n": 0}

    def scandir_fails_once_for_trials(path="."):
        if str(path).endswith("trials") and not failed["n"]:
            failed["n"] += 1
            raise PermissionError(path)
        return real_scandir(path)

    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(job_dir)
    monkeypatch.setattr(os, "scandir", scandir_fails_once_for_trials)
    entry.finalize(None)
    assert failed["n"] == 1
    _assert_finalized_after_snapshot_failure(entry)


@pytest.mark.skipif(not hasattr(os, "link"), reason="hard links")
def test_snapshot_member_vanishing_during_filter_fails_whole_snapshot(tmp_path, monkeypatch):
    job_dir = _hardlinked_config_job_dir(tmp_path, "z.txt")
    real_gettarinfo = tarfile.TarFile.gettarinfo

    def gettarinfo_then_vanish(self, name=None, arcname=None, fileobj=None):
        info = real_gettarinfo(self, name, arcname, fileobj)
        if name is not None and arcname and arcname.endswith("result.json"):
            Path(name).unlink()
        return info

    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(job_dir)
    monkeypatch.setattr(tarfile.TarFile, "gettarinfo", gettarinfo_then_vanish)
    entry.finalize(None)
    _assert_finalized_after_snapshot_failure(entry)


def test_snapshot_archives_plain_copy_of_config_known_gap(tmp_path, monkeypatch):
    """Known gap: a copy (a different inode) under another name is archived as-is.

    A failure here means the gap was closed: update the ledger docstring.
    """
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    (job_dir / "config.json").write_text('{"api_key": "CANARY-COPY"}')
    (job_dir / "config.json.bak").write_text('{"api_key": "CANARY-COPY"}')
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(job_dir)
    entry.finalize(None)
    members = _snapshot_members(entry)
    assert set(members) == {"j1/config.json.bak"}
    assert b"CANARY-COPY" in members["j1/config.json.bak"]
    data = json.loads(entry.command_json_path.read_text())
    assert data["snapshot_excluded"] == ["j1/config.json"]


def test_snapshot_keeps_directories_named_like_config_files(tmp_path, monkeypatch):
    """Only non-directory members are excluded by name; a matching dir keeps its subtree."""
    job_dir = tmp_path / "jobs" / "config.json"
    nested = job_dir / "trials" / "lock.json"
    nested.mkdir(parents=True)
    (job_dir / "result.json").write_text('{"ok": true}')
    (nested / "result.json").write_text('{"ok": true}')
    (nested / "config.json").write_text('{"api_key": "CANARY-DIRNAME"}')
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(job_dir)
    entry.finalize(None)
    with tarfile.open(entry.entry_dir / "job-snapshot.tar.gz", "r:gz") as tf:
        dirs = {m.name for m in tf.getmembers() if m.isdir()}
    assert dirs == {"config.json", "config.json/trials", "config.json/trials/lock.json"}
    assert set(_snapshot_members(entry)) == {
        "config.json/result.json",
        "config.json/trials/lock.json/result.json",
    }
    data = json.loads(entry.command_json_path.read_text())
    assert data["snapshot_status"] == "ok"
    assert data["snapshot_excluded"] == ["config.json/trials/lock.json/config.json"]


def test_snapshot_archives_result_json_verbatim_known_gap(tmp_path, monkeypatch):
    """Known gap: Harbor's ``result.json`` embeds the trial config; it is archived as-is.

    ``TrialResult.config`` is the full ``TrialConfig`` (agent/verifier kwargs, MCP
    servers), and the job-level ``result.json`` nests every ``TrialResult``. A
    failure here means the gap was closed: update the ledger docstring.
    """
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    result = {"config": {"agent": {"kwargs": {"api_key": "CANARY-GAP-RESULT"}}}}
    (job_dir / "result.json").write_text(json.dumps(result))
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(job_dir)
    entry.finalize(None)

    assert b"CANARY-GAP-RESULT" in _snapshot_members(entry)["j1/result.json"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_snapshot_keeps_symlinks_without_following(tmp_path, monkeypatch):
    outside = tmp_path / "outside.txt"
    outside.write_text("CANARY-LINK")
    job_dir = _canary_job_dir(tmp_path)
    (job_dir / "link.txt").symlink_to(outside)
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(job_dir)
    entry.finalize(None)

    with tarfile.open(entry.entry_dir / "job-snapshot.tar.gz", "r:gz") as tf:
        link = tf.getmember("j-canary/link.txt")
        assert link.issym()
        assert link.linkname == str(outside)
    for name, blob in _snapshot_members(entry).items():
        assert b"CANARY" not in blob, f"canary leaked into {name}"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_snapshot_symlinked_root_is_stored_as_link(tmp_path, monkeypatch):
    target = _canary_job_dir(tmp_path)
    (target / "secret.txt").write_text("CANARY-ROOT")
    root = tmp_path / "jobs" / "job-link"
    root.symlink_to(target)
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(root)
    entry.finalize(None)

    with tarfile.open(entry.entry_dir / "job-snapshot.tar.gz", "r:gz") as tf:
        members = tf.getmembers()
    assert [(m.name, m.issym()) for m in members] == [("job-link", True)]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX sockets")
def test_snapshot_skips_unsupported_member_types(tmp_path, monkeypatch):
    # A socket has no tar representation; ``tf.add`` skips it, not fatal.
    import socket

    job_dir = _canary_job_dir(tmp_path)
    monkeypatch.chdir(job_dir)  # bind a short relative path (AF_UNIX path limit)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind("s.sock")
        entry = _open(tmp_path, monkeypatch)
        entry.register_job_dir(job_dir)
        entry.finalize(None)
    finally:
        sock.close()
    with tarfile.open(entry.entry_dir / "job-snapshot.tar.gz", "r:gz") as tf:
        assert "j-canary/s.sock" not in tf.getnames()
    data = json.loads(entry.command_json_path.read_text())
    assert data["snapshot_status"] == "ok"
    assert sorted(data["snapshot_excluded"]) == _CANARY_EXCLUDED


def _assert_finalized_after_snapshot_failure(entry: LedgerEntry) -> None:
    data = json.loads(entry.command_json_path.read_text())
    assert data["snapshot_status"] == "failed"
    # All-or-nothing: no archive, so nothing is reported as excluded from one.
    assert data["snapshot_excluded"] == []
    assert "snapshot_file_left" not in data
    assert data["exit_status"] == "ok"
    assert "finished_at" in data
    # Neither the final archive nor the partial one is left behind.
    assert not (entry.entry_dir / "job-snapshot.tar.gz").exists()
    assert not (entry.entry_dir / "job-snapshot.tar.gz.partial").exists()
    # The run.log handler is detached and closed, and the entry is inactive.
    assert entry._handler is None
    assert not entry._active
    assert not [
        h
        for h in logging.getLogger("aws_bench").handlers
        if isinstance(h, logging.FileHandler) and h.baseFilename.startswith(str(entry.entry_dir))
    ]


def test_snapshot_failure_does_not_raise(tmp_path, monkeypatch):
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(tmp_path / "does" / "not" / "exist")
    entry.finalize(None)  # must not raise
    _assert_finalized_after_snapshot_failure(entry)
    data = json.loads(entry.command_json_path.read_text())
    assert data["snapshot_excluded"] == []


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="POSIX non-root modes")
@pytest.mark.parametrize("unreadable", ["trials/t1/result.json", "trials/t1"])
def test_snapshot_unreadable_member_fails_without_partial_archive(
    tmp_path, monkeypatch, unreadable
):
    job_dir = _canary_job_dir(tmp_path)
    target = job_dir / unreadable
    target.chmod(0)
    try:
        entry = _open(tmp_path, monkeypatch)
        entry.register_job_dir(job_dir)
        entry.finalize(None)
    finally:
        target.chmod(0o700)
    _assert_finalized_after_snapshot_failure(entry)


def test_snapshot_tarfile_error_fails_without_partial_archive(tmp_path, monkeypatch):
    def boom(self, *args, **kwargs):
        raise tarfile.TarError("injected")

    job_dir = _canary_job_dir(tmp_path)
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(job_dir)
    monkeypatch.setattr(tarfile.TarFile, "add", boom)
    entry.finalize(None)
    _assert_finalized_after_snapshot_failure(entry)


def test_snapshot_unexpected_error_fails_without_partial_archive(tmp_path, monkeypatch):
    def boom(self, *args, **kwargs):
        raise RuntimeError("injected")

    job_dir = _canary_job_dir(tmp_path)
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(job_dir)
    monkeypatch.setattr(tarfile.TarFile, "add", boom)
    entry.finalize(None)
    _assert_finalized_after_snapshot_failure(entry)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_snapshot_failure_records_owner_only_file_left_behind(tmp_path, monkeypatch, caplog):
    """A partial archive that cannot be removed is ``0o600`` and named in the record."""

    def boom(self, *args, **kwargs):
        raise tarfile.TarError("injected")

    def unlink_fails(self, missing_ok=False):
        raise OSError("injected")

    old_umask = os.umask(0o022)  # a permissive umask must not widen the partial
    try:
        entry = _open(tmp_path, monkeypatch)
        entry.register_job_dir(_canary_job_dir(tmp_path))
        monkeypatch.setattr(tarfile.TarFile, "add", boom)
        monkeypatch.setattr(Path, "unlink", unlink_fails)
        with caplog.at_level(logging.WARNING, logger="aws_bench"):
            entry.finalize(None)  # must not raise
    finally:
        os.umask(old_umask)
    data = json.loads(entry.command_json_path.read_text())
    assert data["snapshot_status"] == "failed"
    assert data["snapshot_excluded"] == []
    assert data["snapshot_file_left"] == "job-snapshot.tar.gz.partial"
    partial = entry.entry_dir / "job-snapshot.tar.gz.partial"
    assert stat.S_IMODE(partial.stat().st_mode) == 0o600
    assert not (entry.entry_dir / "job-snapshot.tar.gz").exists()
    assert entry._handler is None
    assert not entry._active
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "snapshot failed" in warnings[0].getMessage()
    assert "job-snapshot.tar.gz.partial" in warnings[0].getMessage()
    assert "CANARY" not in warnings[0].getMessage()


def test_snapshot_failure_logs_one_warning(tmp_path, monkeypatch, caplog):
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(_canary_job_dir(tmp_path))
    monkeypatch.setattr(tarfile.TarFile, "add", lambda self, *a, **k: 1 / 0)
    with caplog.at_level(logging.WARNING, logger="aws_bench"):
        entry.finalize(None)
    _assert_finalized_after_snapshot_failure(entry)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert [r.getMessage().count("\n") for r in warnings] == [0]


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_snapshot_interrupt_is_recorded_then_reraised(tmp_path, monkeypatch, interrupt):
    """An interrupt inside ``tf.add`` records the failure, removes the partial, propagates."""
    real_addfile = tarfile.TarFile.addfile
    calls = {"n": 0}

    def addfile_then_interrupt(self, tarinfo, fileobj=None):
        calls["n"] += 1
        if calls["n"] > 2:  # after some bytes have been written
            raise interrupt()
        return real_addfile(self, tarinfo, fileobj)

    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(_canary_job_dir(tmp_path))
    monkeypatch.setattr(tarfile.TarFile, "addfile", addfile_then_interrupt)
    with pytest.raises(interrupt):
        entry.finalize(None)
    assert calls["n"] > 2
    _assert_finalized_after_snapshot_failure(entry)


def test_snapshot_is_written_to_partial_then_renamed(tmp_path, monkeypatch):
    """The final name only appears via ``os.replace`` of the closed partial archive."""
    seen: list[tuple[str, str, int]] = []
    real_replace = os.replace

    def spy_replace(src, dst):
        seen.append((Path(src).name, Path(dst).name, stat.S_IMODE(os.stat(src).st_mode)))
        assert not Path(dst).exists()
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(_canary_job_dir(tmp_path))
    entry.finalize(None)
    assert seen == [("job-snapshot.tar.gz.partial", "job-snapshot.tar.gz", 0o600)]
    assert not (entry.entry_dir / "job-snapshot.tar.gz.partial").exists()
    assert set(_snapshot_members(entry)) == {
        "j-canary/trials/t1/result.json",
        "j-canary/trials/t1/trial.log",
    }


def test_snapshot_rename_failure_leaves_no_archive(tmp_path, monkeypatch):
    def replace_fails(src, dst):
        raise OSError("injected")

    monkeypatch.setattr(os, "replace", replace_fails)
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(_canary_job_dir(tmp_path))
    entry.finalize(None)
    _assert_finalized_after_snapshot_failure(entry)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_snapshot_interrupt_after_rename_names_finished_archive_left(tmp_path, monkeypatch, caplog):
    """If cleanup fails after the rename, the finished archive is named in the record."""
    real_replace = os.replace

    def replace_then_interrupt(src, dst):
        real_replace(src, dst)
        raise KeyboardInterrupt

    def unlink_fails(self, missing_ok=False):
        raise OSError("injected")

    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(_canary_job_dir(tmp_path))
    monkeypatch.setattr(os, "replace", replace_then_interrupt)
    monkeypatch.setattr(Path, "unlink", unlink_fails)
    with caplog.at_level(logging.WARNING, logger="aws_bench"), pytest.raises(KeyboardInterrupt):
        entry.finalize(None)
    data = json.loads(entry.command_json_path.read_text())
    assert data["snapshot_status"] == "failed"
    assert data["snapshot_excluded"] == []
    assert data["snapshot_file_left"] == "job-snapshot.tar.gz"
    final = entry.entry_dir / "job-snapshot.tar.gz"
    assert stat.S_IMODE(final.stat().st_mode) == 0o600
    assert not (entry.entry_dir / "job-snapshot.tar.gz.partial").exists()
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "job-snapshot.tar.gz" in warnings[0]
    assert "none written" not in warnings[0]
    assert not entry._active


def test_snapshot_fdopen_failure_closes_descriptor(tmp_path, monkeypatch):
    opened: list[int] = []
    closed: list[int] = []
    real_open, real_close = os.open, os.close

    def spy_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if str(path).endswith(".partial"):
            opened.append(fd)
        return fd

    def spy_close(fd):
        closed.append(fd)
        return real_close(fd)

    def fdopen_fails(fd, *args, **kwargs):
        raise OSError("injected")

    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(_canary_job_dir(tmp_path))
    monkeypatch.setattr(os, "open", spy_open)
    monkeypatch.setattr(os, "close", spy_close)
    monkeypatch.setattr(os, "fdopen", fdopen_fails)
    entry.finalize(None)
    assert len(opened) == 1
    assert opened[0] in closed
    _assert_finalized_after_snapshot_failure(entry)


def test_snapshot_replaces_stale_partial(tmp_path, monkeypatch):
    """A leftover ``.partial`` is replaced, not a reason to lose the snapshot."""
    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(_canary_job_dir(tmp_path))
    (entry.entry_dir / "job-snapshot.tar.gz.partial").write_text("stale")
    entry.finalize(None)
    data = json.loads(entry.command_json_path.read_text())
    assert data["snapshot_status"] == "ok"
    assert not (entry.entry_dir / "job-snapshot.tar.gz.partial").exists()
    assert set(_snapshot_members(entry)) == {
        "j-canary/trials/t1/result.json",
        "j-canary/trials/t1/trial.log",
    }


def test_finalize_completes_when_snapshot_raises(tmp_path, monkeypatch):
    """Even an exception escaping ``_write_snapshot`` cannot abort ``finalize``."""

    def boom(self):
        raise RuntimeError("injected")

    entry = _open(tmp_path, monkeypatch)
    entry.register_job_dir(_canary_job_dir(tmp_path))
    monkeypatch.setattr(LedgerEntry, "_write_snapshot", boom)
    entry.finalize(ValueError("cmd failed"))  # must not raise

    data = json.loads(entry.command_json_path.read_text())
    assert data["exit_status"] == "error"
    assert data["error_type"] == "ValueError"
    assert data["snapshot_status"] == "failed"
    assert data["snapshot_excluded"] == []
    assert entry._handler is None
    assert not entry._active


def _assert_torn_down(entry: LedgerEntry, handler: logging.Handler) -> None:
    assert handler not in logging.getLogger("aws_bench").handlers
    assert entry._handler is None
    assert not entry._active
    assert entry.run_log_path is None


def test_finalize_teardown_survives_non_oserror_from_flush(tmp_path, monkeypatch):
    def flush_fails(self):
        raise RuntimeError("injected")

    entry = _open(tmp_path, monkeypatch)
    handler = entry._handler
    assert handler is not None
    monkeypatch.setattr(LedgerEntry, "_flush", flush_fails)
    entry.finalize(None)  # must not raise
    _assert_torn_down(entry, handler)
    entry.finalize(None)  # idempotent: a second call is a no-op


def test_finalize_teardown_interrupt_in_flush_still_detaches_and_closes(tmp_path, monkeypatch):
    def flush_interrupted(self):
        raise KeyboardInterrupt

    entry = _open(tmp_path, monkeypatch)
    handler = entry._handler
    assert isinstance(handler, logging.FileHandler)
    monkeypatch.setattr(LedgerEntry, "_flush", flush_interrupted)
    with pytest.raises(KeyboardInterrupt):
        entry.finalize(None)
    _assert_torn_down(entry, handler)
    assert handler.stream is None


def test_finalize_teardown_survives_handler_close_error(tmp_path, monkeypatch):
    entry = _open(tmp_path, monkeypatch)
    handler = entry._handler
    assert isinstance(handler, logging.FileHandler)

    def close_fails():
        raise RuntimeError("injected")

    monkeypatch.setattr(handler, "close", close_fails)
    try:
        entry.finalize(None)  # must not raise
        _assert_torn_down(entry, handler)
        before = entry.command_json_path.read_bytes()
        assert json.loads(before)["exit_status"] == "ok"
        entry.finalize(ValueError("second"))  # idempotent: record unchanged
        assert entry.command_json_path.read_bytes() == before
    finally:
        logging.FileHandler.close(handler)


def test_finalize_teardown_survives_remove_handler_error(tmp_path, monkeypatch):
    entry = _open(tmp_path, monkeypatch)
    handler = entry._handler
    assert isinstance(handler, logging.FileHandler)
    root = logging.getLogger("aws_bench")
    real_remove = logging.Logger.removeHandler

    def remove_fails(self, hdlr):
        raise RuntimeError("injected")

    monkeypatch.setattr(logging.Logger, "removeHandler", remove_fails)
    try:
        entry.finalize(None)  # must not raise
        assert entry._handler is None
        assert not entry._active
        assert handler.stream is None  # still closed despite the failed removal
    finally:
        monkeypatch.setattr(logging.Logger, "removeHandler", real_remove)
        root.removeHandler(handler)


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
