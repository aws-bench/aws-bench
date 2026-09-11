"""Tests for host-local file locking."""

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from filelock import SoftFileLock

from aws_bench.utils.locking import LOCK_SUFFIX, file_lock

_HOLDER = """
import time
from pathlib import Path
from aws_bench.utils.locking import file_lock

with file_lock(Path({guarded!r})):
    Path({flag!r}).write_text("held")
    time.sleep({hold})
"""


def test_lock_is_taken_on_a_sibling_of_the_guarded_path(tmp_path: Path):
    guarded = tmp_path / "state.json"

    with file_lock(guarded):
        assert (tmp_path / f"state.json{LOCK_SUFFIX}").exists()


def test_guarded_file_can_be_replaced_while_the_lock_is_held(tmp_path: Path):
    """Locking a sibling rather than the file itself is what allows os.replace."""
    guarded = tmp_path / "state.json"
    guarded.write_text("old")
    incoming = tmp_path / "incoming.tmp"
    incoming.write_text("new")

    with file_lock(guarded):
        incoming.replace(guarded)

    assert guarded.read_text() == "new"


def test_threads_never_overlap_in_the_critical_section(tmp_path: Path):
    guarded = tmp_path / "state.json"
    events: list[tuple[str, int]] = []

    def worker(index: int) -> None:
        with file_lock(guarded):
            events.append(("enter", index))
            time.sleep(0.05)
            events.append(("exit", index))

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(events) == 8
    for position in range(0, len(events), 2):
        assert events[position][0] == "enter"
        assert events[position + 1] == ("exit", events[position][1])


def test_lock_held_by_another_process_blocks_this_one(tmp_path: Path):
    guarded = tmp_path / "state.json"
    flag = tmp_path / "held.flag"
    hold_seconds = 0.6
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _HOLDER.format(guarded=str(guarded), flag=str(flag), hold=hold_seconds),
        ]
    )
    try:
        deadline = time.monotonic() + 30
        while not flag.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert flag.exists(), "the child never acquired the lock"

        started = time.monotonic()
        with file_lock(guarded):
            waited = time.monotonic() - started
    finally:
        child.wait(timeout=30)

    assert waited > hold_seconds / 3, f"acquired after {waited:.3f}s; the child did not block us"


def test_a_lock_other_processes_could_break_is_refused(tmp_path: Path, monkeypatch):
    """SoftFileLock grants the lock but is breakable, so it is not mutual exclusion."""
    monkeypatch.setattr("aws_bench.utils.locking.FileLock", SoftFileLock)

    with pytest.raises(OSError, match="no kernel file locking"):
        with file_lock(tmp_path / "state.json"):
            pass
