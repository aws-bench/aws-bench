"""Advisory file locking for host-local state files."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_DIR_MODE = 0o700

# Appended to the guarded path to name its lock.
LOCK_SUFFIX = ".lock"


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock covering ``path`` for the block's duration.

    The lock is taken on a sibling ``<path>.lock`` rather than ``path`` itself, so
    the guarded file can be replaced by ``os.replace`` while the lock is held.
    Blocks until the lock is available.

    ``fcntl.flock`` is per-open-file-description, so this serializes concurrent
    threads in one process and concurrent processes on one host. It carries no
    meaning across hosts sharing the path over a network filesystem.
    """
    lock_path = path.with_name(path.name + LOCK_SUFFIX)
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
