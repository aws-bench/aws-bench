"""File locking for host-local state files."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, SoftFileLock

_DIR_MODE = 0o700
_LOCK_FILE_MODE = 0o600

# Appended to the guarded path to name its lock.
LOCK_SUFFIX = ".lock"


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock covering ``path`` for the block's duration.

    The lock is taken on a sibling ``<path>.lock`` rather than ``path`` itself, so
    the guarded file can be replaced by ``os.replace`` while the lock is held.
    Polls until another holder releases the lock, with no timeout.

    The lock serializes concurrent threads in one process and concurrent processes
    on one host, on both Unix and Windows. It carries no meaning across hosts
    sharing the path over a network filesystem.

    Raises:
        OSError: If the filesystem offers no kernel locking, or the lock file
            cannot be opened.
        RuntimeError: If this thread already holds ``path`` through another
            ``file_lock`` block.
    """
    lock_path = path.with_name(path.name + LOCK_SUFFIX)
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)

    # A fresh instance per call starts the reentrancy counter at zero, so each
    # acquire takes a real kernel lock on its own descriptor.
    lock = FileLock(lock_path, mode=_LOCK_FILE_MODE)
    with lock:
        # ``filelock`` answers a filesystem without ``flock`` by rewriting its own
        # class to ``SoftFileLock``, which grants the lock and warns. That lock is
        # breakable by any process that believes the holder is gone, so refuse it
        # rather than let a caller mistake it for mutual exclusion.
        if isinstance(lock, SoftFileLock):
            raise OSError(
                f"{lock_path} is on a filesystem with no kernel file locking, so "
                "concurrent aws-bench processes cannot be serialized safely. Put "
                "this state on local disk instead."
            )
        yield
