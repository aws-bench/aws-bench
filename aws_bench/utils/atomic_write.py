"""Helpers for making a replaced file durable."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def fsync_dir(path: Path) -> None:
    """Flush ``path``'s own directory entry, so a rename into it is durable.

    A no-op on Windows, which cannot open a directory as a file descriptor.
    """
    if sys.platform == "win32":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
