"""Filesystem-backed storage with content-hash optimistic locking."""

import hashlib
import os
import tempfile
from pathlib import Path

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.storage.exceptions import (
    StorageConflictError,
    StorageError,
    StorageNotFoundError,
)
from aws_bench.utils.filelock import LOCK_SUFFIX, file_lock

logger = get_logger(__name__)

DEFAULT_PREFIX = "snapshots/"

# Directories are owner-only: snapshots enumerate every resource in an account.
_DIR_MODE = 0o700
_FILE_MODE = 0o600


def _etag(data: bytes) -> str:
    """Return the content hash used as this backend's ETag."""
    return hashlib.sha256(data).hexdigest()


class LocalStorageBackend:
    """Filesystem storage exposing the same contract as :class:`S3StorageBackend`.

    ETags are content hashes rather than S3's opaque tokens, which gives the same
    compare-and-set behavior on ``save``: a caller holding a stale ETag is
    rejected with :class:`StorageConflictError`.

    Each key's read-modify-write is serialized by a sibling lock file, so
    concurrent workers in one process (or concurrent processes on one host)
    cannot interleave a check against a write. State is host-local: runs spread
    across hosts do not share it.
    """

    def __init__(self, root: Path, prefix: str = DEFAULT_PREFIX):
        """Store objects under ``root``, with every key nested below ``prefix``.

        Args:
            root: Directory holding the store. Created if absent.
            prefix: Key prefix for all operations.
        """
        self._root = root.expanduser()
        self._prefix = prefix
        self._root.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        logger.debug(f"Initialized local storage backend at {self._root}")

    def _path_for(self, key: str) -> Path:
        """Resolve ``key`` to a path, rejecting anything outside the store.

        Raises:
            StorageError: If ``key`` would resolve outside ``root``.
        """
        full_key = f"{self._prefix}{key}"
        candidate = (self._root / full_key).resolve()
        root = self._root.resolve()
        if root != candidate and root not in candidate.parents:
            raise StorageError(f"Key escapes the storage root: {key!r}")
        return candidate

    def _write_atomic(self, path: Path, data: bytes) -> None:
        """Replace ``path`` with ``data`` via a same-directory temp file."""
        path.parent.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
            os.chmod(temporary, _FILE_MODE)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save(self, key: str, data: bytes, expected_etag: str | None) -> str:
        """Save ``data`` at ``key``, guarded by ``expected_etag``.

        Args:
            key: Relative key within prefix
            data: Data to save
            expected_etag: ETag the caller last read (None for an initial write)

        Returns:
            ETag of the data just written

        Raises:
            StorageConflictError: If the stored content no longer matches ``expected_etag``
            StorageNotFoundError: If ``expected_etag`` is given but the key is absent
            StorageError: If the write fails
        """
        path = self._path_for(key)
        logger.debug(f"Saving to {path} (expected_etag={expected_etag})")

        try:
            with file_lock(path):
                if expected_etag is not None:
                    if not path.exists():
                        raise StorageNotFoundError(key=str(path))
                    actual = _etag(path.read_bytes())
                    if actual != expected_etag:
                        raise StorageConflictError(
                            key=str(path), expected=expected_etag, actual=actual
                        )
                self._write_atomic(path, data)
        except OSError as e:
            raise StorageError(f"Failed to save {path}: {e}") from e

        new_etag = _etag(data)
        logger.debug(f"Saved {path} with ETag {new_etag[:8]}...")
        return new_etag

    def load(self, key: str) -> tuple[bytes, str]:
        """Load the data and ETag stored at ``key``.

        Args:
            key: Relative key within prefix

        Returns:
            Tuple of (data, etag)

        Raises:
            StorageNotFoundError: If key doesn't exist
            StorageError: If the read fails
        """
        path = self._path_for(key)
        logger.debug(f"Loading from {path}")

        try:
            with file_lock(path):
                if not path.exists():
                    raise StorageNotFoundError(key=str(path))
                data = path.read_bytes()
        except OSError as e:
            raise StorageError(f"Failed to load {path}: {e}") from e

        etag = _etag(data)
        logger.debug(f"Loaded {path} ({len(data)} bytes, ETag {etag[:8]}...)")
        return (data, etag)

    def exists(self, key: str) -> bool:
        """Return whether ``key`` is present."""
        path = self._path_for(key)
        present = path.is_file()
        logger.debug(f"Key {'exists' if present else 'does not exist'}: {path}")
        return present

    def delete(self, key: str) -> None:
        """Delete ``key`` (idempotent).

        Raises:
            StorageError: If the delete fails
        """
        path = self._path_for(key)
        try:
            with file_lock(path):
                path.unlink(missing_ok=True)
        except OSError as e:
            raise StorageError(f"Failed to delete {path}: {e}") from e
        logger.debug(f"Deleted {path}")

    def list_keys(self, prefix: str) -> list[str]:
        """List keys under ``prefix``, relative to the backend prefix."""
        base = self._path_for(prefix)
        search_root = base if base.is_dir() else base.parent
        if not search_root.is_dir():
            return []

        # Skip this backend's own sidecars: ".<name>.tmp" mid-write, "<name>.lock" always.
        offset = len(self._prefix)
        keys = [
            str(path.relative_to(self._root))[offset:]
            for path in sorted(search_root.rglob("*"))
            if path.is_file() and not path.name.startswith(".") and path.suffix != LOCK_SUFFIX
        ]
        logger.debug(f"Found {len(keys)} keys with prefix {base}")
        return keys

    def bulk_delete(self, keys: list[str]) -> dict[str, Exception | None]:
        """Delete ``keys``, reporting per-key outcome instead of raising."""
        results: dict[str, Exception | None] = {}
        for key in keys:
            try:
                self.delete(key)
                results[key] = None
            except StorageError as e:
                results[key] = e
        return results
