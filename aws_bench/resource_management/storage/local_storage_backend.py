"""Filesystem-backed storage with content-hash optimistic locking."""

import hashlib
import os
import sys
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

# Mirrors S3's DeleteObjects limit so a caller sized against this backend does
# not break against the S3 one.
BATCH_DELETE_MAX_KEYS = 1000

_DIR_MODE = 0o700
_FILE_MODE = 0o600


def _etag(data: bytes) -> str:
    """Return the content hash used as this backend's ETag."""
    return hashlib.sha256(data).hexdigest()


def _fsync_dir(path: Path) -> None:
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


class LocalStorageBackend:
    """Filesystem storage exposing the same contract as :class:`S3StorageBackend`.

    ETags are content hashes rather than S3's opaque tokens, which gives the same
    compare-and-set behavior on ``save``: a caller holding a stale ETag is
    rejected with :class:`StorageConflictError`.

    Within ``save``, the ETag check and the write are held under a sibling lock
    file, so concurrent workers in one process (or concurrent processes on one
    host) cannot interleave them. A caller's own ``load`` then ``save`` cycle is
    not covered; the ETag is what guards that. State is host-local: runs spread
    across hosts do not share it.
    """

    def __init__(self, root: Path, prefix: str = DEFAULT_PREFIX):
        """Store objects under ``root``, with every key nested below ``prefix``.

        Args:
            root: Directory holding the store. Created if absent.
            prefix: Key prefix for all operations.
        """
        expanded = root.expanduser()
        expanded.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        expanded.chmod(_DIR_MODE)
        # Resolved once here so paths from ``_path_for`` are comparable to it.
        self._root = expanded.resolve()
        self._prefix = prefix
        logger.debug(f"Initialized local storage backend at {self._root}")

    def _make_full_key(self, key: str) -> str:
        """Return the prefix-joined key, the form errors report."""
        return f"{self._prefix}{key}"

    def _path_for(self, key: str) -> Path:
        """Resolve ``key`` to a path, rejecting anything outside the store.

        Raises:
            StorageError: If ``key`` would resolve outside ``root``.
        """
        base = self._root / self._prefix
        candidate = (base / key).resolve()
        if base != candidate and base not in candidate.parents:
            raise StorageError(f"Key escapes the storage root: {key!r}")
        return candidate

    def _write_atomic(self, path: Path, data: bytes) -> None:
        """Replace ``path`` with ``data`` via a same-directory temp file."""
        path.parent.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                # Flushed before the rename so a crash cannot publish empty content.
                os.fsync(stream.fileno())
            os.chmod(temporary, _FILE_MODE)
            os.replace(temporary, path)
            _fsync_dir(path.parent)
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
                        raise StorageNotFoundError(key=self._make_full_key(key))
                    actual = _etag(path.read_bytes())
                    if actual != expected_etag:
                        raise StorageConflictError(
                            key=self._make_full_key(key), expected=expected_etag, actual=actual
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

        # Tested before locking, which would otherwise create the directory and a
        # lock sidecar for a key that was never written.
        if not path.exists():
            raise StorageNotFoundError(key=self._make_full_key(key))

        try:
            with file_lock(path):
                if not path.exists():
                    raise StorageNotFoundError(key=self._make_full_key(key))
                data = path.read_bytes()
        except OSError as e:
            raise StorageError(f"Failed to load {path}: {e}") from e

        etag = _etag(data)
        logger.debug(f"Loaded {path} ({len(data)} bytes, ETag {etag[:8]}...)")
        return (data, etag)

    def exists(self, key: str) -> bool:
        """Return whether ``key`` is present.

        Raises:
            StorageError: If the check fails
        """
        path = self._path_for(key)
        try:
            present = path.is_file()
        except OSError as e:
            raise StorageError(f"Failed to check {path}: {e}") from e
        logger.debug(f"Key {'exists' if present else 'does not exist'}: {path}")
        return present

    def delete(self, key: str) -> None:
        """Delete ``key`` (idempotent).

        Raises:
            StorageError: If the delete fails
        """
        path = self._path_for(key)

        # Tested before locking, which would otherwise create the directory and a
        # lock sidecar for a key that was never written.
        if not path.exists():
            logger.debug(f"Delete is a no-op; {path} is absent")
            return

        try:
            with file_lock(path):
                path.unlink(missing_ok=True)
        except OSError as e:
            raise StorageError(f"Failed to delete {path}: {e}") from e
        logger.debug(f"Deleted {path}")

    def list_keys(self, prefix: str) -> list[str]:
        """List keys starting with ``prefix``, matched as a string as S3 does.

        Keys are relative to the backend prefix. This backend's own sidecars are
        never keys: ``.<name>.tmp`` while a write is in flight, ``<name>.lock``
        always.

        Raises:
            StorageError: If the walk fails
        """
        base = self._root / self._prefix
        if not base.is_dir():
            return []

        keys: list[str] = []
        try:
            for path in base.rglob("*"):
                if not path.is_file() or path.name.startswith(".") or path.suffix == LOCK_SUFFIX:
                    continue
                key = path.relative_to(base).as_posix()
                if key.startswith(prefix):
                    keys.append(key)
        except OSError as e:
            raise StorageError(f"Failed to list keys under {base}: {e}") from e

        # Sorted as strings, so the order matches S3's lexicographic key order.
        keys.sort()
        logger.debug(f"Found {len(keys)} keys with prefix {prefix!r}")
        return keys

    def bulk_delete(self, keys: list[str]) -> dict[str, Exception | None]:
        """Delete ``keys``, reporting per-key outcome instead of raising.

        Raises:
            StorageError: If given more than ``BATCH_DELETE_MAX_KEYS`` keys
        """
        if len(keys) > BATCH_DELETE_MAX_KEYS:
            raise StorageError(
                f"Batch delete supports max {BATCH_DELETE_MAX_KEYS} keys, got {len(keys)}"
            )

        results: dict[str, Exception | None] = {}
        for key in keys:
            try:
                self.delete(key)
                results[key] = None
            except StorageError as e:
                logger.warning(f"Failed to delete {key}: {e}")
                results[key] = e
        return results
