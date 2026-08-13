"""Storage backends for snapshot persistence."""

from typing import Protocol


class SnapshotStorage(Protocol):
    """Key/value storage with compare-and-set writes.

    ``save`` accepts the ETag the caller last read and rejects the write when the
    stored content has moved on, so a read-modify-write cycle cannot silently
    overwrite a concurrent update. Implementations differ in what an ETag is
    (S3's opaque token, a content hash) and in how far the guarantee reaches:
    :class:`~aws_bench.resource_management.storage.s3_backend.S3StorageBackend`
    is shared by every host, while
    :class:`~aws_bench.resource_management.storage.local_backend.LocalStorageBackend`
    is host-local.
    """

    def save(self, key: str, data: bytes, expected_etag: str | None) -> str:
        """Write ``data`` at ``key`` and return its new ETag.

        Raises:
            StorageConflictError: If the stored ETag no longer matches ``expected_etag``.
            StorageNotFoundError: If ``expected_etag`` is given but ``key`` is absent.
        """
        ...

    def load(self, key: str) -> tuple[bytes, str]:
        """Return the data at ``key`` and its ETag.

        Raises:
            StorageNotFoundError: If ``key`` is absent.
        """
        ...

    def exists(self, key: str) -> bool:
        """Return whether ``key`` is present."""
        ...

    def delete(self, key: str) -> None:
        """Delete ``key`` (idempotent)."""
        ...
