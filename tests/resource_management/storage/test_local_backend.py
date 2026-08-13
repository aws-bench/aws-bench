"""Tests for LocalStorageBackend."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from aws_bench.resource_management.storage.exceptions import (
    StorageConflictError,
    StorageError,
    StorageNotFoundError,
)
from aws_bench.resource_management.storage.local_backend import LocalStorageBackend


def _backend(tmp_path: Path) -> LocalStorageBackend:
    return LocalStorageBackend(root=tmp_path / "state")


def test_save_creates_root_and_nested_key(tmp_path: Path):
    backend = _backend(tmp_path)
    etag = backend.save("env/pre-setup/111122223333/baseline.json", b'{"a": 1}', None)

    written = tmp_path / "state" / "snapshots" / "env" / "pre-setup" / "111122223333/baseline.json"
    assert written.read_bytes() == b'{"a": 1}'
    assert etag


def test_root_directory_is_owner_only(tmp_path: Path):
    """Snapshots enumerate every resource in an account, so the store is not world-readable."""
    backend = _backend(tmp_path)
    backend.save("k", b"x", None)

    assert (tmp_path / "state").stat().st_mode & 0o777 == 0o700


def test_load_returns_data_and_matching_etag(tmp_path: Path):
    backend = _backend(tmp_path)
    saved_etag = backend.save("k", b"payload", None)

    data, loaded_etag = backend.load("k")

    assert data == b"payload"
    assert loaded_etag == saved_etag


def test_load_missing_key_raises_not_found(tmp_path: Path):
    with pytest.raises(StorageNotFoundError):
        _backend(tmp_path).load("absent")


def test_exists_reflects_presence(tmp_path: Path):
    backend = _backend(tmp_path)
    assert backend.exists("k") is False
    backend.save("k", b"x", None)
    assert backend.exists("k") is True


def test_delete_is_idempotent(tmp_path: Path):
    backend = _backend(tmp_path)
    backend.save("k", b"x", None)

    backend.delete("k")
    backend.delete("k")

    assert backend.exists("k") is False


def test_save_with_matching_etag_succeeds(tmp_path: Path):
    backend = _backend(tmp_path)
    first = backend.save("k", b"v1", None)

    second = backend.save("k", b"v2", first)

    assert backend.load("k")[0] == b"v2"
    assert second != first


def test_save_with_stale_etag_is_rejected(tmp_path: Path):
    """A caller holding a pre-modification etag must not clobber the newer write."""
    backend = _backend(tmp_path)
    stale = backend.save("k", b"v1", None)
    backend.save("k", b"v2", stale)

    with pytest.raises(StorageConflictError):
        backend.save("k", b"v3", stale)

    assert backend.load("k")[0] == b"v2"


def test_save_with_etag_on_absent_key_raises_not_found(tmp_path: Path):
    with pytest.raises(StorageNotFoundError):
        _backend(tmp_path).save("absent", b"v", "some-etag")


def test_key_escaping_the_root_is_rejected(tmp_path: Path):
    backend = _backend(tmp_path)
    with pytest.raises(StorageError, match="escapes the storage root"):
        backend.save("../../escaped.json", b"x", None)


def test_list_keys_returns_keys_relative_to_prefix(tmp_path: Path):
    backend = _backend(tmp_path)
    backend.save("env/pre-setup/111122223333/baseline.json", b"a", None)
    backend.save("env/post-setup/111122223333/baseline.json", b"b", None)
    backend.save("other/post-setup/999988887777/baseline.json", b"c", None)

    assert backend.list_keys("env/") == [
        "env/post-setup/111122223333/baseline.json",
        "env/pre-setup/111122223333/baseline.json",
    ]


def test_list_keys_on_absent_prefix_is_empty(tmp_path: Path):
    assert _backend(tmp_path).list_keys("nothing/here/") == []


def test_bulk_delete_reports_per_key_outcome(tmp_path: Path):
    backend = _backend(tmp_path)
    backend.save("a", b"x", None)

    results = backend.bulk_delete(["a", "never-existed"])

    assert results == {"a": None, "never-existed": None}
    assert backend.exists("a") is False


def test_concurrent_writers_do_not_interleave(tmp_path: Path):
    """The per-key lock serializes read-modify-write, so exactly one writer wins each round."""
    backend = _backend(tmp_path)
    backend.save("k", b"seed", None)

    def append(index: int) -> str | None:
        data, etag = backend.load("k")
        try:
            return backend.save("k", data + str(index).encode(), etag)
        except StorageConflictError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(append, range(8)))

    # Every writer read the same seed etag, so at most one CAS can succeed.
    assert sum(1 for outcome in outcomes if outcome is not None) == 1
    assert backend.load("k")[0].startswith(b"seed")
