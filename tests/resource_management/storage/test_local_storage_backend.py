"""Tests for LocalStorageBackend."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from aws_bench.resource_management.storage.exceptions import (
    StorageConflictError,
    StorageError,
    StorageNotFoundError,
)
from aws_bench.resource_management.storage.local_storage_backend import LocalStorageBackend


def _backend(tmp_path: Path) -> LocalStorageBackend:
    return LocalStorageBackend(root=tmp_path / "state")


def test_errors_report_the_storage_key_not_a_filesystem_path(tmp_path: Path):
    """S3StorageBackend reports the prefix-joined key; a caller must not have to care which."""
    backend = _backend(tmp_path)
    with pytest.raises(StorageNotFoundError) as absent:
        backend.load("env/baseline.json")
    assert absent.value.key == "snapshots/env/baseline.json"

    stale = backend.save("env/baseline.json", b"v1", None)
    backend.save("env/baseline.json", b"v2", stale)
    with pytest.raises(StorageConflictError) as conflict:
        backend.save("env/baseline.json", b"v3", stale)
    assert conflict.value.key == "snapshots/env/baseline.json"


def test_save_creates_root_and_nested_key(tmp_path: Path):
    backend = _backend(tmp_path)
    etag = backend.save("env/pre-setup/111122223333/baseline.json", b'{"a": 1}', None)

    written = tmp_path / "state" / "snapshots" / "env" / "pre-setup" / "111122223333/baseline.json"
    assert written.read_bytes() == b'{"a": 1}'
    assert etag


def test_root_directory_is_owner_only(tmp_path: Path):
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


@pytest.mark.parametrize(
    "key",
    [
        "../../escaped.json",
        "../sibling-of-prefix.json",
        "a/../../sibling-of-prefix.json",
        "/absolute/../../../escaped.json",
    ],
)
def test_key_escaping_the_prefix_is_rejected(tmp_path: Path, key: str):
    """The boundary is the prefix, not the root, which also holds the contamination file."""
    backend = _backend(tmp_path)
    with pytest.raises(StorageError, match="escapes the storage root"):
        backend.save(key, b"x", None)


def test_traversal_cannot_write_beside_the_prefix(tmp_path: Path):
    root = tmp_path / "state"
    backend = LocalStorageBackend(root=root)
    with pytest.raises(StorageError):
        backend.save("../contamination.json", b"x", None)

    assert not (root / "contamination.json").exists()


def test_symlinked_root_is_usable(tmp_path: Path):
    """An unresolved root would make list_keys raise a bare ValueError."""
    real = tmp_path / "real_store"
    real.mkdir()
    link = tmp_path / "linked_store"
    link.symlink_to(real)
    backend = LocalStorageBackend(root=link)

    backend.save("k.json", b"x", None)

    assert backend.load("k.json")[0] == b"x"
    assert backend.list_keys("") == ["k.json"]


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


def test_list_keys_on_absent_prefix_does_not_widen_to_the_parent(tmp_path: Path):
    """An absent prefix returns nothing rather than every sibling environment's keys."""
    backend = _backend(tmp_path)
    backend.save("envA/pre-setup/111122223333/baseline.json", b"a", None)
    backend.save("envB/pre-setup/999988887777/baseline.json", b"b", None)

    assert backend.list_keys("nope/") == []
    assert backend.list_keys("envA/") == ["envA/pre-setup/111122223333/baseline.json"]


def test_list_keys_matches_a_partial_string_prefix(tmp_path: Path):
    backend = _backend(tmp_path)
    backend.save("env-one/baseline.json", b"a", None)
    backend.save("env-two/baseline.json", b"b", None)

    assert backend.list_keys("env-o") == ["env-one/baseline.json"]


def test_reads_of_absent_keys_leave_no_trace_on_disk(tmp_path: Path):
    """S3 creates nothing on a miss; neither may a load, an exists, or a delete here."""
    root = tmp_path / "state"
    backend = LocalStorageBackend(root=root)

    backend.exists("a/b/c")
    with pytest.raises(StorageNotFoundError):
        backend.load("a/b/c")
    backend.delete("x/y/z")

    assert list(root.rglob("*")) == []


def test_delete_leaves_no_lock_sidecar_for_an_absent_key(tmp_path: Path):
    backend = _backend(tmp_path)
    backend.save("kept.json", b"x", None)

    backend.delete("never-written.json")

    assert backend.list_keys("") == ["kept.json"]
    assert not (tmp_path / "state" / "snapshots" / "never-written.json.lock").exists()


def test_bulk_delete_rejects_more_keys_than_s3_accepts(tmp_path: Path):
    """Staying inside S3's limit keeps a caller sized here working against S3."""
    with pytest.raises(StorageError, match="max 1000 keys"):
        _backend(tmp_path).bulk_delete([f"k{index}" for index in range(1001)])


def test_bulk_delete_reports_per_key_outcome(tmp_path: Path):
    backend = _backend(tmp_path)
    backend.save("a", b"x", None)

    results = backend.bulk_delete(["a", "never-existed"])

    assert results == {"a": None, "never-existed": None}
    assert backend.exists("a") is False


def test_concurrent_writers_sharing_an_etag_admit_one(tmp_path: Path):
    """Concurrent compare-and-set against one etag lets exactly one writer through.

    The key then holds that writer's payload whole, never a mixture of two.
    """
    backend = _backend(tmp_path)
    seed_etag = backend.save("k", b"seed", None)

    def overwrite(index: int) -> str | None:
        try:
            return backend.save("k", f"writer-{index}".encode(), seed_etag)
        except StorageConflictError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(overwrite, range(8)))

    assert sum(1 for outcome in outcomes if outcome is not None) == 1
    assert backend.load("k")[0] in {f"writer-{index}".encode() for index in range(8)}
