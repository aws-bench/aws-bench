"""Unit tests for ``scripts/update_registry.py``.

The script is a standalone tool (not part of the ``aws_bench`` package), so it
is loaded by path via importlib. Tests exercise a throwaway git repo that
mimics the aws-bench-datasets layout (``tasks/<scenario>/<task>``, ``scenarios/<scenario>``,
``shared/steering/*.md``, ``metric/metric.py``) so the path-driven commit
resolution runs against real ``git log`` output.

Focus areas mirror the refactor's guarantees:
  * registry is the source of truth and refresh is non-destructive;
  * commit refresh is path-driven, so composite/manual mirror datasets pick up
    upstream changes without being directory-backed;
  * directory-backed scenarios are reconciled from disk;
  * the unified ``all`` dataset is sourced only from directory-backed envs.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "update_registry.py"


@pytest.fixture(scope="module")
def ur():
    """Import the standalone script module by path."""
    spec = importlib.util.spec_from_file_location("update_registry", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# git repo helpers
# --------------------------------------------------------------------------- #
def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    # Isolate from any ambient global hooks path.
    _git(repo, "config", "core.hooksPath", str(repo / ".nohooks"))


def _write(path: Path, content: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _commit(repo: Path, msg: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _add_scenario(repo: Path, scenario: str, tasks: list[str], description: str = "") -> None:
    for t in tasks:
        _write(repo / "tasks" / scenario / t / "task.toml", f"name='{t}'\n")
    _write(
        repo / "scenarios" / scenario / "scenario.toml",
        f'description = "{description or scenario + " scenario"}"\n',
    )


@pytest.fixture
def datasets(tmp_path):
    """A committed datasets repo with shared files and one scenario 'alpha' (2 tasks)."""
    repo = tmp_path / "aws-bench-datasets"
    _init_repo(repo)
    _write(repo / "metric" / "metric.py", "print('m')\n")
    _write(repo / "shared" / "steering" / "aws-awareness.md", "aware\n")
    _write(repo / "shared" / "steering" / "concise-answer.md", "concise\n")
    _add_scenario(repo, "alpha", ["t1", "t2"], description="Alpha scenario")
    _commit(repo, "init alpha")
    return repo


# --------------------------------------------------------------------------- #
# run helper
# --------------------------------------------------------------------------- #
def _run(ur, datasets_path: Path, existing: Path, output: Path, *extra: str) -> list[dict]:
    argv = [
        "update_registry.py",
        "--datasets-path",
        str(datasets_path),
        "--existing-registry",
        str(existing),
        "--output",
        str(output),
        *extra,
    ]
    old = sys.argv
    sys.argv = argv
    try:
        ur.main()
    finally:
        sys.argv = old
    return json.loads(output.read_text())


def _by_name(registry: list[dict]) -> dict[str, dict]:
    return {e["name"]: e for e in registry}


def _pins(entry: dict) -> dict[str, str]:
    return {t["path"]: t["git_commit_id"] for t in entry["tasks"]}


# --------------------------------------------------------------------------- #
# bootstrap / discovery
# --------------------------------------------------------------------------- #
def test_bootstrap_discovers_scenario(ur, datasets, tmp_path):
    out = tmp_path / "registry.json"
    missing = tmp_path / "does-not-exist.json"
    registry = _run(ur, datasets, missing, out)

    by = _by_name(registry)
    assert set(by) == {"alpha"}
    alpha = by["alpha"]
    assert alpha["version"] == "0.1.0"
    assert {t["name"] for t in alpha["tasks"]} == {"t1", "t2"}
    assert [s["name"] for s in alpha["scenarios"]] == ["alpha"]
    assert {p["path"] for p in alpha["extra_instruction_paths"]} == {
        "shared/steering/aws-awareness.md",
        "shared/steering/concise-answer.md",
    }
    assert alpha["metrics"][0]["kwargs"]["script_path"] == "metric/metric.py"
    assert alpha["description"].endswith("2 aws-introspection tasks against alpha.")


def test_empty_registry_and_no_scenarios_errors(ur, tmp_path):
    empty_repo = tmp_path / "empty"
    _init_repo(empty_repo)
    _write(empty_repo / "readme.md", "hi\n")
    _commit(empty_repo, "seed")
    with pytest.raises(SystemExit):
        _run(ur, empty_repo, tmp_path / "none.json", tmp_path / "out.json")


# --------------------------------------------------------------------------- #
# non-destructive retention + path-driven refresh of composite datasets
# --------------------------------------------------------------------------- #
def test_composite_dataset_retained_and_refreshed_by_path(ur, datasets, tmp_path):
    """A manual mirror dataset (not directory-backed) survives and is refreshed.

    Its commit pins are refreshed by path, matching the directory-backed scenario
    it mirrors.
    """
    reg_path = tmp_path / "registry.json"
    out = tmp_path / "out.json"

    # Bootstrap alpha, capture its initial pins.
    boot = _by_name(_run(ur, datasets, tmp_path / "none.json", reg_path))
    old_alpha_pins = _pins(boot["alpha"])

    # Author a composite 'mirror' dataset reusing alpha's paths, pinned stale.
    stale = "0" * 40
    mirror = {
        "name": "mirror",
        "version": "0.3.0",
        "description": "Manual smoke-test mirror of alpha.",
        "tasks": [
            {"name": t["name"], "git_url": t["git_url"], "git_commit_id": stale, "path": t["path"]}
            for t in boot["alpha"]["tasks"]
        ],
        "scenarios": [
            {"name": s["name"], "git_url": s["git_url"], "git_commit_id": stale, "path": s["path"]}
            for s in boot["alpha"]["scenarios"]
        ],
        "metrics": boot["alpha"]["metrics"],
    }
    registry = json.loads(reg_path.read_text())
    registry.append(mirror)
    reg_path.write_text(json.dumps(registry))

    # Upstream change: modify only tasks/alpha/t1 and commit.
    _write(datasets / "tasks" / "alpha" / "t1" / "task.toml", "name='t1'\nchanged=true\n")
    new_commit = _commit(datasets, "touch t1")

    result = _by_name(_run(ur, datasets, reg_path, out))

    # Non-destructive: both datasets present.
    assert set(result) == {"alpha", "mirror"}

    mir = result["mirror"]
    # Description + version identity preserved (only commit content changed -> bump).
    assert mir["description"] == "Manual smoke-test mirror of alpha."
    assert mir["version"] == "0.3.1"
    # Path-driven refresh: stale pins replaced; t1 == new commit, and mirror
    # pins now equal the directory-backed alpha pins for the shared paths.
    assert stale not in _pins(mir).values()
    assert _pins(mir)["tasks/alpha/t1"] == new_commit
    assert _pins(mir) == _pins(result["alpha"])
    # t2 was untouched, so it kept alpha's original pin.
    assert _pins(mir)["tasks/alpha/t2"] == old_alpha_pins["tasks/alpha/t2"]


def test_idempotent_no_change_no_bump(ur, datasets, tmp_path):
    reg_path = tmp_path / "registry.json"
    _run(ur, datasets, tmp_path / "none.json", reg_path)
    first = reg_path.read_text()
    # Re-run with no upstream change; output must be byte-identical.
    out2 = tmp_path / "out2.json"
    _run(ur, datasets, reg_path, out2)
    assert out2.read_text() == first
    assert _by_name(json.loads(first))["alpha"]["version"] == "0.1.0"


# --------------------------------------------------------------------------- #
# directory-backed reconciliation + new scenario discovery
# --------------------------------------------------------------------------- #
def test_directory_backed_reconcile_picks_up_new_task(ur, datasets, tmp_path):
    reg_path = tmp_path / "registry.json"
    _run(ur, datasets, tmp_path / "none.json", reg_path)

    # Add a third task to alpha upstream.
    _write(datasets / "tasks" / "alpha" / "t3" / "task.toml", "name='t3'\n")
    _commit(datasets, "add t3")

    out = tmp_path / "out.json"
    alpha = _by_name(_run(ur, datasets, reg_path, out))["alpha"]
    assert {t["name"] for t in alpha["tasks"]} == {"t1", "t2", "t3"}
    assert alpha["description"].endswith("3 aws-introspection tasks against alpha.")
    assert alpha["version"] == "0.1.1"  # content changed -> patch bump


def test_new_scenario_appended(ur, datasets, tmp_path):
    reg_path = tmp_path / "registry.json"
    _run(ur, datasets, tmp_path / "none.json", reg_path)

    _add_scenario(datasets, "beta", ["b1"], description="Beta scenario")
    _commit(datasets, "add beta")

    out = tmp_path / "out.json"
    result = _by_name(_run(ur, datasets, reg_path, out))
    assert set(result) == {"alpha", "beta"}
    assert result["beta"]["version"] == "0.1.0"
    assert {t["name"] for t in result["beta"]["tasks"]} == {"b1"}


def test_include_scenarios_restricts_reconcile_scope(ur, datasets, tmp_path):
    """--include-scenarios limits which scenarios are reconciled from disk.

    Scenarios outside the set are still commit-refreshed but do not pick up new
    task dirs, and a brand-new scenario outside the set is not appended.
    """
    reg_path = tmp_path / "registry.json"
    _add_scenario(datasets, "beta", ["b1"], description="Beta scenario")
    _commit(datasets, "add beta")
    _run(ur, datasets, tmp_path / "none.json", reg_path)  # bootstrap alpha + beta

    # Upstream: new task in alpha and beta, plus a brand-new scenario gamma.
    _write(datasets / "tasks" / "alpha" / "t3" / "task.toml", "name='t3'\n")
    _write(datasets / "tasks" / "beta" / "b2" / "task.toml", "name='b2'\n")
    _add_scenario(datasets, "gamma", ["g1"], description="Gamma scenario")
    _commit(datasets, "grow alpha+beta, add gamma")

    out = tmp_path / "out.json"
    result = _by_name(_run(ur, datasets, reg_path, out, "--include-scenarios", "alpha"))

    # alpha reconciled from disk -> t3 added; beta only commit-refreshed -> no b2;
    # gamma not in the include set -> not appended.
    assert {t["name"] for t in result["alpha"]["tasks"]} == {"t1", "t2", "t3"}
    assert {t["name"] for t in result["beta"]["tasks"]} == {"b1"}
    assert "gamma" not in result


# --------------------------------------------------------------------------- #
# unified dataset sourcing
# --------------------------------------------------------------------------- #
def test_unified_sources_only_directory_backed_scenarios(ur, datasets, tmp_path):
    reg_path = tmp_path / "registry.json"
    out = tmp_path / "out.json"

    # Add a second real scenario and a composite mirror of alpha.
    _add_scenario(datasets, "beta", ["b1"], description="Beta scenario")
    _commit(datasets, "add beta")
    boot = _by_name(_run(ur, datasets, tmp_path / "none.json", reg_path))
    mirror = {
        "name": "mirror",
        "version": "0.1.0",
        "description": "mirror",
        "tasks": boot["alpha"]["tasks"],
        "scenarios": boot["alpha"]["scenarios"],
        "metrics": boot["alpha"]["metrics"],
    }
    registry = json.loads(reg_path.read_text())
    registry.append(mirror)
    reg_path.write_text(json.dumps(registry))

    result = _by_name(
        _run(ur, datasets, reg_path, out, "--add-unified", "--unified-exclude", "beta")
    )

    # Everything retained.
    assert {"alpha", "beta", "mirror", "all"} <= set(result)
    all_ds = result["all"]
    scenario_names = [s["name"] for s in all_ds["scenarios"]]
    task_names = [t["name"] for t in all_ds["tasks"]]
    # Composite 'mirror' and excluded 'beta' must NOT contribute; no dup names.
    assert scenario_names == ["alpha"]
    assert sorted(task_names) == ["t1", "t2"]
    assert len(task_names) == len(set(task_names))
    assert "beta" in all_ds["description"]  # noted as excluded
    # Shared steering files are carried over (deduplicated across scenarios).
    assert all_ds["extra_instruction_paths"] == result["alpha"]["extra_instruction_paths"]
    assert all_ds["extra_instruction_paths"]


# --------------------------------------------------------------------------- #
# pinning / safety
# --------------------------------------------------------------------------- #
def test_git_commit_pins_everything(ur, datasets, tmp_path):
    reg_path = tmp_path / "registry.json"
    pin = "a" * 40
    alpha = _by_name(_run(ur, datasets, tmp_path / "none.json", reg_path, "--git-commit", pin))[
        "alpha"
    ]
    ids = (
        [t["git_commit_id"] for t in alpha["tasks"]]
        + [s["git_commit_id"] for s in alpha["scenarios"]]
        + [p["git_commit_id"] for p in alpha["extra_instruction_paths"]]
        + [alpha["metrics"][0]["kwargs"]["git_commit_id"]]
    )
    assert set(ids) == {pin}


def test_dirty_working_tree_errors(ur, datasets, tmp_path):
    reg_path = tmp_path / "registry.json"
    _run(ur, datasets, tmp_path / "none.json", reg_path)
    # Uncommitted edit under a registry path.
    _write(datasets / "tasks" / "alpha" / "t1" / "task.toml", "dirty\n")
    with pytest.raises(SystemExit) as exc:
        _run(ur, datasets, reg_path, tmp_path / "out.json")
    assert "uncommitted changes" in str(exc.value)


def test_unresolvable_path_kept_and_warned(ur, datasets, tmp_path, capsys):
    """An entry with an unresolvable path keeps its pin and warns, not aborts.

    A composite entry referencing a path with no git history keeps its previous
    commit pin and emits a warning rather than aborting the run.
    """
    reg_path = tmp_path / "registry.json"
    _run(ur, datasets, tmp_path / "none.json", reg_path)

    keep = "b" * 40
    ghost = {
        "name": "ghost",
        "version": "0.1.0",
        "description": "references a non-existent path",
        "tasks": [
            {
                "name": "gone",
                "git_url": "ssh://x/pkg/aws-bench-datasets",
                "git_commit_id": keep,
                "path": "tasks/ghost/gone",
            }
        ],
        "scenarios": [],
        "metrics": [],
    }
    registry = json.loads(reg_path.read_text())
    registry.append(ghost)
    reg_path.write_text(json.dumps(registry))

    out = tmp_path / "out.json"
    result = _by_name(_run(ur, datasets, reg_path, out))
    assert result["ghost"]["tasks"][0]["git_commit_id"] == keep  # pin preserved
    assert "no resolvable git history" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# pure-function units
# --------------------------------------------------------------------------- #
def test_bump_patch(ur):
    assert ur.bump_patch("0.1.9") == "0.1.10"
    assert ur.bump_patch("1.2.3") == "1.2.4"


def test_bump_patch_rejects_non_semver(ur):
    with pytest.raises(SystemExit):
        ur.bump_patch("1.2")


def test_content_fingerprint_ignores_version(ur):
    a = {"name": "x", "version": "1.0.0", "tasks": []}
    b = {"name": "x", "version": "9.9.9", "tasks": []}
    assert ur._content_fingerprint(a) == ur._content_fingerprint(b)


def test_resolve_version_new_entry_keeps_default(ur):
    entry = {"name": "x", "version": ur.DEFAULT_VERSION}
    bumped: list = []
    ur.resolve_version(entry, None, no_auto_bump=False, bumped=bumped)
    assert entry["version"] == ur.DEFAULT_VERSION
    assert bumped == []


def test_resolve_version_identical_preserves(ur):
    prev = {"name": "x", "version": "2.5.0", "tasks": []}
    entry = {"name": "x", "version": ur.DEFAULT_VERSION, "tasks": []}
    bumped: list = []
    ur.resolve_version(entry, prev, no_auto_bump=False, bumped=bumped)
    assert entry["version"] == "2.5.0"
    assert bumped == []


def test_resolve_version_change_bumps_unless_suppressed(ur):
    prev = {"name": "x", "version": "2.5.0", "tasks": [{"a": 1}]}
    entry = {"name": "x", "version": ur.DEFAULT_VERSION, "tasks": [{"a": 2}]}
    bumped: list = []
    ur.resolve_version(entry, prev, no_auto_bump=False, bumped=bumped)
    assert entry["version"] == "2.5.1"
    assert bumped == [("x", "2.5.0", "2.5.1")]

    entry2 = {"name": "x", "version": ur.DEFAULT_VERSION, "tasks": [{"a": 3}]}
    bumped2: list = []
    ur.resolve_version(entry2, prev, no_auto_bump=True, bumped=bumped2)
    assert entry2["version"] == "2.5.0"  # suppressed
    assert bumped2 == []


def test_refresh_entry_commits_repins_by_path(ur, datasets):
    """Direct unit test of the path-driven refresh helper."""
    head = _git(datasets, "rev-parse", "HEAD")
    entry = {
        "name": "x",
        "tasks": [
            {"name": "t1", "git_commit_id": "stale", "path": "tasks/alpha/t1"},
        ],
        "scenarios": [
            {"name": "alpha", "git_commit_id": "stale", "path": "scenarios/alpha"},
        ],
        "extra_instruction_paths": [
            {"git_commit_id": "stale", "path": "shared/steering/aws-awareness.md"},
        ],
        "metrics": [
            {
                "type": "uv-script",
                "kwargs": {"git_commit_id": "stale", "script_path": "metric/metric.py"},
            }
        ],
    }
    unresolved = ur.refresh_entry_commits(entry, datasets, pin_commit=None)
    assert unresolved == []
    assert entry["tasks"][0]["git_commit_id"] == head
    assert entry["scenarios"][0]["git_commit_id"] == head
    assert entry["extra_instruction_paths"][0]["git_commit_id"] == head
    assert entry["metrics"][0]["kwargs"]["git_commit_id"] == head


def test_refresh_entry_commits_pin_overrides(ur, datasets):
    pin = "c" * 40
    entry = {
        "name": "x",
        "tasks": [{"name": "t1", "git_commit_id": "old", "path": "tasks/alpha/t1"}],
    }
    assert ur.refresh_entry_commits(entry, datasets, pin_commit=pin) == []
    assert entry["tasks"][0]["git_commit_id"] == pin
