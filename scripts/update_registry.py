#!/usr/bin/env python3
"""Update registry.json from the aws-bench-datasets repo.

Usage:
    python scripts/update_registry.py \
        --datasets-path /path/to/aws-bench-datasets \
        --include-scenarios ec2-multiregion troubleshooting-multiservice ...

If --include-scenarios is omitted, all scenarios that have both a scenarios/
and tasks/ directory are included. Output is written to registry.json in the
current working directory (or --output).

Per-path commit pinning
-----------------------
By default each task and scenario entry is pinned to the most recent commit
that actually touched its path, resolved via:

    git -C <datasets-path> log -1 --format=%H -- <path>

This means rerunning the script only changes the ``git_commit_id`` of tasks
or scenarios whose contents (or any descendant) were modified upstream;
unchanged entries keep their previous commit, byte-for-byte. Pass
``--git-commit <sha>`` to instead pin every task and scenario to a single hash.

A path with no git history (untracked, never committed) is a hard error.

Working-tree cleanliness
------------------------
``git log -1 -- <path>`` only sees committed history, so uncommitted edits
under ``tasks/<scenario>/<name>``, ``scenarios/<scenario>``, or
``shared/steering`` would be silently missed. Before resolving commits the
script runs ``git status --porcelain`` over the involved paths and exits with
an error if anything is dirty. Commit (or stash) your changes in the datasets
repo before regenerating the registry.

Registry as source of truth
---------------------------
The existing registry (``--existing-registry``, default ``registry.json``) is
the authoritative list of which datasets exist and which repo paths each one
pins. The script is **non-destructive**: every entry already in the registry
is retained. There are two ways an entry gets refreshed:

  * Directory-backed scenarios (a name with both ``tasks/<name>/`` and
    ``scenarios/<name>/`` on disk) are reconciled from disk -- their task
    membership, description, and commit pins are rebuilt. ``--include-scenarios``
    restricts this reconciliation set; omitted means all discovered ones.
  * Every other entry -- composite/manual datasets such as ``aws-bench-quickstart``
    (whose task entries point at another scenario's ``tasks/<scenario>/...``
    paths) and the unified ``all`` dataset -- is refreshed *by path*: each
    ``git_commit_id`` is re-resolved from its own ``path`` field while name,
    description, and membership are preserved verbatim.

Because refresh is keyed on the ``path`` field rather than the dataset name,
manual mirror datasets pick up upstream changes automatically without the
script needing to know they exist. Newly discovered scenarios that are not
yet in the registry are appended as new entries.

Existing-entry handling
-----------------------
Each refreshed/rebuilt entry is diffed against its previous form (excluding the
version field):

  * Identical content → previous version is preserved.
  * Different content → patch component of the previous version is bumped
    (e.g. 0.1.0 -> 0.1.1) and a line is logged to stderr. ``--no-auto-bump``
    suppresses the bump (version is left untouched even when content changes).
  * New entry → starts at 0.1.0.

``--preserve-existing`` is now the default behaviour and is accepted only as a
deprecated no-op.

Unified dataset
---------------
``--add-unified`` appends a synthetic ``all`` dataset whose scenarios and tasks
are the union of every real scenario's (each entry's per-path commit pins are
preserved verbatim). This lets a single ``aws-bench run -d all`` cover the
whole suite in one run. By default nothing is excluded -- the unified entry
spans every discovered scenario (``--unified-exclude`` can restrict the set).
The unified entry participates in the same content-diff/auto-bump
logic as real entries.
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


def git_last_commit_for_path(repo_path: Path, rel_path: str, strict: bool = True) -> str | None:
    """Return the most recent commit hash that touched rel_path within repo_path.

    ``strict=True`` (the default, used when building directory-backed entries)
    exits with an error if the path has no git history. ``strict=False`` (used
    when refreshing an existing entry in place) returns ``None`` instead, so a
    single unresolvable path degrades to "keep the previous pin" rather than
    aborting the whole refresh.
    """
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", rel_path],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    sha = result.stdout.strip()
    if not sha:
        if strict:
            sys.exit(f"ERROR: no git history for {rel_path} in {repo_path}; is the path committed?")
        return None
    return sha


def git_dirty_paths(repo_path: Path, rel_paths: list[str]) -> list[str]:
    """Return the subset of rel_paths that have uncommitted changes."""
    if not rel_paths:
        return []
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *rel_paths],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    dirty = []
    for line in result.stdout.splitlines():
        # Porcelain format: "XY <path>" where X/Y are status codes (1 char each)
        # plus a space. Path starts at column 3.
        if len(line) > 3:
            dirty.append(line[3:].split(" -> ", 1)[-1].strip())
    return dirty


def read_scenario_description(scenario_toml: Path) -> str:
    """Extract the description field from a scenario.toml file."""
    for line in scenario_toml.read_text().splitlines():
        if line.startswith("description"):
            # Parse: description = "..."
            _, _, value = line.partition("=")
            return value.strip().strip('"')
    return ""


DEFAULT_VERSION = "0.1.0"

SHARED_INSTRUCTION_DIR = "shared/steering"
METRIC_SCRIPT_PATH = "metric/metric.py"


def discover_instruction_paths(datasets_path: Path) -> list[str]:
    """Return sorted repo-relative paths of the shared steering ``.md`` files.

    Empty if the directory is absent. Sorted for deterministic registry output.
    """
    steering_dir = datasets_path / SHARED_INSTRUCTION_DIR
    if not steering_dir.is_dir():
        return []
    return sorted(f"{SHARED_INSTRUCTION_DIR}/{p.name}" for p in steering_dir.glob("*.md"))


# Synthetic dataset that co-lists every real scenario's scenario + tasks so a
# single run can target them all via -d <UNIFIED_DATASET_NAME>. Regenerated
# fresh on every run. By default nothing is excluded -- the unified dataset
# spans every discovered scenario; restrict the set with --unified-exclude.
UNIFIED_DATASET_NAME = "all"
DEFAULT_UNIFIED_EXCLUDE = ()


def build_metrics(git_url: str, metric_commit: str) -> list[dict]:
    """Build the registry metrics list: a single git uv-script entry."""
    return [
        {
            "type": "uv-script",
            "kwargs": {
                "git_url": git_url,
                "git_commit_id": metric_commit,
                "script_path": METRIC_SCRIPT_PATH,
            },
        }
    ]


def build_unified_entry(
    name: str, source_entries: list[dict], exclude: set[str], metrics: list[dict]
) -> dict:
    """Build a synthetic dataset unifying the scenarios + tasks of source_entries.

    ``source_entries`` are the real per-scenario entries; each contributing
    scenario's ``scenarios`` and ``tasks`` are concatenated verbatim (their
    per-path ``git_commit_id`` pins are preserved) and their
    ``extra_instruction_paths`` are unioned by path. Scenarios named in
    ``exclude`` — and the unified entry itself — are left out. Scenario/task
    names are unique across scenarios, so co-listing them under one dataset
    satisfies the framework's per-dataset uniqueness checks.
    """
    included = [e for e in source_entries if e["name"] not in exclude and e["name"] != name]
    scenarios = [s for e in included for s in e.get("scenarios", [])]
    tasks = [t for e in included for t in e.get("tasks", [])]
    # Every scenario pins the same shared/steering files; dedupe by path,
    # keeping first occurrence order.
    instructions = list(
        {i["path"]: i for e in included for i in e.get("extra_instruction_paths", [])}.values()
    )
    included_names = sorted(e["name"] for e in included)

    all_names = {e["name"] for e in source_entries}
    excluded_present = sorted(all_names & exclude)

    description = (
        f"Unified dataset spanning {len(included_names)} scenario(s) "
        f"({', '.join(included_names)}): {len(scenarios)} scenarios, {len(tasks)} tasks."
    )
    if excluded_present:
        description += f" Excludes: {', '.join(excluded_present)}."

    return {
        "name": name,
        "version": DEFAULT_VERSION,
        "description": description,
        "tasks": tasks,
        "scenarios": scenarios,
        "extra_instruction_paths": instructions,
        "metrics": metrics,
    }


def build_scenario_entry(
    scenario_name: str,
    datasets_path: Path,
    git_url: str,
    pin_commit: str | None,
    instruction_paths: list[str],
) -> dict | None:
    """Build a registry entry for a single scenario.

    If ``pin_commit`` is given, every task/scenario is pinned to that hash.
    Otherwise each path is pinned to the last commit that touched it.
    ``instruction_paths`` are the shared steering files (see
    ``discover_instruction_paths``); each is pinned the same way.
    """
    tasks_dir = datasets_path / "tasks" / scenario_name
    scenario_dir = datasets_path / "scenarios" / scenario_name

    if not tasks_dir.is_dir():
        print(f"WARNING: no tasks directory for {scenario_name}, skipping", file=sys.stderr)
        return None

    task_names = sorted(d.name for d in tasks_dir.iterdir() if d.is_dir())

    scenario_toml = scenario_dir / "scenario.toml" if scenario_dir.is_dir() else None
    description = ""
    if scenario_toml and scenario_toml.exists():
        description = read_scenario_description(scenario_toml)

    if description:
        description = description.rstrip(".")
        description = (
            f"{description}; {len(task_names)} aws-introspection tasks against {scenario_name}."
        )
    else:
        description = f"{len(task_names)} aws-introspection tasks against {scenario_name}."

    def commit_for(rel_path: str) -> str:
        if pin_commit is not None:
            return pin_commit
        return git_last_commit_for_path(datasets_path, rel_path)

    tasks = []
    for name in task_names:
        rel = f"tasks/{scenario_name}/{name}"
        tasks.append(
            {
                "name": name,
                "git_url": git_url,
                "git_commit_id": commit_for(rel),
                "path": rel,
            }
        )

    scenarios = []
    if scenario_dir and scenario_dir.is_dir():
        rel = f"scenarios/{scenario_name}"
        scenarios.append(
            {
                "name": scenario_name,
                "git_url": git_url,
                "git_commit_id": commit_for(rel),
                "path": rel,
            }
        )

    extra_instruction_paths = [
        {
            "git_url": git_url,
            "git_commit_id": commit_for(rel),
            "path": rel,
        }
        for rel in instruction_paths
    ]

    metrics = build_metrics(git_url, commit_for(METRIC_SCRIPT_PATH))

    return {
        "name": scenario_name,
        "version": DEFAULT_VERSION,
        "description": description,
        "tasks": tasks,
        "scenarios": scenarios,
        "extra_instruction_paths": extra_instruction_paths,
        "metrics": metrics,
    }


def _content_fingerprint(entry: dict) -> str:
    """Stable JSON of an entry excluding the version field, for change detection."""
    return json.dumps({k: v for k, v in entry.items() if k != "version"}, sort_keys=True)


def bump_patch(version: str) -> str:
    """Increment the patch component of a semver-ish version string."""
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        sys.exit(
            f"ERROR: cannot bump non-semver version {version!r}; "
            "pass --no-auto-bump and update it manually."
        )
    major, minor, patch = (int(p) for p in parts)
    return f"{major}.{minor}.{patch + 1}"


# Keys whose list items each carry a ``path`` + ``git_commit_id`` pair.
_PINNED_PATH_KEYS = ("tasks", "scenarios", "extra_instruction_paths")


def refresh_entry_commits(entry: dict, repo_path: Path, pin_commit: str | None) -> list[str]:
    """Re-resolve ``git_commit_id`` for every pinned path in ``entry`` in place.

    Path-driven and name-agnostic: it walks the ``tasks``, ``scenarios`` and
    ``extra_instruction_paths`` lists plus the ``metrics`` script path, and
    repins each to the latest commit that touched its ``path``. This is what
    lets composite/manual datasets (e.g. ``aws-bench-quickstart``, whose entries
    point at another scenario's ``tasks/<scenario>/...`` paths) pick up upstream
    changes without the script needing to know they exist.

    If ``pin_commit`` is given, every pin is set to that hash instead. Paths
    with no resolvable history are left at their previous pin and returned so
    the caller can warn. Returns the list of paths that could not be resolved.
    """
    unresolved: list[str] = []

    def commit_for(rel_path: str, current: str | None) -> str | None:
        if pin_commit is not None:
            return pin_commit
        sha = git_last_commit_for_path(repo_path, rel_path, strict=False)
        if sha is None:
            unresolved.append(rel_path)
            return current
        return sha

    for key in _PINNED_PATH_KEYS:
        for item in entry.get(key, []):
            rel = item.get("path")
            if rel:
                item["git_commit_id"] = commit_for(rel, item.get("git_commit_id"))

    for metric in entry.get("metrics", []):
        kwargs = metric.get("kwargs", {})
        rel = kwargs.get("script_path")
        if rel:
            kwargs["git_commit_id"] = commit_for(rel, kwargs.get("git_commit_id"))

    return unresolved


def resolve_version(
    entry: dict,
    prev: dict | None,
    no_auto_bump: bool,
    bumped: list[tuple[str, str, str]],
) -> None:
    """Set ``entry['version']`` based on its diff against the previous entry.

    * No previous entry → keep the entry's own version (a fresh build seeds
      ``DEFAULT_VERSION``).
    * Content identical (ignoring the version field) → keep previous version.
    * Content changed → patch-bump previous version, unless ``no_auto_bump``.
      Bumps are appended to ``bumped`` for logging.
    """
    if prev is None:
        return
    prev_version = prev.get("version", DEFAULT_VERSION)
    if _content_fingerprint(prev) == _content_fingerprint(entry):
        entry["version"] = prev_version
    elif no_auto_bump:
        entry["version"] = prev_version
    else:
        new_version = bump_patch(prev_version)
        entry["version"] = new_version
        bumped.append((entry["name"], prev_version, new_version))


def discover_scenarios(datasets_path: Path) -> set[str]:
    """Find scenarios that have both tasks/ and scenarios/ directories."""
    tasks_dir = datasets_path / "tasks"
    scenarios_dir = datasets_path / "scenarios"
    scenarios = set()
    if tasks_dir.is_dir():
        scenarios.update(d.name for d in tasks_dir.iterdir() if d.is_dir())
    if scenarios_dir.is_dir():
        scenarios &= {d.name for d in scenarios_dir.iterdir() if d.is_dir()}
    return scenarios


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets-path",
        type=Path,
        required=True,
        help="Path to the aws-bench-datasets repo checkout",
    )
    parser.add_argument(
        "--include-scenarios",
        nargs="*",
        help=(
            "Directory-backed scenarios to reconcile from disk (rebuild task "
            "membership + description). Default: all discovered. All other "
            "registry entries are still refreshed by path regardless of this flag."
        ),
    )
    parser.add_argument(
        "--git-commit",
        default=None,
        help=(
            "Pin every task and scenario to this commit. "
            "Default: pin each path to the last commit that touched it."
        ),
    )
    parser.add_argument(
        "--git-url",
        default="https://github.com/aws-bench/aws-bench-datasets.git",
        help="Git URL for the datasets repo",
    )
    parser.add_argument(
        "--existing-registry",
        type=Path,
        default=Path("registry.json"),
        help="Existing registry path (default: registry.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("registry.json"),
        help="Output path (default: registry.json)",
    )
    parser.add_argument(
        "--preserve-existing",
        action="store_true",
        help=(
            "DEPRECATED no-op. The registry is now always the source of truth "
            "and existing entries are never dropped, so this flag has no effect."
        ),
    )
    parser.add_argument(
        "--add-unified",
        action="store_true",
        help=(
            f"Append a synthetic '{UNIFIED_DATASET_NAME}' dataset that unifies all "
            "scenarios and their tasks into one descriptor "
            "(spans every discovered scenario by default; see --unified-exclude)."
        ),
    )
    parser.add_argument(
        "--unified-name",
        default=UNIFIED_DATASET_NAME,
        help=f"Name of the unified dataset (default: {UNIFIED_DATASET_NAME}).",
    )
    parser.add_argument(
        "--unified-exclude",
        nargs="*",
        default=list(DEFAULT_UNIFIED_EXCLUDE),
        help=(
            "Scenario names to exclude from the unified dataset "
            "(default: none -- include every discovered scenario)."
        ),
    )
    parser.add_argument(
        "--no-auto-bump",
        action="store_true",
        help=(
            "Do not auto-bump the patch version of changed entries. "
            "Without this flag, any rebuilt entry whose content differs from "
            "the existing one (including task commit IDs) gets its version "
            "patch-bumped."
        ),
    )
    args = parser.parse_args()

    datasets_path = args.datasets_path.resolve()
    if not datasets_path.is_dir():
        sys.exit(f"ERROR: datasets path does not exist: {datasets_path}")

    discovered = discover_scenarios(datasets_path)

    # Directory-backed scenarios to reconcile with disk (refresh task
    # membership + description, not just commit pins). Default: all discovered.
    if args.include_scenarios:
        requested = set(args.include_scenarios)
        unknown = sorted(requested - discovered)
        if unknown:
            print(
                "WARNING: requested scenarios not found on disk "
                f"(skipped for reconciliation): {', '.join(unknown)}",
                file=sys.stderr,
            )
        reconcile_scenarios = sorted(requested & discovered)
    else:
        reconcile_scenarios = sorted(discovered)

    print(
        f"Discovered scenarios: {', '.join(sorted(discovered)) or '(none)'}",
        file=sys.stderr,
    )
    if reconcile_scenarios:
        print(f"Reconciling from disk: {', '.join(reconcile_scenarios)}", file=sys.stderr)

    instruction_paths = discover_instruction_paths(datasets_path)
    if instruction_paths:
        print(f"Shared instructions: {', '.join(instruction_paths)}", file=sys.stderr)

    # Load the existing registry: it is the source of truth for which datasets
    # exist and which paths each one pins.
    existing: list[dict] = []
    existing_by_name: dict[str, dict] = {}
    if args.existing_registry.exists():
        try:
            existing = json.loads(args.existing_registry.read_text())
        except json.JSONDecodeError as exc:
            sys.exit(f"ERROR: cannot parse existing {args.existing_registry}: {exc}")
        if not isinstance(existing, list):
            sys.exit(f"ERROR: existing {args.existing_registry} is not a JSON array")
        existing_by_name = {e["name"]: e for e in existing if "name" in e}

    if not existing and not discovered:
        sys.exit("ERROR: empty registry and no scenarios discovered; nothing to do")

    # Collect every repo path we will resolve so the cleanliness check covers
    # them all -- including paths referenced only by composite/manual datasets.
    paths_to_check: set[str] = {METRIC_SCRIPT_PATH, *instruction_paths}
    for scenario_name in reconcile_scenarios:
        paths_to_check.add(f"tasks/{scenario_name}")
        if (datasets_path / "scenarios" / scenario_name).is_dir():
            paths_to_check.add(f"scenarios/{scenario_name}")
    for entry in existing:
        for key in _PINNED_PATH_KEYS:
            for item in entry.get(key, []):
                if item.get("path"):
                    paths_to_check.add(item["path"])
        for metric in entry.get("metrics", []):
            script_path = metric.get("kwargs", {}).get("script_path")
            if script_path:
                paths_to_check.add(script_path)

    dirty = git_dirty_paths(datasets_path, sorted(paths_to_check))
    if dirty:
        sys.exit(
            "ERROR: datasets repo has uncommitted changes under registry paths:\n  "
            + "\n  ".join(dirty)
            + "\nCommit or stash these changes before regenerating the registry."
        )

    if args.git_commit is not None:
        print(f"Pinning all entries to commit: {args.git_commit}", file=sys.stderr)
    else:
        print("Pinning each entry to its last-touching commit", file=sys.stderr)

    bumped: list[tuple[str, str, str]] = []
    unresolved_paths: list[str] = []
    registry: list[dict] = []
    seen: set[str] = set()

    # 1. Walk the existing registry in order -- an entry is NEVER dropped.
    for prev in existing:
        name = prev.get("name")
        if name is None:
            registry.append(prev)  # malformed but preserved verbatim
            continue
        seen.add(name)

        # The unified dataset is regenerated wholesale below when --add-unified
        # is set; skip it here so it isn't duplicated.
        if args.add_unified and name == args.unified_name:
            continue

        if name in reconcile_scenarios:
            # Directory-backed: rebuild from disk (refreshes task membership,
            # description, and all commit pins). Fall back to an in-place
            # refresh if the directory vanished mid-run, so we still keep it.
            entry = build_scenario_entry(
                name, datasets_path, args.git_url, args.git_commit, instruction_paths
            )
            if entry is None:
                entry = copy.deepcopy(prev)
                unresolved_paths.extend(
                    refresh_entry_commits(entry, datasets_path, args.git_commit)
                )
        else:
            # Composite/manual dataset (or a directory-backed scenario outside
            # the reconcile set): refresh commit pins by path, preserve
            # everything else (name, description, membership) exactly as authored.
            entry = copy.deepcopy(prev)
            unresolved_paths.extend(refresh_entry_commits(entry, datasets_path, args.git_commit))

        resolve_version(entry, prev, args.no_auto_bump, bumped)
        registry.append(entry)

    # 2. Append newly discovered scenarios not already in the registry.
    added: list[str] = []
    for scenario_name in reconcile_scenarios:
        if scenario_name in seen:
            continue
        entry = build_scenario_entry(
            scenario_name, datasets_path, args.git_url, args.git_commit, instruction_paths
        )
        if entry:
            registry.append(entry)  # version already seeded to DEFAULT_VERSION
            added.append(scenario_name)
    if added:
        print(f"Added {len(added)} new scenarios: {', '.join(added)}", file=sys.stderr)

    # 3. Regenerate the unified dataset from the (now refreshed) real entries.
    if args.add_unified:
        exclude = set(args.unified_exclude)
        # Source ONLY from directory-backed scenarios. Composite/manual datasets
        # (e.g. aws-bench-quickstart) and any prior unified entry are never folded in,
        # otherwise their re-used scenario/task names would collide with the real
        # scenarios inside the union.
        source_entries = [e for e in registry if e["name"] in discovered]
        metrics = build_metrics(
            args.git_url,
            args.git_commit or git_last_commit_for_path(datasets_path, METRIC_SCRIPT_PATH),
        )
        unified = build_unified_entry(args.unified_name, source_entries, exclude, metrics)
        resolve_version(unified, existing_by_name.get(args.unified_name), args.no_auto_bump, bumped)
        registry = [e for e in registry if e["name"] != args.unified_name] + [unified]
        print(
            f"Unified dataset '{args.unified_name}': "
            f"{len(unified['scenarios'])} scenarios, {len(unified['tasks'])} tasks",
            file=sys.stderr,
        )

    if args.preserve_existing:
        print(
            "NOTE: --preserve-existing is now the default and is a no-op; the "
            "registry is always the source of truth and no entries are dropped.",
            file=sys.stderr,
        )

    if unresolved_paths:
        uniq = sorted(set(unresolved_paths))
        print(
            f"WARNING: {len(uniq)} path(s) had no resolvable git history; kept "
            "their previous commit pins:\n  " + "\n  ".join(uniq),
            file=sys.stderr,
        )

    for name, old, new in bumped:
        print(f"Bumped {name}: {old} -> {new}", file=sys.stderr)

    args.output.write_text(json.dumps(registry, indent=2) + "\n")
    print(f"Wrote {len(registry)} datasets to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
