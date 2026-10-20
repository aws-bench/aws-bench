"""Local-mode tests for AwsBenchDatasetConfig.get_scenario_configs."""

from pathlib import Path

import pytest

from aws_bench.dataset.config import AwsBenchDatasetConfig
from aws_bench.dataset.exceptions import TaskConfigInvalidError
from aws_bench.dataset.task_config import AwsBenchTask
from aws_bench.scenario.exceptions import DuplicateScenarioNameError
from aws_bench.scenario.locator import ScenarioConfig
from tests.conftest import make_scenario_layout


def _make_task_layout(task_dir: Path, scenario_id: str = "ec2-small") -> Path:
    """Build a minimal valid aws-bench task dir (Harbor instruction+test + env def)."""
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(
        f'[task]\nname = "aws-bench/{task_dir.name}"\n\n[scenario]\nscenario_id = "{scenario_id}"\n'
    )
    (task_dir / "instruction.md").write_text("do it\n")
    (task_dir / "tests").mkdir(exist_ok=True)
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task_dir / "environment").mkdir(exist_ok=True)
    (task_dir / "environment" / "Dockerfile").write_text("FROM scratch\n")
    return task_dir


def test_local_directory_returns_scenario_configs(tmp_path):
    make_scenario_layout(tmp_path / "ec2-small", "ec2-small")
    make_scenario_layout(tmp_path / "rds-mysql", "rds-mysql")
    cfg = AwsBenchDatasetConfig(scenarios_path=tmp_path)
    configs = cfg.get_scenario_configs()
    assert {c.name for c in configs} == {"ec2-small", "rds-mysql"}
    assert all(isinstance(c, ScenarioConfig) for c in configs)
    assert all(c.git_url is None for c in configs)


@pytest.mark.asyncio
async def test_get_scenarios_rejects_duplicate_manifest_names(tmp_path):
    """Two distinct dirs declaring the same [scenario].name fail loud, naming both.

    The manifest name is the run-wide key for account mapping and task->scenario
    resolution, so a silent last-wins collapse would bind work to the wrong
    scenario. get_scenarios rejects it even though the two dirs are distinct
    source dirs.
    """
    make_scenario_layout(tmp_path / "dir-a", "collision")
    make_scenario_layout(tmp_path / "dir-b", "collision")
    cfg = AwsBenchDatasetConfig(scenarios_path=tmp_path)
    with pytest.raises(DuplicateScenarioNameError, match="collision"):
        await cfg.get_scenarios()


def test_include_filter_keeps_matching(tmp_path):
    make_scenario_layout(tmp_path / "ec2-small", "ec2-small")
    make_scenario_layout(tmp_path / "rds-mysql", "rds-mysql")
    cfg = AwsBenchDatasetConfig(scenarios_path=tmp_path, include_scenario_names=["ec2*"])
    configs = cfg.get_scenario_configs()
    assert {c.name for c in configs} == {"ec2-small"}


def test_exclude_filter_drops_matching(tmp_path):
    make_scenario_layout(tmp_path / "ec2-small", "ec2-small")
    make_scenario_layout(tmp_path / "rds-mysql", "rds-mysql")
    cfg = AwsBenchDatasetConfig(scenarios_path=tmp_path, exclude_scenario_names=["ec2*"])
    configs = cfg.get_scenario_configs()
    assert {c.name for c in configs} == {"rds-mysql"}


def test_validation_error_when_both_local_and_registry(tmp_path):
    with pytest.raises(Exception) as exc:  # pydantic.ValidationError or ValueError
        AwsBenchDatasetConfig(scenarios_path=tmp_path, name="aws-bench-all")
    assert "not both" in str(exc.value).lower() or "both" in str(exc.value).lower()


def test_validation_error_when_neither_local_nor_registry():
    with pytest.raises(Exception) as exc:
        AwsBenchDatasetConfig()
    assert "no dataset source" in str(exc.value).lower() or "must be set" in str(exc.value).lower()


# ── get_tasks ──


@pytest.mark.asyncio
async def test_get_tasks_pairs_each_config_with_its_task(tmp_path):
    """get_tasks returns (config, AwsBenchTask) pairs, each built from its own config."""
    tasks_dir = tmp_path / "tasks"
    a = _make_task_layout(tasks_dir / "task-a")
    b = _make_task_layout(tasks_dir / "task-b")

    cfg = AwsBenchDatasetConfig(path=tasks_dir)
    pairs = await cfg.get_tasks()

    assert all(isinstance(t, AwsBenchTask) for _, t in pairs)
    # Each task is built from its OWN config (no positional mismatch).
    by_path = {tc.get_local_path(): task for tc, task in pairs}
    task_a = by_path[a].config.task
    task_b = by_path[b].config.task
    assert task_a is not None and task_b is not None
    assert task_a.name == "aws-bench/task-a"
    assert task_b.name == "aws-bench/task-b"


@pytest.mark.asyncio
async def test_get_tasks_dir_with_toml_but_no_instruction_fails_loud(tmp_path):
    """A dir with task.toml but no instruction.md fails loud (named), not silent-drop.

    The dir is an intended task (has task.toml) but is malformed: it ships no
    instruction.md, so it fails Harbor's _validate_tests at AwsBenchTask
    construction. The listing predicate lists every task.toml-bearing dir, so
    get_tasks's construction gate names it rather than silently dropping it.
    """
    tasks_dir = tmp_path / "tasks"
    _make_task_layout(tasks_dir / "good-task")
    broken = tasks_dir / "broken-task"
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "task.toml").write_text(
        '[task]\nname = "aws-bench/broken-task"\n\n[scenario]\nscenario_id = "ec2-small"\n'
    )
    (broken / "environment").mkdir(exist_ok=True)
    (broken / "environment" / "Dockerfile").write_text("FROM scratch\n")
    # No instruction.md and no tests/ → fails Harbor's _validate_tests.

    cfg = AwsBenchDatasetConfig(path=tasks_dir)
    with pytest.raises(TaskConfigInvalidError, match="broken-task"):
        await cfg.get_tasks()


@pytest.mark.asyncio
async def test_get_tasks_skips_non_task_dirs(tmp_path):
    """Dirs/files without task.toml are not tasks: skipped, not errors."""
    tasks_dir = tmp_path / "tasks"
    good = _make_task_layout(tasks_dir / "good-task")
    # A sibling dir with no task.toml (e.g. docs): not a task, must be skipped.
    readme_dir = tasks_dir / "docs"
    readme_dir.mkdir(parents=True, exist_ok=True)
    (readme_dir / "README.md").write_text("# notes\n")
    # A stray loose file directly in the tasks dir: not a task, must be skipped.
    (tasks_dir / "NOTES.txt").write_text("stray\n")

    cfg = AwsBenchDatasetConfig(path=tasks_dir)
    pairs = await cfg.get_tasks()

    assert [tc.get_local_path() for tc, _ in pairs] == [good]


@pytest.mark.asyncio
async def test_local_task_configs_sorted_and_stable(tmp_path):
    """Local resolution is sorted: two get_task_configs() calls agree in order."""
    tasks_dir = tmp_path / "tasks"
    for name in ("zebra", "alpha", "mango", "beta"):
        _make_task_layout(tasks_dir / name)

    cfg = AwsBenchDatasetConfig(path=tasks_dir)
    first = [tc.get_local_path().name for tc in await cfg.get_task_configs()]
    second = [tc.get_local_path().name for tc in await cfg.get_task_configs()]

    assert first == second
    assert first == sorted(first)


@pytest.mark.asyncio
async def test_local_task_configs_n_tasks_same_subset(tmp_path):
    """Under n_tasks the sorted listing yields a stable prefix subset."""
    tasks_dir = tmp_path / "tasks"
    for name in ("zebra", "alpha", "mango", "beta"):
        _make_task_layout(tasks_dir / name)

    cfg = AwsBenchDatasetConfig(path=tasks_dir, n_tasks=2)
    first = {tc.get_local_path().name for tc in await cfg.get_task_configs()}
    second = {tc.get_local_path().name for tc in await cfg.get_task_configs()}

    assert first == second
    # Sorted prefix: the two lexicographically-first task names.
    assert first == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_local_task_configs_name_filter_same_subset(tmp_path):
    """A name filter selects the same subset across repeated calls."""
    tasks_dir = tmp_path / "tasks"
    for name in ("zebra", "alpha", "mango", "beta"):
        _make_task_layout(tasks_dir / name)

    # task_names filters on the task id name (the local dir basename here).
    cfg = AwsBenchDatasetConfig(path=tasks_dir, task_names=["alpha", "beta"])
    first = {tc.get_local_path().name for tc in await cfg.get_task_configs()}
    second = {tc.get_local_path().name for tc in await cfg.get_task_configs()}

    assert first == second
    assert first == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_get_tasks_lists_tasks_grouped_by_scenario(tmp_path):
    """A <tasks>/<scenario>/<task> layout yields every task, across scenarios.

    Dataset repos keep tasks grouped per scenario. Listing them in one run is
    what lets a single run cover several scenarios, as a registry dataset does.
    """
    tasks_dir = tmp_path / "tasks"
    a = _make_task_layout(tasks_dir / "ec2-small" / "task-a", scenario_id="ec2-small")
    b = _make_task_layout(tasks_dir / "ec2-small" / "task-b", scenario_id="ec2-small")
    c = _make_task_layout(tasks_dir / "rds-mysql" / "task-c", scenario_id="rds-mysql")

    cfg = AwsBenchDatasetConfig(path=tasks_dir)
    pairs = await cfg.get_tasks()

    assert [tc.get_local_path() for tc, _ in pairs] == [a, b, c]
    assert {task.config.scenario.scenario_id for _, task in pairs} == {
        "ec2-small",
        "rds-mysql",
    }


@pytest.mark.asyncio
async def test_get_tasks_mixes_flat_and_grouped_task_dirs(tmp_path):
    """A flat task dir and a grouped one coexist; a task dir is not descended into."""
    tasks_dir = tmp_path / "tasks"
    flat = _make_task_layout(tasks_dir / "flat-task")
    # A nested dir under a task dir is part of that task, never a task itself.
    _make_task_layout(tasks_dir / "flat-task" / "not-a-task")
    grouped = _make_task_layout(tasks_dir / "ec2-small" / "grouped-task")

    cfg = AwsBenchDatasetConfig(path=tasks_dir)
    pairs = await cfg.get_tasks()

    assert [tc.get_local_path() for tc, _ in pairs] == [grouped, flat]
