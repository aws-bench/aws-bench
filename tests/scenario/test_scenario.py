"""Tests for the Scenario class."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from aws_bench.scenario.events import ScenarioPhase
from aws_bench.scenario.scenario import Scenario, ScenarioRegionMap

VALID_TOML = """\
schema_version = "1.0"

[scenario]
name = "scenario-test"
account_tags = ["PRIMARY"]
regions = ["us-east-1"]
"""


def _make_layout(
    root: Path,
    *,
    config: str | None = VALID_TOML,
    dockerfile: bool = True,
    phase_scripts: tuple[ScenarioPhase, ...] = (ScenarioPhase.DEPLOY,),
) -> Path:
    """Build a scenario directory; materialize a ``<phase>/<phase>.sh`` per entry.

    Defaults are a valid scenario: scenario.toml + Dockerfile + deploy.sh.
    """
    root.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (root / "scenario.toml").write_text(config)
    if dockerfile:
        (root / "scenario").mkdir(exist_ok=True)
        (root / "scenario" / "Dockerfile").write_text("FROM alpine\n")
    for phase in phase_scripts:
        (root / phase).mkdir(exist_ok=True)
        (root / phase / f"{phase}.sh").write_text("#!/bin/sh\n")
    return root


def test_scenario_constructs_from_valid_dir(tmp_path):
    d = _make_layout(tmp_path / "s")
    scenario = Scenario(d)
    assert scenario.name == "scenario-test"
    assert scenario.scenario_dir == d.resolve()
    assert scenario.manifest.scenario.name == "scenario-test"
    assert scenario.paths.config_path == d.resolve() / "scenario.toml"


def test_scenario_raises_on_missing_dockerfile(tmp_path):
    d = _make_layout(tmp_path / "s", dockerfile=False)
    with pytest.raises(FileNotFoundError, match="Dockerfile"):
        Scenario(d)


def test_scenario_raises_on_missing_deploy_script(tmp_path):
    d = _make_layout(tmp_path / "s", phase_scripts=())
    with pytest.raises(FileNotFoundError, match="deploy.sh"):
        Scenario(d)


def test_scenario_accepts_phase_dir_without_script(tmp_path):
    """A phase dir with no entry script is valid; that phase is simply skipped.

    The script file on disk is the run gate (``has_phase_script``), so a bare
    verify/ dir with no verify.sh is a benign no-op rather than an error.
    """
    d = _make_layout(tmp_path / "s")
    (d / "verify").mkdir()  # dir but no verify.sh
    scenario = Scenario(d)  # must not raise
    assert scenario.has_phase_script(ScenarioPhase.VERIFY) is False


def test_scenario_accepts_optional_phases(tmp_path):
    d = _make_layout(
        tmp_path / "s",
        phase_scripts=(ScenarioPhase.DEPLOY, ScenarioPhase.VERIFY, ScenarioPhase.CLEANUP),
    )
    scenario = Scenario(d)
    assert scenario.has_phase_script(ScenarioPhase.VERIFY)
    assert scenario.has_phase_script(ScenarioPhase.CLEANUP)


def test_is_valid_dir_returns_true_for_valid_layout(tmp_path):
    d = _make_layout(tmp_path / "s")
    assert Scenario.is_valid_dir(d) is True


def test_is_valid_dir_returns_true_for_bare_scenario_toml(tmp_path):
    """A dir with ONLY scenario.toml (no Dockerfile/deploy.sh) STILL lists.

    Structural-only: a malformed-but-intended scenario must surface so it can
    fail loud at construction, NOT be silently filtered. The missing Dockerfile
    is caught by Scenario(...), not by is_valid_dir.
    """
    d = tmp_path / "s"
    d.mkdir()
    (d / "scenario.toml").write_text(VALID_TOML)
    assert Scenario.is_valid_dir(d) is True


def test_is_valid_dir_returns_false_without_scenario_toml(tmp_path):
    """A dir with no scenario.toml is not an intended scenario → False."""
    d = _make_layout(tmp_path / "s", config=None)
    assert Scenario.is_valid_dir(d) is False


def test_scenario_constructor_still_raises_on_missing_dockerfile(tmp_path):
    """is_valid_dir lists a Dockerfile-less dir; construction fails loud."""
    d = _make_layout(tmp_path / "s", dockerfile=False)
    with pytest.raises(FileNotFoundError, match="Dockerfile"):
        Scenario(d)


def test_is_valid_dir_returns_true_for_malformed_toml(tmp_path):
    """Structural-only check passes for malformed TOML.

    Content errors surface at Scenario(...) construction so they aggregate
    via ScenarioDiscoveryError instead of being silently filtered out.
    """
    d = _make_layout(tmp_path / "s", config="this is not [valid toml\n")
    assert Scenario.is_valid_dir(d) is True


def test_is_valid_dir_returns_true_for_schema_violation(tmp_path):
    """Structural-only check passes for schema violations.

    Content errors surface at Scenario(...) construction.
    """
    bad = """\
schema_version = "1.0"
[scenario]
name = "bad"
account_tags = ["A", "A"]
regions = ["us-east-1"]
"""
    d = _make_layout(tmp_path / "s", config=bad)
    assert Scenario.is_valid_dir(d) is True


def test_scenario_constructor_still_raises_on_malformed_toml(tmp_path):
    """Pair structural-only is_valid_dir with strict construction.

    Scenario(...) must surface TOML parse failures.
    """
    import tomllib

    d = _make_layout(tmp_path / "s", config="this is not [valid toml\n")
    with pytest.raises(tomllib.TOMLDecodeError):
        Scenario(d)


def test_scenario_constructor_still_raises_on_schema_violation(tmp_path):
    """Scenario(...) must surface schema failures."""
    from pydantic import ValidationError

    bad = """\
schema_version = "1.0"
[scenario]
name = "bad"
account_tags = ["A", "A"]
regions = ["us-east-1"]
"""
    d = _make_layout(tmp_path / "s", config=bad)
    with pytest.raises(ValidationError):
        Scenario(d)


def test_is_valid_dir_returns_false_for_nonexistent_path(tmp_path):
    assert Scenario.is_valid_dir(tmp_path / "does-not-exist") is False


# ── ScenarioRegionMap ──


def _fake_scenario(name: str, regions: list[str]) -> Scenario:
    s = MagicMock()
    s.name = name
    s.manifest.scenario.regions = regions
    return cast("Scenario", s)


def test_scenario_region_map_from_scenarios_builds_and_looks_up():
    """from_scenarios keys by name; regions_for returns the declared list."""
    region_map = ScenarioRegionMap.from_scenarios(
        [
            _fake_scenario("scn-a", ["us-east-1", "us-west-2"]),
            _fake_scenario("scn-b", ["eu-west-1"]),
        ]
    )
    assert region_map.regions_for("scn-a") == ["us-east-1", "us-west-2"]
    assert region_map.regions_for("scn-b") == ["eu-west-1"]


def test_scenario_region_map_regions_for_unknown_raises_with_name():
    """regions_for on an unknown scenario raises KeyError naming the scenario."""
    region_map = ScenarioRegionMap.from_scenarios([_fake_scenario("scn-a", ["us-east-1"])])
    with pytest.raises(KeyError, match="scn-missing"):
        region_map.regions_for("scn-missing")


# ── Provenance (git_url / git_commit_id) ──
