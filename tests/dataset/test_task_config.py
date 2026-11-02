"""Tests for the typed aws-bench task-model layer.

Covers ``AwsBenchTaskConfig`` (single-pass parse of the aws-bench task.toml
sections) and ``AwsBenchTask`` (Harbor ``Task`` + aws-bench construction-time
validation + phase-script discovery).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from aws_bench.dataset.models import RoleType, ScriptType
from aws_bench.dataset.task_config import (
    AwsBenchTask,
    AwsBenchTaskConfig,
    ConcurrencyConfig,
    ConcurrencyMode,
    PhaseScript,
    ScenarioRef,
)

# A task.toml that exercises every aws-bench section in one document.
FULL_TOML = """\
schema_version = "1.2"

[task]
name = "aws-bench/example"

[environment]
docker_image = "public.ecr.aws/example:latest"

[scenario]
scenario_id = "ec2-small"
agent_role_name = "AgentRole"
verifier_role_name = "VerifierRole"
pre_invoke_role_name = "PreRole"
post_invoke_role_name = "PostRole"

[concurrency]
mode = "read-only"

[pre_invoke]
timeout_sec = 42.0
env = { FOO = "bar" }

[post_invoke]
timeout_sec = 7.0
"""


def _write_task_layout(
    task_dir: Path,
    *,
    toml: str,
    with_dockerfile: bool = True,
    docker_image: bool = False,
    phase_scripts: tuple[ScriptType, ...] = (),
) -> Path:
    """Build a task directory that satisfies Harbor's instruction + test gate.

    Writes ``task.toml`` (caller-supplied), ``instruction.md``, ``tests/test.sh``
    and (by default) ``environment/Dockerfile`` so ``Task.__init__`` and
    ``AwsBenchTask._validate_layout`` both pass. ``with_dockerfile=False`` plus
    ``docker_image=False`` produces a task with no environment definition.
    Each entry in ``phase_scripts`` materializes ``<type>/<type>.sh``.
    """
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(toml)
    (task_dir / "instruction.md").write_text("do the thing\n")
    (task_dir / "tests").mkdir(exist_ok=True)
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    if with_dockerfile:
        (env_dir / "Dockerfile").write_text("FROM scratch\n")
    for st in phase_scripts:
        phase_dir = task_dir / st
        phase_dir.mkdir(exist_ok=True)
        (phase_dir / f"{st}.sh").write_text("#!/bin/sh\nexit 0\n")
    return task_dir


# ── AwsBenchTaskConfig (single-pass parse) ──


def _parse(toml: str) -> AwsBenchTaskConfig:
    """Parse and return as the subtype.

    ``model_validate_toml`` is cls-bound (returns AwsBenchTaskConfig at runtime),
    but its declared return type is the base TaskConfig; cast so the type checker
    sees the nested aws-bench fields.
    """
    return cast(AwsBenchTaskConfig, AwsBenchTaskConfig.model_validate_toml(toml))


def test_config_parses_base_and_nested_sections_in_one_pass():
    """model_validate_toml populates base fields AND nested aws-bench sections."""
    cfg = _parse(FULL_TOML)

    # Base Harbor fields survive on the subtype.
    assert cfg.task is not None
    assert cfg.task.name == "aws-bench/example"
    assert cfg.environment.docker_image == "public.ecr.aws/example:latest"

    # Nested aws-bench sections populate as typed models.
    assert isinstance(cfg.scenario, ScenarioRef)
    assert cfg.scenario.scenario_id == "ec2-small"
    assert isinstance(cfg.concurrency, ConcurrencyConfig)
    assert isinstance(cfg.pre_invoke, PhaseScript)
    assert isinstance(cfg.post_invoke, PhaseScript)


def test_config_pre_invoke_timeout_lands_on_phase_model():
    """[pre_invoke] timeout_sec parses onto cfg.pre_invoke."""
    cfg = _parse(FULL_TOML)
    assert cfg.pre_invoke is not None
    assert cfg.pre_invoke.timeout_sec == 42.0
    assert cfg.pre_invoke.env == {"FOO": "bar"}


def test_config_concurrency_read_only():
    """Mode "read-only" parses to READ_ONLY."""
    cfg = _parse(FULL_TOML)
    assert cfg.concurrency.mode is ConcurrencyMode.READ_ONLY


def test_config_concurrency_defaults_to_mutating_when_absent():
    """A task.toml with no [concurrency] section defaults to MUTATING."""
    cfg = _parse('[scenario]\nscenario_id = "ec2-small"\n')
    assert cfg.concurrency.mode is ConcurrencyMode.MUTATING


def test_config_concurrency_bogus_mode_raises():
    """An unknown concurrency mode is a ValidationError."""
    with pytest.raises(ValidationError):
        AwsBenchTaskConfig.model_validate_toml(
            '[scenario]\nscenario_id = "ec2-small"\n\n[concurrency]\nmode = "bogus"\n'
        )


def test_config_missing_scenario_id_raises():
    """[scenario] is required; a bare scenario table without scenario_id fails."""
    with pytest.raises(ValidationError):
        AwsBenchTaskConfig.model_validate_toml("[scenario]\n")


def test_config_missing_scenario_section_raises():
    """No [scenario] section at all is a ValidationError (field is required)."""
    with pytest.raises(ValidationError):
        AwsBenchTaskConfig.model_validate_toml('[task]\nname = "aws-bench/x"\n')


# ── ScenarioRef.role_name ──


@pytest.mark.parametrize(
    "role_type,expected",
    [
        (RoleType.AGENT, "AgentRole"),
        (RoleType.VERIFIER, "VerifierRole"),
        (RoleType.PRE_INVOKE, "PreRole"),
        (RoleType.POST_INVOKE, "PostRole"),
    ],
)
def test_scenario_ref_role_name(role_type, expected):
    ref = ScenarioRef(
        scenario_id="ec2-small",
        agent_role_name="AgentRole",
        verifier_role_name="VerifierRole",
        pre_invoke_role_name="PreRole",
        post_invoke_role_name="PostRole",
    )
    assert ref.role_name(role_type) == expected


def test_scenario_ref_role_name_none_when_unset():
    ref = ScenarioRef(scenario_id="ec2-small")
    assert all(ref.role_name(rt) is None for rt in RoleType)


def test_scenario_ref_agent_account_tags_reject_empty():
    with pytest.raises(ValidationError):
        ScenarioRef(scenario_id="ec2-small", agent_account_tags=[])


# ── AwsBenchTask construction ──


def test_task_carries_awsbench_task_config(tmp_path):
    """A constructed task's config is an AwsBenchTaskConfig with nested fields."""
    # FULL_TOML declares [pre_invoke]/[post_invoke], so their scripts must exist.
    task_dir = _write_task_layout(
        tmp_path / "task",
        toml=FULL_TOML,
        phase_scripts=(ScriptType.PRE_INVOKE, ScriptType.POST_INVOKE),
    )
    task = AwsBenchTask(task_dir)
    assert isinstance(task.config, AwsBenchTaskConfig)
    assert task.config.scenario.scenario_id == "ec2-small"
    assert task.config.concurrency.mode is ConcurrencyMode.READ_ONLY


def test_task_has_phase_script_finds_sh(tmp_path):
    """has_phase_script/phase_script_path locate an existing pre_invoke.sh."""
    task_dir = _write_task_layout(
        tmp_path / "task",
        toml=FULL_TOML,
        phase_scripts=(ScriptType.PRE_INVOKE, ScriptType.POST_INVOKE),
    )
    task = AwsBenchTask(task_dir)
    assert task.has_phase_script(ScriptType.PRE_INVOKE) is True
    path = task.phase_script_path(ScriptType.PRE_INVOKE)
    assert path == task_dir / "pre_invoke" / "pre_invoke.sh"


def test_task_phase_script_absent_returns_none(tmp_path):
    """phase_script_path returns None and has_phase_script False when absent."""
    # SETUP has no declared section and no file → absent is fine.
    task_dir = _write_task_layout(tmp_path / "task", toml=FULL_TOML, phase_scripts=())
    # Drop the declared pre/post sections so the missing files are not an error.
    minimal = '[task]\nname = "aws-bench/x"\n\n[scenario]\nscenario_id = "ec2-small"\n'
    (task_dir / "task.toml").write_text(minimal)
    task = AwsBenchTask(task_dir)
    assert task.phase_script_path(ScriptType.PRE_INVOKE) is None
    assert task.has_phase_script(ScriptType.PRE_INVOKE) is False


def test_task_declared_phase_without_file_is_valid(tmp_path):
    """A [pre_invoke] section without a script file constructs; the phase just won't run.

    The script file on disk is the run gate (``has_phase_script``), so a declared
    section with no script is a benign no-op rather than an error.
    """
    task_dir = _write_task_layout(tmp_path / "task", toml=FULL_TOML, phase_scripts=())
    task = AwsBenchTask(task_dir)  # must not raise
    assert task.has_phase_script(ScriptType.PRE_INVOKE) is False


def test_task_empty_phase_dir_without_section_is_valid(tmp_path):
    """A bare pre_invoke/ dir with NO [pre_invoke] section is valid."""
    minimal = '[task]\nname = "aws-bench/x"\n\n[scenario]\nscenario_id = "ec2-small"\n'
    task_dir = _write_task_layout(tmp_path / "task", toml=minimal, phase_scripts=())
    (task_dir / "pre_invoke").mkdir()  # empty dir, no section declared
    task = AwsBenchTask(task_dir)  # must not raise
    assert task.has_phase_script(ScriptType.PRE_INVOKE) is False


def test_task_missing_environment_definition_raises(tmp_path):
    """environment/ with no Dockerfile/compose and no docker_image → error."""
    minimal = '[task]\nname = "aws-bench/x"\n\n[scenario]\nscenario_id = "ec2-small"\n'
    task_dir = _write_task_layout(tmp_path / "task", toml=minimal, with_dockerfile=False)
    with pytest.raises(FileNotFoundError, match="environment"):
        AwsBenchTask(task_dir)


def test_task_docker_image_only_environment_is_valid(tmp_path):
    """[environment].docker_image with no Dockerfile is a valid environment."""
    image_only = (
        '[task]\nname = "aws-bench/x"\n\n'
        '[environment]\ndocker_image = "public.ecr.aws/x:1"\n\n'
        '[scenario]\nscenario_id = "ec2-small"\n'
    )
    task_dir = _write_task_layout(tmp_path / "task", toml=image_only, with_dockerfile=False)
    task = AwsBenchTask(task_dir)  # must not raise
    assert task.config.environment.docker_image == "public.ecr.aws/x:1"


def test_task_separate_verifier_mode_raises(tmp_path):
    """A separate-environment verifier is refused at construction.

    aws-bench verifies AWS account state in the agent container (shared mode); a
    separate verifier environment tears that container down before verification.
    """
    separate = (
        '[task]\nname = "aws-bench/x"\n\n'
        '[scenario]\nscenario_id = "ec2-small"\n\n'
        '[verifier]\nenvironment_mode = "separate"\n'
    )
    task_dir = _write_task_layout(tmp_path / "task", toml=separate)
    with pytest.raises(ValueError, match="separate-environment verifiers are not supported"):
        AwsBenchTask(task_dir)


# ── AwsBenchTask.is_valid_dir (structural-only discovery filter) ──


def test_is_valid_dir_true_for_task_toml_dir(tmp_path):
    """A fully-built task dir holds a task.toml → True."""
    task_dir = _write_task_layout(tmp_path / "task", toml=FULL_TOML)
    assert AwsBenchTask.is_valid_dir(task_dir) is True


def test_is_valid_dir_true_for_bare_task_toml(tmp_path):
    """A dir with ONLY task.toml (no instruction/test/env) STILL lists.

    Structural-only: a malformed-but-intended task must surface so it can fail
    loud at construction, NOT be silently filtered. This is the assertion that
    distinguishes this override from Harbor's thorough ``Task.is_valid_dir``.
    """
    task_dir = tmp_path / "bare"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text('[scenario]\nscenario_id = "ec2-small"\n')
    assert AwsBenchTask.is_valid_dir(task_dir) is True


def test_is_valid_dir_false_without_task_toml(tmp_path):
    """A dir with a README but no task.toml is not an intended task → False."""
    not_a_task = tmp_path / "docs"
    not_a_task.mkdir()
    (not_a_task / "README.md").write_text("not a task\n")
    assert AwsBenchTask.is_valid_dir(not_a_task) is False


def test_is_valid_dir_false_for_loose_file(tmp_path):
    """A path to a loose file (not a dir) has no task.toml under it → False."""
    loose = tmp_path / "loose.txt"
    loose.write_text("stray\n")
    assert AwsBenchTask.is_valid_dir(loose) is False


def test_scenario_ref_rejects_unknown_keys():
    """A misspelled key must fail, not silently give the agent every account."""
    with pytest.raises(ValidationError, match="agent_account_tag"):
        ScenarioRef.model_validate({"scenario_id": "ec2-small", "agent_account_tag": ["PRIMARY"]})
