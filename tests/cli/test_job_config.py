"""AwsBenchJobConfig: run-command job config (subclass of harbor JobConfig)."""

from pathlib import Path

import pytest
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import TaskConfig
from pydantic import ValidationError

from aws_bench.cli.job_config import AwsBenchJobConfig
from aws_bench.dataset.config import AwsBenchDatasetConfig


def test_rejects_authored_tasks_key():
    with pytest.raises(ValidationError, match="tasks"):
        AwsBenchJobConfig.model_validate({"tasks": [{"path": "tasks/t", "source": "x"}]})


def test_rejects_authored_datasets_key():
    with pytest.raises(ValidationError, match="dataset"):
        AwsBenchJobConfig.model_validate({"datasets": [{"name": "x"}]})


def test_dataset_deserializes_as_awsbench_dataset_config():
    cfg = AwsBenchJobConfig.model_validate({"dataset": {"name": "my-dataset", "version": "1.0.0"}})
    assert isinstance(cfg.dataset, AwsBenchDatasetConfig)


def test_dataset_defaults_none():
    assert AwsBenchJobConfig().dataset is None


def test_rejects_non_docker_environment_type():
    with pytest.raises(ValidationError, match="[Dd]ocker"):
        AwsBenchJobConfig.model_validate({"environment": {"type": "daytona"}})


def test_rejects_environment_import_path():
    with pytest.raises(ValidationError, match="import_path"):
        AwsBenchJobConfig.model_validate({"environment": {"import_path": "pkg.mod:Env"}})


def test_rejects_install_only():
    # Harbor 0.22.0 disables the verifier for install_only, but the aws-bench
    # trial config never receives the flag, so the agent would run in full unverified.
    with pytest.raises(ValidationError, match="install_only"):
        AwsBenchJobConfig.model_validate({"install_only": True})


def test_install_only_off_is_valid():
    assert AwsBenchJobConfig.model_validate({}).install_only is False
    assert AwsBenchJobConfig.model_validate({"install_only": False}).install_only is False


def test_full_run_config_still_validates():
    cfg = AwsBenchJobConfig(
        dataset=AwsBenchDatasetConfig(name="aws-bench-all"), env_name="ou", metrics=[]
    )
    assert cfg.install_only is False


def test_docker_environment_is_valid():
    cfg = AwsBenchJobConfig.model_validate({"environment": {"type": "docker"}})
    assert cfg.environment.type is not None


def test_environment_type_none_is_valid():
    cfg = AwsBenchJobConfig()
    assert cfg.environment.type is not None  # harbor validator -> DOCKER


def test_n_concurrent_trials_defaults_to_four():
    assert AwsBenchJobConfig().n_concurrent_trials == 4


def test_accepts_concurrent_trials():
    # The per-scenario admission gate keeps same-scenario mutating trials from
    # overlapping, so a value above 1 is valid.
    assert AwsBenchJobConfig(n_concurrent_trials=8).n_concurrent_trials == 8


def test_env_name_persisted_in_dump():
    cfg = AwsBenchJobConfig(env_name="my-ou")
    assert cfg.model_dump()["env_name"] == "my-ou"


def test_verify_defaults_on_and_persisted():
    cfg = AwsBenchJobConfig()
    assert cfg.verify is True
    assert cfg.model_dump()["verify"] is True


def test_env_name_is_part_of_resume_identity():
    # A different target environment must refuse resume.
    assert AwsBenchJobConfig(env_name="ou-a") != AwsBenchJobConfig(env_name="ou-b")


def test_verify_excluded_from_resume_identity():
    # Toggling verification is operational, not a different run.
    assert AwsBenchJobConfig(verify=True) == AwsBenchJobConfig(verify=False)


def test_test_environment_defaults_none():
    assert AwsBenchJobConfig().test_environment is None


def test_resume_round_trip_equal_under_base_jobconfig():
    cfg = AwsBenchJobConfig()
    cfg.tasks = [TaskConfig(path=Path("tasks/t"), source="ds")]
    dumped = cfg.model_dump_json()
    assert JobConfig.model_validate_json(dumped) == JobConfig.model_validate_json(dumped)


def test_plain_tasks_assignment_does_not_trigger_validator():
    cfg = AwsBenchJobConfig()
    cfg.tasks = [TaskConfig(path=Path("tasks/t"), source="ds")]  # no raise
    assert len(cfg.tasks) == 1
