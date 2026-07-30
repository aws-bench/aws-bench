"""Tests for AwsBenchTrialConfig — the per-trial AWS context on the trial config."""

from __future__ import annotations

from pathlib import Path

from harbor.models.trial.config import TaskConfig

from aws_bench.dataset.task_config import ConcurrencyMode
from aws_bench.scenario.locator import ScenarioConfig
from aws_bench.task.trial_config import AwsBenchTrialConfig


def _task_config() -> TaskConfig:
    return TaskConfig(path=Path("tasks/example"))


def _scenario_desc(name: str = "ec2-small") -> ScenarioConfig:
    return ScenarioConfig(name=name, path=Path(f"scenarios/{name}"))


def test_carries_scenario_descriptor():
    cfg = AwsBenchTrialConfig(
        task=_task_config(),
        scenario=_scenario_desc(),
        scenario_id="ec2-small",
        concurrency_mode=ConcurrencyMode.MUTATING,
        regions=["us-east-1"],
        account_mapping={"PRIMARY": "123456789012"},
    )
    assert cfg.scenario.name == "ec2-small"
    assert cfg.scenario.path == Path("scenarios/ec2-small")


def test_scenario_descriptor_rides_resume_identity():
    """A differing scenario descriptor makes two configs unequal (forces re-run)."""
    base = AwsBenchTrialConfig(
        task=_task_config(),
        scenario=_scenario_desc("ec2-small"),
        scenario_id="ec2-small",
        concurrency_mode=ConcurrencyMode.MUTATING,
        regions=["us-east-1"],
        account_mapping={"PRIMARY": "111111111111"},
    )
    diff_scenario = AwsBenchTrialConfig(
        task=_task_config(),
        scenario=_scenario_desc("ec2-large"),
        scenario_id="ec2-small",
        concurrency_mode=ConcurrencyMode.MUTATING,
        regions=["us-east-1"],
        account_mapping={"PRIMARY": "111111111111"},
    )
    assert base != diff_scenario


def test_carries_scenario_account_mapping_and_exports():
    cfg = AwsBenchTrialConfig(
        task=_task_config(),
        scenario=_scenario_desc("ec2-small"),
        scenario_id="ec2-small",
        concurrency_mode=ConcurrencyMode.MUTATING,
        regions=["us-east-1"],
        account_mapping={"PRIMARY": "123456789012"},
        exports={"PRIMARY": {"BucketName": "my-bucket"}},
    )
    assert cfg.scenario_id == "ec2-small"
    assert cfg.concurrency_mode is ConcurrencyMode.MUTATING
    assert cfg.account_mapping == {"PRIMARY": "123456789012"}
    assert cfg.exports == {"PRIMARY": {"BucketName": "my-bucket"}}


def test_exports_defaults_to_empty():
    cfg = AwsBenchTrialConfig(
        task=_task_config(),
        scenario=_scenario_desc("ec2-small"),
        scenario_id="ec2-small",
        concurrency_mode=ConcurrencyMode.MUTATING,
        regions=["us-east-1"],
        account_mapping={"PRIMARY": "123456789012"},
    )
    assert cfg.exports == {}


def test_exports_field_is_tag_keyed():
    from aws_bench.task.trial_config import AwsBenchTrialConfig

    field = AwsBenchTrialConfig.model_fields["exports"]
    assert field.annotation == dict[str, dict[str, str]]
    assert field.default_factory is dict


def test_exports_is_excluded_from_serialization():
    """Exports stays in memory (sensitive values) but never reaches a serialized form."""
    cfg = AwsBenchTrialConfig(
        task=_task_config(),
        scenario=_scenario_desc("ec2-small"),
        scenario_id="ec2-small",
        concurrency_mode=ConcurrencyMode.MUTATING,
        regions=["us-east-1"],
        account_mapping={"PRIMARY": "123456789012"},
        exports={"PRIMARY": {"SecretishExport": "s3cr3t-value"}},
    )
    assert cfg.exports == {"PRIMARY": {"SecretishExport": "s3cr3t-value"}}
    assert "exports" not in cfg.model_dump()
    assert "exports" not in cfg.model_dump(mode="json")
    assert "s3cr3t-value" not in cfg.model_dump_json()


def test_scenario_and_account_ride_inherited_eq_but_exports_do_not():
    """scenario_id + account_mapping ride TrialConfig.__eq__; exports is excluded.

    ``exports`` is ``exclude=True``, so it is absent from the ``model_dump`` the
    inherited ``__eq__`` compares: two trials that differ only by export value are
    equal (so resume reuses the trial rather than re-running on an export change),
    while a scenario or account change still forces a re-run.
    """
    base = AwsBenchTrialConfig(
        task=_task_config(),
        scenario=_scenario_desc("ec2-small"),
        scenario_id="ec2-small",
        concurrency_mode=ConcurrencyMode.MUTATING,
        regions=["us-east-1"],
        account_mapping={"PRIMARY": "111111111111"},
        exports={"PRIMARY": {"K": "v1"}},
    )
    same = AwsBenchTrialConfig(
        task=_task_config(),
        scenario=_scenario_desc("ec2-small"),
        scenario_id="ec2-small",
        concurrency_mode=ConcurrencyMode.MUTATING,
        regions=["us-east-1"],
        account_mapping={"PRIMARY": "111111111111"},
        exports={"PRIMARY": {"K": "v1"}},
    )
    diff_account = AwsBenchTrialConfig(
        task=_task_config(),
        scenario=_scenario_desc("ec2-small"),
        scenario_id="ec2-small",
        concurrency_mode=ConcurrencyMode.MUTATING,
        regions=["us-east-1"],
        account_mapping={"PRIMARY": "222222222222"},
        exports={"PRIMARY": {"K": "v1"}},
    )
    diff_exports = AwsBenchTrialConfig(
        task=_task_config(),
        scenario=_scenario_desc("ec2-small"),
        scenario_id="ec2-small",
        concurrency_mode=ConcurrencyMode.MUTATING,
        regions=["us-east-1"],
        account_mapping={"PRIMARY": "111111111111"},
        exports={"PRIMARY": {"K": "v2"}},
    )
    assert base == same
    assert base != diff_account
    # Differs only by export value -> still equal, because exports is excluded.
    assert base == diff_exports
