"""Tests for aws_bench.scenario.config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aws_bench.scenario.config import (
    EnvironmentConfig,
    ScenarioManifest,
    TrialEnvironmentConfig,
)
from aws_bench.scenario.events import ScenarioPhase


def _minimal_data(**overrides):
    base = {
        "schema_version": "1.0",
        "scenario": {
            "name": "lambda-broken-env-vars",
            "account_tags": ["PRIMARY"],
            "regions": ["us-east-1"],
        },
    }
    base.update(overrides)
    return base


class TestScenarioManifest:
    def test_minimal_parses(self):
        cfg = ScenarioManifest.model_validate(_minimal_data())
        assert cfg.scenario.name == "lambda-broken-env-vars"
        assert cfg.scenario.account_tags == ["PRIMARY"]
        # Defaults populated
        assert cfg.environment.cpus == 1
        assert cfg.environment.memory_mb == 2048
        assert cfg.deploy.timeout_sec == 600.0
        assert cfg.verify.timeout_sec == 120.0
        assert cfg.cleanup.timeout_sec == 300.0
        assert cfg.reset.timeout_sec == 600.0
        assert cfg.deploy.env == {}
        assert cfg.verify.env == {}
        assert cfg.cleanup.env == {}
        assert cfg.reset.env == {}
        assert cfg.quotas == []

    def test_manifest_exposes_every_phase_attribute(self):
        """Manifest must carry a PhaseConfig field per ScenarioPhase value.

        ``ScenarioTrial._execute`` does ``getattr(manifest, phase)`` for every
        phase, including RESET; a missing ``reset`` field made a scenario's
        reset.sh silently never run (the AttributeError was swallowed).
        """
        cfg = ScenarioManifest.model_validate(_minimal_data())
        for phase in (
            ScenarioPhase.DEPLOY,
            ScenarioPhase.VERIFY,
            ScenarioPhase.CLEANUP,
            ScenarioPhase.RESET,
        ):
            phase_cfg = getattr(cfg, phase)  # must not raise AttributeError
            assert phase_cfg.timeout_sec > 0

    def test_reset_phase_env_vars_parse(self):
        data = _minimal_data(reset={"timeout_sec": 1800.0, "env": {"DMS_CLEANUP": "1"}})
        cfg = ScenarioManifest.model_validate(data)
        assert cfg.reset.timeout_sec == 1800.0
        assert cfg.reset.env == {"DMS_CLEANUP": "1"}

    def test_phase_env_vars_parse(self):
        data = _minimal_data(
            deploy={
                "timeout_sec": 900.0,
                "env": {"CDK_DEFAULT_REGION": "us-east-1", "LOG_LEVEL": "DEBUG"},
            },
            verify={"env": {"VERIFY_DEEP": "1"}},
            cleanup={"env": {"FORCE": "true"}},
        )
        cfg = ScenarioManifest.model_validate(data)
        assert cfg.deploy.timeout_sec == 900.0
        assert cfg.deploy.env == {"CDK_DEFAULT_REGION": "us-east-1", "LOG_LEVEL": "DEBUG"}
        assert cfg.verify.env == {"VERIFY_DEEP": "1"}
        assert cfg.cleanup.env == {"FORCE": "true"}

    def test_full_example_from_lld_parses(self):
        data = {
            "schema_version": "1.0",
            "scenario": {
                "name": "lambda-broken-env-vars",
                "description": "Lambda with misconfigured environment variables",
                "authors": [{"name": "Author One", "email": "author@example.com"}],
                "keywords": ["lambda", "debugging"],
                "account_tags": ["PRIMARY"],
                "regions": ["us-east-1", "us-west-2"],
            },
            "environment": {
                "build_timeout_sec": 600.0,
                "cpus": 2,
                "memory_mb": 4096,
            },
            "deploy": {"timeout_sec": 600.0},
            "verify": {"timeout_sec": 120.0},
            "cleanup": {"timeout_sec": 300.0},
            "quotas": [
                {
                    "account_tag": "PRIMARY",
                    "region": "us-east-1",
                    "service_code": "lambda",
                    "quota_code": "L-B99A9384",
                    "desired_value": 1000,
                },
                {
                    "account_tag": "PRIMARY",
                    "region": "us-west-2",
                    "service_code": "vpc",
                    "quota_code": "L-F678F1CE",
                    "desired_value": 50,
                },
            ],
        }
        cfg = ScenarioManifest.model_validate(data)
        assert cfg.environment.cpus == 2
        assert cfg.environment.memory_mb == 4096
        assert len(cfg.quotas) == 2
        assert cfg.quotas[0].region == "us-east-1"
        assert cfg.quotas[1].service_code == "vpc"

    def test_account_tags_must_be_length_one(self):
        data = _minimal_data()
        data["scenario"]["account_tags"] = ["A", "B"]
        with pytest.raises(ValidationError) as exc:
            ScenarioManifest.model_validate(data)
        assert "account_tags must have exactly one entry" in str(exc.value)

    def test_account_tags_empty_list_rejected(self):
        data = _minimal_data()
        data["scenario"]["account_tags"] = []
        with pytest.raises(ValidationError):
            ScenarioManifest.model_validate(data)

    def test_account_tags_reject_duplicates(self):
        data = _minimal_data()
        data["scenario"]["account_tags"] = ["PRIMARY", "PRIMARY"]
        with pytest.raises(ValidationError) as exc:
            ScenarioManifest.model_validate(data)
        assert "account_tags must not contain duplicates" in str(exc.value)

    def test_account_tag_with_shell_metacharacters_rejected(self):
        """Tags become AWS profile names + heredoc interpolations — sanitize at parse."""
        data = _minimal_data()
        data["scenario"]["account_tags"] = ["BAD;rm -rf /"]
        with pytest.raises(ValidationError) as exc:
            ScenarioManifest.model_validate(data)
        assert "Invalid account_tag" in str(exc.value)

    def test_account_tag_with_newline_rejected(self):
        data = _minimal_data()
        data["scenario"]["account_tags"] = ["X\nAWSBENCH_EOF\nrm -rf /"]
        with pytest.raises(ValidationError):
            ScenarioManifest.model_validate(data)

    def test_account_tag_max_length_accepted(self):
        data = _minimal_data()
        data["scenario"]["account_tags"] = ["A" + "B" * 31]
        cfg = ScenarioManifest.model_validate(data)
        assert len(cfg.scenario.account_tags[0]) == 32

    def test_account_tag_over_length_rejected(self):
        data = _minimal_data()
        data["scenario"]["account_tags"] = ["A" + "B" * 32]
        with pytest.raises(ValidationError) as exc:
            ScenarioManifest.model_validate(data)
        assert "Invalid account_tag" in str(exc.value)

    def test_quotas_must_reference_known_tag(self):
        data = _minimal_data(
            quotas=[
                {
                    "account_tag": "SECONDARY",  # not declared
                    "region": "us-east-1",
                    "service_code": "lambda",
                    "quota_code": "L-X",
                    "desired_value": 1.0,
                }
            ]
        )
        with pytest.raises(ValidationError) as exc:
            ScenarioManifest.model_validate(data)
        # Pydantic error message points at the offending entry's index
        assert "quotas[0]" in str(exc.value)
        assert "SECONDARY" in str(exc.value)

    def test_account_tag_case_sensitive(self):
        # PRIMARY != primary — match must be exact, no case-folding.
        data = _minimal_data(
            quotas=[
                {
                    "account_tag": "primary",
                    "region": "us-east-1",
                    "service_code": "lambda",
                    "quota_code": "L-X",
                    "desired_value": 1.0,
                }
            ]
        )
        with pytest.raises(ValidationError):
            ScenarioManifest.model_validate(data)

    def test_regions_must_be_non_empty(self):
        data = _minimal_data()
        data["scenario"]["regions"] = []
        with pytest.raises(ValidationError) as exc:
            ScenarioManifest.model_validate(data)
        assert "scenario.regions must have at least one entry" in str(exc.value)

    def test_regions_reject_blank_entry(self):
        data = _minimal_data()
        data["scenario"]["regions"] = ["us-east-1", "  "]
        with pytest.raises(ValidationError) as exc:
            ScenarioManifest.model_validate(data)
        assert "scenario.regions entries must be non-empty" in str(exc.value)

    def test_regions_reject_duplicates(self):
        data = _minimal_data()
        data["scenario"]["regions"] = ["us-east-1", "us-east-1"]
        with pytest.raises(ValidationError) as exc:
            ScenarioManifest.model_validate(data)
        assert "scenario.regions must not contain duplicates" in str(exc.value)

    def test_regions_field_required(self):
        data = _minimal_data()
        del data["scenario"]["regions"]
        with pytest.raises(ValidationError):
            ScenarioManifest.model_validate(data)

    def test_quotas_must_reference_known_region(self):
        data = _minimal_data(
            quotas=[
                {
                    "account_tag": "PRIMARY",
                    "region": "eu-west-1",  # not in scenario.regions
                    "service_code": "lambda",
                    "quota_code": "L-X",
                    "desired_value": 1.0,
                }
            ]
        )
        with pytest.raises(ValidationError) as exc:
            ScenarioManifest.model_validate(data)
        assert "quotas[0]" in str(exc.value)
        assert "eu-west-1" in str(exc.value)

    def test_unknown_top_level_field_does_not_fail(self):
        # Pydantic's default ignores unknown keys; we don't restrict here.
        data = _minimal_data()
        data["future_extension"] = {"foo": "bar"}
        ScenarioManifest.model_validate(data)

    def test_model_validate_toml_accepts_str(self):
        toml = (
            'schema_version = "1.0"\n[scenario]\nname = "x"\n'
            'account_tags = ["PRIMARY"]\nregions = ["us-east-1"]\n'
        )
        cfg = ScenarioManifest.model_validate_toml(toml)
        assert cfg.scenario.name == "x"

    def test_model_validate_toml_accepts_utf8_bytes(self):
        """Bytes input is decoded as UTF-8, never the platform locale."""
        toml = (
            'schema_version = "1.0"\n[scenario]\nname = "lambda-é"\n'
            'account_tags = ["PRIMARY"]\nregions = ["us-east-1"]\n'
        ).encode("utf-8")
        cfg = ScenarioManifest.model_validate_toml(toml)
        assert cfg.scenario.name == "lambda-é"

    def test_model_validate_toml_rejects_non_utf8_bytes(self):
        """Non-UTF-8 bytes raise UnicodeDecodeError (a ValueError)."""
        with pytest.raises(UnicodeDecodeError):
            ScenarioManifest.model_validate_toml(b"\xff\xfe garbage")


class TestTrialEnvironmentConfigApplyTo:
    def test_no_overrides_leaves_author_env_untouched(self):
        author = EnvironmentConfig(cpus=2, memory_mb=4096)
        trial = TrialEnvironmentConfig()
        trial.apply_to(author)
        assert author.cpus == 2
        assert author.memory_mb == 4096

    def test_individual_overrides_take_effect(self):
        author = EnvironmentConfig(cpus=1, memory_mb=2048)
        trial = TrialEnvironmentConfig(
            override_cpus=4,
            override_memory_mb=8192,
            override_build_timeout_sec=900.0,
        )
        trial.apply_to(author)
        assert author.cpus == 4
        assert author.memory_mb == 8192
        assert author.build_timeout_sec == 900.0

    def test_partial_override_only_touches_specified_fields(self):
        author = EnvironmentConfig(cpus=2, memory_mb=4096)
        trial = TrialEnvironmentConfig(override_memory_mb=8192)
        trial.apply_to(author)
        assert author.cpus == 2  # untouched
        assert author.memory_mb == 8192  # overridden

    def test_mounts_json_override_replaces_author_mounts(self):
        author = EnvironmentConfig(mounts_json=[{"type": "bind", "source": "/a", "target": "/b"}])
        trial = TrialEnvironmentConfig(
            mounts_json=[{"type": "volume", "source": "vol", "target": "/data"}]
        )
        trial.apply_to(author)
        assert len(author.mounts_json) == 1
        assert author.mounts_json[0]["source"] == "vol"

    def test_mounts_json_none_inherits_author_mounts(self):
        author = EnvironmentConfig(
            mounts_json=[{"type": "bind", "source": "/sock", "target": "/sock"}]
        )
        trial = TrialEnvironmentConfig(mounts_json=None)
        trial.apply_to(author)
        assert len(author.mounts_json) == 1
        assert author.mounts_json[0]["source"] == "/sock"

    def test_mounts_json_empty_list_clears_author_mounts(self):
        author = EnvironmentConfig(mounts_json=[{"type": "bind", "source": "/a", "target": "/b"}])
        trial = TrialEnvironmentConfig(mounts_json=[])
        trial.apply_to(author)
        assert author.mounts_json == []


class TestMountConfig:
    def test_mounts_json_parses_bind_mount(self):
        data = _minimal_data(
            environment={
                "mounts_json": [
                    {
                        "type": "bind",
                        "source": "/var/run/docker.sock",
                        "target": "/var/run/docker.sock",
                    }
                ]
            }
        )
        cfg = ScenarioManifest.model_validate(data)
        assert len(cfg.environment.mounts_json) == 1
        m = cfg.environment.mounts_json[0]
        assert m["type"] == "bind"
        assert m["source"] == "/var/run/docker.sock"
        assert m["target"] == "/var/run/docker.sock"
        assert m.get("read_only") is None

    def test_mounts_json_parses_read_only_volume(self):
        data = _minimal_data(
            environment={
                "mounts_json": [
                    {"type": "volume", "source": "my-vol", "target": "/data", "read_only": True}
                ]
            }
        )
        cfg = ScenarioManifest.model_validate(data)
        m = cfg.environment.mounts_json[0]
        assert m["type"] == "volume"
        assert m["source"] == "my-vol"
        assert m["target"] == "/data"
        assert m.get("read_only") is True

    def test_mounts_json_defaults_to_empty_list(self):
        data = _minimal_data()
        cfg = ScenarioManifest.model_validate(data)
        assert cfg.environment.mounts_json == []

    def test_mounts_json_rejects_invalid_type(self):
        data = _minimal_data(
            environment={"mounts_json": [{"type": "tmpfs", "source": "none", "target": "/tmp"}]}
        )
        with pytest.raises(ValidationError):
            ScenarioManifest.model_validate(data)

    def test_mounts_json_multiple_mounts(self):
        data = _minimal_data(
            environment={
                "mounts_json": [
                    {
                        "type": "bind",
                        "source": "/var/run/docker.sock",
                        "target": "/var/run/docker.sock",
                    },
                    {
                        "type": "volume",
                        "source": "cache-vol",
                        "target": "/cache",
                        "read_only": True,
                    },
                ]
            }
        )
        cfg = ScenarioManifest.model_validate(data)
        assert len(cfg.environment.mounts_json) == 2
        assert cfg.environment.mounts_json[0]["type"] == "bind"
        assert cfg.environment.mounts_json[1]["type"] == "volume"
        assert cfg.environment.mounts_json[1].get("read_only") is True
