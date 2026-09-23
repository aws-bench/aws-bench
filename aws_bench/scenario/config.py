"""Pydantic models parsed from scenario.toml."""

from __future__ import annotations

import re
import tomllib

from harbor.models.trial.config import ServiceVolumeConfig
from pydantic import BaseModel, Field, field_validator, model_validator

# Account tags become AWS profile names inside the scenario container,
# get expanded into the deploy/verify/cleanup env as ``<TAG>=<account_id>``,
# and are interpolated into a heredoc when aws-bench writes ~/.aws/config.
# Constraining them to a conservative identifier shape closes shell-injection
# and INI-parser breakage at the parse boundary so downstream layers can
# trust the values verbatim. Length is capped at 32 chars to match the trial
# name prefix and keep INI lines readable.
_ACCOUNT_TAG_MAX_LEN = 32
_ACCOUNT_TAG_RE = re.compile(rf"^[A-Za-z][A-Za-z0-9_]{{0,{_ACCOUNT_TAG_MAX_LEN - 1}}}$")


class Author(BaseModel):
    """One scenario author."""

    name: str
    email: str | None = None


class ScenarioInfo(BaseModel):
    """The [scenario] section."""

    name: str
    description: str = ""
    authors: list[Author] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    account_tags: list[str]
    regions: list[str]

    @field_validator("account_tags")
    @classmethod
    def _validate_account_tags(cls, value: list[str]) -> list[str]:
        for tag in value:
            if not _ACCOUNT_TAG_RE.match(tag):
                raise ValueError(
                    f"Invalid account_tag {tag!r}: must match "
                    f"[A-Za-z][A-Za-z0-9_]{{0,{_ACCOUNT_TAG_MAX_LEN - 1}}} "
                    f"(letter prefix; ≤{_ACCOUNT_TAG_MAX_LEN} chars)"
                )
        if len(set(value)) != len(value):
            raise ValueError("account_tags must not contain duplicates")
        # Multi-account per scenario is a future extension.
        if len(value) != 1:
            raise ValueError("account_tags must have exactly one entry")
        return value

    @field_validator("regions")
    @classmethod
    def _validate_regions(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("scenario.regions must have at least one entry")
        if any(not region.strip() for region in value):
            raise ValueError("scenario.regions entries must be non-empty")
        if len(set(value)) != len(value):
            raise ValueError("scenario.regions must not contain duplicates")
        return value


class EnvironmentConfig(BaseModel):
    """Author-side resource limits for the scenario container."""

    build_timeout_sec: float = 600.0
    cpus: int = 1
    memory_mb: int = 2048
    mounts_json: list[ServiceVolumeConfig] = Field(default_factory=list)


class PhaseConfig(BaseModel):
    """A deploy / verify / cleanup phase: timeout + env.

    Per-phase `env` vars merge with framework-injected credentials at script
    invocation; phase-set values do not override credential keys. Each phase's
    default ``timeout_sec`` is set where the phase is declared on
    ``ScenarioManifest``; the class default applies only to a partial phase
    table that omits ``timeout_sec``.
    """

    timeout_sec: float = 600.0
    env: dict[str, str] = Field(default_factory=dict)


class QuotaIncrease(BaseModel):
    """One service-quota increase request, scoped to one (account_tag, region)."""

    account_tag: str
    region: str
    service_code: str
    quota_code: str
    desired_value: float


class ScenarioManifest(BaseModel):
    """Parsed scenario.toml.

    Cross-field invariants (single-field rules live on ScenarioInfo):
      * Every quotas[i].region must appear in scenario.regions.
      * Every quotas[i].account_tag must appear in scenario.account_tags.
        Tags are case-sensitive and never normalized.
    """

    schema_version: str = "1.0"
    scenario: ScenarioInfo
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    deploy: PhaseConfig = Field(default_factory=lambda: PhaseConfig(timeout_sec=600.0))
    verify: PhaseConfig = Field(default_factory=lambda: PhaseConfig(timeout_sec=120.0))
    cleanup: PhaseConfig = Field(default_factory=lambda: PhaseConfig(timeout_sec=300.0))
    # reset runs an optional reset/reset.sh before the framework reset; the field
    # must exist for every ScenarioPhase so ScenarioTrial._execute's
    # getattr(manifest, phase) resolves (a missing field made reset.sh silently
    # never run — the AttributeError was swallowed into the trial's script_error).
    reset: PhaseConfig = Field(default_factory=lambda: PhaseConfig(timeout_sec=600.0))
    quotas: list[QuotaIncrease] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_quotas_reference_known_tags(self) -> ScenarioManifest:
        valid = set(self.scenario.account_tags)
        for i, q in enumerate(self.quotas):
            if q.account_tag not in valid:
                raise ValueError(
                    f"quotas[{i}].account_tag {q.account_tag!r} not in "
                    f"scenario.account_tags {sorted(valid)}"
                )
        return self

    @model_validator(mode="after")
    def _check_quotas_reference_known_regions(self) -> ScenarioManifest:
        valid = set(self.scenario.regions)
        for i, q in enumerate(self.quotas):
            if q.region not in valid:
                raise ValueError(
                    f"quotas[{i}].region {q.region!r} not in scenario.regions {sorted(valid)}"
                )
        return self

    @classmethod
    def model_validate_toml(cls, toml_data: str | bytes) -> ScenarioManifest:
        """Parse a scenario.toml document into a ``ScenarioManifest``.

        Accepts UTF-8 bytes or a ``str``. Bytes are decoded by ``tomllib``
        per the TOML spec (UTF-8); strings are passed to ``tomllib.loads``
        unchanged. Prefer bytes when reading from disk so the platform's
        locale encoding cannot misinterpret the file.
        """
        if isinstance(toml_data, bytes):
            return cls.model_validate(tomllib.loads(toml_data.decode("utf-8")))
        return cls.model_validate(tomllib.loads(toml_data))


class TrialEnvironmentConfig(BaseModel):
    """Operator-side environment spec — overrides plus runtime knobs.

    `override_*` fields default to None ("inherit author's value"). Merging
    with the author-side EnvironmentConfig is deferred until environment
    instantiation so the persisted trial config retains both sides for audit.
    """

    force_build: bool = False
    delete: bool = True
    override_build_timeout_sec: float | None = None
    override_cpus: int | None = None
    override_memory_mb: int | None = None
    mounts_json: list[ServiceVolumeConfig] | None = None

    def apply_to(self, author_env: EnvironmentConfig) -> None:
        """Mutate author_env in-place with these overrides."""
        if self.override_cpus is not None:
            author_env.cpus = self.override_cpus
        if self.override_memory_mb is not None:
            author_env.memory_mb = self.override_memory_mb
        if self.override_build_timeout_sec is not None:
            author_env.build_timeout_sec = self.override_build_timeout_sec
        if self.mounts_json is not None:
            author_env.mounts_json = list(self.mounts_json)
