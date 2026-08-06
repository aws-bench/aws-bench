"""Configuration and local safety state for pre-existing AWS accounts.

Pre-existing mode lets aws-bench operate accounts owned by an external control
plane (for example, AWS Control Tower).  The config file is an explicit
allowlist: aws-bench may resolve only the scenario/account-tag pairs declared
there and never creates, moves, tags, or closes an AWS account.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal, TextIO

import yaml
from pydantic import BaseModel, Field, StringConstraints, model_validator

from aws_bench.account_management.exceptions import AccountResolutionError
from aws_bench.account_management.models import OrgInfo, ScenarioAccount, TestEnvironment

ACCOUNT_CONFIG_ENV_VAR = "AWSBENCH_ACCOUNT_CONFIG"

AccountId = Annotated[str, StringConstraints(pattern=r"^[0-9]{12}$")]


class PreexistingEnvironmentConfig(BaseModel):
    """Static scenario-to-account mapping owned outside aws-bench."""

    schema_version: Literal["1.0"] = "1.0"
    mode: Literal["preexisting"] = "preexisting"
    name: str = Field(min_length=1)
    accounts: dict[str, dict[str, AccountId]]
    runner_role: str | None = Field(default=None, min_length=1)
    state_file: Path | None = None

    @model_validator(mode="after")
    def _validate_account_assignments(self) -> "PreexistingEnvironmentConfig":
        if not self.accounts:
            raise ValueError("accounts must contain at least one scenario mapping")

        assignments: dict[str, str] = {}
        for scenario_name, tags in self.accounts.items():
            if not scenario_name or not tags:
                raise ValueError("every scenario must contain at least one account tag")
            for account_tag, account_id in tags.items():
                if not account_tag:
                    raise ValueError("account tags must not be empty")
                assignment = f"{scenario_name}/{account_tag}"
                previous = assignments.get(account_id)
                if previous is not None:
                    raise ValueError(
                        f"account {account_id} is assigned to both {previous} and {assignment}; "
                        "one account cannot host concurrent scenario baselines"
                    )
                assignments[account_id] = assignment
        return self

    def resolve_state_file(self, config_path: Path) -> Path:
        """Return the persistent contamination-state path for this config."""
        if self.state_file is None:
            return config_path.with_suffix(config_path.suffix + ".state.json")
        if self.state_file.is_absolute():
            return self.state_file
        return config_path.parent / self.state_file

    def to_test_environment(
        self,
        *,
        required_by_scenario: dict[str, set[str]] | None = None,
    ) -> TestEnvironment:
        """Build the normal resolved environment model from the static allowlist."""
        selected = self.accounts
        if required_by_scenario is not None:
            selected = {}
            for scenario_name, required_tags in required_by_scenario.items():
                configured = self.accounts.get(scenario_name)
                if configured is None:
                    raise AccountResolutionError(
                        f"Scenario {scenario_name!r} is not present in the pre-existing "
                        "account config. Add an explicit mapping before running it."
                    )
                missing = required_tags - configured.keys()
                if missing:
                    raise AccountResolutionError(
                        f"Scenario {scenario_name!r} is missing account tag(s) "
                        f"{sorted(missing)} in the pre-existing account config."
                    )
                selected[scenario_name] = {tag: configured[tag] for tag in sorted(required_tags)}

        accounts = {
            scenario_name: {
                tag: ScenarioAccount(
                    account_id=account_id,
                    email="",
                    scenario_name=scenario_name,
                    account_tag=tag,
                )
                for tag, account_id in tags.items()
            }
            for scenario_name, tags in selected.items()
        }
        # TestEnvironment predates alternate account backends and requires org
        # metadata.  These sentinel values make the mode explicit in persisted
        # configs without pretending the runner owns the real Organization.
        org = OrgInfo(
            org_id="preexisting",
            root_id="preexisting",
            management_account_id="000000000000",
            management_account_email="",
        )
        return TestEnvironment(
            org=org,
            ou_id="preexisting",
            ou_name=self.name,
            accounts=accounts,
        )


def activate_account_config(path: Path) -> None:
    """Validate and activate a pre-existing account config for this process."""
    resolved = path.expanduser().resolve()
    load_account_config(resolved)
    os.environ[ACCOUNT_CONFIG_ENV_VAR] = str(resolved)


def load_account_config(path: Path) -> PreexistingEnvironmentConfig:
    """Load and validate one YAML/JSON pre-existing account config."""
    if not path.is_file():
        raise FileNotFoundError(f"Pre-existing account config not found: {path}")
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
    else:
        raw = yaml.safe_load(path.read_text())
    return PreexistingEnvironmentConfig.model_validate(raw)


def active_account_config() -> tuple[PreexistingEnvironmentConfig, Path] | None:
    """Return the active config and its resolved path, if mode is enabled."""
    raw_path = os.environ.get(ACCOUNT_CONFIG_ENV_VAR)
    if not raw_path:
        return None
    path = Path(raw_path).expanduser().resolve()
    return load_account_config(path), path


class PreexistingStateStore:
    """Small, lock-protected local store for fail-closed contamination flags."""

    def __init__(self, path: Path) -> None:
        """Store state at ``path`` and serialize access with a sibling lock."""
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")

    @contextmanager
    def _locked(self) -> Iterator[TextIO]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield lock_file
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def contaminated(self) -> set[str]:
        """Return all locally flagged account IDs."""
        with self._locked():
            return self._read_unlocked()

    def mark(self, account_id: str) -> None:
        """Persist a contamination flag."""
        with self._locked():
            values = self._read_unlocked()
            values.add(account_id)
            self._write_unlocked(values)

    def clear(self, account_id: str) -> None:
        """Clear a contamination flag idempotently."""
        with self._locked():
            values = self._read_unlocked()
            values.discard(account_id)
            self._write_unlocked(values)

    def _read_unlocked(self) -> set[str]:
        if not self.path.exists():
            return set()
        payload = json.loads(self.path.read_text())
        return set(payload.get("contaminated_account_ids", []))

    def _write_unlocked(self, values: set[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": "1.0", "contaminated_account_ids": sorted(values)}
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
