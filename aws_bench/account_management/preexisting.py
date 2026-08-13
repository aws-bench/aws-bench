"""Configuration and local safety state for pre-existing AWS accounts.

Pre-existing mode lets aws-bench operate accounts owned by an external control
plane (for example, AWS Control Tower).  The config file is an explicit
allowlist: aws-bench may resolve only the scenario/account-tag pairs declared
there and never creates, moves, tags, or closes an AWS account.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, StringConstraints, model_validator

from aws_bench.account_management.exceptions import (
    AccountResolutionError,
    ContaminationStateMissingError,
)
from aws_bench.account_management.models import OrgInfo, ScenarioAccount, TestEnvironment
from aws_bench.constants import STATE_DIR
from aws_bench.utils.filelock import file_lock

ACCOUNT_CONFIG_ENV_VAR = "AWSBENCH_ACCOUNT_CONFIG"

AccountId = Annotated[str, StringConstraints(pattern=r"^[0-9]{12}$")]


class PreexistingEnvironmentConfig(BaseModel):
    """Static scenario-to-account mapping owned outside aws-bench."""

    schema_version: Literal["1.0"] = "1.0"
    mode: Literal["preexisting"] = "preexisting"
    name: str = Field(min_length=1)
    accounts: dict[str, dict[str, AccountId]]
    runner_role: str = Field(min_length=1)
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
        """Return the persistent contamination-state path for this config.

        Defaults under ``STATE_DIR``, alongside the baseline snapshots, so no
        benchmark state lands in an account under test. A relative ``state_file``
        override resolves against the config's directory.
        """
        if self.state_file is None:
            return STATE_DIR / f"{self.name}-contamination.json"
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
        """Store state at ``path``, serializing access with a sibling lock."""
        self.path = path

    def initialize(self) -> None:
        """Create the state file with no accounts flagged, if it does not exist.

        Every other operation treats an absent file as an error, so this is the
        one place the file comes into being.
        """
        with file_lock(self.path):
            if not self.path.exists():
                self._write_unlocked(set())

    def contaminated(self) -> set[str]:
        """Return all locally flagged account IDs."""
        with file_lock(self.path):
            return self._read_unlocked()

    def mark(self, account_id: str) -> None:
        """Persist a contamination flag."""
        with file_lock(self.path):
            values = self._read_unlocked()
            values.add(account_id)
            self._write_unlocked(values)

    def clear(self, account_id: str) -> None:
        """Clear a contamination flag idempotently."""
        with file_lock(self.path):
            values = self._read_unlocked()
            values.discard(account_id)
            self._write_unlocked(values)

    def _read_unlocked(self) -> set[str]:
        """Read the flagged account IDs.

        Raises:
            ContaminationStateMissingError: If the file is absent. An empty set
                means "nothing is contaminated", so a deleted or misplaced file
                must not be readable as one.
        """
        if not self.path.exists():
            raise ContaminationStateMissingError(
                f"Contamination state file is missing: {self.path}. It records which "
                "accounts are unsafe to reuse, so aws-bench cannot treat its absence "
                "as 'no accounts contaminated'. Run 'aws-bench env init' to create it, "
                "or restore the file from where it was moved."
            )
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
