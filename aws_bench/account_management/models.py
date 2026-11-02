"""Data models for account management."""

from __future__ import annotations

from pydantic import BaseModel, Field


class OrgInfo(BaseModel):
    """AWS Organization metadata."""

    org_id: str
    root_id: str
    management_account_id: str
    management_account_email: str


class ScenarioAccount(BaseModel):
    """A member account provisioned for a containerized scenario.

    The (scenario_name, account_tag) pair is unique within an OU and is
    encoded into the ``aws-bench:scenario`` tag value as ``<name>/<tag>``.
    """

    account_id: str
    email: str
    scenario_name: str
    account_tag: str
    status: str = "ACTIVE"


class TestEnvironment(BaseModel):
    """The AWS Organization OU that serves as the testing environment.

    ``accounts`` indexes the provisioned member accounts as a two-level map,
    ``{scenario_name: {account_tag: ScenarioAccount}}``. The dict structurally
    enforces the (scenario, account_tag) uniqueness invariant. The task side
    knows the outer key as ``scenario_id``; the two are equal in a resolved
    environment.
    """

    org: OrgInfo
    ou_id: str
    ou_name: str
    accounts: dict[str, dict[str, ScenarioAccount]] = Field(default_factory=dict)

    def to_scenario_account_mappings(self) -> dict[str, dict[str, str]]:
        """Return ``{scenario_name: {account_tag: account_id}}``."""
        return {
            scenario_name: {tag: account.account_id for tag, account in tag_map.items()}
            for scenario_name, tag_map in self.accounts.items()
        }

    def mapping_for(self, scenario_name: str) -> dict[str, str]:
        """Return ``{account_tag: account_id}`` for one scenario, or ``{}`` if unknown."""
        return {
            tag: account.account_id for tag, account in self.accounts.get(scenario_name, {}).items()
        }
