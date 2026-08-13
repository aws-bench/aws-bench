"""Tests for externally owned account mappings and safety state."""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from aws_bench.account_management.constants import CFN_OPS_ROLE_NAME
from aws_bench.account_management.exceptions import (
    AccountResolutionError,
    ContaminationStateInvalidError,
    ContaminationStateMissingError,
)
from aws_bench.account_management.manager import AccountManager
from aws_bench.account_management.preexisting import (
    ACCOUNT_CONFIG_ENV_VAR,
    PreexistingStateStore,
    effective_cfn_role,
    load_account_config,
)
from aws_bench.constants import STATE_DIR


def _write_config(tmp_path: Path, body: str | None = None) -> Path:
    path = tmp_path / "accounts.yaml"
    path.write_text(
        body
        or """
schema_version: "1.0"
mode: preexisting
name: aws-bench
runner_role: AWSBenchRunner
cfn_role: cfn-service-execution
state_file: ./state.json
accounts:
  scenario-a:
    PRIMARY: "111122223333"
"""
    )
    return path


def test_load_and_resolve_mapping(tmp_path: Path):
    config = load_account_config(_write_config(tmp_path))
    environment = config.to_test_environment(required_by_scenario={"scenario-a": {"PRIMARY"}})
    assert environment.mapping_for("scenario-a") == {"PRIMARY": "111122223333"}
    assert environment.ou_id == "preexisting"


def test_duplicate_account_assignment_is_rejected(tmp_path: Path):
    path = _write_config(
        tmp_path,
        """
mode: preexisting
name: aws-bench
runner_role: AWSBenchRunner
cfn_role: cfn-service-execution
accounts:
  scenario-a: {PRIMARY: "111122223333"}
  scenario-b: {PRIMARY: "111122223333"}
""",
    )
    with pytest.raises(ValidationError, match="cannot host concurrent scenario baselines"):
        load_account_config(path)


def test_runner_role_is_required(tmp_path: Path):
    path = _write_config(
        tmp_path,
        """
mode: preexisting
name: aws-bench
accounts:
  scenario-a: {PRIMARY: "111122223333"}
""",
    )
    with pytest.raises(ValidationError, match="runner_role"):
        load_account_config(path)


def test_state_file_defaults_outside_the_config_directory(tmp_path: Path):
    """The default keeps contamination state with the snapshots, not next to the config."""
    config = load_account_config(
        _write_config(
            tmp_path,
            """
mode: preexisting
name: acme-benchmark
runner_role: AWSBenchRunner
cfn_role: cfn-service-execution
accounts:
  scenario-a: {PRIMARY: "111122223333"}
""",
        )
    )
    assert config.resolve_state_file(tmp_path / "accounts.yaml") == (
        STATE_DIR / "acme-benchmark-contamination.json"
    )


def test_cfn_role_is_required(tmp_path: Path):
    path = _write_config(
        tmp_path,
        """
mode: preexisting
name: aws-bench
runner_role: AWSBenchRunner
accounts:
  scenario-a: {PRIMARY: "111122223333"}
""",
    )
    with pytest.raises(ValidationError, match="cfn_role"):
        load_account_config(path)


def test_effective_cfn_role_follows_the_active_backend(tmp_path: Path, monkeypatch):
    """Managed accounts carry the role aws-bench made; external ones name their own."""
    assert effective_cfn_role() == CFN_OPS_ROLE_NAME

    config = _write_config(
        tmp_path,
        """
mode: preexisting
name: aws-bench
runner_role: OrganizationAccountAccessRole
cfn_role: OrganizationAccountAccessRole
accounts:
  scenario-a: {PRIMARY: "111122223333"}
""",
    )
    monkeypatch.setenv(ACCOUNT_CONFIG_ENV_VAR, str(config))
    assert effective_cfn_role() == "OrganizationAccountAccessRole"


def test_missing_scenario_or_tag_is_rejected(tmp_path: Path):
    config = load_account_config(_write_config(tmp_path))
    with pytest.raises(AccountResolutionError, match="not present"):
        config.to_test_environment(required_by_scenario={"other": {"PRIMARY"}})
    with pytest.raises(AccountResolutionError, match="missing account tag"):
        config.to_test_environment(required_by_scenario={"scenario-a": {"SECONDARY"}})


def test_state_store_persists_and_clears(tmp_path: Path):
    store = PreexistingStateStore(tmp_path / "state.json")
    store.initialize()
    store.mark("111122223333")
    assert PreexistingStateStore(tmp_path / "state.json").contaminated() == {"111122223333"}
    store.clear("111122223333")
    assert store.contaminated() == set()


def test_missing_state_file_is_not_read_as_clean(tmp_path: Path):
    """A deleted or misplaced file must not clear contamination for every account."""
    store = PreexistingStateStore(tmp_path / "state.json")
    with pytest.raises(ContaminationStateMissingError, match="missing"):
        store.contaminated()

    store.initialize()
    store.mark("111122223333")
    (tmp_path / "state.json").unlink()
    with pytest.raises(ContaminationStateMissingError, match="missing"):
        store.contaminated()


def test_initialize_is_idempotent_and_preserves_flags(tmp_path: Path):
    store = PreexistingStateStore(tmp_path / "state.json")
    store.initialize()
    assert store.contaminated() == set()
    store.mark("111122223333")
    store.initialize()
    assert store.contaminated() == {"111122223333"}


def test_initialize_reports_whether_it_created_the_file(tmp_path: Path):
    """Callers warn on creation, since it cannot be told from a lost file."""
    store = PreexistingStateStore(tmp_path / "state.json")
    assert store.initialize() is True
    assert store.initialize() is False


def test_mark_creates_an_absent_file_rather_than_losing_the_flag(tmp_path: Path):
    """Adding a flag only narrows reuse, so an absent file starts from empty here."""
    store = PreexistingStateStore(tmp_path / "state.json")

    store.mark("111122223333")

    assert store.contaminated() == {"111122223333"}


def test_clear_still_fails_closed_on_an_absent_file(tmp_path: Path):
    """Unlike mark, clearing against an absent file would silently drop flags."""
    store = PreexistingStateStore(tmp_path / "state.json")
    with pytest.raises(ContaminationStateMissingError):
        store.clear("111122223333")


@pytest.mark.parametrize(
    ("label", "body"),
    [
        ("empty object", "{}"),
        ("renamed field", '{"schema_version": "2.0", "contaminated": ["111122223333"]}'),
        ("unknown schema version", '{"schema_version": "9.9", "contaminated_account_ids": []}'),
        ("bare string for the list", '{"schema_version": "1.0", "contaminated_account_ids": "1"}'),
        ("top-level list", '["111122223333"]'),
        ("truncated", '{"schema_version": "1.0", "contaminated_acc'),
        ("empty file", ""),
    ],
)
def test_unparseable_state_is_not_read_as_clean(tmp_path: Path, label: str, body: str):
    """An empty set means 'nothing contaminated', so a file we cannot parse must raise."""
    path = tmp_path / "state.json"
    path.write_text(body)

    with pytest.raises(ContaminationStateInvalidError, match="unreadable"):
        PreexistingStateStore(path).contaminated()


@pytest.mark.parametrize("name", ["../escape", "with/slash", "", ".leading-dot"])
def test_name_that_could_traverse_the_state_directory_is_rejected(tmp_path: Path, name: str):
    """Name becomes a path component in the default state-file location."""
    path = _write_config(
        tmp_path,
        f"""
mode: preexisting
name: "{name}"
runner_role: AWSBenchRunner
cfn_role: cfn-service-execution
accounts:
  scenario-a: {{PRIMARY: "111122223333"}}
""",
    )
    with pytest.raises(ValidationError):
        load_account_config(path)


@pytest.mark.asyncio
async def test_manager_skips_org_mutations_and_tracks_contamination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(ACCOUNT_CONFIG_ENV_VAR, str(_write_config(tmp_path)))
    with patch("aws_bench.account_management.manager.OrganizationsClient") as org_cls:
        manager = AccountManager()
        assert manager.init_organization("aws-bench") == "preexisting"
        mapping = await manager.ensure_scenario_accounts("aws-bench", "scenario-a", {"PRIMARY"})
        assert mapping == {"PRIMARY": "111122223333"}
        manager.ensure_region_restriction_scp("scenario-a", ["us-east-1"], ["111122223333"])
        await manager.mark_contaminated("111122223333")
        assert manager.get_contaminated_accounts(["111122223333"]) == ["111122223333"]
        await manager.clear_contaminated("111122223333")
        assert manager.get_contaminated_accounts(["111122223333"]) == []

    org = org_cls.return_value
    assert not any(
        getattr(org, name).called
        for name in (
            "create_organization",
            "create_ou",
            "ensure_org_role_protection_scp",
            "ensure_region_restriction_scp",
            "tag_resource",
            "untag_resource",
        )
    )


def test_manager_rejects_unallowlisted_account_and_termination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(ACCOUNT_CONFIG_ENV_VAR, str(_write_config(tmp_path)))
    manager = AccountManager()
    with pytest.raises(AccountResolutionError, match="not in the active pre-existing allowlist"):
        manager.ensure_region_restriction_scp("scenario-a", ["us-east-1"], ["111111111111"])
    with pytest.raises(RuntimeError, match="disabled"):
        manager.terminate_environment("aws-bench")


@pytest.fixture(autouse=True)
def _clear_account_config(monkeypatch: pytest.MonkeyPatch):
    """Keep this module independent from the developer's shell environment."""
    monkeypatch.delenv(ACCOUNT_CONFIG_ENV_VAR, raising=False)
    yield
