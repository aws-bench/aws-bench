"""Tests for aws_bench.account_management.manager."""

from __future__ import annotations

import asyncio
import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from tenacity import wait_none

from aws_bench.account_management.exceptions import AccountCreationError, AccountManagementError
from aws_bench.account_management.manager import AccountManager
from aws_bench.account_management.models import OrgInfo


def _make_org_info() -> OrgInfo:
    return OrgInfo(
        org_id="o-abc",
        root_id="r-root1",
        management_account_id="111111111111",
        management_account_email="mgmt@example.com",
    )


@pytest.fixture()
def manager():
    """Create an AccountManager with a mocked OrganizationsClient."""
    with patch("aws_bench.account_management.manager.OrganizationsClient") as mock_org_cls:
        mock_org = MagicMock()
        mock_org_cls.return_value = mock_org
        mgr = AccountManager()
        assert mgr._org is mock_org
        yield mgr


# ── init_organization ──


def test_init_organization_creates_new_ou(manager):
    """Verify init_organization creates a new OU when none exists."""
    org_info = _make_org_info()
    manager._org.get_org_info.return_value = org_info
    manager._org.find_ou_by_name.return_value = None
    manager._org.create_ou.return_value = "ou-new"

    ou_id = manager.init_organization("test-env")

    manager._org.create_organization.assert_called_once()
    manager._org.create_ou.assert_called_once_with("r-root1", "test-env")
    assert ou_id == "ou-new"


def test_init_organization_idempotent_when_ou_exists(manager):
    """Verify init_organization is idempotent when OU already exists."""
    org_info = _make_org_info()
    manager._org.get_org_info.return_value = org_info
    manager._org.find_ou_by_name.return_value = "ou-existing"

    ou_id = manager.init_organization("test-env")

    manager._org.create_ou.assert_not_called()
    manager._org.list_accounts_in_ou.assert_not_called()
    assert ou_id == "ou-existing"


def test_init_organization_attaches_scp(manager):
    """Verify init_organization calls ensure_org_role_protection_scp."""
    org_info = _make_org_info()
    manager._org.get_org_info.return_value = org_info
    manager._org.find_ou_by_name.return_value = "ou-existing"

    manager.init_organization("test-env")

    manager._org.ensure_org_role_protection_scp.assert_called_once_with("ou-existing")


def test_init_organization_attaches_scp_to_new_ou(manager):
    """Verify SCP is attached even when creating a new OU."""
    org_info = _make_org_info()
    manager._org.get_org_info.return_value = org_info
    manager._org.find_ou_by_name.return_value = None
    manager._org.create_ou.return_value = "ou-new"

    manager.init_organization("test-env")

    manager._org.ensure_org_role_protection_scp.assert_called_once_with("ou-new")


def test_init_organization_enables_account_management_after_creating_org(manager):
    """Trusted access for Account Management needs the organization to exist first."""
    manager._org.get_org_info.return_value = _make_org_info()
    manager._org.find_ou_by_name.return_value = "ou-existing"

    manager.init_organization("test-env")

    assert [c[0] for c in manager._org.mock_calls[:2]] == [
        "create_organization",
        "enable_account_management_access",
    ]


def test_init_organization_skips_account_management_in_preexisting_mode(manager):
    manager._preexisting = MagicMock()
    manager._preexisting.name = "test-env"

    assert manager.init_organization("test-env") == "preexisting"

    manager._org.enable_account_management_access.assert_not_called()


# ── scenario-aware methods ──

from aws_bench.account_management.constants import (  # noqa: E402
    CONTAMINATED_TAG_KEY,
    CONTAMINATION_TAG_MAX_ATTEMPTS,
    REGION_OPT_IN_POLL_INTERVAL_SEC,
    REGION_OPT_IN_TIMEOUT_SEC,
    SCENARIO_ACCOUNT_TAG_KEY,
    SCENARIO_SHA_TAG_KEY,
)
from aws_bench.account_management.exceptions import (  # noqa: E402
    AccountResolutionError,
    DuplicateScenarioAccountError,
)


def _ou(manager, ou_id="ou-1"):
    """Wire org info + OU for scenario tests."""
    manager._org.get_org_info.return_value = _make_org_info()
    manager._org.find_ou_by_name.return_value = ou_id


def test_ensure_scenario_accounts_reuses_existing(manager):
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = [
        {"Id": "111", "Email": "a@b.com", "Status": "ACTIVE"}
    ]
    manager._org.get_tags.return_value = {SCENARIO_ACCOUNT_TAG_KEY: "lambda-x/PRIMARY"}

    out = asyncio.run(manager.ensure_scenario_accounts("env", "lambda-x", {"PRIMARY"}))

    assert out == {"PRIMARY": "111"}
    manager._org.create_account.assert_not_called()


def test_ensure_scenario_accounts_skips_suspended(manager):
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = [
        {"Id": "111", "Email": "a@b.com", "Status": "SUSPENDED"}
    ]
    manager._org.get_tags.return_value = {SCENARIO_ACCOUNT_TAG_KEY: "lambda-x/PRIMARY"}
    manager._org.create_account = AsyncMock(return_value="222")
    manager._org.move_account_to_ou = AsyncMock()
    manager._org.tag_resource = AsyncMock()

    out = asyncio.run(manager.ensure_scenario_accounts("env", "lambda-x", {"PRIMARY"}))

    assert out == {"PRIMARY": "222"}
    manager._org.tag_resource.assert_any_await("222", SCENARIO_ACCOUNT_TAG_KEY, "lambda-x/PRIMARY")
    manager._org.tag_resource.assert_any_await("222", SCENARIO_SHA_TAG_KEY, "lambda-x")


def test_ensure_scenario_accounts_creates_when_missing(manager):
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = []
    manager._org.create_account = AsyncMock(return_value="333")
    manager._org.move_account_to_ou = AsyncMock()
    manager._org.tag_resource = AsyncMock()

    out = asyncio.run(manager.ensure_scenario_accounts("env", "lambda-x", {"PRIMARY"}))

    assert out == {"PRIMARY": "333"}
    manager._org.tag_resource.assert_any_await("333", SCENARIO_ACCOUNT_TAG_KEY, "lambda-x/PRIMARY")
    manager._org.tag_resource.assert_any_await("333", SCENARIO_SHA_TAG_KEY, "lambda-x")


def test_ensure_scenario_accounts_does_not_reuse_other_scenarios_account(manager):
    """A different scenario's account must not be reused even if account_tag matches."""
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = [
        {"Id": "111", "Email": "a@b.com", "Status": "ACTIVE"}
    ]
    # Different scenario, same account_tag.
    manager._org.get_tags.return_value = {SCENARIO_ACCOUNT_TAG_KEY: "other-scenario/PRIMARY"}
    manager._org.create_account = AsyncMock(return_value="222")
    manager._org.move_account_to_ou = AsyncMock()
    manager._org.tag_resource = AsyncMock()

    out = asyncio.run(manager.ensure_scenario_accounts("env", "lambda-x", {"PRIMARY"}))

    assert out == {"PRIMARY": "222"}


def test_ensure_scenario_accounts_lists_only_once_for_set(manager):
    """Set-based ensure must call list_accounts_in_ou exactly once for N tags."""
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = [
        {"Id": "111", "Email": "a@b.com", "Status": "ACTIVE"},
        {"Id": "222", "Email": "c@d.com", "Status": "ACTIVE"},
    ]
    manager._org.get_tags.side_effect = [
        {SCENARIO_ACCOUNT_TAG_KEY: "lambda-x/PRIMARY"},
        {SCENARIO_ACCOUNT_TAG_KEY: "lambda-x/SECONDARY"},
    ]

    out = asyncio.run(manager.ensure_scenario_accounts("env", "lambda-x", {"PRIMARY", "SECONDARY"}))

    assert out == {"PRIMARY": "111", "SECONDARY": "222"}
    assert manager._org.list_accounts_in_ou.call_count == 1


def test_ensure_scenario_accounts_raises_on_duplicate(manager):
    """Two ACTIVE accounts tagged for the same (scenario, account_tag) is an integrity error."""
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = [
        {"Id": "111", "Email": "a@b.com", "Status": "ACTIVE"},
        {"Id": "222", "Email": "c@d.com", "Status": "ACTIVE"},
    ]
    manager._org.get_tags.side_effect = [
        {SCENARIO_ACCOUNT_TAG_KEY: "lambda-x/PRIMARY"},
        {SCENARIO_ACCOUNT_TAG_KEY: "lambda-x/PRIMARY"},
    ]

    with pytest.raises(DuplicateScenarioAccountError):
        asyncio.run(manager.ensure_scenario_accounts("env", "lambda-x", {"PRIMARY"}))


def _email_already_exists() -> AccountCreationError:
    """The typed error raised when CreateAccount reports EMAIL_ALREADY_EXISTS."""
    return AccountCreationError("CreateAccount failed for 'lambda-x-PRIMARY': EMAIL_ALREADY_EXISTS")


def test_create_scenario_account_retries_on_email_collision_with_fresh_email(manager, monkeypatch):
    """EMAIL_ALREADY_EXISTS is retried, and each attempt regenerates a distinct email.

    A per-second timestamp is the only entropy in the generated email, so a bare
    retry with the same email would re-collide forever. The retry must call
    generate_account_email again so the second attempt uses a different address.
    """
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = []
    manager._org.move_account_to_ou = AsyncMock()
    manager._org.tag_resource = AsyncMock()

    # First create_account attempt collides; second succeeds.
    manager._org.create_account = AsyncMock(side_effect=[_email_already_exists(), "444"])
    # Distinct emails across attempts (simulates the advanced timestamp).
    emails = iter(
        [
            "lambda-x-PRIMARY-20260704120000@example.com",
            "lambda-x-PRIMARY-20260704120005@example.com",
        ]
    )
    gen = MagicMock(side_effect=lambda *a, **k: next(emails))
    monkeypatch.setattr("aws_bench.account_management.manager.generate_account_email", gen)
    monkeypatch.setattr(manager._create_scenario_account.retry, "wait", wait_none())

    out = asyncio.run(manager.ensure_scenario_accounts("env", "lambda-x", {"PRIMARY"}))

    assert out == {"PRIMARY": "444"}
    assert manager._org.create_account.await_count == 2
    # The two attempts used different emails.
    used_emails = [c.kwargs["email"] for c in manager._org.create_account.await_args_list]
    assert used_emails[0] != used_emails[1]


def test_create_scenario_account_reraises_after_exhausting_email_retries(manager, monkeypatch):
    """A persistently colliding email exhausts the retry cap and reraises AccountCreationError."""
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = []
    manager._org.move_account_to_ou = AsyncMock()
    manager._org.tag_resource = AsyncMock()
    manager._org.create_account = AsyncMock(side_effect=_email_already_exists())
    monkeypatch.setattr(manager._create_scenario_account.retry, "wait", wait_none())

    with pytest.raises(AccountCreationError, match="EMAIL_ALREADY_EXISTS"):
        asyncio.run(manager.ensure_scenario_accounts("env", "lambda-x", {"PRIMARY"}))

    # Retried more than once before giving up.
    assert manager._org.create_account.await_count >= 2


def test_resolve_test_environment_returns_mapping(manager):
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = [
        {"Id": "111", "Email": "a@b.com", "Status": "ACTIVE"},
        {"Id": "222", "Email": "c@d.com", "Status": "ACTIVE"},
        {"Id": "333", "Email": "e@f.com", "Status": "ACTIVE"},
    ]
    manager._org.get_tags.side_effect = [
        {SCENARIO_ACCOUNT_TAG_KEY: "lambda-x/PRIMARY"},
        {SCENARIO_ACCOUNT_TAG_KEY: "lambda-x/SECONDARY"},
        {SCENARIO_ACCOUNT_TAG_KEY: "other/PRIMARY"},  # different scenario
    ]

    out = manager.resolve_test_environment(
        "env", {"lambda-x": {"PRIMARY", "SECONDARY"}}
    ).to_scenario_account_mappings()

    assert out == {"lambda-x": {"PRIMARY": "111", "SECONDARY": "222"}}


def test_resolve_test_environment_raises_when_none(manager):
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = []

    with pytest.raises(AccountResolutionError):
        manager.resolve_test_environment("env", {"lambda-x": {"PRIMARY"}})


def test_resolve_test_environment_raises_on_duplicate(manager):
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = [
        {"Id": "111", "Email": "a@b.com", "Status": "ACTIVE"},
        {"Id": "222", "Email": "c@d.com", "Status": "ACTIVE"},
    ]
    manager._org.get_tags.side_effect = [
        {SCENARIO_ACCOUNT_TAG_KEY: "lambda-x/PRIMARY"},
        {SCENARIO_ACCOUNT_TAG_KEY: "lambda-x/PRIMARY"},  # duplicate
    ]

    with pytest.raises(DuplicateScenarioAccountError):
        manager.resolve_test_environment("env", {"lambda-x": {"PRIMARY"}})


def test_resolve_test_environment_skips_suspended(manager):
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = [
        {"Id": "111", "Email": "a@b.com", "Status": "SUSPENDED"},
        {"Id": "222", "Email": "c@d.com", "Status": "ACTIVE"},
    ]
    # Only the ACTIVE account triggers get_tags.
    manager._org.get_tags.return_value = {SCENARIO_ACCOUNT_TAG_KEY: "lambda-x/SECONDARY"}

    out = manager.resolve_test_environment(
        "env", {"lambda-x": {"SECONDARY"}}
    ).to_scenario_account_mappings()

    assert out == {"lambda-x": {"SECONDARY": "222"}}


def test_resolve_test_environment_case_sensitive(manager):
    """Tag values are not normalized — primary != PRIMARY."""
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = [
        {"Id": "111", "Email": "a@b.com", "Status": "ACTIVE"}
    ]
    manager._org.get_tags.return_value = {SCENARIO_ACCOUNT_TAG_KEY: "Lambda-X/primary"}

    with pytest.raises(AccountResolutionError):
        manager.resolve_test_environment("env", {"lambda-x": {"PRIMARY"}})


def test_resolve_test_environment_returns_when_tags_match(manager):
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = [
        {"Id": "111", "Email": "a@b.com", "Status": "ACTIVE"},
        {"Id": "222", "Email": "c@d.com", "Status": "ACTIVE"},
    ]
    manager._org.get_tags.side_effect = [
        {SCENARIO_ACCOUNT_TAG_KEY: "lambda-x/PRIMARY"},
        {SCENARIO_ACCOUNT_TAG_KEY: "lambda-x/SECONDARY"},
    ]

    out = manager.resolve_test_environment(
        "env", {"lambda-x": {"PRIMARY", "SECONDARY"}}
    ).to_scenario_account_mappings()
    assert out == {"lambda-x": {"PRIMARY": "111", "SECONDARY": "222"}}


def test_resolve_test_environment_raises_on_missing_tag(manager):
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = [
        {"Id": "111", "Email": "a@b.com", "Status": "ACTIVE"},
    ]
    manager._org.get_tags.return_value = {SCENARIO_ACCOUNT_TAG_KEY: "lambda-x/SECONDARY"}

    with pytest.raises(AccountResolutionError) as exc_info:
        manager.resolve_test_environment("env", {"lambda-x": {"PRIMARY"}})
    msg = str(exc_info.value)
    assert "PRIMARY" in msg
    assert "Missing tag(s)" in msg
    assert "env init" in msg


def test_resolve_test_environment_propagates_no_accounts(manager):
    """When no accounts at all, AccountResolutionError surfaces with env-init hint."""
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = []

    with pytest.raises(AccountResolutionError, match="No accounts in OU"):
        manager.resolve_test_environment("env", {"lambda-x": {"PRIMARY"}})


def test_resolve_test_environment_bulk_first_failure_names_scenario(manager):
    """When passing 3 scenarios and the 1st fails, the error names that scenario."""
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = []  # no accounts at all

    with pytest.raises(AccountResolutionError) as exc_info:
        manager.resolve_test_environment(
            "env",
            {
                "lambda-x": {"PRIMARY"},
                "lambda-y": {"PRIMARY"},
                "lambda-z": {"PRIMARY"},
            },
        )
    # First scenario in iteration order should be named in the error.
    assert "lambda-x" in str(exc_info.value)


def test_list_scenario_accounts_returns_only_tagged(manager):
    _ou(manager)
    manager._org.list_accounts_in_ou.return_value = [
        {"Id": "111", "Email": "a@b.com", "Status": "ACTIVE"},
        {"Id": "222", "Email": "c@d.com", "Status": "ACTIVE"},
        {"Id": "333", "Email": "e@f.com", "Status": "ACTIVE"},
    ]
    manager._org.get_tags.side_effect = [
        {SCENARIO_ACCOUNT_TAG_KEY: "lambda-x/PRIMARY"},
        {"EnvironmentId": "old-env"},  # legacy tag, ignored
        {SCENARIO_ACCOUNT_TAG_KEY: "malformed-no-slash"},  # bad value, skipped
    ]

    out = manager.list_scenario_accounts("env")

    assert len(out) == 1
    assert out[0].account_id == "111"
    assert out[0].scenario_name == "lambda-x"
    assert out[0].account_tag == "PRIMARY"


# ── terminate_environment ──


def test_terminate_environment_detaches_scps_from_accounts(manager):
    """Verify terminate_environment detaches SCPs from each account before moving to root."""
    org_info = _make_org_info()
    manager._org.get_org_info.return_value = org_info
    manager._org.find_ou_by_name.return_value = "ou-test"
    manager._org.list_accounts_in_ou.side_effect = [
        [{"Id": "111", "Status": "ACTIVE"}, {"Id": "222", "Status": "ACTIVE"}],
        [],  # second call for "remaining" check
    ]
    manager._org.get_tags.return_value = {}  # no SHA tag → not skipped

    results = manager.terminate_environment("env")

    # detach_all_scps called for each account AND the OU
    assert manager._org.detach_all_scps.call_count == 3
    manager._org.detach_all_scps.assert_any_call("111")
    manager._org.detach_all_scps.assert_any_call("222")
    manager._org.detach_all_scps.assert_any_call("ou-test")
    # Accounts closed
    assert results == {"111": "CLOSED", "222": "CLOSED"}


def test_terminate_does_not_skip_accounts_with_sha_tag(mocker):
    """The vestigial SHA tag (written at creation, never removed) must not gate terminate."""
    mgr = AccountManager()
    mgr._org = MagicMock()
    mgr._org.get_org_info.return_value = SimpleNamespace(
        root_id="r-root", management_account_email="a@b.com"
    )
    mgr._org.find_ou_by_name.return_value = "ou-123"
    mgr._org.list_accounts_in_ou.side_effect = [
        [{"Id": "111111111111", "Status": "ACTIVE", "Email": "x@y.com"}],
        [],  # second call after handling: OU now empty
    ]
    mgr._org.get_tags.return_value = {SCENARIO_SHA_TAG_KEY: "some-scenario"}
    mgr._org.close_account = MagicMock()

    results = mgr.terminate_environment("my-ou")

    assert results["111111111111"] == "CLOSED"
    mgr._org.close_account.assert_called_once_with("111111111111")


# ── ensure_region_restriction_scp ──


def test_ensure_region_restriction_scp_delegates_to_org(manager):
    """The public seam forwards verbatim to OrganizationsClient."""
    manager.ensure_region_restriction_scp("lambda-x", ["us-east-1", "eu-west-1"], ["111", "222"])

    manager._org.ensure_region_restriction_scp.assert_called_once_with(
        "lambda-x", ["us-east-1", "eu-west-1"], ["111", "222"]
    )


# ── ensure_regions_enabled ──


@pytest.fixture()
def slept(monkeypatch):
    """Record region-poll sleeps and advance the manager's monotonic clock by them."""
    durations: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        durations.append(seconds)

    monkeypatch.setattr(
        "aws_bench.account_management.manager.time.monotonic", lambda: sum(durations)
    )
    monkeypatch.setattr("aws_bench.account_management.manager.asyncio.sleep", fake_sleep)
    return durations


def _stub_region_statuses(manager, statuses: dict[str, list[str]]) -> None:
    """Replay each region's RegionOptStatus sequence, then repeat its last status."""
    replay = {
        region: itertools.chain(sequence, itertools.repeat(sequence[-1]))
        for region, sequence in statuses.items()
    }

    def get_region_opt_status(_account_id: str, region: str) -> str:
        return next(replay[region])

    manager._org.get_region_opt_status.side_effect = get_region_opt_status


def test_ensure_regions_enabled_skips_enabled_regions(manager, slept):
    _stub_region_statuses(manager, {"us-east-1": ["ENABLED_BY_DEFAULT"], "eu-south-2": ["ENABLED"]})

    asyncio.run(manager.ensure_regions_enabled("111", ["us-east-1", "eu-south-2"]))

    manager._org.enable_region.assert_not_called()
    assert slept == []


def test_ensure_regions_enabled_requests_once_and_polls_regions_together(manager, slept):
    """Each region is requested once, and every pass reads all pending regions."""
    _stub_region_statuses(
        manager,
        {
            "eu-south-2": ["DISABLED", "ENABLING", "ENABLED"],
            # Still DISABLED on the pass after the request; must not be requested again.
            "ap-east-2": ["DISABLED", "DISABLED", "ENABLED"],
        },
    )

    asyncio.run(manager.ensure_regions_enabled("111", ["eu-south-2", "ap-east-2"]))

    assert [c.args for c in manager._org.enable_region.call_args_list] == [
        ("111", "eu-south-2"),
        ("111", "ap-east-2"),
    ]
    assert [c.args[1] for c in manager._org.get_region_opt_status.call_args_list] == [
        "eu-south-2",
        "ap-east-2",
    ] * 3
    assert slept == [REGION_OPT_IN_POLL_INTERVAL_SEC] * 2


def test_ensure_regions_enabled_rejects_disabling_region(manager, slept):
    _stub_region_statuses(manager, {"eu-south-2": ["DISABLING"]})

    with pytest.raises(
        AccountManagementError, match="Account 111 region eu-south-2: cannot enable from DISABLING"
    ):
        asyncio.run(manager.ensure_regions_enabled("111", ["eu-south-2"]))

    manager._org.enable_region.assert_not_called()


def test_ensure_regions_enabled_times_out_on_one_shared_deadline(manager, slept):
    """Regions still pending at the deadline are named; a DISABLED region is requested once."""
    _stub_region_statuses(manager, {"eu-south-2": ["DISABLED"], "ap-east-2": ["ENABLING"]})

    with pytest.raises(AccountManagementError, match="eu-south-2, ap-east-2; rerun env init"):
        asyncio.run(manager.ensure_regions_enabled("111", ["eu-south-2", "ap-east-2"]))

    manager._org.enable_region.assert_called_once_with("111", "eu-south-2")
    assert sum(slept) == REGION_OPT_IN_TIMEOUT_SEC


def test_ensure_regions_enabled_propagates_enable_error(manager, slept):
    _stub_region_statuses(manager, {"eu-south-2": ["DISABLED"]})
    error = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "not authorized"}},
        "EnableRegion",
    )
    manager._org.enable_region.side_effect = error

    with pytest.raises(ClientError) as caught:
        asyncio.run(manager.ensure_regions_enabled("111", ["eu-south-2"]))

    assert caught.value is error
    assert slept == []


def _enable_region_conflict() -> ClientError:
    message = "Account 111 and region eu-south-2 is currently being enabled or disabled."
    return ClientError({"Error": {"Code": "ConflictException", "Message": message}}, "EnableRegion")


def test_ensure_regions_enabled_waits_after_conflict_while_region_is_switching(manager, slept):
    _stub_region_statuses(manager, {"eu-south-2": ["DISABLED", "ENABLING", "ENABLED"]})
    manager._org.enable_region.side_effect = _enable_region_conflict()

    asyncio.run(manager.ensure_regions_enabled("111", ["eu-south-2"]))

    manager._org.enable_region.assert_called_once_with("111", "eu-south-2")
    assert slept == [REGION_OPT_IN_POLL_INTERVAL_SEC] * 2


def test_ensure_regions_enabled_resends_after_conflict_when_region_is_still_off(manager, slept):
    _stub_region_statuses(manager, {"eu-south-2": ["DISABLED", "DISABLED", "ENABLING", "ENABLED"]})
    manager._org.enable_region.side_effect = [_enable_region_conflict(), None]

    asyncio.run(manager.ensure_regions_enabled("111", ["eu-south-2"]))

    assert manager._org.enable_region.call_count == 2
    assert slept == [REGION_OPT_IN_POLL_INTERVAL_SEC] * 3


def test_ensure_regions_enabled_rejects_disabling_after_conflict(manager, slept):
    _stub_region_statuses(manager, {"eu-south-2": ["DISABLED", "DISABLING"]})
    manager._org.enable_region.side_effect = _enable_region_conflict()

    with pytest.raises(
        AccountManagementError, match="Account 111 region eu-south-2: cannot enable from DISABLING"
    ):
        asyncio.run(manager.ensure_regions_enabled("111", ["eu-south-2"]))

    manager._org.enable_region.assert_called_once_with("111", "eu-south-2")


def test_ensure_regions_enabled_only_validates_preexisting_accounts(manager):
    manager._preexisting = MagicMock()
    manager._preexisting.accounts = {"scenario": {"PRIMARY": "111"}}

    asyncio.run(manager.ensure_regions_enabled("111", ["eu-south-2"]))
    with pytest.raises(AccountResolutionError):
        asyncio.run(manager.ensure_regions_enabled("222", ["eu-south-2"]))

    manager._org.get_region_opt_status.assert_not_called()
    manager._org.enable_region.assert_not_called()


# ── contamination helpers ──


def test_mark_contaminated_tags_account(mocker):
    mgr = AccountManager()
    mgr._org = MagicMock()
    mgr._org.tag_resource = AsyncMock()
    asyncio.run(mgr.mark_contaminated("111111111111"))
    mgr._org.tag_resource.assert_awaited_once_with("111111111111", CONTAMINATED_TAG_KEY, "true")


def test_clear_contaminated_untags_account(mocker):
    mgr = AccountManager()
    mgr._org = MagicMock()
    mgr._org.untag_resource = AsyncMock()
    asyncio.run(mgr.clear_contaminated("111111111111"))
    mgr._org.untag_resource.assert_awaited_once_with("111111111111", [CONTAMINATED_TAG_KEY])


def test_get_contaminated_accounts_returns_tagged_subset():
    mgr = AccountManager()
    mgr._org = MagicMock()
    mgr._org.get_tags.side_effect = lambda aid: (
        {CONTAMINATED_TAG_KEY: "true"} if aid == "222222222222" else {}
    )
    assert mgr.get_contaminated_accounts(["111111111111", "222222222222"]) == ["222222222222"]


def test_mark_contaminated_retries_then_reraises(monkeypatch):
    mgr = AccountManager()
    mgr._org = MagicMock()
    mgr._org.tag_resource = AsyncMock(side_effect=RuntimeError("org throttled"))
    monkeypatch.setattr(AccountManager.mark_contaminated.retry, "wait", wait_none())  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="org throttled"):
        asyncio.run(mgr.mark_contaminated("111111111111"))
    # Retried up to the attempt cap, then re-raised (reraise=True).
    assert mgr._org.tag_resource.await_count == CONTAMINATION_TAG_MAX_ATTEMPTS


def test_clear_contaminated_retries_then_reraises(monkeypatch):
    mgr = AccountManager()
    mgr._org = MagicMock()
    mgr._org.untag_resource = AsyncMock(side_effect=RuntimeError("org throttled"))
    monkeypatch.setattr(AccountManager.clear_contaminated.retry, "wait", wait_none())  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="org throttled"):
        asyncio.run(mgr.clear_contaminated("111111111111"))
    assert mgr._org.untag_resource.await_count == CONTAMINATION_TAG_MAX_ATTEMPTS
