"""Account manager — orchestrates org and account lifecycle."""

from __future__ import annotations

import asyncio

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from aws_bench.account_management.constants import (
    CONTAMINATED_TAG_KEY,
    CONTAMINATION_TAG_MAX_ATTEMPTS,
    EMAIL_COLLISION_MAX_ATTEMPTS,
    SCENARIO_ACCOUNT_TAG_KEY,
    SCENARIO_SHA_TAG_KEY,
)
from aws_bench.account_management.exceptions import (
    AccountCreationError,
    AccountResolutionError,
    DuplicateScenarioAccountError,
    TestEnvironmentNotFoundError,
)
from aws_bench.account_management.models import (
    OrgInfo,
    ScenarioAccount,
    TestEnvironment,
)
from aws_bench.account_management.organizations import OrganizationsClient
from aws_bench.account_management.preexisting import (
    PreexistingEnvironmentConfig,
    PreexistingStateStore,
    active_account_config,
)
from aws_bench.account_management.utils import generate_account_email
from aws_bench.logging.logger import get_logger

logger = get_logger(__name__)


def _is_email_collision(exc: BaseException) -> bool:
    """Return True for an EMAIL_ALREADY_EXISTS account-creation failure.

    It surfaces as AccountCreationError (a poll FailureReason, not a ClientError),
    so it can only be matched on the message.
    """
    return isinstance(exc, AccountCreationError) and "EMAIL_ALREADY_EXISTS" in str(exc)


def format_scenario_tag_value(scenario_name: str, account_tag: str) -> str:
    """Build the canonical ``aws-bench:scenario`` tag value."""
    return f"{scenario_name}/{account_tag}"


def parse_scenario_tag_value(value: str) -> tuple[str, str] | None:
    """Split a tag value into ``(scenario_name, account_tag)``.

    Returns None if the value does not have the expected ``<name>/<tag>`` shape.
    """
    parts = value.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


class AccountManager:
    """Orchestrates the testing environment lifecycle."""

    def __init__(self) -> None:
        """Initialize the account manager."""
        self._org = OrganizationsClient()
        active = active_account_config()
        self._preexisting: PreexistingEnvironmentConfig | None = active[0] if active else None
        self._preexisting_path = active[1] if active else None
        self._state_store = (
            PreexistingStateStore(self._preexisting.resolve_state_file(self._preexisting_path))
            if self._preexisting is not None and self._preexisting_path is not None
            else None
        )

    @property
    def is_preexisting(self) -> bool:
        """Whether account lifecycle is owned by an external control plane."""
        return self._preexisting is not None

    @property
    def runner_role(self) -> str | None:
        """Account-local runner role configured for pre-existing mode."""
        return self._preexisting.runner_role if self._preexisting is not None else None

    # ── Init ──

    def init_organization(self, ou_name: str) -> str:
        """One-time setup: enable Organizations and create the testing-environment OU.

        Idempotent — safe to call multiple times. Returns the OU id.
        """
        if self._preexisting is not None:
            self._require_preexisting_name(ou_name)
            logger.info("Pre-existing account mode: skipping Organization, OU, and SCP creation.")
            return "preexisting"

        self._org.create_organization()
        org_info = self._org.get_org_info()

        existing = self._org.find_ou_by_name(org_info.root_id, ou_name)
        if existing:
            logger.info(f"Testing environment OU '{ou_name}' already exists: {existing}")
            ou_id = existing
        else:
            ou_id = self._org.create_ou(org_info.root_id, ou_name)

        self._org.ensure_org_role_protection_scp(ou_id)
        return ou_id

    def _require_ou(self, org_info: OrgInfo, ou_name: str) -> str:
        """Return ou_id or raise if not initialized."""
        ou_id = self._org.find_ou_by_name(org_info.root_id, ou_name)
        if not ou_id:
            raise TestEnvironmentNotFoundError(
                f"No testing environment OU '{ou_name}' found. Run env init first."
            )
        return ou_id

    def _iter_scenario_accounts(self, ou_id: str, scenario_name: str) -> list[tuple[str, str]]:
        """Return ``(account_tag, account_id)`` for ACTIVE accounts in the OU.

        Filters to accounts whose ``aws-bench:scenario`` tag belongs to
        ``scenario_name``.
        """
        out: list[tuple[str, str]] = []
        for acct in self._org.list_accounts_in_ou(ou_id):
            if acct["Status"] != "ACTIVE":
                continue
            value = self._org.get_tags(acct["Id"]).get(SCENARIO_ACCOUNT_TAG_KEY)
            parsed = parse_scenario_tag_value(value) if value else None
            if parsed is None or parsed[0] != scenario_name:
                continue
            out.append((parsed[1], acct["Id"]))
        return out

    def resolve_test_environment(
        self,
        ou_name: str,
        required_by_scenario: dict[str, set[str]] | None = None,
        statuses: frozenset[str] | None = frozenset({"ACTIVE"}),
    ) -> TestEnvironment:
        """Resolve the OU into a ``TestEnvironment`` with its scenario accounts.

        Lists scenario-tagged accounts in ``ou_name`` once, sorted by
        ``(scenario_name, account_tag)``. ``statuses`` keeps only accounts whose
        AWS status is in the set (default ``{"ACTIVE"}``); pass ``None`` to keep
        every status. When ``required_by_scenario`` is given
        (``{scenario_name: {tag}}``), keeps only those scenarios' accounts and
        requires each named scenario to have an account for each required tag.

        Raises:
            AccountResolutionError: A required scenario has no account, or is
                missing a required tag.
            DuplicateScenarioAccountError: Two accounts share a
                ``(scenario, account_tag)`` pair. Checked across every tagged
                account in the OU, not just ``required_by_scenario``.
            TestEnvironmentNotFoundError: The OU does not exist.
        """
        if self._preexisting is not None:
            self._require_preexisting_name(ou_name)
            return self._preexisting.to_test_environment(required_by_scenario=required_by_scenario)

        org_info = self._org.get_org_info()
        ou_id = self._require_ou(org_info, ou_name)

        all_accounts = sorted(
            (
                account
                for account in self._list_scenario_accounts_by_ou_id(ou_id)
                if statuses is None or account.status in statuses
            ),
            key=lambda a: (a.scenario_name, a.account_tag),
        )
        by_scenario: dict[str, dict[str, ScenarioAccount]] = {}
        for account in all_accounts:
            tag_map = by_scenario.setdefault(account.scenario_name, {})
            if account.account_tag in tag_map:
                raise DuplicateScenarioAccountError(
                    f"Account-tag {account.account_tag!r} for scenario "
                    f"{account.scenario_name!r} resolves to multiple accounts "
                    f"({tag_map[account.account_tag].account_id}, {account.account_id})."
                )
            tag_map[account.account_tag] = account

        if required_by_scenario is None:
            return TestEnvironment(org=org_info, ou_id=ou_id, ou_name=ou_name, accounts=by_scenario)

        selected: dict[str, dict[str, ScenarioAccount]] = {}
        for scenario_name, required_tags in required_by_scenario.items():
            resolved = by_scenario.get(scenario_name, {})
            if not resolved:
                raise AccountResolutionError(
                    f"No accounts in OU {ou_name!r} found tagged with "
                    f"{SCENARIO_ACCOUNT_TAG_KEY}=<{scenario_name}/...>. "
                    f"Run 'aws-bench env init' to provision."
                )
            missing = required_tags - resolved.keys()
            if missing:
                raise AccountResolutionError(
                    f"Scenario {scenario_name!r} declares account_tags="
                    f"{sorted(required_tags)} but the OU only has accounts for "
                    f"{sorted(resolved.keys())}. Missing tag(s): {sorted(missing)}. "
                    f"Run 'aws-bench env init --env-name {ou_name}' to provision "
                    f"the missing accounts."
                )
            selected[scenario_name] = {tag: resolved[tag] for tag in sorted(required_tags)}

        return TestEnvironment(org=org_info, ou_id=ou_id, ou_name=ou_name, accounts=selected)

    async def ensure_scenario_accounts(
        self, ou_name: str, scenario_name: str, account_tags: set[str]
    ) -> dict[str, str]:
        """Ensure one account per ``(scenario_name, account_tag)`` pair.

        Lists accounts in ``ou_name`` once, reuses any already tagged with
        ``aws-bench:scenario = <scenario_name>/<account_tag>``, then creates
        and tags accounts for the missing tags. Returns
        ``{account_tag: account_id}``.

        Raises:
            DuplicateScenarioAccountError: Two accounts share the same
                ``(scenario, account_tag)`` pair.
            TestEnvironmentNotFoundError: The OU does not exist.
        """
        if self._preexisting is not None:
            self._require_preexisting_name(ou_name)
            configured = self._preexisting.accounts.get(scenario_name)
            if configured is None:
                raise AccountResolutionError(
                    f"Scenario {scenario_name!r} is not present in the pre-existing account config."
                )
            missing = account_tags - configured.keys()
            if missing:
                raise AccountResolutionError(
                    f"Scenario {scenario_name!r} is missing pre-existing account "
                    f"tag(s): {sorted(missing)}"
                )
            return {tag: configured[tag] for tag in sorted(account_tags)}

        org_info = self._org.get_org_info()
        ou_id = self._require_ou(org_info, ou_name)

        existing: dict[str, str] = {}
        for tag, aid in self._iter_scenario_accounts(ou_id, scenario_name):
            if tag not in account_tags:
                continue
            if tag in existing:
                raise DuplicateScenarioAccountError(
                    f"Account-tag {tag!r} for scenario {scenario_name!r} "
                    f"resolves to multiple accounts ({existing[tag]}, {aid})."
                )
            existing[tag] = aid

        result: dict[str, str] = {}
        for tag in sorted(account_tags):
            if tag in existing:
                logger.debug(f"Reusing account {existing[tag]} for {scenario_name}/{tag}.")
                result[tag] = existing[tag]
            else:
                result[tag] = await self._create_scenario_account(
                    org_info, ou_id, scenario_name, tag
                )
        return result

    @retry(
        retry=retry_if_exception(_is_email_collision),
        wait=wait_random_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(EMAIL_COLLISION_MAX_ATTEMPTS),
        reraise=True,
    )
    async def _create_scenario_account(
        self,
        org_info: OrgInfo,
        ou_id: str,
        scenario_name: str,
        account_tag: str,
    ) -> str:
        # Re-entered on EMAIL_ALREADY_EXISTS retry, so the email regenerates
        # rather than reusing the colliding address.
        domain = org_info.management_account_email.split("@")[1]
        email = generate_account_email(domain, scenario_name, account_tag)
        account_name = f"{scenario_name}-{account_tag}"
        account_id = await self._org.create_account(name=account_name, email=email)
        await self._org.move_account_to_ou(account_id, ou_id)
        await self._org.tag_resource(
            account_id,
            SCENARIO_ACCOUNT_TAG_KEY,
            format_scenario_tag_value(scenario_name, account_tag),
        )
        await self._org.tag_resource(
            account_id,
            SCENARIO_SHA_TAG_KEY,
            scenario_name,
        )
        return account_id

    def _list_scenario_accounts_by_ou_id(self, ou_id: str) -> list[ScenarioAccount]:
        """List every scenario-tagged account under a resolved OU id.

        Accounts without a parseable ``aws-bench:scenario`` tag are skipped.
        Takes the OU id so callers that already resolved it avoid re-fetching
        the org and OU.
        """
        accounts: list[ScenarioAccount] = []
        for acct in self._org.list_accounts_in_ou(ou_id):
            value = self._org.get_tags(acct["Id"]).get(SCENARIO_ACCOUNT_TAG_KEY)
            if value is None:
                continue
            parsed = parse_scenario_tag_value(value)
            if parsed is None:
                logger.debug(
                    f"Account {acct['Id']} has malformed {SCENARIO_ACCOUNT_TAG_KEY} "
                    f"tag {value!r}; skipping."
                )
                continue
            scenario_name, account_tag = parsed
            accounts.append(
                ScenarioAccount(
                    account_id=acct["Id"],
                    email=acct["Email"],
                    scenario_name=scenario_name,
                    account_tag=account_tag,
                    status=acct["Status"],
                )
            )
        return accounts

    def list_scenario_accounts(self, ou_name: str) -> list[ScenarioAccount]:
        """List every scenario-tagged account in the OU.

        Accounts without a parseable ``aws-bench:scenario`` tag are skipped.
        """
        if self._preexisting is not None:
            environment = self.resolve_test_environment(ou_name)
            return [account for tags in environment.accounts.values() for account in tags.values()]

        org_info = self._org.get_org_info()
        ou_id = self._require_ou(org_info, ou_name)
        return self._list_scenario_accounts_by_ou_id(ou_id)

    def ensure_region_restriction_scp(
        self, scenario_name: str, allowed_regions: list[str], account_ids: list[str]
    ) -> None:
        """Lock ``account_ids`` to ``allowed_regions`` via a per-scenario SCP.

        Public seam over :class:`OrganizationsClient` so callers (the CLI and
        the trial lifecycle) don't reach into ``_org`` directly. Idempotent:
        reuses the policy by name, updates its content when the region set
        changes, and skips accounts that already have it attached.
        """
        if self._preexisting is not None:
            self._validate_allowlisted_accounts(account_ids)
            logger.info(
                "Pre-existing account mode: external IaC owns the region restriction "
                "for %s; skipping SCP mutation.",
                scenario_name,
            )
            return
        self._org.ensure_region_restriction_scp(scenario_name, allowed_regions, account_ids)

    @retry(
        wait=wait_random_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(CONTAMINATION_TAG_MAX_ATTEMPTS),
        reraise=True,
    )
    async def mark_contaminated(self, account_id: str) -> None:
        """Flag an account contaminated after a failed reset. Presence is the signal.

        Retries transient Organizations throttling; re-raises on exhaustion so the
        caller can decide (log for a failed reset, surface for a succeeded one).
        """
        if self._state_store is not None:
            self._validate_allowlisted_accounts([account_id])
            await asyncio.to_thread(self._state_store.mark, account_id)
            return
        await self._org.tag_resource(account_id, CONTAMINATED_TAG_KEY, "true")

    @retry(
        wait=wait_random_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(CONTAMINATION_TAG_MAX_ATTEMPTS),
        reraise=True,
    )
    async def clear_contaminated(self, account_id: str) -> None:
        """Remove the contamination flag (idempotent: a no-op if the tag is absent).

        Retries transient Organizations throttling; re-raises on exhaustion.
        """
        if self._state_store is not None:
            self._validate_allowlisted_accounts([account_id])
            await asyncio.to_thread(self._state_store.clear, account_id)
            return
        await self._org.untag_resource(account_id, [CONTAMINATED_TAG_KEY])

    def get_contaminated_accounts(self, account_ids: list[str]) -> list[str]:
        """Return the subset of ``account_ids`` currently carrying the contamination tag."""
        if self._state_store is not None:
            self._validate_allowlisted_accounts(account_ids)
            contaminated = self._state_store.contaminated()
            return [account_id for account_id in account_ids if account_id in contaminated]
        return [aid for aid in account_ids if CONTAMINATED_TAG_KEY in self._org.get_tags(aid)]

    def terminate_environment(
        self,
        ou_name: str,
        no_close: bool = False,
        account_filter: list[str] | None = None,
    ) -> dict[str, str]:
        """Terminate the testing environment: close accounts, delete OU.

        Moves accounts to org root, closes them, detaches SCPs, and deletes
        the OU. Returns ``{account_id: status}`` for each account.

        If ``no_close`` is True, removes the scenario tag and moves accounts
        to root without closing them.

        Accounts enter a 90-day suspension period after closure.
        """
        if self._preexisting is not None:
            raise RuntimeError(
                "env terminate is disabled in pre-existing account mode; external IaC owns "
                "the accounts and Organizational Unit"
            )

        org_info = self._org.get_org_info()
        ou_id = self._require_ou(org_info, ou_name)

        accounts = self._org.list_accounts_in_ou(ou_id)
        results: dict[str, str] = {}

        for acct in accounts:
            account_id = acct["Id"]
            if acct["Status"] != "ACTIVE":
                results[account_id] = f"SKIPPED ({acct['Status']})"
                continue
            if account_filter and account_id not in account_filter:
                continue

            try:
                self._org.detach_all_scps(account_id)
                self._org.move_account_to_root(account_id, ou_id, org_info.root_id)
                if no_close:
                    # Just remove the scenario tag, don't close
                    asyncio.run(self._org.untag_resource(account_id, [SCENARIO_ACCOUNT_TAG_KEY]))
                    results[account_id] = "UNTAGGED"
                else:
                    self._org.close_account(account_id)
                    results[account_id] = "CLOSED"
            except Exception as e:
                results[account_id] = f"FAILED ({e})"
                logger.error(f"Failed to terminate account {account_id}: {e}")

        # Detach SCPs and delete the OU if all accounts handled
        self._org.detach_all_scps(ou_id)
        remaining = self._org.list_accounts_in_ou(ou_id)
        if not remaining:
            try:
                self._org.delete_organizational_unit(ou_id)
            except Exception as e:
                logger.warning(f"Could not delete OU {ou_id}: {e}")

        return results

    def _require_preexisting_name(self, ou_name: str) -> None:
        assert self._preexisting is not None
        if ou_name != self._preexisting.name:
            raise TestEnvironmentNotFoundError(
                f"Active pre-existing config names environment {self._preexisting.name!r}, "
                f"not {ou_name!r}."
            )

    def _validate_allowlisted_accounts(self, account_ids: list[str]) -> None:
        assert self._preexisting is not None
        allowed = {
            account_id
            for tags in self._preexisting.accounts.values()
            for account_id in tags.values()
        }
        unexpected = set(account_ids) - allowed
        if unexpected:
            raise AccountResolutionError(
                f"Account(s) are not in the active pre-existing allowlist: {sorted(unexpected)}"
            )
