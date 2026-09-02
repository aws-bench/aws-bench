"""Scenario provisioning orchestration.

Provisions accounts and submits quota increases for a set of scenarios:

  Provisioning — bounded by ``n_concurrent``. For each
    ``(scenario, account_tag)`` pair, ensure an account exists, wait for
    the org-access role, and submit each scenario's ``[[quotas]]`` without
    waiting for approval.

  Account-limit reaction — if account creation fails because the AWS
    Organizations "maximum number of accounts" limit is hit, file a Service
    Quotas increase to ``DEFAULT_ORG_ACCOUNT_QUOTA`` (skipping the submit if a
    request is already pending) and raise ``AccountLimitExceededError`` so the
    operator knows to re-run once the increase is approved.

  Approval wait — opt-in via ``wait_for_quotas``. Polls each quota's
    current effective value (``QuotaManager.verify_quotas``) until every
    quota meets its desired value or ``quota_timeout`` elapses. The same
    primitive verifies preconditions before a scenario runs, so the
    two sides agree on what "ready" means.

Clients pass an async ``on_event`` callback to receive typed
``ProvisionHookEvent`` objects during provisioning. UIs render live
progress against this without coupling to the orchestrator.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from botocore.exceptions import ClientError
from pydantic import BaseModel, Field

from aws_bench.account_management.constants import CFN_OPS_ROLE_NAME, ORG_ACCESS_ROLE
from aws_bench.account_management.exceptions import AccountCreationError
from aws_bench.account_management.manager import AccountManager
from aws_bench.account_management.preexisting import effective_cfn_role
from aws_bench.exceptions import OperationCancelled
from aws_bench.logging.logger import get_logger, log_context
from aws_bench.resource_management.fastscan.lambda_deploy import ensure_deployed
from aws_bench.resource_management.manager import ResourceManager
from aws_bench.resource_management.models import (
    QuotaConfiguration,
    QuotaIncreaseRequest,
    QuotaIncreaseResult,
    QuotaStatus,
)
from aws_bench.resource_management.quota_manager import (
    ORG_ACCOUNT_QUOTA_CODE,
    QuotaManager,
)
from aws_bench.resource_management.scanner import _FASTSCAN_LAMBDA, scan_method
from aws_bench.resource_management.snapshot.models import SnapshotResult
from aws_bench.scenario.config import QuotaIncrease, ScenarioManifest
from aws_bench.scenario.exceptions import (
    AccountLimitExceededError,
    InsufficientQuotaError,
    UnmetQuota,
)
from aws_bench.scenario.job import ScenarioJob
from aws_bench.scenario.scenario import Scenario
from aws_bench.utils.credentials_provider import CredentialProvider, build_session_name

logger = get_logger(__name__)

DEFAULT_QUOTA_POLL_INTERVAL = 60
DEFAULT_QUOTA_MAX_WAIT = 1800

# Value requested for the AWS Organizations "maximum number of accounts" quota
# when account creation hits the limit. Increases are subject to manual review
# (not auto-approved on new orgs), so this is filed reactively and the operator
# re-runs once it is granted.
DEFAULT_ORG_ACCOUNT_QUOTA = 15


class ProvisionEvent(Enum):
    """Lifecycle events for a single (scenario, account_tag) provisioning unit.

    Phase-transition events are named ``<RESOURCE>_START`` so a generic
    UI can render any provisioning step without knowing the specific
    resource type.
    """

    START = "start"
    ACCOUNT_START = "account_start"
    ROLE_START = "role_start"
    QUOTAS_START = "quotas_start"
    SNAPSHOT_START = "snapshot_start"
    END = "end"
    CANCEL = "cancel"


class ProvisionHookEvent(BaseModel):
    """Event object passed to provisioning lifecycle hooks.

    ``account_id`` is set once provisioning resolves it; ``error`` is set only on
    END for units that raised. ``succeeded`` is the authoritative END outcome —
    False also covers a cancel or a snapshot/quota failure that leaves ``error``
    None — so the overall progress count reflects successes, not attempts.
    """

    event: ProvisionEvent
    scenario_name: str
    account_tag: str
    account_id: str | None = None
    error: str | None = None
    succeeded: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


HookCallback = Callable[[ProvisionHookEvent], Awaitable[None]]


@dataclass
class SubmittedQuotaBatch:
    """A quota batch submitted during provisioning, revisited when waiting for approval."""

    scenario_name: str
    account_tag: str
    account_id: str
    config: QuotaConfiguration
    results: list[QuotaIncreaseResult] = field(default_factory=list)
    """Per-quota submit outcome, aligned with ``config.increases``.

    Anything but ``ALREADY_MET`` is not met yet; the non-wait path seeds
    ``ProvisioningSummary.unmet_quotas`` from these.
    """


@dataclass
class ProvisionedAccount:
    """Outcome of provisioning one (scenario, account_tag) account."""

    scenario_name: str
    account_tag: str
    account_id: str | None = None
    newly_created: bool = False
    """True if the account was freshly created (not reused from a previous init)."""
    error: Exception | None = None
    submitted_quotas: list[SubmittedQuotaBatch] = field(default_factory=list)
    submit_failures: list[Exception] = field(default_factory=list)
    snapshot_result: SnapshotResult | None = None
    """PRE_SETUP baseline outcome; ``None`` if capture never ran (earlier step failed)."""

    @property
    def provisioned(self) -> bool:
        """True iff the account provisioned cleanly and its baseline was captured.

        Requires no error, no quota submit failures, and (when capture ran) a
        successful snapshot.
        """
        if self.error is not None or self.submit_failures:
            return False
        return self.snapshot_result is None or self.snapshot_result.success


@dataclass
class ProvisioningSummary:
    """Aggregated result of provisioning a set of scenarios."""

    accounts: list[ProvisionedAccount] = field(default_factory=list)
    unmet_quotas: list[UnmetQuota] = field(default_factory=list)
    """Quotas not yet satisfied, each carrying its ``QuotaIncreaseResult.status``.

    Seeded from the submit outcome; the ``wait_for_quotas`` pass replaces it
    with whatever remained pending when polling ended."""

    waited: bool = False
    """Whether the ``wait_for_quotas`` pass ran; see :attr:`all_succeeded`."""

    @property
    def n_total(self) -> int:
        """Total accounts processed."""
        return len(self.accounts)

    @property
    def n_provisioned(self) -> int:
        """Accounts that completed all provisioning steps."""
        return sum(1 for a in self.accounts if a.provisioned)

    @property
    def n_failed(self) -> int:
        """Accounts that failed at any provisioning step."""
        return self.n_total - self.n_provisioned

    @property
    def all_succeeded(self) -> bool:
        """True iff every account provisioned and no quota blocks the run.

        A pending quota fails the run only if we waited for it; an errored
        request (``status.is_failure``) always fails it.
        """
        if self.n_failed != 0:
            return False
        if any(q.result.status.is_failure for q in self.unmet_quotas):
            return False
        return self.waited is False or not self.unmet_quotas


async def provision_scenarios(
    scenarios: list[Scenario],
    ou_name: str,
    *,
    n_concurrent: int,
    wait_for_quotas: bool,
    quota_timeout: int = DEFAULT_QUOTA_MAX_WAIT,
    poll_interval: int = DEFAULT_QUOTA_POLL_INTERVAL,
    account_manager: AccountManager | None = None,
    quota_manager: QuotaManager | None = None,
    cred_provider: CredentialProvider | None = None,
    on_event: HookCallback | None = None,
) -> ProvisioningSummary:
    """Provision accounts and submit quota increases for ``scenarios``.

    Runs at most ``n_concurrent`` accounts in parallel. If account creation
    fails because the AWS Organizations account limit is exceeded, files a
    Service Quotas increase (or reports an already-pending one) and raises
    ``AccountLimitExceededError`` — the org limit can't be raised in time, so
    the operator re-runs once it's approved.

    When ``wait_for_quotas`` is True, the current effective quota value is then
    polled until every quota meets its desired value or ``quota_timeout``
    elapses. The same primitive (``QuotaManager.verify_quotas``) gates
    both this exit and ``ScenarioJob``'s precondition check, so the two
    sides cannot drift.

    Each account's pristine PRE_SETUP baseline is captured as the final step
    of provisioning that account (the baseline ``env cleanup`` later
    subtracts). A capture failure fails that account's ``provisioned`` status.

    ``on_event`` (optional) receives one typed event per lifecycle
    transition per account. Exceptions raised by the callback are logged
    and swallowed so a faulty observer cannot break provisioning.

    Raises:
        AccountLimitExceededError: Account creation hit the org account limit;
            a quota increase was filed (or was already pending).
    """
    creds = cred_provider or CredentialProvider.get()
    am = account_manager or AccountManager()
    qm = quota_manager or QuotaManager(creds)

    # Only deploy when the Lambda backend is selected. A deploy failure halts provisioning
    # rather than proceeding to a scan that will fail later.
    if scan_method() == _FASTSCAN_LAMBDA and am.is_preexisting is not True:
        await asyncio.to_thread(ensure_deployed)

    pairs = [(sc.manifest, tag) for sc in scenarios for tag in sc.manifest.scenario.account_tags]
    out = ProvisioningSummary()
    out.accounts = await _provision_all(
        pairs,
        am,
        qm,
        creds,
        ou_name,
        n_concurrent=n_concurrent,
        on_event=on_event,
    )
    _log_summary(out.accounts)

    # React to the org account-count ceiling: if any account failed because the
    # org is at its maximum-accounts limit, file the increase (or surface an
    # existing pending request) and abort with clear guidance.
    if any(_is_account_limit_exceeded(a.error) for a in out.accounts):
        await _handle_account_limit_exceeded(qm)

    # Seed from the submit outcome so the summary is honest without the wait pass.
    out.unmet_quotas = _unmet_from_submits(out.accounts)

    if wait_for_quotas:
        await run_quota_wait_pass(
            out,
            scenarios,
            cred_provider=creds,
            quota_timeout=quota_timeout,
            poll_interval=poll_interval,
            n_concurrent=n_concurrent,
        )

    return out


async def run_quota_wait_pass(
    summary: ProvisioningSummary,
    scenarios: list[Scenario],
    *,
    cred_provider: CredentialProvider,
    quota_timeout: int,
    poll_interval: int,
    n_concurrent: int,
    on_poll: Callable[[int, int, list[UnmetQuota]], None] | None = None,
) -> None:
    """Run the approval-wait pass and record its outcome on ``summary`` in place.

    Single owner of the ``waited`` / ``unmet_quotas`` contract so the quiet and
    progress paths cannot diverge. ``on_poll`` drives an optional progress bar.
    """
    summary.waited = True
    account_mappings = _build_account_mappings(summary.accounts)
    if not account_mappings:
        return
    try:
        await _await_quota_approvals(
            scenarios,
            account_mappings,
            cred_provider=cred_provider,
            quota_timeout=quota_timeout,
            poll_interval=poll_interval,
            n_concurrent=n_concurrent,
            on_poll=on_poll,
        )
        summary.unmet_quotas = []  # wait pass verified every quota met
    except InsufficientQuotaError as exc:
        summary.unmet_quotas = exc.failures


def _unmet_from_submits(accounts: list[ProvisionedAccount]) -> list[UnmetQuota]:
    """Build the not-yet-met quotas from each account's submit outcome.

    Anything but ``ALREADY_MET`` becomes an ``UnmetQuota``.
    """
    unmet: list[UnmetQuota] = []
    for acct in accounts:
        for batch in acct.submitted_quotas:
            for res in batch.results:
                if res.status == QuotaStatus.ALREADY_MET:
                    continue
                unmet.append(
                    UnmetQuota(
                        scenario_name=batch.scenario_name,
                        account_id=batch.account_id,
                        region=batch.config.region,
                        result=res,
                    )
                )
    return unmet


def _is_account_limit_exceeded(exc: Exception | None) -> bool:
    """True if ``exc`` indicates the org hit its maximum-accounts limit.

    Covers both surfaces: the async ``CreateAccount`` FailureReason
    (``ACCOUNT_LIMIT_EXCEEDED``, raised as ``AccountCreationError``) and the
    synchronous ``ConstraintViolationException`` from ``CreateAccount``
    ("you have exceeded the allowed number of AWS accounts").
    """
    if exc is None:
        return False
    text = str(exc).upper()
    if isinstance(exc, AccountCreationError) and "ACCOUNT_LIMIT_EXCEEDED" in text:
        return True
    if (
        isinstance(exc, ClientError)
        and exc.response.get("Error", {}).get("Code") == "ConstraintViolationException"
        and "ALLOWED NUMBER OF AWS ACCOUNTS" in text
    ):
        return True
    return False


async def _handle_account_limit_exceeded(quota_manager: QuotaManager) -> None:
    """File an org account-count increase (or find an existing one), then raise.

    Reacts to ``ACCOUNT_LIMIT_EXCEEDED`` during account creation: submits an
    increase to ``DEFAULT_ORG_ACCOUNT_QUOTA`` unless one is already pending
    (checked first, to avoid a duplicate request). Always raises
    ``AccountLimitExceededError`` — the limit can't be lifted in time, so init
    aborts with guidance to re-run after approval.

    Raises:
        AccountLimitExceededError: Always. The ``detail`` says whether an
            increase was filed or one was already pending.
    """
    result = await asyncio.to_thread(
        quota_manager.request_org_account_quota_if_absent, DEFAULT_ORG_ACCOUNT_QUOTA
    )
    if result.status == QuotaStatus.ALREADY_PENDING:
        detail = (
            f"A request to increase the org account limit is already pending "
            f"(organizations / {ORG_ACCOUNT_QUOTA_CODE})."
        )
    elif result.status.is_failure:
        detail = (
            f"Attempt to file an increase to {DEFAULT_ORG_ACCOUNT_QUOTA} failed: "
            f"{result.error_message or result.status.value}."
        )
    else:
        detail = (
            f"Filed a request to increase the org account limit to "
            f"{DEFAULT_ORG_ACCOUNT_QUOTA} (status: {result.status.value})."
        )
    logger.warning("Account limit exceeded. %s", detail)
    raise AccountLimitExceededError(detail)


def _build_account_mappings(
    accounts: list[ProvisionedAccount],
) -> dict[str, dict[str, str]]:
    """Map scenario_name -> account_tag -> account_id from successful provisions."""
    mappings: dict[str, dict[str, str]] = defaultdict(dict)
    for acct in accounts:
        if acct.account_id is not None and acct.error is None:
            mappings[acct.scenario_name][acct.account_tag] = acct.account_id
    return dict(mappings)


async def _emit(
    on_event: HookCallback | None,
    event: ProvisionEvent,
    *,
    scenario_name: str,
    account_tag: str,
    account_id: str | None = None,
    error: str | None = None,
    succeeded: bool = False,
) -> None:
    """Build a payload and invoke ``on_event``. Hook errors are swallowed."""
    if on_event is None:
        return
    payload = ProvisionHookEvent(
        event=event,
        scenario_name=scenario_name,
        account_tag=account_tag,
        account_id=account_id,
        error=error,
        succeeded=succeeded,
    )
    try:
        await on_event(payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("on_event for %s/%s raised: %s", scenario_name, account_tag, exc)


_ACCOUNT_CREATION_MAX_CONCURRENT = 3
_ACCOUNT_CONVERGENCE_DELAY_SEC = 180
"""Hard cap on concurrent account-creation calls regardless of ``n_concurrent``."""


async def _provision_all(
    pairs: list[tuple[ScenarioManifest, str]],
    account_manager: AccountManager,
    quota_manager: QuotaManager,
    cred_provider: CredentialProvider,
    ou_name: str,
    *,
    n_concurrent: int,
    on_event: HookCallback | None,
) -> list[ProvisionedAccount]:
    """Provision every (scenario, account_tag) pair in two phases.

    Phase 1 — Account creation, bounded by ``min(n_concurrent, _ACCOUNT_CREATION_MAX_CONCURRENT)``.
    Phase 2 — Role wait + quota submission + baseline capture, bounded by ``n_concurrent``.
    """
    # Phase 1: Create accounts with hard cap of 3 concurrent.
    logger.info(
        "Phase 1: Creating %d account(s) (max %d concurrent)",
        len(pairs),
        min(n_concurrent, _ACCOUNT_CREATION_MAX_CONCURRENT),
    )
    creation_sem = asyncio.Semaphore(min(n_concurrent, _ACCOUNT_CREATION_MAX_CONCURRENT))

    async def _create_one(scenario: ScenarioManifest, account_tag: str) -> ProvisionedAccount:
        async with creation_sem:
            with log_context(scenario.scenario.name), log_context(account_tag):
                return await _create_account(
                    account_manager,
                    ou_name,
                    scenario,
                    account_tag,
                    on_event,
                )

    async with asyncio.TaskGroup() as tg:
        creation_tasks = [tg.create_task(_create_one(sc, tag)) for sc, tag in pairs]
    creation_results = [t.result() for t in creation_tasks]

    # Phase 2: For successful accounts, run role wait + quota + snapshot.
    lifecycle_sem = asyncio.Semaphore(n_concurrent)

    # Build index from (scenario_name, account_tag) -> (ScenarioManifest, result)
    pair_map: dict[tuple[str, str], ScenarioManifest] = {
        (sc.scenario.name, tag): sc for sc, tag in pairs
    }

    async def _provision_one(result: ProvisionedAccount) -> ProvisionedAccount:
        async with lifecycle_sem:
            scenario = pair_map[(result.scenario_name, result.account_tag)]
            with log_context(result.scenario_name), log_context(result.account_tag):
                return await _provision_account_lifecycle(
                    quota_manager,
                    cred_provider,
                    scenario,
                    result,
                    on_event,
                    preexisting=account_manager.is_preexisting is True,
                    runner_role=account_manager.runner_role,
                )

    # Split into successes (proceed to Phase 2) and failures (pass through).
    succeeded = [r for r in creation_results if r.error is None]
    failed = [r for r in creation_results if r.error is not None]

    # Wait for newly created accounts to converge (services, IAM, etc.).
    # Freshly created org accounts need time for service subscriptions to activate.
    # Skip the delay if all accounts were reused (not freshly created).
    has_new_accounts = any(r.newly_created for r in succeeded)
    if succeeded and has_new_accounts:
        logger.info(
            "Waiting %ds for account convergence before provisioning...",
            _ACCOUNT_CONVERGENCE_DELAY_SEC,
        )
        await asyncio.sleep(_ACCOUNT_CONVERGENCE_DELAY_SEC)
    elif succeeded:
        logger.info("All accounts reused — skipping convergence delay")

    logger.info(
        "Phase 2: Provisioning %d account(s) (max %d concurrent)",
        len(succeeded),
        n_concurrent,
    )

    if succeeded:
        async with asyncio.TaskGroup() as tg:
            lifecycle_tasks = [tg.create_task(_provision_one(r)) for r in succeeded]
        lifecycle_results = [t.result() for t in lifecycle_tasks]
    else:
        lifecycle_results = []

    return lifecycle_results + failed


async def _create_account(
    account_manager: AccountManager,
    ou_name: str,
    scenario: ScenarioManifest,
    account_tag: str,
    on_event: HookCallback | None,
) -> ProvisionedAccount:
    """Phase 1: Create/ensure a single account and emit START + ACCOUNT_START events.

    On failure, records the error in ``result.error`` and emits END.
    On success, returns a result with ``account_id`` set (no END emitted yet —
    Phase 2 emits END after completing the lifecycle).
    """
    name = scenario.scenario.name
    result = ProvisionedAccount(scenario_name=name, account_tag=account_tag)
    cancelled = False

    async def emit(event: ProvisionEvent, **kw) -> None:
        await _emit(on_event, event, scenario_name=name, account_tag=account_tag, **kw)

    try:
        await emit(ProvisionEvent.START)

        await emit(ProvisionEvent.ACCOUNT_START)
        try:
            t0 = asyncio.get_event_loop().time()
            mapping = await account_manager.ensure_scenario_accounts(ou_name, name, {account_tag})
            result.account_id = mapping[account_tag]
            # Account creation takes 10+ seconds; reuse is near-instant.
            result.newly_created = (asyncio.get_event_loop().time() - t0) > 5.0
        except Exception as exc:  # noqa: BLE001
            logger.error("Account provisioning failed for %s/%s: %s", name, account_tag, exc)
            result.error = exc
            return result

        return result

    except (asyncio.CancelledError, OperationCancelled):
        cancelled = True
        await emit(ProvisionEvent.CANCEL, account_id=result.account_id)
        raise
    finally:
        # Only emit END here if Phase 1 itself failed (error set) or was cancelled.
        # Successful accounts will get their END from Phase 2.
        if result.error is not None or cancelled:
            await emit(
                ProvisionEvent.END,
                account_id=result.account_id,
                error=str(result.error) if result.error is not None else None,
                succeeded=False,
            )


def _ensure_cfn_ops_role(cred_provider: CredentialProvider, account_id: str) -> None:
    """Create the CFN operations role in the child account (idempotent).

    Passed as RoleARN on DeleteStack so deletion doesn't depend on the CDK
    bootstrap cfn-exec role. AdministratorAccess — same scope as cfn-exec.
    """
    session = cred_provider.get_session_for_account(
        account_id, ORG_ACCESS_ROLE, build_session_name("cfn-ops-role-setup")
    )
    iam = session.client("iam")

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "cloudformation.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        iam.create_role(
            RoleName=CFN_OPS_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="CloudFormation service execution role for stack lifecycle operations",
        )
        logger.debug("Created CFN ops role '%s' in account %s", CFN_OPS_ROLE_NAME, account_id)
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
        logger.debug("CFN ops role '%s' already exists in %s", CFN_OPS_ROLE_NAME, account_id)

    iam.attach_role_policy(
        RoleName=CFN_OPS_ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/AdministratorAccess",
    )


def _validate_cfn_ops_role(cred_provider: CredentialProvider, account_id: str) -> None:
    """Validate the externally managed CloudFormation execution role exists."""
    session = cred_provider.get_session_for_account(
        account_id, ORG_ACCESS_ROLE, build_session_name("cfn-ops-role-validate")
    )
    session.client("iam").get_role(RoleName=effective_cfn_role())


async def _provision_account_lifecycle(
    quota_manager: QuotaManager,
    cred_provider: CredentialProvider,
    scenario: ScenarioManifest,
    result: ProvisionedAccount,
    on_event: HookCallback | None,
    *,
    preexisting: bool = False,
    runner_role: str | None = None,
) -> ProvisionedAccount:
    """Phase 2: Role wait + quota submission + baseline capture for a provisioned account.

    Expects ``result.account_id`` to be set (Phase 1 succeeded).
    Emits ROLE_START, QUOTAS_START, SNAPSHOT_START, and END events.
    """
    assert result.account_id is not None, "Phase 2 requires a Phase 1 account_id"
    name = scenario.scenario.name
    account_tag = result.account_tag
    cancelled = False
    # Phase 1 guarantees this (see docstring); assert to narrow str | None -> str.
    assert result.account_id is not None
    account_id = result.account_id

    async def emit(event: ProvisionEvent, **kw) -> None:
        await _emit(on_event, event, scenario_name=name, account_tag=account_tag, **kw)

    try:
        await emit(ProvisionEvent.ROLE_START, account_id=account_id)
        if preexisting:
            try:
                session = cred_provider.get_session_for_account(
                    account_id,
                    ORG_ACCESS_ROLE,
                    build_session_name("runner-role-validate"),
                )
                await asyncio.to_thread(session.client("sts").get_caller_identity)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Runner role %s is unavailable in %s: %s",
                    runner_role,
                    account_id,
                    exc,
                )
                result.error = exc
                return result
        else:
            try:
                await asyncio.to_thread(cred_provider.wait_for_role, account_id, ORG_ACCESS_ROLE)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Role %s never became assumable in %s: %s",
                    ORG_ACCESS_ROLE,
                    account_id,
                    exc,
                )
                result.error = exc
                return result

        try:
            role_operation = _validate_cfn_ops_role if preexisting else _ensure_cfn_ops_role
            await asyncio.to_thread(role_operation, cred_provider, account_id)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "CFN ops role %s failed in %s: %s",
                "validation" if preexisting else "creation",
                account_id,
                exc,
            )
            result.error = exc
            return result

        await emit(ProvisionEvent.QUOTAS_START, account_id=account_id)
        for region, batch in _group_quotas_for_tag(scenario, account_tag).items():
            config = QuotaConfiguration(
                region=region,
                increases=[
                    QuotaIncreaseRequest(
                        service_code=q.service_code,
                        quota_code=q.quota_code,
                        desired_value=q.desired_value,
                    )
                    for q in batch
                ],
            )
            try:
                quota_operation = (
                    quota_manager.verify_quotas if preexisting else quota_manager.request_quotas
                )
                submit_results = await asyncio.to_thread(
                    quota_operation,
                    config,
                    account_id,
                    ORG_ACCESS_ROLE,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Quota submit failed for %s/%s in %s/%s: %s",
                    name,
                    account_tag,
                    account_id,
                    region,
                    exc,
                )
                result.submit_failures.append(exc)
                continue
            if preexisting:
                unmet = [
                    quota for quota in submit_results if quota.status != QuotaStatus.ALREADY_MET
                ]
                if unmet:
                    result.submit_failures.append(
                        InsufficientQuotaError(
                            [
                                UnmetQuota(
                                    scenario_name=name,
                                    account_id=account_id,
                                    region=region,
                                    result=quota,
                                )
                                for quota in unmet
                            ]
                        )
                    )
            result.submitted_quotas.append(
                SubmittedQuotaBatch(
                    scenario_name=name,
                    account_tag=account_tag,
                    account_id=account_id,
                    config=config,
                    results=submit_results,
                )
            )

        await emit(ProvisionEvent.SNAPSHOT_START, account_id=account_id)
        try:
            result.snapshot_result = await asyncio.to_thread(
                ResourceManager.capture_pre_setup_baseline,
                account_id,
                name,
                list(scenario.scenario.regions),
                cred_provider=cred_provider,
            )
        except Exception as exc:  # noqa: BLE001
            # Guard: a bug in capture must not abort the TaskGroup.
            logger.error("Baseline capture failed for %s/%s: %s", name, account_tag, exc)
            result.snapshot_result = SnapshotResult(
                account_id=account_id, success=False, error_message=str(exc)
            )

        return result

    except (asyncio.CancelledError, OperationCancelled):
        cancelled = True
        await emit(ProvisionEvent.CANCEL, account_id=account_id)
        raise
    finally:
        await emit(
            ProvisionEvent.END,
            account_id=account_id,
            error=str(result.error) if result.error is not None else None,
            succeeded=result.provisioned and not cancelled,
        )


async def _await_quota_approvals(
    scenarios: list[Scenario],
    account_mappings: dict[str, dict[str, str]],
    *,
    cred_provider: CredentialProvider,
    quota_timeout: int,
    poll_interval: int,
    n_concurrent: int,
    on_poll: Callable[[int, int, list[UnmetQuota]], None] | None = None,
) -> None:
    """Poll current quota values until every quota meets its desired value.

    Uses ``QuotaManager.verify_quotas`` (current effective value), not
    request-history polling — that's the single source of truth for
    "ready", so a successful exit here matches a successful precondition
    check elsewhere.

    Transient errors raised by ``verify_quotas`` (for example
    ``DeploymentError`` wrapping a throttle/credential refresh) are logged
    and retried until ``quota_timeout`` elapses.

    ``on_poll`` (optional) is invoked after each verify pass with
    ``(met_count, total_count, unmet)`` so callers can drive a progress
    UI. ``unmet`` is the list of remaining ``UnmetQuota`` (empty on the
    success-exit pass). Exceptions raised by the callback are logged and
    swallowed.

    Raises:
        InsufficientQuotaError: One or more quotas remained below their
            desired value past ``quota_timeout``. The exception carries
            every unmet quota in one pass.
    """
    total = _count_quotas_to_verify(scenarios, account_mappings)
    deadline = time.monotonic() + quota_timeout

    def _notify(met: int, unmet: list[UnmetQuota]) -> None:
        if on_poll is None:
            return
        try:
            on_poll(met, total, unmet)
        except Exception as cb_exc:  # noqa: BLE001
            logger.warning("on_poll callback raised: %s", cb_exc)

    while True:
        try:
            await ScenarioJob._check_quota_sufficiency(
                scenarios,
                account_mappings,
                cred_provider,
                n_concurrent=n_concurrent,
            )
            _notify(total, [])
            logger.info("All %d quota(s) verified — proceeding.", total)
            return
        except InsufficientQuotaError as exc:
            _notify(total - len(exc.failures), list(exc.failures))
            if time.monotonic() >= deadline:
                raise
            logger.info(
                "Still waiting for %d quota(s); polling again in %ds...",
                len(exc.failures),
                poll_interval,
            )
        except Exception as exc:  # noqa: BLE001
            if time.monotonic() >= deadline:
                raise
            logger.warning(
                "Transient verify_quotas failure: %s; retrying in %ds...",
                exc,
                poll_interval,
            )
        await asyncio.sleep(poll_interval)


def _count_quotas_to_verify(
    scenarios: list[Scenario],
    account_mappings: dict[str, dict[str, str]],
) -> int:
    """Count quotas that ``_check_quota_sufficiency`` would actually verify."""
    total = 0
    for sc in scenarios:
        mapping = account_mappings.get(sc.manifest.scenario.name, {})
        for q in sc.manifest.quotas:
            if q.account_tag in mapping:
                total += 1
    return total


def _group_quotas_for_tag(
    scenario: ScenarioManifest, account_tag: str
) -> dict[str, list[QuotaIncrease]]:
    """Return ``{region: [quotas]}`` for one (scenario, account_tag) pair."""
    groups: dict[str, list[QuotaIncrease]] = defaultdict(list)
    for q in scenario.quotas:
        if q.account_tag == account_tag:
            groups[q.region].append(q)
    return groups


def _log_summary(accounts: list[ProvisionedAccount]) -> None:
    failed = [a for a in accounts if a.error is not None]
    submit_failures = sum(len(a.submit_failures) for a in accounts)
    if failed:
        logger.warning(
            "Provisioning: %d/%d account(s) failed before quota submission.",
            len(failed),
            len(accounts),
        )
    if submit_failures:
        logger.warning("Provisioning: %d quota submission(s) failed.", submit_failures)
