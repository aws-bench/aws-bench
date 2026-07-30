"""MediaLive cleanup handlers — Channel → Input → InputSecurityGroup, none CCAPI-deletable.

The custom-delete wave is unordered, so each handler deletes its own blockers. Deletes are
tombstoned (``delete_*`` 200s, ``describe_*`` keeps serving), so ``State == "DELETED"`` is the
gone-signal, not ``NotFoundException``, and every delete is safe to re-issue. Each loop
re-describes before acting, so blockers are re-derived per poll rather than read once.

Contract (as for ``ipam``): CONFIRMED gone (DELETED / never existed) or FAILED.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import boto3
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.models import HandlerResult, HandlerStatus
from aws_bench.utils.concurrent import build_client, raise_if_shutdown

logger = get_logger(__name__)

_NOT_FOUND_CODES = ("NotFoundException",)
_DELETED_STATE = "DELETED"

# Faults that fail identically however often they are re-issued, so the loop stops rather
# than spending its budget on a 4xx it cannot clear.
_PERMANENT_ERROR_CODES = frozenset(
    {"BadRequestException", "ForbiddenException", "UnprocessableEntityException"}
)

# StopChannel is valid only from these; IDLE and *_FAILED delete directly. Transient states
# (CREATING, UPDATING, STOPPING) are absent by design — the loop re-reads and stops once the
# channel becomes stoppable, rather than depending on this set being exhaustive.
_CHANNEL_LIVE_STATES = frozenset({"RUNNING", "STARTING", "RECOVERING"})

_POLL_INTERVAL_SEC = 10
_RAMP_STEP_SEC = 0.5  # a DETACHED input reaches DELETED in ~1.5s, so open far tighter
# One budget per handler invocation, shared by every nested delete: a live channel drains its
# pipelines for ~600s and an input may sit behind two of them.
_HANDLER_BUDGET_SEC = 1800


@dataclass(frozen=True)
class ResourceApi:
    """The describe/delete op pair for one MediaLive type, so the polling can be shared."""

    describe_op: str
    delete_op: str
    id_param: str
    label: str


CHANNEL_API = ResourceApi("describe_channel", "delete_channel", "ChannelId", "channel")
INPUT_API = ResourceApi("describe_input", "delete_input", "InputId", "input")
ISG_API = ResourceApi(
    "describe_input_security_group",
    "delete_input_security_group",
    "InputSecurityGroupId",
    "input security group",
)


@dataclass(frozen=True)
class DeleteConfirmation:
    """Whether a resource was confirmed gone, plus the message to report either way."""

    confirmed: bool
    message: str
    never_existed: bool = False


# Clears what blocks a resource's delete, given its id and freshly-read description.
BlockerClearer = Callable[[str, dict], None]


def _error_code(fault: Exception) -> str:
    """Return an AWS error code, or the exception class for a non-AWS fault."""
    if isinstance(fault, ClientError):
        return fault.response.get("Error", {}).get("Code", "")
    return type(fault).__name__


def _poll_gaps() -> Iterator[float]:
    """Yield sleep gaps opening at the ramp step and widening to the poll interval.

    A teardown usually settles in seconds, so a flat interval would dominate the fast case.
    """
    gap = _RAMP_STEP_SEC
    while True:
        yield gap
        gap = min(gap + _RAMP_STEP_SEC, _POLL_INTERVAL_SEC)


@dataclass
class _FaultLog:
    """Records the faults a retry loop swallows, logging each distinct code once.

    Otherwise the loop is silent for its whole budget, or spams one line per poll.
    """

    label: str
    resource_id: str
    last: Exception | None = None
    _logged_codes: set[str] = field(default_factory=set)

    def record(self, fault: Exception) -> None:
        """Remember the fault and log it if its code has not been seen yet."""
        self.last = fault
        code = _error_code(fault)
        if code in self._logged_codes:
            return
        self._logged_codes.add(code)
        logger.warning(
            f"MediaLive {self.label} '{self.resource_id}' not deleted yet ({code}), retrying: "
            f"{fault}"
        )


def _describe(client: BaseClient, api: ResourceApi, resource_id: str) -> dict:
    """Return the describe response; only a never-existed id raises (a deleted one is served)."""
    return getattr(client, api.describe_op)(**{api.id_param: resource_id})


def _never_existed(fault: Exception) -> bool:
    """Return True if the fault says the id does not exist in this account and region."""
    return _error_code(fault) in _NOT_FOUND_CODES


def _sleep_before_next_poll(deadline: float, gaps: Iterator[float]) -> bool:
    """Wait for the next poll, returning False once the shared deadline leaves no time."""
    raise_if_shutdown()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    time.sleep(min(next(gaps), remaining))
    return True


def _delete_and_confirm_gone(
    client: BaseClient,
    api: ResourceApi,
    resource_id: str,
    *,
    deadline: float,
    clear_blockers: BlockerClearer | None = None,
) -> DeleteConfirmation:
    """Delete the resource and poll until it is DELETED, or FAIL — never an unconfirmed defer.

    Re-describes every pass, so ``clear_blockers`` acts on fresh state and an unreadable
    resource is retried rather than mistaken for gone. A permanent fault stops at once.
    """
    faults = _FaultLog(api.label, resource_id)
    gaps = _poll_gaps()
    recheck_at_once = False

    while True:
        try:
            described = _describe(client, api, resource_id)
        except (ClientError, BotoCoreError) as fault:
            if _never_existed(fault):
                return DeleteConfirmation(
                    True, f"MediaLive {api.label} does not exist", never_existed=True
                )
            faults.record(fault)
        else:
            if described.get("State", "") == _DELETED_STATE:
                logger.debug(f"MediaLive {api.label} '{resource_id}' confirmed deleted")
                return DeleteConfirmation(True, f"MediaLive {api.label} deleted")
            if clear_blockers is not None:
                clear_blockers(resource_id, described)
            accepted = _submit_delete(client, api, resource_id, faults)
            if accepted.permanent_fault is not None:
                return DeleteConfirmation(
                    False,
                    f"MediaLive {api.label} '{resource_id}' cannot be deleted: "
                    f"{accepted.permanent_fault}",
                )
            # A resource that deletes outright (a DETACHED input goes in ~1.5s) is confirmed
            # without waiting; the skip happens once, so a slow one still settles into the ramp.
            if accepted.submitted and not recheck_at_once:
                recheck_at_once = True
                continue

        if not _sleep_before_next_poll(deadline, gaps):
            reason = f": {faults.last}" if faults.last is not None else " within budget"
            return DeleteConfirmation(
                False, f"MediaLive {api.label} '{resource_id}' not confirmed deleted{reason}"
            )


@dataclass(frozen=True)
class _DeleteAttempt:
    """Whether the delete was accepted, and the permanent fault to stop the loop on."""

    submitted: bool = False
    permanent_fault: Exception | None = None


def _submit_delete(
    client: BaseClient, api: ResourceApi, resource_id: str, faults: _FaultLog
) -> _DeleteAttempt:
    """Issue the delete, recording a transient rejection and surfacing a permanent one."""
    try:
        getattr(client, api.delete_op)(**{api.id_param: resource_id})
    except (ClientError, BotoCoreError) as fault:
        if _never_existed(fault):
            return _DeleteAttempt()  # The confirming read rules on this, not the delete.
        if _error_code(fault) in _PERMANENT_ERROR_CODES:
            return _DeleteAttempt(permanent_fault=fault)
        faults.record(fault)
        return _DeleteAttempt()
    return _DeleteAttempt(submitted=True)


def _result(resource: Resource, action: str, status: HandlerStatus, message: str) -> HandlerResult:
    """Build a HandlerResult for *resource* (keeps the handlers below to one statement)."""
    return HandlerResult(
        resource_id=resource.identifier,
        resource_type=resource.type,
        action=action,
        status=status,
        message=message,
    )


def _confirmation_result(resource: Resource, confirmation: DeleteConfirmation) -> HandlerResult:
    """Map a confirm-gone outcome to the handler contract (as for ``ipam``).

    Gone is SUCCESS, or SKIPPED when there was nothing to delete; anything unconfirmed FAILS.
    """
    if not confirmation.confirmed:
        return _result(resource, "delete", HandlerStatus.FAILED, confirmation.message)
    status = HandlerStatus.SKIPPED if confirmation.never_existed else HandlerStatus.SUCCESS
    return _result(resource, "delete", status, confirmation.message)


@dataclass
class BlockerReaper:
    """Deletes the resources blocking a delete, on one shared budget.

    One instance per handler invocation, never shared across the wave's threads. Confirmed
    deletions are memoized: inputs routinely share a channel, and DELETED is terminal.
    """

    client: BaseClient
    deadline: float
    _confirmed_gone: set[tuple[str, str]] = field(default_factory=set)

    def stop_channel_if_live(self, channel_id: str, described: dict) -> None:
        """Stop the channel when it is live, so DeleteChannel is accepted (never raises).

        Re-run per poll on fresh state, so a channel UPDATING at entry is still stopped later.
        """
        if described.get("State", "") not in _CHANNEL_LIVE_STATES:
            return

        try:
            self.client.stop_channel(ChannelId=channel_id)
            logger.debug(f"Stopping MediaLive channel '{channel_id}' to delete it")
        except (ClientError, BotoCoreError) as e:
            # A channel that raced into STOPPING rejects a second stop; the loop re-checks.
            logger.debug(f"Could not stop MediaLive channel '{channel_id}': {e}")

    def release_input(self, input_id: str, described: dict) -> None:
        """Delete the channels pinning the input, since an ATTACHED input rejects DeleteInput.

        A channel must reach DELETED to release its input, so each is confirmed, not just asked.
        """
        for channel_id in described.get("AttachedChannels", []):
            self.delete_channel(channel_id)

    def drain_group(self, group_id: str, described: dict) -> None:
        """Delete the inputs referencing the security group; any of them blocks its delete.

        DETACHED inputs block it too, so every referencing input goes, reaped or not.
        """
        for input_id in described.get("Inputs", []):
            self.delete_input(input_id)

    def delete_channel(self, channel_id: str) -> DeleteConfirmation:
        """Stop the channel if needed, delete it, and confirm it reached DELETED."""
        return self.reap(CHANNEL_API, channel_id, self.stop_channel_if_live)

    def delete_input(self, input_id: str) -> DeleteConfirmation:
        """Delete the input and the channels pinning it, and confirm it reached DELETED."""
        return self.reap(INPUT_API, input_id, self.release_input)

    def reap(
        self, api: ResourceApi, resource_id: str, clear_blockers: BlockerClearer
    ) -> DeleteConfirmation:
        """Delete one resource against the shared deadline, at most once per invocation."""
        memo_key = (api.label, resource_id)
        if memo_key in self._confirmed_gone:
            return DeleteConfirmation(True, f"MediaLive {api.label} deleted")

        confirmation = _delete_and_confirm_gone(
            self.client, api, resource_id, deadline=self.deadline, clear_blockers=clear_blockers
        )
        if confirmation.confirmed:
            self._confirmed_gone.add(memo_key)
        else:
            logger.warning(f"MediaLive {api.label} not deleted: {confirmation.message}")
        return confirmation


def _reaper(session: boto3.Session) -> BlockerReaper:
    """Build the reaper that owns this handler invocation's whole budget.

    No prepare handler is registered: the registry chains prepare into delete, so one would
    only repeat the loop's own blocker-clearing on a second budget.
    """
    client = build_client(session, "medialive")
    return BlockerReaper(client, time.monotonic() + _HANDLER_BUDGET_SEC)


# ── Registered delete handlers ───────────────────────────────────────


@resource_handler("AWS::MediaLive::Channel", role="delete")
def _delete_channel(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Stop the channel if it is live, delete it, and confirm it reached DELETED.

    Confirming DELETED is the point: that is when the channel releases its inputs.
    """
    reaper = _reaper(session)
    return _confirmation_result(resource, reaper.delete_channel(resource.identifier))


@resource_handler("AWS::MediaLive::Input", role="delete")
def _delete_input(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete the channels attached to the input, then the input, and confirm it is DELETED."""
    reaper = _reaper(session)
    return _confirmation_result(resource, reaper.delete_input(resource.identifier))


@resource_handler("AWS::MediaLive::InputSecurityGroup", role="delete")
def _delete_input_security_group(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete the inputs referencing the group, then the group, and confirm it is DELETED."""
    reaper = _reaper(session)
    confirmation = reaper.reap(ISG_API, resource.identifier, reaper.drain_group)
    return _confirmation_result(resource, confirmation)
