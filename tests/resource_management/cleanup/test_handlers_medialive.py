"""Tests for the MediaLive cleanup handlers (Channel → Input → InputSecurityGroup)."""

from __future__ import annotations

import copy
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import (
    CUSTOM_DELETION_REGISTRY,
    PREPARE_REGISTRY,
)
from aws_bench.resource_management.cleanup.handlers import medialive
from aws_bench.resource_management.cleanup.models import HandlerStatus
from aws_bench.resource_management.cleanup.resource_cleaner import ResourceCleaner

_CHANNEL_TYPE = "AWS::MediaLive::Channel"
_INPUT_TYPE = "AWS::MediaLive::Input"
_ISG_TYPE = "AWS::MediaLive::InputSecurityGroup"

# The ids that leaked in the scheduled-runner run this module fixes.
_INPUT_ID = "1592653"
_OTHER_INPUT_ID = "4887889"
_ISG_ID = "44476"
_CHANNEL_ID = "9876543"

_ISG_CONFLICT = (
    f"Input Security Group {_ISG_ID} cannot be deleted because it is still being used by inputs."
)


class VirtualClock:
    """A per-thread virtual clock: ``sleep`` advances time instead of passing.

    Lets the real confirm loops be exercised instantly. Thread-local because
    ``_custom_delete`` uses a pool, and one worker's sleep must not burn another's budget.
    """

    def __init__(self) -> None:
        self._local = threading.local()
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        """Return this thread's virtual now (each thread starts at zero)."""
        return getattr(self._local, "now", 0.0)

    def sleep(self, seconds: float) -> None:
        """Advance this thread's clock and record the gap asked for."""
        self._local.now = self.monotonic() + seconds
        with self._lock:
            self.sleeps.append(seconds)


@contextmanager
def _instant_polling() -> Iterator[VirtualClock]:
    """Run the handlers on a virtual clock, yielding it so tests can assert the sleep ramp.

    Swaps only the handler module's ``time`` reference, so nothing else sees a frozen clock.
    """
    clock = VirtualClock()
    with patch.object(medialive, "time", clock):
        yield clock


def _client_error(code: str, message: str, op: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, op)


@dataclass
class FakeMediaLive:
    """In-memory MediaLive whose deletes honour the live-verified teardown semantics.

    Models the traps the handlers must survive: deleted ids linger as ``State == "DELETED"``
    tombstones and re-delete 200s, an ATTACHED input rejects DeleteInput, any live input
    blocks DeleteInputSecurityGroup, and only a DELETED channel releases its inputs.
    """

    inputs: dict[str, dict] = field(default_factory=dict)
    channels: dict[str, dict] = field(default_factory=dict)
    groups: dict[str, dict] = field(default_factory=dict)
    client: MagicMock = field(default_factory=MagicMock)

    def __post_init__(self) -> None:
        """Wire the MagicMock client's ops to this fake's in-memory state."""
        self.client.describe_input.side_effect = lambda InputId: self._describe(
            self.inputs, InputId, "DescribeInput"
        )
        self.client.describe_channel.side_effect = lambda ChannelId: self._describe(
            self.channels, ChannelId, "DescribeChannel"
        )
        self.client.describe_input_security_group.side_effect = lambda InputSecurityGroupId: (
            self._describe(self.groups, InputSecurityGroupId, "DescribeInputSecurityGroup")
        )
        self.client.delete_input.side_effect = self._delete_input
        self.client.delete_channel.side_effect = self._delete_channel
        self.client.delete_input_security_group.side_effect = self._delete_group
        self.client.stop_channel.side_effect = self._stop_channel

    @property
    def session(self) -> MagicMock:
        """A boto3-session mock handing out this fake as the ``medialive`` client."""
        session = MagicMock()
        session.client.return_value = self.client
        return session

    def _record(self, store: dict[str, dict], resource_id: str, op: str) -> dict:
        if resource_id not in store:
            # Only an id that never existed in this account+region 404s.
            raise _client_error("NotFoundException", f"{resource_id} not found", op)
        return store[resource_id]

    def _describe(self, store: dict[str, dict], resource_id: str, op: str) -> dict:
        # Deep copy: the real API returns a fresh response, not a live handle.
        return copy.deepcopy(self._record(store, resource_id, op))

    def _delete_input(self, InputId: str) -> dict:
        record = self._record(self.inputs, InputId, "DeleteInput")
        if record["State"] == "ATTACHED":
            raise _client_error(
                "ConflictException",
                f"Input {InputId} is busy, it cannot be deleted.",
                "DeleteInput",
            )
        record["State"] = "DELETED"
        return {}

    def _delete_channel(self, ChannelId: str) -> dict:
        record = self._record(self.channels, ChannelId, "DeleteChannel")
        if record["State"] in ("RUNNING", "STARTING", "RECOVERING"):
            raise _client_error(
                "ConflictException",
                f"Channel {ChannelId} must be idle, is {record['State']}.",
                "DeleteChannel",
            )
        record["State"] = "DELETED"
        for input_record in self.inputs.values():
            attached = input_record.get("AttachedChannels", [])
            if ChannelId not in attached:
                continue
            attached.remove(ChannelId)
            if not attached and input_record["State"] == "ATTACHED":
                input_record["State"] = "DETACHED"
        return {}

    def _delete_group(self, InputSecurityGroupId: str) -> dict:
        record = self._record(self.groups, InputSecurityGroupId, "DeleteInputSecurityGroup")
        blockers = [
            input_id
            for input_id in record.get("Inputs", [])
            if self.inputs.get(input_id, {}).get("State") in ("CREATING", "DETACHED", "ATTACHED")
        ]
        if blockers:
            raise _client_error("ConflictException", _ISG_CONFLICT, "DeleteInputSecurityGroup")
        record["State"] = "DELETED"
        return {}

    def _stop_channel(self, ChannelId: str) -> dict:
        self._record(self.channels, ChannelId, "StopChannel")["State"] = "IDLE"
        return {}


def _delete_via_registry(resource_type: str, identifier: str, session: MagicMock):
    """Run the registered delete handler exactly as the cleanup wave does."""
    handler = CUSTOM_DELETION_REGISTRY[resource_type]
    return handler(Resource(type=resource_type, identifier=identifier), session)


def _reaper(client: MagicMock, budget: float | None = None) -> medialive.BlockerReaper:
    """A reaper on the virtual clock's budget, as a handler invocation would build it."""
    budget = medialive._HANDLER_BUDGET_SEC if budget is None else budget
    return medialive.BlockerReaper(client, medialive.time.monotonic() + budget)


# -- registration: all three types need a delete path, none is CCAPI-deletable --


class TestRegistration:
    def test_all_three_types_have_a_delete_handler(self):
        for resource_type in (_CHANNEL_TYPE, _INPUT_TYPE, _ISG_TYPE):
            assert resource_type in CUSTOM_DELETION_REGISTRY

    def test_no_prepare_handler_doubles_the_budget(self):
        """Each delete clears its own blockers, so a chained prepare would only repeat it."""
        for resource_type in (_CHANNEL_TYPE, _INPUT_TYPE, _ISG_TYPE):
            assert resource_type not in PREPARE_REGISTRY


# -- AWS::MediaLive::Input --


class TestDeleteInput:
    def test_detached_input_deletes(self):
        fake = FakeMediaLive(inputs={_INPUT_ID: {"State": "DETACHED", "AttachedChannels": []}})
        with _instant_polling():
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, fake.session)

        assert result.status is HandlerStatus.SUCCESS
        fake.client.delete_input.assert_called_once_with(InputId=_INPUT_ID)
        fake.client.delete_channel.assert_not_called()
        assert fake.inputs[_INPUT_ID]["State"] == "DELETED"

    def test_attached_input_deletes_its_blocking_channel_first(self):
        fake = FakeMediaLive(
            inputs={_INPUT_ID: {"State": "ATTACHED", "AttachedChannels": [_CHANNEL_ID]}},
            channels={_CHANNEL_ID: {"State": "IDLE"}},
        )
        with _instant_polling():
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, fake.session)

        assert result.status is HandlerStatus.SUCCESS
        fake.client.delete_channel.assert_called_once_with(ChannelId=_CHANNEL_ID)
        fake.client.stop_channel.assert_not_called()  # IDLE deletes directly
        assert fake.channels[_CHANNEL_ID]["State"] == "DELETED"
        assert fake.inputs[_INPUT_ID]["State"] == "DELETED"

    def test_input_that_never_existed_is_skipped(self):
        fake = FakeMediaLive()
        with _instant_polling():
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, fake.session)

        assert result.status is HandlerStatus.SKIPPED
        fake.client.delete_input.assert_not_called()

    def test_input_stuck_attached_fails(self):
        """A channel that never reaches DELETED must FAIL, not be reported as deleted."""
        fake = FakeMediaLive(
            inputs={_INPUT_ID: {"State": "ATTACHED", "AttachedChannels": [_CHANNEL_ID]}},
            channels={_CHANNEL_ID: {"State": "IDLE"}},
        )
        # The channel delete never lands, so the input stays ATTACHED and busy.
        fake.client.delete_channel.side_effect = _client_error(
            "ConflictException", "channel busy", "DeleteChannel"
        )
        with _instant_polling():
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, fake.session)

        assert result.status is HandlerStatus.FAILED
        assert "not confirmed deleted" in result.message


# -- AWS::MediaLive::InputSecurityGroup --


class TestDeleteInputSecurityGroup:
    def test_prepare_drains_referencing_inputs_so_the_group_deletes(self):
        """Regression for run 2f3ee360: the group leaked while its inputs still referenced it."""
        fake = FakeMediaLive(
            inputs={
                _INPUT_ID: {"State": "ATTACHED", "AttachedChannels": [_CHANNEL_ID]},
                _OTHER_INPUT_ID: {"State": "DETACHED", "AttachedChannels": []},
            },
            channels={_CHANNEL_ID: {"State": "IDLE"}},
            groups={_ISG_ID: {"State": "IN_USE", "Inputs": [_INPUT_ID, _OTHER_INPUT_ID]}},
        )
        with _instant_polling():
            result = _delete_via_registry(_ISG_TYPE, _ISG_ID, fake.session)

        assert result.status is HandlerStatus.SUCCESS
        assert fake.inputs[_INPUT_ID]["State"] == "DELETED"
        assert fake.inputs[_OTHER_INPUT_ID]["State"] == "DELETED"
        assert fake.channels[_CHANNEL_ID]["State"] == "DELETED"
        assert fake.groups[_ISG_ID]["State"] == "DELETED"

    def test_a_drain_that_cannot_clear_its_inputs_fails(self):
        """The production shape: an input that will not go must keep the group FAILED."""
        fake = FakeMediaLive(
            inputs={_INPUT_ID: {"State": "DETACHED", "AttachedChannels": []}},
            groups={_ISG_ID: {"State": "IN_USE", "Inputs": [_INPUT_ID]}},
        )
        fake.client.delete_input.side_effect = _client_error(
            "ConflictException", "busy", "DeleteInput"
        )
        with _instant_polling():
            result = _delete_via_registry(_ISG_TYPE, _ISG_ID, fake.session)

        assert result.status is HandlerStatus.FAILED
        assert "ConflictException" in result.message
        assert fake.groups[_ISG_ID]["State"] == "IN_USE"

    def test_conflict_that_clears_on_a_later_poll_succeeds(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_input_security_group.side_effect = [
            {"State": "IN_USE", "Inputs": []},  # prepare
            {"State": "IN_USE", "Inputs": []},  # poll after the rejected delete
            {"State": "DELETED", "Inputs": []},  # poll after the retry landed
        ]
        client.delete_input_security_group.side_effect = [
            _client_error("ConflictException", _ISG_CONFLICT, "DeleteInputSecurityGroup"),
            {},
        ]

        with _instant_polling():
            result = _delete_via_registry(_ISG_TYPE, _ISG_ID, session)

        assert result.status is HandlerStatus.SUCCESS
        assert client.delete_input_security_group.call_count == 2

    def test_conflict_that_never_clears_fails(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_input_security_group.return_value = {"State": "IN_USE", "Inputs": []}
        client.delete_input_security_group.side_effect = _client_error(
            "ConflictException", _ISG_CONFLICT, "DeleteInputSecurityGroup"
        )

        with _instant_polling():
            result = _delete_via_registry(_ISG_TYPE, _ISG_ID, session)

        assert result.status is HandlerStatus.FAILED
        assert "not confirmed deleted" in result.message
        assert "ConflictException" in result.message

    def test_deletes_by_the_emitted_id(self):
        """The scanned Id is the CCAPI primary identifier, i.e. what the delete op wants."""
        fake = FakeMediaLive(groups={_ISG_ID: {"State": "IDLE", "Inputs": []}})
        with _instant_polling():
            result = _delete_via_registry(_ISG_TYPE, _ISG_ID, fake.session)

        assert result.status is HandlerStatus.SUCCESS
        fake.client.delete_input_security_group.assert_called_once_with(
            InputSecurityGroupId=_ISG_ID
        )


# -- AWS::MediaLive::Channel --


class TestDeleteChannel:
    def test_running_channel_is_stopped_first(self):
        fake = FakeMediaLive(channels={_CHANNEL_ID: {"State": "RUNNING"}})
        with _instant_polling():
            result = _delete_via_registry(_CHANNEL_TYPE, _CHANNEL_ID, fake.session)

        assert result.status is HandlerStatus.SUCCESS
        fake.client.stop_channel.assert_called_once_with(ChannelId=_CHANNEL_ID)
        fake.client.delete_channel.assert_called_once_with(ChannelId=_CHANNEL_ID)
        assert fake.channels[_CHANNEL_ID]["State"] == "DELETED"

    def test_idle_channel_is_not_stopped(self):
        fake = FakeMediaLive(channels={_CHANNEL_ID: {"State": "IDLE"}})
        with _instant_polling():
            result = _delete_via_registry(_CHANNEL_TYPE, _CHANNEL_ID, fake.session)

        assert result.status is HandlerStatus.SUCCESS
        fake.client.stop_channel.assert_not_called()
        fake.client.delete_channel.assert_called_once_with(ChannelId=_CHANNEL_ID)

    def test_create_failed_channel_is_not_stopped(self):
        fake = FakeMediaLive(channels={_CHANNEL_ID: {"State": "CREATE_FAILED"}})
        with _instant_polling():
            result = _delete_via_registry(_CHANNEL_TYPE, _CHANNEL_ID, fake.session)

        assert result.status is HandlerStatus.SUCCESS
        fake.client.stop_channel.assert_not_called()

    def test_channel_that_never_existed_is_skipped(self):
        fake = FakeMediaLive()
        with _instant_polling():
            result = _delete_via_registry(_CHANNEL_TYPE, _CHANNEL_ID, fake.session)

        assert result.status is HandlerStatus.SKIPPED
        fake.client.delete_channel.assert_not_called()


# -- tombstone semantics --


class TestTombstones:
    def test_an_already_deleted_input_is_confirmed_without_re_deleting(self):
        """The tombstone already proves it is gone, so no delete is needed at all."""
        fake = FakeMediaLive(inputs={_INPUT_ID: {"State": "DELETED", "AttachedChannels": []}})
        with _instant_polling():
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, fake.session)

        assert result.status is HandlerStatus.SUCCESS
        fake.client.delete_input.assert_not_called()

    def test_re_deleting_a_deleted_id_is_harmless(self):
        """MediaLive returns 200 (not NotFoundException) for a delete on a DELETED id."""
        fake = FakeMediaLive(inputs={_INPUT_ID: {"State": "DETACHED", "AttachedChannels": []}})
        with _instant_polling():
            first = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, fake.session)
            fake.client.delete_input(InputId=_INPUT_ID)  # a re-issue against the tombstone
            second = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, fake.session)

        assert first.status is HandlerStatus.SUCCESS
        assert second.status is HandlerStatus.SUCCESS
        assert fake.inputs[_INPUT_ID]["State"] == "DELETED"

    def test_gone_check_keys_on_deleted_state_not_on_not_found(self):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        # A tombstone: describe keeps answering, so NotFoundException never arrives.
        client.describe_input.return_value = {"State": "DELETED", "AttachedChannels": []}
        client.delete_input.return_value = {}

        with _instant_polling():
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, session)

        assert result.status is HandlerStatus.SUCCESS
        assert "deleted" in result.message

    def test_accepted_delete_without_a_deleted_state_fails(self):
        """A 200 is not proof of deletion; only State == DELETED is."""
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_input.return_value = {"State": "DETACHED", "AttachedChannels": []}
        client.delete_input.return_value = {}

        with _instant_polling():
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, session)

        assert result.status is HandlerStatus.FAILED
        assert "not confirmed deleted" in result.message


# -- a second pass over an already-torn-down chain must stay safe (cleanup retries) --


class TestRepeatedPassesAreSafe:
    def test_deleting_the_group_twice_stays_successful(self):
        fake = FakeMediaLive(
            inputs={_INPUT_ID: {"State": "DETACHED", "AttachedChannels": []}},
            groups={_ISG_ID: {"State": "IN_USE", "Inputs": [_INPUT_ID]}},
        )
        with _instant_polling():
            first = _delete_via_registry(_ISG_TYPE, _ISG_ID, fake.session)
            second = _delete_via_registry(_ISG_TYPE, _ISG_ID, fake.session)

        assert first.status is HandlerStatus.SUCCESS
        assert second.status is HandlerStatus.SUCCESS
        # The second pass sees the tombstone and re-deletes nothing.
        assert fake.client.delete_input_security_group.call_count == 1
        assert fake.inputs[_INPUT_ID]["State"] == "DELETED"

    def test_deleting_a_running_channel_twice_stops_it_once(self):
        fake = FakeMediaLive(channels={_CHANNEL_ID: {"State": "RUNNING"}})
        with _instant_polling():
            first = _delete_via_registry(_CHANNEL_TYPE, _CHANNEL_ID, fake.session)
            second = _delete_via_registry(_CHANNEL_TYPE, _CHANNEL_ID, fake.session)

        assert first.status is HandlerStatus.SUCCESS
        assert second.status is HandlerStatus.SUCCESS
        fake.client.stop_channel.assert_called_once_with(ChannelId=_CHANNEL_ID)

    def test_deleting_an_attached_input_twice_deletes_the_channel_once(self):
        fake = FakeMediaLive(
            inputs={_INPUT_ID: {"State": "ATTACHED", "AttachedChannels": [_CHANNEL_ID]}},
            channels={_CHANNEL_ID: {"State": "IDLE"}},
        )
        with _instant_polling():
            first = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, fake.session)
            second = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, fake.session)

        assert first.status is HandlerStatus.SUCCESS
        assert second.status is HandlerStatus.SUCCESS
        fake.client.delete_channel.assert_called_once_with(ChannelId=_CHANNEL_ID)


# -- the unordered custom-delete wave that leaked the group in production --


class TestConcurrentWave:
    def test_whole_chain_deletes_in_one_unordered_wave(self):
        """No ordering between types, so the whole chain must go regardless of who wins."""
        fake = FakeMediaLive(
            inputs={
                _INPUT_ID: {"State": "ATTACHED", "AttachedChannels": [_CHANNEL_ID]},
                _OTHER_INPUT_ID: {"State": "DETACHED", "AttachedChannels": []},
            },
            channels={_CHANNEL_ID: {"State": "RUNNING"}},
            groups={_ISG_ID: {"State": "IN_USE", "Inputs": [_INPUT_ID, _OTHER_INPUT_ID]}},
        )
        wave = [
            Resource(type=_ISG_TYPE, identifier=_ISG_ID),
            Resource(type=_INPUT_TYPE, identifier=_INPUT_ID),
            Resource(type=_INPUT_TYPE, identifier=_OTHER_INPUT_ID),
            Resource(type=_CHANNEL_TYPE, identifier=_CHANNEL_ID),
        ]

        with _instant_polling():
            result = ResourceCleaner(fake.session)._custom_delete(wave)

        assert result.failed == {}
        assert sorted(r.identifier for r in result.succeeded) == sorted(r.identifier for r in wave)
        assert fake.groups[_ISG_ID]["State"] == "DELETED"
        assert fake.inputs[_INPUT_ID]["State"] == "DELETED"
        assert fake.inputs[_OTHER_INPUT_ID]["State"] == "DELETED"
        assert fake.channels[_CHANNEL_ID]["State"] == "DELETED"


# -- a transient fault must cost one poll, never the whole pass --


class TestTransientFaultsDoNotAbortTheDelete:
    def test_an_unreadable_first_poll_still_deletes(self):
        fake = FakeMediaLive(inputs={_INPUT_ID: {"State": "DETACHED", "AttachedChannels": []}})
        fake.client.describe_input.side_effect = [
            _client_error("TooManyRequestsException", "slow down", "DescribeInput"),
            {"State": "DETACHED", "AttachedChannels": []},
            {"State": "DELETED", "AttachedChannels": []},
        ]
        with _instant_polling():
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, fake.session)

        assert result.status is HandlerStatus.SUCCESS
        fake.client.delete_input.assert_called_once_with(InputId=_INPUT_ID)

    def test_a_rejected_stop_still_lets_the_channel_delete_run(self):
        fake = FakeMediaLive(channels={_CHANNEL_ID: {"State": "RUNNING"}})
        fake.client.stop_channel.side_effect = _client_error(
            "ConflictException", "channel is not stoppable", "StopChannel"
        )
        with _instant_polling():
            result = _delete_via_registry(_CHANNEL_TYPE, _CHANNEL_ID, fake.session)

        # The stop is best-effort; the delete is what rules, and RUNNING blocks it.
        assert result.status is HandlerStatus.FAILED
        assert fake.client.delete_channel.call_count >= 1

    def test_an_unreadable_group_read_still_deletes(self):
        fake = FakeMediaLive(groups={_ISG_ID: {"State": "IDLE", "Inputs": []}})
        fake.client.describe_input_security_group.side_effect = [
            _client_error("TooManyRequestsException", "slow down", "DescribeInputSecurityGroup"),
            {"State": "IDLE", "Inputs": []},
            {"State": "DELETED", "Inputs": []},
        ]
        with _instant_polling():
            result = _delete_via_registry(_ISG_TYPE, _ISG_ID, fake.session)

        assert result.status is HandlerStatus.SUCCESS
        fake.client.delete_input_security_group.assert_called_once_with(
            InputSecurityGroupId=_ISG_ID
        )


# -- polling efficiency: a resource that settles fast must be observed fast --


class TestPollingEfficiency:
    def test_a_settled_resource_costs_no_sleep_at_all(self):
        """The delete and its confirming read are one predicate pass, so nothing is waited on."""
        fake = FakeMediaLive(inputs={_INPUT_ID: {"State": "DETACHED", "AttachedChannels": []}})
        with _instant_polling() as clock:
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, fake.session)

        assert result.status is HandlerStatus.SUCCESS
        assert clock.sleeps == []

    def test_first_re_check_is_fast_not_a_full_interval(self):
        """A resource DELETED by the second read is seen after the ramp's first step."""
        fake = FakeMediaLive(inputs={_INPUT_ID: {"State": "SETTLING", "AttachedChannels": []}})
        # The delete lands but the state trails it, as a real settle does.
        fake.client.delete_input.side_effect = None
        fake.client.delete_input.return_value = {}
        fake.client.describe_input.side_effect = [
            {"State": "SETTLING", "AttachedChannels": []},  # prepare's read
            {"State": "SETTLING", "AttachedChannels": []},  # still settling
            {"State": "DELETED", "AttachedChannels": []},
        ]

        with _instant_polling() as clock:
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, fake.session)

        assert result.status is HandlerStatus.SUCCESS
        assert clock.sleeps == [medialive._RAMP_STEP_SEC]
        assert sum(clock.sleeps) < medialive._POLL_INTERVAL_SEC

    def test_ramp_widens_toward_the_interval_and_stops_there(self):
        """Gaps grow by the ramp step each poll and cap at the interval, never hammering."""
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_input.return_value = {"State": "DETACHED", "AttachedChannels": []}
        client.delete_input.return_value = {}

        with _instant_polling() as clock:
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, session)

        assert result.status is HandlerStatus.FAILED  # never reached DELETED
        # The accepted delete buys one immediate re-check, then the ramp opens and widens.
        assert clock.sleeps[:4] == [0.5, 1.0, 1.5, 2.0]
        assert max(clock.sleeps) == medialive._POLL_INTERVAL_SEC
        # The loop trims its last gap to the deadline, so the budget is never overshot.
        assert sum(clock.sleeps) <= medialive._HANDLER_BUDGET_SEC

    def test_a_slow_channel_still_succeeds_within_budget(self):
        """A channel draining pipelines past the ramp is still confirmed, not timed out."""
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_channel.side_effect = [{"State": "IDLE"}] * 31 + [{"State": "DELETED"}]
        client.delete_channel.return_value = {}

        with _instant_polling() as clock:
            result = _delete_via_registry(_CHANNEL_TYPE, _CHANNEL_ID, session)

        assert result.status is HandlerStatus.SUCCESS
        assert sum(clock.sleeps) < medialive._HANDLER_BUDGET_SEC
        client.stop_channel.assert_not_called()  # IDLE deletes directly

    def test_slow_resource_polls_are_bounded_by_the_interval(self):
        """Over a long wait the API is hit at the flat interval, not at the ramp's opening rate."""
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_input_security_group.return_value = {"State": "IN_USE", "Inputs": []}
        client.delete_input_security_group.side_effect = _client_error(
            "ConflictException", _ISG_CONFLICT, "DeleteInputSecurityGroup"
        )

        with _instant_polling() as clock:
            result = _delete_via_registry(_ISG_TYPE, _ISG_ID, session)

        assert result.status is HandlerStatus.FAILED
        # A flat ramp-step gap would be 3600 polls; the widening keeps it near budget/interval.
        flat_interval_polls = medialive._HANDLER_BUDGET_SEC / medialive._POLL_INTERVAL_SEC
        assert len(clock.sleeps) < 2 * flat_interval_polls


# -- a blocker confirmed gone is not re-deleted or re-confirmed in the same invocation --


class TestBlockerConfirmedOncePerInvocation:
    def test_a_channel_already_confirmed_is_not_re_deleted(self):
        """The second ask for a channel already seen DELETED skips the delete and the loop."""
        fake = FakeMediaLive(channels={_CHANNEL_ID: {"State": "IDLE"}})
        with _instant_polling():
            reaper = _reaper(fake.client)
            first = reaper.delete_channel(_CHANNEL_ID)
            calls_after_first = fake.client.describe_channel.call_count
            second = reaper.delete_channel(_CHANNEL_ID)

        assert first.confirmed and second.confirmed
        fake.client.delete_channel.assert_called_once_with(ChannelId=_CHANNEL_ID)
        # No describe either: the memoized confirmation short-circuits the whole loop.
        assert fake.client.describe_channel.call_count == calls_after_first

    def test_two_inputs_sharing_a_channel_confirm_it_once(self):
        """A primary/backup pair on one channel: the second release reuses the confirmation."""
        fake = FakeMediaLive(
            inputs={
                _INPUT_ID: {"State": "ATTACHED", "AttachedChannels": [_CHANNEL_ID]},
                _OTHER_INPUT_ID: {"State": "ATTACHED", "AttachedChannels": [_CHANNEL_ID]},
            },
            channels={_CHANNEL_ID: {"State": "IDLE"}},
        )
        with _instant_polling():
            reaper = _reaper(fake.client)
            reaper.delete_input(_INPUT_ID)
            reaper.delete_input(_OTHER_INPUT_ID)

        fake.client.delete_channel.assert_called_once_with(ChannelId=_CHANNEL_ID)
        assert fake.channels[_CHANNEL_ID]["State"] == "DELETED"

    def test_memoization_is_per_invocation_not_shared_across_calls(self):
        """No cross-invocation cache: a fresh reaper re-reads instead of trusting the last."""
        fake = FakeMediaLive(channels={_CHANNEL_ID: {"State": "IDLE"}})
        with _instant_polling():
            _reaper(fake.client).delete_channel(_CHANNEL_ID)
            reads_after_first = fake.client.describe_channel.call_count
            _reaper(fake.client).delete_channel(_CHANNEL_ID)

        assert fake.client.describe_channel.call_count > reads_after_first

    def test_an_unconfirmed_channel_is_retried_not_memoized(self):
        """Only a confirmed-DELETED channel is memoized, so a stuck one is asked again."""
        fake = FakeMediaLive(channels={_CHANNEL_ID: {"State": "IDLE"}})
        fake.client.delete_channel.side_effect = _client_error(
            "ConflictException", "channel busy", "DeleteChannel"
        )
        with _instant_polling():
            reaper = _reaper(fake.client, budget=60)
            first = reaper.delete_channel(_CHANNEL_ID)
            calls_after_first = fake.client.delete_channel.call_count
            second = reaper.delete_channel(_CHANNEL_ID)

        assert not first.confirmed and not second.confirmed
        assert fake.client.delete_channel.call_count > calls_after_first

    def test_a_stuck_shared_channel_still_fails_the_group(self):
        """End to end: an undeletable channel keeps the group FAILED, never papered over."""
        fake = FakeMediaLive(
            inputs={_INPUT_ID: {"State": "ATTACHED", "AttachedChannels": [_CHANNEL_ID]}},
            channels={_CHANNEL_ID: {"State": "IDLE"}},
            groups={_ISG_ID: {"State": "IN_USE", "Inputs": [_INPUT_ID]}},
        )
        fake.client.delete_channel.side_effect = _client_error(
            "ConflictException", "channel busy", "DeleteChannel"
        )
        with _instant_polling():
            result = _delete_via_registry(_ISG_TYPE, _ISG_ID, fake.session)

        assert result.status is HandlerStatus.FAILED
        assert fake.groups[_ISG_ID]["State"] == "IN_USE"


# -- an unreadable state is never mistaken for gone (the contract's load-bearing branch) --


class TestUnreadableStateIsNotGone:
    def test_a_describe_that_never_recovers_fails(self):
        """A permanently unreadable resource must FAIL, never be reported deleted."""
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_input.side_effect = _client_error(
            "TooManyRequestsException", "slow down", "DescribeInput"
        )

        with _instant_polling():
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, session)

        assert result.status is HandlerStatus.FAILED
        assert "TooManyRequestsException" in result.message

    def test_a_delete_that_404s_defers_to_the_confirming_read(self):
        """Describe found it but delete says it never existed: the read, not the delete, rules."""
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_input.side_effect = [
            {"State": "DETACHED", "AttachedChannels": []},
            {"State": "DELETED", "AttachedChannels": []},
        ]
        client.delete_input.side_effect = _client_error("NotFoundException", "gone", "DeleteInput")

        with _instant_polling():
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, session)

        assert result.status is HandlerStatus.SUCCESS

    def test_a_non_aws_fault_is_also_not_gone(self):
        """A BotoCoreError (no error code) must not read as a not-found either."""
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_input.side_effect = BotoCoreError()

        with _instant_polling():
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, session)

        assert result.status is HandlerStatus.FAILED

    def test_blockers_are_rediscovered_after_a_failed_read(self):
        """A first-poll read fault must not lose the blocker set: the next poll re-derives it."""
        fake = FakeMediaLive(
            inputs={_INPUT_ID: {"State": "ATTACHED", "AttachedChannels": [_CHANNEL_ID]}},
            channels={_CHANNEL_ID: {"State": "IDLE"}},
        )
        real_describe = fake.client.describe_input.side_effect
        faults = [_client_error("TooManyRequestsException", "slow down", "DescribeInput")] * 2

        def flaky(InputId: str):
            if faults:
                raise faults.pop()
            return real_describe(InputId)

        fake.client.describe_input.side_effect = flaky

        with _instant_polling():
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, fake.session)

        assert result.status is HandlerStatus.SUCCESS
        # The blocking channel was still found and deleted, despite the opening faults.
        fake.client.delete_channel.assert_called_once_with(ChannelId=_CHANNEL_ID)
        assert fake.inputs[_INPUT_ID]["State"] == "DELETED"


# -- permanent faults fail fast; transient ones are retried --


class TestErrorClassification:
    def test_a_permanent_fault_stops_immediately(self):
        """A 4xx that cannot clear must not burn the budget: one delete attempt, then FAIL."""
        fake = FakeMediaLive(inputs={_INPUT_ID: {"State": "DETACHED", "AttachedChannels": []}})
        fake.client.delete_input.side_effect = _client_error(
            "ForbiddenException", "not authorized", "DeleteInput"
        )

        with _instant_polling() as clock:
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, fake.session)

        assert result.status is HandlerStatus.FAILED
        assert "cannot be deleted" in result.message
        assert fake.client.delete_input.call_count == 1
        assert clock.sleeps == []  # failed fast rather than polling

    def test_a_transient_fault_is_retried(self):
        """A ConflictException is the blocker-still-present case, so it must keep trying."""
        fake = FakeMediaLive(inputs={_INPUT_ID: {"State": "DETACHED", "AttachedChannels": []}})
        real_delete = fake.client.delete_input.side_effect
        rejections = [_client_error("ConflictException", "busy", "DeleteInput")] * 3

        def flaky(InputId: str):
            if rejections:
                raise rejections.pop()
            return real_delete(InputId)

        fake.client.delete_input.side_effect = flaky

        with _instant_polling():
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, fake.session)

        assert result.status is HandlerStatus.SUCCESS
        assert fake.client.delete_input.call_count == 4

    def test_each_distinct_fault_is_logged_once_not_per_poll(self, caplog):
        """A retry loop must leave a trace, but one unchanged rejection must not spam the log."""
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_input.return_value = {"State": "DETACHED", "AttachedChannels": []}
        client.delete_input.side_effect = _client_error("ConflictException", "busy", "DeleteInput")

        with caplog.at_level(logging.WARNING), _instant_polling():
            result = _delete_via_registry(_INPUT_TYPE, _INPUT_ID, session)

        assert result.status is HandlerStatus.FAILED
        retry_lines = [r for r in caplog.records if "retrying" in r.getMessage()]
        assert len(retry_lines) == 1
        assert "ConflictException" in retry_lines[0].getMessage()
        assert _INPUT_ID in retry_lines[0].getMessage()


# -- the channel stop is decided on fresh state, every poll --


class TestChannelStopStates:
    def test_a_channel_stoppable_only_later_is_still_stopped(self):
        """UPDATING at entry then RUNNING: the per-poll re-read is what catches it."""
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        client.describe_channel.side_effect = [
            {"State": "UPDATING"},  # prepare's read: not stoppable yet
            {"State": "UPDATING"},
            {"State": "RUNNING"},  # now stoppable
            {"State": "DELETED"},
        ]
        client.delete_channel.return_value = {}

        with _instant_polling():
            result = _delete_via_registry(_CHANNEL_TYPE, _CHANNEL_ID, session)

        assert result.status is HandlerStatus.SUCCESS
        client.stop_channel.assert_called_once_with(ChannelId=_CHANNEL_ID)

    def test_transient_states_are_not_stopped(self):
        """StopChannel is invalid from these, so it must not be issued."""
        for state in ("IDLE", "STOPPING", "CREATING", "UPDATING", "DELETING", "UPDATE_FAILED"):
            fake = FakeMediaLive(channels={_CHANNEL_ID: {"State": state}})
            with _instant_polling():
                _delete_via_registry(_CHANNEL_TYPE, _CHANNEL_ID, fake.session)

            assert fake.client.stop_channel.call_count == 0, state

    def test_live_states_are_stopped(self):
        """A channel must be stopped from any of these before DeleteChannel is accepted."""
        for state in ("RUNNING", "STARTING", "RECOVERING"):
            fake = FakeMediaLive(channels={_CHANNEL_ID: {"State": state}})
            with _instant_polling():
                result = _delete_via_registry(_CHANNEL_TYPE, _CHANNEL_ID, fake.session)

            assert result.status is HandlerStatus.SUCCESS, state
            fake.client.stop_channel.assert_called_with(ChannelId=_CHANNEL_ID)


# -- one budget per handler invocation, shared by every nested delete --


class TestSharedBudget:
    def test_draining_many_inputs_shares_one_budget(self):
        """Each stuck input must draw on the same clock, not receive a fresh full budget."""
        stuck_inputs = {
            f"input-{i}": {"State": "DETACHED", "AttachedChannels": []} for i in range(4)
        }
        fake = FakeMediaLive(
            inputs=stuck_inputs,
            groups={_ISG_ID: {"State": "IN_USE", "Inputs": list(stuck_inputs)}},
        )
        # Nothing ever deletes, so every nested loop runs to the shared deadline.
        fake.client.delete_input.side_effect = _client_error(
            "ConflictException", "busy", "DeleteInput"
        )
        fake.client.delete_input_security_group.side_effect = _client_error(
            "ConflictException", _ISG_CONFLICT, "DeleteInputSecurityGroup"
        )

        with _instant_polling() as clock:
            result = _delete_via_registry(_ISG_TYPE, _ISG_ID, fake.session)

        assert result.status is HandlerStatus.FAILED
        # 4 inputs x a fresh budget would be 4x this; one shared deadline caps the whole handler.
        assert sum(clock.sleeps) <= medialive._HANDLER_BUDGET_SEC

    def test_an_input_reached_twice_in_one_drain_is_deleted_once(self):
        """Two groups sharing an input, or a repeated id, must not re-run its whole loop."""
        fake = FakeMediaLive(inputs={_INPUT_ID: {"State": "DETACHED", "AttachedChannels": []}})
        with _instant_polling():
            reaper = _reaper(fake.client)
            reaper.drain_group(_ISG_ID, {"Inputs": [_INPUT_ID, _INPUT_ID]})

        fake.client.delete_input.assert_called_once_with(InputId=_INPUT_ID)

    def test_an_exhausted_budget_stops_the_drain(self):
        """Once the deadline passes, no further blocker delete is attempted."""
        fake = FakeMediaLive(
            inputs={_INPUT_ID: {"State": "DETACHED", "AttachedChannels": []}},
        )
        fake.client.delete_input.side_effect = _client_error(
            "ConflictException", "busy", "DeleteInput"
        )
        with _instant_polling():
            reaper = _reaper(fake.client, budget=0)
            confirmation = reaper.delete_input(_INPUT_ID)

        assert not confirmation.confirmed
        assert fake.client.delete_input.call_count == 1  # the opening pass only
