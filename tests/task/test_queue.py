"""Tests for AwsBenchTrialQueue — the per-scenario gate + the AwsBenchTrial build loop.

Two surfaces:

- ``_execute_trial_with_retries`` builds via ``AwsBenchTrial.create`` and runs the
  retry loop.
- ``_run_trial`` admits a trial through its scenario's readers-writer gate before
  taking the global semaphore: different scenarios co-run, same-scenario mutating
  trials serialize, same-scenario read-only trials co-run, and the global ``-n``
  cap bounds whatever the gate admits.

The gate tests drive ``_run_trial`` with ``_execute_trial_with_retries`` patched to
a barrier that records in-flight counts (per scenario and global), so they assert
scheduling without building real trials.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from harbor.models.job.config import RetryConfig
from harbor.models.trial.config import TaskConfig, TrialConfig

from aws_bench.dataset.task_config import ConcurrencyMode
from aws_bench.task import queue as queue_mod
from aws_bench.task.queue import AwsBenchTrialQueue, _ScenarioAdmissionGate
from aws_bench.task.trial_config import AwsBenchTrialConfig


def _ok_result():
    return SimpleNamespace(exception_info=None)


# --- _execute_trial_with_retries: AwsBenchTrial construction + retry --------


@pytest.mark.asyncio
async def test_execute_builds_awsbench_trial(mocker):
    """The override constructs via AwsBenchTrial.create, not the base Trial.create."""
    trial = MagicMock()
    trial.run = AsyncMock(return_value=_ok_result())
    create = mocker.patch.object(queue_mod.AwsBenchTrial, "create", AsyncMock(return_value=trial))

    q = AwsBenchTrialQueue(n_concurrent=1)
    config = MagicMock()
    result = await q._execute_trial_with_retries(config)

    create.assert_awaited_once_with(config)
    assert result.exception_info is None


@pytest.mark.asyncio
async def test_create_failure_propagates(mocker):
    """A first-attempt create() failure surfaces as itself."""
    mocker.patch.object(
        queue_mod.AwsBenchTrial, "create", AsyncMock(side_effect=RuntimeError("boom"))
    )
    q = AwsBenchTrialQueue(n_concurrent=1)
    with pytest.raises(RuntimeError, match="boom"):
        await q._execute_trial_with_retries(MagicMock())


@pytest.mark.asyncio
async def test_retries_on_retryable_exception(mocker):
    """A retryable exception re-creates the trial; success on the retry is returned."""
    fail = SimpleNamespace(exception_info=SimpleNamespace(exception_type="SomeError"))
    ok = _ok_result()
    failing_trial = MagicMock()
    failing_trial.run = AsyncMock(return_value=fail)
    failing_trial.paths = SimpleNamespace(trial_dir=MagicMock())
    ok_trial = MagicMock()
    ok_trial.run = AsyncMock(return_value=ok)

    create = mocker.patch.object(
        queue_mod.AwsBenchTrial,
        "create",
        AsyncMock(side_effect=[failing_trial, ok_trial]),
    )
    mocker.patch.object(queue_mod.shutil, "rmtree")
    mocker.patch.object(queue_mod.asyncio, "sleep", AsyncMock())

    q = AwsBenchTrialQueue(n_concurrent=1, retry_config=RetryConfig(max_retries=1))
    result = await q._execute_trial_with_retries(MagicMock())

    assert create.await_count == 2
    assert result.exception_info is None


# --- n_concurrent guard -----------------------------------------------------


def test_rejects_non_positive_n_concurrent():
    with pytest.raises(ValueError, match="n_concurrent must be >= 1"):
        AwsBenchTrialQueue(n_concurrent=0)


# --- gate scheduling --------------------------------------------------------


def _trial_config(
    task_id: str, *, trial_name: str, scenario_id: str, mode: ConcurrencyMode
) -> AwsBenchTrialConfig:
    """An AwsBenchTrialConfig carrying its own scenario_id + concurrency_mode."""
    from aws_bench.scenario.locator import ScenarioConfig

    return AwsBenchTrialConfig(
        task=TaskConfig(path=Path(f"tasks/{task_id}")),
        scenario=ScenarioConfig(name=scenario_id, path=Path(f"scenarios/{scenario_id}")),
        trial_name=trial_name,
        scenario_id=scenario_id,
        concurrency_mode=mode,
        account_mapping={"PRIMARY": "111111111111"},
        regions=["us-east-1"],
    )


class _Barrier:
    """Records concurrent trials while each "trial" holds for a fixed duration.

    Patched in for ``_execute_trial_with_retries`` so a "trial" is just a hold;
    ``peak[scenario_id]`` is the most that ran at once for a scenario and
    ``global_peak`` is the most that ran at once across all scenarios. No lock is
    needed: asyncio is single-threaded, so the counter updates between ``await``
    points are atomic.
    """

    def __init__(self, *, hold_sec: float = 0.02) -> None:
        self._hold_sec = hold_sec
        self._in_flight: dict[str, int] = {}
        self._global_in_flight = 0
        self.peak: dict[str, int] = {}
        self.global_peak = 0

    async def run(self, trial_config: AwsBenchTrialConfig):
        scenario_id = trial_config.scenario_id
        self._in_flight[scenario_id] = self._in_flight.get(scenario_id, 0) + 1
        self._global_in_flight += 1
        self.peak[scenario_id] = max(self.peak.get(scenario_id, 0), self._in_flight[scenario_id])
        self.global_peak = max(self.global_peak, self._global_in_flight)
        await asyncio.sleep(self._hold_sec)
        self._in_flight[scenario_id] -= 1
        self._global_in_flight -= 1
        return _ok_result()


def _gated_queue(*, n_concurrent: int, mocker) -> tuple[AwsBenchTrialQueue, _Barrier]:
    q = AwsBenchTrialQueue(n_concurrent=n_concurrent)
    barrier = _Barrier()
    mocker.patch.object(q, "_execute_trial_with_retries", side_effect=barrier.run)
    return q, barrier


@pytest.mark.asyncio
async def test_different_scenarios_co_run(mocker):
    """Trials of different scenarios run in parallel; each scenario stays serial.

    Two mutating trials per scenario: within a scenario they serialize (peak 1),
    but the two scenarios overlap, so the global peak reaches 2 — which a queue
    that serialized across scenarios could never produce.
    """
    q, barrier = _gated_queue(n_concurrent=8, mocker=mocker)
    m = ConcurrencyMode.MUTATING
    await asyncio.gather(
        q._run_trial(_trial_config("a", trial_name="a-0", scenario_id="scenario-a", mode=m)),
        q._run_trial(_trial_config("a", trial_name="a-1", scenario_id="scenario-a", mode=m)),
        q._run_trial(_trial_config("b", trial_name="b-0", scenario_id="scenario-b", mode=m)),
        q._run_trial(_trial_config("b", trial_name="b-1", scenario_id="scenario-b", mode=m)),
    )
    assert barrier.peak == {"scenario-a": 1, "scenario-b": 1}  # each scenario serial
    assert barrier.global_peak == 2  # but the two scenarios ran at the same time


@pytest.mark.asyncio
async def test_same_scenario_mutating_serializes(mocker):
    """Two mutating trials of one scenario never overlap on the account."""
    q, barrier = _gated_queue(n_concurrent=8, mocker=mocker)
    m = ConcurrencyMode.MUTATING
    await asyncio.gather(
        q._run_trial(_trial_config("a", trial_name="a-0", scenario_id="scenario-a", mode=m)),
        q._run_trial(_trial_config("a", trial_name="a-1", scenario_id="scenario-a", mode=m)),
    )
    assert barrier.peak == {"scenario-a": 1}


@pytest.mark.asyncio
async def test_same_scenario_read_only_co_runs(mocker):
    """Read-only trials of one scenario co-run on the single account."""
    q, barrier = _gated_queue(n_concurrent=8, mocker=mocker)
    ro = ConcurrencyMode.READ_ONLY
    await asyncio.gather(
        q._run_trial(_trial_config("a", trial_name="a-0", scenario_id="scenario-a", mode=ro)),
        q._run_trial(_trial_config("a", trial_name="a-1", scenario_id="scenario-a", mode=ro)),
        q._run_trial(_trial_config("a", trial_name="a-2", scenario_id="scenario-a", mode=ro)),
    )
    assert barrier.peak == {"scenario-a": 3}


@pytest.mark.asyncio
async def test_global_cap_beats_gate(mocker):
    """The global -n semaphore bounds even gate-eligible (read-only) trials."""
    q, barrier = _gated_queue(n_concurrent=2, mocker=mocker)
    ro = ConcurrencyMode.READ_ONLY
    await asyncio.gather(
        *(
            q._run_trial(_trial_config("a", trial_name=f"a-{i}", scenario_id="scenario-a", mode=ro))
            for i in range(5)
        )
    )
    assert barrier.peak["scenario-a"] == 2


@pytest.mark.asyncio
async def test_non_awsbench_config_raises(mocker):
    """A plain TrialConfig (no scenario_id/mode) is a wiring bug: refuse to run it ungated."""
    q = AwsBenchTrialQueue(n_concurrent=1)
    # _execute_trial_with_retries must never be reached for a non-aws-bench config.
    mocker.patch.object(q, "_execute_trial_with_retries", AsyncMock())

    with pytest.raises(TypeError, match="AwsBenchTrialConfig"):
        await q._run_trial(TrialConfig(task=TaskConfig(path=Path("tasks/orphan"))))


# --- gate unit behavior -----------------------------------------------------


@pytest.mark.asyncio
async def test_gate_excludes_reader_while_writer_holds():
    """A reader cannot enter while a writer holds the scenario."""
    gate = _ScenarioAdmissionGate()

    async def hold_writer(started: asyncio.Event, release: asyncio.Event) -> None:
        async with gate.hold(ConcurrencyMode.MUTATING):
            started.set()
            await release.wait()

    writer_started, release_writer = asyncio.Event(), asyncio.Event()
    writer = asyncio.create_task(hold_writer(writer_started, release_writer))
    await writer_started.wait()

    reader_admitted = asyncio.Event()

    async def hold_reader() -> None:
        async with gate.hold(ConcurrencyMode.READ_ONLY):
            reader_admitted.set()

    reader = asyncio.create_task(hold_reader())
    await asyncio.sleep(0.01)
    assert not reader_admitted.is_set()  # blocked behind the writer

    release_writer.set()
    await asyncio.wait_for(asyncio.gather(writer, reader), timeout=1.0)
    assert reader_admitted.is_set()  # admitted once the writer drained


@pytest.mark.asyncio
async def test_gate_released_on_exception_result(mocker):
    """A trial whose result carries exception_info still releases the gate (finally)."""
    q = AwsBenchTrialQueue(n_concurrent=8)
    m = ConcurrencyMode.MUTATING
    failed = SimpleNamespace(exception_info=SimpleNamespace(exception_type="SomeError"))
    mocker.patch.object(q, "_execute_trial_with_retries", AsyncMock(return_value=failed))

    # First trial fails; the gate must release so the next mutating trial on the
    # same scenario can still acquire it (otherwise this second call would hang).
    await q._run_trial(_trial_config("a", trial_name="a-0", scenario_id="scenario-a", mode=m))
    await asyncio.wait_for(
        q._run_trial(_trial_config("a", trial_name="a-1", scenario_id="scenario-a", mode=m)),
        timeout=1.0,
    )


@pytest.mark.asyncio
async def test_gate_released_when_trial_raises(mocker):
    """A raised exception (not a result) still releases the gate (hold() exit)."""
    q = AwsBenchTrialQueue(n_concurrent=8)
    m = ConcurrencyMode.MUTATING
    # First call raises; a follow-up mutating trial on the same scenario must still
    # acquire the gate (it would hang here if the raise leaked the hold).
    mocker.patch.object(
        q,
        "_execute_trial_with_retries",
        AsyncMock(side_effect=[RuntimeError("boom"), _ok_result()]),
    )

    with pytest.raises(RuntimeError, match="boom"):
        await q._run_trial(_trial_config("a", trial_name="a-0", scenario_id="scenario-a", mode=m))
    await asyncio.wait_for(
        q._run_trial(_trial_config("a", trial_name="a-1", scenario_id="scenario-a", mode=m)),
        timeout=1.0,
    )


@pytest.mark.asyncio
async def test_reader_stream_does_not_deadlock_waiting_writer(mocker):
    """A finite read-only stream drains so a waiting mutating trial eventually runs (liveness)."""
    q, _ = _gated_queue(n_concurrent=8, mocker=mocker)

    readers = [
        q._run_trial(
            _trial_config(
                "ro", trial_name=f"ro-{i}", scenario_id="scenario-a", mode=ConcurrencyMode.READ_ONLY
            )
        )
        for i in range(4)
    ]
    writer = q._run_trial(
        _trial_config(
            "rw", trial_name="rw-0", scenario_id="scenario-a", mode=ConcurrencyMode.MUTATING
        )
    )
    # The writer must complete; if readers could starve it this gather would hang
    # and wait_for would raise TimeoutError. Completing within the budget is the
    # liveness assertion.
    *_, writer_result = await asyncio.wait_for(asyncio.gather(*readers, writer), timeout=2.0)
    assert writer_result.exception_info is None
