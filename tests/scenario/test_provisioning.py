"""Direct unit tests for aws_bench.scenario.provisioning.provision_scenarios.

Mocks AccountManager, QuotaManager, and CredentialProvider so the
orchestrator's lifecycle (events, error paths, summary aggregation, the
optional approval-wait pass) can be exercised without AWS calls.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from aws_bench.account_management.exceptions import AccountCreationError
from aws_bench.resource_management.models import (
    QuotaConfiguration,
    QuotaIncreaseResult,
    QuotaStatus,
)
from aws_bench.resource_management.snapshot.models import SnapshotResult
from aws_bench.scenario.config import ScenarioManifest
from aws_bench.scenario.exceptions import (
    AccountLimitExceededError,
    InsufficientQuotaError,
    UnmetQuota,
)
from aws_bench.scenario.provisioning import (
    DEFAULT_ORG_ACCOUNT_QUOTA,
    ProvisionedAccount,
    ProvisionEvent,
    ProvisionHookEvent,
    ProvisioningSummary,
    SubmittedQuotaBatch,
    _await_quota_approvals,
    _create_account,
    _emit,
    _handle_account_limit_exceeded,
    _is_account_limit_exceeded,
    provision_scenarios,
)
from aws_bench.scenario.scenario import Scenario


def _make_scenario_config(
    name="sc",
    *,
    account_tags=("PRIMARY",),
    regions=("us-east-1",),
    quotas=None,
) -> ScenarioManifest:
    data: dict = {
        "schema_version": "1.0",
        "scenario": {
            "name": name,
            "account_tags": list(account_tags),
            "regions": list(regions),
        },
    }
    if quotas:
        data["quotas"] = quotas
    return ScenarioManifest.model_validate(data)


def _make_scenario_with_quota(
    sdir_root: Path,
    name: str,
    *,
    account_tags=("PRIMARY",),
    regions=("us-east-1",),
    quotas: list[dict] | None = None,
) -> Scenario:
    """Create a scenario dir with optional quotas in scenario.toml."""
    sd = sdir_root / name
    sd.mkdir(parents=True, exist_ok=True)
    quota_block = ""
    if quotas:
        for q in quotas:
            quota_block += (
                "\n[[quotas]]\n"
                f'account_tag = "{q["account_tag"]}"\n'
                f'region = "{q["region"]}"\n'
                f'service_code = "{q["service_code"]}"\n'
                f'quota_code = "{q["quota_code"]}"\n'
                f"desired_value = {q['desired_value']}\n"
            )
    tags_literal = ", ".join(f'"{t}"' for t in account_tags)
    regions_literal = ", ".join(f'"{r}"' for r in regions)
    toml = (
        'schema_version = "1.0"\n\n'
        "[scenario]\n"
        f'name = "{name}"\n'
        f"account_tags = [{tags_literal}]\n"
        f"regions = [{regions_literal}]\n"
        f"{quota_block}"
    )
    (sd / "scenario.toml").write_text(toml)
    (sd / "scenario").mkdir()
    (sd / "scenario" / "Dockerfile").write_text("FROM alpine\n")
    (sd / "deploy").mkdir()
    (sd / "deploy" / "deploy.sh").write_text("#!/bin/sh\nexit 0\n")
    return Scenario(sd)


def _ensure_returning(account_id: str = "111111111111"):
    """Build an AsyncMock for ensure_scenario_accounts that returns a one-tag dict.

    The provisioning unit calls ``ensure_scenario_accounts(ou, scenario, {tag})``
    expecting back ``{tag: account_id}``.
    """

    async def _impl(_ou, _scenario, tags):
        return dict.fromkeys(tags, account_id)

    return AsyncMock(side_effect=_impl)


@pytest.fixture
def mocks():
    """Return mocked AccountManager / QuotaManager / CredentialProvider triples."""
    am = MagicMock()
    am.ensure_scenario_accounts = _ensure_returning("111111111111")
    qm = MagicMock()
    qm.request_quotas = MagicMock(return_value=[])
    cp = MagicMock()
    cp.wait_for_role = MagicMock(return_value=None)
    return am, qm, cp


@pytest.fixture(autouse=True)
def _stub_capture():
    """Stub the PRE_SETUP capture so provisioning tests never touch S3/AWS.

    _provision_one_account always runs ``ResourceManager.capture_pre_setup_baseline`` as
    its final step; patch it to a success by default. Tests that exercise capture
    outcomes override the return value / side effect on the yielded mock.
    """
    with patch(
        "aws_bench.scenario.provisioning.ResourceManager.capture_pre_setup_baseline"
    ) as capture:
        capture.side_effect = lambda account_id, scenario_name, regions, **_: SnapshotResult(
            account_id=account_id, success=True, regions_captured=list(regions)
        )
        yield capture


@pytest.fixture(autouse=True)
def _zero_convergence_delay():
    """Zero out the convergence delay in tests to avoid 60s waits."""
    with patch("aws_bench.scenario.provisioning._ACCOUNT_CONVERGENCE_DELAY_SEC", 0):
        yield


@pytest.fixture(autouse=True)
def _stub_deploy():
    """Stub the discovery-Lambda deploy so tests never touch AWS.

    provision_scenarios deploys the discovery Lambda once per run via
    ``ensure_deployed``, which uses the CredentialProvider singleton, not the
    injected mock, so it must be patched separately.
    """
    with patch("aws_bench.scenario.provisioning.ensure_deployed") as deploy:
        yield deploy


# -- ProvisioningSummary properties ------------------------------------------


def test_summary_all_succeeded_true_when_empty():
    assert ProvisioningSummary().all_succeeded is True


def test_summary_all_succeeded_false_with_a_failure():
    s = ProvisioningSummary(
        accounts=[
            ProvisionedAccount("a", "PRIMARY", account_id="1"),
            ProvisionedAccount("b", "PRIMARY", error=RuntimeError("nope")),
        ]
    )
    assert s.all_succeeded is False
    assert s.n_total == 2
    assert s.n_provisioned == 1
    assert s.n_failed == 1


def _pending_unmet(quota_code="L-1", status=QuotaStatus.REQUESTED):
    return UnmetQuota(
        scenario_name="sc",
        account_id="111111111111",
        region="us-east-1",
        result=QuotaIncreaseResult(
            service_code="ec2",
            quota_code=quota_code,
            desired_value=10.0,
            status=status,
        ),
    )


def test_summary_all_succeeded_true_with_pending_quota_when_not_waited():
    """Without --wait-for-quotas a still-pending quota is expected, not a failure."""
    s = ProvisioningSummary(
        accounts=[ProvisionedAccount("sc", "PRIMARY", account_id="1")],
        unmet_quotas=[_pending_unmet()],
        waited=False,
    )
    assert s.all_succeeded is True


def test_summary_all_succeeded_false_with_pending_quota_after_waiting():
    """After --wait-for-quotas, a quota still pending past the timeout is a failure."""
    s = ProvisioningSummary(
        accounts=[ProvisionedAccount("sc", "PRIMARY", account_id="1")],
        unmet_quotas=[_pending_unmet()],
        waited=True,
    )
    assert s.all_succeeded is False


def test_summary_all_succeeded_false_with_failed_quota_even_when_not_waited():
    """A quota whose request errored (is_failure) fails the run regardless of waiting."""
    s = ProvisioningSummary(
        accounts=[ProvisionedAccount("sc", "PRIMARY", account_id="1")],
        unmet_quotas=[_pending_unmet(status=QuotaStatus.FAILED)],
        waited=False,
    )
    assert s.all_succeeded is False


# -- provision_scenarios end-to-end ------------------------------------------


def test_provision_scenarios_happy_path(mocks, tmp_path):
    am, qm, cp = mocks
    sc = _make_scenario_with_quota(
        tmp_path,
        "sc",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "lambda",
                "quota_code": "L-1",
                "desired_value": 10.0,
            }
        ],
    )
    result = asyncio.run(
        provision_scenarios(
            [sc],
            "ou",
            n_concurrent=2,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
        )
    )
    assert result.all_succeeded
    assert result.n_total == 1
    am.ensure_scenario_accounts.assert_awaited_once_with("ou", "sc", {"PRIMARY"})
    cp.wait_for_role.assert_called_once()
    qm.request_quotas.assert_called_once()


def test_preexisting_mode_validates_without_provisioning(mocks, tmp_path, _stub_deploy):
    """External accounts use read-only readiness checks during env init."""
    am, qm, cp = mocks
    am.is_preexisting = True
    am.runner_role = "AWSBenchRunner"
    cp.get_session_for_account.return_value.client.return_value.get_caller_identity.return_value = {
        "Account": "111111111111"
    }
    qm.verify_quotas.return_value = [
        QuotaIncreaseResult(
            service_code="lambda",
            quota_code="L-1",
            desired_value=10.0,
            status=QuotaStatus.ALREADY_MET,
        )
    ]
    sc = _make_scenario_with_quota(
        tmp_path,
        "sc",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "lambda",
                "quota_code": "L-1",
                "desired_value": 10.0,
            }
        ],
    )
    with patch("aws_bench.scenario.provisioning._validate_cfn_ops_role") as validate_role:
        result = asyncio.run(
            provision_scenarios(
                [sc],
                "ou",
                n_concurrent=1,
                wait_for_quotas=False,
                account_manager=am,
                quota_manager=qm,
                cred_provider=cp,
            )
        )
    assert result.all_succeeded
    _stub_deploy.assert_not_called()
    cp.wait_for_role.assert_not_called()
    validate_role.assert_called_once_with(cp, "111111111111")
    qm.request_quotas.assert_not_called()
    qm.verify_quotas.assert_called_once()


def test_provision_scenarios_deploys_discovery_lambda_once(mocks, tmp_path, _stub_deploy):
    """The discovery Lambda is deployed once per run on the happy path."""
    am, qm, cp = mocks
    sc = _make_scenario_with_quota(tmp_path, "sc")
    asyncio.run(
        provision_scenarios(
            [sc],
            "ou",
            n_concurrent=2,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
        )
    )
    _stub_deploy.assert_called_once()


def test_provision_scenarios_propagates_discovery_lambda_failure(mocks, tmp_path, _stub_deploy):
    """A deploy failure halts provisioning: the Lambda scan is the point, host scan is broken."""
    am, qm, cp = mocks
    _stub_deploy.side_effect = RuntimeError("deploy boom")
    sc = _make_scenario_with_quota(tmp_path, "sc")
    with pytest.raises(RuntimeError, match="deploy boom"):
        asyncio.run(
            provision_scenarios(
                [sc],
                "ou",
                n_concurrent=2,
                wait_for_quotas=False,
                account_manager=am,
                quota_manager=qm,
                cred_provider=cp,
            )
        )


def test_provision_scenarios_skips_lambda_deploy_when_scan_method_not_lambda(
    mocks, tmp_path, _stub_deploy, monkeypatch
):
    """The Lambda is deployed only when the Lambda scan backend is selected.

    Under ``AWSBENCH_SCAN_METHOD=ccapi`` (or host-only ``fastscan``) the discovery
    Lambda is never invoked, so deploying it would be dead work; the gate skips
    ensure_deployed entirely.
    """
    am, qm, cp = mocks
    monkeypatch.setenv("AWSBENCH_SCAN_METHOD", "ccapi")
    sc = _make_scenario_with_quota(tmp_path, "sc")
    result = asyncio.run(
        provision_scenarios(
            [sc],
            "ou",
            n_concurrent=2,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
        )
    )
    assert result.all_succeeded  # provisioning still completes; only the deploy is skipped
    _stub_deploy.assert_not_called()


def test_provision_scenarios_account_failure_skips_quota(mocks, tmp_path):
    am, qm, cp = mocks
    am.ensure_scenario_accounts = AsyncMock(side_effect=RuntimeError("create failed"))
    sc = _make_scenario_with_quota(
        tmp_path,
        "sc",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "x",
                "quota_code": "L-1",
                "desired_value": 1.0,
            }
        ],
    )
    result = asyncio.run(
        provision_scenarios(
            [sc],
            "ou",
            n_concurrent=2,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
        )
    )
    assert result.n_failed == 1
    assert isinstance(result.accounts[0].error, RuntimeError)
    cp.wait_for_role.assert_not_called()
    qm.request_quotas.assert_not_called()


def test_provision_scenarios_role_wait_failure_skips_quota(mocks, tmp_path):
    am, qm, cp = mocks
    cp.wait_for_role = MagicMock(side_effect=TimeoutError("role never came up"))
    sc = _make_scenario_with_quota(
        tmp_path,
        "sc",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "x",
                "quota_code": "L-1",
                "desired_value": 1.0,
            }
        ],
    )
    result = asyncio.run(
        provision_scenarios(
            [sc],
            "ou",
            n_concurrent=2,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
        )
    )
    assert result.n_failed == 1
    assert isinstance(result.accounts[0].error, TimeoutError)
    qm.request_quotas.assert_not_called()


def test_provision_scenarios_quota_submit_failure_recorded_per_batch(mocks, tmp_path):
    """A submit failure in one region records a submit_failure but provisioning continues."""
    am, qm, cp = mocks
    qm.request_quotas = MagicMock(side_effect=RuntimeError("submit denied"))
    sc = _make_scenario_with_quota(
        tmp_path,
        "sc",
        regions=("us-east-1", "us-west-2"),
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "x",
                "quota_code": "L-1",
                "desired_value": 1.0,
            },
            {
                "account_tag": "PRIMARY",
                "region": "us-west-2",
                "service_code": "y",
                "quota_code": "L-2",
                "desired_value": 1.0,
            },
        ],
    )
    result = asyncio.run(
        provision_scenarios(
            [sc],
            "ou",
            n_concurrent=2,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
        )
    )
    acct = result.accounts[0]
    assert acct.error is None
    assert len(acct.submit_failures) == 2
    assert not acct.provisioned  # submit_failures sets provisioned=False


def test_provision_scenarios_n_concurrent_caps_in_flight(mocks, tmp_path):
    """Semaphore actually bounds parallel work."""
    am, _qm, cp = mocks
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def slow_create(_ou, _scenario, tags):
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        async with lock:
            in_flight -= 1
        return dict.fromkeys(tags, "111")

    am.ensure_scenario_accounts = AsyncMock(side_effect=slow_create)
    qm = MagicMock()
    qm.request_quotas = MagicMock(return_value=[])
    qm.get_org_account_quota = MagicMock(return_value=1000.0)
    scenarios = [_make_scenario_with_quota(tmp_path, f"sc{i}") for i in range(8)]
    asyncio.run(
        provision_scenarios(
            scenarios,
            "ou",
            n_concurrent=2,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
        )
    )
    assert peak <= 2


def test_provision_scenarios_emits_full_lifecycle(mocks, tmp_path):
    am, qm, cp = mocks
    events: list[ProvisionHookEvent] = []

    async def cb(e: ProvisionHookEvent) -> None:
        events.append(e)

    sc = _make_scenario_with_quota(
        tmp_path,
        "sc",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "x",
                "quota_code": "L-1",
                "desired_value": 1.0,
            }
        ],
    )
    asyncio.run(
        provision_scenarios(
            [sc],
            "ou",
            n_concurrent=2,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
            on_event=cb,
        )
    )
    fired = [e.event for e in events]
    assert fired == [
        ProvisionEvent.START,
        ProvisionEvent.ACCOUNT_START,
        ProvisionEvent.ROLE_START,
        ProvisionEvent.QUOTAS_START,
        ProvisionEvent.SNAPSHOT_START,
        ProvisionEvent.END,
    ]
    end = events[-1]
    assert end.error is None
    assert end.account_id == "111111111111"


def test_provision_scenarios_hook_exception_does_not_abort(mocks, tmp_path):
    """A faulty observer must not break provisioning."""
    am, qm, cp = mocks

    async def faulty(_e: ProvisionHookEvent) -> None:
        raise RuntimeError("buggy hook")

    sc = _make_scenario_with_quota(tmp_path, "sc")
    result = asyncio.run(
        provision_scenarios(
            [sc],
            "ou",
            n_concurrent=1,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
            on_event=faulty,
        )
    )
    assert result.all_succeeded


def test_provision_scenarios_failure_event_carries_error_string(mocks, tmp_path):
    am, qm, cp = mocks
    am.ensure_scenario_accounts = AsyncMock(side_effect=ValueError("boom"))
    captured: list[ProvisionHookEvent] = []

    async def cb(e: ProvisionHookEvent) -> None:
        captured.append(e)

    asyncio.run(
        provision_scenarios(
            [_make_scenario_with_quota(tmp_path, "sc")],
            "ou",
            n_concurrent=1,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
            on_event=cb,
        )
    )
    end = next(e for e in captured if e.event == ProvisionEvent.END)
    assert end.error == "boom"
    assert end.succeeded is False


def test_provision_end_event_succeeded_true_on_clean_provision(mocks, tmp_path, _stub_capture):
    """A cleanly provisioned account's END event carries succeeded=True."""
    am, qm, cp = mocks
    captured: list[ProvisionHookEvent] = []

    async def cb(e: ProvisionHookEvent) -> None:
        captured.append(e)

    asyncio.run(
        provision_scenarios(
            [_make_scenario_with_quota(tmp_path, "sc")],
            "ou",
            n_concurrent=1,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
            on_event=cb,
        )
    )
    end = next(e for e in captured if e.event == ProvisionEvent.END)
    assert end.succeeded is True
    assert end.error is None


def test_provision_end_event_succeeded_false_on_snapshot_failure(mocks, tmp_path, _stub_capture):
    """A failed baseline capture sets succeeded=False even though error is None.

    Snapshot failure lands in snapshot_result.success, not result.error, so the
    bar must key on succeeded (mirroring ProvisionedAccount.provisioned), not error.
    """
    am, qm, cp = mocks
    _stub_capture.side_effect = lambda account_id, scenario_name, regions, **_: SnapshotResult(
        account_id=account_id, success=False, error_message="InvalidClientTokenId"
    )
    captured: list[ProvisionHookEvent] = []

    async def cb(e: ProvisionHookEvent) -> None:
        captured.append(e)

    asyncio.run(
        provision_scenarios(
            [_make_scenario_with_quota(tmp_path, "sc")],
            "ou",
            n_concurrent=1,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
            on_event=cb,
        )
    )
    end = next(e for e in captured if e.event == ProvisionEvent.END)
    assert end.succeeded is False
    assert end.error is None


# -- PRE_SETUP baseline capture ----------------------------------------------


def test_provision_captures_baseline_after_quotas(mocks, tmp_path, _stub_capture):
    """Capture runs as the final step, receiving (account_id, scenario, regions)."""
    am, qm, cp = mocks
    sc = _make_scenario_with_quota(tmp_path, "sc", regions=("us-east-1", "us-west-2"))
    result = asyncio.run(
        provision_scenarios(
            [sc],
            "ou",
            n_concurrent=1,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
        )
    )
    _stub_capture.assert_called_once()
    args = _stub_capture.call_args
    assert args.args[0] == "111111111111"
    assert args.args[1] == "sc"
    assert args.args[2] == ["us-east-1", "us-west-2"]
    assert result.accounts[0].snapshot_result is not None
    assert result.accounts[0].provisioned is True


def test_provision_snapshot_failure_fails_account(mocks, tmp_path, _stub_capture):
    """A failed SnapshotResult flips provisioned to False and counts as a failure."""
    am, qm, cp = mocks
    _stub_capture.side_effect = lambda account_id, *a, **k: SnapshotResult(
        account_id=account_id, success=False, error_message="scan failed"
    )
    sc = _make_scenario_with_quota(tmp_path, "sc")
    result = asyncio.run(
        provision_scenarios(
            [sc],
            "ou",
            n_concurrent=1,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
        )
    )
    acct = result.accounts[0]
    assert acct.error is None
    assert acct.snapshot_result is not None and acct.snapshot_result.success is False
    assert acct.provisioned is False
    assert result.n_failed == 1


def test_provision_snapshot_exception_is_trapped(mocks, tmp_path, _stub_capture):
    """A capture that raises is caught and converted to a failed result.

    The raised exception must not abort the account's TaskGroup.
    """
    am, qm, cp = mocks
    _stub_capture.side_effect = RuntimeError("unexpected capture crash")
    sc = _make_scenario_with_quota(tmp_path, "sc")
    result = asyncio.run(
        provision_scenarios(
            [sc],
            "ou",
            n_concurrent=1,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
        )
    )
    acct = result.accounts[0]
    assert acct.snapshot_result is not None and acct.snapshot_result.success is False
    assert "unexpected capture crash" in (acct.snapshot_result.error_message or "")
    assert acct.provisioned is False


# -- approval-wait pass ------------------------------------------------------


def test_provision_scenarios_wait_returns_when_quotas_already_met(mocks, tmp_path):
    """End-to-end: wait_for_quotas=True with verify_quotas returning ALREADY_MET succeeds."""
    am, qm, cp = mocks
    qm.verify_quotas = MagicMock(
        return_value=[
            QuotaIncreaseResult(
                service_code="x",
                quota_code="L-1",
                desired_value=1.0,
                status=QuotaStatus.ALREADY_MET,
            )
        ]
    )
    sc = _make_scenario_with_quota(
        tmp_path,
        "sc",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "x",
                "quota_code": "L-1",
                "desired_value": 1.0,
            }
        ],
    )
    with patch("aws_bench.scenario.job.QuotaManager", return_value=qm):
        result = asyncio.run(
            provision_scenarios(
                [sc],
                "ou",
                n_concurrent=1,
                wait_for_quotas=True,
                quota_timeout=10,
                poll_interval=0,
                account_manager=am,
                quota_manager=qm,
                cred_provider=cp,
            )
        )
    assert result.unmet_quotas == []
    assert qm.verify_quotas.call_count >= 1
    qm.request_quotas.assert_called_once()


def test_provision_scenarios_wait_records_unmet_quota_on_timeout(mocks, tmp_path):
    """End-to-end: a quota that stays PENDING past quota_timeout surfaces in unmet_quotas."""
    am, qm, cp = mocks
    qm.verify_quotas = MagicMock(
        return_value=[
            QuotaIncreaseResult(
                service_code="x",
                quota_code="L-1",
                desired_value=1.0,
                status=QuotaStatus.ALREADY_PENDING,
                error_message="current=0.0, required=1.0 (PENDING)",
            )
        ]
    )
    sc = _make_scenario_with_quota(
        tmp_path,
        "sc",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "x",
                "quota_code": "L-1",
                "desired_value": 1.0,
            }
        ],
    )
    with patch("aws_bench.scenario.job.QuotaManager", return_value=qm):
        result = asyncio.run(
            provision_scenarios(
                [sc],
                "ou",
                n_concurrent=1,
                wait_for_quotas=True,
                quota_timeout=0,
                poll_interval=0,
                account_manager=am,
                quota_manager=qm,
                cred_provider=cp,
            )
        )
    assert len(result.unmet_quotas) == 1
    unmet = result.unmet_quotas[0]
    assert unmet.scenario_name == "sc"
    assert unmet.account_id == "111111111111"
    assert unmet.region == "us-east-1"
    assert unmet.result.quota_code == "L-1"
    assert unmet.result.status == QuotaStatus.ALREADY_PENDING
    assert "PENDING" in unmet.result.error_message


def test_provision_scenarios_seeds_unmet_from_submit_without_waiting(mocks, tmp_path):
    """Without --wait-for-quotas, a freshly REQUESTED quota lands in unmet_quotas.

    The submit-time status (REQUESTED, not ALREADY_MET) is the honest signal that
    the quota is not yet satisfied, so the summary can show it and the pending
    table without any extra verify call or the wait pass.
    """
    am, qm, cp = mocks
    qm.request_quotas = MagicMock(
        return_value=[
            QuotaIncreaseResult(
                service_code="ec2",
                quota_code="L-MET",
                desired_value=10.0,
                status=QuotaStatus.ALREADY_MET,
            ),
            QuotaIncreaseResult(
                service_code="ec2",
                quota_code="L-PENDING",
                desired_value=10.0,
                status=QuotaStatus.REQUESTED,
            ),
        ]
    )
    sc = _make_scenario_with_quota(
        tmp_path,
        "sc",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "ec2",
                "quota_code": "L-MET",
                "desired_value": 10.0,
            },
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "ec2",
                "quota_code": "L-PENDING",
                "desired_value": 10.0,
            },
        ],
    )
    result = asyncio.run(
        provision_scenarios(
            [sc],
            "ou",
            n_concurrent=1,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
        )
    )
    # Only the REQUESTED quota is unmet; the ALREADY_MET one is not listed.
    assert len(result.unmet_quotas) == 1
    unmet = result.unmet_quotas[0]
    assert unmet.scenario_name == "sc"
    assert unmet.account_id == "111111111111"
    assert unmet.region == "us-east-1"
    assert unmet.result.quota_code == "L-PENDING"
    assert unmet.result.status == QuotaStatus.REQUESTED
    # Not waited, and no hard failure -> the run still succeeds (exit 0).
    assert result.waited is False
    assert result.all_succeeded is True
    # Submit was not re-verified: no verify_quotas call in the non-wait path.
    qm.verify_quotas.assert_not_called()


def test_provision_scenarios_no_unmet_when_all_submits_already_met(mocks, tmp_path):
    """Every quota ALREADY_MET at submit -> empty unmet list, clean success."""
    am, qm, cp = mocks
    qm.request_quotas = MagicMock(
        return_value=[
            QuotaIncreaseResult(
                service_code="ec2",
                quota_code="L-1",
                desired_value=10.0,
                status=QuotaStatus.ALREADY_MET,
            )
        ]
    )
    sc = _make_scenario_with_quota(
        tmp_path,
        "sc",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "ec2",
                "quota_code": "L-1",
                "desired_value": 10.0,
            }
        ],
    )
    result = asyncio.run(
        provision_scenarios(
            [sc],
            "ou",
            n_concurrent=1,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
        )
    )
    assert result.unmet_quotas == []
    assert result.all_succeeded is True


def test_provision_scenarios_wait_sets_waited_flag(mocks, tmp_path):
    """The wait pass records waited=True so a leftover pending quota fails the run."""
    am, qm, cp = mocks
    qm.request_quotas = MagicMock(
        return_value=[
            QuotaIncreaseResult(
                service_code="x",
                quota_code="L-1",
                desired_value=1.0,
                status=QuotaStatus.REQUESTED,
            )
        ]
    )
    qm.verify_quotas = MagicMock(
        return_value=[
            QuotaIncreaseResult(
                service_code="x",
                quota_code="L-1",
                desired_value=1.0,
                status=QuotaStatus.ALREADY_PENDING,
                error_message="current=0.0, required=1.0 (PENDING)",
            )
        ]
    )
    sc = _make_scenario_with_quota(
        tmp_path,
        "sc",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "x",
                "quota_code": "L-1",
                "desired_value": 1.0,
            }
        ],
    )
    with patch("aws_bench.scenario.job.QuotaManager", return_value=qm):
        result = asyncio.run(
            provision_scenarios(
                [sc],
                "ou",
                n_concurrent=1,
                wait_for_quotas=True,
                quota_timeout=0,
                poll_interval=0,
                account_manager=am,
                quota_manager=qm,
                cred_provider=cp,
            )
        )
    assert result.waited is True
    assert len(result.unmet_quotas) == 1
    assert result.all_succeeded is False


# -- _await_quota_approvals direct ------------------------------------------


@pytest.mark.asyncio
async def test_await_quota_approvals_returns_when_all_already_met(tmp_path):
    """First poll finds every quota at current >= desired -> exit fast."""
    scenario = _make_scenario_with_quota(
        tmp_path,
        "alpha",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "ec2",
                "quota_code": "L-1216C47A",
                "desired_value": 50,
            }
        ],
    )
    account_mappings = {"alpha": {"PRIMARY": "111111111111"}}

    qm = MagicMock()
    qm.verify_quotas.return_value = [
        QuotaIncreaseResult(
            service_code="ec2",
            quota_code="L-1216C47A",
            desired_value=50.0,
            status=QuotaStatus.ALREADY_MET,
        )
    ]

    cp = MagicMock()
    with patch("aws_bench.scenario.job.QuotaManager", return_value=qm):
        await _await_quota_approvals(
            [scenario],
            account_mappings,
            cred_provider=cp,
            quota_timeout=60,
            poll_interval=1,
            n_concurrent=4,
        )
    assert qm.verify_quotas.call_count == 1


@pytest.mark.asyncio
async def test_await_quota_approvals_polls_until_current_value_reaches_desired(tmp_path):
    """First two polls return PENDING; third returns ALREADY_MET -> exit."""
    scenario = _make_scenario_with_quota(
        tmp_path,
        "alpha",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "ec2",
                "quota_code": "L-1216C47A",
                "desired_value": 50,
            }
        ],
    )
    account_mappings = {"alpha": {"PRIMARY": "111111111111"}}

    qm = MagicMock()
    pending = QuotaIncreaseResult(
        service_code="ec2",
        quota_code="L-1216C47A",
        desired_value=50.0,
        status=QuotaStatus.ALREADY_PENDING,
        error_message="current=8.0, required=50.0 (PENDING)",
    )
    met = QuotaIncreaseResult(
        service_code="ec2",
        quota_code="L-1216C47A",
        desired_value=50.0,
        status=QuotaStatus.ALREADY_MET,
    )
    qm.verify_quotas.side_effect = [[pending], [pending], [met]]

    cp = MagicMock()
    with patch("aws_bench.scenario.job.QuotaManager", return_value=qm):
        await _await_quota_approvals(
            [scenario],
            account_mappings,
            cred_provider=cp,
            quota_timeout=120,
            poll_interval=0,
            n_concurrent=4,
        )
    assert qm.verify_quotas.call_count == 3


@pytest.mark.asyncio
async def test_await_quota_approvals_raises_on_timeout(tmp_path):
    """Quota stays PENDING past quota_timeout -> InsufficientQuotaError listing the unmet quotas."""
    scenario = _make_scenario_with_quota(
        tmp_path,
        "alpha",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "ec2",
                "quota_code": "L-1216C47A",
                "desired_value": 50,
            }
        ],
    )
    account_mappings = {"alpha": {"PRIMARY": "111111111111"}}

    qm = MagicMock()
    qm.verify_quotas.return_value = [
        QuotaIncreaseResult(
            service_code="ec2",
            quota_code="L-1216C47A",
            desired_value=50.0,
            status=QuotaStatus.ALREADY_PENDING,
            error_message="current=8.0, required=50.0 (PENDING)",
        )
    ]

    cp = MagicMock()
    with patch("aws_bench.scenario.job.QuotaManager", return_value=qm):
        with pytest.raises(InsufficientQuotaError) as excinfo:
            await _await_quota_approvals(
                [scenario],
                account_mappings,
                cred_provider=cp,
                quota_timeout=0,
                poll_interval=0,
                n_concurrent=4,
            )
    assert len(excinfo.value.failures) == 1
    assert excinfo.value.failures[0].scenario_name == "alpha"
    assert excinfo.value.failures[0].region == "us-east-1"


@pytest.mark.asyncio
async def test_await_quota_approvals_retries_transient_verify_failures(tmp_path):
    """A non-InsufficientQuotaError exception (e.g. throttle) does not abort the wait."""
    scenario = _make_scenario_with_quota(
        tmp_path,
        "alpha",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "ec2",
                "quota_code": "L-1216C47A",
                "desired_value": 50,
            }
        ],
    )
    account_mappings = {"alpha": {"PRIMARY": "111111111111"}}

    qm = MagicMock()
    met = QuotaIncreaseResult(
        service_code="ec2",
        quota_code="L-1216C47A",
        desired_value=50.0,
        status=QuotaStatus.ALREADY_MET,
    )
    qm.verify_quotas.side_effect = [RuntimeError("throttled"), [met]]

    cp = MagicMock()
    with patch("aws_bench.scenario.job.QuotaManager", return_value=qm):
        await _await_quota_approvals(
            [scenario],
            account_mappings,
            cred_provider=cp,
            quota_timeout=60,
            poll_interval=0,
            n_concurrent=4,
        )
    assert qm.verify_quotas.call_count == 2


@pytest.mark.asyncio
async def test_await_quota_approvals_transient_error_raised_after_timeout(tmp_path):
    """A transient error past the deadline propagates instead of looping forever."""
    scenario = _make_scenario_with_quota(
        tmp_path,
        "alpha",
        quotas=[
            {
                "account_tag": "PRIMARY",
                "region": "us-east-1",
                "service_code": "ec2",
                "quota_code": "L-1216C47A",
                "desired_value": 50,
            }
        ],
    )
    account_mappings = {"alpha": {"PRIMARY": "111111111111"}}

    qm = MagicMock()
    qm.verify_quotas.side_effect = RuntimeError("still throttled")

    cp = MagicMock()
    with patch("aws_bench.scenario.job.QuotaManager", return_value=qm):
        with pytest.raises(RuntimeError, match="still throttled"):
            await _await_quota_approvals(
                [scenario],
                account_mappings,
                cred_provider=cp,
                quota_timeout=0,
                poll_interval=0,
                n_concurrent=4,
            )


# -- _create_account cancellation ---------------------------------------------


def test_create_account_cancel_emits_cancel_then_reraises(mocks):
    """asyncio.CancelledError fires CANCEL, runs END in finally, re-raises."""
    am, qm, cp = mocks
    am.ensure_scenario_accounts = AsyncMock(side_effect=asyncio.CancelledError())
    captured: list[ProvisionHookEvent] = []

    async def cb(e: ProvisionHookEvent) -> None:
        captured.append(e)

    async def go():
        await _create_account(am, "ou", _make_scenario_config(), "PRIMARY", cb)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(go())

    fired = [e.event for e in captured]
    assert ProvisionEvent.CANCEL in fired
    assert ProvisionEvent.END in fired


# -- _emit -------------------------------------------------------------------


def test_emit_no_callback_is_noop():
    asyncio.run(_emit(None, ProvisionEvent.START, scenario_name="x", account_tag="P"))


def test_emit_swallows_callback_exception():
    async def boom(_e):
        raise RuntimeError("hook failed")

    asyncio.run(_emit(boom, ProvisionEvent.START, scenario_name="x", account_tag="P"))


# -- ProvisionedAccount / SubmittedQuotaBatch construction -------------------


def test_provisioned_account_provisioned_flag_logic():
    base = ProvisionedAccount("s", "P", account_id="1")
    assert base.provisioned is True

    err = ProvisionedAccount("s", "P", error=RuntimeError("x"))
    assert err.provisioned is False

    partial = ProvisionedAccount("s", "P", account_id="1", submit_failures=[RuntimeError("x")])
    assert partial.provisioned is False

    snap_ok = ProvisionedAccount(
        "s", "P", account_id="1", snapshot_result=SnapshotResult(account_id="1", success=True)
    )
    assert snap_ok.provisioned is True

    snap_bad = ProvisionedAccount(
        "s",
        "P",
        account_id="1",
        snapshot_result=SnapshotResult(account_id="1", success=False, error_message="boom"),
    )
    assert snap_bad.provisioned is False


def test_submitted_quota_batch_carries_full_context():
    cfg = QuotaConfiguration(region="us-east-1", increases=[])
    batch = SubmittedQuotaBatch(
        scenario_name="sc", account_tag="PRIMARY", account_id="111", config=cfg
    )
    assert batch.scenario_name == "sc"
    assert batch.account_id == "111"
    assert batch.config.region == "us-east-1"


# -- Three-phase provisioning tests -------------------------------------------


def test_account_creation_capped_at_3_concurrent(mocks, tmp_path, _stub_capture):
    """Phase 1: Account creation should run at most 3 concurrently, regardless of n_concurrent."""
    am, qm, cp = mocks

    # Track concurrent account creation calls
    import threading

    max_concurrent = 0
    current_concurrent = 0
    lock = threading.Lock()

    original_ensure = am.ensure_scenario_accounts

    async def tracked_ensure(*args, **kwargs):
        nonlocal max_concurrent, current_concurrent
        with lock:
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
        await asyncio.sleep(0.01)  # Simulate a brief delay
        result = await original_ensure(*args, **kwargs)
        with lock:
            current_concurrent -= 1
        return result

    am.ensure_scenario_accounts = AsyncMock(side_effect=tracked_ensure)

    # Create 5 scenarios (more than the cap of 3)
    scenarios = [_make_scenario_with_quota(tmp_path / f"sc-{i}", f"sc-{i}") for i in range(5)]
    summary = asyncio.run(
        provision_scenarios(
            scenarios,
            "test-ou",
            n_concurrent=10,  # High n_concurrent, but account creation should still cap at 3
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
        )
    )

    assert max_concurrent <= 3, f"Account creation concurrency was {max_concurrent}, expected <= 3"
    assert len(summary.accounts) == 5


def test_account_creation_failures_dont_block_phase_2(mocks, tmp_path, _stub_capture):
    """Phase 2 should proceed with only the successfully created accounts."""
    am, qm, cp = mocks

    # First 2 accounts succeed, 3rd fails
    call_count = 0

    async def selective_ensure(_ou, _scenario, tags):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise RuntimeError("Account creation failed")
        return dict.fromkeys(tags, f"11111111111{call_count}")

    am.ensure_scenario_accounts = AsyncMock(side_effect=selective_ensure)

    scenarios = [_make_scenario_with_quota(tmp_path / f"sc-{i}", f"sc-{i}") for i in range(3)]
    summary = asyncio.run(
        provision_scenarios(
            scenarios,
            "test-ou",
            n_concurrent=5,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
        )
    )

    # 2 should be provisioned, 1 should have an error
    provisioned = [a for a in summary.accounts if a.provisioned]
    failed = [a for a in summary.accounts if a.error is not None]
    assert len(provisioned) == 2
    assert len(failed) == 1


def test_quota_and_snapshot_use_n_concurrent(mocks, tmp_path, _stub_capture):
    """Phase 2 (quota + snapshot) should respect n_concurrent, not the account creation cap."""
    am, qm, cp = mocks

    # Track concurrent quota submission calls
    import threading

    max_concurrent_quota = 0
    current_concurrent_quota = 0
    lock = threading.Lock()

    original_request = qm.request_quotas

    def tracked_request(*args, **kwargs):
        nonlocal max_concurrent_quota, current_concurrent_quota
        with lock:
            current_concurrent_quota += 1
            max_concurrent_quota = max(max_concurrent_quota, current_concurrent_quota)
        import time

        time.sleep(0.01)
        result = original_request(*args, **kwargs)
        with lock:
            current_concurrent_quota -= 1
        return result

    qm.request_quotas = MagicMock(side_effect=tracked_request)

    scenarios = [_make_scenario_with_quota(tmp_path / f"sc-{i}", f"sc-{i}") for i in range(5)]
    summary = asyncio.run(
        provision_scenarios(
            scenarios,
            "test-ou",
            n_concurrent=5,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
        )
    )

    # Phase 2 should allow up to n_concurrent (5) concurrent quota submissions
    # (vs Phase 1 which caps at 3 for account creation)
    assert max_concurrent_quota <= 5
    assert summary.all_succeeded


def test_convergence_delay_between_phases(mocks, tmp_path, _stub_capture):
    """There should be a delay between Phase 1 (account creation) and Phase 2 (lifecycle)."""
    import time

    am, qm, cp = mocks

    # Track when Phase 2 starts (role wait is the first thing in Phase 2)
    phase2_start_time = None

    # Simulate a slow account creation (> 5s) to trigger newly_created=True
    async def slow_ensure(_ou, _scenario, tags):
        await asyncio.sleep(5.1)  # > 5s threshold triggers newly_created
        return dict.fromkeys(tags, "111111111111")

    am.ensure_scenario_accounts = AsyncMock(side_effect=slow_ensure)

    original_wait = cp.wait_for_role

    def timed_wait(*args, **kwargs):
        nonlocal phase2_start_time
        if phase2_start_time is None:
            phase2_start_time = time.monotonic()
        return original_wait(*args, **kwargs)

    cp.wait_for_role = MagicMock(side_effect=timed_wait)

    scenarios = [_make_scenario_with_quota(tmp_path / "sc-delay", "sc-delay")]

    # Patch the delay to 0.1s for fast testing (don't wait 60s in tests)
    with patch("aws_bench.scenario.provisioning._ACCOUNT_CONVERGENCE_DELAY_SEC", 0.1):
        start = time.monotonic()
        asyncio.run(
            provision_scenarios(
                scenarios,
                "test-ou",
                n_concurrent=5,
                wait_for_quotas=False,
                account_manager=am,
                quota_manager=qm,
                cred_provider=cp,
            )
        )

    # Phase 2 should start after the convergence delay
    assert phase2_start_time is not None
    # The total time should include the 5.1s creation + 0.1s delay
    total = phase2_start_time - start
    assert total >= 5.1, f"Expected >= 5.1s total, got {total:.3f}s"


# ---------------------------------------------------------------------------
# Reactive account-limit handling
# ---------------------------------------------------------------------------

_ASYNC_LIMIT_ERR = AccountCreationError(
    "CreateAccount failed for 'sc-PRIMARY': ACCOUNT_LIMIT_EXCEEDED"
)
_OTHER_CONSTRAINT_ERR = ClientError(
    {"Error": {"Code": "ConstraintViolationException", "Message": "Email already exists"}},
    "CreateAccount",
)


def _account_limit_client_error() -> ClientError:
    """The synchronous CreateAccount error when the org is at its account limit."""
    return ClientError(
        {
            "Error": {
                "Code": "ConstraintViolationException",
                "Message": "You have exceeded the allowed number of AWS accounts.",
            }
        },
        "CreateAccount",
    )


@pytest.mark.parametrize(
    "exc, expected",
    [
        (None, False),
        (RuntimeError("boom"), False),
        (_ASYNC_LIMIT_ERR, True),
        (_OTHER_CONSTRAINT_ERR, False),
    ],
)
def test_is_account_limit_exceeded(exc, expected):
    """Detects both the sync ConstraintViolationException and async ACCOUNT_LIMIT_EXCEEDED."""
    assert _is_account_limit_exceeded(exc) is expected


def test_is_account_limit_exceeded_sync_client_error():
    """The synchronous 'allowed number of AWS accounts' ClientError is detected."""
    assert _is_account_limit_exceeded(_account_limit_client_error()) is True


def test_handle_account_limit_exceeded_files_request():
    """Files an increase (none pending) and raises with a 'filed' detail."""
    qm = MagicMock()
    qm.request_org_account_quota_if_absent = MagicMock(
        return_value=QuotaIncreaseResult(
            "organizations", "L-E619E033", DEFAULT_ORG_ACCOUNT_QUOTA, QuotaStatus.REQUESTED
        )
    )

    with pytest.raises(AccountLimitExceededError) as exc_info:
        asyncio.run(_handle_account_limit_exceeded(qm))

    qm.request_org_account_quota_if_absent.assert_called_once_with(DEFAULT_ORG_ACCOUNT_QUOTA)
    assert "Filed a request" in str(exc_info.value)


def test_handle_account_limit_exceeded_reports_existing_pending():
    """When a request is already pending, the raised message says so (no duplicate)."""
    qm = MagicMock()
    qm.request_org_account_quota_if_absent = MagicMock(
        return_value=QuotaIncreaseResult(
            "organizations", "L-E619E033", DEFAULT_ORG_ACCOUNT_QUOTA, QuotaStatus.ALREADY_PENDING
        )
    )

    with pytest.raises(AccountLimitExceededError) as exc_info:
        asyncio.run(_handle_account_limit_exceeded(qm))

    assert "already pending" in str(exc_info.value)


def test_provision_scenarios_reacts_to_account_limit(mocks, tmp_path):
    """Account creation hitting the org limit files a request and raises."""
    am, qm, cp = mocks
    am.ensure_scenario_accounts = AsyncMock(side_effect=_account_limit_client_error())
    qm.request_org_account_quota_if_absent = MagicMock(
        return_value=QuotaIncreaseResult(
            "organizations", "L-E619E033", DEFAULT_ORG_ACCOUNT_QUOTA, QuotaStatus.REQUESTED
        )
    )
    sc = _make_scenario_with_quota(tmp_path, "sc")

    with pytest.raises(AccountLimitExceededError):
        asyncio.run(
            provision_scenarios(
                [sc],
                "ou",
                n_concurrent=2,
                wait_for_quotas=False,
                account_manager=am,
                quota_manager=qm,
                cred_provider=cp,
            )
        )

    qm.request_org_account_quota_if_absent.assert_called_once_with(DEFAULT_ORG_ACCOUNT_QUOTA)


def test_provision_scenarios_no_quota_action_without_limit_error(mocks, tmp_path):
    """Happy path: no account-limit error, so no reactive quota request is filed."""
    am, qm, cp = mocks
    qm.request_org_account_quota_if_absent = MagicMock()
    sc = _make_scenario_with_quota(tmp_path, "sc")

    result = asyncio.run(
        provision_scenarios(
            [sc],
            "ou",
            n_concurrent=2,
            wait_for_quotas=False,
            account_manager=am,
            quota_manager=qm,
            cred_provider=cp,
        )
    )

    qm.request_org_account_quota_if_absent.assert_not_called()
    assert result.n_total == 1
