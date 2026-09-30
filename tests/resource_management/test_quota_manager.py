"""Unit tests for QuotaManager."""

from __future__ import annotations

import logging
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from aws_bench.resource_management.exceptions import ConfigurationError, DeploymentError
from aws_bench.resource_management.models import (
    QuotaConfiguration,
    QuotaIncreaseRequest,
    QuotaIncreaseResult,
    QuotaStatus,
)
from aws_bench.resource_management.quota_manager import (
    ORG_ACCOUNT_QUOTA_CODE,
    ORG_QUOTA_REGION,
    ORG_QUOTA_SERVICE_CODE,
    QuotaManager,
)

_TEST_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client_error(code: str, message: str = "mocked") -> ClientError:
    """Build a botocore ClientError with the given error code."""
    return ClientError(
        {"Error": {"Code": code, "Message": message}},
        "request_service_quota_increase",
    )


API_SUCCESS = "success"
API_ALREADY_EXISTS = "ResourceAlreadyExistsException"
API_ILLEGAL_ARG = "IllegalArgumentException"


def _build_mock_client(scenarios: list[str]) -> MagicMock:
    """Return a mock service-quotas client.

    The request_service_quota_increase side effects match the given
    scenario list (one per call).
    """
    client = MagicMock()
    side_effects = []
    for scenario in scenarios:
        if scenario == API_SUCCESS:
            side_effects.append({"RequestedQuota": {"Id": "mock-id"}})
        elif scenario == API_ALREADY_EXISTS:
            side_effects.append(_make_client_error("ResourceAlreadyExistsException"))
        elif scenario == API_ILLEGAL_ARG:
            side_effects.append(_make_client_error("IllegalArgumentException"))
        else:
            side_effects.append(_make_client_error(scenario, f"Unexpected: {scenario}"))

    def _side_effect(**kwargs):
        val = side_effects.pop(0)
        if isinstance(val, Exception):
            raise val
        return val

    client.request_service_quota_increase.side_effect = _side_effect
    client.list_requested_service_quota_change_history_by_quota.return_value = {
        "RequestedQuotas": [{"Status": "APPROVED"}],
    }
    return client


def _build_manager_with_mock_client(mock_client: MagicMock) -> QuotaManager:
    """Return a QuotaManager whose credential provider yields the given mock client."""
    mock_session = MagicMock()
    mock_session.client.return_value = mock_client

    mock_cred_provider = MagicMock()
    mock_cred_provider.get_session_for_account.return_value = mock_session

    return QuotaManager(mock_cred_provider)


# ---------------------------------------------------------------------------
# Invalid configuration is always rejected (Req 1.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "service_code, quota_code, desired_value, expected_field",
    [
        ("", "L-F678F1CE", 10.0, "service_code"),
        ("   ", "L-F678F1CE", 10.0, "service_code"),
        ("\t\n", "L-F678F1CE", 10.0, "service_code"),
        ("vpc", "", 10.0, "quota_code"),
        ("vpc", "   ", 10.0, "quota_code"),
        ("vpc", "L-F678F1CE", 0.0, "desired_value"),
        ("vpc", "L-F678F1CE", -5.0, "desired_value"),
    ],
)
def test_invalid_request_raises_configuration_error(
    service_code,
    quota_code,
    desired_value,
    expected_field,
):
    """Invalid QuotaIncreaseRequest raises ConfigurationError identifying the bad field."""
    bad_request = QuotaIncreaseRequest(service_code, quota_code, desired_value)
    config = QuotaConfiguration(increases=[bad_request])

    with pytest.raises(ConfigurationError) as exc_info:
        QuotaManager.validate_config(config)

    assert expected_field in str(exc_info.value)


# ---------------------------------------------------------------------------
# Result count equals request count (Req 2.1, 6.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenarios",
    [
        [API_SUCCESS],
        [API_SUCCESS, API_SUCCESS, API_SUCCESS],
        [API_ALREADY_EXISTS],
        [API_ILLEGAL_ARG, API_SUCCESS],
        [API_SUCCESS, API_ALREADY_EXISTS, API_ILLEGAL_ARG],
    ],
)
def test_result_count_matches_request_count(scenarios):
    """request_quotas returns exactly N results for N requests."""
    requests = [QuotaIncreaseRequest(f"svc-{i}", f"q-{i}", 10.0) for i in range(len(scenarios))]
    mock_client = _build_mock_client(scenarios)
    manager = _build_manager_with_mock_client(mock_client)
    config = QuotaConfiguration(increases=requests)

    results = manager.request_quotas(config, "123456789012", "TestRole")

    assert len(results) == len(requests)
    for result in results:
        assert isinstance(result, QuotaIncreaseResult)


# ---------------------------------------------------------------------------
# API outcome maps to correct status (Req 2.2, 2.3, 2.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario, expected_status",
    [
        (API_SUCCESS, QuotaStatus.REQUESTED),
        (API_ALREADY_EXISTS, QuotaStatus.ALREADY_PENDING),
        (API_ILLEGAL_ARG, QuotaStatus.ALREADY_MET),
    ],
)
def test_status_matches_api_outcome(scenario, expected_status):
    """The result status is determined by the API response."""
    request = QuotaIncreaseRequest("vpc", "L-F678F1CE", 10.0)
    mock_client = _build_mock_client([scenario])
    manager = _build_manager_with_mock_client(mock_client)

    result = manager._request_increase(mock_client, request, _TEST_LOG)

    assert result.status == expected_status
    assert result.service_code == request.service_code
    assert result.quota_code == request.quota_code
    assert result.desired_value == request.desired_value


# ---------------------------------------------------------------------------
# Unexpected API errors produce DeploymentError (Req 2.5, 6.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error_code",
    [
        "AccessDenied",
        "AccessDeniedException",
        "UnauthorizedOperation",
        "ThrottlingException",
        "ServiceException",
    ],
)
def test_unexpected_error_raises_deployment_error_with_quota_code(error_code):
    """Unexpected ClientError raises DeploymentError containing the quota code."""
    request = QuotaIncreaseRequest("vpc", "L-F678F1CE", 10.0)
    mock_client = MagicMock()
    mock_client.request_service_quota_increase.side_effect = _make_client_error(
        error_code, "Something went wrong"
    )
    manager = _build_manager_with_mock_client(mock_client)

    with pytest.raises(DeploymentError) as exc_info:
        manager._request_increase(mock_client, request, _TEST_LOG)

    msg = str(exc_info.value)
    assert request.quota_code in msg
    assert error_code in msg
    mock_client.request_service_quota_increase.assert_called_once()


# ---------------------------------------------------------------------------
# RegionDisabledException retry / exhaustion (STS race condition fix)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        _make_client_error("RegionDisabledException"),
        _make_client_error("AccessDeniedException", "explicit deny in a service control policy"),
    ],
)
def test_region_disabled_retries_then_succeeds(error: ClientError) -> None:
    """Explicit regional access rejection on first attempt retries and succeeds."""
    request = QuotaIncreaseRequest("vpc", "L-F678F1CE", 10.0)
    mock_client = MagicMock()
    mock_client.request_service_quota_increase.side_effect = [
        error,
        {"RequestedQuota": {"Id": "mock-id"}},
    ]
    manager = _build_manager_with_mock_client(mock_client)

    with patch("aws_bench.resource_management.quota_manager.time.sleep") as mock_sleep:
        result = manager._request_increase(mock_client, request, _TEST_LOG, max_retries=3)

    assert result.status == QuotaStatus.REQUESTED
    mock_sleep.assert_called_once_with(10)
    assert mock_client.request_service_quota_increase.call_count == 2


@pytest.mark.parametrize(
    "error",
    [
        _make_client_error("RegionDisabledException"),
        _make_client_error("AccessDeniedException", "explicit deny in a service control policy"),
    ],
)
def test_region_disabled_exhausts_retries_raises_deployment_error(error: ClientError) -> None:
    """Persistent regional access rejection exhausts the existing retry limit."""
    request = QuotaIncreaseRequest("vpc", "L-F678F1CE", 10.0)
    mock_client = MagicMock()
    mock_client.request_service_quota_increase.side_effect = error
    manager = _build_manager_with_mock_client(mock_client)

    with patch("aws_bench.resource_management.quota_manager.time.sleep") as mock_sleep:
        with pytest.raises(DeploymentError, match="Exhausted retries"):
            manager._request_increase(mock_client, request, _TEST_LOG, max_retries=3)

    assert mock_client.request_service_quota_increase.call_count == 4
    assert mock_sleep.call_count == 3


# ---------------------------------------------------------------------------
# _diagnose_unmet_quota — AWS change-history status -> QuotaStatus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "aws_status, expected_status, reason_substr",
    [
        ("PENDING", QuotaStatus.ALREADY_PENDING, "pending"),
        ("CASE_OPENED", QuotaStatus.ALREADY_PENDING, "pending"),
        ("APPROVED", QuotaStatus.APPROVED, "propagat"),
        ("CASE_CLOSED", QuotaStatus.CASE_CLOSED, "propagat"),
        ("DENIED", QuotaStatus.DENIED, "denied"),
        ("NOT_APPROVED", QuotaStatus.DENIED, "denied"),
        ("INVALID_REQUEST", QuotaStatus.FAILED, "INVALID_REQUEST"),
    ],
)
def test_diagnose_unmet_quota_maps_aws_status(aws_status, expected_status, reason_substr):
    """An unmet quota's AWS change-history status maps to the right QuotaStatus.

    CASE_CLOSED and APPROVED are not failures: the increase is granted but the
    effective value may not have propagated yet. Only genuinely terminal-bad
    statuses (INVALID_REQUEST, unknown) become FAILED.
    """
    client = MagicMock()
    client.list_requested_service_quota_change_history_by_quota.return_value = {
        "RequestedQuotas": [{"Status": aws_status}],
    }

    status, reason = QuotaManager._diagnose_unmet_quota(client, "ec2", "L-1216C47A")

    assert status == expected_status
    assert reason_substr.lower() in reason.lower()


def test_diagnose_unmet_quota_no_history_is_failure():
    """No request in history means nothing was submitted — a real failure."""
    client = MagicMock()
    client.list_requested_service_quota_change_history_by_quota.return_value = {
        "RequestedQuotas": [],
    }

    status, reason = QuotaManager._diagnose_unmet_quota(client, "ec2", "L-1216C47A")

    assert status == QuotaStatus.FAILED
    assert "no increase requested" in reason


def test_diagnose_unmet_quota_history_error_is_failure():
    """A ClientError reading history is reported as FAILED, not crashed."""
    client = MagicMock()
    client.list_requested_service_quota_change_history_by_quota.side_effect = _make_client_error(
        "AccessDeniedException"
    )

    status, reason = QuotaManager._diagnose_unmet_quota(client, "ec2", "L-1216C47A")

    assert status == QuotaStatus.FAILED
    assert "unable to check" in reason


# ---------------------------------------------------------------------------
# QuotaStatus predicates — single source of truth for status classification
# ---------------------------------------------------------------------------

# (status, is_pending, is_success, is_failure). Exactly one column is True per
# row, and every QuotaStatus member appears — the predicates partition the enum.
_STATUS_CLASSIFICATION = [
    (QuotaStatus.REQUESTED, True, False, False),
    (QuotaStatus.ALREADY_PENDING, True, False, False),
    (QuotaStatus.ALREADY_MET, False, True, False),
    (QuotaStatus.APPROVED, False, True, False),
    (QuotaStatus.CASE_CLOSED, False, True, False),
    (QuotaStatus.DENIED, False, False, True),
    (QuotaStatus.FAILED, False, False, True),
]


@pytest.mark.parametrize("status, is_pending, is_success, is_failure", _STATUS_CLASSIFICATION)
def test_quota_status_predicates(status, is_pending, is_success, is_failure):
    """Each status is exactly one of pending, success, or failure."""
    assert status.is_pending is is_pending
    assert status.is_success is is_success
    assert status.is_failure is is_failure


def test_quota_status_predicates_partition_the_enum():
    """Every QuotaStatus is classified, and the classification table is complete.

    Guards against a new enum member being added without a predicate row — the
    classification would silently default and this assertion would fail.
    """
    classified = {row[0] for row in _STATUS_CLASSIFICATION}
    assert classified == set(QuotaStatus)
    for status in QuotaStatus:
        assert [status.is_pending, status.is_success, status.is_failure].count(True) == 1


@pytest.mark.parametrize(
    "aws_status, expected",
    [
        ("PENDING", QuotaStatus.ALREADY_PENDING),
        ("CASE_OPENED", QuotaStatus.ALREADY_PENDING),
        ("APPROVED", QuotaStatus.APPROVED),
        ("CASE_CLOSED", QuotaStatus.CASE_CLOSED),
        ("DENIED", QuotaStatus.DENIED),
        ("NOT_APPROVED", QuotaStatus.DENIED),
        ("INVALID_REQUEST", QuotaStatus.FAILED),
        ("", QuotaStatus.FAILED),
        ("SOMETHING_NEW", QuotaStatus.FAILED),
    ],
)
def test_quota_status_from_aws_status(aws_status, expected):
    """AWS Service Quotas wire statuses map to the right QuotaStatus.

    Unknown or empty statuses are treated as failures rather than silently
    assumed successful.
    """
    assert QuotaStatus.from_aws_status(aws_status) is expected


# ---------------------------------------------------------------------------
# QuotaConfiguration defaults (Task 6)
# ---------------------------------------------------------------------------


def test_quota_config_default_region():
    """Verify QuotaConfiguration defaults region to us-east-1."""
    config = QuotaConfiguration(increases=[QuotaIncreaseRequest("svc", "q1", 10.0)])
    assert config.region == "us-east-1"


# ---------------------------------------------------------------------------
# CredentialProvider called with correct args
# ---------------------------------------------------------------------------


def test_get_session_called_with_correct_account_and_role():
    """Verify CredentialProvider is called with correct account and role."""
    mock_client = MagicMock()
    mock_client.request_service_quota_increase.return_value = {
        "RequestedQuota": {"Id": "mock-id"},
    }

    mock_session = MagicMock()
    mock_session.client.return_value = mock_client

    mock_cred_provider = MagicMock()
    mock_cred_provider.get_session_for_account.return_value = mock_session

    manager = QuotaManager(mock_cred_provider)
    config = QuotaConfiguration(
        increases=[QuotaIncreaseRequest("vpc", "L-F678F1CE", 10.0)],
    )

    manager.request_quotas(config, account_id="123456789012", role_name="TestRole")

    mock_cred_provider.get_session_for_account.assert_called_once()
    call_args = mock_cred_provider.get_session_for_account.call_args
    assert call_args[0][0] == "123456789012"
    assert call_args[0][1] == "TestRole"
    assert isinstance(call_args[0][2], str)


# ---------------------------------------------------------------------------
# request_org_account_quota — AWS Organizations account-count quota
# ---------------------------------------------------------------------------


def _build_manager_with_mgmt_client(mock_client: MagicMock) -> tuple[QuotaManager, MagicMock]:
    """Return (manager, mgmt_session) whose management session yields mock_client."""
    mock_session = MagicMock()
    mock_session.client.return_value = mock_client

    mock_cred_provider = MagicMock()
    mock_cred_provider.get_management_session.return_value = mock_session

    return QuotaManager(mock_cred_provider), mock_session


def test_request_org_account_quota_success():
    """Submitting the org account quota returns REQUESTED with org identifiers."""
    mock_client = _build_mock_client([API_SUCCESS])
    manager, _ = _build_manager_with_mgmt_client(mock_client)

    result = manager.request_org_account_quota(15)

    assert result.status == QuotaStatus.REQUESTED
    assert result.service_code == ORG_QUOTA_SERVICE_CODE == "organizations"
    assert result.quota_code == ORG_ACCOUNT_QUOTA_CODE == "L-E619E033"
    assert result.desired_value == 15
    mock_client.request_service_quota_increase.assert_called_once_with(
        ServiceCode="organizations", QuotaCode="L-E619E033", DesiredValue=15
    )


def test_request_org_account_quota_uses_management_session_in_us_east_1():
    """The request uses the management session (no role assumption) in us-east-1."""
    mock_client = _build_mock_client([API_SUCCESS])
    manager, mock_session = _build_manager_with_mgmt_client(mock_client)

    manager.request_org_account_quota(15)

    cred = cast(MagicMock, manager._credential_provider)
    cred.get_management_session.assert_called_once()
    cred.get_session_for_account.assert_not_called()
    mock_session.client.assert_called_once_with("service-quotas", region_name=ORG_QUOTA_REGION)
    assert ORG_QUOTA_REGION == "us-east-1"


@pytest.mark.parametrize(
    "scenario, expected_status",
    [
        (API_ILLEGAL_ARG, QuotaStatus.ALREADY_MET),
        (API_ALREADY_EXISTS, QuotaStatus.ALREADY_PENDING),
    ],
)
def test_request_org_account_quota_idempotent_statuses(scenario, expected_status):
    """Already-met / already-pending API outcomes map to their statuses."""
    mock_client = _build_mock_client([scenario])
    manager, _ = _build_manager_with_mgmt_client(mock_client)

    result = manager.request_org_account_quota(15)

    assert result.status == expected_status


def test_request_org_account_quota_unexpected_error_returns_failed():
    """An unexpected API error is returned as FAILED, not raised (non-fatal for init)."""
    mock_client = MagicMock()
    mock_client.request_service_quota_increase.side_effect = _make_client_error("AccessDenied")
    manager, _ = _build_manager_with_mgmt_client(mock_client)

    result = manager.request_org_account_quota(15)

    assert result.status == QuotaStatus.FAILED
    assert result.error_message


# ---------------------------------------------------------------------------
# request_org_account_quota_if_absent — submit only when none pending
# ---------------------------------------------------------------------------


def test_request_org_account_quota_if_absent_skips_when_pending():
    """An already-pending request is detected and NOT resubmitted."""
    mock_client = MagicMock()
    mock_client.list_requested_service_quota_change_history_by_quota.return_value = {
        "RequestedQuotas": [{"Status": "CASE_OPENED"}],
    }
    manager, _ = _build_manager_with_mgmt_client(mock_client)

    result = manager.request_org_account_quota_if_absent(15)

    assert result.status == QuotaStatus.ALREADY_PENDING
    mock_client.request_service_quota_increase.assert_not_called()


def test_request_org_account_quota_if_absent_submits_when_none_pending():
    """With no pending request in history, a new increase is submitted."""
    mock_client = MagicMock()
    mock_client.list_requested_service_quota_change_history_by_quota.return_value = {
        "RequestedQuotas": [],
    }
    mock_client.request_service_quota_increase.return_value = {"RequestedQuota": {"Id": "x"}}
    manager, _ = _build_manager_with_mgmt_client(mock_client)

    result = manager.request_org_account_quota_if_absent(15)

    assert result.status == QuotaStatus.REQUESTED
    mock_client.request_service_quota_increase.assert_called_once_with(
        ServiceCode=ORG_QUOTA_SERVICE_CODE, QuotaCode=ORG_ACCOUNT_QUOTA_CODE, DesiredValue=15
    )


# ---------------------------------------------------------------------------
# diagnose_org_account_quota — map the org account-quota request status
# ---------------------------------------------------------------------------


def test_diagnose_org_account_quota_maps_history_status():
    """Maps the latest request-history status to a QuotaStatus + reason."""
    mock_client = MagicMock()
    mock_client.list_requested_service_quota_change_history_by_quota.return_value = {
        "RequestedQuotas": [{"Status": "DENIED"}],
    }
    manager, mock_session = _build_manager_with_mgmt_client(mock_client)

    status, reason = manager.diagnose_org_account_quota()

    assert status == QuotaStatus.DENIED
    assert "denied" in reason.lower()
    cast(MagicMock, manager._credential_provider).get_management_session.assert_called_once()
    mock_session.client.assert_called_once_with("service-quotas", region_name=ORG_QUOTA_REGION)
    mock_client.list_requested_service_quota_change_history_by_quota.assert_called_once_with(
        ServiceCode=ORG_QUOTA_SERVICE_CODE, QuotaCode=ORG_ACCOUNT_QUOTA_CODE
    )


def test_request_increase_does_not_retry_uncertain_network_failure() -> None:
    client = MagicMock()
    error = EndpointConnectionError(endpoint_url="https://servicequotas.example.invalid")
    client.request_service_quota_increase.side_effect = error
    manager = _build_manager_with_mock_client(client)
    with patch("aws_bench.resource_management.quota_manager.time.sleep") as sleep:
        with pytest.raises(EndpointConnectionError) as caught:
            manager._request_increase(
                client, QuotaIncreaseRequest("vpc", "L-F678F1CE", 10.0), _TEST_LOG
            )
    assert caught.value is error
    client.request_service_quota_increase.assert_called_once()
    sleep.assert_not_called()
