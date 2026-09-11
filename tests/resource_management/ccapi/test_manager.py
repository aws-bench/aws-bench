"""Tests for aws_bench.resource_management.ccapi.manager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from aws_bench.resource_management.ccapi.exceptions import (
    ResourceExistenceCheckError,
    ResourceExistenceHandlerFailureError,
    ResourceExistenceThrottledError,
    ResourceExistenceTransientError,
    ResourceExistenceUnsupportedError,
)
from aws_bench.resource_management.ccapi.manager import CloudControlManager
from aws_bench.resource_management.ccapi.models import (
    CCAPI_CLIENT_CONFIG,
    EXISTENCE_CHECK_CLIENT_CONFIG,
    Resource,
)

# -- __init__ --


def test_creates_cloudcontrol_client_with_retry_config():
    session = MagicMock()
    session.region_name = "us-east-1"
    CloudControlManager(session)
    # Two cloudcontrol clients: the retrying scan/list/delete client, and a dedicated
    # no-retry client for the fail-closed existence check (so a broken-handler type is not
    # retried to exhaustion on every check).
    session.client.assert_any_call(
        "cloudcontrol", region_name="us-east-1", config=CCAPI_CLIENT_CONFIG
    )
    session.client.assert_any_call(
        "cloudcontrol", region_name="us-east-1", config=EXISTENCE_CHECK_CLIENT_CONFIG
    )


def test_existence_check_client_config_disables_retries():
    """The existence-check client is built with retries disabled (no adaptive burn)."""
    assert EXISTENCE_CHECK_CLIENT_CONFIG.retries == {"max_attempts": 0}  # type: ignore[attr-defined]
    # The scan/list/delete path still retries.
    assert CCAPI_CLIENT_CONFIG.retries["max_attempts"] == 8  # type: ignore[attr-defined]


def test_raises_on_client_failure():
    session = MagicMock()
    session.client.side_effect = Exception("no endpoint")
    with pytest.raises(Exception, match="no endpoint"):
        CloudControlManager(session)


def test_passes_region_name_to_type_registry():
    """TypeRegistry should receive region_name when CloudControlManager has an override."""
    session = MagicMock()
    session.region_name = "us-west-1"

    with patch("aws_bench.resource_management.ccapi.manager.TypeRegistry") as MockTypeRegistry:
        CloudControlManager(session, region_name="eu-west-1")

        # TypeRegistry should be called with region_name matching manager's override
        MockTypeRegistry.assert_called_once()
        call_args = MockTypeRegistry.call_args
        assert call_args[0][0] == session
        assert call_args[1].get("region_name") == "eu-west-1"


# -- resource_exists --


def test_resource_exists_returns_true_when_found():
    session = MagicMock()
    ccm = CloudControlManager(session)
    ccm._client.get_resource.return_value = {"ResourceDescription": {}}

    result = ccm.resource_exists(Resource("AWS::S3::Bucket", "my-bucket"))

    assert result is True
    ccm._client.get_resource.assert_called_once_with(
        TypeName="AWS::S3::Bucket",
        Identifier="my-bucket",
    )


def test_resource_exists_returns_false_on_not_found():
    session = MagicMock()
    ccm = CloudControlManager(session)
    error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "not found"}}
    exc = ClientError(error_response, "GetResource")
    ccm._client.get_resource.side_effect = exc
    ccm._client.exceptions.ResourceNotFoundException = type(exc)

    assert ccm.resource_exists(Resource("AWS::S3::Bucket", "gone-bucket")) is False


def test_resource_exists_raises_custom_exception_on_non_not_found_errors():
    session = MagicMock()
    ccm = CloudControlManager(session)
    # Mirrors a Cloud Control handler InternalFailure on GetResource. This maps to the
    # dedicated handler-failure subclass — which is a ResourceExistenceCheckError but NOT the
    # unsupported subclass, so the deleter still attempts the delete rather than skipping and
    # leaking a live resource. It is raised on the FIRST call: the no-retry existence client
    # means no adaptive retry burn (~18s) on the broken-handler types.
    error_response = {
        "Error": {"Code": "HandlerInternalFailureException", "Message": "Internal error occurred"}
    }
    exc = ClientError(error_response, "GetResource")
    ccm._existence_client.get_resource.side_effect = exc
    ccm._existence_client.exceptions.ResourceNotFoundException = type("RNF", (Exception,), {})

    with pytest.raises(ResourceExistenceCheckError) as exc_info:
        ccm.resource_exists(Resource("AWS::Kinesis::Stream", "bench-stream-193512"))
    assert isinstance(exc_info.value, ResourceExistenceHandlerFailureError)
    assert not isinstance(exc_info.value, ResourceExistenceUnsupportedError)
    # No retry burn: the doomed check is raised on the first (and only) attempt.
    assert ccm._existence_client.get_resource.call_count == 1


@pytest.mark.parametrize(
    "throttle_code",
    ["ThrottlingException", "Throttling", "TooManyRequestsException", "RequestLimitExceeded"],
)
def test_resource_exists_raises_throttled_error_on_throttle_codes(throttle_code):
    """A throttle raises the throttle subclass, not the generic unsupported error.

    Otherwise verification buckets it as SKIPPED and silently drops the resource. A throttle is
    RECOVERABLE, so the existence check retries it up to EXISTENCE_CHECK_MAX_ATTEMPTS before
    re-raising (keeping the reset/verification callers resilient to a momentary throttle).
    """
    from aws_bench.resource_management.ccapi.models import EXISTENCE_CHECK_MAX_ATTEMPTS

    session = MagicMock()
    ccm = CloudControlManager(session)
    exc = ClientError({"Error": {"Code": throttle_code, "Message": "slow down"}}, "GetResource")
    ccm._existence_client.get_resource.side_effect = exc
    ccm._existence_client.exceptions.ResourceNotFoundException = type("RNF", (Exception,), {})

    with (
        patch("aws_bench.resource_management.ccapi.manager._existence_check_sleep"),
        pytest.raises(ResourceExistenceThrottledError),
    ):
        ccm.resource_exists(Resource("AWS::S3::Bucket", "my-bucket"))
    # Recoverable: retried the full attempt budget before giving up.
    assert ccm._existence_client.get_resource.call_count == EXISTENCE_CHECK_MAX_ATTEMPTS


@pytest.mark.parametrize(
    "transient",
    [
        ClientError(
            {
                "Error": {"Code": "ServiceInternalError", "Message": "boom"},
                "ResponseMetadata": {"HTTPStatusCode": 500},
            },
            "GetResource",
        ),
        EndpointConnectionError(endpoint_url="https://cloudcontrol.us-east-1.amazonaws.com"),
    ],
)
def test_resource_exists_retries_transient_faults(transient):
    """A transient fault (5xx or connection error) is RECOVERABLE and retried, then re-raised."""
    from aws_bench.resource_management.ccapi.models import EXISTENCE_CHECK_MAX_ATTEMPTS

    session = MagicMock()
    ccm = CloudControlManager(session)
    ccm._existence_client.get_resource.side_effect = transient
    ccm._existence_client.exceptions.ResourceNotFoundException = type("RNF", (Exception,), {})

    with (
        patch("aws_bench.resource_management.ccapi.manager._existence_check_sleep"),
        pytest.raises(ResourceExistenceTransientError),
    ):
        ccm.resource_exists(Resource("AWS::S3::Bucket", "my-bucket"))
    assert ccm._existence_client.get_resource.call_count == EXISTENCE_CHECK_MAX_ATTEMPTS


def test_resource_exists_handler_failure_is_not_retried():
    """A broken-handler fault is NON-recoverable: raised on the first attempt (no retry burn)."""
    session = MagicMock()
    ccm = CloudControlManager(session)
    exc = ClientError(
        {"Error": {"Code": "HandlerInternalFailureException", "Message": "internal"}},
        "GetResource",
    )
    ccm._existence_client.get_resource.side_effect = exc
    ccm._existence_client.exceptions.ResourceNotFoundException = type("RNF", (Exception,), {})

    with (
        patch("aws_bench.resource_management.ccapi.manager._existence_check_sleep") as mock_sleep,
        pytest.raises(ResourceExistenceHandlerFailureError),
    ):
        ccm.resource_exists(Resource("AWS::ControlTower::EnabledBaseline", "arn:...:baseline/x"))
    assert ccm._existence_client.get_resource.call_count == 1
    mock_sleep.assert_not_called()


def test_resource_exists_bare_internal_failure_is_retried():
    """Bare ``InternalFailure`` is retried, NOT treated as a broken handler.

    It is AWS's generic code for a transient server-side 500 as well as the
    handler fault, so only the handler-specific code may skip retries. A
    transient 500 that fails closed with zero retries on the reset survivor
    check reports a spurious survivor and escalates to a reset failure.
    """
    from aws_bench.resource_management.ccapi.models import EXISTENCE_CHECK_MAX_ATTEMPTS

    session = MagicMock()
    ccm = CloudControlManager(session)
    exc = ClientError(
        {
            "Error": {"Code": "InternalFailure", "Message": "transient blip"},
            "ResponseMetadata": {"HTTPStatusCode": 500},
        },
        "GetResource",
    )
    ccm._existence_client.get_resource.side_effect = exc
    ccm._existence_client.exceptions.ResourceNotFoundException = type("RNF", (Exception,), {})

    with (
        patch("aws_bench.resource_management.ccapi.manager._existence_check_sleep"),
        pytest.raises(ResourceExistenceTransientError),
    ):
        ccm.resource_exists(Resource("AWS::S3::Bucket", "my-bucket"))
    assert ccm._existence_client.get_resource.call_count == EXISTENCE_CHECK_MAX_ATTEMPTS


def test_resource_exists_unknown_error_defaults_to_recoverable():
    """An unrecognized error code gets the bounded retry, then still fails closed.

    The asymmetry favors retrying: a permanent unknown error wastes seconds,
    while refusing to retry a transient one (e.g. a throttle variant missing
    from THROTTLE_ERROR_CODES) fails the check closed on the first blip. The
    raised error is still a ResourceExistenceCheckError, so every caller keeps
    the resource exactly as before.
    """
    from aws_bench.resource_management.ccapi.models import EXISTENCE_CHECK_MAX_ATTEMPTS

    session = MagicMock()
    ccm = CloudControlManager(session)
    exc = ClientError(
        {"Error": {"Code": "RequestThrottledException", "Message": "slow down"}},
        "GetResource",
    )
    ccm._existence_client.get_resource.side_effect = exc
    ccm._existence_client.exceptions.ResourceNotFoundException = type("RNF", (Exception,), {})

    with (
        patch("aws_bench.resource_management.ccapi.manager._existence_check_sleep"),
        pytest.raises(ResourceExistenceCheckError),
    ):
        ccm.resource_exists(Resource("AWS::S3::Bucket", "my-bucket"))
    assert ccm._existence_client.get_resource.call_count == EXISTENCE_CHECK_MAX_ATTEMPTS


def test_throttled_error_is_a_resource_existence_check_error():
    """The throttle subclass stays catchable as ResourceExistenceCheckError.

    Existing ``except ResourceExistenceCheckError`` handlers must keep working.
    """
    assert issubclass(ResourceExistenceThrottledError, ResourceExistenceCheckError)


def test_resource_exists_raises_custom_exception_on_unsupported_action():
    session = MagicMock()
    ccm = CloudControlManager(session)
    exc = ClientError(
        {"Error": {"Code": "UnsupportedActionException", "Message": "unsupported"}},
        "GetResource",
    )
    ccm._client.get_resource.side_effect = exc
    ccm._client.exceptions.ResourceNotFoundException = type("RNF", (Exception,), {})
    with pytest.raises(ResourceExistenceUnsupportedError):
        ccm.resource_exists(Resource("AWS::Unsupported::Type", "x"))


def test_resource_exists_returns_false_on_general_service_does_not_exist():
    session = MagicMock()
    ccm = CloudControlManager(session)
    exc = ClientError(
        {"Error": {"Code": "GeneralServiceException", "Message": "Resource does not exist"}},
        "GetResource",
    )
    ccm._client.get_resource.side_effect = exc
    ccm._client.exceptions.ResourceNotFoundException = type("RNF", (Exception,), {})
    assert ccm.resource_exists(Resource("AWS::S3::Bucket", "gone")) is False


def test_resource_exists_returns_false_on_general_service_not_found():
    session = MagicMock()
    ccm = CloudControlManager(session)
    exc = ClientError(
        {
            "Error": {
                "Code": "GeneralServiceException",
                "Message": "Resource could not be found",
            }
        },
        "GetResource",
    )
    ccm._client.get_resource.side_effect = exc
    ccm._client.exceptions.ResourceNotFoundException = type("RNF", (Exception,), {})
    assert ccm.resource_exists(Resource("AWS::EC2::Route", "rtb-123|0.0.0.0/0")) is False


# -- delegation --


@patch("aws_bench.resource_management.ccapi.manager.Scanner")
def test_get_scannable_types_passes_session(MockScanner):
    session = MagicMock()
    mock_scanner_instance = MagicMock()
    MockScanner.return_value = mock_scanner_instance
    mock_scanner_instance.get_scannable_types.return_value = ["AWS::S3::Bucket"]

    ccm = CloudControlManager(session)
    result = ccm.get_scannable_types()

    assert result == ["AWS::S3::Bucket"]
    mock_scanner_instance.get_scannable_types.assert_called_once()


@patch("aws_bench.resource_management.ccapi.manager.Scanner")
def test_scan_resources_passes_types(MockScanner):
    session = MagicMock()
    mock_scanner_instance = MagicMock()
    MockScanner.return_value = mock_scanner_instance
    from aws_bench.resource_management.ccapi.models import ScanResult

    expected = ScanResult(detected={"AWS::S3::Bucket": [{"Identifier": "b1"}]}, failed={})
    mock_scanner_instance.scan_resources.return_value = expected

    ccm = CloudControlManager(session)
    result = ccm.scan_resources(["AWS::S3::Bucket"])

    assert result == expected
    mock_scanner_instance.scan_resources.assert_called_once_with(["AWS::S3::Bucket"])


@patch("aws_bench.resource_management.ccapi.manager.Deleter")
def test_delete_resources_delegates_to_deleter(MockDeleter):
    session = MagicMock()
    mock_deleter_instance = MagicMock()
    MockDeleter.return_value = mock_deleter_instance
    mock_deleter_instance.delete_resources.return_value = {}
    resources = [Resource("AWS::S3::Bucket", "b")]

    ccm = CloudControlManager(session)
    ccm.delete_resources(resources)

    mock_deleter_instance.delete_resources.assert_called_once_with(resources)


@patch("aws_bench.resource_management.ccapi.manager.TypeRegistry")
def test_generate_skip_types_delegates(MockTypeRegistry):
    session = MagicMock()
    MockTypeRegistry.return_value.generate_skip_types.return_value = {"AWS::Bad::Type"}
    ccm = CloudControlManager(session)
    result = ccm.generate_skip_types()
    assert result == {"AWS::Bad::Type"}
