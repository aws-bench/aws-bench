"""Tests for the Kinesis Data Analytics cleanup handlers.

Neither ``kinesisanalyticsv2.delete_application`` (unimplemented in moto) nor the
v1 ``kinesisanalytics`` service (no moto backend) can be exercised end-to-end,
so these tests drive mocked boto3 clients — the same approach the IoT handler
tests use for their failure paths. They assert the handler reads the
``CreateTimestamp`` via ``DescribeApplication`` and passes it to
``DeleteApplication``, and that not-found / error paths behave correctly.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock

from botocore.exceptions import ClientError, EndpointConnectionError

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import CUSTOM_DELETION_REGISTRY
from aws_bench.resource_management.cleanup.handlers.kinesisanalytics import _delete_v1, _delete_v2
from aws_bench.resource_management.cleanup.models import HandlerStatus

_REGION = "us-east-1"
_ACCOUNT = "123456789012"
_APP_NAME = "bench-flink-studio"
_V2_TYPE = "AWS::KinesisAnalyticsV2::Application"
_V1_TYPE = "AWS::KinesisAnalytics::Application"
_V1_ARN = f"arn:aws:kinesisanalytics:{_REGION}:{_ACCOUNT}:application/{_APP_NAME}"
_CREATE_TS = datetime.datetime(2026, 9, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)


def _resource(resource_type: str, identifier: str) -> Resource:
    return Resource(type=resource_type, identifier=identifier)


def _mock_session(client: MagicMock) -> MagicMock:
    """A session whose ``client(service)`` returns the given mock (via build_client)."""
    session = MagicMock()
    session.client.return_value = client
    return session


def _app_client(create_ts: datetime.datetime = _CREATE_TS) -> MagicMock:
    """A mock KDA client that describes an existing application."""
    client = MagicMock()
    client.describe_application.return_value = {"ApplicationDetail": {"CreateTimestamp": create_ts}}
    return client


def _client_error(code: str, op: str) -> ClientError:
    return ClientError({"Error": {"Code": code}}, op)


# -- registration --


def test_handlers_registered_for_both_kda_types():
    """Both application types must be registered so the scan does not fall through to CCAPI.

    CCAPI cannot delete the v1 ``AWS::KinesisAnalytics::Application`` type; without
    this registration an agent-created Flink Studio application leaks and fails
    reset (the bug these handlers fix).
    """
    import aws_bench.resource_management.cleanup.handlers  # noqa: F401

    assert _V2_TYPE in CUSTOM_DELETION_REGISTRY
    assert _V1_TYPE in CUSTOM_DELETION_REGISTRY


# -- happy path: describe then delete with CreateTimestamp --


def test_v2_describes_then_deletes_with_create_timestamp():
    """The v2 handler reads CreateTimestamp and passes it to delete_application."""
    client = _app_client()

    result = _delete_v2(_resource(_V2_TYPE, _APP_NAME), _mock_session(client))

    assert result.status == HandlerStatus.SUCCESS
    client.describe_application.assert_called_once_with(ApplicationName=_APP_NAME)
    client.delete_application.assert_called_once_with(
        ApplicationName=_APP_NAME, CreateTimestamp=_CREATE_TS
    )


def test_v1_describes_then_deletes_with_create_timestamp():
    """The v1 handler reads CreateTimestamp and passes it to delete_application."""
    client = _app_client()

    result = _delete_v1(_resource(_V1_TYPE, _V1_ARN), _mock_session(client))

    assert result.status == HandlerStatus.SUCCESS
    client.delete_application.assert_called_once_with(
        ApplicationName=_APP_NAME, CreateTimestamp=_CREATE_TS
    )


def test_v1_derives_application_name_from_arn():
    """The v1 identifier is an ARN; the handler deletes by the bare application name."""
    client = _app_client()

    _delete_v1(_resource(_V1_TYPE, _V1_ARN), _mock_session(client))

    client.describe_application.assert_called_once_with(ApplicationName=_APP_NAME)


def test_v2_accepts_bare_name_identifier():
    """The v2 identifier is a bare application name and is used unchanged."""
    client = _app_client()

    _delete_v2(_resource(_V2_TYPE, _APP_NAME), _mock_session(client))

    client.describe_application.assert_called_once_with(ApplicationName=_APP_NAME)


# -- idempotency --


def test_v2_already_gone_on_describe_is_idempotent_success():
    """A v2 application that no longer exists yields SUCCESS without a delete call."""
    client = _app_client()
    client.describe_application.side_effect = _client_error(
        "ResourceNotFoundException", "DescribeApplication"
    )

    result = _delete_v2(_resource(_V2_TYPE, _APP_NAME), _mock_session(client))

    assert result.status == HandlerStatus.SUCCESS
    client.delete_application.assert_not_called()


def test_v1_already_gone_on_delete_is_idempotent_success():
    """A v1 application that vanishes between describe and delete yields SUCCESS."""
    client = _app_client()
    client.delete_application.side_effect = _client_error(
        "ResourceNotFoundException", "DeleteApplication"
    )

    result = _delete_v1(_resource(_V1_TYPE, _V1_ARN), _mock_session(client))

    assert result.status == HandlerStatus.SUCCESS


# -- failure paths --


def test_delete_reports_failure_on_client_error():
    """A non-not-found ClientError from delete_application maps to FAILED, not SUCCESS."""
    client = _app_client()
    client.delete_application.side_effect = _client_error(
        "InvalidArgumentException", "DeleteApplication"
    )

    result = _delete_v2(_resource(_V2_TYPE, _APP_NAME), _mock_session(client))

    assert result.status == HandlerStatus.FAILED


def test_delete_reports_failure_on_botocore_error():
    """A connection-level BotoCoreError maps to FAILED."""
    client = _app_client()
    client.delete_application.side_effect = EndpointConnectionError(
        endpoint_url="https://kinesisanalytics"
    )

    result = _delete_v1(_resource(_V1_TYPE, _V1_ARN), _mock_session(client))

    assert result.status == HandlerStatus.FAILED


# -- Bug 1: v1 entry for a v2-SDK-managed Studio application --

_STUDIO_NAME = "environment-2h384hj-flink-studio"
_STUDIO_V2_ARN = f"arn:aws:kinesisanalytics:{_REGION}:{_ACCOUNT}:application/{_STUDIO_NAME}"


def _v2_managed_error(op: str) -> ClientError:
    return ClientError(
        {
            "Error": {
                "Code": "UnsupportedOperationException",
                "Message": (
                    f"{_STUDIO_NAME} was created/updated by kinesisanalyticsv2 SDK. "
                    "Please use kinesisanalyticsv2 SDK to make changes."
                ),
            }
        },
        op,
    )


def test_v1_treats_v2_managed_application_as_handled():
    """The v1 entry for a v2-SDK-managed Studio app is deduped, not failed."""
    client = _app_client()
    client.describe_application.side_effect = _v2_managed_error("DescribeApplication")

    result = _delete_v1(_resource(_V1_TYPE, _STUDIO_V2_ARN), _mock_session(client))

    assert result.status == HandlerStatus.SUCCESS
    client.delete_application.assert_not_called()


def test_v1_unsupported_operation_without_v2_hint_still_fails():
    """A generic UnsupportedOperationException (not v2-managed) remains a failure."""
    client = _app_client()
    client.describe_application.side_effect = ClientError(
        {"Error": {"Code": "UnsupportedOperationException", "Message": "something else"}},
        "DescribeApplication",
    )

    result = _delete_v1(_resource(_V1_TYPE, _V1_ARN), _mock_session(client))

    assert result.status == HandlerStatus.FAILED


def test_v2_managed_dedupe_does_not_apply_to_v2_handler():
    """The v2 handler does not swallow UnsupportedOperationException as 'handled'."""
    client = _app_client()
    client.describe_application.side_effect = _v2_managed_error("DescribeApplication")

    result = _delete_v2(_resource(_V2_TYPE, _STUDIO_NAME), _mock_session(client))

    assert result.status == HandlerStatus.FAILED


# -- Bug 2: v2 handler waits for the service-managed Studio stack to auto-remove --


def _multi_session(kda_client: MagicMock, cfn_client: MagicMock) -> MagicMock:
    """A session whose client(service) returns the KDA or CloudFormation mock."""
    session = MagicMock()
    clients = {"kinesisanalyticsv2": kda_client, "cloudformation": cfn_client}
    session.client.side_effect = lambda service, *a, **k: clients[service]
    return session


def _requested_services(session: MagicMock) -> list[str]:
    return [call.args[0] for call in session.client.call_args_list]


def test_v2_waits_for_studio_stack_removal_without_deleting_it():
    """After deleting a Studio app, the handler WAITS for stack removal, never DeleteStack."""
    kda = _app_client()
    cfn = MagicMock()
    session = _multi_session(kda, cfn)

    result = _delete_v2(_resource(_V2_TYPE, _STUDIO_NAME), session)

    assert result.status == HandlerStatus.SUCCESS
    kda.delete_application.assert_called_once_with(
        ApplicationName=_STUDIO_NAME, CreateTimestamp=_CREATE_TS
    )
    cfn.get_waiter.assert_called_once_with("stack_delete_complete")
    cfn.get_waiter.return_value.wait.assert_called_once()
    assert cfn.get_waiter.return_value.wait.call_args.kwargs["StackName"] == _STUDIO_NAME
    cfn.delete_stack.assert_not_called()  # AWS auto-removes it; never delete directly


def test_v2_non_studio_application_does_not_touch_cloudformation():
    """A regular (non-Studio) v2 application triggers no stack wait."""
    kda = _app_client()
    cfn = MagicMock()
    session = _multi_session(kda, cfn)

    result = _delete_v2(_resource(_V2_TYPE, "my-regular-app"), session)

    assert result.status == HandlerStatus.SUCCESS
    assert "cloudformation" not in _requested_services(session)
    cfn.get_waiter.assert_not_called()


def test_v2_stack_wait_failure_does_not_fail_the_application_delete():
    """A stack-removal wait timeout is best-effort: the app delete still SUCCEEDS."""
    from botocore.exceptions import WaiterError

    kda = _app_client()
    cfn = MagicMock()
    cfn.get_waiter.return_value.wait.side_effect = WaiterError(
        name="StackDeleteComplete", reason="Max attempts exceeded", last_response={}
    )
    session = _multi_session(kda, cfn)

    result = _delete_v2(_resource(_V2_TYPE, _STUDIO_NAME), session)

    assert result.status == HandlerStatus.SUCCESS
