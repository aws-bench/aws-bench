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
from unittest.mock import MagicMock, call, patch

from botocore.exceptions import ClientError, EndpointConnectionError, WaiterError

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
    """A mock KDA client that describes an existing application.

    Also configured so the post-delete teardown poll (``collect`` over
    ``list_applications``) sees the application already absent, so the wait
    returns immediately for the plain happy-path tests.
    """
    client = MagicMock()
    client.describe_application.return_value = {"ApplicationDetail": {"CreateTimestamp": create_ts}}
    client.can_paginate.return_value = False
    client.list_applications.return_value = {"ApplicationSummaries": []}
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


# -- Bug 2: the deleted KDA app lingers in the scanner LISTING; poll it, not describe --


def _requested_services(session: MagicMock) -> list[str]:
    return [call.args[0] for call in session.client.call_args_list]


_SLEEP = "aws_bench.resource_management.cleanup.handlers.kinesisanalytics.time.sleep"


def _detail() -> dict:
    return {"ApplicationDetail": {"CreateTimestamp": _CREATE_TS}}


def _stack_gone_error() -> ClientError:
    return ClientError(
        {"Error": {"Code": "ValidationError", "Message": f"Stack {_STUDIO_NAME} does not exist"}},
        "DescribeStacks",
    )


def _list_page(app_ids: list[dict]) -> dict:
    return {"ApplicationSummaries": app_ids}


def _v2_list_client(name: str, present_polls: int) -> MagicMock:
    """v2 client: supports the delete path and lists ``name`` present then absent."""
    client = MagicMock()
    client.describe_application.return_value = _detail()
    client.can_paginate.return_value = False
    present = _list_page([{"ApplicationName": name}])
    client.list_applications.side_effect = [present] * present_polls + [_list_page([])] * 50
    return client


def _v1_list_client(name: str, present_polls: int) -> MagicMock:
    """v1 client: supports the delete path and lists ``name`` (by ARN) present then absent."""
    client = MagicMock()
    client.describe_application.return_value = _detail()
    client.can_paginate.return_value = False
    arn = f"arn:aws:kinesisanalytics:{_REGION}:{_ACCOUNT}:application/{name}"
    present = _list_page([{"ApplicationARN": arn}])
    client.list_applications.side_effect = [present] * present_polls + [_list_page([])] * 50
    return client


def _empty_list_client() -> MagicMock:
    """A KDA client that lists nothing (the app is absent from this service)."""
    client = MagicMock()
    client.can_paginate.return_value = False
    client.list_applications.return_value = _list_page([])
    return client


def _multi3_session(v2: MagicMock, v1: MagicMock, cfn: MagicMock) -> MagicMock:
    session = MagicMock()
    clients = {"kinesisanalyticsv2": v2, "kinesisanalytics": v1, "cloudformation": cfn}
    session.client.side_effect = lambda service, *a, **k: clients[service]
    return session


@patch(_SLEEP)
def test_v2_stackless_waits_until_app_absent_from_both_listings(_sleep: MagicMock):
    """Stack-less Studio app: poll the v2 AND v1 listings until the app clears."""
    name = "studio-notebook-2h384hj"  # not an environment-*-flink-studio name
    v2 = _v2_list_client(name, present_polls=2)
    v1 = _v1_list_client(name, present_polls=2)
    cfn = MagicMock()
    session = _multi3_session(v2, v1, cfn)

    result = _delete_v2(_resource(_V2_TYPE, name), session)

    assert result.status == HandlerStatus.SUCCESS
    v2.delete_application.assert_called_once_with(ApplicationName=name, CreateTimestamp=_CREATE_TS)
    # Kept polling the LISTING (not describe) until absent: 2 present + 1 absent.
    assert v2.list_applications.call_count == 3
    assert v1.list_applications.call_count == 3
    # Stack-less variant: no stack delete, no stack polling.
    cfn.describe_stacks.assert_not_called()
    cfn.delete_stack.assert_not_called()


@patch(_SLEEP)
def test_teardown_polls_the_listing_not_describe(_sleep: MagicMock):
    """Regression: the wait consults list_applications each poll, NEVER DescribeApplication.

    Guards against the "2 ms return" seen when the wait was gated on a managed
    stack and/or polled DescribeApplication (which reports the app gone at once).
    Here the listing returns the app for several polls; the handler must keep
    waiting and only return once the LISTING is clear.
    """
    name = "studio-notebook-2h384hj"  # stack-less variant — nothing to gate on
    v2 = _v2_list_client(name, present_polls=3)
    v1 = _empty_list_client()
    session = _multi3_session(v2, v1, MagicMock())

    _delete_v2(_resource(_V2_TYPE, name), session)

    # describe_application is called exactly once — for the delete, not for polling.
    assert v2.describe_application.call_count == 1
    # the listing was polled until the app cleared: 3 present + 1 absent.
    assert v2.list_applications.call_count == 4


@patch(_SLEEP)
def test_v2_stackbacked_waits_for_listings_and_stack_never_deletestack(_sleep: MagicMock):
    """Stack-backed Studio app: also wait for the managed stack, still no DeleteStack."""
    name = _STUDIO_NAME  # environment-2h384hj-flink-studio
    v2 = _v2_list_client(name, present_polls=2)
    v1 = _v1_list_client(name, present_polls=2)
    cfn = MagicMock()
    cfn.describe_stacks.side_effect = [
        {"Stacks": [{"StackStatus": "DELETE_IN_PROGRESS"}]},
        {"Stacks": [{"StackStatus": "DELETE_IN_PROGRESS"}]},
        _stack_gone_error(),
    ] + [_stack_gone_error()] * 50
    session = _multi3_session(v2, v1, cfn)

    result = _delete_v2(_resource(_V2_TYPE, name), session)

    assert result.status == HandlerStatus.SUCCESS
    assert cfn.describe_stacks.call_count == 3  # waited out the transient DELETE_IN_PROGRESS
    cfn.delete_stack.assert_not_called()


@patch(_SLEEP)
def test_v1_genuine_delete_also_waits_for_listing(_sleep: MagicMock):
    """A real v1-only delete waits for the app to clear the listing too."""
    name = "legacy-analytics-app"
    v1 = _v1_list_client(name, present_polls=1)
    v2 = _empty_list_client()  # no v2 entry for a v1-only app
    cfn = MagicMock()
    session = _multi3_session(v2, v1, cfn)

    arn = f"arn:aws:kinesisanalytics:{_REGION}:{_ACCOUNT}:application/{name}"
    result = _delete_v1(_resource(_V1_TYPE, arn), session)

    assert result.status == HandlerStatus.SUCCESS
    v1.delete_application.assert_called_once()
    assert v1.list_applications.call_count >= 2  # waited past the first (present) poll


@patch(_SLEEP)
def test_teardown_timeout_is_best_effort_success(_sleep: MagicMock):
    """If the app never clears the listing within the window, the delete still SUCCEEDS."""
    name = _STUDIO_NAME
    v2 = MagicMock()
    v2.describe_application.return_value = _detail()
    v2.can_paginate.return_value = False
    v2.list_applications.return_value = _list_page([{"ApplicationName": name}])  # always present
    v1 = _empty_list_client()
    cfn = MagicMock()
    cfn.describe_stacks.return_value = {"Stacks": [{"StackStatus": "DELETE_IN_PROGRESS"}]}
    session = _multi3_session(v2, v1, cfn)

    result = _delete_v2(_resource(_V2_TYPE, name), session)

    assert result.status == HandlerStatus.SUCCESS
    cfn.delete_stack.assert_not_called()


@patch(_SLEEP)
def test_teardown_backs_off_between_polls(sleep_mock: MagicMock):
    """The poll interval grows (backoff) and is capped, rather than a fixed delay."""
    name = "studio-notebook-2h384hj"
    v2 = MagicMock()
    v2.describe_application.return_value = _detail()
    v2.can_paginate.return_value = False
    v2.list_applications.return_value = _list_page([{"ApplicationName": name}])  # always present
    v1 = _empty_list_client()
    session = _multi3_session(v2, v1, MagicMock())

    _delete_v2(_resource(_V2_TYPE, name), session)

    delays = [call.args[0] for call in sleep_mock.call_args_list]
    assert delays[0] == 10  # initial delay
    assert delays == sorted(delays)  # non-decreasing (backoff)
    assert max(delays) <= 60  # capped
    assert delays[-1] == 60  # reaches the cap over the window


# -- Stack membership: a stack-managed app is deleted through its stack --

_STACK_ID = f"arn:aws:cloudformation:{_REGION}:{_ACCOUNT}:stack/agent-studio/1111"


def _stack_member(
    stack_id: str = _STACK_ID, resource_type: str = "AWS::KinesisAnalyticsV2::Application"
):
    return {"StackResources": [{"ResourceType": resource_type, "StackId": stack_id}]}


def _cfn_for_stack(stack_id: str = _STACK_ID) -> MagicMock:
    """A CloudFormation client that reports the application as a member of ``stack_id``."""
    cfn = MagicMock()
    cfn.describe_stack_resources.return_value = _stack_member(stack_id)
    return cfn


def _waiter_error() -> WaiterError:
    return WaiterError("StackDeleteComplete", "Waiter encountered a terminal failure state", {})


@patch(_SLEEP)
def test_unmanaged_app_is_deleted_natively_and_the_stack_is_left_alone(_sleep: MagicMock):
    """An application that is in no stack is deleted with DeleteApplication; no stack is touched."""
    name = "environment-2h384hj-studio-notebook"
    v2 = _v2_list_client(name, present_polls=0)
    cfn = MagicMock()
    cfn.describe_stack_resources.return_value = {"StackResources": []}
    session = _multi3_session(v2, _empty_list_client(), cfn)

    result = _delete_v2(_resource(_V2_TYPE, name), session)

    assert result.status == HandlerStatus.SUCCESS
    v2.delete_application.assert_called_once_with(ApplicationName=name, CreateTimestamp=_CREATE_TS)
    cfn.describe_stack_resources.assert_called_once_with(PhysicalResourceId=name)
    cfn.delete_stack.assert_not_called()


@patch(_SLEEP)
def test_stack_managed_app_is_deleted_by_deleting_its_stack(_sleep: MagicMock):
    """CloudFormation removes the logging option while the app still exists, then the app.

    Deleting the application first would leave the stack's
    ``ApplicationCloudWatchLoggingOption`` child unable to delete (its parent is gone).
    """
    name = "agent-chosen-studio"
    v2 = _v2_list_client(name, present_polls=0)
    cfn = _cfn_for_stack()
    session = _multi3_session(v2, _empty_list_client(), cfn)

    result = _delete_v2(_resource(_V2_TYPE, name), session)

    assert result.status == HandlerStatus.SUCCESS
    assert _STACK_ID in result.message
    cfn.describe_stack_resources.assert_called_once_with(PhysicalResourceId=name)
    cfn.delete_stack.assert_called_once_with(StackName=_STACK_ID)
    cfn.get_waiter.assert_called_once_with("stack_delete_complete")
    v2.delete_application.assert_not_called()  # the stack owns the application's deletion


@patch(_SLEEP)
def test_stack_managed_app_still_waits_for_the_scanner_listing(_sleep: MagicMock):
    """The cascaded delete has the same listing lag as a native one, so the wait still runs."""
    name = "agent-chosen-studio"
    v2 = _v2_list_client(name, present_polls=2)
    cfn = _cfn_for_stack()
    session = _multi3_session(v2, _empty_list_client(), cfn)

    result = _delete_v2(_resource(_V2_TYPE, name), session)

    assert result.status == HandlerStatus.SUCCESS
    assert v2.list_applications.call_count == 3  # 2 present + 1 absent


@patch(_SLEEP)
def test_stack_stuck_in_delete_failed_is_retried_retaining_the_failed_children(_sleep: MagicMock):
    """A stack already broken by an earlier app-first delete is retried with RetainResources."""
    name = "agent-chosen-studio"
    v2 = _v2_list_client(name, present_polls=0)
    cfn = _cfn_for_stack()
    waiter = MagicMock()
    waiter.wait.side_effect = [_waiter_error(), None]
    cfn.get_waiter.return_value = waiter
    # The first lookup finds the owning stack; the second lists its resources' statuses.
    cfn.describe_stack_resources.side_effect = [
        _stack_member(),
        {
            "StackResources": [
                {"LogicalResourceId": "StudioLoggingOption", "ResourceStatus": "DELETE_FAILED"},
                {"LogicalResourceId": "StudioApp", "ResourceStatus": "DELETE_COMPLETE"},
            ]
        },
    ]
    session = _multi3_session(v2, _empty_list_client(), cfn)

    result = _delete_v2(_resource(_V2_TYPE, name), session)

    assert result.status == HandlerStatus.SUCCESS
    assert cfn.delete_stack.call_args_list == [
        call(StackName=_STACK_ID),
        call(StackName=_STACK_ID, RetainResources=["StudioLoggingOption"]),
    ]


@patch(_SLEEP)
def test_stack_failure_with_nothing_to_retain_is_reported_as_failed(_sleep: MagicMock):
    """If the stack does not delete and no resource is DELETE_FAILED, there is nothing to retry."""
    name = "agent-chosen-studio"
    cfn = _cfn_for_stack()
    waiter = MagicMock()
    waiter.wait.side_effect = _waiter_error()
    cfn.get_waiter.return_value = waiter
    cfn.describe_stack_resources.side_effect = [_stack_member(), {"StackResources": []}]
    session = _multi3_session(_v2_list_client(name, present_polls=0), _empty_list_client(), cfn)

    result = _delete_v2(_resource(_V2_TYPE, name), session)

    assert result.status == HandlerStatus.FAILED
    assert _STACK_ID in result.message
    assert cfn.delete_stack.call_count == 1


def test_delete_stack_error_is_reported_as_failed():
    """A ClientError from DeleteStack maps to FAILED, not an exception."""
    name = "agent-chosen-studio"
    cfn = _cfn_for_stack()
    cfn.delete_stack.side_effect = _client_error("ValidationError", "DeleteStack")
    session = _multi3_session(_v2_list_client(name, present_polls=0), _empty_list_client(), cfn)

    result = _delete_v2(_resource(_V2_TYPE, name), session)

    assert result.status == HandlerStatus.FAILED


@patch(_SLEEP)
def test_membership_lookup_error_falls_back_to_the_native_delete(_sleep: MagicMock):
    """If stack membership cannot be determined, behave as before: delete the application."""
    name = "agent-chosen-studio"
    v2 = _v2_list_client(name, present_polls=0)
    cfn = MagicMock()
    cfn.describe_stack_resources.side_effect = _client_error(
        "AccessDenied", "DescribeStackResources"
    )
    session = _multi3_session(v2, _empty_list_client(), cfn)

    result = _delete_v2(_resource(_V2_TYPE, name), session)

    assert result.status == HandlerStatus.SUCCESS
    v2.delete_application.assert_called_once_with(ApplicationName=name, CreateTimestamp=_CREATE_TS)
    cfn.delete_stack.assert_not_called()


@patch(_SLEEP)
def test_a_stack_resource_of_another_type_does_not_make_the_app_stack_managed(_sleep: MagicMock):
    """Only a KDA application resource counts; an unrelated resource sharing the id does not."""
    name = "agent-chosen-studio"
    v2 = _v2_list_client(name, present_polls=0)
    cfn = MagicMock()
    cfn.describe_stack_resources.return_value = _stack_member(resource_type="AWS::Logs::LogGroup")
    session = _multi3_session(v2, _empty_list_client(), cfn)

    result = _delete_v2(_resource(_V2_TYPE, name), session)

    assert result.status == HandlerStatus.SUCCESS
    v2.delete_application.assert_called_once()
    cfn.delete_stack.assert_not_called()
