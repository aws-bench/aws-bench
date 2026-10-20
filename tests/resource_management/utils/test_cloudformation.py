"""Tests for CloudFormation utility functions."""

from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from aws_bench.resource_management.utils.cloudformation import (
    disable_termination_protection,
    get_stack_resource_drifts,
    is_stack_not_found,
)


def test_get_stack_resource_drifts_single_page():
    """Returns all drifts when results fit in single page."""
    mock_client = MagicMock()
    mock_client.describe_stack_resource_drifts.return_value = {
        "StackResourceDrifts": [
            {"LogicalResourceId": "Resource1", "StackResourceDriftStatus": "IN_SYNC"},
            {"LogicalResourceId": "Resource2", "StackResourceDriftStatus": "MODIFIED"},
        ]
    }

    result = get_stack_resource_drifts(mock_client, "test-stack")

    assert len(result) == 2
    assert result[0]["LogicalResourceId"] == "Resource1"
    assert result[1]["LogicalResourceId"] == "Resource2"
    mock_client.describe_stack_resource_drifts.assert_called_once_with(StackName="test-stack")


def test_get_stack_resource_drifts_multiple_pages():
    """Paginates through multiple pages to get all drifts."""
    mock_client = MagicMock()

    # First page with NextToken
    mock_client.describe_stack_resource_drifts.side_effect = [
        {
            "StackResourceDrifts": [
                {"LogicalResourceId": "Resource1", "StackResourceDriftStatus": "IN_SYNC"}
            ],
            "NextToken": "token-1",
        },
        {
            "StackResourceDrifts": [
                {"LogicalResourceId": "Resource2", "StackResourceDriftStatus": "MODIFIED"}
            ],
            "NextToken": "token-2",
        },
        {
            "StackResourceDrifts": [
                {"LogicalResourceId": "Resource3", "StackResourceDriftStatus": "DELETED"}
            ]
        },
    ]

    result = get_stack_resource_drifts(mock_client, "test-stack")

    assert len(result) == 3
    assert result[0]["LogicalResourceId"] == "Resource1"
    assert result[1]["LogicalResourceId"] == "Resource2"
    assert result[2]["LogicalResourceId"] == "Resource3"

    # Verify pagination calls
    assert mock_client.describe_stack_resource_drifts.call_count == 3
    calls = mock_client.describe_stack_resource_drifts.call_args_list
    assert calls[0][1] == {"StackName": "test-stack"}
    assert calls[1][1] == {"StackName": "test-stack", "NextToken": "token-1"}
    assert calls[2][1] == {"StackName": "test-stack", "NextToken": "token-2"}


def test_get_stack_resource_drifts_empty():
    """Returns empty list when no drifts found."""
    mock_client = MagicMock()
    mock_client.describe_stack_resource_drifts.return_value = {"StackResourceDrifts": []}

    result = get_stack_resource_drifts(mock_client, "test-stack")

    assert result == []


def test_is_stack_not_found_returns_true():
    """Returns True for ValidationError with 'does not exist' message."""
    from botocore.exceptions import ClientError

    error = ClientError(
        error_response={
            "Error": {"Code": "ValidationError", "Message": "Stack with id my-stack does not exist"}
        },
        operation_name="DescribeStacks",
    )

    assert is_stack_not_found(error)


def test_is_stack_not_found_returns_false_for_other_validation_errors():
    """Returns False for ValidationError without 'does not exist' message."""
    from botocore.exceptions import ClientError

    error = ClientError(
        error_response={"Error": {"Code": "ValidationError", "Message": "Invalid parameter"}},
        operation_name="DescribeStacks",
    )

    assert not is_stack_not_found(error)


def test_is_stack_not_found_returns_false_for_other_errors():
    """Returns False for non-ValidationError codes."""
    from botocore.exceptions import ClientError

    error = ClientError(
        error_response={"Error": {"Code": "AccessDenied", "Message": "Access denied"}},
        operation_name="DescribeStacks",
    )

    assert not is_stack_not_found(error)


# -- disable_termination_protection --


def _client_with_protection(enabled: bool) -> MagicMock:
    client = MagicMock()
    client.describe_stacks.return_value = {"Stacks": [{"EnableTerminationProtection": enabled}]}
    return client


def test_disable_termination_protection_turns_protection_off():
    """A protected stack gets UpdateTerminationProtection(False); reports it did."""
    client = _client_with_protection(enabled=True)

    assert disable_termination_protection(client, "my-stack") is True
    client.update_termination_protection.assert_called_once_with(
        EnableTerminationProtection=False, StackName="my-stack"
    )


def test_disable_termination_protection_noop_when_off():
    """An unprotected stack is left alone."""
    client = _client_with_protection(enabled=False)

    assert disable_termination_protection(client, "my-stack") is False
    client.update_termination_protection.assert_not_called()


def test_disable_termination_protection_swallows_access_denied():
    """Permission errors are logged, not raised; DeleteStack surfaces the real failure."""
    client = MagicMock()
    client.describe_stacks.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "forbidden"}}, "DescribeStacks"
    )

    assert disable_termination_protection(client, "my-stack") is False
    client.update_termination_protection.assert_not_called()


def test_disable_termination_protection_swallows_stack_not_found():
    client = MagicMock()
    client.describe_stacks.side_effect = ClientError(
        {"Error": {"Code": "ValidationError", "Message": "Stack with id my-stack does not exist"}},
        "DescribeStacks",
    )

    assert disable_termination_protection(client, "my-stack") is False


def test_disable_termination_protection_swallows_update_error():
    client = _client_with_protection(enabled=True)
    client.update_termination_protection.side_effect = ClientError(
        {"Error": {"Code": "ValidationError", "Message": "Stack is in DELETE_IN_PROGRESS"}},
        "UpdateTerminationProtection",
    )

    assert disable_termination_protection(client, "my-stack") is False
