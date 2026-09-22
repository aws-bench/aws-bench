"""Tests for aws_bench.resource_management.ccapi.exceptions."""

from __future__ import annotations

from botocore.exceptions import ClientError

from aws_bench.exceptions import AWSBenchError
from aws_bench.resource_management.ccapi.exceptions import (
    CloudControlError,
    CloudControlResourceDeletionException,
    ResourceExistenceCheckError,
    ResourceExistenceHandlerFailureError,
    ResourceExistenceUnsupportedError,
    is_not_found_error,
)
from aws_bench.resource_management.ccapi.models import Resource

# -- CloudControlResourceDeletionException --


def test_deletion_exception_inherits_from_awsbench_error():
    resource = Resource(type="AWS::S3::Bucket", identifier="my-bucket")
    exc = CloudControlResourceDeletionException(resource, ValueError("denied"))
    assert isinstance(exc, AWSBenchError)
    assert isinstance(exc, CloudControlError)


def test_deletion_exception_message_includes_resource_info():
    resource = Resource(type="AWS::S3::Bucket", identifier="my-bucket")
    original = ValueError("access denied")
    exc = CloudControlResourceDeletionException(resource, original)
    assert "AWS::S3::Bucket" in str(exc)
    assert "my-bucket" in str(exc)
    assert exc.resource is resource
    assert exc.original_error is original


# -- ResourceExistenceCheckError --


def test_existence_check_error_inherits_from_awsbench_error():
    exc = ResourceExistenceCheckError("check failed")
    assert isinstance(exc, AWSBenchError)
    assert isinstance(exc, CloudControlError)


def test_unsupported_error_is_a_resource_existence_check_error():
    """The unsupported subclass stays catchable as ResourceExistenceCheckError.

    Existing ``except ResourceExistenceCheckError`` handlers (verification manager, account
    scanner) must keep catching it, while callers that care can distinguish it — the deleter
    skips an unsupported type but attempts a generic/unverified failure.
    """
    exc = ResourceExistenceUnsupportedError("CCAPI does not support X")
    assert isinstance(exc, ResourceExistenceCheckError)
    assert issubclass(ResourceExistenceUnsupportedError, ResourceExistenceCheckError)


def test_handler_failure_error_is_check_error_but_not_unsupported():
    """A broken-handler failure is a check error, not the unsupported subclass.

    The distinction is load-bearing: the deleter SKIPS an unsupported type but must ATTEMPT a
    handler-failure type (its handler is broken, not the type unsupported), or a real orphan of
    that type would leak.
    """
    exc = ResourceExistenceHandlerFailureError("HandlerInternalFailureException")
    assert isinstance(exc, ResourceExistenceCheckError)
    assert not isinstance(exc, ResourceExistenceUnsupportedError)
    assert issubclass(ResourceExistenceHandlerFailureError, ResourceExistenceCheckError)


# -- is_not_found_error --


def test_is_not_found_error_general_service_does_not_exist():
    exc = ClientError(
        {"Error": {"Code": "GeneralServiceException", "Message": "routeTable does not exist"}},
        "GetResource",
    )
    assert is_not_found_error(exc) is True


def test_is_not_found_error_general_service_not_found():
    exc = ClientError(
        {
            "Error": {
                "Code": "GeneralServiceException",
                "Message": "Resource could not be found",
            }
        },
        "GetResource",
    )
    assert is_not_found_error(exc) is True


def test_is_not_found_error_general_service_other_message():
    exc = ClientError(
        {"Error": {"Code": "GeneralServiceException", "Message": "access denied"}},
        "GetResource",
    )
    assert is_not_found_error(exc) is False


def test_is_not_found_error_non_general_service_error():
    exc = ClientError(
        {"Error": {"Code": "ValidationException", "Message": "does not exist"}},
        "GetResource",
    )
    assert is_not_found_error(exc) is False


def test_is_not_found_error_non_client_error():
    assert is_not_found_error(RuntimeError("does not exist")) is False
