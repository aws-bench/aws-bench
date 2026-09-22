"""SageMaker cleanup handlers.

Cloud Control (CCAPI) does not support ``AWS::SageMaker::EndpointConfig``
(``UnsupportedActionException``), so endpoint configs left behind after their
endpoint is deleted are never swept. This custom handler removes them via
the SageMaker API.
"""

from __future__ import annotations

import boto3

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.handlers._service_delete import service_delete
from aws_bench.resource_management.cleanup.models import HandlerResult

# SageMaker returns ValidationException when the resource does not exist.
_SM_NOT_FOUND_CODES = ("ValidationException",)


@resource_handler("AWS::SageMaker::EndpointConfig", role="delete")
def _delete_endpoint_config(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete a SageMaker EndpointConfig via the SageMaker API."""
    return service_delete(
        resource,
        session,
        client_name="sagemaker",
        op_name="delete_endpoint_config",
        id_param="EndpointConfigName",
        not_found_codes=_SM_NOT_FOUND_CODES,
        already_gone_message="SageMaker EndpointConfig already gone",
        log_label="SageMaker EndpointConfig",
    )
