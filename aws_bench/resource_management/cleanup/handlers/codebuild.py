"""CodeBuild project cleanup handler.

CloudControl (CCAPI) cannot operate on ``AWS::CodeBuild::Project`` — a
``get_resource`` / ``delete_resource`` returns ``UnsupportedActionException``,
which the CCAPI deleter classifies as "unsupported" and SKIPS. With no custom
handler, an agent-created project therefore survives the reset as an orphan and
contaminates the account. This handler deletes it via the CodeBuild service API.

The scanned identifier is the project NAME (the ``list_projects`` lister emits
names, and the existence verifier confirms by ``batch_get_projects(names=[...])``).

``delete_project`` succeeds even for a project with build history, EXCEPT when the
project ALSO has a resource policy — AWS rejects that until the policy is removed
(see https://docs.aws.amazon.com/codebuild/latest/userguide/delete-project.html).
A prepare step therefore best-effort removes any resource policy first; projects
without one (the common agent-created case) are unaffected.
"""

from __future__ import annotations

import boto3
from botocore.exceptions import ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.handlers._service_delete import service_delete
from aws_bench.resource_management.cleanup.models import HandlerResult, HandlerStatus
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)

_PROJECT_TYPE = "AWS::CodeBuild::Project"
# delete_project is idempotent, but surface the not-found path as already-gone.
_NOT_FOUND_CODES = ("ResourceNotFoundException",)


@resource_handler(_PROJECT_TYPE, role="prepare")
def _prepare_codebuild_project(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Remove a resource policy so delete_project is not rejected.

    A project with both builds and a resource policy cannot be deleted until the
    policy is removed. The policy ARN is the project ARN; we resolve it from the
    name via batch_get_projects. Best-effort: a project that is missing or has no
    resource policy is a clean no-op (SUCCESS), so the delete step proceeds.
    """
    name = resource.identifier
    try:
        client = build_client(session, "codebuild")
        resp = client.batch_get_projects(names=[name])
        projects = resp.get("projects", [])
        if not projects:
            return HandlerResult(
                resource_id=name,
                resource_type=resource.type,
                action="prepare",
                status=HandlerStatus.SUCCESS,
                message="Project not found; nothing to prepare",
            )
        project_arn = projects[0].get("arn")
        if not project_arn:
            return HandlerResult(
                resource_id=name,
                resource_type=resource.type,
                action="prepare",
                status=HandlerStatus.SUCCESS,
                message="Project has no ARN; nothing to prepare",
            )
        try:
            client.delete_resource_policy(resourceArn=project_arn)
            logger.debug("Removed resource policy on CodeBuild project '%s'", name)
        except ClientError as e:
            # No policy attached (or already gone) is the common case and not an error.
            if e.response.get("Error", {}).get("Code", "") not in ("ResourceNotFoundException",):
                raise
    except ClientError as e:
        return HandlerResult(
            resource_id=name,
            resource_type=resource.type,
            action="prepare",
            status=HandlerStatus.FAILED,
            message=f"Failed to clear CodeBuild resource policy: {e}",
        )
    return HandlerResult(
        resource_id=name,
        resource_type=resource.type,
        action="prepare",
        status=HandlerStatus.SUCCESS,
    )


@resource_handler(_PROJECT_TYPE, role="delete")
def _delete_codebuild_project(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete a CodeBuild project via the service API (CCAPI cannot)."""
    return service_delete(
        resource,
        session,
        client_name="codebuild",
        op_name="delete_project",
        id_param="name",
        not_found_codes=_NOT_FOUND_CODES,
        already_gone_message="CodeBuild project already gone",
        log_label="CodeBuild project",
    )
