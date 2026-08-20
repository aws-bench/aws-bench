"""Bedrock Knowledge Base cleanup handler.

Deleting a task-created ``AWS::Bedrock::KnowledgeBase`` on reset has three
wrinkles, each handled in a dedicated helper below:
  * it owns data sources that block deletion — prepare deletes them first;
  * its delete assumes the KB's own IAM execution role, which an agent may not
    have scoped for teardown — prepare grants a transient teardown policy on that
    role (``_ensure_role_can_teardown``), and the delete step removes it once the
    KB is gone (``_revoke_teardown_grant``) so it never lingers on a role reset
    preserves;
  * the delete is asynchronous — the delete step waits for terminal deletion,
    re-issuing on ``DELETE_UNSUCCESSFUL`` (``_wait_for_terminal_deletion``).
"""

from __future__ import annotations

import json

import boto3
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.models import (
    PROTECTED_IAM_ROLE_NAMES,
    SERVICE_ROLE_PREFIX,
    Resource,
)
from aws_bench.resource_management.cleanup.handler_registry import resource_handler
from aws_bench.resource_management.cleanup.handlers._service_delete import (
    prepare_error_result,
    service_delete,
)
from aws_bench.resource_management.cleanup.models import HandlerResult, HandlerStatus
from aws_bench.resource_management.utils.polling import wait_until
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)

_NOT_FOUND_CODES = ("ResourceNotFoundException", "NotFoundException")
_IAM_NOT_FOUND_CODES = ("NoSuchEntity", "NoSuchEntityException")

# Actions a knowledge base's execution role must be allowed to call so Bedrock
# can tear down the KB's backing store during deletion. Covers every KB
# backing-store family (Redshift structured, OpenSearch/Aurora/S3 vector) since
# the handler is type-agnostic. The grant is transient: the delete step removes it
# once the KB is gone, so it never outlives the KB even on a role reset preserves.
_KB_TEARDOWN_ACTIONS = [
    "sqlworkbench:*",
    "redshift:*",
    "redshift-data:*",
    "redshift-serverless:*",
    "aoss:*",
    "es:*",
    "rds:*",
    "rds-data:*",
    "s3:*",
    "bedrock:*",
]
_KB_TEARDOWN_POLICY_NAME = "AwsBenchKBTeardown"

# Bounded terminal-deletion polling so a stuck KB cannot hang reset forever:
# 30 attempts x 10s = up to 300s, ample for KB (and its vector store) teardown.
_WAITER_TIMEOUT_SEC = 300
_WAITER_INTERVAL_SEC = 10
_DELETE_UNSUCCESSFUL = "DELETE_UNSUCCESSFUL"
# A DELETE_UNSUCCESSFUL KB is terminal — Bedrock does NOT retry on its own. Re-issue
# delete_knowledge_base a bounded number of times so a transient cause clears (most
# often IAM: the teardown policy just granted in _prepare has not yet propagated to
# the role's assumed session when Bedrock's first delete tries sqlworkbench, so that
# one attempt is denied; a re-issue on a later poll, once the policy is effective,
# succeeds). Re-issues are paced by the poll interval (_WAITER_INTERVAL_SEC).
_DELETE_REISSUE_MAX = 5


def _wait_for_terminal_deletion(client: BaseClient, kb_id: str) -> None:
    """Block until the knowledge base is actually gone, re-issuing on failure.

    ``delete_knowledge_base`` is asynchronous; the post-run reset verification
    fails if it races a still-present (``DELETING``) KB, so we must not return
    until ``get_knowledge_base`` reports it gone. ``bedrock-agent`` ships no
    deletion waiter, so poll manually.

    A KB that reaches ``DELETE_UNSUCCESSFUL`` is a TERMINAL failure that Bedrock
    never retries — commonly because the execution-role teardown policy granted in
    ``_prepare`` had not propagated when Bedrock's first delete assumed the role
    (an IAM eventual-consistency race). We re-issue the delete (bounded); once the
    policy is effective it finalizes in seconds. If it still cannot be deleted
    after the re-issues, raise so ``service_delete`` maps it to FAILED.
    """
    reissues = 0

    def _gone() -> bool:
        nonlocal reissues
        try:
            kb = client.get_knowledge_base(knowledgeBaseId=kb_id)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code", "") in _NOT_FOUND_CODES:
                return True  # Fully deleted.
            raise  # transient (throttling/etc.) — wait_until swallows and retries
        status = (kb.get("knowledgeBase") or {}).get("status", "")
        if status == _DELETE_UNSUCCESSFUL and reissues < _DELETE_REISSUE_MAX:
            # Terminal failure that won't self-heal — re-issue the delete now that
            # the teardown policy has had time to propagate.
            reissues += 1
            logger.debug(
                "KB '%s' is %s; re-issuing delete (attempt %d/%d)",
                kb_id,
                _DELETE_UNSUCCESSFUL,
                reissues,
                _DELETE_REISSUE_MAX,
            )
            try:
                client.delete_knowledge_base(knowledgeBaseId=kb_id)
            except ClientError as e:
                if e.response.get("Error", {}).get("Code", "") in _NOT_FOUND_CODES:
                    return True
                raise
        return False

    if wait_until(_gone, timeout=_WAITER_TIMEOUT_SEC, interval=_WAITER_INTERVAL_SEC):
        return
    try:
        client.get_knowledge_base(knowledgeBaseId=kb_id)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") in _NOT_FOUND_CODES:
            return  # Actually gone despite the poll timeout.
        raise
    # Still present after the bounded wait — raise so service_delete maps it to FAILED.
    raise ClientError(
        {
            "Error": {
                "Code": "DeletionTimeout",
                "Message": f"still present after {_WAITER_TIMEOUT_SEC}s ({reissues} re-issue(s))",
            }
        },
        "GetKnowledgeBase",
    )


def _kb_teardown_role(client: BaseClient, kb_id: str) -> str | None:
    """Return the KB's execution-role name the teardown policy applies to, or None.

    Reads the KB's ``roleArn``. Returns ``None`` when there is nothing to
    grant/revoke: the KB is gone/unreadable, has no role, or the role is a
    service-linked or protected role we must never mutate (a KB cannot legitimately
    use one as its execution role, and IAM would reject the inline policy anyway).
    """
    try:
        kb = client.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"]
    except (ClientError, BotoCoreError, KeyError):
        return None  # KB already gone or unreadable.
    role_arn = kb.get("roleArn", "")
    role_name = role_arn.rsplit("/", 1)[-1] if role_arn else ""
    if not role_name:
        return None
    if role_name.startswith(SERVICE_ROLE_PREFIX) or role_name in PROTECTED_IAM_ROLE_NAMES:
        logger.debug("Skipping teardown grant for protected/service role %s", role_name)
        return None
    return role_name


def _ensure_role_can_teardown(session: boto3.Session, role_name: str) -> None:
    """Grant ``role_name`` the actions Bedrock needs to tear down the KB's store.

    ``delete_knowledge_base`` assumes the KB's own execution role to delete its
    backing store; an agent-created role scoped to build/query the KB can lack
    the teardown actions, wedging the delete in ``DELETE_UNSUCCESSFUL``. Attach a
    teardown-scoped inline policy so the async delete can complete.

    The grant is safe to apply to any role because the delete step removes it once
    the KB is gone (``_revoke_teardown_grant``) — so it never lingers on a role
    reset preserves. Best-effort: any failure here (role already gone, IAM denied)
    is swallowed so prepare still proceeds — the delete step surfaces a genuinely
    stuck KB.
    """
    iam = build_client(session, "iam")
    try:
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName=_KB_TEARDOWN_POLICY_NAME,
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {"Effect": "Allow", "Action": _KB_TEARDOWN_ACTIONS, "Resource": "*"}
                    ],
                }
            ),
        )
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") not in _IAM_NOT_FOUND_CODES:
            # Log-worthy but not fatal: let the delete step report a stuck KB.
            logger.warning("Could not grant teardown policy to role %s: %s", role_name, e)
    except BotoCoreError as e:
        logger.warning("Could not grant teardown policy to role %s: %s", role_name, e)


def _revoke_teardown_grant(session: boto3.Session, role_name: str) -> None:
    """Remove the transient teardown policy from ``role_name`` after KB deletion.

    Keeps the broad grant from outliving the KB. Best-effort and idempotent: the
    role is usually deleted right after the KB (an agent-created new resource), so
    a missing role/policy is expected and swallowed; only unexpected errors log.
    """
    iam = build_client(session, "iam")
    try:
        iam.delete_role_policy(RoleName=role_name, PolicyName=_KB_TEARDOWN_POLICY_NAME)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") not in _IAM_NOT_FOUND_CODES:
            logger.warning("Could not revoke teardown policy from role %s: %s", role_name, e)
    except BotoCoreError as e:
        logger.warning("Could not revoke teardown policy from role %s: %s", role_name, e)


@resource_handler("AWS::Bedrock::KnowledgeBase", role="prepare")
def _prepare(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete the KB's data sources and grant its role teardown permissions."""
    client = build_client(session, "bedrock-agent")
    kb_id = resource.identifier
    role_name = _kb_teardown_role(client, kb_id)
    if role_name:
        _ensure_role_can_teardown(session, role_name)
    try:
        for page in client.get_paginator("list_data_sources").paginate(knowledgeBaseId=kb_id):
            for ds in page.get("dataSourceSummaries", []):
                # Delete each data source best-effort per item: one already gone (a retried
                # prepare, or a concurrent delete) must not abort the loop and leave the rest
                # undeleted — which would then block the KB delete and stall reset. A
                # not-found on a single data source is skipped; any other error propagates to
                # the outer handler and maps to a FAILED result.
                try:
                    client.delete_data_source(
                        knowledgeBaseId=kb_id, dataSourceId=ds["dataSourceId"]
                    )
                except ClientError as e:
                    if e.response.get("Error", {}).get("Code", "") not in _NOT_FOUND_CODES:
                        raise
    except (ClientError, BotoCoreError) as e:
        return prepare_error_result(
            e,
            resource,
            not_found_codes=_NOT_FOUND_CODES,
            not_found_message="Knowledge base not found",
            failed_message_prefix="Failed to delete data sources",
        )
    return HandlerResult(
        resource_id=kb_id,
        resource_type=resource.type,
        action="prepare",
        status=HandlerStatus.SUCCESS,
        message="Deleted knowledge base data sources",
    )


@resource_handler("AWS::Bedrock::KnowledgeBase", role="delete")
def _delete(resource: Resource, session: boto3.Session) -> HandlerResult:
    """Delete the knowledge base via the bedrock-agent API, then wait for terminal deletion.

    Captures the KB's execution role up front (it is readable until the KB is gone)
    so the transient teardown policy granted in prepare can be removed once the KB
    is deleted — the ``finally`` runs whether the delete succeeded or got stuck, so
    the broad grant never lingers on a role reset preserves.
    """
    client = build_client(session, "bedrock-agent")
    role_name = _kb_teardown_role(client, resource.identifier)
    try:
        return service_delete(
            resource,
            session,
            client_name="bedrock-agent",
            op_name="delete_knowledge_base",
            id_param="knowledgeBaseId",
            not_found_codes=_NOT_FOUND_CODES,
            already_gone_message="Knowledge base already gone",
            log_label="Bedrock knowledge base",
            post_delete=lambda c: _wait_for_terminal_deletion(c, resource.identifier),
        )
    finally:
        if role_name:
            _revoke_teardown_grant(session, role_name)
