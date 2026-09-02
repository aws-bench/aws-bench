"""Credential and env-resolution helpers for AWS trials.

Functions take primitive arguments (``account_id``, ``task_name``, ``job_id``)
that the trial reads off ``self.config`` / ``self.task`` and passes in.
"""

from __future__ import annotations

from uuid import UUID

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.dataset.models import RoleType
from aws_bench.logging.logger import get_logger
from aws_bench.utils.credentials_provider import CredentialProvider, build_session_name
from aws_bench.utils.placeholders import substitute_placeholders

logger = get_logger(__name__)


def session_name(*, task_name: str, role_type: RoleType, job_id: UUID | None) -> str:
    r"""Build an STS RoleSessionName for CloudTrail auditing (<=64 chars, [\w+=,.@-]).

    Ordered ``app-<role>-<task>-<job>`` so that if the name exceeds 64 chars,
    ``build_session_name``'s trim drops the job-id tail rather than the
    audit-meaningful role and task. '/' in an org/name task name becomes '-'
    (STS allows only [\w+=,.@-]).
    """
    safe_name = task_name.replace("/", "-")
    segments = [str(role_type), safe_name]
    if job_id:
        segments.append(str(job_id))
    return build_session_name(*segments)


def resolve_env_with_creds(
    *,
    raw_env: dict[str, str],
    placeholders: dict[str, dict[str, str]],
    creds: dict[str, str],
) -> dict[str, str]:
    """Resolve ``{{placeholder}}`` patterns in env vars and append credentials.

    ``placeholders`` is the tag-keyed export map. Credentials are applied last so
    they override any conflicting keys. Used for pre-invoke, post-invoke, and
    verifier env sections.
    """
    env = {k: substitute_placeholders(v, placeholders) for k, v in raw_env.items()}
    env.update(creds)
    return env


def assume_role_for_script(
    *,
    account_id: str,
    role_name: str | None,
    role_type: RoleType,
    task_name: str,
    job_id: UUID | None,
) -> dict[str, str]:
    """Assume an IAM role for a script/verifier, falling back to org access role."""
    if not role_name:
        role_name = ORG_ACCESS_ROLE
        logger.debug(
            f"No custom role for {role_type} in {task_name}, using default org access role"
        )

    return CredentialProvider.get().chain_assume_role(
        account_id=account_id,
        role_name=role_name,
        session_name=session_name(task_name=task_name, role_type=role_type, job_id=job_id),
    )
