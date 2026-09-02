"""Credential provider for assuming roles into member accounts."""

from __future__ import annotations

import time

import boto3
from botocore.config import Config
from botocore.credentials import DeferredRefreshableCredentials, create_assume_role_refresher
from botocore.exceptions import ClientError

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.account_management.exceptions import AccountResolutionError
from aws_bench.account_management.preexisting import active_account_config
from aws_bench.constants import DEFAULT_REGION
from aws_bench.exceptions import CredentialError
from aws_bench.logging.logger import get_logger
from aws_bench.utils.concurrent import build_client, build_session, raise_if_shutdown

logger = get_logger(__name__)

# Default retry policy for every session this provider builds. aws-bench fans
# out many concurrent clients against one account+region, so adaptive mode's
# client-side rate limiter (vs. botocore's legacy default) damps throttle bursts.
_RETRY_DEFAULTS = {"max_attempts": 8, "mode": "adaptive"}


def _default_client_config() -> Config:
    """A fresh Config per call, so no two sessions share the mutable retries dict."""
    return Config(retries=dict(_RETRY_DEFAULTS))


def _apply_client_defaults(session: boto3.Session) -> boto3.Session:
    """Set the retry default on ``session``; a client's own ``config=`` still wins."""
    session._session.set_default_client_config(_default_client_config())
    return session


# Building blocks for STS RoleSessionNames. Every name is composed as
# ``app[-<segment>...]`` so CloudTrail entries are uniformly attributable and the
# name stays neutral — it must not reveal to an evaluated agent that it is running
# inside aws-bench. ``SESSION_NAME_PREFIX`` is the single source of truth.
SESSION_NAME_PREFIX = "app"
# STS caps RoleSessionName at 64 chars.
MAX_SESSION_NAME_LEN = 64


def build_session_name(*segments: str) -> str:
    """Compose a ``SESSION_NAME_PREFIX``-prefixed STS RoleSessionName from ``segments``.

    Joins ``SESSION_NAME_PREFIX`` and ``segments`` with ``-`` and truncates to STS's
    64-char limit. This is the single constructor for session names, so the
    prefix lives in exactly one place.

    Example:
        ``build_session_name("rm", "cleanup")`` -> ``"app-rm-cleanup"``
    """
    return "-".join([SESSION_NAME_PREFIX, *segments])[:MAX_SESSION_NAME_LEN]


def enforce_session_name(session_name: str) -> str:
    """Validate and normalize an STS RoleSessionName for CloudTrail attribution.

    Backstop at the generic STS choke points all assume-role paths funnel
    through: even a hand-written name (not built via :func:`build_session_name`)
    must carry the ``app-`` prefix, so the convention is enforced at runtime.

    Returns the name truncated to STS's 64-char limit; callers historically
    relied on this truncation (e.g. the ``app-<role>-<task>-<job>`` builder
    drops its job-id tail rather than the audit-meaningful prefix).

    Raises:
        ValueError: If the name does not start with ``app-``.
    """
    required_prefix = SESSION_NAME_PREFIX + "-"
    if not session_name.startswith(required_prefix):
        raise ValueError(
            f"RoleSessionName must start with {required_prefix!r} for CloudTrail "
            f"attribution, got {session_name!r}"
        )
    return session_name[:MAX_SESSION_NAME_LEN]


def env_credentials_dict_to_session(creds: dict[str, str]) -> boto3.Session:
    """Convert environment variable-type AWS credentials dict to a boto3 Session.

    Expects the dict to contain: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN.
    Sets region_name to DEFAULT_REGION so clients created from this session have a default region.
    """
    return _apply_client_defaults(
        boto3.Session(
            aws_access_key_id=creds["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=creds["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=creds["AWS_SESSION_TOKEN"],
            region_name=DEFAULT_REGION,
        )
    )


def session_to_env_credentials(session: boto3.Session) -> dict[str, str]:
    """Snapshot a boto3 Session's credentials as an env-var dict.

    Returns a dict with AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and
    AWS_SESSION_TOKEN (empty string if no session token), suitable for
    :func:`env_credentials_dict_to_session`.

    The snapshot is static (no auto-refresh), so for refreshable credentials we
    first force a full re-mint — the recipient (container, worker thread) gets the
    maximum duration, not the aged tail of the parent's current credential.

    Raises:
        CredentialError: If the session has no credentials to snapshot.
    """
    creds = session.get_credentials()
    if creds is None:
        raise CredentialError("Session has no credentials to snapshot")
    # RefreshableCredentials only re-mints when near expiry; force a full-duration
    # refresh so the frozen snapshot starts fresh, not with the parent's remainder.
    protected_refresh = getattr(creds, "_protected_refresh", None)
    if callable(protected_refresh):
        protected_refresh(is_mandatory=True)
        # Log the snapshot's expiry so a downstream credential failure (the recipient
        # is stuck with this static token) can be traced to when it was due to die.
        expiry = getattr(creds, "_expiry_time", None)
        if expiry is not None:
            logger.debug("Snapshotted credentials expire at %s", expiry.isoformat())
    frozen = creds.get_frozen_credentials()
    return {
        "AWS_ACCESS_KEY_ID": frozen.access_key,
        "AWS_SECRET_ACCESS_KEY": frozen.secret_key,
        "AWS_SESSION_TOKEN": frozen.token or "",
    }


def session_to_credential_process(session: boto3.Session) -> dict[str, object]:
    """Snapshot a Session's credentials as an AWS ``credential_process`` JSON dict.

    Returns the ``credential_process`` stdout shape — ``Version``, the three key
    fields, and the RFC3339 ``Expiration`` the SDK uses to re-invoke the process.
    Pass a refreshable session (e.g. ``get_session_for_account``).

    Raises:
        CredentialError: If the session has no credentials, or they carry no
            expiry (without ``Expiration`` the SDK treats the output as static).
    """
    creds = session.get_credentials()
    if creds is None:
        raise CredentialError("Session has no credentials to snapshot")
    frozen = creds.get_frozen_credentials()
    expiry = getattr(creds, "_expiry_time", None)
    if expiry is None:
        raise CredentialError("Session credentials have no expiry; cannot build credential_process")
    return {
        "Version": 1,
        "AccessKeyId": frozen.access_key,
        "SecretAccessKey": frozen.secret_key,
        "SessionToken": frozen.token,
        "Expiration": expiry.isoformat(),
    }


def assumed_credentials_dict_to_credentials_env(creds: dict[str, str]) -> dict[str, str]:
    """Convert "sts.assume_role"-type credentials to env vars."""
    return {
        "AWS_ACCESS_KEY_ID": creds["AccessKeyId"],
        "AWS_SECRET_ACCESS_KEY": creds["SecretAccessKey"],
        "AWS_SESSION_TOKEN": creds["SessionToken"],
    }


def _credentials_block(profile: str, creds: dict[str, str]) -> str:
    return (
        f"[{profile}]\n"
        f"aws_access_key_id={creds['AWS_ACCESS_KEY_ID']}\n"
        f"aws_secret_access_key={creds['AWS_SECRET_ACCESS_KEY']}\n"
        f"aws_session_token={creds['AWS_SESSION_TOKEN']}\n"
    )


def build_aws_credentials_file(per_tag_creds: dict[str, dict[str, str]]) -> str:
    """Render an ``~/.aws/credentials`` body with static creds, one block per tag.

    ``per_tag_creds`` maps each account tag to its already-assumed STS creds
    (the ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_SESSION_TOKEN``
    keys). Each block is named for its tag; the caller selects one with
    ``AWS_PROFILE=<tag>`` (or ``--profile <tag>``). No ``[default]`` block is
    written: a tag is used only when named explicitly.
    """
    parts: list[str] = []
    for tag, creds in per_tag_creds.items():
        parts.append(_credentials_block(tag, creds))
    return "\n".join(parts)


def _create_refreshable_session(
    parent_session: boto3.Session,
    role_arn: str,
    session_name: str,
    region: str = DEFAULT_REGION,
) -> boto3.Session:
    """Create a boto3 session with auto-refreshable assume role credentials.

    Credentials will use the default 1-hour duration and automatically refresh
    ~15 minutes before expiry.

    Args:
        parent_session: Parent boto3 session whose credentials are used to
            assume the role. Must itself have refreshable credentials
            (IAM role, SSO, credential_process, etc.) for long-running
            operations.
        role_arn: ARN of the role to assume
        session_name: Session name for CloudTrail auditing
        region: AWS region for the session (default: DEFAULT_REGION)

    Returns:
        boto3.Session with refreshable credentials
    """
    params = {
        "RoleArn": role_arn,
        "RoleSessionName": enforce_session_name(session_name),
    }

    def refresh():
        sts_client = build_client(parent_session, "sts")
        return create_assume_role_refresher(sts_client, params)()

    credentials = DeferredRefreshableCredentials(refresh_using=refresh, method="sts-assume-role")

    session = build_session(lambda: boto3.Session(region_name=region))
    session._session._credentials = credentials
    return _apply_client_defaults(session)


def create_regional_session(parent_session: boto3.Session, region: str) -> boto3.Session:
    """Create a session for ``region`` sharing ``parent_session``'s credentials.

    Sharing the credential provider preserves auto-refresh behavior across
    regional sessions.
    """
    regional_session = build_session(lambda: boto3.Session(region_name=region))
    regional_session._session._credentials = parent_session._session._credentials
    return _apply_client_defaults(regional_session)


class CredentialProvider:
    """Assumes roles and provides caller identity information.

    Singleton — use CredentialProvider.get() to get or create the shared instance.
    """

    _instance: CredentialProvider | None = None

    def __init__(self, session: boto3.Session | None = None) -> None:
        """Default session uses the ambient credentials chain in DEFAULT_REGION.

        A caller-supplied session is stamped too, so a session built elsewhere
        still gets aws-bench's default client config (retry policy).
        """
        self._session = _apply_client_defaults(session or boto3.Session(region_name=DEFAULT_REGION))
        self._caller_account_id: str | None = None

    @property
    def session(self) -> boto3.Session:
        """The underlying boto3 session."""
        return self._session

    def get_management_session(self) -> boto3.Session:
        """Get session for management account (same as main session).

        This method is provided for clarity and future extensibility if we need
        to differentiate management account sessions.

        Returns:
            boto3.Session: Session for management account operations
        """
        return self._session

    @property
    def _sts(self):
        """Create a fresh STS client from the current session.

        This ensures we always use up-to-date credentials, even if the underlying
        session has refreshable credentials that have been renewed.
        """
        return self._session.client("sts")

    @classmethod
    def get(cls, session: boto3.Session | None = None) -> CredentialProvider:
        """Return the shared instance, creating it on first call.

        The session parameter is only used on the first call. Subsequent calls
        return the existing instance regardless of the session argument.
        """
        if cls._instance is None:
            cls._instance = cls(session)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton. Useful for testing."""
        cls._instance = None

    def get_caller_account_id(self) -> str:
        """Return the AWS account ID of the current caller. Cached after first call."""
        if self._caller_account_id is None:
            self._caller_account_id = self._sts.get_caller_identity()["Account"]
        account_id = self._caller_account_id
        if account_id is None:
            raise CredentialError("Failed to resolve caller account ID")
        return account_id

    def assume_role(
        self,
        account_id: str,
        role_name: str,
        session_name: str,
        duration_seconds: int = 3600,
    ) -> dict[str, str]:
        """Assume a role in a member account and returns credentials.

        Args:
            account_id: The target AWS account ID.
            role_name: IAM role name to assume.
            session_name: Session name for CloudTrail auditing.
            duration_seconds: How long the credentials are valid.

        Returns:
            Dict with AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN.
        """
        role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
        logger.debug(f"Assuming role {role_arn} (session={session_name}).")

        response = self._sts.assume_role(
            RoleArn=role_arn,
            RoleSessionName=enforce_session_name(session_name),
            DurationSeconds=duration_seconds,
        )
        return assumed_credentials_dict_to_credentials_env(response["Credentials"])

    def _preexisting_role(self, account_id: str, role_name: str | None) -> str:
        """Resolve the direct role used for an externally owned account.

        ``OrganizationAccountAccessRole`` is an implementation detail of accounts
        created by aws-bench.  In pre-existing mode it means "the configured
        runner identity" instead.  Explicit task roles remain explicit.

        Raises:
            CredentialError: If no config is active.
            AccountResolutionError: If ``account_id`` is outside the allowlist.
        """
        active = active_account_config()
        if active is None:
            raise CredentialError(
                f"No pre-existing account config is active; cannot resolve a role for "
                f"account {account_id}"
            )
        config, _ = active
        allowed = {
            configured_id for tags in config.accounts.values() for configured_id in tags.values()
        }
        if account_id not in allowed:
            raise AccountResolutionError(
                f"Account {account_id} is not in the active pre-existing allowlist"
            )
        if role_name in (None, ORG_ACCESS_ROLE):
            return config.runner_role
        return role_name

    def _ambient_is_target_role(self, account_id: str, role_name: str) -> bool:
        """Return whether the ambient STS identity already is ``role_name``."""
        identity = self._sts.get_caller_identity()
        arn = str(identity.get("Arn", ""))
        marker = f"arn:aws:sts::{account_id}:assumed-role/{role_name}/"
        return arn.startswith(marker)

    def chain_assume_role(
        self,
        account_id: str,
        session_name: str,
        role_name: str | None = None,
        duration_seconds: int = 3600,
    ) -> dict[str, str]:
        """Assume into a member account, always via the org access role.

        Hop 1 always assumes ORG_ACCESS_ROLE in the target account; this hop
        is never skipped. If ``role_name`` is given and differs from
        ORG_ACCESS_ROLE, hop 2 chains from that session into ``role_name``.
        Otherwise the hop-1 credentials are returned directly.

        Args:
            account_id: The target AWS account ID.
            role_name: IAM role name for the optional second hop. When unset
                or equal to ORG_ACCESS_ROLE, only the first hop runs.
            session_name: Session name for CloudTrail auditing.
            duration_seconds: How long the credentials are valid.

        Returns:
            Dict with AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN.
        """
        parent_session = self._session
        preexisting = active_account_config()
        if preexisting is not None:
            config, _ = preexisting
            target_role = self._preexisting_role(account_id, role_name)
            # Already running as the target role — self-assume would fail, so reuse it.
            if self._ambient_is_target_role(account_id, target_role):
                return session_to_env_credentials(self._session)
            if target_role != config.runner_role and not self._ambient_is_target_role(
                account_id, config.runner_role
            ):
                runner_creds = self.assume_role(
                    account_id,
                    config.runner_role,
                    build_session_name("runner", account_id[-6:]),
                    duration_seconds=duration_seconds,
                )
                parent_session = env_credentials_dict_to_session(runner_creds)
            if parent_session is not self._session:
                role_arn = f"arn:aws:iam::{account_id}:role/{target_role}"
                response = build_client(parent_session, "sts").assume_role(
                    RoleArn=role_arn,
                    RoleSessionName=enforce_session_name(session_name),
                    DurationSeconds=duration_seconds,
                )
                return assumed_credentials_dict_to_credentials_env(response["Credentials"])
            return self.assume_role(
                account_id,
                target_role,
                session_name,
                duration_seconds=duration_seconds,
            )

        # Hop 1: always go through the org access role
        hop1_session_name = (
            session_name
            if (not role_name or role_name == ORG_ACCESS_ROLE)
            else build_session_name("org", account_id[-6:])
        )
        try:
            org_creds = self.assume_role(
                account_id,
                ORG_ACCESS_ROLE,
                hop1_session_name,
                duration_seconds=duration_seconds,
            )
        except Exception as e:
            logger.error(f"First hop: Failed to assume {ORG_ACCESS_ROLE} in {account_id}: {e}")
            raise e

        # Single-hop case: caller wants the org-access session itself, no chained role.
        if not role_name or role_name == ORG_ACCESS_ROLE:
            if not role_name:
                logger.debug(f"Assuming {ORG_ACCESS_ROLE} as no role was specified for second hop.")
            return org_creds

        # Hop 2: from the member account session, assume the target role
        member_sts = build_client(env_credentials_dict_to_session(org_creds), "sts")

        role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
        logger.debug(f"Chained assume: {role_arn} (session={session_name}).")

        try:
            response = member_sts.assume_role(
                RoleArn=role_arn,
                RoleSessionName=enforce_session_name(session_name),
                DurationSeconds=duration_seconds,
            )
        except Exception as e:
            logger.error(f"Second hop: Failed to assume {role_arn} in {account_id}: {e}")
            raise e

        return assumed_credentials_dict_to_credentials_env(response["Credentials"])

    def wait_for_role(
        self,
        account_id: str,
        role_name: str,
        timeout: int = 180,
        interval: int = 5,
    ) -> None:
        """Block until a role in a member account is assumable.

        Newly created accounts may not have their roles available immediately
        due to IAM eventual consistency. This method polls ``sts:AssumeRole``
        until it succeeds or the timeout is reached.

        Args:
            account_id: Target AWS account ID.
            role_name: IAM role name to wait for.
            timeout: Maximum seconds to wait.
            interval: Seconds between attempts.

        Raises:
            CredentialError: If the role is still not assumable after *timeout*.
        """
        role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
        deadline = time.monotonic() + timeout
        last_error: ClientError | None = None

        while time.monotonic() < deadline:
            # Poll for shutdown: this retry loop can run long on fresh-account IAM lag.
            raise_if_shutdown()
            try:
                self._sts.assume_role(
                    RoleArn=role_arn,
                    RoleSessionName=build_session_name("role-probe"),
                    DurationSeconds=900,
                )
                logger.debug("Role %s is now assumable.", role_arn)
                return
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in (
                    "AccessDenied",
                    "AccessDeniedException",
                ):
                    raise
                last_error = exc
                logger.debug("Role %s not yet assumable, retrying in %ds...", role_arn, interval)
                time.sleep(interval)

        raise CredentialError(f"Role {role_arn} not assumable after {timeout}s: {last_error}")

    def get_session_for_account(
        self,
        account_id: str,
        role_name: str,
        session_name: str,
        region: str = DEFAULT_REGION,
    ) -> boto3.Session:
        """Return a boto3 Session with auto-refreshable credentials for a member account.

        Credentials automatically refresh ~15 minutes before the 1-hour expiry.

        Args:
            account_id: The target AWS account ID.
            role_name: IAM role name to assume.
            session_name: Session name for CloudTrail auditing.
            region: AWS region for the session (default: DEFAULT_REGION).

        Returns:
            A boto3.Session with refreshable credentials that auto-refresh before expiry.
        """
        parent_session = self._session
        preexisting = active_account_config()
        if preexisting is not None:
            config, _ = preexisting
            target_role = self._preexisting_role(account_id, role_name)
            # Already running as the target role — self-assume would fail, so reuse it.
            if self._ambient_is_target_role(account_id, target_role):
                return create_regional_session(self._session, region)
            parent_session = self._session
            if target_role != config.runner_role and not self._ambient_is_target_role(
                account_id, config.runner_role
            ):
                runner_arn = f"arn:aws:iam::{account_id}:role/{config.runner_role}"
                parent_session = _create_refreshable_session(
                    self._session,
                    runner_arn,
                    build_session_name("runner", account_id[-6:]),
                    region,
                )
            role_name = target_role
        role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
        return _create_refreshable_session(
            parent_session,
            role_arn,
            session_name,
            region,
        )
