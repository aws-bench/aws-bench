"""Credential provider for assuming roles into member accounts."""

from __future__ import annotations

import asyncio
import configparser
import json
import logging
import re
import shlex
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import PurePosixPath

import boto3
from botocore.config import Config
from botocore.configloader import build_profile_map
from botocore.credentials import (
    DeferredRefreshableCredentials,
    ReadOnlyCredentials,
    RefreshableCredentials,
    create_assume_role_refresher,
)
from botocore.exceptions import ClientError

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.account_management.exceptions import AccountResolutionError
from aws_bench.account_management.preexisting import active_account_config
from aws_bench.constants import DEFAULT_REGION
from aws_bench.exceptions import CredentialError
from aws_bench.logging.logger import get_logger
from aws_bench.utils.concurrent import build_client, build_session, raise_if_shutdown

logger = get_logger(__name__)

CREDS_DIR = PurePosixPath(".aws/creds")
CREDENTIAL_ENV_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_CREDENTIAL_EXPIRATION",
    "AWS_ACCOUNT_ID",
    "AWS_CONFIG_FILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_CREDENTIAL_FILE",
    "BOTO_CONFIG",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
    "AWS_EC2_METADATA_SERVICE_ENDPOINT",
    "AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE",
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
)
_CRED_REFRESH_SKEW_SEC = 900
CRED_REFRESH_MIN_SLEEP_SEC = 30
_CRED_REFRESH_RETRY_SEC = 60

_ACCOUNT_TAG_MAX_LEN = 32
_ACCOUNT_TAG_RE = re.compile(rf"[A-Za-z][A-Za-z0-9_]{{0,{_ACCOUNT_TAG_MAX_LEN - 1}}}")
_ACCOUNT_TAG_RESERVED = frozenset(CREDENTIAL_ENV_VARS) | {
    "HOME",
    "PATH",
    "MANAGEMENT_ROLE",
    "AWS_EC2_METADATA_DISABLED",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_BEARER_TOKEN_BEDROCK",
}
_PROFILE_CREDENTIAL_SETTINGS = frozenset(
    {
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "aws_security_token",
        "aws_account_id",
        "credential_source",
        "source_profile",
        "role_arn",
        "role_session_name",
        "web_identity_token_file",
        "external_id",
        "mfa_serial",
        "duration_seconds",
        "sso_session",
        "sso_start_url",
        "sso_region",
        "sso_account_id",
        "sso_role_name",
        "login_session",
    }
)

# Default retry policy for every session this provider builds. aws-bench fans
# out many concurrent clients against one account+region, so adaptive mode's
# client-side rate limiter (vs. botocore's legacy default) damps throttle bursts.
_RETRY_DEFAULTS = {"max_attempts": 8, "mode": "adaptive"}


def validate_account_tag(tag: str) -> str:
    """Validate a tag used as a profile, file name and exported account variable."""
    if not _ACCOUNT_TAG_RE.fullmatch(tag):
        raise ValueError(
            "Invalid account_tag: must match "
            f"[A-Za-z][A-Za-z0-9_]{{0,{_ACCOUNT_TAG_MAX_LEN - 1}}} "
            f"(letter prefix; ≤{_ACCOUNT_TAG_MAX_LEN} chars)"
        )
    if tag in _ACCOUNT_TAG_RESERVED:
        raise ValueError(f"Invalid account_tag {tag!r}: reserved control variable")
    return tag


def credential_env(profile: str, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Overlay safe transport values without changing the caller's environment.

    Blanks suppress inherited transport overrides without forwarding old secrets.
    The child must also run :func:`credential_command` to remove those variables.
    """
    validate_account_tag(profile)
    return {
        **(env or {}),
        **dict.fromkeys(CREDENTIAL_ENV_VARS, ""),
        "AWS_PROFILE": profile,
        "AWS_EC2_METADATA_DISABLED": "true",
    }


def credential_command(command: str, profile: str) -> str:
    """Remove inherited credential sources in the child shell before ``command``."""
    validate_account_tag(profile)
    return (
        f"unset {' '.join(CREDENTIAL_ENV_VARS)} || exit $?\n"
        f"export AWS_PROFILE={shlex.quote(profile)} AWS_EC2_METADATA_DISABLED=true || exit $?\n"
        f"{command}"
    )


def _parse_aws_ini(existing: str, file_name: str) -> configparser.RawConfigParser:
    """Parse AWS INI text without exposing its potentially secret contents in errors."""
    parsed = configparser.RawConfigParser()
    try:
        parsed.read_string(existing)
    except configparser.Error:
        raise CredentialError(f"Cannot parse AWS {file_name}; check its INI syntax") from None
    return parsed


def build_aws_config(
    account_tags: Iterable[str], existing: str = "", region: str | None = None
) -> str:
    """Add process profiles while preserving existing text and unrelated settings.

    Existing credential settings for a selected profile are conflicts, except
    for the exact process command this function generates. An optional region
    fills missing region settings only.
    """
    tags = dict.fromkeys(validate_account_tag(tag) for tag in account_tags)
    if region is not None and not re.fullmatch(r"[a-z0-9][a-z0-9-]*", region):
        raise ValueError("Invalid AWS region")
    parsed = _parse_aws_ini(existing, "config")
    sections: dict[str, str] = {}
    for section in parsed.sections():
        # Use the SDK's profile-name rules, including quoted names and "default".
        for tag in build_profile_map({section: {}})["profiles"]:
            if tag in tags:
                if tag in sections:
                    raise CredentialError(f"Multiple AWS config sections select profile {tag!r}")
                sections[tag] = section

    lines = existing.splitlines(keepends=True)
    insertions: dict[int, str] = {}
    new_profiles: list[str] = []
    for tag in tags:
        process = f"""sh -c 'cat "$HOME/{CREDS_DIR}/{tag}.json"'"""
        section = sections.get(tag)
        options = parsed[section] if section is not None else parsed.defaults()
        if _PROFILE_CREDENTIAL_SETTINGS.intersection(options) or (
            "credential_process" in options and options["credential_process"] != process
        ):
            raise CredentialError(f"AWS config profile {tag!r} has conflicting credential settings")
        additions = ""
        if section is None or "credential_process" not in options:
            additions += f"credential_process = {process}\n"
        if region is not None and "region" not in options:
            additions += f"region = {region}\n"
        if section is None:
            header = "default" if tag == "default" else f"profile {tag}"
            new_profiles.append(f"[{header}]\n{additions}")
        elif additions:
            headers = [
                i
                for i, line in enumerate(lines)
                if (match := parsed.SECTCRE.match(line.strip()))
                and match.group("header") == section
            ]
            # An apparent header inside a multiline value is ambiguous to edit.
            if len(headers) != 1:
                raise CredentialError(f"Cannot locate AWS config profile {tag!r} unambiguously")
            # Match the next option or header so it cannot become a continuation.
            indent = ""
            for line in lines[headers[0] + 1 :]:
                stripped = line.lstrip()
                if stripped.strip() and not stripped.startswith(("#", ";")):
                    indent = line[: len(line) - len(stripped)]
                    break
            insertions[headers[0]] = "".join(
                indent + line for line in additions.splitlines(keepends=True)
            )

    for index in sorted(insertions, reverse=True):
        if not lines[index].endswith("\n"):
            lines[index] += "\n"
        lines.insert(index + 1, insertions[index])
    result = "".join(lines)
    if new_profiles:
        if result:
            result += "\n" if result.endswith("\n") else "\n\n"
        result += "\n".join(new_profiles)
    return result


def check_static_profiles(account_tags: Iterable[str], existing: str) -> None:
    """Reject selected profiles in the separate shared credentials file."""
    tags = [validate_account_tag(tag) for tag in account_tags]
    parsed = _parse_aws_ini(existing, "credentials file")
    for tag in tags:
        if parsed.has_section(tag):
            raise CredentialError(f"AWS shared credentials file contains selected profile {tag!r}")


def _validate_expiry(expires_at: object) -> datetime:
    """Require an actual, timezone-aware, future credential expiration."""
    if expires_at is None:
        raise CredentialError("Session credentials have no expiry")
    if not isinstance(expires_at, datetime):
        raise CredentialError("Credential expiration must be a datetime")
    if expires_at.tzinfo is None or expires_at.utcoffset() is None:
        raise CredentialError("Credential expiration must be timezone-aware")
    if expires_at <= datetime.now(timezone.utc):
        raise CredentialError("Credential expiration must be in the future")
    return expires_at


def credential_refresh_delay(expires_at: datetime) -> float:
    """Return the normal refresh delay, rejecting credentials that expire before it."""
    expires_at = _validate_expiry(expires_at)
    remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
    if remaining <= CRED_REFRESH_MIN_SLEEP_SEC:
        raise CredentialError("Credential lifetime must exceed the minimum refresh sleep")
    return max(remaining - _CRED_REFRESH_SKEW_SEC, float(CRED_REFRESH_MIN_SLEEP_SEC))


def credential_deadline(expires_at: datetime) -> float:
    """Convert a valid future expiration to the running loop's monotonic clock."""
    expires_at = _validate_expiry(expires_at)
    return (
        asyncio.get_running_loop().time()
        + (expires_at - datetime.now(timezone.utc)).total_seconds()
    )


async def refresh_credentials_loop(
    expires_at: datetime,
    refresh_once: Callable[[], Awaitable[datetime]],
    log: logging.Logger,
) -> None:
    """Refresh until cancelled or the last published credentials expire.

    ``refresh_once`` mints off the event loop and returns the new expiration
    only after successful publication. The owner also validates the initial
    expiration after publication, before starting a consumer.
    """
    loop = asyncio.get_running_loop()
    deadline = credential_deadline(expires_at)
    delay = credential_refresh_delay(expires_at)
    try:
        async with asyncio.timeout_at(deadline) as guard:
            while True:
                await asyncio.sleep(delay)
                if loop.time() >= deadline:
                    raise TimeoutError
                try:
                    renewed = await refresh_once()
                    delay = credential_refresh_delay(renewed)
                except Exception as exc:
                    # Exception text can contain credentials or credential-process output.
                    log.warning(
                        "Credential refresh failed (%s); retrying in %ds",
                        type(exc).__name__,
                        _CRED_REFRESH_RETRY_SEC,
                    )
                    delay = _CRED_REFRESH_RETRY_SEC
                else:
                    # A blocking publication can finish before the timeout callback runs.
                    if loop.time() >= deadline:
                        raise TimeoutError
                    deadline = credential_deadline(renewed)
                    guard.reschedule(deadline)
    except TimeoutError:
        raise CredentialError(
            "Published AWS credentials expired before refresh completed"
        ) from None


def mint_credentials(
    provider: CredentialProvider,
    account_mapping: Mapping[str, str],
    role_name: str,
    session_name: str,
) -> tuple[dict[str, str], datetime]:
    """Mint process JSON bodies and their earliest expiry without publishing files.

    Call from a worker thread. The lifecycle owner captures its account mapping
    and role at entry and checks the expiration again after initial publication.
    """
    accounts = dict(account_mapping)
    if not accounts:
        raise CredentialError("Cannot mint credentials for an empty account mapping")
    for tag in accounts:
        validate_account_tag(tag)
    session_name = enforce_session_name(session_name)
    files: dict[str, str] = {}
    expiries: list[datetime] = []
    for tag, account_id in accounts.items():
        session = provider.get_chained_session_for_account(account_id, role_name, session_name)
        credentials = session_to_credential_process(session)
        files[f"{tag}.json"] = json.dumps(credentials)
        expiries.append(datetime.fromisoformat(str(credentials["Expiration"])))
    expires_at = min(expiries)
    credential_refresh_delay(expires_at)
    return files, expires_at


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
        ``build_session_name("session")`` -> ``"app-session"``

    Segments must stay neutral: they must not reveal to an evaluated agent (via
    ``sts:GetCallerIdentity`` or CloudTrail) that it is running inside aws-bench.
    Callers use the ``"session"`` token plus opaque identifiers (e.g. an account-id
    tail) only — never a task, benchmark, or operation description.
    """
    return "-".join([SESSION_NAME_PREFIX, *segments])[:MAX_SESSION_NAME_LEN]


def enforce_session_name(session_name: str) -> str:
    """Validate and normalize an STS RoleSessionName for CloudTrail attribution.

    Backstop at the generic STS choke points all assume-role paths funnel
    through: even a hand-written name (not built via :func:`build_session_name`)
    must carry the ``app-`` prefix, so the convention is enforced at runtime.

    Returns the name truncated to STS's 64-char limit (e.g. the
    ``app-session-<job>`` builder drops its job-id tail rather than the prefix).

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
        build_session(
            lambda: boto3.Session(
                aws_access_key_id=creds["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=creds["AWS_SECRET_ACCESS_KEY"],
                aws_session_token=creds["AWS_SESSION_TOKEN"],
                region_name=DEFAULT_REGION,
            )
        )
    )


def _snapshot_credentials(
    session: boto3.Session, *, force_refresh: bool = False
) -> tuple[ReadOnlyCredentials, datetime | None]:
    """Read one credential generation under botocore's lock without translating errors."""
    creds = session.get_credentials()
    if creds is None:
        raise CredentialError("Session has no credentials to snapshot")
    if isinstance(creds, RefreshableCredentials):
        if not force_refresh:
            # This may acquire the same non-reentrant lock; call it outside.
            creds.get_frozen_credentials()
        with creds._refresh_lock:
            if force_refresh:
                creds._protected_refresh(is_mandatory=True)
            return creds._frozen_credentials, creds._expiry_time
    return creds.get_frozen_credentials(), getattr(creds, "_expiry_time", None)


def session_to_env_credentials(session: boto3.Session) -> dict[str, str]:
    """Snapshot a boto3 Session's credentials as an env-var dict.

    Returns a dict with AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and
    AWS_SESSION_TOKEN (empty string if no session token), suitable for
    :func:`env_credentials_dict_to_session`.

    The snapshot is static (no auto-refresh), so for refreshable credentials we
    first force a full re-mint — the recipient (container, worker thread) gets the
    maximum duration, not the aged tail of the parent's current credential.
    Retrieval errors propagate unchanged.

    Raises:
        CredentialError: If the session has no credentials to snapshot.
    """
    frozen, expiry = _snapshot_credentials(session, force_refresh=True)
    if expiry is not None:
        logger.debug("Snapshotted credentials expire at %s", expiry.isoformat())
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
        CredentialError: If retrieval fails or credentials lack a valid future
            expiration (without ``Expiration`` the SDK treats the output as static).
    """
    try:
        frozen, expiry = _snapshot_credentials(session)
    except CredentialError:
        raise
    except Exception as exc:
        raise CredentialError("Failed to retrieve session credentials") from exc
    expiry = _validate_expiry(expiry)
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
        validate_account_tag(tag)
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
    regional_session._session._credentials = parent_session.get_credentials()
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
        self._session = _apply_client_defaults(
            session or build_session(lambda: boto3.Session(region_name=DEFAULT_REGION))
        )
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
        return build_client(self._session, "sts")

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
        # STS omits the IAM role path from an assumed-role identity ARN.
        role = role_name.rsplit("/", 1)[-1]
        marker = f"arn:aws:sts::{account_id}:assumed-role/{role}"
        arn_role, _, arn_session = arn.rpartition("/")
        return arn_role == marker and bool(arn_session)

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
                    build_session_name("session", account_id[-6:]),
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
            else build_session_name("session", account_id[-6:])
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
                    RoleSessionName=build_session_name("session"),
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
        """Return a session for the requested account role.

        Managed accounts assume the role directly from the host session.
        Preexisting accounts enforce the allowlist and use their configured runner,
        reusing matching ambient credentials when possible.

        New role sessions refresh ~15 minutes before their one-hour expiry.
        Reused ambient credentials retain their original expiry and renewal
        capabilities; reuse does not extend their lifetime.

        Args:
            account_id: The target AWS account ID.
            role_name: IAM role name to assume.
            session_name: Session name for CloudTrail auditing.
            region: AWS region for the session (default: DEFAULT_REGION).

        Returns:
            A boto3.Session for the requested role and region.
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
                    build_session_name("session", account_id[-6:]),
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

    def get_chained_session_for_account(
        self,
        account_id: str,
        role_name: str | None,
        session_name: str,
        region: str = DEFAULT_REGION,
    ) -> boto3.Session:
        """Return a session through the account's role chain.

        Managed accounts assume the organization access role first, then the
        requested role when different. None selects only the organization role.
        Preexisting accounts retain their runner role, allowlist and ambient
        session reuse through ``get_session_for_account``.

        New role sessions renew before expiry. Reused ambient credentials retain
        their original expiry and renewal capabilities; reuse does not extend them.
        """
        session_name = enforce_session_name(session_name)
        if active_account_config() is not None:
            return self.get_session_for_account(
                account_id, role_name or ORG_ACCESS_ROLE, session_name, region
            )
        chained = role_name not in (None, ORG_ACCESS_ROLE)
        parent = self.get_session_for_account(
            account_id,
            ORG_ACCESS_ROLE,
            build_session_name("session", account_id[-6:]) if chained else session_name,
            region,
        )
        if not chained:
            return parent
        return _create_refreshable_session(
            parent, f"arn:aws:iam::{account_id}:role/{role_name}", session_name, region
        )
