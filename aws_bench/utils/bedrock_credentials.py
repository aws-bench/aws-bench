"""Bedrock service-specific credential management.

Generates long-term Bedrock API keys via IAM service-specific credentials,
cached in SSM Parameter Store.
"""

import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from enum import Enum

import boto3
from botocore.exceptions import ClientError

from aws_bench.logging.logger import get_logger

logger = get_logger(__name__)

IAM_USER_NAME = "bedrock-api-user"
SERVICE_NAME = "bedrock.amazonaws.com"
POLICY_ARN = "arn:aws:iam::aws:policy/AmazonBedrockFullAccess"
SSM_REGION = "us-east-1"
SSM_PARAMETER = "/bedrock-aws-bench/bedrock-api-key"

DEFAULT_DAYS = 30
DEFAULT_MIN_REMAINING_DAYS = 1

# HTTP statuses that indicate a transient condition on the verification probe
# (throttling / server-side load), not a statement about the key's validity.
_TRANSIENT_HTTP_CODES = frozenset({429, 500, 502, 503, 504})


class TokenVerification(Enum):
    """Outcome of probing a bearer token against the Bedrock API.

    The three states exist so the caller can distinguish a key that is actually
    bad from a probe that merely failed under load. Rotating on the latter
    deletes the shared credential and 403s every concurrent run holding it.
    """

    VALID = "valid"  # HTTP 200 — the key authenticated.
    INVALID = "invalid"  # HTTP 403 after the IAM-propagation retry budget.
    INDETERMINATE = "indeterminate"  # 429/5xx/network/timeout — validity unknown.


class BedrockCredentialError(Exception):
    """Raised when credential generation or retrieval fails."""


def _ensure_iam_user(iam_client, user_name: str) -> None:
    """Ensure the IAM user exists and has the Bedrock policy attached."""
    try:
        iam_client.create_user(UserName=user_name)
        logger.info(f"Created IAM user: {user_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
    iam_client.attach_user_policy(UserName=user_name, PolicyArn=POLICY_ARN)


def _get_existing_credentials(iam_client, user_name: str) -> list[dict]:
    """List existing service-specific credentials for Bedrock."""
    response = iam_client.list_service_specific_credentials(
        UserName=user_name, ServiceName=SERVICE_NAME
    )
    return response.get("ServiceSpecificCredentials", [])


def _find_reusable_credential(credentials: list[dict], min_remaining_days: int) -> dict | None:
    """Return the first active credential with enough remaining life, or None."""
    now = datetime.now(timezone.utc)
    min_expiration = now + timedelta(days=min_remaining_days)
    for cred in credentials:
        if cred["Status"] != "Active":
            continue
        expiration = cred.get("ExpirationDate")
        if expiration and expiration <= min_expiration:
            continue
        return cred
    return None


def _delete_all_credentials(iam_client, user_name: str, credentials: list[dict]) -> None:
    """Delete all existing service-specific credentials for the user."""
    for cred in credentials:
        cred_id = cred["ServiceSpecificCredentialId"]
        iam_client.delete_service_specific_credential(
            UserName=user_name, ServiceSpecificCredentialId=cred_id
        )
        logger.info(f"Deleted existing credential: {cred_id}")


def _get_token_from_ssm(ssm_client, parameter_name: str) -> str | None:
    """Read the token from SSM Parameter Store, or None if not found."""
    try:
        response = ssm_client.get_parameter(Name=parameter_name, WithDecryption=True)
        return response["Parameter"]["Value"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "ParameterNotFound":
            return None
        raise


def _store_token_in_ssm(ssm_client, parameter_name: str, api_key: str) -> None:
    """Store the token in SSM Parameter Store as a SecureString."""
    ssm_client.put_parameter(
        Name=parameter_name,
        Value=api_key,
        Type="SecureString",
        Overwrite=True,
        Description="Long-term Bedrock API key managed by aws-bench env creds",
    )
    logger.info(f"Token stored in SSM: {parameter_name}")


def _verify_token(api_key: str, retries: int = 3, delay: int = 5) -> TokenVerification:
    """Probe the token with a lightweight Bedrock API call.

    Returns a tri-state so the caller never rotates a valid, in-use key on a
    transient blip:

    - ``VALID``: authenticated (HTTP 200).
    - ``INVALID``: rejected for auth reasons (HTTP 403) even after the
      IAM-propagation retry budget. Safe to rotate.
    - ``INDETERMINATE``: the probe hit a transient condition (throttling, 5xx,
      other non-auth HTTP status, network/timeout). The key's validity is
      unknown; the caller MUST NOT rotate on this.

    403 (propagation) and transient conditions are both retried with a fixed
    delay; the final state reflects the last attempt.
    """
    url = "https://bedrock.us-east-1.amazonaws.com/foundation-models/amazon.titan-embed-text-v1"
    last_outcome = TokenVerification.INDETERMINATE
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return TokenVerification.VALID
                # A non-200 without an exception is unusual and not an auth
                # signal; treat as inconclusive rather than rotating.
                last_outcome = TokenVerification.INDETERMINATE
                break
        except urllib.error.HTTPError as e:
            if e.code == 403:
                last_outcome = TokenVerification.INVALID
                if attempt < retries - 1:
                    logger.info(f"Got 403, retrying in {delay}s (IAM propagation delay)...")
                    time.sleep(delay)
                    continue
                logger.error(f"Token verification failed: HTTP 403 - {e.reason}")
                break
            if e.code in _TRANSIENT_HTTP_CODES:
                last_outcome = TokenVerification.INDETERMINATE
                if attempt < retries - 1:
                    logger.warning(
                        f"Transient HTTP {e.code} verifying token; retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    continue
                logger.warning(f"Token verification inconclusive after retries: HTTP {e.code}")
                break
            # Other HTTP errors (e.g. 400/404) are not auth failures.
            logger.warning(f"Token verification inconclusive: HTTP {e.code} - {e.reason}")
            last_outcome = TokenVerification.INDETERMINATE
            break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_outcome = TokenVerification.INDETERMINATE
            if attempt < retries - 1:
                logger.warning(
                    f"Transient network error verifying token ({e}); retrying in {delay}s..."
                )
                time.sleep(delay)
                continue
            logger.warning(f"Token verification inconclusive after retries (network): {e}")
            break
    return last_outcome


def _expiration_key(cred: dict) -> datetime:
    """Sort key for choosing which credential to retire first (soonest expiry)."""
    expiration = cred.get("ExpirationDate")
    return expiration or datetime.max.replace(tzinfo=timezone.utc)


def _rotate_credential(
    iam_client,
    ssm_client,
    existing: list[dict],
    *,
    days: int,
    no_verify: bool,
    reclaim_slot: bool = False,
) -> str:
    """Mint a replacement credential gracefully and return its bearer token.

    Rotation is ordered mint -> verify -> store-in-SSM -> delete-old, so a run
    already holding the old key is not invalidated before the new key is proven
    and published. The old credential is retired only after the new one is in
    SSM.

    IAM caps a user at 2 service-specific credentials per service. When both
    slots are full a rotation cannot mint without first deleting an existing
    credential. That delete is destructive and may surprise a caller who
    manages ``bedrock-api-user`` themselves, so it is opt-in: only when
    ``reclaim_slot`` is set do we free the soonest-to-expire credential.
    Otherwise we refuse and tell the caller how to authorize it.
    """
    if len(existing) >= 2:
        if not reclaim_slot:
            raise BedrockCredentialError(
                f"Both service-specific credential slots for IAM user '{IAM_USER_NAME}' are in "
                "use (IAM allows a maximum of 2 per user/service), so rotating requires deleting "
                "one of them. Re-run with --reclaim-slot to authorize deleting the "
                "soonest-to-expire credential, or --force to replace all existing credentials."
            )
        to_free = min(existing, key=_expiration_key)
        logger.warning(
            "Both credential slots in use; --reclaim-slot set, freeing soonest-to-expire "
            f"credential {to_free['ServiceSpecificCredentialId']} to rotate."
        )
        _delete_all_credentials(iam_client, IAM_USER_NAME, [to_free])
        existing = [c for c in existing if c is not to_free]

    logger.info(f"Generating long-term Bedrock key (valid {days} days)...")
    try:
        response = iam_client.create_service_specific_credential(
            UserName=IAM_USER_NAME, ServiceName=SERVICE_NAME, CredentialAgeDays=days
        )
        credential = response["ServiceSpecificCredential"]
        api_key = credential["ServiceCredentialSecret"]
        new_id = credential["ServiceSpecificCredentialId"]
        logger.info(f"Credential ID: {new_id}")
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "LimitExceeded":
            raise BedrockCredentialError(
                "Limit exceeded: max 2 service-specific credentials per user per service. "
                "Use --force to delete existing credentials and create a fresh one."
            ) from e
        raise BedrockCredentialError(f"Failed to generate key: {e}") from e

    # Verify the NEW key. A transient probe failure is not fatal for a key IAM
    # just minted; only a hard 403 after the propagation window is.
    if not no_verify:
        logger.info("Verifying new token against Bedrock API...")
        outcome = _verify_token(api_key)
        if outcome is TokenVerification.INVALID:
            # New key is unusable — roll it back and keep the old credential(s)
            # intact so nothing that still works is destroyed.
            _delete_all_credentials(
                iam_client, IAM_USER_NAME, [{"ServiceSpecificCredentialId": new_id}]
            )
            raise BedrockCredentialError(
                "New token failed verification - the key may not be usable yet."
            )
        if outcome is TokenVerification.INDETERMINATE:
            logger.warning("New token verification inconclusive (transient); storing anyway.")
        else:
            logger.info("Token verified successfully.")

    # Publish the new key first, THEN retire the old credentials.
    _store_token_in_ssm(ssm_client, SSM_PARAMETER, api_key)
    if existing:
        _delete_all_credentials(iam_client, IAM_USER_NAME, existing)
    return api_key


def generate_bearer_token(
    *,
    force: bool = False,
    no_verify: bool = False,
    days: int = DEFAULT_DAYS,
    min_remaining_days: int = DEFAULT_MIN_REMAINING_DAYS,
    reclaim_slot: bool = False,
) -> str:
    """Generate or retrieve a cached Bedrock bearer token.

    Args:
        force: Delete existing credentials and generate fresh.
        no_verify: Skip token verification against Bedrock API.
        days: Credential lifetime in days.
        min_remaining_days: Minimum remaining days to consider a credential reusable.
        reclaim_slot: When both service-specific credential slots are in use and a
            rotation is required, authorize deleting the soonest-to-expire
            credential to free a slot. Off by default so a caller who manages
            ``bedrock-api-user`` themselves is never surprised by a deletion;
            ``force`` implies it.

    Returns:
        The bearer token string.

    Raises:
        BedrockCredentialError: If credential generation or verification fails.
    """
    session = boto3.Session(region_name=SSM_REGION)
    iam_client = session.client("iam")
    ssm_client = session.client("ssm")

    # Step 1: Ensure IAM user exists and check credential expiration
    _ensure_iam_user(iam_client, IAM_USER_NAME)
    existing = _get_existing_credentials(iam_client, IAM_USER_NAME)
    credential_expiring = False

    if existing and not force:
        reusable = _find_reusable_credential(existing, min_remaining_days)
        if not reusable:
            credential_expiring = True
            logger.info(f"Credential expiring within {min_remaining_days}d - will rotate.")

    # Step 2: Try to reuse the token from SSM (fast path)
    need_regenerate = False
    if not force and not credential_expiring:
        cached_token = _get_token_from_ssm(ssm_client, SSM_PARAMETER)
        if cached_token:
            if no_verify:
                logger.info("Reusing token from SSM (verification skipped).")
                return cached_token
            logger.info(f"Found token in SSM ({SSM_PARAMETER}), verifying...")
            outcome = _verify_token(cached_token)
            if outcome is TokenVerification.VALID:
                logger.info("Reusing valid token from SSM.")
                return cached_token
            if outcome is TokenVerification.INDETERMINATE:
                # A probe that failed transiently (throttling / 5xx / network)
                # says nothing about the token's validity. Reuse it WITHOUT
                # rotating: rotation deletes the shared credential and 403s every
                # concurrent run already holding this key.
                logger.warning(
                    "Token verification inconclusive (transient failure); "
                    "reusing cached token without rotation."
                )
                return cached_token
            # INVALID: a hard 403 after the propagation window - the key really
            # is bad, so rotation is warranted.
            logger.info("Token from SSM rejected by Bedrock (403). Rotating...")
            need_regenerate = True
        else:
            logger.info(f"No token found in SSM ({SSM_PARAMETER}).")
            need_regenerate = True

    # Steps 3-5: Rotate gracefully (mint -> verify -> store -> delete-old) so a
    # run holding the old key is not invalidated before the new key is proven.
    if credential_expiring or force or need_regenerate:
        return _rotate_credential(
            iam_client,
            ssm_client,
            existing,
            days=days,
            no_verify=no_verify,
            reclaim_slot=reclaim_slot or force,
        )

    # Unreachable: one of reuse / rotate always returns above. Guard for safety.
    raise BedrockCredentialError("Unable to obtain a Bedrock bearer token.")
