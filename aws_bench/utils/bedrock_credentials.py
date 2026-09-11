"""Bedrock service-specific credential management.

Generates long-term Bedrock API keys via IAM service-specific credentials,
cached in SSM Parameter Store.
"""

import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

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


def _verify_token(api_key: str, retries: int = 3, delay: int = 5) -> bool:
    """Verify the token works with a lightweight Bedrock API call."""
    url = "https://bedrock.us-east-1.amazonaws.com/foundation-models/amazon.titan-embed-text-v1"
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.HTTPError as e:
            if e.code == 403 and attempt < retries - 1:
                logger.info(f"Got 403, retrying in {delay}s (IAM propagation delay)...")
                time.sleep(delay)
                continue
            logger.error(f"Token verification failed: HTTP {e.code} - {e.reason}")
            break
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            break
    return False


def generate_bearer_token(
    *,
    force: bool = False,
    no_verify: bool = False,
    days: int = DEFAULT_DAYS,
    min_remaining_days: int = DEFAULT_MIN_REMAINING_DAYS,
) -> str:
    """Generate or retrieve a cached Bedrock bearer token.

    Args:
        force: Delete existing credentials and generate fresh.
        no_verify: Skip token verification against Bedrock API.
        days: Credential lifetime in days.
        min_remaining_days: Minimum remaining days to consider a credential reusable.

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
            logger.info(f"Found token in SSM ({SSM_PARAMETER}), verifying...")
            # Full retry budget: a credential 403s until IAM propagates, which
            # routinely outlasts a single interval. Reading that as failure would
            # discard a valid token and delete the credential backing it.
            if no_verify or _verify_token(cached_token):
                logger.info("Reusing valid token from SSM.")
                return cached_token
            logger.info("Token from SSM failed verification. Generating a new one...")
            need_regenerate = True
        else:
            logger.info(f"No token found in SSM ({SSM_PARAMETER}).")
            need_regenerate = True

    # Step 3: Rotate - delete expiring/invalid/force credentials
    if credential_expiring or force or need_regenerate:
        if existing:
            _delete_all_credentials(iam_client, IAM_USER_NAME, existing)
            existing = []

    # Step 4: Generate a new credential
    logger.info(f"Generating long-term Bedrock key (valid {days} days)...")
    try:
        response = iam_client.create_service_specific_credential(
            UserName=IAM_USER_NAME, ServiceName=SERVICE_NAME, CredentialAgeDays=days
        )
        credential = response["ServiceSpecificCredential"]
        api_key = credential["ServiceCredentialSecret"]
        logger.info(f"Credential ID: {credential['ServiceSpecificCredentialId']}")
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "LimitExceeded":
            raise BedrockCredentialError(
                "Limit exceeded: max 2 service-specific credentials per user per service. "
                "Use --force to delete existing credentials and create a fresh one."
            ) from e
        raise BedrockCredentialError(f"Failed to generate key: {e}") from e

    # Step 5: Verify and store in SSM
    if not no_verify:
        logger.info("Verifying token against Bedrock API...")
        if not _verify_token(api_key):
            raise BedrockCredentialError(
                "Token verification failed - the key may not be usable yet."
            )
        logger.info("Token verified successfully.")

    _store_token_in_ssm(ssm_client, SSM_PARAMETER, api_key)
    return api_key
