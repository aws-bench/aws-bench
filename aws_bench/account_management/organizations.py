"""Thin wrapper around the boto3 Organizations client.

Manages AWS Organizations resources including Organizational Units (OUs),
member accounts, and tagging.
"""

from __future__ import annotations

import asyncio
import json

from botocore.exceptions import ClientError
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_result,
    stop_after_delay,
    wait_exponential,
    wait_random_exponential,
)

from aws_bench.account_management.constants import (
    CFN_OPS_ROLE_NAME,
    ORG_ACCESS_ROLE,
    POLL_TIMEOUT_SEC,
)
from aws_bench.account_management.exceptions import (
    AccountCreationError,
    AccountCreationTimeoutError,
    NotManagementAccountError,
    OrganizationNotReadyError,
)
from aws_bench.account_management.models import OrgInfo
from aws_bench.account_management.utils import raise_account_creation_timeout
from aws_bench.logging.logger import get_logger
from aws_bench.utils.credentials_provider import CredentialProvider

logger = get_logger(__name__)


def _is_policy_type_not_enabled_error(exc: BaseException) -> bool:
    """Return True if the exception is a PolicyTypeNotEnabledException."""
    return (
        isinstance(exc, ClientError)
        and exc.response["Error"]["Code"] == "PolicyTypeNotEnabledException"
    )


def _is_concurrent_modification_error(exc: BaseException) -> bool:
    """Return True if the exception is a ConcurrentModificationException.

    AWS Organizations serializes mutations on the same entity and rejects
    racing callers with this transient error; safe to retry with backoff.
    """
    return (
        isinstance(exc, ClientError)
        and exc.response["Error"]["Code"] == "ConcurrentModificationException"
    )


class OrganizationsClient:
    """Wrapper around the boto3 Organizations client."""

    def __init__(self) -> None:
        """Initialize the organizations client."""
        self._credentials_provider = CredentialProvider.get()
        self._client = self._credentials_provider.session.client("organizations")

    # ── Org info ──

    def get_org_info(self) -> OrgInfo:
        """Fetch OrgInfo from the Organizations API."""
        raw = self._client.describe_organization()["Organization"]
        roots = self._client.list_roots()["Roots"]
        if not roots:
            raise OrganizationNotReadyError("No organization roots found.")
        return OrgInfo(
            org_id=raw["Id"],
            root_id=roots[0]["Id"],
            management_account_id=raw["MasterAccountId"],
            management_account_email=raw["MasterAccountEmail"],
        )

    # ── Organization lifecycle ──

    def create_organization(self) -> None:
        """Enable Organizations with all features. Idempotent.

        Raises:
            NotManagementAccountError: If the account is a member of an org it doesn't manage.
        """
        try:
            self._client.create_organization(FeatureSet="ALL")
            logger.info("Organization created.")
        except ClientError as e:
            if e.response["Error"]["Code"] == "AlreadyInOrganizationException":
                logger.debug("AWS Organizations already enabled for this account.")
                self._verify_is_management_account()
            else:
                raise

    def _verify_is_management_account(self) -> None:
        """Verify the caller is the management account of the org."""
        org = self._client.describe_organization()["Organization"]
        caller_id = self._credentials_provider.get_caller_account_id()
        if caller_id != org["MasterAccountId"]:
            raise NotManagementAccountError(
                f"Account {caller_id} is a member of organization "
                f"{org['Id']}, but the management account is "
                f"{org['MasterAccountId']}. "
                f"aws-bench requires the management account."
            )

    # ── Organizational Unit (OU) operations ──

    def get_tags(self, resource_id: str) -> dict[str, str]:
        """Get all tags for an Organizations resource."""
        tags: dict[str, str] = {}
        paginator = self._client.get_paginator("list_tags_for_resource")
        for page in paginator.paginate(ResourceId=resource_id):
            for tag in page["Tags"]:
                tags[tag["Key"]] = tag["Value"]
        return tags

    def list_ous(self, root_id: str) -> list[dict]:
        """List all Organizational Units directly under a parent."""
        ous: list[dict] = []
        paginator = self._client.get_paginator("list_organizational_units_for_parent")
        for page in paginator.paginate(ParentId=root_id):
            ous.extend(page["OrganizationalUnits"])
        return ous

    def find_ou_by_name(self, root_id: str, name: str) -> str | None:
        """Find an Organizational Unit (OU) under root by its name.

        Returns:
            The OU ID if found, None otherwise.
        """
        paginator = self._client.get_paginator("list_organizational_units_for_parent")
        for page in paginator.paginate(ParentId=root_id):
            for ou in page["OrganizationalUnits"]:
                if ou["Name"] == name:
                    return ou["Id"]
        return None

    def create_ou(self, root_id: str, name: str) -> str:
        """Create an Organizational Unit (OU) under root.

        Returns:
            The new OU ID.
        """
        result = self._client.create_organizational_unit(ParentId=root_id, Name=name)
        ou_id = result["OrganizationalUnit"]["Id"]
        logger.info(f"Created testing environment OU '{name}' ({ou_id}).")
        return ou_id

    # ── Account operations ──

    def list_accounts_in_ou(self, ou_id: str) -> list[dict]:
        """List all accounts under an OU."""
        accounts: list[dict] = []
        paginator = self._client.get_paginator("list_accounts_for_parent")
        for page in paginator.paginate(ParentId=ou_id):
            accounts.extend(page["Accounts"])
        return accounts

    async def create_account(self, name: str, email: str) -> str:
        """Create a member account and poll until ready.

        Returns:
            The new account ID.
        """
        logger.info(f"Creating account '{name}' ({email})...")
        request_id = await self._submit_create_account(name, email)
        result = await self._poll_account_creation(request_id, name)
        if result is None:
            raise AccountCreationTimeoutError(
                f"Account creation timed out after {POLL_TIMEOUT_SEC}s for '{name}'"
            )
        return result

    @retry(
        retry=retry_if_exception(_is_concurrent_modification_error),
        wait=wait_random_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_delay(180),
        reraise=True,
    )
    async def _submit_create_account(self, name: str, email: str) -> str:
        """Submit CreateAccount, retrying only transient Organizations write-lock conflicts."""
        response = await asyncio.to_thread(
            self._client.create_account, Email=email, AccountName=name
        )
        request_id: str = response["CreateAccountStatus"]["Id"]
        return request_id

    @retry(
        wait=wait_exponential(multiplier=1, min=3, max=15),
        stop=stop_after_delay(POLL_TIMEOUT_SEC),
        retry=retry_if_result(lambda r: r is None),
        retry_error_callback=raise_account_creation_timeout,
    )
    async def _poll_account_creation(self, request_id: str, name: str) -> str | None:
        """Poll CreateAccount status with exponential backoff.

        Returns:
            Account ID on success, None if still in progress (triggers retry).
        """
        status = await asyncio.to_thread(
            self._client.describe_create_account_status,
            CreateAccountRequestId=request_id,
        )
        state = status["CreateAccountStatus"]

        if state["State"] == "SUCCEEDED":
            account_id: str = state["AccountId"]
            logger.info(f"Account created: {account_id}")
            return account_id
        elif state["State"] == "FAILED":
            reason = state.get("FailureReason", "UNKNOWN")
            raise AccountCreationError(f"CreateAccount failed for '{name}': {reason}")

        # Still in progress
        return None

    @retry(
        retry=retry_if_exception(_is_concurrent_modification_error),
        wait=wait_random_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_delay(180),
        reraise=True,
    )
    async def move_account_to_ou(self, account_id: str, ou_id: str) -> None:
        """Move an account into the given OU."""
        parents = await asyncio.to_thread(self._client.list_parents, ChildId=account_id)
        current_parent = parents["Parents"][0]["Id"]
        if current_parent == ou_id:
            return

        await asyncio.to_thread(
            self._client.move_account,
            AccountId=account_id,
            SourceParentId=current_parent,
            DestinationParentId=ou_id,
        )
        logger.info(f"Moved account {account_id} to OU {ou_id}.")

    async def tag_resource(self, resource_id: str, key: str, value: str) -> None:
        """Tag an Organizations resource."""
        await asyncio.to_thread(
            self._client.tag_resource,
            ResourceId=resource_id,
            Tags=[{"Key": key, "Value": value}],
        )

    async def untag_resource(self, resource_id: str, tag_keys: list[str]) -> None:
        """Remove tags from an Organizations resource."""
        await asyncio.to_thread(
            self._client.untag_resource,
            ResourceId=resource_id,
            TagKeys=tag_keys,
        )

    # ── Service Control Policies ──

    SCP_NAME = "awsbench-protect-org-access-role"
    SCP_DESCRIPTION = "Prevent child accounts from modifying framework-managed roles"

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_delay(POLL_TIMEOUT_SEC),
        retry=retry_if_exception(_is_policy_type_not_enabled_error),
    )
    def ensure_org_role_protection_scp(self, ou_id: str) -> None:
        """Create/update and attach the SCP protecting framework-managed roles.

        Reconciles existing policies: if the content diverges from the desired
        state (e.g. a new role was added), the policy is updated in place.
        """
        self._enable_scp_policy_type()

        desired_content = self._build_role_protection_policy(ORG_ACCESS_ROLE)
        policy_id = self._find_scp_by_name(self.SCP_NAME)
        if policy_id is None:
            policy_id = self._create_org_role_scp(desired_content)
            logger.info(f"Created SCP '{self.SCP_NAME}' ({policy_id})")
        else:
            current = self._client.describe_policy(PolicyId=policy_id)
            if json.loads(current["Policy"]["Content"]) != json.loads(desired_content):
                self._client.update_policy(PolicyId=policy_id, Content=desired_content)
                logger.info(f"Updated SCP '{self.SCP_NAME}' with current role set")
            else:
                logger.debug(f"SCP '{self.SCP_NAME}' already up to date ({policy_id})")

        if self._is_policy_attached(policy_id, ou_id):
            logger.debug(f"SCP already attached to OU {ou_id}")
            return

        self._attach_policy_with_retry(policy_id, ou_id)
        logger.info(f"Attached SCP {policy_id} to OU {ou_id}")

    @retry(
        retry=retry_if_exception(_is_policy_type_not_enabled_error),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_delay(60),
        reraise=True,
    )
    def _attach_policy_with_retry(self, policy_id: str, target_id: str) -> None:
        """Attach policy, retrying on PolicyTypeNotEnabledException for propagation."""
        try:
            self._client.attach_policy(PolicyId=policy_id, TargetId=target_id)
        except ClientError as e:
            if e.response["Error"]["Code"] == "DuplicatePolicyAttachmentException":
                return
            raise

    def _enable_scp_policy_type(self) -> None:
        """Enable SERVICE_CONTROL_POLICY on the org root. Idempotent.

        AWS enables the policy type asynchronously, so we poll until the root
        reports it ENABLED before returning. Without this wait the subsequent
        CreatePolicy/AttachPolicy calls can race and fail with
        PolicyTypeNotEnabledException.
        """
        org_info = self.get_org_info()
        try:
            self._client.enable_policy_type(
                RootId=org_info.root_id,
                PolicyType="SERVICE_CONTROL_POLICY",
            )
            logger.info("Enabled SERVICE_CONTROL_POLICY on org root")
        except ClientError as e:
            if e.response["Error"]["Code"] == "PolicyTypeAlreadyEnabledException":
                logger.debug("SERVICE_CONTROL_POLICY already enabled on org root")
            else:
                raise

        self._wait_for_scp_enabled(org_info.root_id)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_delay(POLL_TIMEOUT_SEC),
        retry=retry_if_result(lambda enabled: not enabled),
    )
    def _wait_for_scp_enabled(self, root_id: str) -> bool:
        """Poll until SERVICE_CONTROL_POLICY is ENABLED on the org root.

        Returns:
            True once the policy type is active (stops the retry); False while
            still pending (triggers another poll).
        """
        for root in self._client.list_roots()["Roots"]:
            if root["Id"] != root_id:
                continue
            for pt in root.get("PolicyTypes", []):
                if pt.get("Type") == "SERVICE_CONTROL_POLICY" and pt.get("Status") == "ENABLED":
                    logger.debug("SERVICE_CONTROL_POLICY is ENABLED on org root")
                    return True
        logger.debug("Waiting for SERVICE_CONTROL_POLICY to become ENABLED...")
        return False

    def _find_scp_by_name(self, name: str) -> str | None:
        """Find an SCP by name. Returns policy ID or None."""
        paginator = self._client.get_paginator("list_policies")
        for page in paginator.paginate(Filter="SERVICE_CONTROL_POLICY"):
            for policy in page["Policies"]:
                if policy["Name"] == name:
                    return policy["Id"]
        return None

    def _is_policy_attached(self, policy_id: str, target_id: str) -> bool:
        """Check if a policy is already attached to a target."""
        try:
            targets = self._client.list_targets_for_policy(PolicyId=policy_id)
            return any(t["TargetId"] == target_id for t in targets["Targets"])
        except ClientError:
            return False

    @staticmethod
    def _build_role_protection_policy(role_name: str) -> str:
        """Return the JSON policy document protecting framework-managed roles.

        The org-access role is exempted from the deny: SCPs bind every principal
        inside a member account (including assumed org-access sessions — the
        management-account exemption does not survive the role hop), and the
        framework manages the CFN ops role through exactly those sessions.
        Agents never hold org-access credentials, so they stay blocked.
        """
        policy_doc = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "ProtectFrameworkRoles",
                    "Effect": "Deny",
                    "Action": [
                        "iam:DeleteRole",
                        "iam:UpdateRole",
                        "iam:DeleteRolePolicy",
                        "iam:PutRolePolicy",
                        "iam:AttachRolePolicy",
                        "iam:DetachRolePolicy",
                        "iam:UpdateAssumeRolePolicy",
                    ],
                    "Resource": [
                        f"arn:aws:iam::*:role/{role_name}",
                        f"arn:aws:iam::*:role/{CFN_OPS_ROLE_NAME}",
                    ],
                    "Condition": {
                        "ArnNotLike": {
                            "aws:PrincipalArn": f"arn:aws:iam::*:role/{role_name}",
                        }
                    },
                }
            ],
        }
        return json.dumps(policy_doc)

    def _create_org_role_scp(self, content: str) -> str:
        """Create the SCP from a pre-built policy document."""
        response = self._client.create_policy(
            Content=content,
            Description=self.SCP_DESCRIPTION,
            Name=self.SCP_NAME,
            Type="SERVICE_CONTROL_POLICY",
        )
        return response["Policy"]["PolicySummary"]["Id"]

    # ── Region Restriction SCP ──

    REGION_RESTRICTION_SCP_PREFIX = "awsbench-region-restrict"

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_delay(POLL_TIMEOUT_SEC),
        retry=retry_if_exception(_is_policy_type_not_enabled_error),
    )
    def ensure_region_restriction_scp(
        self, scenario_name: str, allowed_regions: list[str], account_ids: list[str]
    ) -> None:
        """Create a per-scenario SCP and attach it to each account.

        Idempotent — reuses existing policy by name, updates content if
        policy content changed, and skips accounts that already have it attached.
        """
        scp_name = f"{self.REGION_RESTRICTION_SCP_PREFIX}-{scenario_name}"
        self._enable_scp_policy_type()

        policy_id = self._find_scp_by_name(scp_name)
        desired_content = self._build_region_restriction_policy(allowed_regions)

        if policy_id is None:
            policy_id = self._create_region_restriction_scp(scp_name, desired_content)
            logger.info(f"Created SCP '{scp_name}' ({policy_id})")
        else:
            logger.debug(f"SCP '{scp_name}' already exists ({policy_id})")
            current = self._client.describe_policy(PolicyId=policy_id)
            if json.loads(current["Policy"]["Content"]) != json.loads(desired_content):
                self._client.update_policy(PolicyId=policy_id, Content=desired_content)
                logger.info(f"Updated SCP '{scp_name}' policy content")

        for account_id in account_ids:
            if self._is_policy_attached(policy_id, account_id):
                logger.debug(f"SCP already attached to account {account_id}")
                continue
            self._attach_policy_with_retry(policy_id, account_id)
            logger.info(f"Attached SCP {policy_id} to account {account_id}")

    def _build_region_restriction_policy(self, allowed_regions: list[str]) -> str:
        """Return the JSON string of the region restriction policy document."""
        policy_doc = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "DenyOutOfScopeRegions",
                    "Effect": "Deny",
                    # This is mostly to unblock data plane APIs of some global services
                    "NotAction": [
                        # Identity & access (global, always us-east-1)
                        "iam:*",
                        "sts:*",
                        "organizations:*",
                        # Region opt-in uses the global Account Management endpoint
                        "account:GetRegionOptStatus",
                        "account:EnableRegion",
                        # Edge / network (global, fixed region endpoints)
                        "route53:*",
                        "route53domains:*",
                        "cloudfront:*",
                        "globalaccelerator:*",
                        "networkmanager:*",
                        # Security (global, always us-east-1)
                        "shield:*",
                        "wafv2:*",
                        "waf:*",
                        # Billing & observability (global, always us-east-1)
                        "budgets:*",
                        "ce:*",
                        "support:*",
                        "health:*",
                    ],
                    "Resource": "*",
                    "Condition": {
                        "StringNotEquals": {
                            "aws:RequestedRegion": sorted(allowed_regions),
                        }
                    },
                },
                {
                    "Sid": "DenyOutOfScopeRegionOptIn",
                    "Effect": "Deny",
                    "Action": "account:EnableRegion",
                    "Resource": "*",
                    # The global endpoint's RequestedRegion is not the opt-in target.
                    "Condition": {
                        "StringNotEquals": {
                            "account:TargetRegion": sorted(allowed_regions),
                        }
                    },
                },
            ],
        }
        return json.dumps(policy_doc)

    def _create_region_restriction_scp(self, scp_name: str, content: str) -> str:
        """Create a region restriction SCP. Returns the policy ID."""
        response = self._client.create_policy(
            Content=content,
            Description=f"Deny actions outside allowed regions for {scp_name}",
            Name=scp_name,
            Type="SERVICE_CONTROL_POLICY",
        )
        return response["Policy"]["PolicySummary"]["Id"]

    # ── Terminate / Teardown ──

    def close_account(self, account_id: str) -> None:
        """Close a member account. Enters 90-day suspension period."""
        self._client.close_account(AccountId=account_id)
        logger.info(f"Closed account {account_id} (90-day suspension started)")

    @retry(
        retry=retry_if_exception(_is_concurrent_modification_error),
        wait=wait_random_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_delay(180),
        reraise=True,
    )
    def move_account_to_root(self, account_id: str, ou_id: str, root_id: str) -> None:
        """Move an account from an OU back to the org root."""
        self._client.move_account(
            AccountId=account_id,
            SourceParentId=ou_id,
            DestinationParentId=root_id,
        )
        logger.info(f"Moved account {account_id} from OU {ou_id} to root")

    def delete_organizational_unit(self, ou_id: str) -> None:
        """Delete an empty OU."""
        self._client.delete_organizational_unit(OrganizationalUnitId=ou_id)
        logger.info(f"Deleted OU {ou_id}")

    def detach_all_scps(self, target_id: str) -> None:
        """Detach all non-default SCPs from a target (OU or account)."""
        try:
            policies = self._client.list_policies_for_target(
                TargetId=target_id, Filter="SERVICE_CONTROL_POLICY"
            )
            for policy in policies.get("Policies", []):
                # Don't detach the AWS default FullAWSAccess policy
                if policy["Id"] == "p-FullAWSAccess":
                    continue
                self._client.detach_policy(PolicyId=policy["Id"], TargetId=target_id)
                logger.info(f"Detached SCP {policy['Id']} from {target_id}")
        except ClientError as e:
            logger.warning(f"Could not detach SCPs from {target_id}: {e}")
