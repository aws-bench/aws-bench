"""Service quota management: request and poll AWS service quota increases."""

from __future__ import annotations

import logging
import time

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.logging.logger import get_logger, log_context
from aws_bench.resource_management.ccapi.models import MAX_WORKERS_HEAVY
from aws_bench.resource_management.exceptions import ConfigurationError, DeploymentError
from aws_bench.resource_management.models import (
    QuotaConfiguration,
    QuotaEntry,
    QuotaIncreaseRequest,
    QuotaIncreaseResult,
    QuotaStatus,
)
from aws_bench.utils.concurrent import (
    build_client,
    interruptible_executor,
    raise_if_shutdown,
)
from aws_bench.utils.credentials_provider import (
    CredentialProvider,
    build_session_name,
    create_regional_session,
)
from aws_bench.utils.regions import get_enabled_regions

logger = get_logger(__name__)

# AWS Organizations "Maximum number of accounts" quota. Organizations is a
# global service hosted in us-east-1, so its Service Quotas must be requested
# there, and only the management account may submit the increase.
ORG_QUOTA_SERVICE_CODE = "organizations"
# "Maximum number of accounts" (default 10). Verified via
# service-quotas:ListServiceQuotas — the code AWS docs reference elsewhere
ORG_ACCOUNT_QUOTA_CODE = "L-E619E033"
ORG_QUOTA_REGION = "us-east-1"


# APPROVED and CASE_CLOSED appear here because a granted increase can read
# below desired until its new value propagates.
_UNMET_REASONS = {
    QuotaStatus.ALREADY_PENDING: "increase pending",
    QuotaStatus.DENIED: "increase denied",
    QuotaStatus.APPROVED: "approved — increase may still be propagating",
    QuotaStatus.CASE_CLOSED: "case closed — increase may still be propagating",
}


class QuotaManager:
    """Requests and polls AWS service quota increases via boto3."""

    def __init__(self, credential_provider: CredentialProvider) -> None:
        """Store reference to credential provider."""
        self._credential_provider = credential_provider

    @staticmethod
    def validate_config(config: QuotaConfiguration) -> None:
        """Validate all fields in the configuration.

        Raises ConfigurationError if any request has empty service_code,
        empty quota_code, or desired_value <= 0.
        """
        for req in config.increases:
            if not req.service_code or not req.service_code.strip():
                raise ConfigurationError(
                    f"service_code must not be empty (quota_code={req.quota_code!r})"
                )
            if not req.quota_code or not req.quota_code.strip():
                raise ConfigurationError(
                    f"quota_code must not be empty (service_code={req.service_code!r})"
                )
            if req.desired_value <= 0:
                raise ConfigurationError(
                    f"desired_value must be > 0 for quota {req.quota_code!r}, "
                    f"got {req.desired_value}"
                )

    def _request_increase(
        self,
        client: BaseClient,
        request: QuotaIncreaseRequest,
        log: logging.Logger | logging.LoggerAdapter,
        *,
        max_retries: int = 3,
        retry_delay: int = 10,
    ) -> QuotaIncreaseResult:
        """Submit a single quota increase request.

        Returns a result with status REQUESTED, ALREADY_PENDING, or ALREADY_MET.
        Raises DeploymentError on unexpected API errors.
        """
        last_exc: ClientError | None = None
        for attempt in range(max_retries + 1):
            try:
                client.request_service_quota_increase(
                    ServiceCode=request.service_code,
                    QuotaCode=request.quota_code,
                    DesiredValue=request.desired_value,
                )
                log.info(
                    "Requested quota increase: %s/%s to %.1f",
                    request.service_code,
                    request.quota_code,
                    request.desired_value,
                )
                return QuotaIncreaseResult(
                    service_code=request.service_code,
                    quota_code=request.quota_code,
                    desired_value=request.desired_value,
                    status=QuotaStatus.REQUESTED,
                )
            except ClientError as exc:
                last_exc = exc
                error_code = exc.response["Error"]["Code"]
                if error_code == "ResourceAlreadyExistsException":
                    log.debug(
                        "Quota increase already pending: %s/%s",
                        request.service_code,
                        request.quota_code,
                    )
                    return QuotaIncreaseResult(
                        service_code=request.service_code,
                        quota_code=request.quota_code,
                        desired_value=request.desired_value,
                        status=QuotaStatus.ALREADY_PENDING,
                    )
                if error_code == "IllegalArgumentException":
                    log.debug(
                        "Quota already met: %s/%s",
                        request.service_code,
                        request.quota_code,
                    )
                    return QuotaIncreaseResult(
                        service_code=request.service_code,
                        quota_code=request.quota_code,
                        desired_value=request.desired_value,
                        status=QuotaStatus.ALREADY_MET,
                    )

                # RegionDisabledException here is a race: quotas requested too
                # soon after account creation, before STS activates. Retry.
                if error_code == "RegionDisabledException":
                    if attempt < max_retries:
                        log.warning(
                            "STS is possibly not yet active for %s, retrying in %ds (%d/%d)",
                            request.quota_code,
                            retry_delay,
                            attempt + 1,
                            max_retries,
                        )
                        time.sleep(retry_delay)
                    continue
                raise DeploymentError(
                    f"Unexpected error requesting quota increase for {request.quota_code}: {exc}"
                ) from exc

        raise DeploymentError(
            f"Exhausted retries requesting quota increase for {request.quota_code}: {last_exc}"
        )

    def request_quotas(
        self,
        config: QuotaConfiguration,
        account_id: str,
        role_name: str,
    ) -> list[QuotaIncreaseResult]:
        """Submit quota increase requests without waiting for approval.

        Each request returns its immediate outcome (REQUESTED, ALREADY_PENDING,
        ALREADY_MET, or FAILED). Approval is verified separately and
        asynchronously via ``verify_quotas``; this method never blocks on it.

        Args:
            config: Quota configuration with increases.
            account_id: Target AWS account ID.
            role_name: IAM role to assume into the account.

        Returns:
            List of results, one per configured quota increase. A request that
            errors is returned as a FAILED result rather than raising.

        Raises:
            ConfigurationError: If any quota request has invalid fields.
        """
        self.validate_config(config)

        with log_context(account_id), log_context(config.region):
            session = self._credential_provider.get_session_for_account(
                account_id, role_name, build_session_name("session", account_id[-6:])
            )
            client: BaseClient = build_client(session, "service-quotas", region_name=config.region)

            results: list[QuotaIncreaseResult] = []
            for req in config.increases:
                try:
                    result = self._request_increase(client, req, logger)
                except DeploymentError as exc:
                    logger.warning(
                        "Failed to request quota increase for %s/%s: %s",
                        req.service_code,
                        req.quota_code,
                        exc,
                    )
                    result = QuotaIncreaseResult(
                        service_code=req.service_code,
                        quota_code=req.quota_code,
                        desired_value=req.desired_value,
                        status=QuotaStatus.FAILED,
                        error_message=str(exc),
                    )
                results.append(result)
                logger.debug(
                    "Quota %s/%s: %s",
                    result.service_code,
                    result.quota_code,
                    result.status.value,
                )

            return results

    def request_org_account_quota(
        self,
        desired_value: float,
        region: str = ORG_QUOTA_REGION,
    ) -> QuotaIncreaseResult:
        """Request an increase to the AWS Organizations account-count quota.

        Submitted from the management account's own session (no role assumption)
        in ``region`` — us-east-1, since AWS Organizations is a global service
        hosted there and only the management account may request this quota.

        Idempotent: an already-satisfied or already-pending quota returns
        ALREADY_MET / ALREADY_PENDING. Never raises — an unexpected API error is
        returned as a FAILED result so a failed submit does not abort
        environment initialization.
        """
        request = QuotaIncreaseRequest(
            service_code=ORG_QUOTA_SERVICE_CODE,
            quota_code=ORG_ACCOUNT_QUOTA_CODE,
            desired_value=desired_value,
        )
        with log_context(region):
            session = self._credential_provider.get_management_session()
            client: BaseClient = build_client(session, "service-quotas", region_name=region)
            try:
                return self._request_increase(client, request, logger)
            except DeploymentError as exc:
                logger.warning(
                    "Failed to request org account-count quota increase to %s: %s",
                    desired_value,
                    exc,
                )
                return QuotaIncreaseResult(
                    service_code=ORG_QUOTA_SERVICE_CODE,
                    quota_code=ORG_ACCOUNT_QUOTA_CODE,
                    desired_value=desired_value,
                    status=QuotaStatus.FAILED,
                    error_message=str(exc),
                )

    def request_org_account_quota_if_absent(self, desired_value: float) -> QuotaIncreaseResult:
        """Submit an org account-count increase unless one is already pending.

        Checks the request history first (``diagnose_org_account_quota``); if an
        increase is already pending (``CASE_OPENED`` / ``PENDING``), returns an
        ALREADY_PENDING result without filing a duplicate. Otherwise submits a
        new request via ``request_org_account_quota``.
        """
        status, _reason = self.diagnose_org_account_quota()
        if status.is_pending:
            logger.info("Org account-count quota increase already pending; not resubmitting.")
            return QuotaIncreaseResult(
                service_code=ORG_QUOTA_SERVICE_CODE,
                quota_code=ORG_ACCOUNT_QUOTA_CODE,
                desired_value=desired_value,
                status=QuotaStatus.ALREADY_PENDING,
            )
        return self.request_org_account_quota(desired_value)

    def diagnose_org_account_quota(self, region: str = ORG_QUOTA_REGION) -> tuple[QuotaStatus, str]:
        """Return the status + human reason for the org account-count quota request.

        Reads the request change history from the management-account session and
        maps the latest request status (pending / denied / approved-but-
        propagating / none) to a ``QuotaStatus`` and a short reason. Used to
        explain why an increase has not yet raised the effective limit.
        """
        session = self._credential_provider.get_management_session()
        client: BaseClient = build_client(session, "service-quotas", region_name=region)
        return self._diagnose_unmet_quota(client, ORG_QUOTA_SERVICE_CODE, ORG_ACCOUNT_QUOTA_CODE)

    def verify_quotas(
        self,
        config: QuotaConfiguration,
        account_id: str,
        role_name: str,
    ) -> list[QuotaIncreaseResult]:
        """Check that current quota values meet the desired values.

        Does not request increases — only reads current values. When a quota
        is insufficient, checks the change history to determine the reason
        (pending, denied, or no request found).

        Returns:
            List of results with ALREADY_MET or the exact status from history.
        """
        self.validate_config(config)

        with log_context(account_id), log_context(config.region):
            session = self._credential_provider.get_session_for_account(
                account_id,
                role_name,
                build_session_name("session", account_id[-6:]),
            )
            client: BaseClient = build_client(session, "service-quotas", region_name=config.region)

            results: list[QuotaIncreaseResult] = []
            for req in config.increases:
                try:
                    resp = client.get_service_quota(
                        ServiceCode=req.service_code,
                        QuotaCode=req.quota_code,
                    )
                    current = resp["Quota"]["Value"]
                except ClientError as exc:
                    raise DeploymentError(f"Failed to get quota {req.quota_code}: {exc}") from exc

                if current >= req.desired_value:
                    logger.debug(
                        "Quota %s/%s: current=%.1f, met (required=%.1f)",
                        req.service_code,
                        req.quota_code,
                        current,
                        req.desired_value,
                    )
                    results.append(
                        QuotaIncreaseResult(
                            service_code=req.service_code,
                            quota_code=req.quota_code,
                            desired_value=req.desired_value,
                            status=QuotaStatus.ALREADY_MET,
                        )
                    )
                    continue

                # DEBUG, not WARNING: an unmet quota is expected during wait
                # polls and the setup gate, and is surfaced to the operator via
                # the progress UI / InsufficientQuotaError, not this log line.
                status, reason = self._diagnose_unmet_quota(
                    client, req.service_code, req.quota_code
                )
                msg = f"current={current:.1f}, required={req.desired_value:.1f} ({reason})"
                logger.debug(
                    "Quota %s/%s: %s",
                    req.service_code,
                    req.quota_code,
                    msg,
                )
                results.append(
                    QuotaIncreaseResult(
                        service_code=req.service_code,
                        quota_code=req.quota_code,
                        desired_value=req.desired_value,
                        status=status,
                        error_message=msg,
                    )
                )
            return results

    @staticmethod
    def _diagnose_unmet_quota(
        client: BaseClient,
        service_code: str,
        quota_code: str,
    ) -> tuple[QuotaStatus, str]:
        """Check quota change history and return the status with a reason."""
        try:
            resp = client.list_requested_service_quota_change_history_by_quota(
                ServiceCode=service_code,
                QuotaCode=quota_code,
            )
            changes = resp.get("RequestedQuotas", [])
            if not changes:
                return QuotaStatus.FAILED, "no increase requested"

            aws_status = changes[0].get("Status", "")
            status = QuotaStatus.from_aws_status(aws_status)
            return status, _UNMET_REASONS.get(status, f"last request status: {aws_status}")
        except ClientError:
            return QuotaStatus.FAILED, "unable to check request history"

    def collect_requested_quotas(
        self, account_id: str, role_name: str = ORG_ACCESS_ROLE
    ) -> tuple[list[QuotaEntry], str | None]:
        """Report an account's requested quotas and their current values.

        Read-only inventory (distinct from ``verify_quotas``, which checks a
        specific config). Returns ``(entries, error)``: ``error`` is set (and
        entries empty) when the account role can't be assumed. Regions are
        scanned in a bounded wave — each hits a distinct throttle bucket.
        """
        try:
            session = self._credential_provider.get_session_for_account(
                account_id,
                role_name,
                build_session_name("session", account_id[-6:]),
            )
            regions = get_enabled_regions(session)
        except Exception as e:
            return [], f"error assuming role: {e}"

        entries: list[QuotaEntry] = []
        if regions:
            with interruptible_executor(max_workers=min(len(regions), MAX_WORKERS_HEAVY)) as ex:
                for region_entries in ex.map(
                    lambda region: self._region_requested_quotas(session, region),
                    regions,
                ):
                    entries.extend(region_entries)

        # Group by region, then quota id, so multi-region accounts read top-down.
        entries.sort(key=lambda e: (e.region, e.quota_id))
        return entries, None

    @staticmethod
    def _region_requested_quotas(session: boto3.Session, region: str) -> list[QuotaEntry]:
        """Return one region's requested-quota entries (region fan-out worker)."""
        raise_if_shutdown()
        with log_context(region):
            entries: list[QuotaEntry] = []
            try:
                sq = build_client(create_regional_session(session, region), "service-quotas")
                paginator = sq.get_paginator("list_requested_service_quota_change_history")
                for page in paginator.paginate():
                    for req in page.get("RequestedQuotas", []):
                        svc = req.get("ServiceCode", "?")
                        code = req.get("QuotaCode", "?")
                        desired = req.get("DesiredValue", 0)
                        name = req.get("QuotaName", "")
                        try:
                            quota = sq.get_service_quota(ServiceCode=svc, QuotaCode=code)["Quota"]
                            current = quota["Value"]
                            name = name or quota.get("QuotaName", "")
                            met = current >= desired
                        except Exception:
                            current, met = None, False
                        entries.append(QuotaEntry(region, code, name, desired, current, met))
            except Exception:
                return []
            return entries
