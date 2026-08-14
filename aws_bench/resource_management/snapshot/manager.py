# aws_bench/resource_management/snapshot/manager.py
"""Snapshot manager for capturing and loading baseline state."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import tenacity
from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.account_management.preexisting import active_account_config
from aws_bench.constants import STATE_DIR
from aws_bench.logging.logger import get_logger, log_context
from aws_bench.resource_management.ccapi.models import MAX_WORKERS_ACCOUNT, MAX_WORKERS_HEAVY
from aws_bench.resource_management.constants import RESOURCE_MANAGEMENT_SESSION
from aws_bench.resource_management.exceptions import DriftDetectionError, SnapshotNotFoundError
from aws_bench.resource_management.fastscan.engine import _TRANSIENT_SERVER_CODES
from aws_bench.resource_management.scanner import make_scanner, scan_method
from aws_bench.resource_management.snapshot.drift import (
    DRIFT_DETECTION_FAILED,
    detect_stacks_drift,
    get_drift_client,
)
from aws_bench.resource_management.snapshot.models import (
    DriftBaseline,
    ResourceDrift,
    Snapshot,
    SnapshotContext,
    SnapshotKey,
    SnapshotResult,
    SnapshotStage,
    StackMetadata,
)
from aws_bench.resource_management.storage import SnapshotStorage
from aws_bench.resource_management.storage.exceptions import (
    StorageConflictError,
    StorageNotFoundError,
)
from aws_bench.resource_management.storage.local_storage_backend import LocalStorageBackend
from aws_bench.resource_management.storage.s3_backend import S3StorageBackend
from aws_bench.utils.concurrent import build_client, interruptible_executor, raise_if_shutdown
from aws_bench.utils.credentials_provider import (
    CredentialProvider,
    build_session_name,
    create_regional_session,
)
from aws_bench.utils.retry import is_fresh_account_transient

logger = get_logger(__name__)

STATE_BUCKET_PREFIX = "awsbench-state-"


class SnapshotManager:
    """Manages snapshot capture and loading operations.

    The storage backend is lazily initialized on first access to avoid paying
    the initialization cost (STS get_caller_identity + S3 head_bucket + 4 config PUTs)
    in code paths that only use CloudFormation APIs (e.g., drift capture).
    """

    def __init__(self):
        """Initialize snapshot manager."""
        self._backend: SnapshotStorage | None = None
        self._etags: dict[SnapshotKey, str] = {}

    @property
    def _storage(self) -> SnapshotStorage:
        """Lazily initialize the storage backend on first access.

        Pre-existing mode has no management account to hold a state bucket, and a
        bucket in the account under test would put benchmark state inside the
        agent's blast radius, so state goes to host-local disk instead. Managed
        mode derives its bucket from the management account
        (``awsbench-state-{account-id}``).

        Returns:
            Initialized storage backend
        """
        if self._backend is None:
            if active_account_config() is not None:
                logger.debug(f"Initializing local storage backend at {STATE_DIR} (lazy)")
                self._backend = LocalStorageBackend(root=STATE_DIR)
                return self._backend

            logger.debug("Initializing S3 storage backend (lazy)")
            mgmt_session = CredentialProvider.get().get_management_session()

            # Auto-derive bucket name from management account
            sts = build_client(mgmt_session, "sts")
            identity = sts.get_caller_identity()
            account_id = identity["Account"]
            state_bucket = f"{STATE_BUCKET_PREFIX}{account_id}"

            # Create S3 storage backend
            self._backend = S3StorageBackend(
                session=mgmt_session,
                bucket_name=state_bucket,
            )

        return self._backend

    def _make_s3_key(self, env_name: str, account_id: str, stage: SnapshotStage) -> str:
        """Return the storage key for a snapshot."""
        return f"{env_name}/{stage}/{account_id}/baseline.json"

    def _get_snapshot_key(
        self, env_name: str, account_id: str, stage: SnapshotStage
    ) -> SnapshotKey:
        """Return the etag key for a snapshot."""
        return SnapshotKey(env_name, account_id, stage)

    def _count_resources(self, snapshot: Snapshot) -> int:
        """Count total resources in a snapshot."""
        return self._count_resources_from_dict(snapshot.resource_ids)

    def _count_resources_from_dict(self, resource_ids: dict[str, list[dict]]) -> int:
        """Count total resources from resource_ids dict."""
        return sum(len(v) for v in resource_ids.values())

    def save_snapshot(
        self, env_name: str, snapshot: Snapshot, stage: SnapshotStage = SnapshotStage.POST_SETUP
    ) -> None:
        """Save snapshot to S3 storage.

        Args:
            env_name: Environment name
            snapshot: Snapshot to save
            stage: Snapshot lifecycle stage
        """
        logger.debug(f"Saving snapshot for environment '{env_name}', account {snapshot.account_id}")

        data = self._snapshot_to_dict(snapshot)
        resource_count = self._count_resources(snapshot)
        stack_count = len(snapshot.stack_metadata)
        logger.debug(f"Snapshot contains {stack_count} stacks, {resource_count} resources")

        # Serialize to JSON bytes
        json_data = json.dumps(data, indent=2).encode("utf-8")

        # Get keys
        s3_key = self._make_s3_key(env_name, snapshot.account_id, stage)
        etag_key = self._get_snapshot_key(env_name, snapshot.account_id, stage)

        # Get expected ETag for optimistic locking (if we've loaded this snapshot)
        expected_etag = self._etags.get(etag_key)

        # Write to S3 (MUST succeed)
        try:
            new_etag = self._storage.save(s3_key, json_data, expected_etag=expected_etag)
            self._etags[etag_key] = new_etag
            logger.debug(
                f"Saved snapshot for account {snapshot.account_id} (etag={new_etag[:8]}...)"
            )
        except StorageConflictError:
            # Enrich the conflict error with snapshot-level context
            raise StorageConflictError(
                key=s3_key,
                expected=expected_etag or "<none>",
                actual="<modified>",
            ) from None

    def load_snapshot(
        self, env_name: str, account_id: str, stage: SnapshotStage = SnapshotStage.POST_SETUP
    ) -> Snapshot:
        """Load snapshot from S3 storage.

        Args:
            env_name: Environment name
            account_id: AWS account ID
            stage: Stage at which the snapshot was taken

        Returns:
            Loaded snapshot

        Raises:
            SnapshotNotFoundError: If snapshot doesn't exist in S3
            StorageError: For transient S3 errors (retryable)
            json.JSONDecodeError: If snapshot data is corrupted
        """
        logger.debug(f"Loading snapshot for environment '{env_name}', account {account_id}")

        s3_key = self._make_s3_key(env_name, account_id, stage)
        etag_key = self._get_snapshot_key(env_name, account_id, stage)

        try:
            raw_data, etag = self._storage.load(s3_key)
        except StorageNotFoundError as e:
            # Shared read: raises for the caller to classify (terminal for verify, ok for cleanup).
            logger.debug(
                f"Snapshot not found for environment '{env_name}', account {account_id}: {e}"
            )
            raise SnapshotNotFoundError(env_name, account_id, stage) from e
        self._etags[etag_key] = etag
        data = json.loads(raw_data)
        snapshot = self._dict_to_snapshot(data)

        resource_count = self._count_resources(snapshot)
        stack_count = len(snapshot.stack_metadata)
        logger.debug(
            f"Loaded snapshot for {account_id}: {stack_count} stacks, {resource_count} resources"
        )

        return snapshot

    def snapshot_exists(
        self, env_name: str, account_id: str, stage: SnapshotStage = SnapshotStage.POST_SETUP
    ) -> bool:
        """Check if snapshot exists in S3 (fast HEAD operation).

        Args:
            env_name: Environment name
            account_id: AWS account ID
            stage: Stage at which the snapshot was taken

        Returns:
            True if snapshot exists, False otherwise
        """
        s3_key = self._make_s3_key(env_name, account_id, stage)
        return self._storage.exists(s3_key)

    def delete_snapshot(
        self, env_name: str, account_id: str, stage: SnapshotStage = SnapshotStage.POST_SETUP
    ) -> None:
        """Delete snapshot from S3 (idempotent).

        Args:
            env_name: Environment name
            account_id: AWS account ID
            stage: Stage at which the snapshot was taken
        """
        s3_key = self._make_s3_key(env_name, account_id, stage)
        self._storage.delete(s3_key)
        logger.debug(f"Deleted snapshot for {env_name}/{account_id} at stage '{stage}'")

    # Sequential per account+region, so it can afford a long convergence budget.
    @tenacity.retry(
        retry=tenacity.retry_if_exception(is_fresh_account_transient),
        wait=tenacity.wait_exponential(multiplier=2, min=10, max=60) + tenacity.wait_random(0, 5),
        stop=tenacity.stop_after_delay(180),
        reraise=True,
    )
    def _list_active_stacks(self, cfn: Any) -> list[dict[str, Any]]:
        """List active CloudFormation stacks (exclude deleted and nested stacks).

        The snapshot's first AWS call, so a fresh account's unconverged subscription
        surfaces here (see is_fresh_account_transient); the decorator retries it.
        """
        logger.debug("Listing CloudFormation stacks")
        stacks = []
        paginator = cfn.get_paginator("list_stacks")
        for page in paginator.paginate():
            for stack in page["StackSummaries"]:
                if stack["StackStatus"] != "DELETE_COMPLETE" and "ParentId" not in stack:
                    stacks.append(stack)
        return stacks

    def _capture_drift_baselines(
        self, cfn: Any, stacks: list[dict[str, Any]]
    ) -> dict[str, DriftBaseline]:
        """Capture drift baselines for all stacks via :func:`detect_stacks_drift`.

        Raises ``DriftDetectionError`` if any stack is still DETECTION_FAILED
        after retries — saving an incomplete baseline would read as a false drift
        mismatch on a later verify/reset, so fail here and re-run setup instead.
        """
        logger.debug(f"Detecting drift for {len(stacks)} stack(s)")

        drift_baseline = detect_stacks_drift(
            cfn, [(stack["StackName"], stack["StackStatus"]) for stack in stacks]
        )

        failed = sorted(
            name
            for name, baseline in drift_baseline.items()
            if baseline.detection_status == DRIFT_DETECTION_FAILED
        )
        if failed:
            raise DriftDetectionError(
                f"Drift detection failed for {len(failed)} stack(s) after retries: "
                f"{', '.join(failed)}. Snapshot not saved — re-run setup."
            )

        drift_skipped = sum(
            1 for b in drift_baseline.values() if b.detection_status.startswith("SKIPPED_")
        )
        logger.debug(
            f"Drift detection: {len(drift_baseline) - drift_skipped} detected, "
            f"{drift_skipped} skipped"
        )
        return drift_baseline

    def _capture_stack_metadata(
        self, cfn: Any, stacks: list[dict[str, Any]], region: str = ""
    ) -> dict[str, StackMetadata]:
        """Capture metadata for all stacks (tagging each with its region)."""
        logger.debug(f"Capturing metadata for {len(stacks)} stack(s)")

        stack_metadata = {}
        for stack in stacks:
            raise_if_shutdown()
            stack_name = stack["StackName"]

            try:
                stack_detail = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]
                template_body = cfn.get_template(StackName=stack_name)["TemplateBody"]

                # Normalize template_body to dict for consistent hashing
                if isinstance(template_body, str):
                    try:
                        template_body = json.loads(template_body)
                    except json.JSONDecodeError:
                        # YAML template - hash the raw string directly
                        template_hash = hashlib.sha256(template_body.encode()).hexdigest()
                    else:
                        template_hash = hashlib.sha256(
                            json.dumps(template_body, sort_keys=True).encode()
                        ).hexdigest()
                else:
                    # Already a dict
                    template_hash = hashlib.sha256(
                        json.dumps(template_body, sort_keys=True).encode()
                    ).hexdigest()

                stack_metadata[stack_name] = StackMetadata(
                    status=stack_detail["StackStatus"],
                    template_hash=f"sha256:{template_hash}",
                    parameters={
                        p["ParameterKey"]: p["ParameterValue"]
                        for p in stack_detail.get("Parameters", [])
                    },
                    tags={t["Key"]: t["Value"] for t in stack_detail.get("Tags", [])},
                    region=region,
                )
                logger.debug(f"Captured: {stack_name}")
            except (ClientError, BotoCoreError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to capture metadata for stack '{stack_name}': {e}")
                logger.debug(f"Failed: {stack_name} - {e}")
                stack_metadata[stack_name] = StackMetadata(
                    status=stack.get("StackStatus", "UNKNOWN"),
                    template_hash="unavailable",
                    parameters={},
                    tags={},
                    region=region,
                )

        logger.debug("Metadata capture complete")
        return stack_metadata

    def capture_snapshot(
        self,
        scan_session: boto3.Session,
        account_id: str,
        environment_id: str,
        scenario_hash: str,
        region: str,
    ) -> Snapshot:
        """Capture snapshot for an account.

        Args:
            scan_session: boto3 session for scanning target account
            account_id: AWS account ID
            environment_id: Environment/scenario name
            scenario_hash: SHA256 hash of scenario folder
            region: AWS region to scan

        Returns:
            Snapshot with captured state
        """
        logger.debug(f"Capturing snapshot for account {account_id} in region {region}")

        region_session = create_regional_session(scan_session, region)
        cfn = get_drift_client(region_session, region_name=region)
        scan_mgr = make_scanner(region_session, region_name=region, account_id=account_id)

        stacks = self._list_active_stacks(cfn)
        logger.debug(f"Found {len(stacks)} active stack(s) in {region}")

        drift_baseline = self._capture_drift_baselines(cfn, stacks)
        stack_metadata = self._capture_stack_metadata(cfn, stacks, region)

        logger.debug("Scanning resources via %s backend", scan_method())
        scan_result = scan_mgr.scan_resources(region=region)
        resource_count = self._count_resources_from_dict(scan_result.detected)
        logger.debug(f"Fast-scan found {resource_count} resources")

        # Fail the snapshot if transient server errors (5xx/throttle) persisted
        # beyond all retries. A snapshot impaired by these errors would have
        # missing resources, causing incorrect verify/reset/cleanup decisions.
        transient_failures = {
            k: v
            for k, v in scan_result.failed.items()
            if any(pattern in v for pattern in _TRANSIENT_SERVER_CODES)
        }
        if transient_failures:
            raise RuntimeError(
                f"Snapshot aborted for {region}: {len(transient_failures)} lister(s) "
                f"failed with transient server errors after retries. "
                f"Examples: {dict(list(transient_failures.items())[:5])}. "
                f"Re-run the command to retry."
            )

        snapshot = Snapshot(
            timestamp=datetime.now(timezone.utc),
            account_id=account_id,
            environment_id=environment_id,
            scenario_hash=scenario_hash,
            drift_baseline=drift_baseline,
            stack_metadata=stack_metadata,
            resource_ids=scan_result.detected,
            regions=[region],
            failed_resource_types=scan_result.failed,
            empty_resource_types=scan_result.empty,
        )

        logger.debug(f"Snapshot captured: {len(stacks)} stacks, {resource_count} resources")
        return snapshot

    def capture_snapshot_multiregion(
        self,
        scan_session: boto3.Session,
        account_id: str,
        environment_id: str,
        scenario_hash: str,
        regions: list[str],
    ) -> Snapshot:
        """Capture and merge a baseline snapshot across multiple regions.

        Captures regions concurrently (up to MAX_WORKERS_HEAVY at a time) and
        merges the results into a single snapshot whose ``regions`` lists every
        scanned region. This is the correct shape for ``verify``/``reset``,
        which re-scan per region and compare against the merged baseline.

        Args:
            scan_session: boto3 session for scanning target account
            account_id: AWS account ID
            environment_id: Environment/scenario name
            scenario_hash: SHA256 hash of scenario folder
            regions: AWS regions to scan (must be non-empty)

        Returns:
            Merged Snapshot covering all regions.
        """
        if not regions:
            raise ValueError("regions must be non-empty")

        snapshots: list[Snapshot] = []

        def _capture_region(region: str) -> Snapshot:
            """Thread-safe capture of a single region."""
            with log_context(region):
                raise_if_shutdown()
                logger.debug(f"Starting snapshot capture for region {region}")
                snapshot = self.capture_snapshot(
                    scan_session, account_id, environment_id, scenario_hash, region
                )
                logger.debug(f"Completed snapshot capture for region {region}")
                return snapshot

        with interruptible_executor(max_workers=MAX_WORKERS_HEAVY) as executor:
            # Submit all regions for concurrent processing
            future_to_region = {
                executor.submit(_capture_region, region): region for region in regions
            }

            # Collect results as they complete
            for future in as_completed(future_to_region):
                region = future_to_region[future]
                try:
                    snapshot = future.result()
                    snapshots.append(snapshot)
                except Exception as exc:
                    logger.error(f"Region {region} snapshot failed: {exc}")
                    raise

        if not snapshots:
            raise RuntimeError("No snapshots captured successfully")

        # Merge all snapshots
        merged = snapshots[0]
        for part in snapshots[1:]:
            # Merge region-scoped state. Stack names are unique per region in
            # practice; on any collision the later region wins (logged for
            # visibility).
            for stack_name, drift in part.drift_baseline.items():
                if stack_name in merged.drift_baseline:
                    logger.warning(f"Stack name collision across regions: {stack_name}")
                merged.drift_baseline[stack_name] = drift
            merged.stack_metadata.update(part.stack_metadata)

            # Union resources per type (dedupe by Identifier).
            for rtype, resources in part.resource_ids.items():
                existing = merged.resource_ids.setdefault(rtype, [])
                seen = {r.get("Identifier") for r in existing}
                existing.extend(r for r in resources if r.get("Identifier") not in seen)

            merged.failed_resource_types.update(part.failed_resource_types)
            merged.empty_resource_types |= part.empty_resource_types

        # Set regions list to the complete input list (not accumulated via append)
        merged.regions = list(regions)
        logger.debug(
            f"Merged multi-region snapshot for account {account_id}: "
            f"regions={merged.regions}, {len(merged.stack_metadata)} stack(s)"
        )
        return merged

    def _snapshot_to_dict(self, snapshot: Snapshot) -> dict[str, Any]:
        """Convert Snapshot to JSON-serializable dict."""
        data = asdict(snapshot)
        data["timestamp"] = snapshot.timestamp.isoformat()
        data["empty_resource_types"] = list(snapshot.empty_resource_types)
        return data

    def _dict_to_drift_baseline(self, drift_data: dict[str, Any]) -> DriftBaseline:
        """Convert dict to DriftBaseline object."""
        resource_drifts = [ResourceDrift(**rd) for rd in drift_data["resource_drifts"]]
        return DriftBaseline(
            detection_status=drift_data["detection_status"],
            resource_drifts=resource_drifts,
        )

    def _dict_to_snapshot(self, data: dict[str, Any]) -> Snapshot:
        """Convert dict to Snapshot object."""
        drift_baseline = {
            stack: self._dict_to_drift_baseline(drift)
            for stack, drift in data["drift_baseline"].items()
        }
        stack_metadata = {
            stack: StackMetadata(**meta) for stack, meta in data["stack_metadata"].items()
        }
        return Snapshot(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            account_id=data["account_id"],
            environment_id=data["environment_id"],
            scenario_hash=data.get("scenario_hash", data.get("dataset_ref", "unknown")),
            drift_baseline=drift_baseline,
            stack_metadata=stack_metadata,
            resource_ids=data["resource_ids"],
            regions=data.get("regions", []),
            failed_resource_types=data.get("failed_resource_types", {}),
            empty_resource_types=set(data.get("empty_resource_types", [])),
        )

    def _write_snapshot_file(
        self, output_dir: Path, scenario_name: str, snapshot: Snapshot
    ) -> Path:
        """Write a snapshot to ``<output_dir>/<scenario_name>/<account_id>.json``.

        Returns the path written. Used for on-demand OBSERVABILITY captures that go
        to disk instead of the S3 state bucket. ``scenario_name`` is validated to
        stay within ``output_dir`` so a separator- or ``..``-bearing name cannot
        write outside the output tree.

        Raises:
            ValueError: If ``scenario_name`` would resolve outside ``output_dir``.
        """
        dest_dir = output_dir / scenario_name
        if output_dir.resolve() not in dest_dir.resolve().parents:
            raise ValueError(f"Unsafe scenario name for snapshot path: '{scenario_name}'")
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"{snapshot.account_id}.json"
        data = self._snapshot_to_dict(snapshot)
        path.write_text(json.dumps(data, indent=2))
        return path

    def snapshot_account(
        self,
        scan_session: boto3.Session,
        account_id: str,
        ctx: SnapshotContext,
    ) -> SnapshotResult:
        """Snapshot a single account and save to S3.

        Args:
            scan_session: boto3 session for scanning the account
            account_id: AWS account ID
            ctx: Snapshot context with environment, scenario, regions, and stage

        Returns:
            SnapshotResult with success status and regions captured
        """
        try:
            snapshot = self.capture_snapshot_multiregion(
                scan_session, account_id, ctx.scenario_id, ctx.scenario_hash, ctx.regions
            )
            if ctx.output_dir is not None:
                path = self._write_snapshot_file(ctx.output_dir, ctx.scenario_id, snapshot)
                logger.debug(
                    f"Wrote {ctx.stage} snapshot for {ctx.scenario_id}/{account_id} to {path}"
                )
                return SnapshotResult(
                    account_id=account_id,
                    success=True,
                    regions_captured=ctx.regions,
                    output_path=str(path),
                )
            self.save_snapshot(ctx.scenario_id, snapshot, stage=ctx.stage)
            logger.debug(f"Saved {ctx.stage} snapshot for {ctx.scenario_id}/{account_id}")
            return SnapshotResult(
                account_id=account_id,
                success=True,
                regions_captured=ctx.regions,
            )
        except Exception as exc:
            logger.error(f"Snapshot failed for {ctx.scenario_id}/{account_id}: {exc}")
            return SnapshotResult(
                account_id=account_id,
                success=False,
                error_message=str(exc),
            )

    def snapshot_scenarios(
        self,
        contexts: list[SnapshotContext],
    ) -> dict[str, list[SnapshotResult]]:
        """Snapshot multiple scenarios (each with multiple accounts) concurrently.

        For PRE_SETUP snapshots: skips accounts that already have a baseline to maintain
        idempotency and prevent overwriting clean baselines with contaminated ones.

        Args:
            contexts: List of SnapshotContext, each with account_ids populated

        Returns:
            Dict mapping scenario name (environment_id) to list of SnapshotResult per account
        """
        cred_provider = CredentialProvider.get()
        results: dict[str, list[SnapshotResult]] = {}

        def _snapshot_one_account(ctx: SnapshotContext, account_id: str) -> SnapshotResult:
            """Snapshot a single account."""
            with log_context(account_id):
                raise_if_shutdown()
                # PRE_SETUP idempotency: skip if baseline already exists
                if ctx.stage == SnapshotStage.PRE_SETUP:
                    if self.snapshot_exists(ctx.scenario_id, account_id, SnapshotStage.PRE_SETUP):
                        logger.debug(
                            f"Skipping PRE_SETUP for {ctx.scenario_id}/{account_id} - "
                            "baseline exists"
                        )
                        return SnapshotResult(
                            account_id=account_id,
                            success=True,
                            regions_captured=ctx.regions,
                        )

                session = cred_provider.get_session_for_account(
                    account_id,
                    ORG_ACCESS_ROLE,
                    build_session_name(RESOURCE_MANAGEMENT_SESSION, f"snapshot-{ctx.stage}"),
                )
                return self.snapshot_account(session, account_id, ctx)

        # One worker per account, capped at MAX_WORKERS_ACCOUNT; never start more than needed.
        total_accounts = sum(len(ctx.account_ids) for ctx in contexts)
        with interruptible_executor(
            max_workers=max(1, min(total_accounts, MAX_WORKERS_ACCOUNT))
        ) as executor:
            # Submit all accounts across all scenarios
            future_to_context_and_account = {}
            for ctx in contexts:
                for account_id in ctx.account_ids:
                    future = executor.submit(_snapshot_one_account, ctx, account_id)
                    future_to_context_and_account[future] = (ctx.scenario_id, account_id)

            # Collect results grouped by scenario
            for future in as_completed(future_to_context_and_account):
                env_id, account_id = future_to_context_and_account[future]
                try:
                    result = future.result()
                    if env_id not in results:
                        results[env_id] = []
                    results[env_id].append(result)
                except Exception as exc:
                    logger.error(f"Snapshot failed for {env_id}/{account_id}: {exc}")
                    if env_id not in results:
                        results[env_id] = []
                    results[env_id].append(
                        SnapshotResult(
                            account_id=account_id,
                            success=False,
                            error_message=str(exc),
                        )
                    )

        return results
