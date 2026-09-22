"""Data models for snapshot management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path


class SnapshotStage(StrEnum):
    """Stages at which snapshots are captured.

    - PRE_SETUP: the pristine init baseline captured by ``env init`` right after
      accounts are provisioned (only AWS defaults present). Stored in S3;
      ``env cleanup`` subtracts it so account defaults are neither deleted nor
      reported as orphans.
    - POST_SETUP: the deployed baseline captured by ``env setup``. Stored in S3;
      ``verify`` and ``reset`` diff the live account against it.
    - OBSERVABILITY: an on-demand snapshot for inspection/debugging. Written to a
      local JSON file, never stored in S3, and never consumed by
      setup/verify/reset/cleanup (no lifecycle coupling).
    """

    PRE_SETUP = "pre-setup"
    POST_SETUP = "post-setup"
    OBSERVABILITY = "observability"


@dataclass
class ResourceDrift:
    """Represents a single resource's drift information."""

    logical_resource_id: str
    stack_resource_drift_status: str  # IN_SYNC, MODIFIED, DELETED, NOT_CHECKED
    property_differences: list[dict] = field(default_factory=list)


@dataclass
class DriftBaseline:
    """Drift detection results for a stack."""

    detection_status: str  # DETECTION_COMPLETE, DETECTION_FAILED
    resource_drifts: list[ResourceDrift] = field(default_factory=list)
    # CloudFormation's DetectionStatusReason, to tell a permanent resource-gone
    # failure from a transient one. Empty unless detection_status is FAILED.
    detection_status_reason: str = ""


@dataclass
class StackMetadata:
    """Metadata about a CloudFormation stack."""

    status: str  # Stack status (CREATE_COMPLETE, ROLLBACK_FAILED, etc.)
    template_hash: str  # SHA256 hash of template
    parameters: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    region: str = ""  # AWS region the stack lives in (for per-region verify filtering)


@dataclass(frozen=True)
class SnapshotKey:
    """Unique identifier for a snapshot."""

    env_name: str
    account_id: str
    stage: SnapshotStage


@dataclass
class Snapshot:
    """Complete baseline snapshot of account state."""

    timestamp: datetime
    account_id: str
    environment_id: str
    scenario_hash: str
    drift_baseline: dict[str, DriftBaseline]
    stack_metadata: dict[str, StackMetadata]
    resource_ids: dict[str, list[dict]]  # resource_type -> list of resources
    regions: list[str] = field(default_factory=list)  # Regions scanned in this snapshot
    failed_resource_types: dict[str, str] = field(default_factory=dict)  # CCAPI scan failures
    empty_resource_types: set[str] = field(
        default_factory=set
    )  # Types scanned but returned 0 resources


@dataclass
class SnapshotResult:
    """Result of capturing a snapshot for a single account."""

    account_id: str
    success: bool
    regions_captured: list[str] = field(default_factory=list)
    error_message: str | None = None
    output_path: str | None = None  # Local file path (OBSERVABILITY captures only)


@dataclass
class SnapshotContext:
    """Context for snapshot operations to reduce parameter repetition.

    Used both for single-account operations (snapshot_account) and
    multi-account operations (snapshot_scenarios). For multi-account,
    account_ids specifies which accounts to snapshot.

    scenario_hash is empty for PRE_SETUP stage (captured before deployment).
    """

    scenario_id: str
    scenario_hash: str
    regions: list[str]
    stage: SnapshotStage = SnapshotStage.POST_SETUP
    account_ids: list[str] = field(default_factory=list)
    # When set, the snapshot is written to a JSON file under this directory
    # instead of S3; used by OBSERVABILITY captures.
    output_dir: Path | None = None
    # Identifiers a caller flagged as unresolved orphans; a captured baseline
    # containing any of these is refused, never saved (defense-in-depth against
    # absorbing a live orphan into the baseline).
    forbidden_identifiers: set[str] = field(default_factory=set)
