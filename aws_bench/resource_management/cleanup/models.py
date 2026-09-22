"""Stack deletion models and constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from aws_bench.resource_management.ccapi.models import (
    GLOBAL_RESOURCE_TYPES,
    DeletionFailureEvent,
    Resource,
    ScanResult,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLL_MIN_SEC = 5
POLL_MAX_SEC = 30
STACK_DELETE_TIMEOUT = 1800  # 30 minutes initial timeout
STACK_DELETE_MAX_TIMEOUT = 3600  # 1 hour absolute maximum
STACK_DELETE_RETRIES = 2
STACK_DELETE_CONCURRENCY = 10
EVENTUAL_CONSISTENCY_WAIT_SEC = 10  # Wait for S3/IAM eventual consistency after deletion
DELETE_COMPLETE = "DELETE_COMPLETE"
DELETE_FAILED = "DELETE_FAILED"
INFRA_PREFIXES = ("CDKToolkit", "cdk-hnb659fds-")
# Bootstrap version param; the scanner emits its ARN, so it needs a substring (not prefix) match.
INFRA_SUBSTRINGS = ("cdk-bootstrap/hnb659fds/",)

# CDK bootstrap infra that IS safe to reclaim once a region's stacks are gone:
# regional, single-owner assets the CDKToolkit stack retains on stack delete
# (the assets S3 bucket, the bootstrap version SSM param, the image-publishing
# ECR repo). Only these types are re-included by an ``include_infra`` sweep.
# The bootstrap IAM roles (``cdk-hnb659fds-*-role``) are deliberately absent:
# they are global (see ``GLOBAL_RESOURCE_TYPES``) and may still back a surviving
# stack's ``RoleARN`` — or a custom-resource provider — in ANY region, so they
# must never be swept as "orphaned infra". Deleting one wedges every stack that
# references it (CloudFormation can no longer assume the role to delete them).
SWEEPABLE_INFRA_TYPES = frozenset(
    {"AWS::S3::Bucket", "AWS::SSM::Parameter", "AWS::ECR::Repository"}
)

# Resources CCAPI lists in every region but that exist account-wide. The
# per-region sweep path (phases 1 and 3) runs concurrently across regions and
# does NOT dedup globals (only the orphan-report path does), so sweeping these
# per-region races: a region that finished deleting its stacks can delete a
# role a still-deleting region's stack depends on. They are therefore withheld
# from the per-region sweeps and reaped once, after every region finishes stack
# deletion and only when no stack survives anywhere
# (see ``CleanupManager._sweep_global_leftovers``).
#
# This is the single canonical set, shared with the reset path
# (``reset.manager``) so the two teardown flows never diverge. WAFv2
# (``AWS::WAFv2::WebACL`` etc.) is intentionally absent: a WebACL is global only
# at CLOUDFRONT scope and regional at REGIONAL scope, yet both share one CFN type
# name — blanket-listing it would misclassify every regional WAF as global.


def is_infra_identifier(identifier: str) -> bool:
    """Return True if the identifier belongs to CDK bootstrap/toolkit infrastructure."""
    return identifier.startswith(INFRA_PREFIXES) or any(s in identifier for s in INFRA_SUBSTRINGS)


def partition_by_scope(
    resources: dict[str, list[dict]],
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Split a detected-resources map into ``(regional, global_)`` by resource type.

    Global types (``GLOBAL_RESOURCE_TYPES``) surface in every region's scan and
    must be swept once behind a barrier, not per-region; the regional remainder
    is safe to sweep in its own region once that region's stacks are gone.
    """
    regional: dict[str, list[dict]] = {}
    global_: dict[str, list[dict]] = {}
    for rtype, items in resources.items():
        (global_ if rtype in GLOBAL_RESOURCE_TYPES else regional)[rtype] = items
    return regional, global_


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class StackDeletionStatus(Enum):
    """Outcome of a single stack deletion attempt."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    NOT_FOUND = "NOT_FOUND"


class ExistenceStatus(Enum):
    """Post-deletion resource existence check result.

    Indicates whether a resource still exists after cleanup attempts.
    """

    EXISTS = "exists"
    ABSENT = "absent"
    SKIPPED = "skipped"
    UNKNOWN = "unknown"
    UNCHECKED_SUBRESOURCE = "unchecked_subresource"


@dataclass(frozen=True)
class SnapshotResources:
    """A snapshot's resources plus the types that failed to enumerate.

    ``failed_types`` matters for the destructive cleanup diff: a type that failed
    to enumerate in the snapshot (so it is absent from ``resource_ids``) must be
    skipped, not treated as "had zero resources" — otherwise every live resource
    of that type reads as new and would be deleted (e.g. a default VPC).
    """

    resource_ids: dict[str, list[dict]]
    failed_types: dict[str, str]


@dataclass
class StackResource:
    """A single resource within a CloudFormation stack."""

    logical_id: str
    physical_id: str
    resource_type: str
    status: str


@dataclass
class ResourceVerificationResult:
    """Result of verifying a resource's existence after cleanup.

    Includes the original stack resource info plus verification outcome and metadata.
    """

    logical_id: str
    physical_id: str
    resource_type: str
    cfn_status: str  # Original CloudFormation resource status
    existence_status: ExistenceStatus
    metadata: dict[str, str] = field(default_factory=dict)  # For ccapi_cleanup, etc.


@dataclass
class StackDeletionResult:
    """Result of deleting a single stack, including surviving resources."""

    stack_name: str
    status: StackDeletionStatus
    reason: str = ""
    resources: list[StackResource] = field(default_factory=list)
    # Resources FORCE_DELETE_STACK could not delete and left live in the account.
    # The single-stack cleanup path runs no orphan scan, so this is the only record
    # of them and must be surfaced, not dropped.
    abandoned_resources: list[StackResource] = field(default_factory=list)
    deferred: bool = False
    """The stack did not delete this run, but its only remaining blockers are
    eventually-deletable (subnet/VPC pinned by requester-managed ENIs that AWS
    releases ~20-40 min after the owning service is deleted). Such a stack — and
    the resources it left behind — are recorded in the run's deferred registry so
    the post-cleanup orphan scan excludes them and the run is NOT failed for a
    state that self-heals. ``status`` stays ``FAILED`` (honest: the stack is still
    present), but a deferred result is excluded from the failure verdict and is
    reaped by a later run once the owner has released the ENIs."""


@dataclass
class DeletionSummary:
    """Aggregated results across multiple stack deletions."""

    results: list[StackDeletionResult] = field(default_factory=list)

    @property
    def succeeded(self) -> list[StackDeletionResult]:
        """Return results with SUCCESS status."""
        return [result for result in self.results if result.status == StackDeletionStatus.SUCCESS]

    @property
    def failed(self) -> list[StackDeletionResult]:
        """Return results that failed to delete, excluding deferred ones.

        A deferred result kept ``FAILED`` status for honest diagnostics (the stack
        is still present), but its only blockers are eventually-deletable
        requester-managed ENIs, so it must not count toward the failure verdict.
        """
        return [
            result
            for result in self.results
            if result.status == StackDeletionStatus.FAILED and not result.deferred
        ]

    @property
    def deferred(self) -> list[StackDeletionResult]:
        """Return results deferred to a later run (requester-managed ENI pins)."""
        return [result for result in self.results if result.deferred]

    @property
    def all_succeeded(self) -> bool:
        """Return True if no results failed (deferred results do not count as failures)."""
        return not self.failed


@dataclass
class CustomDeletionResult:
    """Result of running custom deletion handlers on a batch of resources.

    skipped   — resources with no registered custom handler (need CCAPI fallback).
    succeeded — resources whose custom handler completed without error.
    failed    — resources whose custom handler raised, mapped to the failure reason.
    """

    skipped: list[Resource]
    succeeded: list[Resource]
    failed: dict[Resource, DeletionFailureEvent]


class HandlerStatus(Enum):
    """Outcome of a single handler operation."""

    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class HandlerResult:
    """Result of a single handler operation on a resource."""

    resource_id: str
    resource_type: str
    action: str
    status: HandlerStatus
    message: str = ""


def to_ccapi_resources(resources: list[StackResource]) -> list[Resource]:
    """Convert StackResources to CCAPI Resources, skipping those without a physical ID."""
    return [
        Resource(type=r.resource_type, identifier=r.physical_id) for r in resources if r.physical_id
    ]


def exclude_infra_resources(
    resources: dict[str, list[dict]],
    *,
    keep_types: frozenset[str] = frozenset(),
) -> dict[str, list[dict]]:
    """Remove known infrastructure resources (CDKToolkit, cdk-hnb659fds-*, bootstrap param).

    ``keep_types`` re-includes infra resources whose CloudFormation type is in the
    set — used by an ``include_infra`` sweep to reclaim the CDKToolkit stack's
    retained *regional* assets (see ``SWEEPABLE_INFRA_TYPES``). Every other infra
    resource — critically the global ``cdk-hnb659fds-*`` bootstrap IAM roles —
    stays protected regardless, so an infra sweep can never delete a role a
    surviving stack still needs.
    """
    filtered: dict[str, list[dict]] = {}
    for rtype, items in resources.items():
        if rtype in keep_types:
            filtered[rtype] = list(items)
            continue
        remaining = [item for item in items if not is_infra_identifier(item.get("Identifier", ""))]
        if remaining:
            filtered[rtype] = remaining
    return filtered


@dataclass
class RegionResult:
    """Cleanup result for a single region."""

    region: str
    stacks_found: int = 0
    stacks_deleted: int = 0
    stacks_failed: list[str] = field(default_factory=list)
    orphan_count: int = 0
    error: str = ""

    @property
    def stacks_failed_count(self) -> int:
        """Number of stacks that failed to delete."""
        return len(self.stacks_failed)


@dataclass
class CleanupSummary:
    """Overall deep cleanup summary across all regions."""

    regions: list[RegionResult] = field(default_factory=list)
    orphaned_resources: dict[str, list[str]] = field(default_factory=dict)
    run_dir: str = ""
    scan_incomplete: bool = False
    """The post-cleanup orphan scan did not finish (failed or cancelled).

    When True, an empty ``orphaned_resources`` does NOT mean the account is
    orphan-free — regions went unscanned.
    """

    @property
    def total_stacks(self) -> int:
        """Total stacks found across all regions."""
        return sum(r.stacks_found for r in self.regions)

    @property
    def total_deleted(self) -> int:
        """Total stacks successfully deleted."""
        return sum(r.stacks_deleted for r in self.regions)

    @property
    def total_failed(self) -> int:
        """Total stacks that failed to delete."""
        return sum(r.stacks_failed_count for r in self.regions)

    @property
    def total_orphaned(self) -> int:
        """Total orphaned resources across all types."""
        return sum(len(ids) for ids in self.orphaned_resources.values())

    @property
    def all_stacks_succeeded(self) -> bool:
        """True if every stack was deleted, no region errored, and the scan finished.

        Does not check orphaned_resources (a region can delete every stack yet
        still have orphans). An incomplete scan fails this: an unfinished scan
        can't attest the account is clean.
        """
        return (
            self.total_failed == 0
            and not any(r.error for r in self.regions)
            and not self.scan_incomplete
        )

    @property
    def is_clean(self) -> bool:
        """True iff every stack was deleted and no orphaned resources remain.

        Stricter than :attr:`all_stacks_succeeded`, which stays orphan-agnostic
        for reset and the untag/terminate gate.
        """
        return self.all_stacks_succeeded and self.total_orphaned == 0

    @property
    def failure_reason(self) -> str:
        """Why this summary is not clean, for the cleanup verdict. Empty iff clean.

        Aggregates every non-empty cause, joined by "; " in most- to least-severe
        order, so a single dirty account surfaces all its problems at once rather
        than masking all but the most-severe (a region can raise with no failed
        stacks or orphans, and orphans can coexist with failed stacks).
        """
        causes: list[str] = []
        if self.scan_incomplete:
            causes.append("post-cleanup orphan scan incomplete — cannot attest clean")
        if self.total_failed:
            causes.append(f"{self.total_failed} stack(s) failed to delete")
        if self.total_orphaned:
            causes.append(f"{self.total_orphaned} orphaned resource(s) remain")
        errored = [r.region for r in self.regions if r.error]
        if errored:
            causes.append(f"{len(errored)} region(s) errored during cleanup: {', '.join(errored)}")
        return "; ".join(causes)


@dataclass
class RegionScanAggregate:
    """Aggregated scan results from multiple regions."""

    scan_result: ScanResult
    """Combined scan results from all regions."""

    region_counts: dict[str, int]
    """Count of orphaned resources per region."""


@dataclass
class AccountScanResult:
    """Result of scanning an account for orphaned resources after cleanup."""

    orphaned_resources: dict[str, list[str]]
    """Orphaned resources grouped by resource type, with their identifiers."""

    region_counts: dict[str, int]
    """Count of orphaned resources per region."""

    failed_regions: dict[str, str] = field(default_factory=dict)
    """Regions whose scan failed or was cancelled, keyed to the error.

    Non-empty means the scan was incomplete: an empty ``orphaned_resources``
    does NOT imply the account is clean.
    """


@dataclass
class AccountCleanupResult:
    """Result of cleaning up a single account."""

    account_id: str
    summary: CleanupSummary | None
    error: str | None


@dataclass
class EnvironmentCleanupResult:
    """Result of cleaning up all accounts in an environment."""

    environment_name: str
    results: list[AccountCleanupResult]

    @property
    def succeeded_count(self) -> int:
        """Number of accounts that cleaned successfully."""
        return sum(1 for r in self.results if r.error is None)

    @property
    def failed_count(self) -> int:
        """Number of accounts that failed to clean."""
        return sum(1 for r in self.results if r.error is not None)
