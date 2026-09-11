"""Tests for the three-phase cleanup flow in CleanupManager."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aws_bench.exceptions import OperationCancelled
from aws_bench.resource_management.ccapi.models import ScanResult
from aws_bench.resource_management.cleanup.manager import CleanupManager
from aws_bench.resource_management.cleanup.models import RegionResult, SnapshotResources


def _snapshot(resource_ids, failed_types=None):
    return SnapshotResources(resource_ids=resource_ids, failed_types=failed_types or {})


@pytest.mark.asyncio
async def test_delete_residuals_deletes_current_minus_setup():
    mgr = CleanupManager(MagicMock(), env_name="scn")
    setup = _snapshot({"AWS::S3::Bucket": [{"Identifier": "baseline-bucket"}]})
    current = ScanResult(
        detected={
            "AWS::S3::Bucket": [{"Identifier": "baseline-bucket"}, {"Identifier": "run-bucket"}]
        },
        failed={},
    )
    with (
        patch.object(mgr, "_scan_region_resources", return_value=current),
        patch("aws_bench.resource_management.cleanup.manager.ResourceSweeper") as RS,
    ):
        RS.return_value.delete = AsyncMock(return_value={})
        await mgr._delete_resources_created_after_setup("us-east-1", setup)
    swept = RS.return_value.delete.call_args.args[0]
    assert swept == {"AWS::S3::Bucket": [{"Identifier": "run-bucket"}]}


@pytest.mark.asyncio
async def test_sweep_deletes_current_minus_init():
    mgr = CleanupManager(MagicMock(), env_name="scn")
    init = _snapshot({"AWS::EC2::VPC": [{"Identifier": "vpc-default"}]})
    current = ScanResult(
        detected={"AWS::EC2::VPC": [{"Identifier": "vpc-default"}, {"Identifier": "vpc-leftover"}]},
        failed={},
    )
    with (
        patch.object(mgr, "_scan_region_resources", return_value=current),
        patch("aws_bench.resource_management.cleanup.manager.ResourceSweeper") as RS,
    ):
        RS.return_value.delete = AsyncMock(return_value={})
        await mgr._delete_resources_not_in_init_snapshot("us-east-1", init)
    swept = RS.return_value.delete.call_args.args[0]
    assert swept == {"AWS::EC2::VPC": [{"Identifier": "vpc-leftover"}]}


@pytest.mark.asyncio
async def test_sweep_does_not_existence_check_baselined_defaults():
    """Change A: a baselined default is excluded by the diff and never CCAPI-checked in a sweep.

    Only the genuine post-diff residual reaches the host-side re-check — the fix for the F4 storm,
    where the sweep re-checked the entire live inventory (incl. ControlTower/SecurityHub defaults)
    on every wave.
    """
    mgr = CleanupManager(MagicMock(), env_name="scn")
    setup = _snapshot({"AWS::ControlTower::EnabledBaseline": [{"Identifier": "eb-default"}]})
    current = ScanResult(
        detected={
            "AWS::ControlTower::EnabledBaseline": [{"Identifier": "eb-default"}],
            "AWS::S3::Bucket": [{"Identifier": "run-bucket"}],
        },
        failed={},
    )
    checked: list[str] = []

    def _exists(resource):
        checked.append(resource.identifier)
        return True  # residual still exists -> kept

    with (
        patch.object(mgr, "_scan_region_resources", return_value=current),
        patch(
            "aws_bench.resource_management.cleanup.manager.create_regional_session",
            return_value=MagicMock(),
        ),
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.CloudControlManager"
        ) as ccm_cls,
        patch("aws_bench.resource_management.cleanup.manager.ResourceSweeper") as RS,
    ):
        ccm_cls.return_value.resource_exists.side_effect = _exists
        RS.return_value.delete = AsyncMock(return_value={})
        await mgr._delete_resources_created_after_setup("us-east-1", setup)

    # The baselined ControlTower default was excluded by the diff and never existence-checked.
    assert "eb-default" not in checked
    # Only the genuine residual was host-side re-checked, and (still existing) swept.
    assert checked == ["run-bucket"]
    swept = RS.return_value.delete.call_args.args[0]
    assert swept == {"AWS::S3::Bucket": [{"Identifier": "run-bucket"}]}


@pytest.mark.asyncio
async def test_sweep_drops_residual_confirmed_absent_before_delete():
    """Change A regression: a residual the host-side re-check confirms gone is dropped, not swept."""  # noqa: E501
    mgr = CleanupManager(MagicMock(), env_name="scn")
    setup = _snapshot({})
    current = ScanResult(detected={"AWS::S3::Bucket": [{"Identifier": "ghost-bucket"}]}, failed={})
    with (
        patch.object(mgr, "_scan_region_resources", return_value=current),
        patch(
            "aws_bench.resource_management.cleanup.manager.create_regional_session",
            return_value=MagicMock(),
        ),
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.CloudControlManager"
        ) as ccm_cls,
        patch("aws_bench.resource_management.cleanup.manager.ResourceSweeper") as RS,
    ):
        ccm_cls.return_value.resource_exists.return_value = False  # confirmed gone
        RS.return_value.delete = AsyncMock(return_value={})
        await mgr._delete_resources_created_after_setup("us-east-1", setup)

    # Residual confirmed gone -> nothing to sweep.
    RS.return_value.delete.assert_not_called()


@pytest.mark.asyncio
async def test_residual_phase_skipped_when_no_setup_snapshot():
    mgr = CleanupManager(MagicMock(), env_name="scn")
    with (
        patch.object(mgr, "_scan_region_resources") as scan,
        patch("aws_bench.resource_management.cleanup.manager.ResourceSweeper") as RS,
    ):
        await mgr._delete_resources_created_after_setup("us-east-1", None)
    scan.assert_not_called()
    RS.assert_not_called()


# -- FIX 1: failed-type guard on the destructive diff --


@pytest.mark.asyncio
async def test_phase1_skips_type_that_failed_at_snapshot():
    """A type that failed to enumerate at setup is NOT swept even if it appears live.

    Without the failed-type guard, a baseline enumeration failure looks like the
    type had zero resources, so every live resource of that type reads as "new"
    and gets deleted — e.g. a default VPC.
    """
    mgr = CleanupManager(MagicMock(), env_name="scn")
    setup = _snapshot(
        resource_ids={},  # VPC absent because enumeration failed at baseline
        failed_types={"AWS::EC2::VPC": "throttled"},
    )
    current = ScanResult(
        detected={"AWS::EC2::VPC": [{"Identifier": "vpc-default"}]},
        failed={},
    )
    with (
        patch.object(mgr, "_scan_region_resources", return_value=current),
        patch("aws_bench.resource_management.cleanup.manager.ResourceSweeper") as RS,
    ):
        RS.return_value.delete = AsyncMock(return_value={})
        await mgr._delete_resources_created_after_setup("us-east-1", setup)
    RS.return_value.delete.assert_not_called()


@pytest.mark.asyncio
async def test_phase3_skips_type_that_failed_live():
    """A type that failed to enumerate in the live scan is NOT swept."""
    mgr = CleanupManager(MagicMock(), env_name="scn")
    init = _snapshot({"AWS::EC2::VPC": [{"Identifier": "vpc-default"}]})
    current = ScanResult(
        detected={"AWS::EC2::VPC": [{"Identifier": "vpc-default"}]},
        failed={"AWS::EC2::VPC": "throttled"},
    )
    with (
        patch.object(mgr, "_scan_region_resources", return_value=current),
        patch("aws_bench.resource_management.cleanup.manager.ResourceSweeper") as RS,
    ):
        RS.return_value.delete = AsyncMock(return_value={})
        await mgr._delete_resources_not_in_init_snapshot("us-east-1", init)
    RS.return_value.delete.assert_not_called()


# -- FIX 2: fail closed when the init snapshot is absent --


@pytest.mark.asyncio
async def test_phase3_skipped_when_no_init_snapshot(caplog):
    """A missing init snapshot must FAIL CLOSED: skip the sweep, warn loudly."""
    mgr = CleanupManager(MagicMock(), env_name="scn")
    with (
        patch.object(mgr, "_scan_region_resources") as scan,
        patch("aws_bench.resource_management.cleanup.manager.ResourceSweeper") as RS,
        caplog.at_level(logging.WARNING),
    ):
        await mgr._delete_resources_not_in_init_snapshot("us-east-1", None)
    scan.assert_not_called()
    RS.assert_not_called()
    assert "init snapshot" in caplog.text.lower()


@pytest.mark.asyncio
async def test_phase3_sweeps_against_present_but_empty_init_snapshot():
    """A genuinely present (empty) init snapshot still sweeps."""
    mgr = CleanupManager(MagicMock(), env_name="scn")
    init = _snapshot({})  # present, empty
    current = ScanResult(
        detected={"AWS::S3::Bucket": [{"Identifier": "leftover"}]},
        failed={},
    )
    with (
        patch.object(mgr, "_scan_region_resources", return_value=current),
        patch("aws_bench.resource_management.cleanup.manager.ResourceSweeper") as RS,
    ):
        RS.return_value.delete = AsyncMock(return_value={})
        await mgr._delete_resources_not_in_init_snapshot("us-east-1", init)
    swept = RS.return_value.delete.call_args.args[0]
    assert swept == {"AWS::S3::Bucket": [{"Identifier": "leftover"}]}


# -- FIX 3: keep the phase-2 region result if phase 3 raises --


@pytest.mark.asyncio
async def test_phase3_error_still_returns_phase2_result():
    """A generic error in phase 3 is absorbed; the phase-2 stack result survives."""
    mgr = CleanupManager(MagicMock(), env_name="scn")
    phase2_result = RegionResult(region="us-east-1", stacks_found=3, stacks_deleted=3)
    with (
        patch.object(
            mgr, "_load_snapshot_resource_ids", side_effect=[_snapshot({}), _snapshot({})]
        ),
        patch.object(mgr, "_delete_resources_created_after_setup", AsyncMock()),
        patch.object(mgr, "_cleanup_single_region", AsyncMock(return_value=phase2_result)),
        patch.object(
            mgr,
            "_delete_resources_not_in_init_snapshot",
            AsyncMock(side_effect=RuntimeError("phase 3 blew up")),
        ),
    ):
        result = await mgr._cleanup_region_in_phases("us-east-1", MagicMock())
    assert result is phase2_result
    assert result.stacks_deleted == 3


@pytest.mark.asyncio
async def test_phase_cancellation_propagates():
    """A cooperative-shutdown cancel in a phase must propagate, not be absorbed."""
    mgr = CleanupManager(MagicMock(), env_name="scn")
    phase2_result = RegionResult(region="us-east-1", stacks_found=1, stacks_deleted=1)
    with (
        patch.object(
            mgr, "_load_snapshot_resource_ids", side_effect=[_snapshot({}), _snapshot({})]
        ),
        patch.object(mgr, "_delete_resources_created_after_setup", AsyncMock()),
        patch.object(mgr, "_cleanup_single_region", AsyncMock(return_value=phase2_result)),
        patch.object(
            mgr,
            "_delete_resources_not_in_init_snapshot",
            AsyncMock(side_effect=OperationCancelled("shutdown")),
        ),
        pytest.raises(OperationCancelled),
    ):
        await mgr._cleanup_region_in_phases("us-east-1", MagicMock())


# -- FIX 5: surface ResourceSweeper.delete failures --


@pytest.mark.asyncio
async def test_phase1_warns_on_sweep_failures(caplog):
    """When delete returns failures, phase 1 logs a WARNING summarizing them."""
    mgr = CleanupManager(MagicMock(), env_name="scn")
    setup = _snapshot({})
    current = ScanResult(
        detected={"AWS::S3::Bucket": [{"Identifier": "run-bucket"}]},
        failed={},
    )
    failure_dict = {MagicMock(): MagicMock()}
    with (
        patch.object(mgr, "_scan_region_resources", return_value=current),
        patch("aws_bench.resource_management.cleanup.manager.ResourceSweeper") as RS,
        caplog.at_level(logging.WARNING),
    ):
        RS.return_value.delete = AsyncMock(return_value=failure_dict)
        await mgr._delete_resources_created_after_setup("us-east-1", setup)
    assert "could not" in caplog.text.lower() or "failed to sweep" in caplog.text.lower()


@pytest.mark.asyncio
async def test_phase3_warns_on_sweep_failures(caplog):
    """When delete returns failures, phase 3 logs a WARNING summarizing them."""
    mgr = CleanupManager(MagicMock(), env_name="scn")
    init = _snapshot({})
    current = ScanResult(
        detected={"AWS::S3::Bucket": [{"Identifier": "leftover"}]},
        failed={},
    )
    failure_dict = {MagicMock(): MagicMock()}
    with (
        patch.object(mgr, "_scan_region_resources", return_value=current),
        patch("aws_bench.resource_management.cleanup.manager.ResourceSweeper") as RS,
        caplog.at_level(logging.WARNING),
    ):
        RS.return_value.delete = AsyncMock(return_value=failure_dict)
        await mgr._delete_resources_not_in_init_snapshot("us-east-1", init)
    assert "could not" in caplog.text.lower() or "failed to sweep" in caplog.text.lower()


# ============================================================================
# Regression tests for the Phase-3 bootstrap/global-IAM-role wedge fix.
#
# Bug: the concurrent per-region Phase-3 sweep deleted the CDK cfn-exec-role and
# app custom-resource provider roles (global IAM) out from under stacks that were
# still DELETE_FAILED — in the same region and, because IAM is global, in other
# regions still mid-teardown. CloudFormation could then no longer assume the role
# to delete those stacks, wedging them unrecoverably.
# ============================================================================


# -- Fix 2: global resources are withheld from the per-region sweeps --


@pytest.mark.asyncio
async def test_phase1_withholds_global_iam_roles():
    """Phase 1 sweeps regional residuals only; global IAM is left for the barrier sweep."""
    mgr = CleanupManager(MagicMock(), env_name="scn")
    setup = _snapshot({})
    current = ScanResult(
        detected={
            "AWS::S3::Bucket": [{"Identifier": "run-bucket"}],
            "AWS::IAM::Role": [{"Identifier": "run-role"}],
        },
        failed={},
    )
    with (
        patch.object(mgr, "_scan_region_resources", return_value=current),
        patch("aws_bench.resource_management.cleanup.manager.ResourceSweeper") as RS,
    ):
        RS.return_value.delete = AsyncMock(return_value={})
        await mgr._delete_resources_created_after_setup("us-east-1", setup)
    swept = RS.return_value.delete.call_args.args[0]
    assert swept == {"AWS::S3::Bucket": [{"Identifier": "run-bucket"}]}
    assert "AWS::IAM::Role" not in swept


@pytest.mark.asyncio
async def test_phase3_withholds_global_iam_roles():
    """Phase 3 sweeps regional leftovers only; global IAM is never swept per-region."""
    mgr = CleanupManager(MagicMock(), env_name="scn")
    init = _snapshot({})
    current = ScanResult(
        detected={
            "AWS::EC2::Subnet": [{"Identifier": "subnet-leftover"}],
            "AWS::IAM::Role": [{"Identifier": "provider-role-leftover"}],
        },
        failed={},
    )
    with (
        patch.object(mgr, "_scan_region_resources", return_value=current),
        patch("aws_bench.resource_management.cleanup.manager.ResourceSweeper") as RS,
    ):
        RS.return_value.delete = AsyncMock(return_value={})
        await mgr._delete_resources_not_in_init_snapshot("us-east-1", init)
    swept = RS.return_value.delete.call_args.args[0]
    assert swept == {"AWS::EC2::Subnet": [{"Identifier": "subnet-leftover"}]}
    assert "AWS::IAM::Role" not in swept


# -- Phase 3 runs as last-resort cleanup regardless of phase 2 outcome --


@pytest.mark.asyncio
async def test_phase3_runs_even_when_region_has_surviving_stacks():
    """Phase 3 runs even when a stack survived phase 2.

    Orphaned resources (agent-created, not stack-managed) must still be reaped;
    the sweep is best-effort.
    """
    mgr = CleanupManager(MagicMock(), env_name="scn")
    phase2 = RegionResult(
        region="us-east-1", stacks_found=3, stacks_deleted=2, stacks_failed=["stuck-stack"]
    )
    sweep = AsyncMock()
    with (
        patch.object(
            mgr, "_load_snapshot_resource_ids", side_effect=[_snapshot({}), _snapshot({})]
        ),
        patch.object(mgr, "_delete_resources_created_after_setup", AsyncMock()),
        patch.object(mgr, "_cleanup_single_region", AsyncMock(return_value=phase2)),
        patch.object(mgr, "_delete_resources_not_in_init_snapshot", sweep),
    ):
        result = await mgr._cleanup_region_in_phases("us-east-1", MagicMock())
    assert result is phase2
    sweep.assert_called_once()


@pytest.mark.asyncio
async def test_phase3_runs_when_region_fully_clean():
    """A region that deleted every stack DOES run its Phase-3 regional sweep."""
    mgr = CleanupManager(MagicMock(), env_name="scn")
    phase2 = RegionResult(region="us-east-1", stacks_found=2, stacks_deleted=2)
    with (
        patch.object(
            mgr, "_load_snapshot_resource_ids", side_effect=[_snapshot({}), _snapshot({})]
        ),
        patch.object(mgr, "_delete_resources_created_after_setup", AsyncMock()),
        patch.object(mgr, "_cleanup_single_region", AsyncMock(return_value=phase2)),
        patch.object(mgr, "_delete_resources_not_in_init_snapshot", AsyncMock()) as sweep,
    ):
        await mgr._cleanup_region_in_phases("us-east-1", MagicMock())
    sweep.assert_called_once()


@pytest.mark.asyncio
async def test_phase3_runs_even_when_region_errored():
    """Phase 3 runs even when the region errored — last-resort cleanup."""
    mgr = CleanupManager(MagicMock(), env_name="scn")
    phase2 = RegionResult(region="us-east-1", error="boom")
    sweep = AsyncMock()
    with (
        patch.object(
            mgr, "_load_snapshot_resource_ids", side_effect=[_snapshot({}), _snapshot({})]
        ),
        patch.object(mgr, "_delete_resources_created_after_setup", AsyncMock()),
        patch.object(mgr, "_cleanup_single_region", AsyncMock(return_value=phase2)),
        patch.object(mgr, "_delete_resources_not_in_init_snapshot", sweep),
    ):
        await mgr._cleanup_region_in_phases("us-east-1", MagicMock())
    sweep.assert_called_once()


# -- Fix 2: the global sweep runs once, behind the all-regions barrier --


@pytest.mark.asyncio
async def test_global_sweep_runs_even_when_region_has_surviving_stacks():
    """Global sweep always runs as last-resort cleanup regardless of phase-2 outcome.

    Orphaned global resources (agent-created IAM roles etc.) must be reaped even
    when stacks survived. The sweep itself is best-effort.
    """
    mgr = CleanupManager(MagicMock(), env_name="scn")
    clean = RegionResult(region="us-east-1", stacks_found=1, stacks_deleted=1)
    stuck = RegionResult(region="us-west-2", stacks_found=1, stacks_failed=["stuck"])
    with (
        patch.object(mgr, "_resolve_regions", AsyncMock(return_value=["us-east-1", "us-west-2"])),
        patch.object(mgr, "_write_metadata"),
        patch.object(mgr, "_cleanup_region_in_phases", AsyncMock(side_effect=[clean, stuck])),
        patch.object(mgr, "_sweep_global_leftovers", AsyncMock()) as global_sweep,
        patch.object(
            mgr,
            "_scan_orphaned_resources",
            AsyncMock(
                return_value=MagicMock(orphaned_resources={}, region_counts={}, failed_regions={})
            ),
        ),
        patch.object(mgr, "_log_summary"),
        patch.object(mgr, "_save_summary"),
    ):
        await mgr._cleanup_all_stacks_impl(["us-east-1", "us-west-2"])
    global_sweep.assert_called_once()


@pytest.mark.asyncio
async def test_global_sweep_runs_once_when_all_regions_clean():
    """With every region's stacks gone, the account-wide sweep runs exactly once."""
    mgr = CleanupManager(MagicMock(), env_name="scn")
    r1 = RegionResult(region="us-east-1", stacks_found=1, stacks_deleted=1)
    r2 = RegionResult(region="us-west-2", stacks_found=1, stacks_deleted=1)
    with (
        patch.object(mgr, "_resolve_regions", AsyncMock(return_value=["us-east-1", "us-west-2"])),
        patch.object(mgr, "_write_metadata"),
        patch.object(mgr, "_cleanup_region_in_phases", AsyncMock(side_effect=[r1, r2])),
        patch.object(mgr, "_sweep_global_leftovers", AsyncMock()) as global_sweep,
        patch.object(
            mgr,
            "_scan_orphaned_resources",
            AsyncMock(
                return_value=MagicMock(orphaned_resources={}, region_counts={}, failed_regions={})
            ),
        ),
        patch.object(mgr, "_log_summary"),
        patch.object(mgr, "_save_summary"),
    ):
        await mgr._cleanup_all_stacks_impl(["us-east-1", "us-west-2"])
    global_sweep.assert_called_once()


# -- _sweep_global_leftovers behavior --


@pytest.mark.asyncio
async def test_sweep_global_leftovers_deletes_only_global_diff():
    """Sweeps global (current − init) leftovers; regional leftovers are left alone."""
    mgr = CleanupManager(MagicMock(), env_name="scn")
    init = _snapshot({"AWS::IAM::Role": [{"Identifier": "baseline-role"}]})
    current = ScanResult(
        detected={
            "AWS::IAM::Role": [{"Identifier": "baseline-role"}, {"Identifier": "provider-role"}],
            "AWS::EC2::Subnet": [{"Identifier": "subnet-x"}],
        },
        failed={},
    )
    with (
        patch.object(mgr, "_load_snapshot_resource_ids", return_value=init),
        patch.object(mgr, "_scan_region_resources", return_value=current),
        patch("aws_bench.resource_management.cleanup.manager.ResourceSweeper") as RS,
    ):
        RS.return_value.delete = AsyncMock(return_value={})
        await mgr._sweep_global_leftovers(["us-east-1"])
    swept = RS.return_value.delete.call_args.args[0]
    assert swept == {"AWS::IAM::Role": [{"Identifier": "provider-role"}]}


@pytest.mark.asyncio
async def test_global_sweep_proceeds_for_deferred_only_region():
    """A region whose only non-deleted stack is *deferred* does not block the sweep.

    A deferred stack sets neither ``stacks_failed`` nor ``error`` (it is a distinct
    third bucket), so it is not a survivor. That is intentional and safe: deferral
    only happens when the sole blockers are requester-managed ENIs — the stack's
    custom-resource provider roles are already done, and its cfn-exec-role is
    infra-protected — so sweeping account-wide leftovers cannot strand it.
    """
    mgr = CleanupManager(MagicMock(), env_name="scn")
    # stacks_found=1 but neither deleted nor failed -> the stack was deferred.
    deferred_region = RegionResult(
        region="us-east-1", stacks_found=1, stacks_deleted=0, stacks_failed=[]
    )
    with (
        patch.object(mgr, "_resolve_regions", AsyncMock(return_value=["us-east-1"])),
        patch.object(mgr, "_write_metadata"),
        patch.object(mgr, "_cleanup_region_in_phases", AsyncMock(side_effect=[deferred_region])),
        patch.object(mgr, "_sweep_global_leftovers", AsyncMock()) as global_sweep,
        patch.object(
            mgr,
            "_scan_orphaned_resources",
            AsyncMock(
                return_value=MagicMock(orphaned_resources={}, region_counts={}, failed_regions={})
            ),
        ),
        patch.object(mgr, "_log_summary"),
        patch.object(mgr, "_save_summary"),
    ):
        await mgr._cleanup_all_stacks_impl(["us-east-1"])
    global_sweep.assert_called_once()


@pytest.mark.asyncio
async def test_sweep_global_leftovers_fails_closed_without_init(caplog):
    """No init snapshot → skip the global sweep (never delete the account's own roles)."""
    mgr = CleanupManager(MagicMock(), env_name="scn")
    with (
        patch.object(mgr, "_load_snapshot_resource_ids", return_value=None),
        patch.object(mgr, "_scan_region_resources") as scan,
        patch("aws_bench.resource_management.cleanup.manager.ResourceSweeper") as RS,
        caplog.at_level(logging.WARNING),
    ):
        await mgr._sweep_global_leftovers(["us-east-1"])
    scan.assert_not_called()
    RS.assert_not_called()
    assert "init snapshot" in caplog.text.lower()


@pytest.mark.asyncio
async def test_sweep_global_leftovers_scans_all_regions_and_dedups():
    """Region-homed globals are not missed; a truly-global resource is swept once.

    Regression guard for the single-region assumption: ``ResourceExplorer2::Index``
    is a GLOBAL_RESOURCE_TYPE but region-homed — it only surfaces in its own
    region's scan and its delete must target that region. A truly-global IAM role
    surfaces in every region and must be deleted exactly once (deduped).
    """
    mgr = CleanupManager(MagicMock(), env_name="scn")
    init = _snapshot({})
    index_arn = "arn:aws:resource-explorer-2:us-east-1:123:index/abc"
    east = ScanResult(
        detected={
            "AWS::IAM::Role": [{"Identifier": "shared-role"}],
            "AWS::ResourceExplorer2::Index": [{"Identifier": index_arn}],
        },
        failed={},
    )
    west = ScanResult(
        detected={"AWS::IAM::Role": [{"Identifier": "shared-role"}]},  # same global role
        failed={},
    )

    def scan_side_effect(region, *, include_infra=False):
        return east if region == "us-east-1" else west

    with (
        patch.object(mgr, "_load_snapshot_resource_ids", return_value=init),
        patch.object(mgr, "_scan_region_resources", side_effect=scan_side_effect),
        patch("aws_bench.resource_management.cleanup.manager.ResourceSweeper") as RS,
    ):
        RS.return_value.delete = AsyncMock(return_value={})
        await mgr._sweep_global_leftovers(["us-east-1", "us-west-2"])

    # Aggregate everything swept across all delete() calls.
    swept: dict[str, list[str]] = {}
    for call in RS.return_value.delete.call_args_list:
        for rtype, items in call.args[0].items():
            swept.setdefault(rtype, []).extend(i["Identifier"] for i in items)

    # Region-homed index is reaped (would be missed by a single-region scan)...
    assert swept.get("AWS::ResourceExplorer2::Index") == [index_arn]
    # ...and the truly-global role is deleted exactly once despite appearing twice.
    assert swept.get("AWS::IAM::Role", []).count("shared-role") == 1
