"""Tests for aws_bench.resource_management.cleanup.account_scanner."""

from __future__ import annotations

import json
from unittest.mock import ANY, MagicMock, patch

import pytest

from aws_bench.resource_management.ccapi.models import ScanResult
from aws_bench.resource_management.cleanup.account_scanner import (
    AccountScanner,
    ExistenceCheckCache,
)
from aws_bench.resource_management.cleanup.models import RegionScanAggregate
from aws_bench.resource_management.deferred import deferred_scope, mark_deferred

# -- AccountScanner.run --


def test_returns_empty_for_no_regions(tmp_path):
    result = AccountScanner(MagicMock()).run(tmp_path, [])
    assert result.orphaned_resources == {}
    assert result.region_counts == {}


# -- AccountScanner._scan_region --


def test_scan_region_threads_account_id_into_make_scanner():
    """AccountScanner routes its per-region scan to the management-account Lambda.

    make_scanner only targets the Lambda when given an account_id; without it the
    scan degrades to the throttled host path, where a failed lister lands in
    scan_result.failed and the cleanup phases skip that type — leaving orphaned
    resources undeleted. The account id is what makes those failures surface.
    """
    session = MagicMock()
    scanner = AccountScanner(session, account_id="123456789012")
    with (
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.create_regional_session",
            return_value=session,
        ),
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.make_scanner"
        ) as mock_scanner_cls,
    ):
        mock_scanner_cls.return_value.scan_resources.return_value = ScanResult(
            detected={}, failed={}
        )
        scanner._scan_region("us-east-1", MagicMock())

    assert mock_scanner_cls.call_args.kwargs.get("account_id") == "123456789012"


def test_scan_region_account_id_defaults_to_none():
    """Omitting account_id threads None as a keyword (no positional breakage)."""
    session = MagicMock()
    scanner = AccountScanner(session)
    with (
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.create_regional_session",
            return_value=session,
        ),
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.make_scanner"
        ) as mock_scanner_cls,
    ):
        mock_scanner_cls.return_value.scan_resources.return_value = ScanResult(
            detected={}, failed={}
        )
        scanner._scan_region("us-east-1", MagicMock())

    assert "account_id" in mock_scanner_cls.call_args.kwargs
    assert mock_scanner_cls.call_args.kwargs["account_id"] is None


def test_scan_region_returns_filtered_resources():
    session = MagicMock()
    scanner = AccountScanner(session)
    with (
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.create_regional_session",
            return_value=session,
        ),
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.make_scanner"
        ) as mock_scanner_cls,
    ):
        mock_scanner = MagicMock()
        mock_scanner.scan_resources.return_value = ScanResult(
            detected={"AWS::S3::Bucket": [{"Identifier": "my-bucket"}]},
            failed={},
        )
        mock_scanner_cls.return_value = mock_scanner
        resolver = MagicMock()
        resolver.filter_resources_by_region.return_value = [MagicMock(identifier="my-bucket")]
        result = scanner._scan_region("us-east-1", resolver)
    assert "AWS::S3::Bucket" in result.detected


def _scan_region_with_detected(detected: dict, resource_exists, *, include_infra: bool = False):
    """Run _scan_region with a mocked scanner returning ``detected`` and a patched CCAPI check."""
    session = MagicMock()
    scanner = AccountScanner(session)
    with (
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.create_regional_session",
            return_value=session,
        ),
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.make_scanner"
        ) as mock_scanner_cls,
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.CloudControlManager"
        ) as mock_ccm_cls,
    ):
        mock_scanner = MagicMock()
        mock_scanner.scan_resources.return_value = ScanResult(detected=detected, failed={})
        mock_scanner_cls.return_value = mock_scanner
        mock_ccm = MagicMock()
        mock_ccm.resource_exists.side_effect = resource_exists
        mock_ccm_cls.return_value = mock_ccm
        resolver = MagicMock()
        resolver.filter_resources_by_region.side_effect = lambda region, resources: resources
        return scanner._scan_region("us-east-1", resolver, include_infra=include_infra)


def test_scan_region_drops_orphan_confirmed_absent_by_host_recheck():
    """A phantom orphan (fast-scan lag) that host-side CCAPI reports gone is dropped."""
    result = _scan_region_with_detected(
        {"AWS::Cognito::IdentityPool": [{"Identifier": "us-east-1:ghost"}]},
        resource_exists=lambda resource: False,  # GetResource -> not found
    )
    assert "AWS::Cognito::IdentityPool" not in result.detected


def test_scan_region_verify_absent_false_skips_host_recheck():
    """Change A: the sweep scan (verify_absent=False) returns inventory without any CCAPI check.

    The full-inventory host-side re-check is exactly the F4 storm; the sweep path diffs against
    the baseline first and re-checks only the surviving residuals (via the cleanup manager's
    ``_drop_stale_residuals``), so no CloudControlManager is even constructed here.
    """
    session = MagicMock()
    scanner = AccountScanner(session)
    with (
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.create_regional_session",
            return_value=session,
        ),
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.make_scanner"
        ) as mock_scanner_cls,
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.CloudControlManager"
        ) as mock_ccm_cls,
    ):
        mock_scanner_cls.return_value.scan_resources.return_value = ScanResult(
            detected={"AWS::Cognito::IdentityPool": [{"Identifier": "us-east-1:pool"}]},
            failed={},
        )
        resolver = MagicMock()
        resolver.filter_resources_by_region.side_effect = lambda region, resources: resources
        result = scanner._scan_region("us-east-1", resolver, verify_absent=False)

    # Inventory returned unmodified, and no existence check was performed.
    assert result.detected["AWS::Cognito::IdentityPool"] == [{"Identifier": "us-east-1:pool"}]
    mock_ccm_cls.assert_not_called()


def test_scan_region_keeps_orphan_that_still_exists():
    """A resource the host-side re-check confirms still EXISTS is kept as a real orphan."""
    result = _scan_region_with_detected(
        {"AWS::Cognito::IdentityPool": [{"Identifier": "us-east-1:real"}]},
        resource_exists=lambda resource: True,
    )
    assert result.detected["AWS::Cognito::IdentityPool"] == [{"Identifier": "us-east-1:real"}]


def test_scan_region_keeps_orphan_when_recheck_unsupported_or_errors():
    """CCAPI-unsupported (e.g. ACM) or transient errors must KEEP the orphan, never mask it."""
    from aws_bench.resource_management.ccapi.exceptions import ResourceExistenceCheckError

    def _raises(resource):
        raise ResourceExistenceCheckError("CCAPI does not support type")

    result = _scan_region_with_detected(
        {"AWS::CertificateManager::Certificate": [{"Identifier": "arn:acm:cert/1"}]},
        resource_exists=_raises,
    )
    assert result.detected["AWS::CertificateManager::Certificate"] == [
        {"Identifier": "arn:acm:cert/1"}
    ]


def test_scan_region_excludes_cdk_infra_by_default():
    """By default CDK bootstrap/toolkit resources are filtered out of the scan."""
    result = _scan_region_with_detected(
        {"AWS::S3::Bucket": [{"Identifier": "cdk-hnb659fds-assets-123-us-east-1"}]},
        resource_exists=lambda resource: True,
    )
    assert "AWS::S3::Bucket" not in result.detected


def test_scan_region_keeps_cdk_infra_when_include_infra():
    """With include_infra=True the retained CDKToolkit assets bucket is kept as an orphan."""
    result = _scan_region_with_detected(
        {"AWS::S3::Bucket": [{"Identifier": "cdk-hnb659fds-assets-123-us-east-1"}]},
        resource_exists=lambda resource: True,
        include_infra=True,
    )
    assert result.detected["AWS::S3::Bucket"] == [
        {"Identifier": "cdk-hnb659fds-assets-123-us-east-1"}
    ]


def test_scan_region_excludes_bootstrap_iam_role_even_with_include_infra():
    """include_infra reclaims regional CDK assets but NEVER the bootstrap IAM roles.

    Regression guard for the wedge: the cfn-exec-role is a global resource a
    surviving stack still references as its RoleARN. Even with the infra opt-in on
    (to reach the assets bucket), it must stay filtered out so it is never swept.
    """
    result = _scan_region_with_detected(
        {
            "AWS::S3::Bucket": [{"Identifier": "cdk-hnb659fds-assets-123-us-east-1"}],
            "AWS::IAM::Role": [{"Identifier": "cdk-hnb659fds-cfn-exec-role-123-us-east-1"}],
        },
        resource_exists=lambda resource: True,
        include_infra=True,
    )
    assert result.detected["AWS::S3::Bucket"] == [
        {"Identifier": "cdk-hnb659fds-assets-123-us-east-1"}
    ]
    assert "AWS::IAM::Role" not in result.detected


def test_scan_region_raises_on_shutdown_before_touching_aws():
    """A flag set before the scan starts bails at the worker-entry checkpoint."""
    from aws_bench.exceptions import OperationCancelled
    from aws_bench.utils import concurrent

    scanner = AccountScanner(MagicMock())
    concurrent.reset_shutdown()
    concurrent.request_shutdown()
    try:
        with (
            patch(
                "aws_bench.resource_management.cleanup.account_scanner.create_regional_session"
            ) as mock_session,
            pytest.raises(OperationCancelled),
        ):
            scanner._scan_region("us-east-1", MagicMock())
        mock_session.assert_not_called()
    finally:
        concurrent.reset_shutdown()


def test_scan_region_propagates_mid_scan_cancel():
    """A cancel raised inside the scan propagates, it is not recorded as a failure.

    ``OperationCancelled`` is a ``BaseException``, so it bypasses the region's
    ``except Exception`` and unwinds the scan rather than being turned into a
    failed-region record the run would continue past. A genuine scan error
    (``Exception``) is still recorded — see ``test_scan_region_handles_scan_error``.
    """
    from aws_bench.exceptions import OperationCancelled
    from aws_bench.utils import concurrent

    scanner = AccountScanner(MagicMock())
    concurrent.reset_shutdown()
    try:
        with (
            patch(
                "aws_bench.resource_management.cleanup.account_scanner.create_regional_session",
                return_value=MagicMock(),
            ),
            patch(
                "aws_bench.resource_management.cleanup.account_scanner.make_scanner"
            ) as mock_scanner_cls,
            pytest.raises(OperationCancelled),
        ):
            mock_scanner_cls.return_value.scan_resources.side_effect = OperationCancelled("stop")
            scanner._scan_region("us-east-1", MagicMock())
    finally:
        concurrent.reset_shutdown()


def test_scan_region_handles_scan_error():
    scanner = AccountScanner(MagicMock())
    with patch(
        "aws_bench.resource_management.cleanup.account_scanner.create_regional_session",
        side_effect=Exception("fail"),
    ):
        result = scanner._scan_region("us-east-1", MagicMock())
    assert result.detected == {}
    assert "us-east-1/_scan_error" in result.failed


def test_scan_region_filters_out_non_region_resources():
    session = MagicMock()
    scanner = AccountScanner(session)
    with (
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.create_regional_session",
            return_value=session,
        ),
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.make_scanner"
        ) as mock_scanner_cls,
    ):
        mock_scanner = MagicMock()
        mock_scanner.scan_resources.return_value = ScanResult(
            detected={"AWS::S3::Bucket": [{"Identifier": "other-region-bucket"}]},
            failed={},
        )
        mock_scanner_cls.return_value = mock_scanner
        resolver = MagicMock()
        resolver.filter_resources_by_region.return_value = []
        result = scanner._scan_region("us-east-1", resolver)
    assert "AWS::S3::Bucket" not in result.detected


# -- AccountScanner.scan_region --


def test_scan_region_returns_full_scan_result():
    """The public wrapper returns the full ``_scan_region(region)`` ScanResult.

    Both ``detected`` and ``failed`` must survive so the cleanup phases can skip
    types that failed to enumerate rather than treating them as "new" and
    deleting them.
    """
    scanner = AccountScanner(MagicMock())
    detected = {"AWS::S3::Bucket": [{"Identifier": "live-bucket"}]}
    failed = {"AWS::EC2::VPC": "throttled"}
    scan_result = ScanResult(detected=detected, failed=failed)
    with patch.object(scanner, "_scan_region", return_value=scan_result) as mock_scan:
        result = scanner.scan_region("us-east-1")
    # Passes a fresh RegionResolver through to the internal per-region scan.
    mock_scan.assert_called_once_with("us-east-1", ANY, include_infra=False, verify_absent=True)
    assert result is scan_result
    assert result.detected == detected
    assert result.failed == failed


# -- AccountScanner._scan_all_regions --


def test_scan_all_regions_merges_results():
    session = MagicMock()
    scanner = AccountScanner(session)
    mock_result = ScanResult(
        detected={"AWS::S3::Bucket": [{"Identifier": "b1"}]},
        failed={},
    )
    with patch.object(scanner, "_scan_region", return_value=mock_result):
        aggregate = scanner._scan_all_regions(["us-east-1"])
    assert "AWS::S3::Bucket" in aggregate.scan_result.detected
    assert aggregate.region_counts == {"us-east-1": 1}


def test_scan_all_regions_handles_task_error():
    session = MagicMock()
    scanner = AccountScanner(session)
    with patch.object(scanner, "_scan_region", side_effect=Exception("fail")):
        aggregate = scanner._scan_all_regions(["us-east-1"])
    assert aggregate.scan_result.detected == {}
    assert "us-east-1/_task_error" in aggregate.scan_result.failed
    assert aggregate.scan_result.failed["us-east-1/_task_error"] == "fail"
    assert aggregate.region_counts == {"us-east-1": 0}


def test_scan_all_regions_handles_partial_failure():
    """One region fails, others succeed - all results should be present."""
    session = MagicMock()
    scanner = AccountScanner(session)

    def scan_side_effect(region, resolver, *, include_infra=False):
        if region == "us-west-2":
            raise Exception("network error")
        return ScanResult(
            detected={"AWS::S3::Bucket": [{"Identifier": f"bucket-{region}"}]},
            failed={},
        )

    with patch.object(scanner, "_scan_region", side_effect=scan_side_effect):
        aggregate = scanner._scan_all_regions(["us-east-1", "us-west-2", "eu-west-1"])

    # Successful regions should have results
    assert "AWS::S3::Bucket" in aggregate.scan_result.detected
    assert len(aggregate.scan_result.detected["AWS::S3::Bucket"]) == 2  # us-east-1 and eu-west-1
    identifiers = {item["Identifier"] for item in aggregate.scan_result.detected["AWS::S3::Bucket"]}
    assert "bucket-us-east-1" in identifiers
    assert "bucket-eu-west-1" in identifiers

    # Failed region should be recorded
    assert "us-west-2/_task_error" in aggregate.scan_result.failed
    assert aggregate.region_counts == {"us-east-1": 1, "us-west-2": 0, "eu-west-1": 1}
    assert aggregate.scan_result.failed["us-west-2/_task_error"] == "network error"


def test_scan_all_regions_dedupes_global_resource_across_regions():
    """A global resource surfaced in every region is counted exactly once.

    Global services (e.g. CloudFront) have no region resolver, so
    ``RegionResolver.filter_resources_by_region`` keeps the same resource in every
    region's scan. Merging must dedupe by (type, identifier) so the resource is
    reported once — not once per region — otherwise ``total_orphaned`` and the
    per-region counts are inflated (observed: one CloudFront ConnectionGroup
    listed 3x, total_orphaned 11 instead of 9).
    """
    scanner = AccountScanner(MagicMock())
    regions = ["us-east-1", "us-west-2", "ap-southeast-1"]

    def scan_side_effect(region, resolver, *, include_infra=False):
        # Same global identifier returned by every region (resolver-less path).
        return ScanResult(
            detected={
                "AWS::CloudFront::ConnectionGroup": [
                    {"Identifier": "cg_3GEVQmWo8qj2Oo0EIVooGBMYCla"}
                ]
            },
            failed={},
        )

    with patch.object(scanner, "_scan_region", side_effect=scan_side_effect):
        aggregate = scanner._scan_all_regions(regions)

    connection_groups = aggregate.scan_result.detected["AWS::CloudFront::ConnectionGroup"]
    assert connection_groups == [{"Identifier": "cg_3GEVQmWo8qj2Oo0EIVooGBMYCla"}]
    # Counted once total, attributed to whichever region reported it first
    # (completion order is nondeterministic, so assert the shape, not the region).
    assert sum(aggregate.region_counts.values()) == 1
    assert sorted(aggregate.region_counts.values()) == [0, 0, 1]


def test_scan_all_regions_keeps_distinct_regional_resources():
    """Dedup must not collapse genuinely distinct per-region resources.

    Regional identifiers are unique per region, so a global-resource dedup keyed
    on (type, identifier) leaves regional resources untouched.
    """
    scanner = AccountScanner(MagicMock())

    def scan_side_effect(region, resolver, *, include_infra=False):
        return ScanResult(
            detected={"AWS::EC2::Instance": [{"Identifier": f"i-{region}"}]},
            failed={},
        )

    with patch.object(scanner, "_scan_region", side_effect=scan_side_effect):
        aggregate = scanner._scan_all_regions(["us-east-1", "us-west-2"])

    instances = aggregate.scan_result.detected["AWS::EC2::Instance"]
    assert len(instances) == 2
    assert {i["Identifier"] for i in instances} == {"i-us-east-1", "i-us-west-2"}
    assert aggregate.region_counts == {"us-east-1": 1, "us-west-2": 1}


def test_scan_all_regions_does_not_dedupe_empty_identifiers():
    """Resources without an Identifier are never collapsed onto (rtype, "").

    An empty id carries no identity, so two distinct id-less resources of the
    same type must both survive rather than the second being silently dropped.
    Defensive: CloudControl resources normally always carry an identifier.
    """
    scanner = AccountScanner(MagicMock())

    def scan_side_effect(region, resolver, *, include_infra=False):
        return ScanResult(
            detected={
                "AWS::Foo::Bar": [
                    {"Identifier": "", "Name": "a"},
                    {"Identifier": "", "Name": "b"},
                ]
            },
            failed={},
        )

    with patch.object(scanner, "_scan_region", side_effect=scan_side_effect):
        aggregate = scanner._scan_all_regions(["us-east-1"])

    kept = aggregate.scan_result.detected["AWS::Foo::Bar"]
    assert len(kept) == 2
    assert {item["Name"] for item in kept} == {"a", "b"}
    assert aggregate.region_counts == {"us-east-1": 2}


# -- AccountScanner._write_scan_results --


def test_write_scan_results_creates_json_file(tmp_path):
    """_write_scan_results should create a JSON file with formatted results."""
    scanner = AccountScanner(MagicMock())
    scan_result = ScanResult(
        detected={
            "AWS::S3::Bucket": [{"Identifier": "bucket-1"}, {"Identifier": "bucket-2"}],
            "AWS::EC2::Instance": [{"Identifier": "i-123"}],
        },
        failed={"AWS::Bad::Type": "UnsupportedActionException"},
    )

    scanner._write_scan_results(scan_result, 3, tmp_path)

    output_file = tmp_path / "post_cleanup_scan.json"
    assert output_file.exists()

    with open(output_file) as f:
        data = json.load(f)

    assert data["total_orphaned"] == 3
    assert data["orphaned_resources"]["AWS::S3::Bucket"] == ["bucket-1", "bucket-2"]
    assert data["orphaned_resources"]["AWS::EC2::Instance"] == ["i-123"]
    assert data["types_failed"] == {"AWS::Bad::Type": "UnsupportedActionException"}


def test_write_scan_results_handles_empty_results(tmp_path):
    """_write_scan_results should handle empty scan results correctly."""
    scanner = AccountScanner(MagicMock())
    scan_result = ScanResult(detected={}, failed={})

    scanner._write_scan_results(scan_result, 0, tmp_path)

    output_file = tmp_path / "post_cleanup_scan.json"
    assert output_file.exists()

    with open(output_file) as f:
        data = json.load(f)

    assert data["total_orphaned"] == 0
    assert data["orphaned_resources"] == {}
    assert data["types_failed"] == {}


def test_write_scan_results_extracts_identifiers(tmp_path):
    """_write_scan_results should extract only Identifiers from resource items."""
    scanner = AccountScanner(MagicMock())
    scan_result = ScanResult(
        detected={
            "AWS::S3::Bucket": [
                {"Identifier": "bucket-1", "Properties": {"BucketName": "bucket-1"}},
                {"Identifier": "bucket-2", "Properties": {"BucketName": "bucket-2"}},
            ]
        },
        failed={},
    )

    scanner._write_scan_results(scan_result, 2, tmp_path)

    output_file = tmp_path / "post_cleanup_scan.json"
    with open(output_file) as f:
        data = json.load(f)

    # Should only have identifiers, not full resource objects
    assert data["orphaned_resources"]["AWS::S3::Bucket"] == ["bucket-1", "bucket-2"]
    assert all(isinstance(item, str) for item in data["orphaned_resources"]["AWS::S3::Bucket"])


# -- AccountScanner.run full flow --


def test_run_full_flow(tmp_path):
    session = MagicMock()
    scanner = AccountScanner(session)
    mock_aggregate = RegionScanAggregate(
        scan_result=ScanResult(detected={}, failed={}),
        region_counts={},
    )
    with (
        patch.object(scanner, "_scan_all_regions", return_value=mock_aggregate),
        patch("aws_bench.resource_management.cleanup.account_scanner.write_json"),
    ):
        result = scanner.run(tmp_path, ["us-east-1"])
    assert result.orphaned_resources == {}
    assert result.region_counts == {}


def test_run_surfaces_failed_regions_and_is_not_reported_clean(tmp_path, caplog):
    """A failed region (no orphans seen) yields failed_regions and an INCOMPLETE log.

    An empty orphan set with failures must not read as "clean".
    """
    import logging

    session = MagicMock()
    scanner = AccountScanner(session)
    mock_aggregate = RegionScanAggregate(
        scan_result=ScanResult(detected={}, failed={"us-east-1/_scan_error": "boom"}),
        region_counts={"us-east-1": 0},
    )
    with (
        patch.object(scanner, "_scan_all_regions", return_value=mock_aggregate),
        patch("aws_bench.resource_management.cleanup.account_scanner.write_json"),
        caplog.at_level(logging.WARNING),
    ):
        result = scanner.run(tmp_path, ["us-east-1"])

    assert result.failed_regions == {"us-east-1/_scan_error": "boom"}
    assert "INCOMPLETE" in caplog.text
    # The success verdict line must not fire.
    assert "no orphaned resources found" not in caplog.text


def test_run_ignores_benign_per_lister_failures(tmp_path, caplog):
    """A benign per-lister failure must NOT mark the run INCOMPLETE.

    Under fast-scan the ``failed`` map is always non-empty — hundreds of optional
    per-service List/Describe calls a scenario never uses fail routinely, keyed
    ``<region>/<service>:<Op>``. Only whole-region sentinel keys
    (``<region>/_scan_error`` / ``<region>/_task_error``) are real failures. A run
    whose only failures are per-lister must read as clean.
    """
    import logging

    scanner = AccountScanner(MagicMock())
    mock_aggregate = RegionScanAggregate(
        scan_result=ScanResult(
            detected={},
            failed={"us-east-1/cloudsearch:DescribeDomains": "AccessDenied"},
        ),
        region_counts={"us-east-1": 0},
    )
    with (
        patch.object(scanner, "_scan_all_regions", return_value=mock_aggregate),
        patch("aws_bench.resource_management.cleanup.account_scanner.write_json"),
        caplog.at_level(logging.INFO),
    ):
        result = scanner.run(tmp_path, ["us-east-1"])

    # The benign per-lister failure is not a region failure.
    assert result.failed_regions == {}
    assert "INCOMPLETE" not in caplog.text
    assert "no orphaned resources found" in caplog.text


def test_run_treats_sentinel_key_as_region_failure(tmp_path, caplog):
    """A ``_scan_error`` sentinel amidst benign per-lister failures IS a region failure.

    The sentinel is kept in ``failed_regions`` and drives the INCOMPLETE verdict
    even when benign per-lister failures share the map.
    """
    import logging

    scanner = AccountScanner(MagicMock())
    mock_aggregate = RegionScanAggregate(
        scan_result=ScanResult(
            detected={},
            failed={
                "us-east-1/cloudsearch:DescribeDomains": "AccessDenied",
                "us-east-1/_scan_error": "region blew up",
            },
        ),
        region_counts={"us-east-1": 0},
    )
    with (
        patch.object(scanner, "_scan_all_regions", return_value=mock_aggregate),
        patch("aws_bench.resource_management.cleanup.account_scanner.write_json"),
        caplog.at_level(logging.WARNING),
    ):
        result = scanner.run(tmp_path, ["us-east-1"])

    # Only the sentinel survives into failed_regions; the benign key is dropped.
    assert result.failed_regions == {"us-east-1/_scan_error": "region blew up"}
    assert "INCOMPLETE" in caplog.text


def test_run_logs_orphans(tmp_path):
    session = MagicMock()
    scanner = AccountScanner(session)
    mock_aggregate = RegionScanAggregate(
        scan_result=ScanResult(
            detected={"AWS::S3::Bucket": [{"Identifier": "orphan"}]},
            failed={},
        ),
        region_counts={"us-east-1": 1},
    )
    with (
        patch.object(scanner, "_scan_all_regions", return_value=mock_aggregate),
        patch("aws_bench.resource_management.cleanup.account_scanner.write_json"),
    ):
        result = scanner.run(tmp_path, ["us-east-1"])
    assert "AWS::S3::Bucket" in result.orphaned_resources


def test_run_calls_write_scan_results(tmp_path):
    """Run persists the detected resources and only the region-failure subset.

    The scan's ``failed`` map carries benign per-lister keys under fast-scan; only
    whole-region sentinels are actionable, so the write receives a ScanResult with
    the same detected resources but the failures narrowed to region sentinels.
    """
    session = MagicMock()
    scanner = AccountScanner(session)
    detected = {"AWS::S3::Bucket": [{"Identifier": "b1"}, {"Identifier": "b2"}]}
    mock_scan = ScanResult(
        detected=detected,
        # A benign per-lister failure — not a whole-region sentinel.
        failed={"us-east-1/cloudsearch:DescribeDomains": "error"},
    )
    mock_aggregate = RegionScanAggregate(
        scan_result=mock_scan,
        region_counts={"us-east-1": 2},
    )

    with (
        patch.object(scanner, "_scan_all_regions", return_value=mock_aggregate),
        patch.object(scanner, "_write_scan_results") as mock_write,
    ):
        scanner.run(tmp_path, ["us-east-1"])

    mock_write.assert_called_once()
    written_scan, written_total, written_dir = mock_write.call_args.args
    assert written_scan.detected == detected
    # The benign per-lister failure is dropped; only region sentinels persist.
    assert written_scan.failed == {}
    assert written_total == 2
    assert written_dir == tmp_path


# -- AccountScanner.run — pre-setup baseline subtraction --


def test_run_excludes_predeploy_baseline_resources(tmp_path):
    """Resources present in the pre-setup baseline are not reported as orphans.

    The exclusion runs inside _scan_region, so drive the real scan with CCAPI
    and the region resolver mocked (mocking _scan_region would bypass it).
    """
    session = MagicMock()
    scanner = AccountScanner(session)
    baseline = {"AWS::S3::Bucket": [{"Identifier": "default-bucket"}]}

    with (
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.create_regional_session",
            return_value=session,
        ),
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.make_scanner"
        ) as mock_scanner_cls,
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.RegionResolver"
        ) as mock_resolver_cls,
    ):
        mock_scanner_cls.return_value.scan_resources.return_value = ScanResult(
            detected={
                "AWS::S3::Bucket": [
                    {"Identifier": "default-bucket"},
                    {"Identifier": "scenario-bucket"},
                ]
            },
            failed={},
        )
        # Region resolver keeps both buckets in-region.
        mock_resolver_cls.return_value.filter_resources_by_region.return_value = [
            MagicMock(identifier="default-bucket"),
            MagicMock(identifier="scenario-bucket"),
        ]
        result = scanner.run(tmp_path, ["us-east-1"], predeploy_baseline=baseline)

    # Only the scenario-created bucket remains, reported by identifier.
    assert result.orphaned_resources == {"AWS::S3::Bucket": ["scenario-bucket"]}
    assert result.region_counts["us-east-1"] == 1


def test_run_without_baseline_reports_all(tmp_path):
    """With no baseline, all non-infra resources are reported (legacy behavior)."""
    session = MagicMock()
    scanner = AccountScanner(session)

    with (
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.create_regional_session",
            return_value=session,
        ),
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.make_scanner"
        ) as mock_scanner_cls,
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.RegionResolver"
        ) as mock_resolver_cls,
    ):
        mock_scanner_cls.return_value.scan_resources.return_value = ScanResult(
            detected={"AWS::S3::Bucket": [{"Identifier": "b1"}, {"Identifier": "b2"}]},
            failed={},
        )
        mock_resolver_cls.return_value.filter_resources_by_region.return_value = [
            MagicMock(identifier="b1"),
            MagicMock(identifier="b2"),
        ]
        result = scanner.run(tmp_path, ["us-east-1"], predeploy_baseline=None)

    assert result.region_counts["us-east-1"] == 2


def test_run_baseline_excludes_whole_type(tmp_path):
    """A resource type fully covered by the baseline drops out entirely."""
    session = MagicMock()
    scanner = AccountScanner(session)
    baseline = {"AWS::IAM::Role": [{"Identifier": "default-role"}]}

    with (
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.create_regional_session",
            return_value=session,
        ),
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.make_scanner"
        ) as mock_scanner_cls,
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.RegionResolver"
        ) as mock_resolver_cls,
    ):
        mock_scanner_cls.return_value.scan_resources.return_value = ScanResult(
            detected={"AWS::IAM::Role": [{"Identifier": "default-role"}]},
            failed={},
        )
        mock_resolver_cls.return_value.filter_resources_by_region.return_value = [
            MagicMock(identifier="default-role"),
        ]
        result = scanner.run(tmp_path, ["us-east-1"], predeploy_baseline=baseline)

    assert result.orphaned_resources == {}
    assert result.region_counts["us-east-1"] == 0


# -- AccountScanner._scan_region — deferred-deletion exclusion --


def test_scan_region_excludes_deferred_resources():
    """A resource deferred this run (e.g. a Lambda@Edge master) is not reported as an orphan."""
    session = MagicMock()
    scanner = AccountScanner(session)
    with (
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.create_regional_session",
            return_value=session,
        ),
        patch(
            "aws_bench.resource_management.cleanup.account_scanner.make_scanner"
        ) as mock_scanner_cls,
    ):
        mock_scanner_cls.return_value.scan_resources.return_value = ScanResult(
            detected={
                "AWS::Lambda::Function": [
                    {"Identifier": "cf-edge-headers-json"},
                    {"Identifier": "keep-me"},
                ]
            },
            failed={},
        )
        resolver = MagicMock()
        resolver.filter_resources_by_region.return_value = [
            MagicMock(identifier="cf-edge-headers-json"),
            MagicMock(identifier="keep-me"),
        ]
        with deferred_scope():
            mark_deferred("AWS::Lambda::Function", "cf-edge-headers-json")
            result = scanner._scan_region("us-east-1", resolver)

    assert result.detected == {"AWS::Lambda::Function": [{"Identifier": "keep-me"}]}


# -- ExistenceCheckCache --


def test_existence_check_cache_memoizes_and_defaults_to_none():
    """The cache returns None for an unseen key and the recorded bool otherwise."""
    cache = ExistenceCheckCache()
    assert cache.get("AWS::S3::Bucket", "b1") is None
    cache.record("AWS::S3::Bucket", "b1", confirmed_absent=True)
    cache.record("AWS::SecurityHub::Standard", "std-1", confirmed_absent=False)
    assert cache.get("AWS::S3::Bucket", "b1") is True
    assert cache.get("AWS::SecurityHub::Standard", "std-1") is False
    # Distinct identifiers do not collide.
    assert cache.get("AWS::S3::Bucket", "b2") is None


# -- AccountScanner._confirmed_absent — memoization across waves --


def test_confirmed_absent_caches_absent_and_error_but_not_exists():
    """Absent (drop) and error (keep) are memoized; a still-existing resource is not.

    The broken-handler types error on every wave, so caching the keep is the whole point; an
    absent resource stays absent (cleanup only deletes). An existing resource is re-checked
    because a later sweep phase may delete it.
    """
    from aws_bench.resource_management.ccapi.exceptions import (
        ResourceExistenceHandlerFailureError,
    )

    cache = ExistenceCheckCache()
    scanner = AccountScanner(MagicMock(), existence_cache=cache)
    ccm = MagicMock()

    def exists_side_effect(resource):
        if resource.identifier == "gone":
            return False  # confirmed absent -> drop, cache True
        if resource.identifier == "live":
            return True  # exists -> keep, NOT cached
        raise ResourceExistenceHandlerFailureError("handler failed")  # keep, cache False

    ccm.resource_exists.side_effect = exists_side_effect

    # First wave: three distinct checks.
    assert scanner._confirmed_absent(ccm, "AWS::S3::Bucket", "gone") is True
    assert scanner._confirmed_absent(ccm, "AWS::S3::Bucket", "live") is False
    assert scanner._confirmed_absent(ccm, "AWS::SecurityHub::Standard", "broken") is False
    assert ccm.resource_exists.call_count == 3

    # Second wave: absent + error served from cache (no new calls); exists re-checked.
    assert scanner._confirmed_absent(ccm, "AWS::S3::Bucket", "gone") is True
    assert scanner._confirmed_absent(ccm, "AWS::SecurityHub::Standard", "broken") is False
    assert ccm.resource_exists.call_count == 3  # unchanged — both cached
    assert scanner._confirmed_absent(ccm, "AWS::S3::Bucket", "live") is False
    assert ccm.resource_exists.call_count == 4  # exists re-checked


def test_second_wave_serves_broken_handler_from_cache():
    """Two scan waves share one cache: the broken-handler check runs once, not per wave.

    Mirrors the cleanup structure where CleanupManager builds a fresh AccountScanner per sweep
    wave but threads one shared ExistenceCheckCache through them all.
    """
    from aws_bench.resource_management.ccapi.exceptions import (
        ResourceExistenceHandlerFailureError,
    )

    cache = ExistenceCheckCache()
    detected = {"AWS::ControlTower::EnabledBaseline": [{"Identifier": "eb-1"}]}
    call_count = {"n": 0}

    def exists_side_effect(resource):
        call_count["n"] += 1
        raise ResourceExistenceHandlerFailureError("HandlerInternalFailureException")

    with patch(
        "aws_bench.resource_management.cleanup.account_scanner.CloudControlManager"
    ) as mock_ccm_cls:
        mock_ccm_cls.return_value.resource_exists.side_effect = exists_side_effect
        # Wave 1 and wave 2 each build their own scanner but share the cache.
        for _ in range(2):
            scanner = AccountScanner(MagicMock(), existence_cache=cache)
            kept = scanner.drop_confirmed_absent(MagicMock(), dict(detected))
            # Broken handler errs -> fail-closed keep, every wave.
            assert kept == detected

    assert call_count["n"] == 1  # only the first wave hit CCAPI


def test_confirmed_absent_no_cache_rechecks_every_time():
    """Without a cache (default), every call hits CCAPI — one-shot scans need no memo."""
    scanner = AccountScanner(MagicMock())  # no existence_cache
    ccm = MagicMock()
    ccm.resource_exists.return_value = False  # absent
    assert scanner._confirmed_absent(ccm, "AWS::S3::Bucket", "gone") is True
    assert scanner._confirmed_absent(ccm, "AWS::S3::Bucket", "gone") is True
    assert ccm.resource_exists.call_count == 2


# -- AccountScanner._drop_confirmed_absent — parallel partition + keep-on-error --


def test_drop_confirmed_absent_partitions_across_types_in_parallel():
    """The parallelized check drops confirmed-absent, keeps existing, keeps on error.

    A mixed multi-type batch verifies the flatten/rebuild preserves the keep-on-error
    fail-closed semantic and only drops resources definitively confirmed gone.
    """
    from aws_bench.resource_management.ccapi.exceptions import ResourceExistenceCheckError

    detected = {
        "AWS::S3::Bucket": [{"Identifier": "gone"}, {"Identifier": "live"}],
        "AWS::SecurityHub::Standard": [{"Identifier": "broken"}],
        "AWS::Cognito::IdentityPool": [{"Identifier": "also-gone"}],
    }

    def exists_side_effect(resource):
        if resource.identifier in ("gone", "also-gone"):
            return False  # confirmed absent -> drop
        if resource.identifier == "live":
            return True  # exists -> keep
        raise ResourceExistenceCheckError("handler failed")  # keep on error

    with patch(
        "aws_bench.resource_management.cleanup.account_scanner.CloudControlManager"
    ) as mock_ccm_cls:
        mock_ccm_cls.return_value.resource_exists.side_effect = exists_side_effect
        scanner = AccountScanner(MagicMock())
        result = scanner.drop_confirmed_absent(MagicMock(), detected)

    # Confirmed-absent dropped: the S3 "gone" and the whole Cognito type.
    assert result["AWS::S3::Bucket"] == [{"Identifier": "live"}]
    assert "AWS::Cognito::IdentityPool" not in result
    # Errored broken-handler type kept (fail-closed).
    assert result["AWS::SecurityHub::Standard"] == [{"Identifier": "broken"}]


def test_drop_confirmed_absent_noop_when_empty():
    """No detected resources means no CloudControlManager is even constructed."""
    scanner = AccountScanner(MagicMock())
    with patch(
        "aws_bench.resource_management.cleanup.account_scanner.CloudControlManager"
    ) as mock_ccm_cls:
        assert scanner.drop_confirmed_absent(MagicMock(), {}) == {}
    mock_ccm_cls.assert_not_called()


def test_drop_confirmed_absent_handles_types_with_empty_item_lists():
    """A type key present with an empty list yields no checks (and no zero-worker pool)."""
    scanner = AccountScanner(MagicMock())
    with patch(
        "aws_bench.resource_management.cleanup.account_scanner.CloudControlManager"
    ) as mock_ccm_cls:
        result = scanner.drop_confirmed_absent(MagicMock(), {"AWS::S3::Bucket": []})
    assert result == {}
    mock_ccm_cls.return_value.resource_exists.assert_not_called()
