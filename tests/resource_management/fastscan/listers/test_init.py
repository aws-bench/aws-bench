"""Tests for the assembled lister set: all_listers() + supersession + cfn_type_pins()."""

from aws_bench.resource_management.fastscan.listers import all_listers, cfn_type_pins
from aws_bench.resource_management.fastscan.listers.custom_listers import custom_listers
from aws_bench.resource_management.fastscan.listers.lister_registry import (
    DISABLED_LISTERS,
    SUPERSEDED_BY_CUSTOM_LISTER,
)
from aws_bench.resource_management.fastscan.listers.simple_listers import SIMPLE_LISTERS
from aws_bench.resource_management.fastscan.type_map import cfn_type_resource_nouns, lister_op_noun


def test_all_listers_excludes_superseded_and_disabled():
    keys = {(x.service, x.op) for x in all_listers()}
    code_keys = {(c.service, c.op) for c in custom_listers()}
    for key in SUPERSEDED_BY_CUSTOM_LISTER:
        assert key not in keys or key in code_keys
    for key in DISABLED_LISTERS:
        assert key not in keys, f"disabled lister {key} must not run"


def test_ecs_list_services_simple_lister_is_superseded():
    """The broken no-arg simple ecs:list_services row must be superseded by the custom lister.

    The simple row defaults to the "default" cluster and raises ClusterNotFoundException when
    it is absent, falsely marking AWS::ECS::Service un-enumerable. The custom lister iterates
    every cluster, so only it runs for AWS::ECS::Service.
    """
    assert ("ecs", "list_services") in SUPERSEDED_BY_CUSTOM_LISTER
    # It is a real simple lister (so the supersede is not a dead entry) ...
    assert ("ecs", "list_services") in {(x.service, x.op) for x in SIMPLE_LISTERS}
    # ... and it does not run in the assembled set.
    assert ("ecs", "list_services") not in {(x.service, x.op) for x in all_listers()}
    # The custom lister remains the sole writer for AWS::ECS::Service.
    ecs_service_writers = [
        (c.service, c.op) for c in custom_listers() if c.cfn_type == "AWS::ECS::Service"
    ]
    assert ecs_service_writers == [("ecs", "ListServices")]


def test_disabled_listers_are_real_simple_listers():
    # DISABLED_LISTERS targets simple (data-table) listers only: it suppresses a row by key
    # instead of deleting it. A custom lister is hand-written code — disable-by-key doesn't apply
    # (fix or delete it at its source), so a disabled key MUST be a real simple lister. This also
    # catches a dead entry (typo, or a lister removed since it was disabled) doing nothing.
    simple = {(x.service, x.op) for x in SIMPLE_LISTERS}
    custom = {(x.service, x.op) for x in custom_listers()}
    for key in DISABLED_LISTERS:
        assert key in simple, f"disabled key {key} is not a simple lister"
        assert key not in custom, (
            f"disabled key {key} is custom — fix/remove its code, not disable-by-key"
        )


def test_sagemaker_partner_apps_lister_is_disabled():
    """ListPartnerApps throttles the parallel multi-region snapshot and aborts it.

    Observed in ap-northeast-3. PartnerApp is not a benchmark resource, so the
    lister stays disabled — this guards against it being re-enabled and re-aborting snapshots.
    """
    assert ("sagemaker", "ListPartnerApps") in DISABLED_LISTERS
    assert ("sagemaker", "ListPartnerApps") not in {(x.service, x.op) for x in all_listers()}


def test_region_skip_map_contract():
    # The only in-repo guard for the generated region_skip.py: keys are real listers, values
    # non-empty, and the map is op-level (s3:ListBuckets never skipped, unlike DirectoryBuckets).
    from aws_bench.resource_management.fastscan.listers.region_skip import LISTER_REGION_SKIP

    live = {(lister.service, lister.op) for lister in all_listers()}
    stale = sorted(k for k in LISTER_REGION_SKIP if k not in live)
    assert not stale, f"region_skip references listers not in all_listers(): {stale}"

    empty = sorted(k for k, regions in LISTER_REGION_SKIP.items() if not regions)
    assert not empty, f"region_skip has keys with no regions (generation bug): {empty}"

    # Operation-level, not service-level: the load-bearing invariant. s3:ListBuckets is live in
    # every region; only S3 Express (ListDirectoryBuckets) is regionally absent.
    assert ("s3", "ListBuckets") not in LISTER_REGION_SKIP
    assert ("s3", "ListDirectoryBuckets") in LISTER_REGION_SKIP


def test_unavailable_lister_regions_contract():
    # UNAVAILABLE_LISTER_REGIONS keys must reference real listers with non-empty region sets — a
    # dead key silently does nothing.
    from aws_bench.resource_management.fastscan.listers.region_policy import (
        UNAVAILABLE_LISTER_REGIONS,
    )

    live = {(lister.service, lister.op) for lister in all_listers()}
    stale = sorted(k for k in UNAVAILABLE_LISTER_REGIONS if k not in live)
    assert not stale, f"region_policy references listers not in all_listers(): {stale}"

    empty = sorted(k for k, regions in UNAVAILABLE_LISTER_REGIONS.items() if not regions)
    assert not empty, f"region_policy has keys with no regions: {empty}"


def test_all_listers_has_no_duplicate_scan_keys():
    # Two listers on one "service:op" would clobber each other; the scanner fails loudly on it.
    keys = [f"{lister.service}:{lister.op}" for lister in all_listers()]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, f"duplicate scan keys would crash scan(): {dupes}"


def test_superseded_table_op_never_runs_as_a_table_lister():
    # A superseded op must have at most one writer in the assembled set, and where it survives it
    # is the code lister, not the dropped table row.
    keys = [(lister.service, lister.op) for lister in all_listers()]
    code_keys = {(lister.service, lister.op) for lister in custom_listers()}
    for superseded in SUPERSEDED_BY_CUSTOM_LISTER:
        count = keys.count(superseded)
        assert count <= 1, f"{superseded} must not be run by two listers at once"
        if count == 1:
            assert superseded in code_keys, f"{superseded}'s sole writer must be the code lister"


def test_cfn_type_pins_cover_every_pinned_lister():
    pins = cfn_type_pins()
    pinned = {f"{lister.service}:{lister.op}": lister.cfn_type for lister in all_listers()}
    pinned = {k: v for k, v in pinned.items() if v is not None}
    assert pins == pinned
    # Representative pins from both sources: a code sub-resource pin and a folded data-row pin.
    assert pins["ec2:DescribeAddresses"] == "AWS::EC2::EIP"
    assert pins["acm:list_certificates"] == "AWS::CertificateManager::Certificate"


def test_ambiguous_folded_methods_are_not_pinned():
    # A method feeding several DISTINCT CFN sub-types must stay unpinned (falls to the service
    # catch-all). iam:list_policies and lakeformation:list_permissions were once assumed ambiguous
    # but Phase 2 live-proved each maps to a single CFN type (ManagedPolicy / PrincipalPermissions),
    # so they are now pinned and no longer listed here.
    pins = cfn_type_pins()
    for key in (
        "servicediscovery:list_namespaces",
        "networkmanager:list_attachments",
    ):
        assert key not in pins, f"{key} feeds multiple CFN types and must not be pinned"


def test_every_code_lister_pins_or_matches_its_type():
    """Each code lister either pins a cfn_type or has an op-noun a type could match.

    Guards the attribution bug: a lister with neither would silently fall into the
    AWS::<service>::* catch-all forever. Collects all offenders and asserts the set is empty in
    one shot, so a failure names every offending lister rather than only the first.
    """
    unattributable = [
        f"{lister.service}:{lister.op}"
        for lister in custom_listers()
        if lister.cfn_type is None and not lister_op_noun(lister.op)
    ]
    assert not unattributable, (
        f"listers that neither pin a cfn_type nor yield an op noun: {unattributable}"
    )


def test_pinned_op_nouns_are_consistent_with_their_type_where_expected():
    # For pins that DO align with cfn_type_resource_nouns (the clean ones), lister_op_noun ∈ nouns;
    # the sub-resource pins deliberately don't — those rely purely on the direct cfn_type pin.
    for op, cfn_type in (
        ("ListDistributions", "AWS::CloudFront::Distribution"),
        ("DescribeClusters", "AWS::Redshift::Cluster"),
        ("DescribeLogGroups", "AWS::Logs::LogGroup"),
    ):
        assert lister_op_noun(op) in cfn_type_resource_nouns(cfn_type)
