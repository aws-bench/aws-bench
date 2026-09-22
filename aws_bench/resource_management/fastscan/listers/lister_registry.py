"""Assemble the one lister set the scanner runs: the simple listers + the custom listers."""

from __future__ import annotations

from aws_bench.resource_management.fastscan.listers.custom_listers import custom_listers
from aws_bench.resource_management.fastscan.listers.model import Lister
from aws_bench.resource_management.fastscan.listers.simple_listers import SIMPLE_LISTERS


def _simple_listers() -> list[Lister]:
    """Turn every simple-lister data entry into a :class:`Lister`."""
    return [
        Lister.from_row(
            service=entry.service,
            op=entry.op,
            method=entry.method,
            result_path=entry.result_path,
            id_field=entry.id_field,
            cfn_type=entry.cfn_type,
            status_field=entry.status_field,
            status_filter=entry.status_filter,
            status_exclude=entry.status_exclude,
        )
        for entry in SIMPLE_LISTERS
    ]


# Simple-lister ops superseded by a custom lister sharing the same ``service:op`` scan key. Every
# lister writes its ids under one key, so two listers on one key would clobber each other
# nondeterministically (thread-order dependent); dropping the simple lister leaves exactly one
# writer. An op is superseded either because the custom lister emits the CCAPI-correct identifier
# (the simple lister emits a different id for the same CFN type) or because the custom lister also
# pins the discoveries to their exact CFN type (the simple lister would only reach a service noun).
SUPERSEDED_BY_CUSTOM_LISTER: frozenset[tuple[str, str]] = frozenset(
    {
        # Correct-identifier custom lister supersedes the wrong-id simple lister.
        ("redshift", "DescribeClusters"),  # → AWS::Redshift::Cluster (emits ClusterNamespaceArn)
        ("redshift-serverless", "ListNamespaces"),  # → …::Namespace (emits namespaceArn)
        ("redshift-serverless", "ListWorkgroups"),  # → …::Workgroup (emits workgroupArn)
        ("logs", "DescribeLogGroups"),  # → AWS::Logs::LogGroup (emits arn with :* tail)
        ("logs", "ListLogGroups"),  # → AWS::Logs::LogGroup (capped/arn variant)
        (
            "logs",
            "describe_metric_filters",
        ),  # → AWS::Logs::MetricFilter (emits LogGroupName|Filter)
        ("imagebuilder", "ListComponents"),  # → …::Component (emits version ARN, no build tail)
        ("imagebuilder", "ListImages"),  # → …::Image (custom fans out to build-version ARN …/N)
        # CFN-type pin: the simple lister discovers these too, but the custom lister pins each to
        # its exact CFN type; the pinned variant is the sole writer so discoveries reach the right
        # type instead of a service catch-all.
        ("acm-pca", "ListCertificateAuthorities"),  # → AWS::ACMPCA::CertificateAuthority
        ("amp", "ListWorkspaces"),  # → AWS::APS::Workspace
        ("amp", "ListScrapers"),  # → AWS::APS::Scraper
        # The simple no-arg list_services defaults to the "default" cluster and raises
        # ClusterNotFoundException when it is absent (fresh/scenario accounts have none),
        # falsely marking AWS::ECS::Service un-enumerable. The custom lister iterates every
        # cluster and emits the CCAPI composite id serviceArn|cluster.
        ("ecs", "list_services"),  # → AWS::ECS::Service
        ("ec2", "DescribeAddresses"),  # → AWS::EC2::EIP
        # Default-VPC exclusion: the simple row returns the default VPC's associated
        # set, which is undeletable and must not surface as an orphan.
        ("ec2", "DescribeDhcpOptions"),  # → AWS::EC2::DHCPOptions
        # PrefixList and VPCEndpointService: SimpleLister runs the too-broad
        # API (returns AWS-managed lists / all region-available services);
        # the CustomLister runs the caller-scoped API instead.
        ("ec2", "DescribeManagedPrefixLists"),  # → AWS::EC2::PrefixList
        ("ec2", "DescribeVpcEndpointServiceConfigurations"),  # → AWS::EC2::VPCEndpointService
        ("lexv2-models", "ListBots"),  # → AWS::Lex::Bot
        ("mq", "ListBrokers"),  # → AWS::AmazonMQ::Broker
        ("mq", "ListConfigurations"),  # → AWS::AmazonMQ::Configuration
        # WAF Classic ListLoggingConfigurations needs Limit>=1 (no server default); the simple
        # no-arg row fails ValidationException, so the custom lister passing Limit supersedes it.
        ("waf", "ListLoggingConfigurations"),
        ("waf-regional", "ListLoggingConfigurations"),
    }
)


# Simple (data-table) listers never run, each for the human-verified reason on its line. Only
# simple listers — a custom lister with a needs-an-id problem is fixed at its source, not disabled
# by key. Do NOT add an account-not-onboarded lister here: its ValidationException is account
# state, and it must stay running so an onboarded account enumerates it. The billing group below
# is the one exception: it throttle-rejects permanently on a member account (aws-bench only scans
# member accounts), and left running that permanent ThrottlingException aborts the whole snapshot.
DISABLED_LISTERS: frozenset[tuple[str, str]] = frozenset(
    {
        # Needs an id we don't have.
        ("ec2", "DescribeVpcBlockPublicAccessExclusions"),  # needs an ExclusionId / VPC / subnet id
        ("vpc-lattice", "ListServiceNetworkServiceAssociations"),  # needs serviceNetworkIdentifier
        ("vpc-lattice", "ListServiceNetworkVpcAssociations"),  # needs serviceNetworkIdentifier
        # Payer-only: ThrottlingException on every call from a member account (verified against the
        # payer, where the same calls succeed).
        ("billing", "ListBillingViews"),
        ("bcm-data-exports", "ListExports"),
        ("bcm-pricing-calculator", "ListBillEstimates"),
        ("bcm-pricing-calculator", "ListBillScenarios"),
        ("bcm-pricing-calculator", "ListWorkloadEstimates"),
        # SageMaker Edge Manager (near-EOL): ThrottlingException even from the payer.
        ("sagemaker", "list_devices"),
        ("sagemaker", "list_device_fleets"),
        # OpenSearch Serverless: very low rate limits cause transient ThrottlingException that
        # aborts snapshots. No stable scenario uses this service.
        ("opensearchserverless", "ListVpcEndpoints"),
        ("opensearchserverless", "ListCollectionGroups"),
        ("opensearchserverless", "ListCollections"),
        # SageMaker PartnerApp: low-TPS control-plane API that ThrottlingExceptions under the
        # parallel multi-region snapshot and aborts the whole capture (observed in
        # ap-northeast-3). Not a benchmark resource, so disabling the lister is safe.
        ("sagemaker", "ListPartnerApps"),
    }
)


def all_listers() -> list[Lister]:
    """Every lister the scanner runs: simple (minus superseded and disabled) + custom."""
    simple = [
        lister
        for lister in _simple_listers()
        if (lister.service, lister.op) not in SUPERSEDED_BY_CUSTOM_LISTER
        and (lister.service, lister.op) not in DISABLED_LISTERS
    ]
    return simple + list(custom_listers())


def cfn_type_pins() -> dict[str, str]:
    """Map each ``"service:op"`` scan key to the exact CFN type its lister pins itself to."""
    return {
        f"{lister.service}:{lister.op}": lister.cfn_type
        for lister in all_listers()
        if lister.cfn_type is not None
    }
