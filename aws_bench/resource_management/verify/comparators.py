"""Comparison utilities for verification operations."""

from __future__ import annotations

import json
from typing import Callable

# Registry of AWS-managed resource filters.
# Each predicate takes (identifier, resource_dict) and returns True if the resource
# is AWS-managed and should be excluded from drift comparison.
#
# To add a new filter: add one entry to this dict.
AWS_MANAGED_FILTERS: dict[str, Callable[[str, dict], bool]] = {
    # Service-linked roles (created by AWS services)
    "AWS::IAM::Role": lambda id, _: id.startswith("AWSServiceRoleFor"),
    # Service-linked roles are AWS-created and only the owning service can delete them.
    "AWS::IAM::ServiceLinkedRole": lambda id, _: True,
    # AWS-managed policies have no account ID in the ARN (::aws: partition)
    "AWS::IAM::ManagedPolicy": lambda id, _: "arn:aws:iam::aws:policy/" in id,
    # AWS-managed prefix lists have OwnerId == "AWS" in Properties (CCAPI callers only;
    # fast-scan output goes through the caller-scoped CustomListers instead).
    "AWS::EC2::PrefixList": lambda _, r: _is_aws_owned(r),
    # AWS-reserved ``default.*`` parameter groups (redis/memcached/valkey) are undeletable.
    "AWS::ElastiCache::ParameterGroup": lambda id, _: id.startswith("default."),
    # AWS-reserved default RDS/Neptune parameter & option groups are per-engine, account-created,
    # and undeletable ("Default DBParameterGroup cannot be deleted" / "Default option groups cannot
    # be deleted", verified live). The lister emits the group NAME: param groups are
    # ``default.<eng>`` (e.g. default.mysql8.0), option groups are ``default:<eng>``
    # (e.g. default:mysql-8-0). Task/agent-created groups carry a custom name and are
    # NOT filtered.
    "AWS::RDS::DBParameterGroup": lambda id, _: id.startswith("default."),
    "AWS::RDS::OptionGroup": lambda id, _: id.startswith("default:"),
    "AWS::Neptune::DBParameterGroup": lambda id, _: id.startswith("default."),
    # DB *cluster* parameter groups have the same undeletable AWS-reserved ``default.<eng>``
    # defaults (e.g. default.aurora-mysql8.0, default.neptune1.2, default.docdb5.0). The listers
    # emit the NAME, so the prefix match works; task-created groups carry a custom name.
    "AWS::RDS::DBClusterParameterGroup": lambda id, _: id.startswith("default."),
    "AWS::Neptune::DBClusterParameterGroup": lambda id, _: id.startswith("default."),
    "AWS::DocDB::DBClusterParameterGroup": lambda id, _: id.startswith("default."),
    # AWS-reserved ``system*`` keyspaces (system, system_schema, ...) are undeletable.
    "AWS::Cassandra::Keyspace": lambda id, _: id.startswith("system"),
    # Auto-created by AWS services (EKS, ELB) when attaching ENIs.
    "AWS::EC2::NetworkInterfacePermission": lambda id, _: True,
    # Auto-defined resolver rule associations are created by AWS for every VPC
    # (internet-resolver, etc.) and cannot be deleted (RSLVR-00713).
    "AWS::Route53Resolver::ResolverRuleAssociation": lambda id, _: "rslvr-autodefined-" in id,
    # Reserved capacity are billing constructs, not deletable infrastructure.
    "AWS::RDS::ReservedDBInstance": lambda id, _: True,
    "AWS::ElastiCache::ReservedCacheNode": lambda id, _: True,
    # AWS-managed public model hub. Identifier is the hub ARN; anchor to the
    # suffix so a task hub like "SageMakerPublicHubClone" isn't wrongly excluded.
    "AWS::SageMaker::Hub": lambda id, _: (
        id.endswith("hub/SageMakerPublicHub") or id == "SageMakerPublicHub"
    ),
    "AWS::Redshift::ClusterParameterGroup": lambda id, _: id.startswith("default."),
    "AWS::Redshift::ClusterSubnetGroup": lambda id, _: id == "default",
    # AWS-reserved default IoT domain configurations. Custom domains
    # (which cannot use the ``iot:`` prefix) still surface as real orphans.
    "AWS::IoT::DomainConfiguration": lambda id, _: id.startswith("iot:"),
    # AWS-preconfigured default dashboard (documented in AWS S3 docs)
    "AWS::S3::StorageLens": lambda id, _: id == "default-account-dashboard",
    # AWS-managed RAM permissions live in the ::aws: partition (no account id);
    # they are global, undeletable, and surface/disappear on their own, so they
    # are not account drift. Customer-managed permissions carry an account id.
    "AWS::RAM::Permission": lambda id, _: "arn:aws:ram::aws:permission/" in id,
    # AWS-published Well-Architected lenses live in the ::aws: partition (no account id) — global,
    # read-only, undeletable. Customer/custom lenses carry an account id in the ARN.
    "AWS::WellArchitected::Lens": lambda id, _: "arn:aws:wellarchitected::aws:lens/" in id,
    # AWS built-in Bedrock AgentCore evaluators (Builtin.*) are AWS-published and undeletable; their
    # ARN has an empty account (arn:aws:bedrock-agentcore:::evaluator/Builtin.<name>).
    "AWS::BedrockAgentCore::Evaluator": lambda id, _: (
        ":::evaluator/Builtin." in id or id.startswith("Builtin.")
    ),
    # AWS-published RDS/DocDB CA certificates (rds-ca-*) have an empty account in the ARN
    # (arn:aws:rds:region::cert:rds-ca-*); they are global, rotated by AWS, and not account drift.
    "AWS::RDS::Certificate": lambda id, _: id.startswith("rds-ca-") or "::cert:rds-ca-" in id,
    # Automated (system) RDS snapshots are created on the backup schedule and removed by AWS with
    # the parent DB; they cannot be deleted manually ("automated snapshots cannot be deleted",
    # verified live) and regenerate daily, so treating them as drift/residual is pure churn. Their
    # resource name carries the ``rds:`` owner prefix (…:snapshot:rds:… / bare rds:…); MANUAL
    # snapshots (customer/agent-created, deletable) have a plain name and are NOT filtered.
    "AWS::RDS::DBSnapshot": lambda id, _: "snapshot:rds:" in id or id.startswith("rds:"),
    "AWS::RDS::DBClusterSnapshot": lambda id, _: (
        "cluster-snapshot:rds:" in id or id.startswith("rds:")
    ),
    # AWS-reserved KMS aliases (alias/aws/*, bare name or ARN) are AWS-created and undeletable.
    "AWS::KMS::Alias": lambda id, _: id.startswith("alias/aws/") or ":alias/aws/" in id,
    # AWS-created App Runner default auto scaling configuration.
    "AWS::AppRunner::AutoScalingConfiguration": lambda id, _: (
        "autoscalingconfiguration/DefaultConfiguration/" in id or id == "DefaultConfiguration"
    ),
    # AWS-created default SMS opt-out list (undeletable, RESOURCE_MODIFICATION_NOT_ALLOWED).
    "AWS::SMSVOICE::OptOutList": lambda id, _: id.endswith("opt-out-list/Default"),
    # AWS-created default EventBridge event bus (undeletable).
    "AWS::Events::EventBus": lambda id, _: id.endswith("event-bus/default"),
    # AWS-created default MediaConvert queue. Created lazily by the service on first
    # MediaConvert use (so it is absent from the pre-deploy init snapshot) and cannot be
    # deleted, so the init-snapshot diff would otherwise flag it as a leaked orphan. The
    # lister emits the ARN (…:queues/Default). On-demand queues a task creates carry a
    # custom name and are NOT filtered.
    "AWS::MediaConvert::Queue": lambda id, _: id.endswith("queues/Default"),
    # Service-managed secrets (e.g. a Redshift namespace's admin credentials) are
    # created and rotated BY the owning AWS service, which names them with a "!"
    # marker: ``redshift!…``, ``rds!…``, ``aws!…`` (ARN ``…:secret:redshift!…``).
    # They are managed by (and deleted with) their parent resource — a customer
    # cannot delete them directly — so they are not account drift. Redshift can
    # add one to a baseline namespace when a task connects to it (e.g. a Bedrock
    # KB → Redshift link), surfacing it as a "new" resource. Customer/agent
    # secrets carry a plain name with no "!" and are NOT filtered.
    "AWS::SecretsManager::Secret": lambda id, _: _is_service_managed_secret(id),
}


# AWS services mark secrets they own/rotate with a ``<service>!`` PREFIX on the
# secret name (e.g. ``redshift!…``). Match only these KNOWN service prefixes, not
# "!" anywhere: "!" is legal in a customer secret name, so a broad match could
# wrongly drop a real agent-created leftover from new-resource detection. Extend
# this set if a new AWS-managed-secret producer shows up in a scenario.
_SERVICE_MANAGED_SECRET_PREFIXES = ("redshift!", "rds!", "aws!")


def _is_service_managed_secret(identifier: str) -> bool:
    """Whether a Secrets Manager identifier names a service-managed secret.

    The fast-scan identifier may be the bare name or the full ARN
    (``…:secret:redshift!…``); in the ARN the ``<service>!`` prefix is on the
    trailing name segment. Customer/agent secrets have a plain name (no known
    ``<service>!`` prefix) and are NOT filtered.
    """
    name = identifier.split(":secret:")[-1] if ":secret:" in identifier else identifier
    return name.startswith(_SERVICE_MANAGED_SECRET_PREFIXES)


def _is_aws_owned(resource: dict) -> bool:
    """Check if a resource has OwnerId == 'AWS' in its Properties."""
    props = resource.get("Properties", "")
    if not props:
        return False
    try:
        parsed = json.loads(props) if isinstance(props, str) else props
        return str(parsed.get("OwnerId", "")).lower() == "aws"
    except (json.JSONDecodeError, AttributeError):
        return False


def filter_aws_managed_resources(
    resources: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Filter out AWS-managed resources, custom resources, and malformed types.

    Uses the AWS_MANAGED_FILTERS registry to determine which resources are
    AWS-managed. Each resource type can have a predicate that inspects the
    identifier and/or resource properties.
    """
    filtered = {}

    for resource_type, resource_list in resources.items():
        if "::" not in resource_type or resource_type.startswith("Custom::"):
            continue

        predicate = AWS_MANAGED_FILTERS.get(resource_type)
        filtered_list = [
            r
            for r in resource_list
            if r.get("Identifier") and not (predicate and predicate(r["Identifier"], r))
        ]

        if filtered_list:
            filtered[resource_type] = filtered_list

    return filtered


def find_new_resources(
    current: dict[str, list[dict]],
    snapshot: dict[str, list[dict]],
    current_failed: dict[str, str] | None = None,
    snapshot_failed: dict[str, str] | None = None,
) -> dict[str, list[dict]]:
    """Find resources in current that aren't in snapshot.

    Only compares resource types that succeeded in both scans. Resource types
    that failed in either scan are ignored to avoid false positives from
    inconsistent CCAPI scan results.

    Args:
        current: Current resources by type
        snapshot: Baseline snapshot resources by type
        current_failed: Resource types that failed in current scan (optional)
        snapshot_failed: Resource types that failed in baseline scan (optional)

    Returns:
        Dict of new resources by type
    """
    current_filtered = filter_aws_managed_resources(current)
    snapshot_filtered = filter_aws_managed_resources(snapshot)

    # Determine which resource types to compare
    current_failed_types = set(current_failed.keys()) if current_failed else set()
    snapshot_failed_types = set(snapshot_failed.keys()) if snapshot_failed else set()
    failed_types = current_failed_types | snapshot_failed_types

    new = {}
    for resource_type, resources in current_filtered.items():
        # Skip types that failed in either scan
        if resource_type in failed_types:
            continue

        snapshot_ids = {r["Identifier"] for r in snapshot_filtered.get(resource_type, [])}
        current_ids = {r["Identifier"] for r in resources}
        new_ids = current_ids - snapshot_ids

        if new_ids:
            new[resource_type] = [r for r in resources if r["Identifier"] in new_ids]

    return new


def count_resources(resources: dict[str, list[dict]]) -> int:
    """Count total number of resources across all types."""
    return sum(len(resource_list) for resource_list in resources.values())


def normalize_drift(drift: list[dict]) -> list[dict]:
    """Normalize drift for comparison (sort, extract comparable fields)."""
    # Extract comparable fields
    normalized = []
    for item in drift:
        normalized.append(
            {
                "LogicalResourceId": item["LogicalResourceId"],
                "StackResourceDriftStatus": item.get("StackResourceDriftStatus"),
                "PropertyDifferences": item.get("PropertyDifferences", []),
            }
        )

    # Sort by LogicalResourceId for consistent comparison
    return sorted(normalized, key=lambda x: x["LogicalResourceId"])


def drifts_match(current_drift: list[dict], baseline_drift: list[dict]) -> bool:
    """Compare drift states, accounting for ordering."""
    current_normalized = normalize_drift(current_drift)
    baseline_normalized = normalize_drift(baseline_drift)

    return current_normalized == baseline_normalized
