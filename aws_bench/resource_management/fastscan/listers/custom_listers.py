"""The listers that need real Python — filtering, per-parent fan-out, manual pagination."""

from __future__ import annotations

import json

from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.fastscan.listers.model import Lister, SessionLike
from aws_bench.resource_management.fastscan.runtime import RETRY_CONFIG, collect

logger = get_logger(__name__)


# CloudFormation stack statuses that mean the stack still exists (excludes DELETE_COMPLETE).
_CFN_ACTIVE_STATUSES = [
    "CREATE_IN_PROGRESS",
    "CREATE_COMPLETE",
    "CREATE_FAILED",
    "ROLLBACK_IN_PROGRESS",
    "ROLLBACK_COMPLETE",
    "ROLLBACK_FAILED",
    "DELETE_FAILED",
    "UPDATE_IN_PROGRESS",
    "UPDATE_COMPLETE",
    "UPDATE_COMPLETE_CLEANUP_IN_PROGRESS",
    "UPDATE_ROLLBACK_IN_PROGRESS",
    "UPDATE_ROLLBACK_COMPLETE",
    "UPDATE_ROLLBACK_COMPLETE_CLEANUP_IN_PROGRESS",
    "UPDATE_ROLLBACK_FAILED",
    "UPDATE_FAILED",
    "REVIEW_IN_PROGRESS",
    "IMPORT_IN_PROGRESS",
    "IMPORT_COMPLETE",
    "IMPORT_ROLLBACK_IN_PROGRESS",
    "IMPORT_ROLLBACK_COMPLETE",
    "IMPORT_ROLLBACK_FAILED",
]

# CloudFormation stack statuses that mean the stack is gone / going (excluded from discovery).
_CFN_TERMINAL_STATUSES = {"DELETE_COMPLETE", "DELETE_IN_PROGRESS"}

# SageMaker trial-component primary statuses that mean the component is being/has been removed.
_SAGEMAKER_DELETING_STATES = {
    "deleting",
    "delet_in_progress",
    "delete_in_progress",
    "delete_complete",
    "deleted",
    "terminated",
    "pending_deletion",
    "scheduledfordeletion",
}

# Redshift ClusterStatus values that mean the cluster is being deleted (slow shutdown). A cluster
# in one of these is on its way out, not an orphan; every other state still surfaces.
_REDSHIFT_DELETING_STATES = {"deleting", "final-snapshot"}


# --- List* listers with real logic ----------------------------------------------------


def list_bedrock_inference_profiles(session: SessionLike) -> list[str]:
    """Customer inference profiles (excludes AWS SYSTEM_DEFINED profiles)."""
    client = session.client("bedrock", config=RETRY_CONFIG)
    return [
        p["inferenceProfileArn"]
        for page in client.get_paginator("list_inference_profiles").paginate()
        for p in page.get("inferenceProfileSummaries", [])
        if p.get("type") != "SYSTEM_DEFINED"
    ]


def list_bedrock_prompt_routers(session: SessionLike) -> list[str]:
    """Customer prompt routers (excludes AWS default routers)."""
    client = session.client("bedrock", config=RETRY_CONFIG)
    return [
        p["promptRouterArn"]
        for page in client.get_paginator("list_prompt_routers").paginate()
        for p in page.get("promptRouterSummaries", [])
        if p.get("type") != "default"
    ]


def list_cloudformation_stacks(session: SessionLike) -> list[str]:
    """Stack ids in any non-deleted status."""
    client = session.client("cloudformation", config=RETRY_CONFIG)
    paginator = client.get_paginator("list_stacks")
    return [
        s["StackId"]
        for page in paginator.paginate(StackStatusFilter=_CFN_ACTIVE_STATUSES)
        for s in page.get("StackSummaries", [])
    ]


def list_elasticbeanstalk_platform_versions(session: SessionLike) -> list[str]:
    """Self-owned Elastic Beanstalk platform versions."""
    client = session.client("elasticbeanstalk", config=RETRY_CONFIG)
    self_only = [{"Type": "PlatformOwner", "Operator": "=", "Values": ["self"]}]
    return [
        p["PlatformArn"]
        for page in client.get_paginator("list_platform_versions").paginate(Filters=self_only)
        for p in page.get("PlatformSummaryList", [])
    ]


def list_cognito_identity_pools(session: SessionLike) -> list[str]:
    """Cognito identity pool ids.

    ``ListIdentityPools`` requires ``MaxResults`` (so the spec generator's no-arg probe skipped
    it); the paginator supplies it via ``PageSize``, yielding all pools.
    """
    client = session.client("cognito-identity", config=RETRY_CONFIG)
    return [
        p["IdentityPoolId"]
        for page in client.get_paginator("list_identity_pools").paginate(
            PaginationConfig={"PageSize": 60}
        )
        for p in page.get("IdentityPools", [])
    ]


def list_waf_logging_configurations(session: SessionLike) -> list[str]:
    """WAF Classic (global) logging-config ARNs.

    ``ListLoggingConfigurations`` requires ``Limit`` >= 1 with no server default; PageSize supplies
    it (a no-arg call omits Limit and fails ValidationException).
    """
    client = session.client("waf", config=RETRY_CONFIG)
    return [
        c["ResourceArn"]
        for page in client.get_paginator("list_logging_configurations").paginate(
            PaginationConfig={"PageSize": 100}
        )
        for c in page.get("LoggingConfigurations", [])
        if c.get("ResourceArn")
    ]


def list_waf_regional_logging_configurations(session: SessionLike) -> list[str]:
    """WAF Classic Regional logging-config ARNs.

    Same required-``Limit`` as the global lister, but botocore ships no paginator for
    waf-regional, so Limit goes on a direct call.
    """
    client = session.client("waf-regional", config=RETRY_CONFIG)
    return [
        c["ResourceArn"]
        for c in client.list_logging_configurations(Limit=100).get("LoggingConfigurations", [])
        if c.get("ResourceArn")
    ]


def _wafv2_scopes_for_region(region: str) -> tuple[str, ...]:
    """Scopes to list in this region: REGIONAL always; CLOUDFRONT only in us-east-1.

    WAFv2's ``ListWebACLs``/``ListIPSets`` reject ``Scope=CLOUDFRONT`` outside us-east-1
    (WAFInvalidParameterException); CLOUDFRONT-scoped resources are global and surfaced by the
    us-east-1 scan.
    """
    return ("REGIONAL", "CLOUDFRONT") if region == "us-east-1" else ("REGIONAL",)


def list_wafv2_web_acls(session: SessionLike) -> list[str]:
    """WAFv2 web ACLs across scope, keyed by the CFN composite id ``Name|Id|Scope``.

    WAFv2 has no boto paginator and its list ops are scope-partitioned, so this fans out over the
    scopes valid in the client's region. The emitted ``Name|Id|Scope`` is CCAPI's primaryIdentifier
    (verified live), so the CCAPI delete fallback round-trips; a bare id or ARN would not. Without
    this lister a surviving WebACL (4 scenarios create one) is an undetected orphan.
    """
    client = session.client("wafv2", config=RETRY_CONFIG)
    region = client.meta.region_name
    out: list[str] = []
    for scope in _wafv2_scopes_for_region(region):
        for acl in client.list_web_acls(Scope=scope).get("WebACLs", []):
            out.append(f"{acl['Name']}|{acl['Id']}|{scope}")
    return out


def list_wafv2_ip_sets(session: SessionLike) -> list[str]:
    """WAFv2 IP sets across scope, keyed by the CFN composite id ``Name|Id|Scope``.

    Same scope fan-out and composite-identifier rationale as :func:`list_wafv2_web_acls`.
    """
    client = session.client("wafv2", config=RETRY_CONFIG)
    region = client.meta.region_name
    out: list[str] = []
    for scope in _wafv2_scopes_for_region(region):
        for ip_set in client.list_ip_sets(Scope=scope).get("IPSets", []):
            out.append(f"{ip_set['Name']}|{ip_set['Id']}|{scope}")
    return out


def list_iam_policies(session: SessionLike) -> list[str]:
    """Customer-managed IAM policies (Scope=Local excludes AWS-managed)."""
    client = session.client("iam", config=RETRY_CONFIG)
    return [
        p["Arn"]
        for page in client.get_paginator("list_policies").paginate(Scope="Local")
        for p in page["Policies"]
    ]


def list_iam_roles(session: SessionLike) -> list[str]:
    """IAM roles excluding AWS service-linked roles (/aws-service-role/).

    Emits RoleName (not the ARN): AWS::IAM::Role's CCAPI primaryIdentifier is RoleName, and
    delete_resource with the ARN fails (InvalidRequestException) — verified live.
    """
    client = session.client("iam", config=RETRY_CONFIG)
    return [
        r["RoleName"]
        for page in client.get_paginator("list_roles").paginate()
        for r in page["Roles"]
        if not r["Path"].startswith("/aws-service-role/")
    ]


def list_kms_aliases(session: SessionLike) -> list[str]:
    """Customer KMS aliases (excludes AWS-managed alias/aws/* aliases).

    Emits AliasName (``alias/NAME``, not the ARN): AWS::KMS::Alias's CCAPI primaryIdentifier is
    AliasName, so the ARN would not resolve for delete.
    """
    client = session.client("kms", config=RETRY_CONFIG)
    return [
        a["AliasName"]
        for page in client.get_paginator("list_aliases").paginate()
        for a in page.get("Aliases", [])
        if not a.get("AliasName", "").startswith("alias/aws/")
    ]


def list_kms_keys(session: SessionLike) -> list[str]:
    """Customer-managed KMS keys (excludes AWS-managed and pending-deletion keys)."""
    client = session.client("kms", config=RETRY_CONFIG)
    keys: list[str] = []
    for page in client.get_paginator("list_keys").paginate():
        for k in page.get("Keys", []):
            try:
                meta = client.describe_key(KeyId=k["KeyId"])["KeyMetadata"]
            except (ClientError, BotoCoreError) as exc:
                # A key we can't describe is skipped (it isn't a confirmable customer key),
                # but log it: a throttle/AccessDenied here silently drops a key that MAY be
                # customer-managed, so the drop must leave a trace rather than vanish.
                logger.warning(f"kms.describe_key skipped {k.get('KeyId')}: {exc}")
                continue
            if meta.get("KeyManager") != "AWS" and meta.get("KeyState") not in (
                "PendingDeletion",
                "PendingReplicaDeletion",
            ):
                keys.append(k["KeyArn"])
    return keys


def list_medialive_cloud_watch_alarm_template_groups(session: SessionLike) -> list[str]:
    """Customer-scoped (LOCAL) MediaLive CloudWatch alarm-template groups."""
    client = session.client("medialive", config=RETRY_CONFIG)
    return [
        g["Arn"]
        for page in client.get_paginator("list_cloud_watch_alarm_template_groups").paginate(
            Scope="LOCAL"
        )
        for g in page.get("CloudWatchAlarmTemplateGroups", [])
    ]


def list_medialive_cloud_watch_alarm_templates(session: SessionLike) -> list[str]:
    """Customer-scoped (LOCAL) MediaLive CloudWatch alarm templates."""
    client = session.client("medialive", config=RETRY_CONFIG)
    return [
        t["Arn"]
        for page in client.get_paginator("list_cloud_watch_alarm_templates").paginate(Scope="LOCAL")
        for t in page.get("CloudWatchAlarmTemplates", [])
    ]


def list_ram_permissions(session: SessionLike) -> list[str]:
    """Customer-managed RAM permissions (manual pagination — no boto3 paginator)."""
    client = session.client("ram", config=RETRY_CONFIG)
    arns: list[str] = []
    token = None
    while True:
        kw = {"nextToken": token} if token else {}
        resp = client.list_permissions(**kw)
        for p in resp.get("permissions", []):
            if p.get("permissionType") == "CUSTOMER_MANAGED" and p.get("arn"):
                arns.append(p["arn"])
        token = resp.get("nextToken")
        if not token:
            break
    return arns


def list_sagemaker_trial_components(session: SessionLike) -> list[dict]:
    """Live SageMaker trial components (excludes ones being/already deleted)."""
    client = session.client("sagemaker", config=RETRY_CONFIG)
    results: list[dict] = []
    for page in client.get_paginator("list_trial_components").paginate():
        for tc in page.get("TrialComponentSummaries", []):
            status_obj = tc.get("Status") or {}
            primary = (status_obj.get("PrimaryStatus") or "").strip().lower()
            if primary in _SAGEMAKER_DELETING_STATES:
                continue
            results.append(
                {
                    "id": tc.get("TrialComponentArn"),
                    "name": tc.get("TrialComponentName"),
                    "status": primary,
                    "type": "ListTrialComponents",
                    "service": "sagemaker",
                }
            )
    return results


def list_sns_subscriptions(session: SessionLike) -> list[str]:
    """SNS subscriptions with a real ARN (skips PendingConfirmation placeholders)."""
    arns = collect(
        session.client("sns", config=RETRY_CONFIG),
        "list_subscriptions",
        "Subscriptions",
        "SubscriptionArn",
    )
    return [a for a in arns if a and a.startswith("arn:")]


def list_ssm_documents(session: SessionLike) -> list[str]:
    """Self-owned SSM documents (Owner=Self excludes AWS-owned documents)."""
    client = session.client("ssm", config=RETRY_CONFIG)
    return [
        d["Name"]
        for page in client.get_paginator("list_documents").paginate(
            Filters=[{"Key": "Owner", "Values": ["Self"]}]
        )
        for d in page["DocumentIdentifiers"]
    ]


# --- Describe* listers with real logic -------------------------------------------------


def describe_appstream_images(session: SessionLike) -> list[str]:
    """Customer (PRIVATE) AppStream images."""
    client = session.client("appstream", config=RETRY_CONFIG)
    return [img["Arn"] for img in client.describe_images(Type="PRIVATE").get("Images", [])]


def describe_cloudformation_stacks(session: SessionLike) -> list[dict]:
    """Non-deleted CloudFormation stacks, as id dicts (carries name/status for reporting)."""
    client = session.client("cloudformation", config=RETRY_CONFIG)
    out: list[dict] = []
    for page in client.get_paginator("describe_stacks").paginate():
        for s in page.get("Stacks", []):
            if s.get("StackStatus") in _CFN_TERMINAL_STATUSES:
                continue
            out.append(
                {
                    "id": s.get("StackId") or s.get("StackName"),
                    "name": s.get("StackName"),
                    "status": s.get("StackStatus"),
                    "type": "cloudformation:DescribeStacks",
                    "service": "cloudformation",
                }
            )
    return out


def describe_ec2_fpga_images(session: SessionLike) -> list[str]:
    """Self-owned EC2 FPGA images."""
    client = session.client("ec2", config=RETRY_CONFIG)
    return [
        img["FpgaImageId"]
        for page in client.get_paginator("describe_fpga_images").paginate(Owners=["self"])
        for img in page["FpgaImages"]
    ]


def describe_ec2_images(session: SessionLike) -> list[str]:
    """Self-owned EC2 AMIs."""
    client = session.client("ec2", config=RETRY_CONFIG)
    return [
        img["ImageId"]
        for page in client.get_paginator("describe_images").paginate(Owners=["self"])
        for img in page["Images"]
    ]


def describe_ec2_nat_gateways(session: SessionLike) -> list[str]:
    """NAT gateways in a live state (pending / available / failed)."""
    client = session.client("ec2", config=RETRY_CONFIG)
    out: list[str] = []
    for page in client.get_paginator("describe_nat_gateways").paginate(
        Filter=[{"Name": "state", "Values": ["pending", "available", "failed"]}]
    ):
        for ng in page["NatGateways"]:
            out.append(ng["NatGatewayId"])
    return out


def describe_ec2_snapshots(session: SessionLike) -> list[str]:
    """Self-owned EBS snapshots."""
    client = session.client("ec2", config=RETRY_CONFIG)
    return [
        s["SnapshotId"]
        for page in client.get_paginator("describe_snapshots").paginate(OwnerIds=["self"])
        for s in page["Snapshots"]
    ]


def describe_ec2_prefix_lists(session: SessionLike) -> list[str]:
    """Managed prefix lists owned by the caller (AWS-managed lists filtered server-side)."""
    sts = session.client("sts", config=RETRY_CONFIG)
    account_id = sts.get_caller_identity()["Account"]
    client = session.client("ec2", config=RETRY_CONFIG)
    out: list[str] = []
    for page in client.get_paginator("describe_managed_prefix_lists").paginate(
        Filters=[{"Name": "owner-id", "Values": [account_id]}]
    ):
        for pl in page.get("PrefixLists", []):
            pl_id = pl.get("PrefixListId")
            if pl_id:
                out.append(pl_id)
    return out


def describe_ec2_vpc_endpoint_services(session: SessionLike) -> list[str]:
    """VPC endpoint services published by the caller.

    Uses ``describe_vpc_endpoint_service_configurations`` (caller-scoped) instead of
    ``describe_vpc_endpoint_services`` (returns every service available in the region).
    """
    client = session.client("ec2", config=RETRY_CONFIG)
    out: list[str] = []
    for page in client.get_paginator("describe_vpc_endpoint_service_configurations").paginate():
        for svc in page.get("ServiceConfigurations", []):
            svc_id = svc.get("ServiceId")
            if svc_id:
                out.append(svc_id)
    return out


# --- Supplementary listers: standard paginated ----------------------------------------


def list_cloudfront_list_distributions(session: SessionLike) -> list[str]:
    """CloudFront distributions by bare Id (the CCAPI primaryIdentifier /properties/Id).

    CCAPI's DeleteResource accepts the bare distribution id; the ARN gets a
    ResourceNotFoundException, so a detected distribution emitted as an ARN was undeletable.
    """
    return collect(
        session.client("cloudfront", config=RETRY_CONFIG),
        "list_distributions",
        "DistributionList.Items",
        "Id",
    )


def list_cloudfront_list_cloud_front_origin_access_identities(session: SessionLike) -> list[str]:
    """CloudFront origin access identities (ids)."""
    return collect(
        session.client("cloudfront", config=RETRY_CONFIG),
        "list_cloud_front_origin_access_identities",
        "CloudFrontOriginAccessIdentityList.Items",
        "Id",
    )


def list_cloudfront_connection_groups(session: SessionLike) -> list[str]:
    """CloudFront connection group ids, excluding the AWS-managed default group.

    CloudFront auto-creates exactly one default connection group per account
    (``IsDefault=True``, Name ``CreatedByCloudFront-*``). It is undeletable ("You
    cannot disable the default connection group", verified live) and regenerates on
    its own, so it must never be reported as an orphan. Task/agent-created groups
    (``IsDefault=False``) ARE returned. Filtered here at the lister — not by a
    downstream AWS-managed predicate — because fast-scan emits only identifiers, so
    the ``IsDefault`` property is not visible at filter time. ``list_connection_groups``
    has no boto3 paginator, so pages are walked manually via ``Marker``/``NextMarker``.
    """
    client = session.client("cloudfront", config=RETRY_CONFIG)
    ids: list[str] = []
    marker: str | None = None
    while True:
        resp = client.list_connection_groups(**({"Marker": marker} if marker else {}))
        for group in resp.get("ConnectionGroups", []):
            if group.get("Id") and not group.get("IsDefault", False):
                ids.append(group["Id"])
        marker = resp.get("NextMarker")
        if not marker:
            break
    return ids


def list_acm_pca_list_certificate_authorities(session: SessionLike) -> list[str]:
    """ACM PCA certificate authorities (ARNs), excluding deleted ones.

    A deleted CA lingers in the DELETED state (restore window) and DELETE_IN_PROGRESS is
    transient; neither is an orphan, so keep only live states.
    """
    return collect(
        session.client("acm-pca", config=RETRY_CONFIG),
        "list_certificate_authorities",
        "CertificateAuthorities",
        "Arn",
        status_field="Status",
        status_filter=["CREATING", "PENDING_CERTIFICATE", "ACTIVE", "DISABLED", "EXPIRED"],
    )


def list_amp_list_workspaces(session: SessionLike) -> list[str]:
    """Amazon Managed Prometheus (APS) workspaces (ARNs)."""
    return collect(
        session.client("amp", config=RETRY_CONFIG), "list_workspaces", "workspaces", "arn"
    )


def list_amp_list_scrapers(session: SessionLike) -> list[str]:
    """Amazon Managed Prometheus (APS) scrapers (ARNs)."""
    return collect(session.client("amp", config=RETRY_CONFIG), "list_scrapers", "scrapers", "arn")


def list_mq_list_brokers(session: SessionLike) -> list[str]:
    """Amazon MQ brokers by BrokerId (CCAPI primaryIdentifier), excluding brokers being deleted."""
    return collect(
        session.client("mq", config=RETRY_CONFIG),
        "list_brokers",
        "BrokerSummaries",
        "BrokerId",
        status_field="BrokerState",
        status_exclude=["DELETION_IN_PROGRESS"],
    )


def list_cognito_idp_list_user_pools(session: SessionLike) -> list[str]:
    """Cognito user pools (ids)."""
    client = session.client("cognito-idp", config=RETRY_CONFIG)
    out: list[str] = []
    for page in client.get_paginator("list_user_pools").paginate(MaxResults=60):
        out.extend(p["Id"] for p in page.get("UserPools", []))
    return out


# --- Supplementary listers: irregular pagination (no paginator / required args / per-parent) --


def list_ecs_describe_capacity_providers(session: SessionLike) -> list[str]:
    """ECS capacity providers (ARNs), excluding AWS-managed FARGATE(_SPOT) (manual paging)."""
    client = session.client("ecs", config=RETRY_CONFIG)
    out: list[str] = []
    token: str | None = None
    while True:
        kw = {"nextToken": token} if token else {}
        resp = client.describe_capacity_providers(**kw)
        for cp in resp.get("capacityProviders", []):
            if cp.get("name") not in ("FARGATE", "FARGATE_SPOT"):
                out.append(cp["capacityProviderArn"])
        token = resp.get("nextToken")
        if not token:
            return out


def list_cloudfront_list_functions(session: SessionLike) -> list[str]:
    """CloudFront functions (ARNs). list_functions has no paginator (uses Marker)."""
    client = session.client("cloudfront", config=RETRY_CONFIG)
    out: list[str] = []
    marker: str | None = None
    while True:
        kw = {"Marker": marker} if marker else {}
        resp = client.list_functions(**kw)
        fl = resp.get("FunctionList", {})
        for f in fl.get("Items", []):
            md = f.get("FunctionMetadata", {})
            if md.get("FunctionARN"):
                out.append(md["FunctionARN"])
        marker = fl.get("NextMarker")
        if not marker:
            return out


def list_mq_list_configurations(session: SessionLike) -> list[str]:
    """Amazon MQ configurations by bare Id (CCAPI primaryIdentifier). Paginates by NextToken."""
    client = session.client("mq", config=RETRY_CONFIG)
    out: list[str] = []
    token: str | None = None
    while True:
        kw = {"NextToken": token} if token else {}
        resp = client.list_configurations(**kw)
        out.extend(c["Id"] for c in resp.get("Configurations", []) if c.get("Id"))
        token = resp.get("NextToken")
        if not token:
            return out


def list_lexv2_models_list_bots(session: SessionLike) -> list[str]:
    """Lex V2 bots (ids). list_bots paginates by nextToken."""
    client = session.client("lexv2-models", config=RETRY_CONFIG)
    out: list[str] = []
    token: str | None = None
    while True:
        kw = {"nextToken": token} if token else {}
        resp = client.list_bots(**kw)
        out.extend(b["botId"] for b in resp.get("botSummaries", []) if b.get("botId"))
        token = resp.get("nextToken")
        if not token:
            return out


def list_backup_list_backup_selections(session: SessionLike) -> list[str]:
    """Backup selections across every plan, as CCAPI's ``<SelectionId>_<BackupPlanId>`` id."""
    client = session.client("backup", config=RETRY_CONFIG)
    plan_ids: list[str] = []
    for page in client.get_paginator("list_backup_plans").paginate():
        plan_ids.extend(p["BackupPlanId"] for p in page.get("BackupPlansList", []))
    out: list[str] = []
    for plan_id in plan_ids:
        for page in client.get_paginator("list_backup_selections").paginate(BackupPlanId=plan_id):
            out.extend(
                f"{s['SelectionId']}_{plan_id}"
                for s in page.get("BackupSelectionsList", [])
                if s.get("SelectionId")
            )
    return out


# --- Supplementary listers: EC2 drift-parity describers -------------------------------
# CCAPI enumerates these EC2 sub-resources (rules, associations, attachments, EIPs) as
# distinct types; for DRIFT detection fast-scan must see them too. All are plain describe
# calls; the id CCAPI uses is a field already in the response (verified live).


def list_ec2_describe_addresses(session: SessionLike) -> list[str]:
    """Elastic IPs as the CCAPI composite primaryIdentifier ``PublicIp|AllocationId``.

    CCAPI rejects the bare ``AllocationId`` with ``ValidationException``; the composite is
    required for the CCAPI delete + orphan re-check to work.
    """
    client = session.client("ec2", config=RETRY_CONFIG)
    return [
        f"{a['PublicIp']}|{a['AllocationId']}"
        for a in client.describe_addresses().get("Addresses", [])
        if a.get("AllocationId") and a.get("PublicIp")
    ]


def list_ec2_describe_eip_associations(session: SessionLike) -> list[str]:
    """Elastic-IP associations (eipassoc-...) from describe_addresses."""
    client = session.client("ec2", config=RETRY_CONFIG)
    return [
        a["AssociationId"]
        for a in client.describe_addresses().get("Addresses", [])
        if a.get("AssociationId")
    ]


def list_ec2_describe_nic_attachments(session: SessionLike) -> list[str]:
    """Network-interface attachments (eni-attach-/ela-attach-...) via describe_network_ifaces."""
    client = session.client("ec2", config=RETRY_CONFIG)
    out: list[str] = []
    for page in client.get_paginator("describe_network_interfaces").paginate():
        for ni in page.get("NetworkInterfaces", []):
            att = ni.get("Attachment") or {}
            if att.get("AttachmentId"):
                out.append(att["AttachmentId"])
    return out


def list_ec2_describe_volume_attachments(session: SessionLike) -> list[str]:
    """Volume attachments (vol-... attachment) from describe_volumes Attachments."""
    client = session.client("ec2", config=RETRY_CONFIG)
    out: list[str] = []
    for page in client.get_paginator("describe_volumes").paginate():
        for v in page.get("Volumes", []):
            for att in v.get("Attachments", []):
                if att.get("VolumeId"):
                    out.append(att["VolumeId"])
    return out


def default_vpcs(client) -> list[dict]:
    """The region's default VPC(s) via the server-side ``is-default`` filter.

    AWS provisions exactly one default VPC per enabled region (asynchronously,
    shortly after account creation); zero if it was explicitly deleted. Used to
    classify default-VPC components by their CURRENT role — attached/associated
    to the default VPC — so no baseline snapshot is needed to exclude them.

    Shared with the cleanup IGW-detach hook (``cleanup.handlers.vpc``). It lives
    HERE, not there, because these listers run inside the fast-scan Lambda whose
    zip closure does not ship ``cleanup/`` — the import must point cleanup →
    fastscan. Carries no error policy: listers let a failure propagate (the scan
    runtime records it and consumers skip the type), the detach hook wraps it
    fail-open.
    """
    return [
        vpc
        for page in client.get_paginator("describe_vpcs").paginate(
            Filters=[{"Name": "is-default", "Values": ["true"]}]
        )
        for vpc in page.get("Vpcs", [])
    ]


def default_vpc_ids(client) -> set[str]:
    """Ids of the region's default VPC(s)."""
    return {vpc["VpcId"] for vpc in default_vpcs(client) if vpc.get("VpcId")}


def list_ec2_describe_subnet_route_table_associations(session: SessionLike) -> list[str]:
    """Route-table association ids (rtbassoc-...), excluding each default VPC's main association.

    Every association (incl. main-table ones) is returned to match CCAPI, EXCEPT the
    implicit main route-table association of a *default* VPC. That association is an
    AWS-created, undeletable artifact ("cannot delete the main route table
    association") that regenerates with the default VPC, so it must never be reported
    as an orphan (it survives ``env cleanup`` and would fail the post-cleanup scan).
    Filtered here at the lister — not by a downstream AWS-managed predicate — because
    fast-scan emits only identifiers, so ``Main`` / ``VpcId`` are not visible at filter
    time. Task/agent-created associations — the main association of a NON-default VPC
    and every explicit subnet association — are still returned.
    """
    client = session.client("ec2", config=RETRY_CONFIG)
    default_ids = default_vpc_ids(client)
    out: list[str] = []
    for page in client.get_paginator("describe_route_tables").paginate():
        for rt in page.get("RouteTables", []):
            for a in rt.get("Associations", []):
                assoc_id = a.get("RouteTableAssociationId")
                if not assoc_id:
                    continue
                # Skip only the default VPC's implicit main association (undeletable).
                if a.get("Main") and rt.get("VpcId") in default_ids:
                    continue
                out.append(assoc_id)
    return out


def list_ec2_describe_subnet_nacl_associations(session: SessionLike) -> list[str]:
    """Subnet↔NACL associations (aclassoc-...), excluding each default VPC's default-NACL ones."""
    client = session.client("ec2", config=RETRY_CONFIG)
    default_ids = default_vpc_ids(client)
    out: list[str] = []
    for page in client.get_paginator("describe_network_acls").paginate():
        for n in page.get("NetworkAcls", []):
            # Skip the default VPC's default-NACL associations (undeletable, regenerating).
            if n.get("IsDefault") and n.get("VpcId") in default_ids:
                continue
            for a in n.get("Associations", []):
                if a.get("NetworkAclAssociationId"):
                    out.append(a["NetworkAclAssociationId"])
    return out


def list_ec2_describe_vpc_gateway_attachments(session: SessionLike) -> list[str]:
    """Internet-gateway↔VPC attachments (CCAPI id ``IGW|<vpc>``), excluding the default VPC's.

    The default VPC's IGW attachment is AWS-created with the account and detaching
    it degrades the default VPC (no egress; AWS never re-attaches), so it must
    never surface as an orphan for the sweep. A scenario VPC's attachment dies
    with its stack and is still returned.
    """
    client = session.client("ec2", config=RETRY_CONFIG)
    default_ids = default_vpc_ids(client)
    out: list[str] = []
    for page in client.get_paginator("describe_internet_gateways").paginate():
        for igw in page.get("InternetGateways", []):
            for att in igw.get("Attachments", []):
                if att.get("VpcId") and att["VpcId"] not in default_ids:
                    out.append(f"IGW|{att['VpcId']}")
    return out


# Default-VPC components (VPC, subnets, IGW + attachment, main route table, DHCP
# options, SG/NACL) are AWS-provisioned, undeletable in place, and required by
# scenarios that ``Vpc.fromLookup(isDefault)`` — they must never surface as
# orphans. The pre-setup baseline can't be trusted to exclude them: default-VPC
# provisioning is asynchronous, and a snapshot taken minutes after account
# creation can miss components (observed live), permanently poisoning the
# baseline. These listers therefore classify by CURRENT role — attached or
# associated to the ``IsDefault`` VPC — which needs no baseline and self-heals
# poisoned ones. Filtered here because fast-scan emits only identifiers, so the
# attachment/association is invisible downstream.


def list_ec2_describe_all_security_groups(session: SessionLike) -> list[str]:
    """All security groups EXCEPT each VPC's default (undeletable).

    The default security group (GroupName='default') cannot be deleted — AWS rejects
    the call with "cannot delete a default security group". Excluding it from the scan
    prevents it from appearing as an orphan that fails cleanup.
    """
    client = session.client("ec2", config=RETRY_CONFIG)
    return [
        sg["GroupId"]
        for page in client.get_paginator("describe_security_groups").paginate()
        for sg in page.get("SecurityGroups", [])
        if sg.get("GroupName") != "default"
    ]


def list_ec2_describe_all_network_acls(session: SessionLike) -> list[str]:
    """All network ACLs EXCEPT each VPC's default (undeletable).

    The default network ACL (IsDefault=True) cannot be deleted — AWS rejects
    the call with "cannot delete default network ACL". Excluding it from the scan
    prevents it from appearing as an orphan that fails cleanup.
    """
    client = session.client("ec2", config=RETRY_CONFIG)
    return [
        a["NetworkAclId"]
        for page in client.get_paginator("describe_network_acls").paginate()
        for a in page.get("NetworkAcls", [])
        if not a.get("IsDefault")
    ]


def list_ec2_describe_all_route_tables(session: SessionLike) -> list[str]:
    """All route tables EXCEPT each VPC's main table (undeletable).

    The main route table (the one with a Main=True association) cannot be deleted —
    AWS rejects the call with "cannot delete the main route table". Excluding it from
    the scan prevents it from appearing as an orphan that fails cleanup.
    """
    client = session.client("ec2", config=RETRY_CONFIG)
    return [
        rt["RouteTableId"]
        for page in client.get_paginator("describe_route_tables").paginate()
        for rt in page.get("RouteTables", [])
        if not any(a.get("Main") for a in rt.get("Associations", []))
    ]


def list_s3_bucket_policies(session: SessionLike) -> list[str]:
    """Buckets that have a bucket policy (CCAPI ``AWS::S3::BucketPolicy`` id = bucket name)."""
    client = session.client("s3", config=RETRY_CONFIG)
    out: list[str] = []
    for b in client.list_buckets().get("Buckets", []):
        name = b.get("Name")
        if not name:
            continue
        try:
            client.get_bucket_policy(Bucket=name)
            out.append(name)
        except ClientError as exc:
            # NoSuchBucketPolicy is the expected "bucket has no policy" signal — skip quietly.
            # Anything else (AccessDenied, throttle, region redirect) means we could NOT
            # determine whether a policy exists, so log it rather than silently treat the
            # bucket as policy-free.
            if exc.response.get("Error", {}).get("Code") != "NoSuchBucketPolicy":
                logger.warning(f"s3.get_bucket_policy skipped {name}: {exc}")
            continue
    return out


def list_ec2_describe_all_subnets(session: SessionLike) -> list[str]:
    """All subnets EXCEPT per-AZ default subnets (``DefaultForAz``).

    Keyed on the per-subnet property, NOT default-VPC membership, so an
    agent-created subnet inside the default VPC still surfaces as a real
    orphan/new resource.
    """
    client = session.client("ec2", config=RETRY_CONFIG)
    return [
        s["SubnetId"]
        for page in client.get_paginator("describe_subnets").paginate()
        for s in page.get("Subnets", [])
        if not s.get("DefaultForAz")
    ]


def list_ec2_describe_all_vpcs(session: SessionLike) -> list[str]:
    """All VPCs EXCEPT the default VPC (``IsDefault``).

    Sweeping the default VPC would permanently mutate the account (AWS never
    recreates it) and break the scenarios that ``Vpc.fromLookup(isDefault)``.
    """
    client = session.client("ec2", config=RETRY_CONFIG)
    return [
        v["VpcId"]
        for page in client.get_paginator("describe_vpcs").paginate()
        for v in page.get("Vpcs", [])
        if not v.get("IsDefault")
    ]


def list_ec2_describe_all_internet_gateways(session: SessionLike) -> list[str]:
    """All internet gateways EXCEPT the one attached to the default VPC.

    An IGW has no ``IsDefault`` property, so the default role is the CURRENT
    attachment to the ``IsDefault`` VPC. Attached, it is undeletable
    (DependencyViolation) and detaching it degrades the default VPC. A detached
    (floating) IGW no longer occupies the role — it is deletable and still
    surfaces, so a real leftover is never masked.
    """
    client = session.client("ec2", config=RETRY_CONFIG)
    default_ids = default_vpc_ids(client)
    return [
        igw["InternetGatewayId"]
        for page in client.get_paginator("describe_internet_gateways").paginate()
        for igw in page.get("InternetGateways", [])
        if not any(att.get("VpcId") in default_ids for att in igw.get("Attachments", []))
    ]


def list_ec2_describe_dhcp_options(session: SessionLike) -> list[str]:
    """All DHCP options sets EXCEPT the one associated with the default VPC.

    A DHCP set has no back-reference to its VPCs, so the default role is read
    off the default VPC's ``DhcpOptionsId``. Associated, it is undeletable
    (DependencyViolation) and swapping the association degrades the default
    VPC's DNS/DHCP config. Supersedes the simple ``DescribeDhcpOptions`` row.
    """
    client = session.client("ec2", config=RETRY_CONFIG)
    default_dopt_ids = {
        vpc["DhcpOptionsId"] for vpc in default_vpcs(client) if vpc.get("DhcpOptionsId")
    }
    return [
        d["DhcpOptionsId"]
        for page in client.get_paginator("describe_dhcp_options").paginate()
        for d in page.get("DhcpOptions", [])
        if d.get("DhcpOptionsId") and d["DhcpOptionsId"] not in default_dopt_ids
    ]


def list_codedeploy_registered_on_premises_instances(session: SessionLike) -> list[str]:
    """Registered CodeDeploy on-premises instances (names).

    ``registrationStatus`` is a server-side REQUEST parameter, not a response field, so it must be
    passed to the API call. Without it the API also returns Deregistered instances — stale records
    of instances that no longer exist — which would surface as phantom drift.
    """
    client = session.client("codedeploy", config=RETRY_CONFIG)
    return [
        name
        for page in client.get_paginator("list_on_premises_instances").paginate(
            registrationStatus="Registered"
        )
        for name in page.get("instanceNames", [])
    ]


def list_logs_metric_filters(session: SessionLike) -> list[str]:
    """CloudWatch Logs metric filters by the CFN composite id ``logGroupName|filterName``.

    AWS::Logs::MetricFilter's CCAPI primaryIdentifier is composite [LogGroupName, FilterName]; the
    bare filterName is rejected, so a detected orphaned filter was undeletable. A SimpleLister can
    emit only one flat field, so the composite is built here.
    """
    client = session.client("logs", config=RETRY_CONFIG)
    out: list[str] = []
    for page in client.get_paginator("describe_metric_filters").paginate():
        for mf in page.get("metricFilters", []):
            name, group = mf.get("filterName"), mf.get("logGroupName")
            if name and group:
                out.append(f"{group}|{name}")
    return out


def list_ecs_list_services(session: SessionLike) -> list[str]:
    """ECS services across every cluster, keyed by the CFN composite id ``serviceArn|cluster``.

    AWS::ECS::Service's CCAPI primaryIdentifier is composite [ServiceArn, Cluster]; the bare service
    ARN is rejected (ValidationException), so a detected orphaned service was undeletable. Emit
    ``serviceArn|clusterArn`` (the cluster being paginated) so the CCAPI delete round-trips.
    """
    client = session.client("ecs", config=RETRY_CONFIG)
    clusters: list[str] = []
    for page in client.get_paginator("list_clusters").paginate():
        clusters.extend(page.get("clusterArns", []))
    out: list[str] = []
    for cluster in clusters:
        for page in client.get_paginator("list_services").paginate(cluster=cluster):
            out.extend(f"{svc}|{cluster}" for svc in page.get("serviceArns", []))
    return out


def list_eks_addons(session: SessionLike) -> list[str]:
    """EKS add-ons across every cluster, keyed by the CFN composite id ``clusterName|addonName``."""
    client = session.client("eks", config=RETRY_CONFIG)
    clusters: list[str] = []
    for page in client.get_paginator("list_clusters").paginate():
        clusters.extend(page.get("clusters", []))
    out: list[str] = []
    for cluster in clusters:
        for page in client.get_paginator("list_addons").paginate(clusterName=cluster):
            out.extend(f"{cluster}|{addon}" for addon in page.get("addons", []))
    return out


def list_eks_pod_identity_associations(session: SessionLike) -> list[str]:
    """EKS Pod Identity Associations across every cluster.

    Keyed by the composite id ``clusterName|associationId``.
    """
    client = session.client("eks", config=RETRY_CONFIG)
    clusters: list[str] = []
    for page in client.get_paginator("list_clusters").paginate():
        clusters.extend(page.get("clusters", []))
    results: list[str] = []
    for cluster in clusters:
        try:
            token = None
            while True:
                kwargs: dict = {"clusterName": cluster}
                if token:
                    kwargs["nextToken"] = token
                resp = client.list_pod_identity_associations(**kwargs)
                for assoc in resp.get("associations", []):
                    results.append(f"{cluster}|{assoc['associationId']}")
                token = resp.get("nextToken")
                if not token:
                    break
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                continue
            raise
    return results


def list_eks_nodegroups(session: SessionLike) -> list[str]:
    """EKS Nodegroups across every cluster.

    Keyed by the composite id ``clusterName|nodegroupName``.
    """
    client = session.client("eks", config=RETRY_CONFIG)
    clusters: list[str] = []
    for page in client.get_paginator("list_clusters").paginate():
        clusters.extend(page.get("clusters", []))
    results: list[str] = []
    for cluster in clusters:
        try:
            for page in client.get_paginator("list_nodegroups").paginate(clusterName=cluster):
                for ng in page.get("nodegroups", []):
                    results.append(f"{cluster}|{ng}")
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                continue
            raise
    return results


def list_redshift_clusters_by_identifier(session: SessionLike) -> list[str]:
    """Redshift clusters keyed by ClusterIdentifier (CCAPI's id, not the namespace ARN)."""
    client = session.client("redshift", config=RETRY_CONFIG)
    return [
        c["ClusterIdentifier"]
        for page in client.get_paginator("describe_clusters").paginate()
        for c in page.get("Clusters", [])
        if c.get("ClusterIdentifier") and c.get("ClusterStatus") not in _REDSHIFT_DELETING_STATES
    ]


def list_redshift_serverless_namespaces_by_name(session: SessionLike) -> list[str]:
    """Redshift Serverless namespaces keyed by namespaceName (CCAPI's id, not the ARN)."""
    client = session.client("redshift-serverless", config=RETRY_CONFIG)
    return [
        n["namespaceName"]
        for page in client.get_paginator("list_namespaces").paginate()
        for n in page.get("namespaces", [])
        if n.get("namespaceName")
    ]


def list_redshift_serverless_workgroups_by_name(session: SessionLike) -> list[str]:
    """Redshift Serverless workgroups keyed by workgroupName (CCAPI's id, not the workgroup ARN)."""
    client = session.client("redshift-serverless", config=RETRY_CONFIG)
    return [
        w["workgroupName"]
        for page in client.get_paginator("list_workgroups").paginate()
        for w in page.get("workgroups", [])
        if w.get("workgroupName")
    ]


def list_logs_log_groups_by_name(session: SessionLike) -> list[str]:
    """CloudWatch log groups keyed by logGroupName (CCAPI's id, not the ``:*``-tailed ARN)."""
    client = session.client("logs", config=RETRY_CONFIG)
    return [
        g["logGroupName"]
        for page in client.get_paginator("describe_log_groups").paginate()
        for g in page.get("logGroups", [])
        if g.get("logGroupName")
    ]


def list_imagebuilder_component_build_versions(session: SessionLike) -> list[str]:
    """Image Builder component BUILD-version ARNs (CCAPI's id, per component version)."""
    client = session.client("imagebuilder", config=RETRY_CONFIG)
    version_arns: list[str] = []
    for page in client.get_paginator("list_components").paginate():
        version_arns.extend(c["arn"] for c in page.get("componentVersionList", []) if c.get("arn"))
    out: list[str] = []
    for varn in version_arns:
        for page in client.get_paginator("list_component_build_versions").paginate(
            componentVersionArn=varn
        ):
            out.extend(c["arn"] for c in page.get("componentSummaryList", []) if c.get("arn"))
    return out


def list_imagebuilder_images(session: SessionLike) -> list[str]:
    """Image Builder image BUILD-version ARNs (the ``imageBuildVersionArn`` delete_image needs).

    list_images returns only the image VERSION arn (…/1.0.0); delete_image (and the CCAPI primary
    identifier) requires the BUILD-version arn (…/1.0.0/N), so a version-arn emit is rejected and a
    detected image is undeletable. Fan out per image version via list_image_build_versions and emit
    each build-version arn. Mirrors :func:`list_imagebuilder_component_build_versions`.
    """
    client = session.client("imagebuilder", config=RETRY_CONFIG)
    version_arns: list[str] = []
    for page in client.get_paginator("list_images").paginate():
        version_arns.extend(i["arn"] for i in page.get("imageVersionList", []) if i.get("arn"))
    out: list[str] = []
    for varn in version_arns:
        for page in client.get_paginator("list_image_build_versions").paginate(
            imageVersionArn=varn
        ):
            out.extend(i["arn"] for i in page.get("imageSummaryList", []) if i.get("arn"))
    return out


def list_s3_list_storage_lens_configurations(session: SessionLike) -> list[str]:
    """S3 Storage Lens configuration ids."""
    sts = session.client("sts", config=RETRY_CONFIG)
    account = sts.get_caller_identity()["Account"]
    s3c = session.client("s3control", config=RETRY_CONFIG)
    out: list[str] = []
    token: str | None = None
    while True:
        kw = {"AccountId": account}
        if token:
            kw["NextToken"] = token
        resp = s3c.list_storage_lens_configurations(**kw)
        out.extend(c["Id"] for c in resp.get("StorageLensConfigurationList", []) if c.get("Id"))
        token = resp.get("NextToken")
        if not token:
            return out


# --- Code-lister batch 0: parent->child + discriminator listers (composite `|`-joined ids). ----


_SYNC_TYPES = ("CFN_STACK_SYNC",)


def list_codestar_connections_sync_configurations(session: SessionLike) -> list[str]:
    """Sync configurations across every repository link (id ``ResourceName|SyncType``)."""
    client = session.client("codestar-connections", config=RETRY_CONFIG)
    link_ids: list[str] = []
    token: str | None = None
    while True:
        kw = {"NextToken": token} if token else {}
        resp = client.list_repository_links(**kw)
        link_ids.extend(
            rl["RepositoryLinkId"]
            for rl in resp.get("RepositoryLinks", [])
            if rl.get("RepositoryLinkId")
        )
        token = resp.get("NextToken")
        if not token:
            break
    out: list[str] = []
    for link_id in link_ids:
        for sync_type in _SYNC_TYPES:
            marker: str | None = None
            while True:
                kw = {"RepositoryLinkId": link_id, "SyncType": sync_type}
                if marker:
                    kw["NextToken"] = marker
                try:
                    resp = client.list_sync_configurations(**kw)
                except (ClientError, BotoCoreError):
                    break
                out.extend(
                    f"{sc['ResourceName']}|{sc['SyncType']}"
                    for sc in resp.get("SyncConfigurations", [])
                    if sc.get("ResourceName") and sc.get("SyncType")
                )
                marker = resp.get("NextToken")
                if not marker:
                    break
    return out


def _datasync_all_location_arns(client: object) -> list[str]:
    """Every LocationArn in the region (paginated)."""
    return [
        loc["LocationArn"]
        for page in client.get_paginator("list_locations").paginate()  # type: ignore[attr-defined]
        for loc in page.get("Locations", [])
        if loc.get("LocationArn")
    ]


def _datasync_locations_of_type(session: SessionLike, describe_op: str) -> list[str]:
    """LocationArns that ``describe_op`` resolves — i.e. locations of that exact DataSync type.

    A per-location error is swallowed so one non-matching location (the expected
    InvalidRequestException) or a transient failure never aborts the lister.
    """
    client = session.client("datasync", config=RETRY_CONFIG)
    describe = getattr(client, describe_op)
    out: list[str] = []
    for arn in _datasync_all_location_arns(client):
        try:
            describe(LocationArn=arn)
        except (ClientError, BotoCoreError):
            continue
        out.append(arn)
    return out


def list_datasync_location_s3(session: SessionLike) -> list[str]:
    """DataSync S3 locations."""
    return _datasync_locations_of_type(session, "describe_location_s3")


def list_datasync_location_efs(session: SessionLike) -> list[str]:
    """DataSync EFS locations."""
    return _datasync_locations_of_type(session, "describe_location_efs")


def list_datasync_location_nfs(session: SessionLike) -> list[str]:
    """DataSync NFS locations."""
    return _datasync_locations_of_type(session, "describe_location_nfs")


def list_datasync_location_smb(session: SessionLike) -> list[str]:
    """DataSync SMB locations."""
    return _datasync_locations_of_type(session, "describe_location_smb")


def list_datasync_location_hdfs(session: SessionLike) -> list[str]:
    """DataSync HDFS locations."""
    return _datasync_locations_of_type(session, "describe_location_hdfs")


def list_datasync_location_object_storage(session: SessionLike) -> list[str]:
    """DataSync object-storage locations."""
    return _datasync_locations_of_type(session, "describe_location_object_storage")


def list_datasync_location_azure_blob(session: SessionLike) -> list[str]:
    """DataSync Azure Blob locations."""
    return _datasync_locations_of_type(session, "describe_location_azure_blob")


def list_datasync_location_fsx_lustre(session: SessionLike) -> list[str]:
    """DataSync FSx for Lustre locations."""
    return _datasync_locations_of_type(session, "describe_location_fsx_lustre")


def list_datasync_location_fsx_ontap(session: SessionLike) -> list[str]:
    """DataSync FSx for NetApp ONTAP locations."""
    return _datasync_locations_of_type(session, "describe_location_fsx_ontap")


def list_datasync_location_fsx_openzfs(session: SessionLike) -> list[str]:
    """DataSync FSx for OpenZFS locations."""
    return _datasync_locations_of_type(session, "describe_location_fsx_open_zfs")


def list_datasync_location_fsx_windows(session: SessionLike) -> list[str]:
    """DataSync FSx for Windows File Server locations."""
    return _datasync_locations_of_type(session, "describe_location_fsx_windows")


def list_omics_workflow_versions(session: SessionLike) -> list[str]:
    """Workflow-version ARNs across every workflow (id ``arn``)."""
    client = session.client("omics", config=RETRY_CONFIG)
    workflow_ids: list[str] = []
    for page in client.get_paginator("list_workflows").paginate():
        workflow_ids.extend(w["id"] for w in page.get("items", []) if w.get("id"))
    out: list[str] = []
    for wf_id in workflow_ids:
        try:
            for page in client.get_paginator("list_workflow_versions").paginate(workflowId=wf_id):
                out.extend(v["arn"] for v in page.get("items", []) if v.get("arn"))
        except (ClientError, BotoCoreError):
            continue
    return out


def _has_real_policy(policy: str | None) -> bool:
    """Report whether ``policy`` is a non-empty document (the AWS default is the string ``{}``)."""
    if not policy:
        return False
    try:
        parsed = json.loads(policy)
    except (ValueError, TypeError):
        # A non-JSON non-empty string still means a policy is present.
        return True
    return bool(parsed)


def list_smsvoice_resource_policies(session: SessionLike) -> list[str]:
    """SMS-Voice resources carrying a non-empty resource policy (id ``ResourceArn``).

    The list handler spans phone numbers, sender ids, pools and opt-out lists; only resources
    whose ``GetResourcePolicy`` returns a non-empty document are emitted, which excludes the
    AWS-managed ``Default`` opt-out list (its policy comes back as the empty JSON string ``{}``).
    """
    client = session.client("pinpoint-sms-voice-v2", config=RETRY_CONFIG)
    arns: list[str] = []
    for op, list_key, arn_field in (
        ("describe_phone_numbers", "PhoneNumbers", "PhoneNumberArn"),
        ("describe_pools", "Pools", "PoolArn"),
        ("describe_sender_ids", "SenderIds", "SenderIdArn"),
        ("describe_opt_out_lists", "OptOutLists", "OptOutListArn"),
    ):
        try:
            for page in client.get_paginator(op).paginate():
                arns.extend(
                    item[arn_field] for item in page.get(list_key, []) if item.get(arn_field)
                )
        except (ClientError, BotoCoreError):
            continue
    out: list[str] = []
    for arn in arns:
        try:
            resp = client.get_resource_policy(ResourceArn=arn)
        except (ClientError, BotoCoreError):
            # No policy attached (ResourceNotFoundException) or unreadable — not a policy resource.
            continue
        if _has_real_policy(resp.get("Policy")):
            out.append(arn)
    return out


def list_servicecatalog_service_action_associations(session: SessionLike) -> list[str]:
    """Service-action associations (id ``ProductId|ProvisioningArtifactId|ServiceActionId``).

    Walks every admin-visible product, each of its provisioning artifacts, then the service
    actions associated with that (product, artifact) pair.
    """
    client = session.client("servicecatalog", config=RETRY_CONFIG)
    out: list[str] = []
    for page in client.get_paginator("search_products_as_admin").paginate():
        for pv in page.get("ProductViewDetails", []):
            product_id = (pv.get("ProductViewSummary") or {}).get("ProductId")
            if not product_id:
                continue
            try:
                artifacts = client.list_provisioning_artifacts(ProductId=product_id)
            except (ClientError, BotoCoreError):
                continue
            for art in artifacts.get("ProvisioningArtifactDetails", []):
                pa_id = art.get("Id")
                if not pa_id:
                    continue
                try:
                    pages = client.get_paginator(
                        "list_service_actions_for_provisioning_artifact"
                    ).paginate(ProductId=product_id, ProvisioningArtifactId=pa_id)
                    for sap in pages:
                        out.extend(
                            f"{product_id}|{pa_id}|{sa['Id']}"
                            for sa in sap.get("ServiceActionSummaries", [])
                            if sa.get("Id")
                        )
                except (ClientError, BotoCoreError):
                    continue
    return out


def list_servicecatalog_tag_option_associations(session: SessionLike) -> list[str]:
    """Tag-option associations (id ``TagOptionId|ResourceId``) across every tag option."""
    client = session.client("servicecatalog", config=RETRY_CONFIG)
    tag_option_ids: list[str] = []
    token: str | None = None
    while True:
        kw = {"PageToken": token} if token else {}
        try:
            resp = client.list_tag_options(**kw)
        except ClientError as exc:
            # In an account that never migrated TagOptions, ListTagOptions raises
            # TagOptionNotMigratedException — there are no tag options, hence no associations.
            if exc.response.get("Error", {}).get("Code") == "TagOptionNotMigratedException":
                return []
            raise
        tag_option_ids.extend(t["Id"] for t in resp.get("TagOptionDetails", []) if t.get("Id"))
        token = resp.get("PageToken")
        if not token:
            break
    out: list[str] = []
    for to_id in tag_option_ids:
        marker: str | None = None
        while True:
            kw = {"TagOptionId": to_id}
            if marker:
                kw["PageToken"] = marker
            try:
                resp = client.list_resources_for_tag_option(**kw)
            except (ClientError, BotoCoreError):
                break
            out.extend(f"{to_id}|{r['Id']}" for r in resp.get("ResourceDetails", []) if r.get("Id"))
            marker = resp.get("PageToken")
            if not marker:
                break
    return out


def list_kinesisanalyticsv2_application_outputs(session: SessionLike) -> list[str]:
    """SQL-application outputs (id ``ApplicationName|OutputId``) across every application.

    Enumerated via ``DescribeApplication`` per app; the type has no CCAPI list handler, so this
    is a fast-scan-only drift signal whose id is self-consistent across baseline/current scans.
    """
    client = session.client("kinesisanalyticsv2", config=RETRY_CONFIG)
    names: list[str] = []
    for page in client.get_paginator("list_applications").paginate():
        names.extend(
            a["ApplicationName"]
            for a in page.get("ApplicationSummaries", [])
            if a.get("ApplicationName")
        )
    out: list[str] = []
    for name in names:
        try:
            detail = client.describe_application(ApplicationName=name)["ApplicationDetail"]
        except (ClientError, BotoCoreError):
            continue
        sql = (detail.get("ApplicationConfigurationDescription") or {}).get(
            "SqlApplicationConfigurationDescription"
        ) or {}
        for od in sql.get("OutputDescriptions", []):
            if od.get("OutputId"):
                out.append(f"{name}|{od['OutputId']}")
    return out


# --- Code-lister batch 1: parent->child two-step listers (composite `|`-joined ids). -----------


def list_amplify_domain_associations(session: SessionLike) -> list[str]:
    """Amplify domain associations (``AWS::Amplify::Domain`` id = domainAssociationArn)."""
    client = session.client("amplify", config=RETRY_CONFIG)
    app_ids: list[str] = []
    for page in client.get_paginator("list_apps").paginate():
        app_ids.extend(a["appId"] for a in page.get("apps", []) if a.get("appId"))
    out: list[str] = []
    for app_id in app_ids:
        try:
            for page in client.get_paginator("list_domain_associations").paginate(appId=app_id):
                out.extend(
                    d["domainAssociationArn"]
                    for d in page.get("domainAssociations", [])
                    if d.get("domainAssociationArn")
                )
        except (ClientError, BotoCoreError):
            continue
    return out


def list_config_remediation_configurations(session: SessionLike) -> list[str]:
    """Customer Config remediation configs (id = ConfigRuleName; skips service-created)."""
    client = session.client("config", config=RETRY_CONFIG)
    rule_names: list[str] = []
    for page in client.get_paginator("describe_config_rules").paginate():
        rule_names.extend(
            r["ConfigRuleName"] for r in page.get("ConfigRules", []) if r.get("ConfigRuleName")
        )
    out: list[str] = []
    # DescribeRemediationConfigurations accepts up to 25 rule names per call.
    for start in range(0, len(rule_names), 25):
        batch = rule_names[start : start + 25]
        try:
            resp = client.describe_remediation_configurations(ConfigRuleNames=batch)
        except (ClientError, BotoCoreError):
            continue
        for rc in resp.get("RemediationConfigurations", []):
            # Skip remediations AWS created on the customer's behalf (conformance packs / org
            # rules) — those are not directly-deletable customer resources.
            if rc.get("CreatedByService"):
                continue
            if rc.get("ConfigRuleName"):
                out.append(rc["ConfigRuleName"])
    return out


def list_globalaccelerator_listeners(session: SessionLike) -> list[str]:
    """Global Accelerator listeners (``AWS::GlobalAccelerator::Listener`` id = ListenerArn)."""
    client = session.client("globalaccelerator", config=RETRY_CONFIG)
    accel_arns: list[str] = []
    for page in client.get_paginator("list_accelerators").paginate():
        accel_arns.extend(
            a["AcceleratorArn"] for a in page.get("Accelerators", []) if a.get("AcceleratorArn")
        )
    out: list[str] = []
    for accel_arn in accel_arns:
        try:
            for page in client.get_paginator("list_listeners").paginate(AcceleratorArn=accel_arn):
                out.extend(
                    listener["ListenerArn"]
                    for listener in page.get("Listeners", [])
                    if listener.get("ListenerArn")
                )
        except (ClientError, BotoCoreError):
            continue
    return out


def list_globalaccelerator_endpoint_groups(session: SessionLike) -> list[str]:
    """Global Accelerator endpoint groups (id = EndpointGroupArn) across accel -> listeners."""
    client = session.client("globalaccelerator", config=RETRY_CONFIG)
    accel_arns: list[str] = []
    for page in client.get_paginator("list_accelerators").paginate():
        accel_arns.extend(
            a["AcceleratorArn"] for a in page.get("Accelerators", []) if a.get("AcceleratorArn")
        )
    listener_arns: list[str] = []
    for accel_arn in accel_arns:
        try:
            for page in client.get_paginator("list_listeners").paginate(AcceleratorArn=accel_arn):
                listener_arns.extend(
                    listener["ListenerArn"]
                    for listener in page.get("Listeners", [])
                    if listener.get("ListenerArn")
                )
        except (ClientError, BotoCoreError):
            continue
    out: list[str] = []
    for listener_arn in listener_arns:
        try:
            for page in client.get_paginator("list_endpoint_groups").paginate(
                ListenerArn=listener_arn
            ):
                out.extend(
                    g["EndpointGroupArn"]
                    for g in page.get("EndpointGroups", [])
                    if g.get("EndpointGroupArn")
                )
        except (ClientError, BotoCoreError):
            continue
    return out


def _guardduty_detector_ids(client: object) -> list[str]:
    """All GuardDuty detector ids in the region (parent for every GuardDuty child type)."""
    detector_ids: list[str] = []
    for page in client.get_paginator("list_detectors").paginate():  # type: ignore[attr-defined]
        detector_ids.extend(page.get("DetectorIds", []))
    return detector_ids


def list_guardduty_filters(session: SessionLike) -> list[str]:
    """GuardDuty filters (``AWS::GuardDuty::Filter`` id = ``<DetectorId>|<Name>``)."""
    client = session.client("guardduty", config=RETRY_CONFIG)
    out: list[str] = []
    for detector_id in _guardduty_detector_ids(client):
        try:
            for page in client.get_paginator("list_filters").paginate(DetectorId=detector_id):
                out.extend(f"{detector_id}|{name}" for name in page.get("FilterNames", []))
        except (ClientError, BotoCoreError):
            continue
    return out


def list_guardduty_ip_sets(session: SessionLike) -> list[str]:
    """GuardDuty IP sets (``AWS::GuardDuty::IPSet`` id = ``<IpSetId>|<DetectorId>``)."""
    client = session.client("guardduty", config=RETRY_CONFIG)
    out: list[str] = []
    for detector_id in _guardduty_detector_ids(client):
        try:
            for page in client.get_paginator("list_ip_sets").paginate(DetectorId=detector_id):
                out.extend(f"{set_id}|{detector_id}" for set_id in page.get("IpSetIds", []))
        except (ClientError, BotoCoreError):
            continue
    return out


def list_guardduty_threat_intel_sets(session: SessionLike) -> list[str]:
    """GuardDuty threat-intel sets (``ThreatIntelSet``; id ``<Id>|<DetectorId>``)."""
    client = session.client("guardduty", config=RETRY_CONFIG)
    out: list[str] = []
    for detector_id in _guardduty_detector_ids(client):
        try:
            for page in client.get_paginator("list_threat_intel_sets").paginate(
                DetectorId=detector_id
            ):
                out.extend(
                    f"{set_id}|{detector_id}" for set_id in page.get("ThreatIntelSetIds", [])
                )
        except (ClientError, BotoCoreError):
            continue
    return out


def list_guardduty_threat_entity_sets(session: SessionLike) -> list[str]:
    """GuardDuty threat-entity sets (``ThreatEntitySet``; id ``<Id>|<DetectorId>``)."""
    client = session.client("guardduty", config=RETRY_CONFIG)
    out: list[str] = []
    for detector_id in _guardduty_detector_ids(client):
        try:
            for page in client.get_paginator("list_threat_entity_sets").paginate(
                DetectorId=detector_id
            ):
                out.extend(
                    f"{set_id}|{detector_id}" for set_id in page.get("ThreatEntitySetIds", [])
                )
        except (ClientError, BotoCoreError):
            continue
    return out


def list_guardduty_trusted_entity_sets(session: SessionLike) -> list[str]:
    """GuardDuty trusted-entity sets (id = ``<Id>|<DetectorId>``)."""
    client = session.client("guardduty", config=RETRY_CONFIG)
    out: list[str] = []
    for detector_id in _guardduty_detector_ids(client):
        try:
            for page in client.get_paginator("list_trusted_entity_sets").paginate(
                DetectorId=detector_id
            ):
                out.extend(
                    f"{set_id}|{detector_id}" for set_id in page.get("TrustedEntitySetIds", [])
                )
        except (ClientError, BotoCoreError):
            continue
    return out


def list_guardduty_members(session: SessionLike) -> list[str]:
    """GuardDuty members (``AWS::GuardDuty::Member`` id = ``<DetectorId>|<MemberAccountId>``)."""
    client = session.client("guardduty", config=RETRY_CONFIG)
    out: list[str] = []
    for detector_id in _guardduty_detector_ids(client):
        try:
            for page in client.get_paginator("list_members").paginate(DetectorId=detector_id):
                out.extend(
                    f"{detector_id}|{m['AccountId']}"
                    for m in page.get("Members", [])
                    if m.get("AccountId")
                )
        except (ClientError, BotoCoreError):
            continue
    return out


def list_guardduty_publishing_destinations(session: SessionLike) -> list[str]:
    """GuardDuty publishing destinations (id = ``<DetectorId>|<DestinationId>``).

    ``list_publishing_destinations`` has no boto3 paginator, so pages are walked
    manually by ``NextToken``.
    """
    client = session.client("guardduty", config=RETRY_CONFIG)
    out: list[str] = []
    for detector_id in _guardduty_detector_ids(client):
        token: str | None = None
        while True:
            kwargs = {"DetectorId": detector_id}
            if token:
                kwargs["NextToken"] = token
            try:
                resp = client.list_publishing_destinations(**kwargs)
            except (ClientError, BotoCoreError):
                break
            out.extend(
                f"{detector_id}|{d['DestinationId']}"
                for d in resp.get("Destinations", [])
                if d.get("DestinationId")
            )
            token = resp.get("NextToken")
            if not token:
                break
    return out


def list_guardduty_masters(session: SessionLike) -> list[str]:
    """GuardDuty master accounts (``Master``; id ``<DetectorId>|<MasterAccountId>``)."""
    client = session.client("guardduty", config=RETRY_CONFIG)
    out: list[str] = []
    for detector_id in _guardduty_detector_ids(client):
        try:
            master = client.get_master_account(DetectorId=detector_id).get("Master") or {}
        except (ClientError, BotoCoreError):
            continue
        account_id = master.get("AccountId")
        if account_id:
            out.append(f"{detector_id}|{account_id}")
    return out


def list_location_tracker_consumers(session: SessionLike) -> list[str]:
    """Location tracker consumers (id = ``<TrackerName>|<ConsumerArn>``)."""
    client = session.client("location", config=RETRY_CONFIG)
    tracker_names: list[str] = []
    for page in client.get_paginator("list_trackers").paginate():
        tracker_names.extend(
            t["TrackerName"] for t in page.get("Entries", []) if t.get("TrackerName")
        )
    out: list[str] = []
    for tracker_name in tracker_names:
        try:
            for page in client.get_paginator("list_tracker_consumers").paginate(
                TrackerName=tracker_name
            ):
                out.extend(
                    f"{tracker_name}|{consumer_arn}"
                    for consumer_arn in page.get("ConsumerArns", [])
                )
        except (ClientError, BotoCoreError):
            continue
    return out


def list_organizations_organizational_units(session: SessionLike) -> list[str]:
    """Every OU (``AWS::Organizations::OrganizationalUnit`` id = ou-id), recursing all parents."""
    client = session.client("organizations", config=RETRY_CONFIG)
    parents: list[str] = []
    for page in client.get_paginator("list_roots").paginate():
        parents.extend(r["Id"] for r in page.get("Roots", []) if r.get("Id"))
    out: list[str] = []
    seen: set[str] = set()
    # Breadth-first walk: an OU can be a parent of further OUs, so newly-found OUs are
    # themselves queued as parents until the tree is exhausted.
    while parents:
        parent_id = parents.pop()
        try:
            pages = client.get_paginator("list_organizational_units_for_parent").paginate(
                ParentId=parent_id
            )
        except (ClientError, BotoCoreError):
            continue
        for page in pages:
            for ou in page.get("OrganizationalUnits", []):
                ou_id = ou.get("Id")
                if ou_id and ou_id not in seen:
                    seen.add(ou_id)
                    out.append(ou_id)
                    parents.append(ou_id)
    return out


# --- Code-lister batch 2: parent->child + EC2 IsEgress-discriminator listers. ------------------


def list_connect_instance_ids(client: object) -> list[str]:
    """Every Connect instance id (parent for the per-instance association listers)."""
    ids: list[str] = []
    for page in client.get_paginator("list_instances").paginate():  # type: ignore[attr-defined]
        ids.extend(i["Id"] for i in page.get("InstanceSummaryList", []) if i.get("Id"))
    return ids


def list_connect_approved_origins(session: SessionLike) -> list[str]:
    """Approved origins per instance (CCAPI id ``InstanceId|Origin``)."""
    client = session.client("connect", config=RETRY_CONFIG)
    out: list[str] = []
    for instance_id in list_connect_instance_ids(client):
        try:
            for page in client.get_paginator("list_approved_origins").paginate(
                InstanceId=instance_id
            ):
                out.extend(f"{instance_id}|{origin}" for origin in page.get("Origins", []))
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"connect.list_approved_origins skipped {instance_id}: {exc}")
    return out


def list_connect_security_keys(session: SessionLike) -> list[str]:
    """Security keys per instance (CCAPI id ``InstanceId|AssociationId``)."""
    client = session.client("connect", config=RETRY_CONFIG)
    out: list[str] = []
    for instance_id in list_connect_instance_ids(client):
        try:
            for page in client.get_paginator("list_security_keys").paginate(InstanceId=instance_id):
                out.extend(
                    f"{instance_id}|{k['AssociationId']}"
                    for k in page.get("SecurityKeys", [])
                    if k.get("AssociationId")
                )
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"connect.list_security_keys skipped {instance_id}: {exc}")
    return out


_CONNECT_STORAGE_RESOURCE_TYPES = (
    "CHAT_TRANSCRIPTS",
    "CALL_RECORDINGS",
    "SCHEDULED_REPORTS",
    "MEDIA_STREAMS",
    "CONTACT_TRACE_RECORDS",
    "AGENT_EVENTS",
    "REAL_TIME_CONTACT_ANALYSIS_SEGMENTS",
    "ATTACHMENTS",
    "CONTACT_EVALUATIONS",
    "SCREEN_RECORDINGS",
    "REAL_TIME_CONTACT_ANALYSIS_CHAT_SEGMENTS",
    "REAL_TIME_CONTACT_ANALYSIS_VOICE_SEGMENTS",
    "EMAIL_MESSAGES",
)


def list_connect_instance_storage_configs(session: SessionLike) -> list[str]:
    """Instance storage configs per instance+resource-type (CCAPI id ``InstanceArn|AssocId``)."""
    client = session.client("connect", config=RETRY_CONFIG)
    out: list[str] = []
    for page in client.get_paginator("list_instances").paginate():
        for inst in page.get("InstanceSummaryList", []):
            instance_id = inst.get("Id")
            instance_arn = inst.get("Arn")
            if not instance_id or not instance_arn:
                continue
            for resource_type in _CONNECT_STORAGE_RESOURCE_TYPES:
                try:
                    resp = client.list_instance_storage_configs(
                        InstanceId=instance_id, ResourceType=resource_type
                    )
                except (ClientError, BotoCoreError) as exc:
                    logger.debug(
                        f"connect.list_instance_storage_configs skipped "
                        f"{instance_id}/{resource_type}: {exc}"
                    )
                    continue
                out.extend(
                    f"{instance_arn}|{c['AssociationId']}"
                    for c in resp.get("StorageConfigs", [])
                    if c.get("AssociationId")
                )
    return out


def list_ec2_local_gateway_routes(session: SessionLike) -> list[str]:
    """Static local-gateway routes per route table (CCAPI id ``DestCidr|RouteTableId``)."""
    client = session.client("ec2", config=RETRY_CONFIG)
    table_ids: list[str] = []
    for page in client.get_paginator("describe_local_gateway_route_tables").paginate():
        table_ids.extend(
            t["LocalGatewayRouteTableId"]
            for t in page.get("LocalGatewayRouteTables", [])
            if t.get("LocalGatewayRouteTableId")
        )
    out: list[str] = []
    for table_id in table_ids:
        try:
            resp = client.search_local_gateway_routes(
                LocalGatewayRouteTableId=table_id,
                Filters=[{"Name": "type", "Values": ["static", "propagated"]}],
            )
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"ec2.search_local_gateway_routes skipped {table_id}: {exc}")
            continue
        out.extend(
            f"{r['DestinationCidrBlock']}|{table_id}"
            for r in resp.get("Routes", [])
            if r.get("DestinationCidrBlock")
        )
    return out


def list_ec2_route_server_associations(session: SessionLike) -> list[str]:
    """Route-server<->VPC associations per route server (CCAPI id ``RouteServerId|VpcId``)."""
    client = session.client("ec2", config=RETRY_CONFIG)
    server_ids: list[str] = []
    for page in client.get_paginator("describe_route_servers").paginate():
        server_ids.extend(
            s["RouteServerId"] for s in page.get("RouteServers", []) if s.get("RouteServerId")
        )
    out: list[str] = []
    for server_id in server_ids:
        try:
            resp = client.get_route_server_associations(RouteServerId=server_id)
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"ec2.get_route_server_associations skipped {server_id}: {exc}")
            continue
        out.extend(
            f"{a['RouteServerId']}|{a['VpcId']}"
            for a in resp.get("RouteServerAssociations", [])
            if a.get("RouteServerId") and a.get("VpcId")
        )
    return out


def list_ec2_route_server_propagations(session: SessionLike) -> list[str]:
    """Route-server route-table propagations per server (id ``RouteServerId|RouteTableId``)."""
    client = session.client("ec2", config=RETRY_CONFIG)
    server_ids: list[str] = []
    for page in client.get_paginator("describe_route_servers").paginate():
        server_ids.extend(
            s["RouteServerId"] for s in page.get("RouteServers", []) if s.get("RouteServerId")
        )
    out: list[str] = []
    for server_id in server_ids:
        try:
            resp = client.get_route_server_propagations(RouteServerId=server_id)
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"ec2.get_route_server_propagations skipped {server_id}: {exc}")
            continue
        out.extend(
            f"{p['RouteServerId']}|{p['RouteTableId']}"
            for p in resp.get("RouteServerPropagations", [])
            if p.get("RouteServerId") and p.get("RouteTableId")
        )
    return out


def list_ec2_vpc_endpoint_service_permissions(session: SessionLike) -> list[str]:
    """Endpoint services that have an explicit allow-list (CCAPI id = the ``ServiceId``)."""
    client = session.client("ec2", config=RETRY_CONFIG)
    service_ids: list[str] = []
    for page in client.get_paginator("describe_vpc_endpoint_service_configurations").paginate():
        service_ids.extend(
            c["ServiceId"] for c in page.get("ServiceConfigurations", []) if c.get("ServiceId")
        )
    out: list[str] = []
    for service_id in service_ids:
        try:
            resp = client.describe_vpc_endpoint_service_permissions(ServiceId=service_id)
        except (ClientError, BotoCoreError) as exc:
            logger.debug(
                f"ec2.describe_vpc_endpoint_service_permissions skipped {service_id}: {exc}"
            )
            continue
        if resp.get("AllowedPrincipals"):
            out.append(service_id)
    return out


def list_ec2_security_group_ingress_rules(session: SessionLike) -> list[str]:
    """Ingress security-group rules (``sgr-*`` where IsEgress is False)."""
    client = session.client("ec2", config=RETRY_CONFIG)
    return [
        r["SecurityGroupRuleId"]
        for page in client.get_paginator("describe_security_group_rules").paginate()
        for r in page.get("SecurityGroupRules", [])
        if r.get("SecurityGroupRuleId") and r.get("IsEgress") is False
    ]


def list_ec2_security_group_egress_rules(session: SessionLike) -> list[str]:
    """Egress security-group rules (``sgr-*`` where IsEgress is True)."""
    client = session.client("ec2", config=RETRY_CONFIG)
    return [
        r["SecurityGroupRuleId"]
        for page in client.get_paginator("describe_security_group_rules").paginate()
        for r in page.get("SecurityGroupRules", [])
        if r.get("SecurityGroupRuleId") and r.get("IsEgress") is True
    ]


def list_greengrass_logger_definition_versions(session: SessionLike) -> list[str]:
    """Logger-definition versions per definition (CCAPI id = the version ``Id``)."""
    client = session.client("greengrass", config=RETRY_CONFIG)
    def_ids: list[str] = []
    for page in client.get_paginator("list_logger_definitions").paginate():
        def_ids.extend(d["Id"] for d in page.get("Definitions", []) if d.get("Id"))
    out: list[str] = []
    for def_id in def_ids:
        try:
            for page in client.get_paginator("list_logger_definition_versions").paginate(
                LoggerDefinitionId=def_id
            ):
                out.extend(v["Id"] for v in page.get("Versions", []) if v.get("Id"))
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"greengrass.list_logger_definition_versions skipped {def_id}: {exc}")
    return out


def _verifiedpermissions_policy_store_ids(client: object) -> list[str]:
    """Every Verified Permissions policy-store id (the parent for the per-store listers)."""
    ids: list[str] = []
    for page in client.get_paginator("list_policy_stores").paginate():  # type: ignore[attr-defined]
        ids.extend(
            s["policyStoreId"] for s in page.get("policyStores", []) if s.get("policyStoreId")
        )
    return ids


def list_verifiedpermissions_identity_sources(session: SessionLike) -> list[str]:
    """Identity sources per policy store (CCAPI id ``IdentitySourceId|PolicyStoreId``)."""
    client = session.client("verifiedpermissions", config=RETRY_CONFIG)
    out: list[str] = []
    for store_id in _verifiedpermissions_policy_store_ids(client):
        try:
            for page in client.get_paginator("list_identity_sources").paginate(
                policyStoreId=store_id
            ):
                out.extend(
                    f"{s['identitySourceId']}|{store_id}"
                    for s in page.get("identitySources", [])
                    if s.get("identitySourceId")
                )
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"verifiedpermissions.list_identity_sources skipped {store_id}: {exc}")
    return out


def list_verifiedpermissions_policies(session: SessionLike) -> list[str]:
    """Policies per policy store (CCAPI id ``PolicyId|PolicyStoreId``)."""
    client = session.client("verifiedpermissions", config=RETRY_CONFIG)
    out: list[str] = []
    for store_id in _verifiedpermissions_policy_store_ids(client):
        try:
            for page in client.get_paginator("list_policies").paginate(policyStoreId=store_id):
                out.extend(
                    f"{p['policyId']}|{store_id}"
                    for p in page.get("policies", [])
                    if p.get("policyId")
                )
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"verifiedpermissions.list_policies skipped {store_id}: {exc}")
    return out


def list_pcaconnectorad_service_principal_names(session: SessionLike) -> list[str]:
    """Service principal names per directory registration (id ``ConnectorArn|DirRegArn``)."""
    client = session.client("pca-connector-ad", config=RETRY_CONFIG)
    reg_arns: list[str] = []
    for page in client.get_paginator("list_directory_registrations").paginate():
        reg_arns.extend(r["Arn"] for r in page.get("DirectoryRegistrations", []) if r.get("Arn"))
    out: list[str] = []
    for reg_arn in reg_arns:
        try:
            for page in client.get_paginator("list_service_principal_names").paginate(
                DirectoryRegistrationArn=reg_arn
            ):
                out.extend(
                    f"{spn['ConnectorArn']}|{reg_arn}"
                    for spn in page.get("ServicePrincipalNames", [])
                    if spn.get("ConnectorArn")
                )
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"pca-connector-ad.list_service_principal_names skipped {reg_arn}: {exc}")
    return out


def list_pcaconnectorad_template_group_access_control_entries(session: SessionLike) -> list[str]:
    """Template group access-control entries (id ``GroupSecurityIdentifier|TemplateArn``)."""
    client = session.client("pca-connector-ad", config=RETRY_CONFIG)
    connector_arns: list[str] = []
    for page in client.get_paginator("list_connectors").paginate():
        connector_arns.extend(c["Arn"] for c in page.get("Connectors", []) if c.get("Arn"))
    template_arns: list[str] = []
    for connector_arn in connector_arns:
        try:
            for page in client.get_paginator("list_templates").paginate(ConnectorArn=connector_arn):
                template_arns.extend(t["Arn"] for t in page.get("Templates", []) if t.get("Arn"))
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"pca-connector-ad.list_templates skipped {connector_arn}: {exc}")
    out: list[str] = []
    for template_arn in template_arns:
        try:
            for page in client.get_paginator("list_template_group_access_control_entries").paginate(
                TemplateArn=template_arn
            ):
                out.extend(
                    f"{e['GroupSecurityIdentifier']}|{template_arn}"
                    for e in page.get("AccessControlEntries", [])
                    if e.get("GroupSecurityIdentifier")
                )
        except (ClientError, BotoCoreError) as exc:
            logger.debug(
                f"pca-connector-ad.list_template_group_access_control_entries "
                f"skipped {template_arn}: {exc}"
            )
    return out


def list_rtbfabric_link_routing_rules(session: SessionLike) -> list[str]:
    """Link routing rules across every gateway+link (id ``gatewayId|linkId|ruleId``)."""
    client = session.client("rtbfabric", config=RETRY_CONFIG)
    gateway_ids: list[str] = []
    # Both gateway list ops return ``gatewayIds`` as a flat list of id strings (not structures).
    for op in ("list_requester_gateways", "list_responder_gateways"):
        try:
            for page in client.get_paginator(op).paginate():
                gateway_ids.extend(page.get("gatewayIds", []))
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"rtbfabric.{op} skipped: {exc}")
    out: list[str] = []
    for gateway_id in gateway_ids:
        link_ids: list[str] = []
        try:
            for page in client.get_paginator("list_links").paginate(gatewayId=gateway_id):
                link_ids.extend(
                    link["linkId"] for link in page.get("links", []) if link.get("linkId")
                )
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"rtbfabric.list_links skipped {gateway_id}: {exc}")
            continue
        for link_id in link_ids:
            try:
                for page in client.get_paginator("list_link_routing_rules").paginate(
                    gatewayId=gateway_id, linkId=link_id
                ):
                    out.extend(
                        f"{gateway_id}|{link_id}|{r['ruleId']}"
                        for r in page.get("rules", [])
                        if r.get("ruleId")
                    )
            except (ClientError, BotoCoreError) as exc:
                logger.debug(
                    f"rtbfabric.list_link_routing_rules skipped {gateway_id}/{link_id}: {exc}"
                )
    return out


def list_logs_transformers(session: SessionLike) -> list[str]:
    """Log groups that have a transformer (CCAPI id = the ``logGroupIdentifier``/name)."""
    client = session.client("logs", config=RETRY_CONFIG)
    group_names: list[str] = []
    for page in client.get_paginator("describe_log_groups").paginate():
        group_names.extend(
            g["logGroupName"] for g in page.get("logGroups", []) if g.get("logGroupName")
        )
    out: list[str] = []
    for name in group_names:
        try:
            resp = client.get_transformer(logGroupIdentifier=name)
        except ClientError as exc:
            # ResourceNotFoundException = the group simply has no transformer (the common case);
            # anything else means we could not determine transformer presence, so log it.
            if exc.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
                logger.debug(f"logs.get_transformer skipped {name}: {exc}")
            continue
        except BotoCoreError as exc:
            logger.debug(f"logs.get_transformer skipped {name}: {exc}")
            continue
        if resp.get("transformerConfig"):
            out.append(name)
    return out


# --- Code-lister batch 3: NetworkManager AttachmentType discriminators + parent->child. ----


def _networkmanager_attachments(session: SessionLike, attachment_type: str) -> list[str]:
    """Attachment ids for one ``AttachmentType`` (filtered call + per-item field re-check)."""
    client = session.client("networkmanager", config=RETRY_CONFIG)
    out: list[str] = []
    for page in client.get_paginator("list_attachments").paginate(AttachmentType=attachment_type):
        for a in page.get("Attachments", []):
            if a.get("AttachmentType") == attachment_type and a.get("AttachmentId"):
                out.append(a["AttachmentId"])
    return out


def list_networkmanager_connect_attachments(session: SessionLike) -> list[str]:
    """NetworkManager CONNECT attachments (AWS::NetworkManager::ConnectAttachment)."""
    return _networkmanager_attachments(session, "CONNECT")


def list_networkmanager_site_to_site_vpn_attachments(session: SessionLike) -> list[str]:
    """NetworkManager SITE_TO_SITE_VPN attachments (::SiteToSiteVpnAttachment)."""
    return _networkmanager_attachments(session, "SITE_TO_SITE_VPN")


def list_networkmanager_vpc_attachments(session: SessionLike) -> list[str]:
    """NetworkManager VPC attachments (AWS::NetworkManager::VpcAttachment)."""
    return _networkmanager_attachments(session, "VPC")


def list_networkmanager_direct_connect_gateway_attachments(session: SessionLike) -> list[str]:
    """NetworkManager DIRECT_CONNECT_GATEWAY attachments (::DirectConnectGatewayAttachment)."""
    return _networkmanager_attachments(session, "DIRECT_CONNECT_GATEWAY")


def list_networkmanager_transit_gateway_route_table_attachments(
    session: SessionLike,
) -> list[str]:
    """NetworkManager TRANSIT_GATEWAY_ROUTE_TABLE attachments (::TransitGatewayRouteTableAttach)."""
    return _networkmanager_attachments(session, "TRANSIT_GATEWAY_ROUTE_TABLE")


def list_networkmanager_transit_gateway_registrations(session: SessionLike) -> list[str]:
    """TGW registrations per global network as CCAPI's ``<GlobalNetworkId>|<TransitGatewayArn>``."""
    client = session.client("networkmanager", config=RETRY_CONFIG)
    out: list[str] = []
    for gn_page in client.get_paginator("describe_global_networks").paginate():
        for gn in gn_page.get("GlobalNetworks", []):
            global_network_id = gn.get("GlobalNetworkId")
            if not global_network_id:
                continue
            try:
                for page in client.get_paginator("get_transit_gateway_registrations").paginate(
                    GlobalNetworkId=global_network_id
                ):
                    for reg in page.get("TransitGatewayRegistrations", []):
                        arn = reg.get("TransitGatewayArn")
                        if arn:
                            out.append(f"{global_network_id}|{arn}")
            except (ClientError, BotoCoreError) as exc:
                logger.debug(f"networkmanager registrations skipped {global_network_id}: {exc}")
    return out


def list_iam_user_to_group_additions(session: SessionLike) -> list[str]:
    """IAM group memberships as ``<GroupName>/<UserName>`` (one per user in each group)."""
    client = session.client("iam", config=RETRY_CONFIG)
    out: list[str] = []
    for page in client.get_paginator("list_groups").paginate():
        for group in page.get("Groups", []):
            name = group.get("GroupName")
            if not name:
                continue
            try:
                for gm in client.get_paginator("get_group").paginate(GroupName=name):
                    for user in gm.get("Users", []):
                        if user.get("UserName"):
                            out.append(f"{name}/{user['UserName']}")
            except (ClientError, BotoCoreError) as exc:
                logger.debug(f"iam get_group skipped {name}: {exc}")
    return out


def list_medialive_channel_placement_groups(session: SessionLike) -> list[str]:
    """Channel placement groups per cluster, as CCAPI's ``<Id>|<ClusterId>``."""
    client = session.client("medialive", config=RETRY_CONFIG)
    out: list[str] = []
    for cl_page in client.get_paginator("list_clusters").paginate():
        for cluster in cl_page.get("Clusters", []):
            cluster_id = cluster.get("Id")
            if not cluster_id:
                continue
            try:
                for page in client.get_paginator("list_channel_placement_groups").paginate(
                    ClusterId=cluster_id
                ):
                    for grp in page.get("ChannelPlacementGroups", []):
                        if grp.get("Id"):
                            out.append(f"{grp['Id']}|{cluster_id}")
            except (ClientError, BotoCoreError) as exc:
                logger.debug(f"medialive placement groups skipped {cluster_id}: {exc}")
    return out


def list_qbusiness_data_sources(session: SessionLike) -> list[str]:
    """Q Business data sources as CCAPI's ``<ApplicationId>|<DataSourceId>|<IndexId>``."""
    client = session.client("qbusiness", config=RETRY_CONFIG)
    out: list[str] = []
    application_ids: list[str] = []
    for page in client.get_paginator("list_applications").paginate():
        application_ids.extend(
            a["applicationId"] for a in page.get("applications", []) if a.get("applicationId")
        )
    for application_id in application_ids:
        index_ids: list[str] = []
        try:
            for page in client.get_paginator("list_indices").paginate(applicationId=application_id):
                index_ids.extend(i["indexId"] for i in page.get("indices", []) if i.get("indexId"))
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"qbusiness list_indices skipped {application_id}: {exc}")
            continue
        for index_id in index_ids:
            try:
                for page in client.get_paginator("list_data_sources").paginate(
                    applicationId=application_id, indexId=index_id
                ):
                    for ds in page.get("dataSources", []):
                        if ds.get("dataSourceId"):
                            out.append(f"{application_id}|{ds['dataSourceId']}|{index_id}")
            except (ClientError, BotoCoreError) as exc:
                logger.debug(f"qbusiness list_data_sources skipped {index_id}: {exc}")
    return out


def list_s3files_file_system_policies(session: SessionLike) -> list[str]:
    """S3 file systems that have a policy attached (CCAPI id = fileSystemId)."""
    client = session.client("s3files", config=RETRY_CONFIG)
    out: list[str] = []
    file_system_ids: list[str] = []
    for page in client.get_paginator("list_file_systems").paginate():
        file_system_ids.extend(
            f["fileSystemId"] for f in page.get("fileSystems", []) if f.get("fileSystemId")
        )
    for fid in file_system_ids:
        try:
            if client.get_file_system_policy(fileSystemId=fid).get("policy"):
                out.append(fid)
        except ClientError as exc:
            # "no policy" surfaces as a not-found error and just means this file system has none;
            # any other code (AccessDenied, throttle) means we could not confirm, so log it.
            code = exc.response.get("Error", {}).get("Code")
            if code not in ("ResourceNotFoundException", "PolicyNotFound", "NotFoundException"):
                logger.warning(f"s3files.get_file_system_policy skipped {fid}: {exc}")
        except BotoCoreError as exc:
            logger.warning(f"s3files.get_file_system_policy skipped {fid}: {exc}")
    return out


def list_ecr_registry_policy(session: SessionLike) -> list[str]:
    """The account's ECR registry policy, keyed by RegistryId (its CCAPI primary identifier).

    A registry has at most one policy; ``GetRegistryPolicy`` raises when none is set, in which case
    there is nothing to report.
    """
    client = session.client("ecr", config=RETRY_CONFIG)
    try:
        resp = client.get_registry_policy()
    except (ClientError, BotoCoreError) as exc:
        logger.debug(f"ecr.get_registry_policy skipped: {exc}")
        return []
    registry_id = resp.get("registryId")
    return [registry_id] if registry_id else []


def list_secretsmanager_resource_policies(session: SessionLike) -> list[str]:
    """Secret ARNs that have a resource policy attached (CCAPI id = secret ARN)."""
    client = session.client("secretsmanager", config=RETRY_CONFIG)
    out: list[str] = []
    arns: list[str] = []
    for page in client.get_paginator("list_secrets").paginate():
        arns.extend(s["ARN"] for s in page.get("SecretList", []) if s.get("ARN"))
    for arn in arns:
        try:
            if client.get_resource_policy(SecretId=arn).get("ResourcePolicy"):
                out.append(arn)
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"secretsmanager get_resource_policy skipped {arn}: {exc}")
    return out


def list_secretsmanager_rotation_schedules(session: SessionLike) -> list[str]:
    """Rotation-enabled secrets (CCAPI ::RotationSchedule id = the secret ARN)."""
    client = session.client("secretsmanager", config=RETRY_CONFIG)
    out: list[str] = []
    for page in client.get_paginator("list_secrets").paginate():
        for s in page.get("SecretList", []):
            if s.get("RotationEnabled") and s.get("ARN"):
                out.append(s["ARN"])
    return out


def list_qconnect_ai_agents(session: SessionLike) -> list[str]:
    """Q-in-Connect (Wisdom) AI agents per assistant, as CCAPI's ``<AIAgentId>|<AssistantId>``."""
    client = session.client("qconnect", config=RETRY_CONFIG)
    out: list[str] = []
    assistant_ids: list[str] = []
    for page in client.get_paginator("list_assistants").paginate():
        assistant_ids.extend(
            a["assistantId"] for a in page.get("assistantSummaries", []) if a.get("assistantId")
        )
    for assistant_id in assistant_ids:
        try:
            for page in client.get_paginator("list_ai_agents").paginate(assistantId=assistant_id):
                for agent in page.get("aiAgentSummaries", []):
                    if agent.get("aiAgentId"):
                        out.append(f"{agent['aiAgentId']}|{assistant_id}")
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"qconnect list_ai_agents skipped {assistant_id}: {exc}")
    return out


def list_qconnect_ai_agent_versions(session: SessionLike) -> list[str]:
    """AI-agent versions per agent, as CCAPI's ``<AssistantId>|<AIAgentId>|<VersionNumber>``."""
    client = session.client("qconnect", config=RETRY_CONFIG)
    out: list[str] = []
    assistant_ids: list[str] = []
    for page in client.get_paginator("list_assistants").paginate():
        assistant_ids.extend(
            a["assistantId"] for a in page.get("assistantSummaries", []) if a.get("assistantId")
        )
    for assistant_id in assistant_ids:
        agent_ids: list[str] = []
        try:
            for page in client.get_paginator("list_ai_agents").paginate(assistantId=assistant_id):
                agent_ids.extend(
                    ag["aiAgentId"]
                    for ag in page.get("aiAgentSummaries", [])
                    if ag.get("aiAgentId")
                )
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"qconnect list_ai_agents skipped {assistant_id}: {exc}")
            continue
        for agent_id in agent_ids:
            try:
                for page in client.get_paginator("list_ai_agent_versions").paginate(
                    assistantId=assistant_id, aiAgentId=agent_id
                ):
                    for ver in page.get("aiAgentVersionSummaries", []):
                        version = ver.get("versionNumber")
                        if version is not None:
                            out.append(f"{assistant_id}|{agent_id}|{version}")
            except (ClientError, BotoCoreError) as exc:
                logger.debug(f"qconnect list_ai_agent_versions skipped {agent_id}: {exc}")
    return out


# --- Code-lister batch 4: parent->child + per-resource policy/channel listers. -----------------


def _apigatewayv2_api_ids(client: object) -> list[str]:
    """Every API Gateway V2 API id (parent for the response listers)."""
    ids: list[str] = []
    for page in client.get_paginator("get_apis").paginate():  # type: ignore[attr-defined]
        ids.extend(a["ApiId"] for a in page.get("Items", []) if a.get("ApiId"))
    return ids


def list_apigatewayv2_integration_responses(session: SessionLike) -> list[str]:
    """Integration responses (CCAPI id ``ApiId|IntegrationId|IntegrationResponseId``)."""
    client = session.client("apigatewayv2", config=RETRY_CONFIG)
    out: list[str] = []
    for api_id in _apigatewayv2_api_ids(client):
        try:
            integration_ids = [
                it["IntegrationId"]
                for page in client.get_paginator("get_integrations").paginate(ApiId=api_id)
                for it in page.get("Items", [])
                if it.get("IntegrationId")
            ]
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"apigatewayv2.get_integrations skipped {api_id}: {exc}")
            continue
        for integration_id in integration_ids:
            try:
                for page in client.get_paginator("get_integration_responses").paginate(
                    ApiId=api_id, IntegrationId=integration_id
                ):
                    out.extend(
                        f"{api_id}|{integration_id}|{r['IntegrationResponseId']}"
                        for r in page.get("Items", [])
                        if r.get("IntegrationResponseId")
                    )
            except (ClientError, BotoCoreError) as exc:
                logger.debug(
                    f"apigatewayv2.get_integration_responses skipped "
                    f"{api_id}/{integration_id}: {exc}"
                )
    return out


def list_apigatewayv2_route_responses(session: SessionLike) -> list[str]:
    """Route responses (CCAPI id ``ApiId|RouteId|RouteResponseId``)."""
    client = session.client("apigatewayv2", config=RETRY_CONFIG)
    out: list[str] = []
    for api_id in _apigatewayv2_api_ids(client):
        try:
            route_ids = [
                rt["RouteId"]
                for page in client.get_paginator("get_routes").paginate(ApiId=api_id)
                for rt in page.get("Items", [])
                if rt.get("RouteId")
            ]
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"apigatewayv2.get_routes skipped {api_id}: {exc}")
            continue
        for route_id in route_ids:
            try:
                for page in client.get_paginator("get_route_responses").paginate(
                    ApiId=api_id, RouteId=route_id
                ):
                    out.extend(
                        f"{api_id}|{route_id}|{r['RouteResponseId']}"
                        for r in page.get("Items", [])
                        if r.get("RouteResponseId")
                    )
            except (ClientError, BotoCoreError) as exc:
                logger.debug(f"apigatewayv2.get_route_responses skipped {api_id}/{route_id}: {exc}")
    return out


def list_backup_restore_testing_selections(session: SessionLike) -> list[str]:
    """Restore-testing selections (id ``RestoreTestingPlanName|RestoreTestingSelectionName``)."""
    client = session.client("backup", config=RETRY_CONFIG)
    plan_names = [
        p["RestoreTestingPlanName"]
        for page in client.get_paginator("list_restore_testing_plans").paginate()
        for p in page.get("RestoreTestingPlans", [])
        if p.get("RestoreTestingPlanName")
    ]
    out: list[str] = []
    for plan_name in plan_names:
        try:
            for page in client.get_paginator("list_restore_testing_selections").paginate(
                RestoreTestingPlanName=plan_name
            ):
                out.extend(
                    f"{plan_name}|{s['RestoreTestingSelectionName']}"
                    for s in page.get("RestoreTestingSelections", [])
                    if s.get("RestoreTestingSelectionName")
                )
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"backup.list_restore_testing_selections skipped {plan_name}: {exc}")
    return out


def list_iotsitewise_projects(session: SessionLike) -> list[str]:
    """SiteWise projects across every portal (CCAPI id ``ProjectId``)."""
    client = session.client("iotsitewise", config=RETRY_CONFIG)
    portal_ids = [
        p["id"]
        for page in client.get_paginator("list_portals").paginate()
        for p in page.get("portalSummaries", [])
        if p.get("id")
    ]
    out: list[str] = []
    for portal_id in portal_ids:
        try:
            for page in client.get_paginator("list_projects").paginate(portalId=portal_id):
                out.extend(pr["id"] for pr in page.get("projectSummaries", []) if pr.get("id"))
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"iotsitewise.list_projects skipped {portal_id}: {exc}")
    return out


def list_lambda_urls(session: SessionLike) -> list[str]:
    """Function URLs (CCAPI id ``FunctionArn`` of the URL-owning function/alias)."""
    client = session.client("lambda", config=RETRY_CONFIG)
    function_names = [
        f["FunctionName"]
        for page in client.get_paginator("list_functions").paginate()
        for f in page.get("Functions", [])
        if f.get("FunctionName")
    ]
    out: list[str] = []
    for function_name in function_names:
        try:
            for page in client.get_paginator("list_function_url_configs").paginate(
                FunctionName=function_name
            ):
                out.extend(
                    u["FunctionArn"]
                    for u in page.get("FunctionUrlConfigs", [])
                    if u.get("FunctionArn")
                )
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"lambda.list_function_url_configs skipped {function_name}: {exc}")
    return out


def list_mediapackagev2_origin_endpoint_policies(session: SessionLike) -> list[str]:
    """Origin-endpoint policies (CCAPI id ``ChannelGroupName|ChannelName|OriginEndpointName``).

    Only endpoints that actually carry a resource policy are emitted; a ``ResourceNotFound``
    from ``get_origin_endpoint_policy`` is the normal "no policy" signal and is skipped.
    """
    client = session.client("mediapackagev2", config=RETRY_CONFIG)
    out: list[str] = []
    for group_page in client.get_paginator("list_channel_groups").paginate():
        for group in group_page.get("Items", []):
            group_name = group.get("ChannelGroupName")
            if not group_name:
                continue
            for endpoint in _mediapackagev2_endpoints(client, group_name):
                channel_name, endpoint_name = endpoint
                if _mediapackagev2_has_policy(client, group_name, channel_name, endpoint_name):
                    out.append(f"{group_name}|{channel_name}|{endpoint_name}")
    return out


def _mediapackagev2_endpoints(client: object, group_name: str) -> list[tuple[str, str]]:
    """(ChannelName, OriginEndpointName) pairs under one channel group."""
    pairs: list[tuple[str, str]] = []
    try:
        channel_names = [
            c["ChannelName"]
            for page in client.get_paginator("list_channels").paginate(  # type: ignore[attr-defined]
                ChannelGroupName=group_name
            )
            for c in page.get("Items", [])
            if c.get("ChannelName")
        ]
    except (ClientError, BotoCoreError) as exc:
        logger.debug(f"mediapackagev2.list_channels skipped {group_name}: {exc}")
        return pairs
    for channel_name in channel_names:
        try:
            for page in client.get_paginator("list_origin_endpoints").paginate(  # type: ignore[attr-defined]
                ChannelGroupName=group_name, ChannelName=channel_name
            ):
                pairs.extend(
                    (channel_name, e["OriginEndpointName"])
                    for e in page.get("Items", [])
                    if e.get("OriginEndpointName")
                )
        except (ClientError, BotoCoreError) as exc:
            logger.debug(
                f"mediapackagev2.list_origin_endpoints skipped {group_name}/{channel_name}: {exc}"
            )
    return pairs


def _mediapackagev2_has_policy(
    client: object, group_name: str, channel_name: str, endpoint_name: str
) -> bool:
    """True if the origin endpoint carries a resource policy (ResourceNotFound => no policy)."""
    try:
        resp = client.get_origin_endpoint_policy(  # type: ignore[attr-defined]
            ChannelGroupName=group_name,
            ChannelName=channel_name,
            OriginEndpointName=endpoint_name,
        )
        return bool(resp.get("Policy"))
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            logger.debug(
                f"mediapackagev2.get_origin_endpoint_policy skipped "
                f"{group_name}/{channel_name}/{endpoint_name}: {exc}"
            )
        return False
    except BotoCoreError as exc:
        logger.debug(
            f"mediapackagev2.get_origin_endpoint_policy skipped "
            f"{group_name}/{channel_name}/{endpoint_name}: {exc}"
        )
        return False


def _pinpoint_app_ids(client: object) -> list[str]:
    """Every Pinpoint application id (parent for the per-app channel/settings listers)."""
    ids: list[str] = []
    token: str | None = None
    while True:
        kwargs = {"Token": token} if token else {}
        resp = client.get_apps(**kwargs)  # type: ignore[attr-defined]
        body = resp.get("ApplicationsResponse", {})
        ids.extend(a["Id"] for a in body.get("Item", []) if a.get("Id"))
        token = body.get("NextToken")
        if not token:
            return ids


def _pinpoint_channel_ids(session: SessionLike, get_op: str, response_key: str) -> list[str]:
    """Ids of a single-per-app Pinpoint channel across every app (CCAPI id = channel ``Id``).

    ``get_op`` (e.g. ``get_sms_channel``) returns one channel wrapped in ``response_key``
    (e.g. ``SMSChannelResponse``); a ``NotFoundException`` means the app has no such channel.
    """
    client = session.client("pinpoint", config=RETRY_CONFIG)
    out: list[str] = []
    for app_id in _pinpoint_app_ids(client):
        try:
            resp = getattr(client, get_op)(ApplicationId=app_id)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") != "NotFoundException":
                logger.debug(f"pinpoint.{get_op} skipped {app_id}: {exc}")
            continue
        except BotoCoreError as exc:
            logger.debug(f"pinpoint.{get_op} skipped {app_id}: {exc}")
            continue
        channel = resp.get(response_key, {})
        channel_id = channel.get("Id")
        if channel_id:
            out.append(channel_id)
    return out


def list_pinpoint_sms_channels(session: SessionLike) -> list[str]:
    """SMS channels across every app (CCAPI id = channel ``Id``)."""
    return _pinpoint_channel_ids(session, "get_sms_channel", "SMSChannelResponse")


def list_pinpoint_apns_sandbox_channels(session: SessionLike) -> list[str]:
    """APNS sandbox channels across every app (CCAPI id = channel ``Id``)."""
    return _pinpoint_channel_ids(session, "get_apns_sandbox_channel", "APNSSandboxChannelResponse")


def list_pinpoint_apns_voip_channels(session: SessionLike) -> list[str]:
    """APNS VOIP channels across every app (CCAPI id = channel ``Id``)."""
    return _pinpoint_channel_ids(session, "get_apns_voip_channel", "APNSVoipChannelResponse")


def list_pinpoint_apns_voip_sandbox_channels(session: SessionLike) -> list[str]:
    """APNS VOIP sandbox channels across every app (CCAPI id = channel ``Id``)."""
    return _pinpoint_channel_ids(
        session, "get_apns_voip_sandbox_channel", "APNSVoipSandboxChannelResponse"
    )


def list_pinpoint_application_settings(session: SessionLike) -> list[str]:
    """Application settings per app (CCAPI id = settings ``ApplicationId``)."""
    client = session.client("pinpoint", config=RETRY_CONFIG)
    out: list[str] = []
    for app_id in _pinpoint_app_ids(client):
        try:
            resp = client.get_application_settings(ApplicationId=app_id)
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"pinpoint.get_application_settings skipped {app_id}: {exc}")
            continue
        settings_id = resp.get("ApplicationSettingsResource", {}).get("ApplicationId")
        if settings_id:
            out.append(settings_id)
    return out


def list_pinpoint_in_app_templates(session: SessionLike) -> list[str]:
    """In-app message templates (CCAPI id = ``TemplateName``); filtered by ``TemplateType=INAPP``.

    ``list_templates`` returns every template type intermixed; the ``TemplateType`` input filter
    plus a defensive per-item ``TemplateType == 'INAPP'`` check keep this pinned to exactly the
    ``InAppTemplate`` type (no email/push/SMS/voice templates leak in).
    """
    client = session.client("pinpoint", config=RETRY_CONFIG)
    out: list[str] = []
    token: str | None = None
    while True:
        kwargs = {"TemplateType": "INAPP"}
        if token:
            kwargs["NextToken"] = token
        resp = client.list_templates(**kwargs)
        body = resp.get("TemplatesResponse", {})
        out.extend(
            t["TemplateName"]
            for t in body.get("Item", [])
            if t.get("TemplateName") and t.get("TemplateType") == "INAPP"
        )
        token = body.get("NextToken")
        if not token:
            return out


def list_s3tables_table_bucket_policies(session: SessionLike) -> list[str]:
    """Table buckets that carry a resource policy (CCAPI id = ``TableBucketARN``).

    A ``NotFoundException`` from ``get_table_bucket_policy`` is the normal "no policy" signal.
    """
    client = session.client("s3tables", config=RETRY_CONFIG)
    bucket_arns: list[str] = []
    token: str | None = None
    while True:
        kwargs = {"continuationToken": token} if token else {}
        resp = client.list_table_buckets(**kwargs)
        bucket_arns.extend(b["arn"] for b in resp.get("tableBuckets", []) if b.get("arn"))
        token = resp.get("continuationToken")
        if not token:
            break
    out: list[str] = []
    for arn in bucket_arns:
        try:
            policy = client.get_table_bucket_policy(tableBucketARN=arn)
            if policy.get("resourcePolicy"):
                out.append(arn)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") != "NotFoundException":
                logger.debug(f"s3tables.get_table_bucket_policy skipped {arn}: {exc}")
        except BotoCoreError as exc:
            logger.debug(f"s3tables.get_table_bucket_policy skipped {arn}: {exc}")
    return out


def _sso_instance_arns(client: object) -> list[str]:
    """Every IAM Identity Center instance ARN (parent for the per-instance SSO listers)."""
    arns: list[str] = []
    for page in client.get_paginator("list_instances").paginate():  # type: ignore[attr-defined]
        arns.extend(i["InstanceArn"] for i in page.get("Instances", []) if i.get("InstanceArn"))
    return arns


def list_sso_application_assignments(session: SessionLike) -> list[str]:
    """Application assignments (CCAPI id ``ApplicationArn|PrincipalType|PrincipalId``)."""
    client = session.client("sso-admin", config=RETRY_CONFIG)
    out: list[str] = []
    for instance_arn in _sso_instance_arns(client):
        try:
            application_arns = [
                a["ApplicationArn"]
                for page in client.get_paginator("list_applications").paginate(
                    InstanceArn=instance_arn
                )
                for a in page.get("Applications", [])
                if a.get("ApplicationArn")
            ]
        except (ClientError, BotoCoreError) as exc:
            logger.debug(f"sso-admin.list_applications skipped {instance_arn}: {exc}")
            continue
        for application_arn in application_arns:
            try:
                for page in client.get_paginator("list_application_assignments").paginate(
                    ApplicationArn=application_arn
                ):
                    out.extend(
                        f"{a['ApplicationArn']}|{a['PrincipalType']}|{a['PrincipalId']}"
                        for a in page.get("ApplicationAssignments", [])
                        if a.get("ApplicationArn")
                        and a.get("PrincipalType")
                        and a.get("PrincipalId")
                    )
            except (ClientError, BotoCoreError) as exc:
                logger.debug(
                    f"sso-admin.list_application_assignments skipped {application_arn}: {exc}"
                )
    return out


def list_sso_instance_access_control_attribute_configs(session: SessionLike) -> list[str]:
    """Instances that have an access-control attribute configuration (CCAPI id ``InstanceArn``).

    There is at most one configuration per instance; a ``ResourceNotFoundException`` means the
    instance has none, so only instances that return a configuration are emitted.
    """
    client = session.client("sso-admin", config=RETRY_CONFIG)
    out: list[str] = []
    for instance_arn in _sso_instance_arns(client):
        try:
            client.describe_instance_access_control_attribute_configuration(
                InstanceArn=instance_arn
            )
            out.append(instance_arn)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
                logger.debug(
                    f"sso-admin.describe_instance_access_control_attribute_configuration "
                    f"skipped {instance_arn}: {exc}"
                )
        except BotoCoreError as exc:
            logger.debug(
                f"sso-admin.describe_instance_access_control_attribute_configuration "
                f"skipped {instance_arn}: {exc}"
            )
    return out


# --- Code-lister batch 5: parent->child + type-discriminator listers. --------------------------


def list_amp_resource_policies(session: SessionLike) -> list[str]:
    """APS workspace resource policies (id = WorkspaceArn of each workspace that has one)."""
    client = session.client("amp", config=RETRY_CONFIG)
    out: list[str] = []
    for page in client.get_paginator("list_workspaces").paginate():
        for ws in page.get("workspaces", []):
            wsid = ws.get("workspaceId")
            arn = ws.get("arn")
            if not wsid or not arn:
                continue
            try:
                client.describe_resource_policy(workspaceId=wsid)
            except (ClientError, BotoCoreError):
                # No resource policy on this workspace (or transient) — not a policy resource.
                continue
            out.append(arn)
    return out


def list_kinesis_stream_consumers(session: SessionLike) -> list[str]:
    """Kinesis stream consumers (id = ConsumerARN) across every stream (two-step per stream)."""
    client = session.client("kinesis", config=RETRY_CONFIG)
    stream_arns: list[str] = []
    for page in client.get_paginator("list_streams").paginate():
        stream_arns.extend(
            s["StreamARN"] for s in page.get("StreamSummaries", []) if s.get("StreamARN")
        )
    out: list[str] = []
    for stream_arn in stream_arns:
        try:
            token: str | None = None
            while True:
                kwargs = {"StreamARN": stream_arn}
                if token:
                    kwargs["NextToken"] = token
                resp = client.list_stream_consumers(**kwargs)
                out.extend(
                    c["ConsumerARN"] for c in resp.get("Consumers", []) if c.get("ConsumerARN")
                )
                token = resp.get("NextToken")
                if not token:
                    break
        except (ClientError, BotoCoreError):
            continue
    return out


def list_neptunegraph_private_graph_endpoints(session: SessionLike) -> list[str]:
    """Neptune-graph private endpoints (id = ``<graphId>_<vpcId>``) across every graph."""
    client = session.client("neptune-graph", config=RETRY_CONFIG)
    graph_ids: list[str] = []
    for page in client.get_paginator("list_graphs").paginate():
        graph_ids.extend(g["id"] for g in page.get("graphs", []) if g.get("id"))
    out: list[str] = []
    for graph_id in graph_ids:
        try:
            for page in client.get_paginator("list_private_graph_endpoints").paginate(
                graphIdentifier=graph_id
            ):
                for endpoint in page.get("privateGraphEndpoints", []):
                    vpc = endpoint.get("vpcId")
                    if vpc:
                        out.append(f"{graph_id}_{vpc}")
        except (ClientError, BotoCoreError):
            continue
    return out


def list_route53_hosted_zones(session: SessionLike) -> list[str]:
    """All hosted zones (public + private) as the CCAPI primaryIdentifier: the bare ``Z...`` id.

    ``list_hosted_zones`` returns ``Id`` as ``/hostedzone/<id>``, but CCAPI's
    ``AWS::Route53::HostedZone`` primaryIdentifier is the bare ``<id>`` and rejects the prefixed
    form with ``InvalidRequestException``, so the orphan re-check and CCAPI delete need it stripped.
    """
    client = session.client("route53", config=RETRY_CONFIG)
    zone_ids: list[str] = []
    for page in client.get_paginator("list_hosted_zones").paginate():  # type: ignore[attr-defined]
        for zone in page.get("HostedZones", []):
            zone_id = (zone.get("Id") or "").split("/")[-1]
            if zone_id:
                zone_ids.append(zone_id)
    return zone_ids


def _public_hosted_zone_ids(client: object) -> list[str]:
    """Bare ids (``Z...``) of every public hosted zone (DNSSEC applies only to public zones)."""
    zone_ids: list[str] = []
    for page in client.get_paginator("list_hosted_zones").paginate():  # type: ignore[attr-defined]
        for zone in page.get("HostedZones", []):
            if (zone.get("Config") or {}).get("PrivateZone"):
                continue
            zone_id = (zone.get("Id") or "").split("/")[-1]
            if zone_id:
                zone_ids.append(zone_id)
    return zone_ids


def list_route53_dnssec(session: SessionLike) -> list[str]:
    """Route53 DNSSEC configs (id = HostedZoneId) for zones that have DNSSEC set up."""
    client = session.client("route53", config=RETRY_CONFIG)
    out: list[str] = []
    for zone_id in _public_hosted_zone_ids(client):
        try:
            resp = client.get_dnssec(HostedZoneId=zone_id)
        except (ClientError, BotoCoreError):
            continue
        serve = (resp.get("Status") or {}).get("ServeSignature")
        if resp.get("KeySigningKeys") or serve == "SIGNING":
            out.append(zone_id)
    return out


def list_route53_key_signing_keys(session: SessionLike) -> list[str]:
    """Route53 key-signing keys (id = ``<HostedZoneId>|<Name>``) across every public zone."""
    client = session.client("route53", config=RETRY_CONFIG)
    out: list[str] = []
    for zone_id in _public_hosted_zone_ids(client):
        try:
            resp = client.get_dnssec(HostedZoneId=zone_id)
        except (ClientError, BotoCoreError):
            continue
        for ksk in resp.get("KeySigningKeys", []):
            name = ksk.get("Name")
            if name:
                out.append(f"{zone_id}|{name}")
    return out


def list_datazone_project_profiles(session: SessionLike) -> list[str]:
    """DataZone project profiles (id = ``<domainId>|<profileId>``) across every domain."""
    client = session.client("datazone", config=RETRY_CONFIG)
    domain_ids: list[str] = []
    for page in client.get_paginator("list_domains").paginate():
        domain_ids.extend(d["id"] for d in page.get("items", []) if d.get("id"))
    out: list[str] = []
    for domain_id in domain_ids:
        try:
            for page in client.get_paginator("list_project_profiles").paginate(
                domainIdentifier=domain_id
            ):
                for profile in page.get("items", []):
                    pid = profile.get("id")
                    if pid:
                        out.append(f"{domain_id}|{pid}")
        except (ClientError, BotoCoreError):
            continue
    return out


def list_bedrockagentcore_gateway_targets(session: SessionLike) -> list[str]:
    """BedrockAgentCore gateway targets (id = ``<gatewayId>|<targetId>``) across every gateway."""
    client = session.client("bedrock-agentcore-control", config=RETRY_CONFIG)
    gateway_ids: list[str] = []
    for page in client.get_paginator("list_gateways").paginate():
        gateway_ids.extend(g["gatewayId"] for g in page.get("items", []) if g.get("gatewayId"))
    out: list[str] = []
    for gateway_id in gateway_ids:
        try:
            for page in client.get_paginator("list_gateway_targets").paginate(
                gatewayIdentifier=gateway_id
            ):
                for target in page.get("items", []):
                    tid = target.get("targetId")
                    if tid:
                        out.append(f"{gateway_id}|{tid}")
        except (ClientError, BotoCoreError):
            continue
    return out


def list_bedrockagentcore_browser_custom(session: SessionLike) -> list[str]:
    """Customer BedrockAgentCore browsers (id = browserId; ``type=CUSTOM`` excludes AWS SYSTEM)."""
    client = session.client("bedrock-agentcore-control", config=RETRY_CONFIG)
    return [
        b["browserId"]
        for page in client.get_paginator("list_browsers").paginate(type="CUSTOM")
        for b in page.get("browserSummaries", [])
        if b.get("browserId")
    ]


def list_bedrockagentcore_code_interpreter_custom(session: SessionLike) -> list[str]:
    """Customer BedrockAgentCore code interpreters (id = codeInterpreterId; ``type=CUSTOM``)."""
    client = session.client("bedrock-agentcore-control", config=RETRY_CONFIG)
    return [
        ci["codeInterpreterId"]
        for page in client.get_paginator("list_code_interpreters").paginate(type="CUSTOM")
        for ci in page.get("codeInterpreterSummaries", [])
        if ci.get("codeInterpreterId")
    ]


def list_s3vectors_vector_bucket_policies(session: SessionLike) -> list[str]:
    """S3 Vectors bucket policies (id = vectorBucketArn) for buckets that have a policy."""
    client = session.client("s3vectors", config=RETRY_CONFIG)
    buckets: list[tuple[str, str]] = []
    token: str | None = None
    while True:
        kwargs = {"nextToken": token} if token else {}
        resp = client.list_vector_buckets(**kwargs)
        for bucket in resp.get("vectorBuckets", []):
            name = bucket.get("vectorBucketName")
            arn = bucket.get("vectorBucketArn")
            if name and arn:
                buckets.append((name, arn))
        token = resp.get("nextToken")
        if not token:
            break
    out: list[str] = []
    for name, arn in buckets:
        try:
            client.get_vector_bucket_policy(vectorBucketName=name)
        except (ClientError, BotoCoreError):
            # NotFoundException = bucket has no policy (or transient) — not a policy resource.
            continue
        out.append(arn)
    return out


def list_cleanroomsml_configured_model_algorithm_associations(session: SessionLike) -> list[str]:
    """CleanRoomsML configured-model-algorithm associations (id = the association ARN).

    Enumerates Clean Rooms memberships (the parent required by the ML list op), then the
    associations under each membership.
    """
    cleanrooms = session.client("cleanrooms", config=RETRY_CONFIG)
    cleanroomsml = session.client("cleanroomsml", config=RETRY_CONFIG)
    membership_ids: list[str] = []
    for page in cleanrooms.get_paginator("list_memberships").paginate():
        membership_ids.extend(m["id"] for m in page.get("membershipSummaries", []) if m.get("id"))
    out: list[str] = []
    for membership_id in membership_ids:
        try:
            paginator = cleanroomsml.get_paginator("list_configured_model_algorithm_associations")
            for page in paginator.paginate(membershipIdentifier=membership_id):
                out.extend(
                    a["configuredModelAlgorithmAssociationArn"]
                    for a in page.get("configuredModelAlgorithmAssociations", [])
                    if a.get("configuredModelAlgorithmAssociationArn")
                )
        except (ClientError, BotoCoreError):
            continue
    return out


# Every hand-written lister as a :class:`Lister`. ``op`` mirrors the CFN-facing operation name
# used for type attribution; a ``cfn_type`` (when set) pins the lister's ids to that exact
# CloudFormation type directly instead of guessing by op-noun — required for sub-resource/
# derived types (EIP from DescribeAddresses, BucketPolicy from ListBuckets, the EC2
# *Attachment/*Association types) and the redshift/logs/imagebuilder parity listers.
_LISTERS: tuple[Lister, ...] = (
    # List* listers with real logic.
    Lister(
        "codedeploy",
        "ListOnPremisesInstances",
        list_codedeploy_registered_on_premises_instances,
    ),
    Lister("bedrock", "ListInferenceProfiles", list_bedrock_inference_profiles),
    Lister("bedrock", "ListPromptRouters", list_bedrock_prompt_routers),
    Lister("cloudformation", "ListStacks", list_cloudformation_stacks),
    Lister("elasticbeanstalk", "ListPlatformVersions", list_elasticbeanstalk_platform_versions),
    Lister(
        "cognito-identity",
        "ListIdentityPools",
        list_cognito_identity_pools,
        "AWS::Cognito::IdentityPool",
    ),
    Lister("waf", "ListLoggingConfigurations", list_waf_logging_configurations),
    Lister("waf-regional", "ListLoggingConfigurations", list_waf_regional_logging_configurations),
    Lister("iam", "ListPolicies", list_iam_policies, "AWS::IAM::ManagedPolicy"),
    Lister("iam", "ListRoles", list_iam_roles, "AWS::IAM::Role"),
    Lister("kms", "ListAliases", list_kms_aliases, "AWS::KMS::Alias"),
    Lister("kms", "ListKeys", list_kms_keys, "AWS::KMS::Key"),
    Lister(
        "medialive",
        "ListCloudWatchAlarmTemplateGroups",
        list_medialive_cloud_watch_alarm_template_groups,
    ),
    Lister("medialive", "ListCloudWatchAlarmTemplates", list_medialive_cloud_watch_alarm_templates),
    Lister("ram", "ListPermissions", list_ram_permissions),
    Lister(
        "sagemaker",
        "ListTrialComponents",
        list_sagemaker_trial_components,
        "AWS::SageMaker::TrialComponent",
    ),
    Lister("sns", "ListSubscriptions", list_sns_subscriptions),
    Lister("ssm", "ListDocuments", list_ssm_documents, "AWS::SSM::Document"),
    # Describe* listers with real logic.
    Lister("appstream", "DescribeImages", describe_appstream_images, "AWS::AppStream::Image"),
    Lister("cloudformation", "DescribeStacks", describe_cloudformation_stacks),
    Lister("ec2", "DescribeFpgaImages", describe_ec2_fpga_images, "AWS::EC2::FpgaImage"),
    Lister("ec2", "DescribeImages", describe_ec2_images, "AWS::EC2::Image"),
    Lister("ec2", "DescribeNatGateways", describe_ec2_nat_gateways, "AWS::EC2::NatGateway"),
    Lister("ec2", "DescribeManagedPrefixLists", describe_ec2_prefix_lists, "AWS::EC2::PrefixList"),
    Lister("ec2", "DescribeSnapshots", describe_ec2_snapshots, "AWS::EC2::Snapshot"),
    Lister(
        "ec2",
        "DescribeVpcEndpointServiceConfigurations",
        describe_ec2_vpc_endpoint_services,
        "AWS::EC2::VPCEndpointService",
    ),
    # Supplementary listers pinned to their exact CFN type.
    Lister(
        "cloudfront",
        "ListDistributions",
        list_cloudfront_list_distributions,
        "AWS::CloudFront::Distribution",
    ),
    Lister(
        "cloudfront", "ListFunctions", list_cloudfront_list_functions, "AWS::CloudFront::Function"
    ),
    Lister(
        "cloudfront",
        "ListCloudFrontOriginAccessIdentities",
        list_cloudfront_list_cloud_front_origin_access_identities,
        "AWS::CloudFront::CloudFrontOriginAccessIdentity",
    ),
    Lister(
        "cloudfront",
        "ListConnectionGroups",
        list_cloudfront_connection_groups,
        "AWS::CloudFront::ConnectionGroup",
    ),
    Lister(
        "acm-pca",
        "ListCertificateAuthorities",
        list_acm_pca_list_certificate_authorities,
        "AWS::ACMPCA::CertificateAuthority",
    ),
    Lister("amp", "ListWorkspaces", list_amp_list_workspaces, "AWS::APS::Workspace"),
    Lister("amp", "ListScrapers", list_amp_list_scrapers, "AWS::APS::Scraper"),
    Lister("mq", "ListBrokers", list_mq_list_brokers, "AWS::AmazonMQ::Broker"),
    Lister("mq", "ListConfigurations", list_mq_list_configurations, "AWS::AmazonMQ::Configuration"),
    Lister(
        "cognito-idp", "ListUserPools", list_cognito_idp_list_user_pools, "AWS::Cognito::UserPool"
    ),
    Lister(
        "ecs",
        "DescribeCapacityProviders",
        list_ecs_describe_capacity_providers,
        "AWS::ECS::CapacityProvider",
    ),
    Lister("lexv2-models", "ListBots", list_lexv2_models_list_bots, "AWS::Lex::Bot"),
    Lister(
        "backup",
        "ListBackupSelections",
        list_backup_list_backup_selections,
        "AWS::Backup::BackupSelection",
    ),
    # NOTE: the plain ``DescribeSecurityGroupRules`` lister is intentionally absent. SG rules split
    # into ::SecurityGroupIngress / ::SecurityGroupEgress by IsEgress, and a bare ``sgr-*`` id can't
    # be typed without that flag — so the two direction-split listers below
    # (``DescribeSecurityGroupRulesIngress`` / ``...Egress``) cover every rule, each pinned to its
    # exact CFN type. Pinning the mixed call to one type would mis-type the other direction.
    Lister("ec2", "DescribeAddresses", list_ec2_describe_addresses, "AWS::EC2::EIP"),
    Lister(
        "ec2",
        "DescribeEipAssociations",
        list_ec2_describe_eip_associations,
        "AWS::EC2::EIPAssociation",
    ),
    Lister(
        "ec2",
        "DescribeNicAttachments",
        list_ec2_describe_nic_attachments,
        "AWS::EC2::NetworkInterfaceAttachment",
    ),
    Lister(
        "ec2",
        "DescribeVolumeAttachments",
        list_ec2_describe_volume_attachments,
        "AWS::EC2::VolumeAttachment",
    ),
    Lister(
        "ec2",
        "DescribeSubnetRouteTableAssociations",
        list_ec2_describe_subnet_route_table_associations,
        "AWS::EC2::SubnetRouteTableAssociation",
    ),
    Lister(
        "ec2",
        "DescribeSubnetNaclAssociations",
        list_ec2_describe_subnet_nacl_associations,
        "AWS::EC2::SubnetNetworkAclAssociation",
    ),
    Lister(
        "ec2",
        "DescribeVpcGatewayAttachments",
        list_ec2_describe_vpc_gateway_attachments,
        "AWS::EC2::VPCGatewayAttachment",
    ),
    Lister(
        "ec2",
        "DescribeSecurityGroups",
        list_ec2_describe_all_security_groups,
        "AWS::EC2::SecurityGroup",
    ),
    Lister(
        "ec2", "DescribeNetworkAcls", list_ec2_describe_all_network_acls, "AWS::EC2::NetworkAcl"
    ),
    Lister(
        "ec2", "DescribeRouteTables", list_ec2_describe_all_route_tables, "AWS::EC2::RouteTable"
    ),
    Lister("s3", "ListBucketPolicies", list_s3_bucket_policies, "AWS::S3::BucketPolicy"),
    Lister("ec2", "DescribeSubnets", list_ec2_describe_all_subnets, "AWS::EC2::Subnet"),
    Lister("ec2", "DescribeVpcs", list_ec2_describe_all_vpcs, "AWS::EC2::VPC"),
    Lister(
        "ec2",
        "DescribeInternetGateways",
        list_ec2_describe_all_internet_gateways,
        "AWS::EC2::InternetGateway",
    ),
    Lister(
        "ec2",
        "DescribeDhcpOptions",
        list_ec2_describe_dhcp_options,
        "AWS::EC2::DHCPOptions",
    ),
    Lister("ecs", "ListServices", list_ecs_list_services, "AWS::ECS::Service"),
    Lister("logs", "describe_metric_filters", list_logs_metric_filters, "AWS::Logs::MetricFilter"),
    Lister("eks", "ListAddons", list_eks_addons, "AWS::EKS::Addon"),
    Lister(
        "eks",
        "ListPodIdentityAssociations",
        list_eks_pod_identity_associations,
        "AWS::EKS::PodIdentityAssociation",
    ),
    Lister("eks", "ListNodegroups", list_eks_nodegroups, "AWS::EKS::Nodegroup"),
    Lister(
        "redshift",
        "DescribeClusters",
        list_redshift_clusters_by_identifier,
        "AWS::Redshift::Cluster",
    ),
    Lister(
        "redshift-serverless",
        "ListNamespaces",
        list_redshift_serverless_namespaces_by_name,
        "AWS::RedshiftServerless::Namespace",
    ),
    Lister(
        "redshift-serverless",
        "ListWorkgroups",
        list_redshift_serverless_workgroups_by_name,
        "AWS::RedshiftServerless::Workgroup",
    ),
    Lister("logs", "DescribeLogGroups", list_logs_log_groups_by_name, "AWS::Logs::LogGroup"),
    Lister(
        "imagebuilder",
        "ListComponentBuildVersions",
        list_imagebuilder_component_build_versions,
        "AWS::ImageBuilder::Component",
    ),
    Lister("imagebuilder", "ListImages", list_imagebuilder_images, "AWS::ImageBuilder::Image"),
    Lister(
        "s3",
        "ListStorageLensConfigurations",
        list_s3_list_storage_lens_configurations,
        "AWS::S3::StorageLens",
    ),
    # Code-lister batch 0: parent->child + discriminator listers (composite `|`-joined ids).
    Lister(
        "codestar-connections",
        "ListSyncConfigurations",
        list_codestar_connections_sync_configurations,
        "AWS::CodeStarConnections::SyncConfiguration",
    ),
    Lister(
        "datasync", "DescribeLocationS3", list_datasync_location_s3, "AWS::DataSync::LocationS3"
    ),
    Lister(
        "datasync", "DescribeLocationEfs", list_datasync_location_efs, "AWS::DataSync::LocationEFS"
    ),
    Lister(
        "datasync", "DescribeLocationNfs", list_datasync_location_nfs, "AWS::DataSync::LocationNFS"
    ),
    Lister(
        "datasync", "DescribeLocationSmb", list_datasync_location_smb, "AWS::DataSync::LocationSMB"
    ),
    Lister(
        "datasync",
        "DescribeLocationHdfs",
        list_datasync_location_hdfs,
        "AWS::DataSync::LocationHDFS",
    ),
    Lister(
        "datasync",
        "DescribeLocationObjectStorage",
        list_datasync_location_object_storage,
        "AWS::DataSync::LocationObjectStorage",
    ),
    Lister(
        "datasync",
        "DescribeLocationAzureBlob",
        list_datasync_location_azure_blob,
        "AWS::DataSync::LocationAzureBlob",
    ),
    Lister(
        "datasync",
        "DescribeLocationFsxLustre",
        list_datasync_location_fsx_lustre,
        "AWS::DataSync::LocationFSxLustre",
    ),
    Lister(
        "datasync",
        "DescribeLocationFsxOntap",
        list_datasync_location_fsx_ontap,
        "AWS::DataSync::LocationFSxONTAP",
    ),
    Lister(
        "datasync",
        "DescribeLocationFsxOpenZfs",
        list_datasync_location_fsx_openzfs,
        "AWS::DataSync::LocationFSxOpenZFS",
    ),
    Lister(
        "datasync",
        "DescribeLocationFsxWindows",
        list_datasync_location_fsx_windows,
        "AWS::DataSync::LocationFSxWindows",
    ),
    Lister(
        "omics",
        "ListWorkflowVersions",
        list_omics_workflow_versions,
        "AWS::Omics::WorkflowVersion",
    ),
    Lister(
        "pinpoint-sms-voice-v2",
        "GetResourcePolicy",
        list_smsvoice_resource_policies,
        "AWS::SMSVOICE::ResourcePolicy",
    ),
    Lister(
        "servicecatalog",
        "ListServiceActionsForProvisioningArtifact",
        list_servicecatalog_service_action_associations,
        "AWS::ServiceCatalog::ServiceActionAssociation",
    ),
    Lister(
        "servicecatalog",
        "ListResourcesForTagOption",
        list_servicecatalog_tag_option_associations,
        "AWS::ServiceCatalog::TagOptionAssociation",
    ),
    Lister(
        "kinesisanalyticsv2",
        "DescribeApplicationOutputs",
        list_kinesisanalyticsv2_application_outputs,
        "AWS::KinesisAnalyticsV2::ApplicationOutput",
    ),
    # Code-lister batch 1: parent->child two-step listers (composite `|`-joined ids).
    Lister(
        "amplify",
        "ListDomainAssociations",
        list_amplify_domain_associations,
        "AWS::Amplify::Domain",
    ),
    Lister(
        "config",
        "DescribeRemediationConfigurations",
        list_config_remediation_configurations,
        "AWS::Config::RemediationConfiguration",
    ),
    Lister(
        "globalaccelerator",
        "ListListeners",
        list_globalaccelerator_listeners,
        "AWS::GlobalAccelerator::Listener",
    ),
    Lister(
        "globalaccelerator",
        "ListEndpointGroups",
        list_globalaccelerator_endpoint_groups,
        "AWS::GlobalAccelerator::EndpointGroup",
    ),
    Lister("guardduty", "ListFilters", list_guardduty_filters, "AWS::GuardDuty::Filter"),
    Lister("guardduty", "ListIPSets", list_guardduty_ip_sets, "AWS::GuardDuty::IPSet"),
    Lister(
        "guardduty",
        "ListThreatIntelSets",
        list_guardduty_threat_intel_sets,
        "AWS::GuardDuty::ThreatIntelSet",
    ),
    Lister(
        "guardduty",
        "ListThreatEntitySets",
        list_guardduty_threat_entity_sets,
        "AWS::GuardDuty::ThreatEntitySet",
    ),
    Lister(
        "guardduty",
        "ListTrustedEntitySets",
        list_guardduty_trusted_entity_sets,
        "AWS::GuardDuty::TrustedEntitySet",
    ),
    Lister("guardduty", "ListMembers", list_guardduty_members, "AWS::GuardDuty::Member"),
    Lister(
        "guardduty",
        "ListPublishingDestinations",
        list_guardduty_publishing_destinations,
        "AWS::GuardDuty::PublishingDestination",
    ),
    Lister("guardduty", "GetMasterAccount", list_guardduty_masters, "AWS::GuardDuty::Master"),
    Lister(
        "location",
        "ListTrackerConsumers",
        list_location_tracker_consumers,
        "AWS::Location::TrackerConsumer",
    ),
    Lister(
        "organizations",
        "ListOrganizationalUnitsForParent",
        list_organizations_organizational_units,
        "AWS::Organizations::OrganizationalUnit",
    ),
    # Code-lister batch 2: parent->child + EC2 IsEgress-discriminator listers.
    Lister(
        "connect",
        "ListApprovedOrigins",
        list_connect_approved_origins,
        "AWS::Connect::ApprovedOrigin",
    ),
    Lister(
        "connect",
        "ListInstanceStorageConfigs",
        list_connect_instance_storage_configs,
        "AWS::Connect::InstanceStorageConfig",
    ),
    Lister(
        "connect",
        "ListSecurityKeys",
        list_connect_security_keys,
        "AWS::Connect::SecurityKey",
    ),
    Lister(
        "ec2",
        "SearchLocalGatewayRoutes",
        list_ec2_local_gateway_routes,
        "AWS::EC2::LocalGatewayRoute",
    ),
    Lister(
        "ec2",
        "GetRouteServerAssociations",
        list_ec2_route_server_associations,
        "AWS::EC2::RouteServerAssociation",
    ),
    Lister(
        "ec2",
        "GetRouteServerPropagations",
        list_ec2_route_server_propagations,
        "AWS::EC2::RouteServerPropagation",
    ),
    Lister(
        "ec2",
        "DescribeVpcEndpointServicePermissions",
        list_ec2_vpc_endpoint_service_permissions,
        "AWS::EC2::VPCEndpointServicePermissions",
    ),
    Lister(
        "ec2",
        "DescribeSecurityGroupRulesIngress",
        list_ec2_security_group_ingress_rules,
        "AWS::EC2::SecurityGroupIngress",
    ),
    Lister(
        "ec2",
        "DescribeSecurityGroupRulesEgress",
        list_ec2_security_group_egress_rules,
        "AWS::EC2::SecurityGroupEgress",
    ),
    Lister(
        "greengrass",
        "ListLoggerDefinitionVersions",
        list_greengrass_logger_definition_versions,
        "AWS::Greengrass::LoggerDefinitionVersion",
    ),
    Lister(
        "verifiedpermissions",
        "ListIdentitySources",
        list_verifiedpermissions_identity_sources,
        "AWS::VerifiedPermissions::IdentitySource",
    ),
    Lister(
        "verifiedpermissions",
        "ListPolicies",
        list_verifiedpermissions_policies,
        "AWS::VerifiedPermissions::Policy",
    ),
    Lister(
        "pca-connector-ad",
        "ListServicePrincipalNames",
        list_pcaconnectorad_service_principal_names,
        "AWS::PCAConnectorAD::ServicePrincipalName",
    ),
    Lister(
        "pca-connector-ad",
        "ListTemplateGroupAccessControlEntries",
        list_pcaconnectorad_template_group_access_control_entries,
        "AWS::PCAConnectorAD::TemplateGroupAccessControlEntry",
    ),
    Lister(
        "rtbfabric",
        "ListLinkRoutingRules",
        list_rtbfabric_link_routing_rules,
        "AWS::RTBFabric::LinkRoutingRule",
    ),
    Lister(
        "logs",
        "GetTransformer",
        list_logs_transformers,
        "AWS::Logs::Transformer",
    ),
    # Code-lister batch 3: NetworkManager AttachmentType discriminators + parent->child listers.
    Lister(
        "networkmanager",
        "ListAttachmentsConnect",
        list_networkmanager_connect_attachments,
        "AWS::NetworkManager::ConnectAttachment",
    ),
    Lister(
        "networkmanager",
        "ListAttachmentsSiteToSiteVpn",
        list_networkmanager_site_to_site_vpn_attachments,
        "AWS::NetworkManager::SiteToSiteVpnAttachment",
    ),
    Lister(
        "networkmanager",
        "ListAttachmentsVpc",
        list_networkmanager_vpc_attachments,
        "AWS::NetworkManager::VpcAttachment",
    ),
    Lister(
        "networkmanager",
        "ListAttachmentsDirectConnectGateway",
        list_networkmanager_direct_connect_gateway_attachments,
        "AWS::NetworkManager::DirectConnectGatewayAttachment",
    ),
    Lister(
        "networkmanager",
        "ListAttachmentsTransitGatewayRouteTable",
        list_networkmanager_transit_gateway_route_table_attachments,
        "AWS::NetworkManager::TransitGatewayRouteTableAttachment",
    ),
    Lister(
        "networkmanager",
        "GetTransitGatewayRegistrations",
        list_networkmanager_transit_gateway_registrations,
        "AWS::NetworkManager::TransitGatewayRegistration",
    ),
    Lister(
        "iam",
        "GetGroupMemberships",
        list_iam_user_to_group_additions,
        "AWS::IAM::UserToGroupAddition",
    ),
    Lister(
        "medialive",
        "ListChannelPlacementGroups",
        list_medialive_channel_placement_groups,
        "AWS::MediaLive::ChannelPlacementGroup",
    ),
    Lister(
        "qbusiness",
        "ListDataSources",
        list_qbusiness_data_sources,
        "AWS::QBusiness::DataSource",
    ),
    Lister(
        "s3files",
        "GetFileSystemPolicies",
        list_s3files_file_system_policies,
        "AWS::S3Files::FileSystemPolicy",
    ),
    Lister(
        "ecr",
        "GetRegistryPolicy",
        list_ecr_registry_policy,
        "AWS::ECR::RegistryPolicy",
    ),
    Lister(
        "secretsmanager",
        "GetResourcePolicies",
        list_secretsmanager_resource_policies,
        "AWS::SecretsManager::ResourcePolicy",
    ),
    Lister(
        "secretsmanager",
        "ListRotationSchedules",
        list_secretsmanager_rotation_schedules,
        "AWS::SecretsManager::RotationSchedule",
    ),
    Lister(
        "qconnect",
        "ListAIAgents",
        list_qconnect_ai_agents,
        "AWS::Wisdom::AIAgent",
    ),
    Lister(
        "qconnect",
        "ListAIAgentVersions",
        list_qconnect_ai_agent_versions,
        "AWS::Wisdom::AIAgentVersion",
    ),
    # Code-lister batch 4: parent->child + per-resource policy/channel listers.
    Lister(
        "apigatewayv2",
        "GetIntegrationResponses",
        list_apigatewayv2_integration_responses,
        "AWS::ApiGatewayV2::IntegrationResponse",
    ),
    Lister(
        "apigatewayv2",
        "GetRouteResponses",
        list_apigatewayv2_route_responses,
        "AWS::ApiGatewayV2::RouteResponse",
    ),
    Lister(
        "backup",
        "ListRestoreTestingSelections",
        list_backup_restore_testing_selections,
        "AWS::Backup::RestoreTestingSelection",
    ),
    Lister(
        "iotsitewise",
        "ListProjects",
        list_iotsitewise_projects,
        "AWS::IoTSiteWise::Project",
    ),
    Lister(
        "lambda",
        "ListFunctionUrlConfigs",
        list_lambda_urls,
        "AWS::Lambda::Url",
    ),
    Lister(
        "mediapackagev2",
        "GetOriginEndpointPolicy",
        list_mediapackagev2_origin_endpoint_policies,
        "AWS::MediaPackageV2::OriginEndpointPolicy",
    ),
    Lister(
        "pinpoint",
        "GetSmsChannel",
        list_pinpoint_sms_channels,
        "AWS::Pinpoint::SMSChannel",
    ),
    Lister(
        "pinpoint",
        "GetApnsSandboxChannel",
        list_pinpoint_apns_sandbox_channels,
        "AWS::Pinpoint::APNSSandboxChannel",
    ),
    Lister(
        "pinpoint",
        "GetApnsVoipChannel",
        list_pinpoint_apns_voip_channels,
        "AWS::Pinpoint::APNSVoipChannel",
    ),
    Lister(
        "pinpoint",
        "GetApnsVoipSandboxChannel",
        list_pinpoint_apns_voip_sandbox_channels,
        "AWS::Pinpoint::APNSVoipSandboxChannel",
    ),
    Lister(
        "pinpoint",
        "GetApplicationSettings",
        list_pinpoint_application_settings,
        "AWS::Pinpoint::ApplicationSettings",
    ),
    Lister(
        "pinpoint",
        "ListTemplates",
        list_pinpoint_in_app_templates,
        "AWS::Pinpoint::InAppTemplate",
    ),
    Lister(
        "s3tables",
        "GetTableBucketPolicy",
        list_s3tables_table_bucket_policies,
        "AWS::S3Tables::TableBucketPolicy",
    ),
    Lister(
        "sso-admin",
        "ListApplicationAssignments",
        list_sso_application_assignments,
        "AWS::SSO::ApplicationAssignment",
    ),
    Lister(
        "sso-admin",
        "DescribeInstanceAccessControlAttributeConfiguration",
        list_sso_instance_access_control_attribute_configs,
        "AWS::SSO::InstanceAccessControlAttributeConfiguration",
    ),
    # Code-lister batch 5: parent->child + type-discriminator listers.
    Lister(
        "amp",
        "DescribeResourcePolicy",
        list_amp_resource_policies,
        "AWS::APS::ResourcePolicy",
    ),
    Lister(
        "kinesis",
        "ListStreamConsumers",
        list_kinesis_stream_consumers,
        "AWS::Kinesis::StreamConsumer",
    ),
    Lister(
        "neptune-graph",
        "ListPrivateGraphEndpoints",
        list_neptunegraph_private_graph_endpoints,
        "AWS::NeptuneGraph::PrivateGraphEndpoint",
    ),
    Lister("route53", "ListHostedZones", list_route53_hosted_zones, "AWS::Route53::HostedZone"),
    Lister("wafv2", "ListWebACLs", list_wafv2_web_acls, "AWS::WAFv2::WebACL"),
    Lister("wafv2", "ListIPSets", list_wafv2_ip_sets, "AWS::WAFv2::IPSet"),
    Lister("route53", "GetDNSSEC", list_route53_dnssec, "AWS::Route53::DNSSEC"),
    Lister(
        "route53",
        "GetDNSSECKeySigningKeys",
        list_route53_key_signing_keys,
        "AWS::Route53::KeySigningKey",
    ),
    Lister(
        "datazone",
        "ListProjectProfiles",
        list_datazone_project_profiles,
        "AWS::DataZone::ProjectProfile",
    ),
    Lister(
        "bedrock-agentcore-control",
        "ListGatewayTargets",
        list_bedrockagentcore_gateway_targets,
        "AWS::BedrockAgentCore::GatewayTarget",
    ),
    Lister(
        "bedrock-agentcore-control",
        "ListBrowsersCustom",
        list_bedrockagentcore_browser_custom,
        "AWS::BedrockAgentCore::BrowserCustom",
    ),
    Lister(
        "bedrock-agentcore-control",
        "ListCodeInterpretersCustom",
        list_bedrockagentcore_code_interpreter_custom,
        "AWS::BedrockAgentCore::CodeInterpreterCustom",
    ),
    Lister(
        "s3vectors",
        "GetVectorBucketPolicy",
        list_s3vectors_vector_bucket_policies,
        "AWS::S3Vectors::VectorBucketPolicy",
    ),
    Lister(
        "cleanroomsml",
        "ListConfiguredModelAlgorithmAssociations",
        list_cleanroomsml_configured_model_algorithm_associations,
        "AWS::CleanRoomsML::ConfiguredModelAlgorithmAssociation",
    ),
)


def custom_listers() -> tuple[Lister, ...]:
    """Every hand-written lister (each pins its service + op, and its CFN type when derivable)."""
    return _LISTERS
