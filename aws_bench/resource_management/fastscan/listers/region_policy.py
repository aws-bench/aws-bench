"""Hand-curated per-lister regions where a lister's results can't be obtained.

Same ``(service, op)`` shape and handling as the generated ``region_skip.LISTER_REGION_SKIP``,
but a different reason: region_skip records where a service has *no endpoint* (topology); this
records where the endpoint is reachable but the scan gets no successful response across retries.
Listed here, the lister is recorded ``empty`` (not ``failed``, which would abort the region via
the fail-loud gate) and still scanned in every other region. Op-level, so a working sibling op
is never dropped. Pure data (ships in the Lambda closure).

Each entry requires proof: a lone isolated call returns no success across retries in that region
while succeeding elsewhere (rules out account-wide throttling and concurrency). A resource type
the AWS regional-availability catalog confirms is *not offered* in a region is also proof: its
endpoint resolves but the API rejects the call (AccessDenied "not authorized ... in this region"
/ "not authorized to invoke this API operation", NotAuthorizedException, InvalidParameterValue,
or an empty-message AccessDenied), and a type absent from a region cannot hold an orphan there —
so recording it ``empty`` preserves the fail-closed guarantee.
"""

from __future__ import annotations

# Greengrass V1 + V2 list calls returned no success in us-west-1 (isolated call, no success across
# retries there, <1s elsewhere, s3 control unaffected). Greengrass V1 is also EOL (2026-06-01).
UNAVAILABLE_LISTER_REGIONS: dict[tuple[str, str], frozenset[str]] = {
    ("greengrass", "ListCoreDefinitions"): frozenset({"us-west-1"}),
    ("greengrass", "ListLoggerDefinitionVersions"): frozenset({"us-west-1"}),
    ("greengrass", "ListSubscriptionDefinitions"): frozenset({"us-west-1"}),
    ("greengrass", "list_connector_definitions"): frozenset({"us-west-1"}),
    ("greengrass", "list_device_definitions"): frozenset({"us-west-1"}),
    ("greengrass", "list_function_definitions"): frozenset({"us-west-1"}),
    ("greengrass", "list_groups"): frozenset({"us-west-1"}),
    ("greengrass", "list_logger_definitions"): frozenset({"us-west-1"}),
    ("greengrass", "list_resource_definitions"): frozenset({"us-west-1"}),
    ("greengrassv2", "ListComponents"): frozenset({"us-west-1"}),
    ("greengrassv2", "ListCoreDevices"): frozenset({"us-west-1"}),
    ("greengrassv2", "ListDeployments"): frozenset({"us-west-1"}),
    ("iotwireless", "ListPartnerAccounts"): frozenset(
        {"ap-northeast-1", "ap-southeast-2", "eu-central-1", "eu-west-1", "us-west-2"}
    ),
    ("bedrock", "ListAutomatedReasoningPolicies"): frozenset({"ap-southeast-1", "us-west-1"}),
    ("bedrock-agent", "list_flows"): frozenset({"us-west-1"}),
    ("bedrock-agent", "list_prompts"): frozenset({"us-west-1"}),
    ("bedrock-data-automation", "list_blueprints"): frozenset({"ap-southeast-1"}),
    ("bedrock-data-automation", "list_data_automation_libraries"): frozenset(
        {"ap-southeast-1", "us-east-2"}
    ),
    ("bedrock-data-automation", "list_data_automation_projects"): frozenset({"ap-southeast-1"}),
    ("comprehend", "ListDocumentClassificationJobs"): frozenset({"us-west-1"}),
    ("comprehend", "ListDocumentClassifiers"): frozenset({"us-west-1"}),
    ("comprehend", "ListEntitiesDetectionJobs"): frozenset({"us-west-1"}),
    ("comprehend", "ListSentimentDetectionJobs"): frozenset({"us-west-1"}),
    ("connect", "ListTrafficDistributionGroups"): frozenset({"ap-southeast-1"}),
    ("connectcampaigns", "ListCampaigns"): frozenset({"ap-southeast-1"}),
    ("dax", "describe_parameter_groups"): frozenset({"ap-northeast-2"}),
    ("finspace", "ListEnvironments"): frozenset({"ap-southeast-1"}),
    ("glue", "DescribeIntegrations"): frozenset({"us-west-1"}),
    ("glue", "list_integration_resource_properties"): frozenset({"us-west-1"}),
    ("inspector2", "ListCodeSecurityIntegrations"): frozenset({"us-west-1"}),
    ("inspector2", "ListCodeSecurityScanConfigurations"): frozenset({"us-west-1"}),
    ("omics", "ListAnnotationStores"): frozenset({"us-east-2"}),
    ("omics", "ListVariantStores"): frozenset({"us-east-2"}),
    ("rekognition", "DescribeProjects"): frozenset({"us-west-1"}),
    ("rekognition", "ListStreamProcessors"): frozenset({"ap-southeast-1", "us-west-1"}),
}
