"""Per-lister region skip-set: where each ``(service, op)`` has no endpoint.

GENERATED — do not hand-edit. Regenerate with ``out/listers-coverage/build_lister_skip.py``.
Pure data (no third-party imports) so it ships in the discovery Lambda closure.

``LISTER_REGION_SKIP`` maps a lister's ``(service, op)`` to the frozenset of regions where every
input probe observed ``EndpointConnectionError`` — the service has no endpoint there. Op-level:
ops of one service differ regionally; only probe-measured regions appear.
Source probe(s): probe-results.json. PROVISIONAL (single probe — not two-run validated).
"""

from __future__ import annotations

LISTER_REGION_SKIP: dict[tuple[str, str], frozenset[str]] = {
    ("aiops", "ListInvestigationGroups"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ca-central-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("appfabric", "ListAppBundles"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("appflow", "DescribeConnectorProfiles"): frozenset({"ap-northeast-3", "eu-north-1"}),
    ("appflow", "ListFlows"): frozenset({"ap-northeast-3", "eu-north-1"}),
    ("appflow", "describe_connectors"): frozenset({"ap-northeast-3", "eu-north-1"}),
    ("appintegrations", "ListApplications"): frozenset(
        {"eu-north-1", "eu-west-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("appintegrations", "ListDataIntegrations"): frozenset(
        {"eu-north-1", "eu-west-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("appintegrations", "ListEventIntegrations"): frozenset(
        {"eu-north-1", "eu-west-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("applicationcostprofiler", "ListReportDefinitions"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("apprunner", "ListAutoScalingConfigurations"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "ca-central-1", "eu-north-1", "sa-east-1", "us-west-1"}
    ),
    ("apprunner", "ListConnections"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "ca-central-1", "eu-north-1", "sa-east-1", "us-west-1"}
    ),
    ("apprunner", "ListObservabilityConfigurations"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "ca-central-1", "eu-north-1", "sa-east-1", "us-west-1"}
    ),
    ("apprunner", "ListServices"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "ca-central-1", "eu-north-1", "sa-east-1", "us-west-1"}
    ),
    ("apprunner", "ListVpcConnectors"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "ca-central-1", "eu-north-1", "sa-east-1", "us-west-1"}
    ),
    ("apprunner", "ListVpcIngressConnections"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "ca-central-1", "eu-north-1", "sa-east-1", "us-west-1"}
    ),
    ("appstream", "DescribeAppBlockBuilders"): frozenset({"eu-north-1"}),
    ("appstream", "describe_app_blocks"): frozenset({"eu-north-1"}),
    ("appstream", "describe_applications"): frozenset({"eu-north-1"}),
    ("appstream", "DescribeDirectoryConfigs"): frozenset({"eu-north-1"}),
    ("appstream", "describe_fleets"): frozenset({"eu-north-1"}),
    ("appstream", "DescribeImageBuilders"): frozenset({"eu-north-1"}),
    ("appstream", "DescribeImages"): frozenset({"eu-north-1"}),
    ("appstream", "DescribeStacks"): frozenset({"eu-north-1"}),
    ("auditmanager", "ListAssessmentReports"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1"}
    ),
    ("auditmanager", "ListAssessments"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1"}
    ),
    ("b2bi", "ListCapabilities"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-north-1",
            "eu-west-2",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("b2bi", "ListPartnerships"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-north-1",
            "eu-west-2",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("b2bi", "ListProfiles"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-north-1",
            "eu-west-2",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("b2bi", "ListTransformers"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-north-1",
            "eu-west-2",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("bedrock-agentcore", "ListABTests"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("bedrock-agentcore", "ListBatchEvaluations"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("bedrock-agentcore", "ListRecommendations"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("bedrock-agentcore-control", "list_agent_runtimes"): frozenset(
        {"ap-northeast-3", "us-west-1"}
    ),
    ("bedrock-agentcore-control", "list_api_key_credential_providers"): frozenset(
        {"ap-northeast-3", "us-west-1"}
    ),
    ("bedrock-agentcore-control", "list_browser_profiles"): frozenset(
        {"ap-northeast-3", "us-west-1"}
    ),
    ("bedrock-agentcore-control", "ListBrowsers"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("bedrock-agentcore-control", "ListBrowsersCustom"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("bedrock-agentcore-control", "ListCodeInterpreters"): frozenset(
        {"ap-northeast-3", "us-west-1"}
    ),
    ("bedrock-agentcore-control", "ListCodeInterpretersCustom"): frozenset(
        {"ap-northeast-3", "us-west-1"}
    ),
    ("bedrock-agentcore-control", "list_configuration_bundles"): frozenset(
        {"ap-northeast-3", "us-west-1"}
    ),
    ("bedrock-agentcore-control", "list_datasets"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("bedrock-agentcore-control", "ListEvaluators"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("bedrock-agentcore-control", "ListGatewayTargets"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("bedrock-agentcore-control", "list_gateways"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("bedrock-agentcore-control", "list_harnesses"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("bedrock-agentcore-control", "list_memories"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("bedrock-agentcore-control", "list_oauth2_credential_providers"): frozenset(
        {"ap-northeast-3", "us-west-1"}
    ),
    ("bedrock-agentcore-control", "list_online_evaluation_configs"): frozenset(
        {"ap-northeast-3", "us-west-1"}
    ),
    ("bedrock-agentcore-control", "list_payment_credential_providers"): frozenset(
        {"ap-northeast-3", "us-west-1"}
    ),
    ("bedrock-agentcore-control", "list_payment_managers"): frozenset(
        {"ap-northeast-3", "us-west-1"}
    ),
    ("bedrock-agentcore-control", "list_policy_engines"): frozenset(
        {"ap-northeast-3", "us-west-1"}
    ),
    ("bedrock-agentcore-control", "ListRegistries"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("bedrock-agentcore-control", "list_workload_identities"): frozenset(
        {"ap-northeast-3", "us-west-1"}
    ),
    ("bedrock-data-automation", "list_blueprints"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "us-west-1"}
    ),
    ("bedrock-data-automation", "list_data_automation_projects"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "us-west-1"}
    ),
    ("bedrock-data-automation", "list_data_automation_libraries"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "us-west-1"}
    ),
    ("braket", "search_spending_limits"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-west-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
        }
    ),
    ("chatbot", "DescribeChimeWebhookConfigurations"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-west-1",
        }
    ),
    ("chatbot", "DescribeSlackChannelConfigurations"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-west-1",
        }
    ),
    ("chatbot", "ListCustomActions"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-west-1",
        }
    ),
    ("chatbot", "ListMicrosoftTeamsChannelConfigurations"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-west-1",
        }
    ),
    ("chatbot", "ListMicrosoftTeamsConfiguredTeams"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-west-1",
        }
    ),
    ("chime-sdk-identity", "list_app_instances"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("chime-sdk-media-pipelines", "ListMediaCapturePipelines"): frozenset(
        {
            "ap-northeast-3",
            "eu-north-1",
            "eu-west-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("chime-sdk-media-pipelines", "ListMediaInsightsPipelineConfigurations"): frozenset(
        {
            "ap-northeast-3",
            "eu-north-1",
            "eu-west-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("chime-sdk-media-pipelines", "ListMediaPipelineKinesisVideoStreamPools"): frozenset(
        {
            "ap-northeast-3",
            "eu-north-1",
            "eu-west-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("chime-sdk-media-pipelines", "ListMediaPipelines"): frozenset(
        {
            "ap-northeast-3",
            "eu-north-1",
            "eu-west-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("chime-sdk-voice", "ListPhoneNumberOrders"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("chime-sdk-voice", "ListPhoneNumbers"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("chime-sdk-voice", "ListSipMediaApplications"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("chime-sdk-voice", "ListSipRules"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("chime-sdk-voice", "ListVoiceConnectorGroups"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("chime-sdk-voice", "ListVoiceConnectors"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("chime-sdk-voice", "ListVoiceProfileDomains"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("cleanrooms", "ListCollaborations"): frozenset(
        {"ap-northeast-3", "ap-south-1", "ca-central-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("cleanrooms", "ListConfiguredTables"): frozenset(
        {"ap-northeast-3", "ap-south-1", "ca-central-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("cleanrooms", "ListMemberships"): frozenset(
        {"ap-northeast-3", "ap-south-1", "ca-central-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("cleanroomsml", "ListAudienceExportJobs"): frozenset(
        {"ap-northeast-3", "ap-south-1", "ca-central-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("cleanroomsml", "ListAudienceGenerationJobs"): frozenset(
        {"ap-northeast-3", "ap-south-1", "ca-central-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("cleanroomsml", "ListAudienceModels"): frozenset(
        {"ap-northeast-3", "ap-south-1", "ca-central-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("cleanroomsml", "ListConfiguredAudienceModels"): frozenset(
        {"ap-northeast-3", "ap-south-1", "ca-central-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("cleanroomsml", "ListConfiguredModelAlgorithmAssociations"): frozenset(
        {"ap-northeast-3", "ap-south-1", "ca-central-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("cleanroomsml", "ListConfiguredModelAlgorithms"): frozenset(
        {"ap-northeast-3", "ap-south-1", "ca-central-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("cleanroomsml", "ListTrainingDatasets"): frozenset(
        {"ap-northeast-3", "ap-south-1", "ca-central-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("clouddirectory", "ListDevelopmentSchemaArns"): frozenset({"ap-northeast-3"}),
    ("clouddirectory", "ListDirectories"): frozenset({"ap-northeast-3"}),
    ("clouddirectory", "ListPublishedSchemaArns"): frozenset({"ap-northeast-3"}),
    ("cloudhsm", "ListHapgs"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
        }
    ),
    ("cloudhsm", "ListHsms"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
        }
    ),
    ("cloudhsm", "ListLunaClients"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
        }
    ),
    ("cloudsearch", "DescribeDomains"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
        }
    ),
    ("codeartifact", "ListDomains"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "ca-central-1", "sa-east-1", "us-west-1"}
    ),
    ("codeartifact", "ListRepositories"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "ca-central-1", "sa-east-1", "us-west-1"}
    ),
    ("codeconnections", "ListConnections"): frozenset({"ap-northeast-3"}),
    ("codeconnections", "ListHosts"): frozenset({"ap-northeast-3"}),
    ("codeconnections", "ListRepositoryLinks"): frozenset({"ap-northeast-3"}),
    ("codeguru-reviewer", "ListRepositoryAssociations"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("codeguru-security", "ListScans"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("codeguruprofiler", "ListProfilingGroups"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("codestar-connections", "ListConnections"): frozenset({"ap-northeast-3"}),
    ("codestar-connections", "ListHosts"): frozenset({"ap-northeast-3"}),
    ("codestar-connections", "ListRepositoryLinks"): frozenset({"ap-northeast-3"}),
    ("codestar-connections", "ListSyncConfigurations"): frozenset({"ap-northeast-3"}),
    ("codestar-notifications", "ListNotificationRules"): frozenset({"ap-northeast-3"}),
    ("codestar-notifications", "ListTargets"): frozenset({"ap-northeast-3"}),
    ("cognito-sync", "ListIdentityPoolUsage"): frozenset(
        {"ap-northeast-3", "ca-central-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("comprehend", "ListDocumentClassificationJobs"): frozenset({"ap-northeast-3", "eu-north-1"}),
    ("comprehend", "ListDocumentClassifiers"): frozenset({"ap-northeast-3", "eu-north-1"}),
    ("comprehend", "ListDominantLanguageDetectionJobs"): frozenset(
        {"ap-northeast-3", "eu-north-1"}
    ),
    ("comprehend", "ListEndpoints"): frozenset({"ap-northeast-3", "eu-north-1"}),
    ("comprehend", "ListEntitiesDetectionJobs"): frozenset({"ap-northeast-3", "eu-north-1"}),
    ("comprehend", "ListEntityRecognizers"): frozenset({"ap-northeast-3", "eu-north-1"}),
    ("comprehend", "ListEventsDetectionJobs"): frozenset({"ap-northeast-3", "eu-north-1"}),
    ("comprehend", "ListFlywheels"): frozenset({"ap-northeast-3", "eu-north-1"}),
    ("comprehend", "ListKeyPhrasesDetectionJobs"): frozenset({"ap-northeast-3", "eu-north-1"}),
    ("comprehend", "ListPiiEntitiesDetectionJobs"): frozenset({"ap-northeast-3", "eu-north-1"}),
    ("comprehend", "ListSentimentDetectionJobs"): frozenset({"ap-northeast-3", "eu-north-1"}),
    ("comprehend", "ListTargetedSentimentDetectionJobs"): frozenset(
        {"ap-northeast-3", "eu-north-1"}
    ),
    ("comprehend", "ListTopicsDetectionJobs"): frozenset({"ap-northeast-3", "eu-north-1"}),
    ("comprehendmedical", "ListEntitiesDetectionV2Jobs"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("comprehendmedical", "ListICD10CMInferenceJobs"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("comprehendmedical", "ListPHIDetectionJobs"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("comprehendmedical", "ListRxNormInferenceJobs"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("comprehendmedical", "ListSNOMEDCTInferenceJobs"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("connect", "ListApprovedOrigins"): frozenset(
        {"eu-north-1", "eu-west-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("connect", "ListInstanceStorageConfigs"): frozenset(
        {"eu-north-1", "eu-west-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("connect", "ListInstances"): frozenset(
        {"eu-north-1", "eu-west-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("connect", "ListSecurityKeys"): frozenset(
        {"eu-north-1", "eu-west-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("connect", "ListTrafficDistributionGroups"): frozenset(
        {"eu-north-1", "eu-west-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("connectcampaigns", "ListCampaigns"): frozenset(
        {"eu-north-1", "eu-west-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("connectcampaignsv2", "list_campaigns"): frozenset(
        {"eu-north-1", "eu-west-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("connectcases", "list_domains"): frozenset(
        {
            "ap-northeast-3",
            "eu-north-1",
            "eu-west-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("connecthealth", "ListDomains"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("cur", "DescribeReportDefinitions"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("customer-profiles", "ListDomains"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("databrew", "ListDatasets"): frozenset({"ap-northeast-3"}),
    ("databrew", "ListJobs"): frozenset({"ap-northeast-3"}),
    ("databrew", "ListProjects"): frozenset({"ap-northeast-3"}),
    ("databrew", "ListRecipes"): frozenset({"ap-northeast-3"}),
    ("databrew", "ListRulesets"): frozenset({"ap-northeast-3"}),
    ("databrew", "ListSchedules"): frozenset({"ap-northeast-3"}),
    ("dataexchange", "ListDataGrants"): frozenset(
        {"ap-northeast-3", "ap-south-1", "ca-central-1", "eu-north-1", "eu-west-3", "sa-east-1"}
    ),
    ("dataexchange", "ListDataSets"): frozenset(
        {"ap-northeast-3", "ap-south-1", "ca-central-1", "eu-north-1", "eu-west-3", "sa-east-1"}
    ),
    ("dataexchange", "ListEventActions"): frozenset(
        {"ap-northeast-3", "ap-south-1", "ca-central-1", "eu-north-1", "eu-west-3", "sa-east-1"}
    ),
    ("datapipeline", "ListPipelines"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("datazone", "ListDomains"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("datazone", "ListProjectProfiles"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("dax", "describe_clusters"): frozenset({"ap-northeast-3"}),
    ("dax", "describe_parameter_groups"): frozenset({"ap-northeast-3"}),
    ("dax", "describe_subnet_groups"): frozenset({"ap-northeast-3"}),
    ("deadline", "ListFarms"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("deadline", "ListLicenseEndpoints"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("deadline", "ListMonitors"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("detective", "ListGraphs"): frozenset({"ap-northeast-3"}),
    ("detective", "ListOrganizationAdminAccounts"): frozenset({"ap-northeast-3"}),
    ("detective", "list_invitations"): frozenset({"ap-northeast-3"}),
    ("devicefarm", "ListDeviceInstances"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("devicefarm", "ListInstanceProfiles"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("devicefarm", "ListProjects"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("devicefarm", "ListTestGridProjects"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("devicefarm", "ListVPCEConfigurations"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("devops-agent", "ListAgentSpaces"): frozenset({"ap-northeast-3", "eu-north-1", "us-west-1"}),
    ("devops-agent", "ListPrivateConnections"): frozenset(
        {"ap-northeast-3", "eu-north-1", "us-west-1"}
    ),
    ("devops-agent", "ListServices"): frozenset({"ap-northeast-3", "eu-north-1", "us-west-1"}),
    ("devops-guru", "ListNotificationChannels"): frozenset({"ap-northeast-3"}),
    ("discovery", "DescribeAgents"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("discovery", "DescribeContinuousExports"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("discovery", "DescribeExportConfigurations"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("discovery", "DescribeExportTasks"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("discovery", "DescribeImportTasks"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("docdb-elastic", "ListClusterSnapshots"): frozenset(
        {"ap-northeast-3", "eu-north-1", "us-west-1"}
    ),
    ("docdb-elastic", "ListClusters"): frozenset({"ap-northeast-3", "eu-north-1", "us-west-1"}),
    ("dsql", "ListClusters"): frozenset({"us-west-1"}),
    ("ecr-public", "DescribeRepositories"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("elementalinference", "ListDictionaries"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-southeast-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("elementalinference", "ListFeeds"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-southeast-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("entityresolution", "ListIdMappingWorkflows"): frozenset(
        {"ap-northeast-3", "ap-south-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("entityresolution", "ListIdNamespaces"): frozenset(
        {"ap-northeast-3", "ap-south-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("entityresolution", "ListMatchingWorkflows"): frozenset(
        {"ap-northeast-3", "ap-south-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("entityresolution", "ListSchemaMappings"): frozenset(
        {"ap-northeast-3", "ap-south-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("finspace", "ListEnvironments"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("finspace", "ListKxEnvironments"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("finspace-data", "ListDatasets"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("forecast", "ListDatasetGroups"): frozenset(
        {
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("forecast", "ListDatasetImportJobs"): frozenset(
        {
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("forecast", "ListDatasets"): frozenset(
        {
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("forecast", "ListExplainabilities"): frozenset(
        {
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("forecast", "ListExplainabilityExports"): frozenset(
        {
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("forecast", "ListForecastExportJobs"): frozenset(
        {
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("forecast", "ListForecasts"): frozenset(
        {
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("forecast", "ListMonitors"): frozenset(
        {
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("forecast", "ListPredictorBacktestExportJobs"): frozenset(
        {
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("forecast", "ListPredictors"): frozenset(
        {
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("forecast", "ListWhatIfAnalyses"): frozenset(
        {
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("forecast", "ListWhatIfForecastExports"): frozenset(
        {
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("forecast", "ListWhatIfForecasts"): frozenset(
        {
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("frauddetector", "DescribeModelVersions"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("frauddetector", "GetBatchImportJobs"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("frauddetector", "GetBatchPredictionJobs"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("frauddetector", "GetDetectors"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("frauddetector", "GetEntityTypes"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("frauddetector", "GetEventTypes"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("frauddetector", "GetExternalModels"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("frauddetector", "GetLabels"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("frauddetector", "GetListsMetadata"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("frauddetector", "GetModels"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("frauddetector", "GetOutcomes"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("frauddetector", "GetVariables"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("gamelift", "DescribeFleetAttributes"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3"}
    ),
    ("gamelift", "DescribeGameSessionQueues"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3"}
    ),
    ("gamelift", "DescribeMatchmakingConfigurations"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3"}
    ),
    ("gamelift", "DescribeMatchmakingRuleSets"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3"}
    ),
    ("gamelift", "DescribeVpcPeeringConnections"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3"}
    ),
    ("gamelift", "ListBuilds"): frozenset({"ap-northeast-3", "eu-north-1", "eu-west-3"}),
    ("gamelift", "ListContainerFleets"): frozenset({"ap-northeast-3", "eu-north-1", "eu-west-3"}),
    ("gamelift", "ListContainerGroupDefinitions"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3"}
    ),
    ("gamelift", "ListFleets"): frozenset({"ap-northeast-3", "eu-north-1", "eu-west-3"}),
    ("gamelift", "ListGameServerGroups"): frozenset({"ap-northeast-3", "eu-north-1", "eu-west-3"}),
    ("gamelift", "ListScripts"): frozenset({"ap-northeast-3", "eu-north-1", "eu-west-3"}),
    ("gamelift", "list_aliases"): frozenset({"ap-northeast-3", "eu-north-1", "eu-west-3"}),
    ("gamelift", "list_locations"): frozenset({"ap-northeast-3", "eu-north-1", "eu-west-3"}),
    ("gameliftstreams", "ListApplications"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-west-1",
        }
    ),
    ("gameliftstreams", "ListStreamGroups"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-west-1",
        }
    ),
    ("globalaccelerator", "ListAccelerators"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("globalaccelerator", "ListByoipCidrs"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("globalaccelerator", "ListCrossAccountAttachments"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("globalaccelerator", "ListCustomRoutingAccelerators"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("globalaccelerator", "ListEndpointGroups"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("globalaccelerator", "ListListeners"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("grafana", "ListWorkspaces"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("greengrass", "ListCoreDefinitions"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1"}
    ),
    ("greengrass", "ListLoggerDefinitionVersions"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1"}
    ),
    ("greengrass", "ListSubscriptionDefinitions"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1"}
    ),
    ("greengrass", "list_connector_definitions"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1"}
    ),
    ("greengrass", "list_device_definitions"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1"}
    ),
    ("greengrass", "list_function_definitions"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1"}
    ),
    ("greengrass", "list_groups"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1"}
    ),
    ("greengrass", "list_logger_definitions"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1"}
    ),
    ("greengrass", "list_resource_definitions"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1"}
    ),
    ("greengrassv2", "ListComponents"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1"}
    ),
    ("greengrassv2", "ListCoreDevices"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1"}
    ),
    ("greengrassv2", "ListDeployments"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1"}
    ),
    ("groundstation", "ListConfigs"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-west-2",
            "eu-west-3",
            "us-west-1",
        }
    ),
    ("groundstation", "list_dataflow_endpoint_groups"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-west-2",
            "eu-west-3",
            "us-west-1",
        }
    ),
    ("groundstation", "ListMissionProfiles"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-west-2",
            "eu-west-3",
            "us-west-1",
        }
    ),
    ("healthlake", "ListFHIRDatastores"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-southeast-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("inspector", "ListAssessmentRuns"): frozenset(
        {"ap-northeast-3", "ap-southeast-1", "ca-central-1", "eu-west-3", "sa-east-1"}
    ),
    ("inspector", "list_assessment_targets"): frozenset(
        {"ap-northeast-3", "ap-southeast-1", "ca-central-1", "eu-west-3", "sa-east-1"}
    ),
    ("inspector", "ListAssessmentTemplates"): frozenset(
        {"ap-northeast-3", "ap-southeast-1", "ca-central-1", "eu-west-3", "sa-east-1"}
    ),
    ("inspector", "ListEventSubscriptions"): frozenset(
        {"ap-northeast-3", "ap-southeast-1", "ca-central-1", "eu-west-3", "sa-east-1"}
    ),
    ("interconnect", "ListConnections"): frozenset(
        {"ap-northeast-1", "ap-northeast-3", "ap-south-1"}
    ),
    ("iot", "ListActiveViolations"): frozenset({"ap-northeast-3"}),
    ("iot", "ListAuditSuppressions"): frozenset({"ap-northeast-3"}),
    ("iot", "ListAuthorizers"): frozenset({"ap-northeast-3"}),
    ("iot", "ListBillingGroups"): frozenset({"ap-northeast-3"}),
    ("iot", "ListCACertificates"): frozenset({"ap-northeast-3"}),
    ("iot", "ListCertificateProviders"): frozenset({"ap-northeast-3"}),
    ("iot", "ListCertificates"): frozenset({"ap-northeast-3"}),
    ("iot", "ListCommands"): frozenset({"ap-northeast-3"}),
    ("iot", "ListCustomMetrics"): frozenset({"ap-northeast-3"}),
    ("iot", "ListDimensions"): frozenset({"ap-northeast-3"}),
    ("iot", "ListDomainConfigurations"): frozenset({"ap-northeast-3"}),
    ("iot", "ListFleetMetrics"): frozenset({"ap-northeast-3"}),
    ("iot", "ListIndices"): frozenset({"ap-northeast-3"}),
    ("iot", "ListJobTemplates"): frozenset({"ap-northeast-3"}),
    ("iot", "ListMitigationActions"): frozenset({"ap-northeast-3"}),
    ("iot", "ListOTAUpdates"): frozenset({"ap-northeast-3"}),
    ("iot", "ListPackages"): frozenset({"ap-northeast-3"}),
    ("iot", "ListPolicies"): frozenset({"ap-northeast-3"}),
    ("iot", "ListProvisioningTemplates"): frozenset({"ap-northeast-3"}),
    ("iot", "list_role_aliases"): frozenset({"ap-northeast-3"}),
    ("iot", "ListScheduledAudits"): frozenset({"ap-northeast-3"}),
    ("iot", "ListSecurityProfiles"): frozenset({"ap-northeast-3"}),
    ("iot", "ListStreams"): frozenset({"ap-northeast-3"}),
    ("iot", "ListThingGroups"): frozenset({"ap-northeast-3"}),
    ("iot", "ListThingTypes"): frozenset({"ap-northeast-3"}),
    ("iot", "ListThings"): frozenset({"ap-northeast-3"}),
    ("iot", "ListTopicRuleDestinations"): frozenset({"ap-northeast-3"}),
    ("iot", "ListTopicRules"): frozenset({"ap-northeast-3"}),
    ("iot", "describe_account_audit_configuration"): frozenset({"ap-northeast-3"}),
    ("iot", "describe_encryption_configuration"): frozenset({"ap-northeast-3"}),
    ("iot-data", "ListRetainedMessages"): frozenset({"ap-northeast-3"}),
    ("iot-managed-integrations", "ListAccountAssociations"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iot-managed-integrations", "ListCloudConnectors"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iot-managed-integrations", "ListConnectorDestinations"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iot-managed-integrations", "ListCredentialLockers"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iot-managed-integrations", "ListDestinations"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iot-managed-integrations", "ListEventLogConfigurations"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iot-managed-integrations", "ListManagedThings"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iot-managed-integrations", "ListNotificationConfigurations"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iot-managed-integrations", "ListOtaTaskConfigurations"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iot-managed-integrations", "ListOtaTasks"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iot-managed-integrations", "ListProvisioningProfiles"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iotdeviceadvisor", "list_suite_definitions"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("iotevents", "ListDetectorModels"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("iotevents", "ListInputs"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("iotevents", "list_alarm_models"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("iotfleetwise", "ListCampaigns"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iotfleetwise", "ListDecoderManifests"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iotfleetwise", "ListFleets"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iotfleetwise", "ListModelManifests"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iotfleetwise", "ListSignalCatalogs"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iotfleetwise", "ListStateTemplates"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iotfleetwise", "ListVehicles"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iotsecuretunneling", "ListTunnels"): frozenset({"ap-northeast-3"}),
    ("iotsitewise", "ListAccessPolicies"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("iotsitewise", "ListAssetModels"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("iotsitewise", "ListAssets"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("iotsitewise", "ListComputationModels"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("iotsitewise", "ListGateways"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("iotsitewise", "ListPortals"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("iotsitewise", "ListProjects"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("iotsitewise", "ListTimeSeries"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("iotthingsgraph", "SearchFlowTemplates"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iotthingsgraph", "SearchSystemInstances"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iotthingsgraph", "SearchSystemTemplates"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("iottwinmaker", "ListWorkspaces"): frozenset(
        {
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("iotwireless", "ListDestinations"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("iotwireless", "ListDeviceProfiles"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("iotwireless", "ListFuotaTasks"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("iotwireless", "ListMulticastGroups"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("iotwireless", "ListNetworkAnalyzerConfigurations"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("iotwireless", "ListPartnerAccounts"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("iotwireless", "ListPositionConfigurations"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("iotwireless", "ListServiceProfiles"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("iotwireless", "ListWirelessDeviceImportTasks"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("iotwireless", "ListWirelessDevices"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("iotwireless", "ListWirelessGateways"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("iotwireless", "list_wireless_gateway_task_definitions"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("ivs", "ListAdConfigurations"): frozenset(
        {
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("ivs", "ListChannels"): frozenset(
        {
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("ivs", "ListPlaybackKeyPairs"): frozenset(
        {
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("ivs", "ListPlaybackRestrictionPolicies"): frozenset(
        {
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("ivs", "ListRecordingConfigurations"): frozenset(
        {
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("ivs-realtime", "ListCompositions"): frozenset(
        {
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("ivs-realtime", "list_encoder_configurations"): frozenset(
        {
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("ivs-realtime", "list_ingest_configurations"): frozenset(
        {
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("ivs-realtime", "list_public_keys"): frozenset(
        {
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("ivs-realtime", "list_stages"): frozenset(
        {
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("ivs-realtime", "list_storage_configurations"): frozenset(
        {
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("ivschat", "ListLoggingConfigurations"): frozenset(
        {
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("ivschat", "ListRooms"): frozenset(
        {
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("kendra", "list_indices"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("kendra-ranking", "list_rescore_execution_plans"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("keyspacesstreams", "ListStreams"): frozenset({"ap-northeast-3"}),
    ("kinesisvideo", "ListSignalingChannels"): frozenset(
        {"ap-northeast-3", "eu-north-1", "us-west-1"}
    ),
    ("kinesisvideo", "ListStreams"): frozenset({"ap-northeast-3", "eu-north-1", "us-west-1"}),
    ("lex-models", "GetBots"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("lex-models", "GetIntents"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("lex-models", "GetSlotTypes"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("lexv2-models", "ListBots"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("lexv2-models", "ListTestSets"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("lightsail", "GetActiveNames"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("lightsail", "GetAlarms"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("lightsail", "GetBuckets"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("lightsail", "GetCertificates"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("lightsail", "GetCloudFormationStackRecords"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("lightsail", "GetContactMethods"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("lightsail", "get_container_services"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("lightsail", "GetDiskSnapshots"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("lightsail", "GetDisks"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("lightsail", "GetDistributions"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("lightsail", "GetDomains"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("lightsail", "GetInstanceSnapshots"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("lightsail", "GetInstances"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("lightsail", "GetKeyPairs"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("lightsail", "GetLoadBalancers"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("lightsail", "get_relational_database_snapshots"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("lightsail", "get_relational_databases"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("lightsail", "GetStaticIps"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("location", "ListGeofenceCollections"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "eu-west-3", "us-west-1"}
    ),
    ("location", "ListKeys"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "eu-west-3", "us-west-1"}
    ),
    ("location", "ListMaps"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "eu-west-3", "us-west-1"}
    ),
    ("location", "list_place_indexes"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "eu-west-3", "us-west-1"}
    ),
    ("location", "ListRouteCalculators"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "eu-west-3", "us-west-1"}
    ),
    ("location", "ListTrackerConsumers"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "eu-west-3", "us-west-1"}
    ),
    ("location", "ListTrackers"): frozenset(
        {"ap-northeast-2", "ap-northeast-3", "eu-west-3", "us-west-1"}
    ),
    ("lookoutequipment", "ListDatasets"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("lookoutequipment", "ListInferenceSchedulers"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("lookoutequipment", "ListLabelGroups"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("lookoutequipment", "ListModels"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("lookoutequipment", "ListRetrainingSchedulers"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("machinelearning", "DescribeBatchPredictions"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("machinelearning", "DescribeDataSources"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("machinelearning", "DescribeEvaluations"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("machinelearning", "DescribeMLModels"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("managedblockchain", "ListAccessors"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("managedblockchain", "ListNetworks"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("marketplace-agreement", "SearchAgreements"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("medialive", "ListChannelPlacementGroups"): frozenset({"us-west-1"}),
    ("medialive", "ListChannels"): frozenset({"us-west-1"}),
    ("medialive", "ListCloudWatchAlarmTemplateGroups"): frozenset({"us-west-1"}),
    ("medialive", "ListCloudWatchAlarmTemplates"): frozenset({"us-west-1"}),
    ("medialive", "ListClusters"): frozenset({"us-west-1"}),
    ("medialive", "ListEventBridgeRuleTemplateGroups"): frozenset({"us-west-1"}),
    ("medialive", "ListEventBridgeRuleTemplates"): frozenset({"us-west-1"}),
    ("medialive", "ListInputDevices"): frozenset({"us-west-1"}),
    ("medialive", "ListNetworks"): frozenset({"us-west-1"}),
    ("medialive", "ListReservations"): frozenset({"us-west-1"}),
    ("medialive", "ListSdiSources"): frozenset({"us-west-1"}),
    ("medialive", "ListSignalMaps"): frozenset({"us-west-1"}),
    ("medialive", "list_input_security_groups"): frozenset({"us-west-1"}),
    ("medialive", "list_inputs"): frozenset({"us-west-1"}),
    ("medialive", "list_multiplexes"): frozenset({"us-west-1"}),
    ("mediastore", "list_containers"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("mediatailor", "ListChannels"): frozenset({"us-west-1"}),
    ("mediatailor", "ListPlaybackConfigurations"): frozenset({"us-west-1"}),
    ("mediatailor", "ListSourceLocations"): frozenset({"us-west-1"}),
    ("medical-imaging", "list_datastores"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("memorydb", "DescribeACLs"): frozenset({"ap-northeast-3"}),
    ("memorydb", "DescribeClusters"): frozenset({"ap-northeast-3"}),
    ("memorydb", "DescribeMultiRegionClusters"): frozenset({"ap-northeast-3"}),
    ("memorydb", "DescribeParameterGroups"): frozenset({"ap-northeast-3"}),
    ("memorydb", "DescribeSnapshots"): frozenset({"ap-northeast-3"}),
    ("memorydb", "DescribeSubnetGroups"): frozenset({"ap-northeast-3"}),
    ("memorydb", "DescribeUsers"): frozenset({"ap-northeast-3"}),
    ("mgh", "ListApplicationStates"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("mgh", "ListProgressUpdateStreams"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("migrationhub-config", "DescribeHomeRegionControls"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("migrationhuborchestrator", "ListPlugins"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("migrationhuborchestrator", "ListTemplates"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("migrationhuborchestrator", "ListWorkflows"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("migrationhubstrategy", "ListAnalyzableServers"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("migrationhubstrategy", "ListApplicationComponents"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("migrationhubstrategy", "ListCollectors"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("migrationhubstrategy", "ListImportFileTask"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("migrationhubstrategy", "ListServers"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("mpa", "ListApprovalTeams"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("mpa", "ListIdentitySources"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("mturk", "ListHITs"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("mturk", "ListWorkerBlocks"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("neptunedata", "ListGremlinQueries"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("neptunedata", "ListMLEndpoints"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("neptunedata", "ListMLModelTransformJobs"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("neptunedata", "ListOpenCypherQueries"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("nova-act", "ListWorkflowDefinitions"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("omics", "ListAnnotationStores"): frozenset(
        {"ap-northeast-3", "ap-south-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("omics", "ListConfigurations"): frozenset(
        {"ap-northeast-3", "ap-south-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("omics", "ListReferenceStores"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("omics", "ListRunCaches"): frozenset(
        {"ap-northeast-3", "ap-south-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("omics", "ListRunGroups"): frozenset(
        {"ap-northeast-3", "ap-south-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("omics", "ListSequenceStores"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("omics", "ListVariantStores"): frozenset(
        {"ap-northeast-3", "ap-south-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("omics", "ListWorkflowVersions"): frozenset(
        {"ap-northeast-3", "ap-south-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("omics", "ListWorkflows"): frozenset(
        {"ap-northeast-3", "ap-south-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("panorama", "ListApplicationInstances"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("panorama", "ListDevices"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("panorama", "ListNodes"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("panorama", "ListPackages"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("payment-cryptography", "ListKeys"): frozenset({"ap-northeast-2", "eu-north-1", "us-west-1"}),
    ("payment-cryptography", "list_aliases"): frozenset(
        {"ap-northeast-2", "eu-north-1", "us-west-1"}
    ),
    ("pcs", "ListClusters"): frozenset({"ca-central-1", "us-west-1"}),
    ("personalize", "ListBatchInferenceJobs"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("personalize", "ListBatchSegmentJobs"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("personalize", "ListCampaigns"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("personalize", "ListDataDeletionJobs"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("personalize", "ListDatasetExportJobs"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("personalize", "ListDatasetGroups"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("personalize", "ListDatasetImportJobs"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("personalize", "ListDatasets"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("personalize", "ListEventTrackers"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("personalize", "ListFilters"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("personalize", "ListMetricAttributions"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("personalize", "ListRecommenders"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("personalize", "ListSchemas"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("personalize", "ListSolutions"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("pinpoint", "GetApnsSandboxChannel"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("pinpoint", "GetApnsVoipChannel"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("pinpoint", "GetApnsVoipSandboxChannel"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("pinpoint", "GetApplicationSettings"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("pinpoint", "GetSmsChannel"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("pinpoint", "ListTemplates"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("pinpoint", "get_apps"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("pinpoint-sms-voice", "ListConfigurationSets"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-southeast-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("proton", "ListComponents"): frozenset(
        {"ap-northeast-3", "ap-south-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("proton", "ListEnvironmentTemplates"): frozenset(
        {"ap-northeast-3", "ap-south-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("proton", "ListEnvironments"): frozenset(
        {"ap-northeast-3", "ap-south-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("proton", "ListRepositories"): frozenset(
        {"ap-northeast-3", "ap-south-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("proton", "ListServiceTemplates"): frozenset(
        {"ap-northeast-3", "ap-south-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("proton", "ListServices"): frozenset(
        {"ap-northeast-3", "ap-south-1", "eu-north-1", "eu-west-3", "sa-east-1", "us-west-1"}
    ),
    ("qbusiness", "ListDataSources"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("qbusiness", "list_applications"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("qconnect", "ListAIAgentVersions"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("qconnect", "ListAIAgents"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("qconnect", "ListAssistants"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("qconnect", "ListKnowledgeBases"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("rekognition", "DescribeProjects"): frozenset({"ap-northeast-3", "eu-north-1", "eu-west-3"}),
    ("rekognition", "ListCollections"): frozenset({"ap-northeast-3", "eu-north-1", "eu-west-3"}),
    ("rekognition", "ListMediaAnalysisJobs"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3"}
    ),
    ("rekognition", "ListStreamProcessors"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3"}
    ),
    ("repostspace", "ListSpaces"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("resiliencehub", "ListApps"): frozenset({"ap-northeast-3"}),
    ("resiliencehub", "ListRecommendationTemplates"): frozenset({"ap-northeast-3"}),
    ("resiliencehub", "ListResiliencyPolicies"): frozenset({"ap-northeast-3"}),
    ("resiliencehubv2", "ListPolicies"): frozenset({"ap-northeast-3"}),
    ("resiliencehubv2", "ListServices"): frozenset({"ap-northeast-3"}),
    ("resiliencehubv2", "ListSystems"): frozenset({"ap-northeast-3"}),
    ("route53-recovery-readiness", "ListCells"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("route53-recovery-readiness", "ListReadinessChecks"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("route53-recovery-readiness", "ListRecoveryGroups"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("route53-recovery-readiness", "ListResourceSets"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("route53domains", "ListDomains"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("route53globalresolver", "ListAccessSources"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("route53globalresolver", "ListFirewallDomainLists"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("route53globalresolver", "ListGlobalResolvers"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("rtbfabric", "ListRequesterGateways"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("rtbfabric", "ListResponderGateways"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("s3", "ListDirectoryBuckets"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-southeast-2",
            "ca-central-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-west-1",
        }
    ),
    ("sagemaker-geospatial", "ListVectorEnrichmentJobs"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("sdb", "list_domains"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
        }
    ),
    ("security-ir", "ListCases"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("security-ir", "ListMemberships"): frozenset({"ap-northeast-3", "us-west-1"}),
    ("securityagent", "ListAgentSpaces"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("securityagent", "ListApplications"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("securityagent", "ListIntegrations"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("securityagent", "ListTargetDomains"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("securityagent", "list_security_requirement_packs"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("serverlessrepo", "ListApplications"): frozenset({"ap-northeast-3"}),
    ("signer", "ListSigningProfiles"): frozenset({"ap-northeast-3"}),
    ("simpledbv2", "ListExports"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "us-east-2",
        }
    ),
    ("sms-voice", "ListConfigurationSets"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-southeast-1",
            "ca-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("snow-device-management", "ListDevices"): frozenset({"ap-northeast-3"}),
    ("ssm-contacts", "ListContacts"): frozenset({"ap-northeast-3"}),
    ("ssm-contacts", "ListRotations"): frozenset({"ap-northeast-3"}),
    ("ssm-incidents", "ListIncidentRecords"): frozenset({"ap-northeast-3"}),
    ("ssm-incidents", "ListReplicationSets"): frozenset({"ap-northeast-3"}),
    ("ssm-incidents", "ListResponsePlans"): frozenset({"ap-northeast-3"}),
    ("support-app", "ListSlackChannelConfigurations"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("support-app", "ListSlackWorkspaceConfigurations"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("support-app", "get_account_alias"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("textract", "ListAdapters"): frozenset(
        {"ap-northeast-1", "ap-northeast-3", "eu-north-1", "sa-east-1"}
    ),
    ("timestream-influxdb", "list_db_clusters"): frozenset({"us-west-1"}),
    ("timestream-influxdb", "list_db_instances"): frozenset({"us-west-1"}),
    ("transcribe", "ListCallAnalyticsCategories"): frozenset({"ap-northeast-3"}),
    ("transcribe", "ListLanguageModels"): frozenset({"ap-northeast-3"}),
    ("transcribe", "ListMedicalVocabularies"): frozenset({"ap-northeast-3"}),
    ("transcribe", "ListVocabularies"): frozenset({"ap-northeast-3"}),
    ("transcribe", "ListVocabularyFilters"): frozenset({"ap-northeast-3"}),
    ("translate", "ListParallelData"): frozenset({"ap-northeast-3", "sa-east-1"}),
    ("translate", "ListTerminologies"): frozenset({"ap-northeast-3", "sa-east-1"}),
    ("translate", "ListTextTranslationJobs"): frozenset({"ap-northeast-3", "sa-east-1"}),
    ("voice-id", "ListDomains"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "eu-north-1",
            "eu-west-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("wellarchitected", "ListLenses"): frozenset({"ap-northeast-3"}),
    ("wellarchitected", "ListProfiles"): frozenset({"ap-northeast-3"}),
    ("wellarchitected", "ListReviewTemplates"): frozenset({"ap-northeast-3"}),
    ("wellarchitected", "ListWorkloads"): frozenset({"ap-northeast-3"}),
    ("wickr", "ListNetworks"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "eu-west-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
        }
    ),
    ("wisdom", "ListAssistants"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("wisdom", "ListKnowledgeBases"): frozenset(
        {
            "ap-northeast-3",
            "ap-south-1",
            "eu-north-1",
            "eu-west-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("workdocs", "DescribeUsers"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("workmail", "ListOrganizations"): frozenset(
        {
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ca-central-1",
            "eu-central-1",
            "eu-north-1",
            "eu-west-2",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("workspaces", "DescribeApplications"): frozenset(
        {"ap-northeast-3", "eu-north-1", "us-west-1"}
    ),
    ("workspaces", "describe_connection_aliases"): frozenset(
        {"ap-northeast-3", "eu-north-1", "us-west-1"}
    ),
    ("workspaces", "DescribeIpGroups"): frozenset({"ap-northeast-3", "eu-north-1", "us-west-1"}),
    ("workspaces", "DescribeWorkspaceBundles"): frozenset(
        {"ap-northeast-3", "eu-north-1", "us-west-1"}
    ),
    ("workspaces", "DescribeWorkspaceDirectories"): frozenset(
        {"ap-northeast-3", "eu-north-1", "us-west-1"}
    ),
    ("workspaces", "DescribeWorkspaceImages"): frozenset(
        {"ap-northeast-3", "eu-north-1", "us-west-1"}
    ),
    ("workspaces", "DescribeWorkspaces"): frozenset({"ap-northeast-3", "eu-north-1", "us-west-1"}),
    ("workspaces", "DescribeWorkspacesPools"): frozenset(
        {"ap-northeast-3", "eu-north-1", "us-west-1"}
    ),
    ("workspaces", "ListAccountLinks"): frozenset({"ap-northeast-3", "eu-north-1", "us-west-1"}),
    ("workspaces-instances", "ListWorkspaceInstances"): frozenset({"eu-north-1", "us-west-1"}),
    ("workspaces-thin-client", "ListDevices"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("workspaces-thin-client", "ListEnvironments"): frozenset(
        {
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-southeast-1",
            "ap-southeast-2",
            "eu-north-1",
            "eu-west-3",
            "sa-east-1",
            "us-east-2",
            "us-west-1",
        }
    ),
    ("workspaces-web", "ListBrowserSettings"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("workspaces-web", "ListDataProtectionSettings"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("workspaces-web", "ListIpAccessSettings"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("workspaces-web", "ListNetworkSettings"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("workspaces-web", "ListPortals"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("workspaces-web", "ListSessionLoggers"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("workspaces-web", "ListTrustStores"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("workspaces-web", "ListUserAccessLoggingSettings"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
    ("workspaces-web", "ListUserSettings"): frozenset(
        {"ap-northeast-3", "eu-north-1", "eu-west-3", "sa-east-1", "us-east-2", "us-west-1"}
    ),
}
