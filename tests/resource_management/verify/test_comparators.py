from aws_bench.resource_management.verify.comparators import (
    count_resources,
    drifts_match,
    filter_aws_managed_resources,
    find_new_resources,
    normalize_drift,
)


def test_filter_aws_managed_resources_removes_service_roles():
    """Test filtering removes AWS service roles."""
    resources = {
        "AWS::IAM::Role": [
            {"Identifier": "AWSServiceRoleForECS"},
            {"Identifier": "MyCustomRole"},
            {"Identifier": "AWSServiceRoleForRDS"},
        ]
    }

    filtered = filter_aws_managed_resources(resources)

    assert "AWS::IAM::Role" in filtered
    assert len(filtered["AWS::IAM::Role"]) == 1
    assert filtered["AWS::IAM::Role"][0]["Identifier"] == "MyCustomRole"


def test_filter_aws_managed_resources_removes_aws_managed_ram_permissions():
    """AWS-managed RAM permissions (arn:aws:ram::aws:permission/*) are filtered out.

    These are global, account-independent, and undeletable; reset flagged them as
    'new resources' and then failed forever trying to delete them. Customer-managed
    permissions (which carry an account id) must be preserved.
    """
    resources = {
        "AWS::RAM::Permission": [
            {"Identifier": "arn:aws:ram::aws:permission/AWSRAMDefaultPermissionDNSView"},
            {"Identifier": "arn:aws:ram::aws:permission/AWSRAMPermissionDNSViewFullAccess"},
            {"Identifier": "arn:aws:ram::123456789012:permission/my-custom-permission"},
        ]
    }

    filtered = filter_aws_managed_resources(resources)

    assert "AWS::RAM::Permission" in filtered
    assert len(filtered["AWS::RAM::Permission"]) == 1
    assert (
        filtered["AWS::RAM::Permission"][0]["Identifier"]
        == "arn:aws:ram::123456789012:permission/my-custom-permission"
    )


def test_filter_aws_managed_resources_removes_aws_reserved_kms_aliases():
    """AWS-reserved KMS aliases (alias/aws/*) are filtered out; customer aliases are kept."""
    resources = {
        "AWS::KMS::Alias": [
            {"Identifier": "arn:aws:kms:ap-northeast-1:123456789012:alias/aws/acm"},
            {"Identifier": "alias/aws/ebs"},
            {"Identifier": "arn:aws:kms:us-east-1:123456789012:alias/my-app-key"},
            {"Identifier": "alias/my-aws-key"},
        ]
    }

    filtered = filter_aws_managed_resources(resources)

    assert "AWS::KMS::Alias" in filtered
    kept = {r["Identifier"] for r in filtered["AWS::KMS::Alias"]}
    assert kept == {
        "arn:aws:kms:us-east-1:123456789012:alias/my-app-key",
        "alias/my-aws-key",
    }


def test_filter_aws_managed_resources_removes_default_rds_neptune_param_option_groups():
    """AWS-reserved default RDS/Neptune parameter & option groups are filtered out.

    These per-engine groups are account-created and undeletable ("Default DBParameterGroup
    cannot be deleted" / "Default option groups cannot be deleted"), so the cleanup sweep
    flagged them as orphans and failed forever. Param groups are named ``default.<engine>``,
    option groups ``default:<engine>``. Task/agent-created groups (custom names) are kept.
    """
    resources = {
        "AWS::RDS::DBParameterGroup": [
            {"Identifier": "default.mysql8.0"},
            {"Identifier": "my-custom-pg"},
        ],
        "AWS::RDS::OptionGroup": [
            {"Identifier": "default:mysql-8-0"},
            {"Identifier": "my-custom-og"},
        ],
        "AWS::Neptune::DBParameterGroup": [
            {"Identifier": "default.neptune1.3"},
            {"Identifier": "my-neptune-pg"},
        ],
    }

    filtered = filter_aws_managed_resources(resources)

    assert {r["Identifier"] for r in filtered["AWS::RDS::DBParameterGroup"]} == {"my-custom-pg"}
    assert {r["Identifier"] for r in filtered["AWS::RDS::OptionGroup"]} == {"my-custom-og"}
    assert {r["Identifier"] for r in filtered["AWS::Neptune::DBParameterGroup"]} == {
        "my-neptune-pg"
    }


def test_filter_aws_managed_resources_removes_default_iot_domain_configurations():
    """AWS-reserved default IoT domain configurations are filtered out.

    Every account has the default managed data-plane (``iot:Data-ATS``) and credential-provider
    (``iot:CredentialProvider``) endpoints. Their names are reserved — AWS forbids user/CDK-created
    names starting with ``iot:`` — so the cleanup sweep flagged these ambient account defaults as
    orphans and failed. Custom domains (which cannot use the ``iot:`` prefix) are kept.
    """
    resources = {
        "AWS::IoT::DomainConfiguration": [
            {"Identifier": "iot:Data-ATS"},
            {"Identifier": "iot:CredentialProvider"},
            {"Identifier": "my-custom-domain"},
        ],
    }

    filtered = filter_aws_managed_resources(resources)

    assert {r["Identifier"] for r in filtered["AWS::IoT::DomainConfiguration"]} == {
        "my-custom-domain"
    }


def test_filter_aws_managed_resources_removes_service_managed_secrets():
    """Service-managed secrets ("<service>!") are filtered; customer secrets kept.

    A Redshift namespace's admin secret (``redshift!…``) is created and rotated
    by Redshift and can surface as a "new" resource when a task connects to a
    baseline namespace (e.g. a Bedrock KB → Redshift link). It is managed by its
    parent and undeletable directly, so it must not be treated as account drift.
    """
    resources = {
        "AWS::SecretsManager::Secret": [
            {
                "Identifier": (
                    "arn:aws:secretsmanager:us-east-1:123456789012:secret:"
                    "redshift!123456789012-test-ns-admin-mjlyMw"
                )
            },
            {"Identifier": "redshift!123456789012-test-ns-admin-mjlyMw"},
            {"Identifier": "rds!cluster-abc123"},
            {
                "Identifier": (
                    "arn:aws:secretsmanager:us-east-1:123456789012:secret:my-app-secret-AbCdEf"
                )
            },
            {"Identifier": "my-custom-secret"},
            # "!" mid-name is NOT the <service>! prefix -> a customer secret, must be KEPT.
            {"Identifier": "my!secret"},
            {"Identifier": "prod!db-creds"},
        ]
    }

    filtered = filter_aws_managed_resources(resources)

    kept = {r["Identifier"] for r in filtered.get("AWS::SecretsManager::Secret", [])}
    assert kept == {
        "arn:aws:secretsmanager:us-east-1:123456789012:secret:my-app-secret-AbCdEf",
        "my-custom-secret",
        "my!secret",
        "prod!db-creds",
    }


def test_filter_aws_managed_resources_removes_custom_types():
    """Test filtering removes custom CloudFormation types."""
    resources = {
        "Custom::MyResource": [{"Identifier": "resource-1"}],
        "AWS::Lambda::Function": [{"Identifier": "my-function"}],
        "InvalidType": [{"Identifier": "bad"}],
    }

    filtered = filter_aws_managed_resources(resources)

    assert "Custom::MyResource" not in filtered
    assert "InvalidType" not in filtered
    assert "AWS::Lambda::Function" in filtered
    assert len(filtered["AWS::Lambda::Function"]) == 1


def test_find_new_resources_detects_new_resources():
    """Test finding new resources added after baseline."""
    current = {
        "AWS::S3::Bucket": [
            {"Identifier": "bucket-1"},
            {"Identifier": "bucket-2"},
        ],
        "AWS::IAM::Role": [{"Identifier": "MyRole"}],
    }

    snapshot = {
        "AWS::S3::Bucket": [{"Identifier": "bucket-1"}],
    }

    new = find_new_resources(current, snapshot)

    assert "AWS::S3::Bucket" in new
    assert len(new["AWS::S3::Bucket"]) == 1
    assert new["AWS::S3::Bucket"][0]["Identifier"] == "bucket-2"
    assert "AWS::IAM::Role" in new


def test_find_new_resources_current_minus_snapshot_for_cleanup():
    """Pin the current-minus-snapshot semantics the cleanup sweep relies on.

    Cleanup diffs the live account (current) against a baseline snapshot and sweeps
    exactly the resources whose (type, Identifier) are in current but not the baseline.
    A default VPC present in BOTH is a baseline resource and must be kept; a scenario
    resource present only in current must be swept.
    """
    current = {
        "AWS::EC2::VPC": [
            {"Identifier": "vpc-default"},
            {"Identifier": "vpc-scenario"},
        ],
    }
    snapshot = {
        "AWS::EC2::VPC": [{"Identifier": "vpc-default"}],
    }

    new = find_new_resources(current, snapshot)

    # The default VPC (in both) is excluded; only the scenario VPC (current-only) remains.
    assert new == {"AWS::EC2::VPC": [{"Identifier": "vpc-scenario"}]}


def test_find_new_resources_empty_when_no_changes():
    """Test finding new resources returns empty when state matches."""
    current = {
        "AWS::S3::Bucket": [{"Identifier": "bucket-1"}],
    }

    snapshot = {
        "AWS::S3::Bucket": [{"Identifier": "bucket-1"}],
    }

    new = find_new_resources(current, snapshot)

    assert len(new) == 0


def test_count_resources():
    """Test counting total resources across types."""
    resources = {
        "AWS::S3::Bucket": [{"Identifier": "b1"}, {"Identifier": "b2"}],
        "AWS::IAM::Role": [{"Identifier": "r1"}],
    }

    count = count_resources(resources)

    assert count == 3


def test_normalize_drift():
    """Test drift normalization for comparison."""
    drift = [
        {
            "LogicalResourceId": "MySecurityGroup",
            "StackResourceDriftStatus": "MODIFIED",
            "PropertyDifferences": [{"PropertyPath": "/IpPermissions", "Expected": "[]"}],
        },
        {
            "LogicalResourceId": "MyRole",
            "StackResourceDriftStatus": "IN_SYNC",
            "PropertyDifferences": [],
        },
    ]

    normalized = normalize_drift(drift)

    assert len(normalized) == 2
    assert normalized[0]["LogicalResourceId"] == "MyRole"  # Sorted
    assert normalized[1]["LogicalResourceId"] == "MySecurityGroup"


def test_drifts_match_identical():
    """Test drift matching for identical drifts."""
    drift1 = [
        {
            "LogicalResourceId": "MyRole",
            "StackResourceDriftStatus": "IN_SYNC",
            "PropertyDifferences": [],
        }
    ]

    drift2 = [
        {
            "LogicalResourceId": "MyRole",
            "StackResourceDriftStatus": "IN_SYNC",
            "PropertyDifferences": [],
        }
    ]

    assert drifts_match(drift1, drift2)


def test_drifts_match_different():
    """Test drift matching for different drifts."""
    drift1 = [
        {
            "LogicalResourceId": "MyRole",
            "StackResourceDriftStatus": "IN_SYNC",
            "PropertyDifferences": [],
        }
    ]

    drift2 = [
        {
            "LogicalResourceId": "MyRole",
            "StackResourceDriftStatus": "MODIFIED",
            "PropertyDifferences": [{"PropertyPath": "/AssumeRolePolicyDocument"}],
        }
    ]

    assert not drifts_match(drift1, drift2)


def test_drifts_match_different_order():
    """Test drift matching ignores ordering."""
    drift1 = [
        {"LogicalResourceId": "Resource1", "StackResourceDriftStatus": "IN_SYNC"},
        {"LogicalResourceId": "Resource2", "StackResourceDriftStatus": "MODIFIED"},
    ]

    drift2 = [
        {"LogicalResourceId": "Resource2", "StackResourceDriftStatus": "MODIFIED"},
        {"LogicalResourceId": "Resource1", "StackResourceDriftStatus": "IN_SYNC"},
    ]

    # After normalization, should match
    assert drifts_match(drift1, drift2)


class TestServiceLinkedRoleAndNetworkInterfacePermissionFiltering:
    """Tests that ServiceLinkedRole and NetworkInterfacePermission are filtered from drift."""

    def test_service_linked_role_filtered_as_aws_managed(self):
        from aws_bench.resource_management.verify.comparators import AWS_MANAGED_FILTERS

        predicate = AWS_MANAGED_FILTERS.get("AWS::IAM::ServiceLinkedRole")
        assert predicate is not None
        assert predicate(
            "arn:aws:iam::123456789012:role/aws-service-role/eks.amazonaws.com/AWSServiceRoleForAmazonEKS",
            {},
        )

    def test_network_interface_permission_filtered_as_aws_managed(self):
        from aws_bench.resource_management.verify.comparators import AWS_MANAGED_FILTERS

        predicate = AWS_MANAGED_FILTERS.get("AWS::EC2::NetworkInterfacePermission")
        assert predicate is not None
        assert predicate("eni-perm-abc123", {})


class TestSageMakerHubFiltering:
    """The SageMaker::Hub filter must match only the AWS-managed public hub."""

    def _predicate(self):
        from aws_bench.resource_management.verify.comparators import AWS_MANAGED_FILTERS

        predicate = AWS_MANAGED_FILTERS.get("AWS::SageMaker::Hub")
        assert predicate is not None
        return predicate

    def test_matches_managed_hub_by_arn_suffix(self):
        predicate = self._predicate()
        assert predicate("arn:aws:sagemaker:us-east-1:123456789012:hub/SageMakerPublicHub", {})

    def test_matches_bare_managed_hub_name(self):
        predicate = self._predicate()
        assert predicate("SageMakerPublicHub", {})

    def test_does_not_match_task_hub_containing_the_substring(self):
        """A task-created hub whose name merely contains the string is NOT excluded."""
        predicate = self._predicate()
        assert not predicate(
            "arn:aws:sagemaker:us-east-1:123456789012:hub/SageMakerPublicHubClone", {}
        )
        assert not predicate("SageMakerPublicHubClone", {})
        assert not predicate("MySageMakerPublicHub", {})


class TestPhase2AwsOwnedFilters:
    """Phase-2 filters for AWS-published defaults: match AWS-owned ones, keep customer ones."""

    def _predicate(self, cfn_type):
        from aws_bench.resource_management.verify.comparators import AWS_MANAGED_FILTERS

        predicate = AWS_MANAGED_FILTERS.get(cfn_type)
        assert predicate is not None
        return predicate

    def test_apprunner_default_autoscaling_config_matches_reserved_only(self):
        predicate = self._predicate("AWS::AppRunner::AutoScalingConfiguration")
        # AWS-reserved default (revision ARN or bare name) is filtered — AWS refuses to delete it.
        assert predicate(
            "arn:aws:apprunner:us-east-1:1:autoscalingconfiguration/DefaultConfiguration/1/000",
            {},
        )
        assert predicate("DefaultConfiguration", {})
        # A customer-created config carries a different name and is NOT filtered.
        assert not predicate(
            "arn:aws:apprunner:us-east-1:1:autoscalingconfiguration/my-config/1/abc", {}
        )

    def test_wellarchitected_lens_matches_aws_published_only(self):
        predicate = self._predicate("AWS::WellArchitected::Lens")
        assert predicate("arn:aws:wellarchitected::aws:lens/wellarchitected", {})
        # a customer/custom lens carries an account id and must NOT be filtered
        assert not predicate(
            "arn:aws:wellarchitected:us-east-1:123456789012:lens/my-custom-lens", {}
        )

    def test_bedrock_agentcore_builtin_evaluator_matches_builtins_only(self):
        predicate = self._predicate("AWS::BedrockAgentCore::Evaluator")
        assert predicate("arn:aws:bedrock-agentcore:::evaluator/Builtin.Correctness", {})
        assert predicate("Builtin.Faithfulness", {})
        # a customer-created evaluator has an account id and is NOT filtered
        assert not predicate(
            "arn:aws:bedrock-agentcore:us-east-1:123456789012:evaluator/my-eval", {}
        )

    def test_rds_ca_certificate_matches_aws_certs_only(self):
        predicate = self._predicate("AWS::RDS::Certificate")
        assert predicate("arn:aws:rds:us-east-1::cert:rds-ca-ecc384-g1", {})
        assert predicate("rds-ca-rsa2048-g1", {})
        # a customer-imported cert (hypothetical) without the rds-ca- prefix is NOT filtered
        assert not predicate("arn:aws:rds:us-east-1:123456789012:cert:my-imported-cert", {})

    def test_rds_automated_snapshot_filtered_manual_kept(self):
        predicate = self._predicate("AWS::RDS::DBSnapshot")
        # Automated (system) snapshots carry the rds: owner prefix — filtered (they regenerate daily
        # and cannot be deleted manually).
        assert predicate("arn:aws:rds:us-east-1:1:snapshot:rds:db-2026-07-04-06-07", {})
        assert predicate("rds:db-2026-07-04-06-07", {})
        # A manual snapshot has a plain name — NOT filtered (it is real, deletable drift).
        assert not predicate("arn:aws:rds:us-east-1:1:snapshot:my-manual-snap", {})

    def test_rds_automated_cluster_snapshot_filtered_manual_kept(self):
        predicate = self._predicate("AWS::RDS::DBClusterSnapshot")
        assert predicate("arn:aws:rds:us-east-1:1:cluster-snapshot:rds:cl-2026-07-04-05-40", {})
        assert not predicate("arn:aws:rds:us-east-1:1:cluster-snapshot:my-manual-cl", {})

    def test_default_event_bus_filtered_custom_kept(self):
        predicate = self._predicate("AWS::Events::EventBus")
        # The account's default event bus is AWS-created and undeletable — filtered.
        assert predicate("arn:aws:events:us-east-1:123456789012:event-bus/default", {})
        # A task-created custom bus carries its own name and is NOT filtered.
        assert not predicate("arn:aws:events:us-east-1:123456789012:event-bus/my-bus", {})

    def test_default_mediaconvert_queue_filtered_custom_kept(self):
        predicate = self._predicate("AWS::MediaConvert::Queue")
        # The account's default MediaConvert queue is service-created (lazily) and undeletable —
        # filtered (the lister emits the ARN …:queues/Default).
        assert predicate("arn:aws:mediaconvert:us-east-1:123456789012:queues/Default", {})
        # A task-created on-demand queue carries its own name and is NOT filtered.
        assert not predicate("arn:aws:mediaconvert:us-east-1:123456789012:queues/my-queue", {})
