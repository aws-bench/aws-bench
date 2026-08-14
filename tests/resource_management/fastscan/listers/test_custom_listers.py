"""Behavior tests for the hand-written code listers (mocked boto3 clients).

Each lister is exercised against a fake session/client to assert its real behavior: the
filtering it applies (AWS-managed exclusions, default-VPC skipping, status filtering), the
id/ARN field it emits, and its manual-pagination loops. No AWS is touched.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from aws_bench.resource_management.fastscan.listers import custom_listers as cl


class _Paginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **_kw):
        return iter(self._pages)


class _FakeClient:
    """A boto3-client stand-in: paginated ops return configured pages, direct ops return values.

    ``paginated`` maps a snake_case op name to its list of response pages; ``direct`` maps an
    op name to the single response dict (or a callable receiving kwargs). Any other attribute
    access returns a MagicMock so incidental calls don't blow up.
    """

    def __init__(self, *, paginated=None, direct=None):
        self._paginated = paginated or {}
        self._direct = direct or {}

    def get_paginator(self, op):
        return _Paginator(self._paginated[op])

    def __getattr__(self, name):
        if name in self._direct:
            value = self._direct[name]
            return value if callable(value) else (lambda **_kw: value)
        return MagicMock()


def _session(client_by_service: dict[str, _FakeClient]) -> MagicMock:
    """A session whose ``.client(service, ...)`` returns the configured fake per service."""
    session = MagicMock()
    session.client.side_effect = lambda service, **_kw: client_by_service[service]
    return session


def _one_service(service: str, client: _FakeClient) -> MagicMock:
    return _session({service: client})


# -- filtering listers: the AWS-managed / default exclusions each lister enforces --


def test_bedrock_inference_profiles_excludes_system_defined():
    client = _FakeClient(
        paginated={
            "list_inference_profiles": [
                {
                    "inferenceProfileSummaries": [
                        {"inferenceProfileArn": "arn:custom", "type": "APPLICATION"},
                        {"inferenceProfileArn": "arn:aws-default", "type": "SYSTEM_DEFINED"},
                    ]
                }
            ]
        }
    )
    assert cl.list_bedrock_inference_profiles(_one_service("bedrock", client)) == ["arn:custom"]


def test_codedeploy_on_premises_instances_filters_to_registered():
    # registrationStatus is a server-side REQUEST parameter (not a response field), so the lister
    # must pass registrationStatus="Registered" to the paginator — otherwise deregistered (stale)
    # instances are enumerated as if they still exist.
    session = MagicMock()
    client = MagicMock()
    session.client.return_value = client
    paginator = MagicMock()
    client.get_paginator.return_value = paginator
    paginator.paginate.return_value = iter([{"instanceNames": ["prod-1", "prod-2"]}])

    result = cl.list_codedeploy_registered_on_premises_instances(session)

    assert result == ["prod-1", "prod-2"]
    client.get_paginator.assert_called_once_with("list_on_premises_instances")
    paginator.paginate.assert_called_once_with(registrationStatus="Registered")


def test_bedrock_prompt_routers_excludes_default():
    client = _FakeClient(
        paginated={
            "list_prompt_routers": [
                {
                    "promptRouterSummaries": [
                        {"promptRouterArn": "arn:custom", "type": "custom"},
                        {"promptRouterArn": "arn:default", "type": "default"},
                    ]
                }
            ]
        }
    )
    assert cl.list_bedrock_prompt_routers(_one_service("bedrock", client)) == ["arn:custom"]


def test_iam_roles_excludes_service_linked():
    # Emits RoleName (the AWS::IAM::Role CCAPI primaryIdentifier), not the ARN.
    client = _FakeClient(
        paginated={
            "list_roles": [
                {
                    "Roles": [
                        {"RoleName": "app", "Arn": "arn:role/app", "Path": "/"},
                        {"RoleName": "slr", "Arn": "arn:role/slr", "Path": "/aws-service-role/x"},
                    ]
                }
            ]
        }
    )
    assert cl.list_iam_roles(_one_service("iam", client)) == ["app"]


def test_kms_aliases_excludes_aws_managed():
    client = _FakeClient(
        paginated={
            "list_aliases": [
                {
                    "Aliases": [
                        {"AliasArn": "arn:alias/mine", "AliasName": "alias/mine"},
                        {"AliasArn": "arn:alias/aws", "AliasName": "alias/aws/s3"},
                    ]
                }
            ]
        }
    )
    assert cl.list_kms_aliases(_one_service("kms", client)) == ["alias/mine"]


def test_kms_keys_excludes_aws_managed_and_pending_deletion():
    def describe_key(KeyId):  # noqa: N803 — mirrors boto3 kwarg
        meta = {
            "k-customer": {"KeyManager": "CUSTOMER", "KeyState": "Enabled"},
            "k-aws": {"KeyManager": "AWS", "KeyState": "Enabled"},
            "k-pending": {"KeyManager": "CUSTOMER", "KeyState": "PendingDeletion"},
        }[KeyId]
        return {"KeyMetadata": meta}

    client = _FakeClient(
        paginated={
            "list_keys": [
                {
                    "Keys": [
                        {"KeyId": "k-customer", "KeyArn": "arn:key/customer"},
                        {"KeyId": "k-aws", "KeyArn": "arn:key/aws"},
                        {"KeyId": "k-pending", "KeyArn": "arn:key/pending"},
                    ]
                }
            ]
        },
        direct={"describe_key": describe_key},
    )
    assert cl.list_kms_keys(_one_service("kms", client)) == ["arn:key/customer"]


def test_kms_keys_skips_undescribable_key_with_warning(caplog):
    def describe_key(KeyId):  # noqa: N803
        raise ClientError({"Error": {"Code": "AccessDenied"}}, "DescribeKey")

    client = _FakeClient(
        paginated={"list_keys": [{"Keys": [{"KeyId": "k1", "KeyArn": "arn:key/1"}]}]},
        direct={"describe_key": describe_key},
    )
    import logging

    with caplog.at_level(logging.WARNING):
        assert cl.list_kms_keys(_one_service("kms", client)) == []
    assert "describe_key skipped" in caplog.text


def test_sns_subscriptions_skips_pending_confirmation():
    client = _FakeClient(
        paginated={
            "list_subscriptions": [
                {
                    "Subscriptions": [
                        {"SubscriptionArn": "arn:aws:sns:us-east-1:1:t:sub"},
                        {"SubscriptionArn": "PendingConfirmation"},
                    ]
                }
            ]
        }
    )
    assert cl.list_sns_subscriptions(_one_service("sns", client)) == [
        "arn:aws:sns:us-east-1:1:t:sub"
    ]


def test_cloudformation_stacks_returns_ids():
    client = _FakeClient(
        paginated={"list_stacks": [{"StackSummaries": [{"StackId": "arn:stack/a"}]}]}
    )
    assert cl.list_cloudformation_stacks(_one_service("cloudformation", client)) == ["arn:stack/a"]


def test_describe_cloudformation_stacks_excludes_terminal_and_emits_dicts():
    client = _FakeClient(
        paginated={
            "describe_stacks": [
                {
                    "Stacks": [
                        {
                            "StackId": "arn:live",
                            "StackName": "live",
                            "StackStatus": "CREATE_COMPLETE",
                        },
                        {
                            "StackId": "arn:gone",
                            "StackName": "gone",
                            "StackStatus": "DELETE_COMPLETE",
                        },
                    ]
                }
            ]
        }
    )
    out = cl.describe_cloudformation_stacks(_one_service("cloudformation", client))
    assert [d["id"] for d in out] == ["arn:live"]
    assert out[0]["service"] == "cloudformation"


def test_sagemaker_trial_components_excludes_deleting_states():
    client = _FakeClient(
        paginated={
            "list_trial_components": [
                {
                    "TrialComponentSummaries": [
                        {"TrialComponentArn": "arn:live", "Status": {"PrimaryStatus": "Completed"}},
                        {"TrialComponentArn": "arn:gone", "Status": {"PrimaryStatus": "Deleting"}},
                    ]
                }
            ]
        }
    )
    out = cl.list_sagemaker_trial_components(_one_service("sagemaker", client))
    assert [d["id"] for d in out] == ["arn:live"]


# -- manual-pagination listers: the NextToken/Marker loops --


def test_ram_permissions_pages_and_filters_customer_managed():
    responses = iter(
        [
            {
                "permissions": [
                    {"arn": "arn:p1", "permissionType": "CUSTOMER_MANAGED"},
                    {"arn": "arn:aws", "permissionType": "AWS_MANAGED"},
                ],
                "nextToken": "t2",
            },
            {"permissions": [{"arn": "arn:p2", "permissionType": "CUSTOMER_MANAGED"}]},
        ]
    )
    client = _FakeClient(direct={"list_permissions": lambda **_kw: next(responses)})
    assert cl.list_ram_permissions(_one_service("ram", client)) == ["arn:p1", "arn:p2"]


def test_ecs_capacity_providers_excludes_fargate_and_pages():
    responses = iter(
        [
            {
                "capacityProviders": [
                    {"name": "cp1", "capacityProviderArn": "arn:cp1"},
                    {"name": "FARGATE", "capacityProviderArn": "arn:fargate"},
                ],
                "nextToken": "n2",
            },
            {"capacityProviders": [{"name": "cp2", "capacityProviderArn": "arn:cp2"}]},
        ]
    )
    client = _FakeClient(direct={"describe_capacity_providers": lambda **_kw: next(responses)})
    assert cl.list_ecs_describe_capacity_providers(_one_service("ecs", client)) == [
        "arn:cp1",
        "arn:cp2",
    ]


def test_cloudfront_functions_pages_by_marker():
    responses = iter(
        [
            {
                "FunctionList": {
                    "Items": [{"FunctionMetadata": {"FunctionARN": "arn:f1"}}],
                    "NextMarker": "m2",
                }
            },
            {"FunctionList": {"Items": [{"FunctionMetadata": {"FunctionARN": "arn:f2"}}]}},
        ]
    )
    client = _FakeClient(direct={"list_functions": lambda **_kw: next(responses)})
    assert cl.list_cloudfront_list_functions(_one_service("cloudfront", client)) == [
        "arn:f1",
        "arn:f2",
    ]


def test_cloudfront_connection_groups_excludes_default():
    """The AWS-managed default connection group (IsDefault=True) is filtered out."""
    client = _FakeClient(
        direct={
            "list_connection_groups": {
                "ConnectionGroups": [
                    {"Id": "cg_default", "Name": "CreatedByCloudFront-x", "IsDefault": True},
                    {"Id": "cg_custom", "Name": "my-group", "IsDefault": False},
                ]
            }
        }
    )
    assert cl.list_cloudfront_connection_groups(_one_service("cloudfront", client)) == ["cg_custom"]


def test_cloudfront_connection_groups_pages_by_marker():
    """Manual Marker/NextMarker pagination collects every non-default group."""
    responses = iter(
        [
            {
                "ConnectionGroups": [
                    {"Id": "cg_default", "IsDefault": True},
                    {"Id": "cg_a", "IsDefault": False},
                ],
                "NextMarker": "m2",
            },
            {"ConnectionGroups": [{"Id": "cg_b", "IsDefault": False}]},
        ]
    )
    client = _FakeClient(direct={"list_connection_groups": lambda **_kw: next(responses)})
    assert cl.list_cloudfront_connection_groups(_one_service("cloudfront", client)) == [
        "cg_a",
        "cg_b",
    ]


def test_mq_configurations_pages_by_nexttoken_and_emits_bare_id():
    # Emits the bare configuration Id (CCAPI primaryIdentifier), not the Arn: CCAPI (the only delete
    # path — the mq client has no DeleteConfiguration op) rejects the ARN and deletes with the Id.
    responses = iter(
        [
            {"Configurations": [{"Id": "c-1", "Arn": "arn:c1"}], "NextToken": "t2"},
            {"Configurations": [{"Id": "c-2", "Arn": "arn:c2"}]},
        ]
    )
    client = _FakeClient(direct={"list_configurations": lambda **_kw: next(responses)})
    assert cl.list_mq_list_configurations(_one_service("mq", client)) == ["c-1", "c-2"]


def _wafv2_client(region: str, web_acls_by_scope: dict, ip_sets_by_scope: dict) -> MagicMock:
    """A wafv2 client stand-in whose list ops key off the requested Scope and know their region."""
    client = MagicMock()
    client.meta.region_name = region

    def list_web_acls(Scope, **_kw):  # noqa: N803 — mirrors boto3 kwarg
        return {"WebACLs": web_acls_by_scope.get(Scope, [])}

    def list_ip_sets(Scope, **_kw):  # noqa: N803
        return {"IPSets": ip_sets_by_scope.get(Scope, [])}

    client.list_web_acls.side_effect = list_web_acls
    client.list_ip_sets.side_effect = list_ip_sets
    return client


def test_wafv2_web_acls_emit_name_id_scope_composite_both_scopes_in_us_east_1():
    # WAFv2 has NO fast-scan lister today, yet 4 scenarios create a standalone WebACL -> a survivor
    # is an UNDETECTED leak. CCAPI's primaryIdentifier is the composite Name|Id|Scope (verified
    # live). CLOUDFRONT scope is only listable in us-east-1; REGIONAL in every region.
    client = _wafv2_client(
        "us-east-1",
        web_acls_by_scope={
            "REGIONAL": [{"Name": "reg-acl", "Id": "reg-id", "ARN": "arn:reg"}],
            "CLOUDFRONT": [{"Name": "cf-acl", "Id": "cf-id", "ARN": "arn:cf"}],
        },
        ip_sets_by_scope={},
    )
    session = MagicMock()
    session.client.side_effect = lambda service, **_kw: client
    assert cl.list_wafv2_web_acls(session) == [
        "reg-acl|reg-id|REGIONAL",
        "cf-acl|cf-id|CLOUDFRONT",
    ]


def test_wafv2_web_acls_skip_cloudfront_scope_outside_us_east_1():
    # Listing CLOUDFRONT scope outside us-east-1 raises WAFInvalidParameterException, so the lister
    # must only query REGIONAL there (CLOUDFRONT WebACLs are surfaced by the us-east-1 scan).
    client = _wafv2_client(
        "us-west-2",
        web_acls_by_scope={"REGIONAL": [{"Name": "reg-acl", "Id": "reg-id", "ARN": "arn:reg"}]},
        ip_sets_by_scope={},
    )
    session = MagicMock()
    session.client.side_effect = lambda service, **_kw: client
    assert cl.list_wafv2_web_acls(session) == ["reg-acl|reg-id|REGIONAL"]
    client.list_web_acls.assert_called_once_with(Scope="REGIONAL")


def test_wafv2_ip_sets_emit_name_id_scope_composite():
    client = _wafv2_client(
        "us-east-1",
        web_acls_by_scope={},
        ip_sets_by_scope={"REGIONAL": [{"Name": "blocklist", "Id": "ip-id", "ARN": "arn:ip"}]},
    )
    session = MagicMock()
    session.client.side_effect = lambda service, **_kw: client
    assert cl.list_wafv2_ip_sets(session) == ["blocklist|ip-id|REGIONAL"]


def test_waf_logging_configurations_passes_pagesize_and_returns_arns():
    # The bug this lister fixes: ListLoggingConfigurations needs Limit>=1, and a bare paginate()
    # omits it (ValidationException). Assert BOTH the ARNs and that PageSize is supplied (which
    # botocore maps to the required Limit) — a regression here silently reintroduces the failure.
    session = MagicMock()
    client = MagicMock()
    session.client.return_value = client
    paginator = MagicMock()
    client.get_paginator.return_value = paginator
    paginator.paginate.return_value = iter(
        [{"LoggingConfigurations": [{"ResourceArn": "arn:a"}, {"ResourceArn": "arn:b"}]}]
    )

    result = cl.list_waf_logging_configurations(session)

    assert result == ["arn:a", "arn:b"]
    client.get_paginator.assert_called_once_with("list_logging_configurations")
    paginator.paginate.assert_called_once_with(PaginationConfig={"PageSize": 100})


def test_waf_regional_logging_configurations_passes_limit_on_direct_call():
    # waf-regional has no paginator, so Limit must ride the direct call (same required-Limit bug).
    client = MagicMock()
    client.list_logging_configurations.return_value = {
        "LoggingConfigurations": [{"ResourceArn": "arn:r1"}]
    }
    session = MagicMock()
    session.client.return_value = client

    result = cl.list_waf_regional_logging_configurations(session)

    assert result == ["arn:r1"]
    client.list_logging_configurations.assert_called_once_with(Limit=100)


def test_lexv2_bots_pages_by_nexttoken():
    responses = iter(
        [
            {"botSummaries": [{"botId": "b1"}], "nextToken": "t2"},
            {"botSummaries": [{"botId": "b2"}]},
        ]
    )
    client = _FakeClient(direct={"list_bots": lambda **_kw: next(responses)})
    assert cl.list_lexv2_models_list_bots(_one_service("lexv2-models", client)) == ["b1", "b2"]


def test_cognito_user_pools_returns_ids():
    client = _FakeClient(paginated={"list_user_pools": [{"UserPools": [{"Id": "pool-1"}]}]})
    assert cl.list_cognito_idp_list_user_pools(_one_service("cognito-idp", client)) == ["pool-1"]


def test_backup_selections_composite_id_across_plans():
    client = _FakeClient(
        paginated={
            "list_backup_plans": [{"BackupPlansList": [{"BackupPlanId": "plan-1"}]}],
            "list_backup_selections": [{"BackupSelectionsList": [{"SelectionId": "sel-1"}]}],
        }
    )
    assert cl.list_backup_list_backup_selections(_one_service("backup", client)) == ["sel-1_plan-1"]


# -- EC2 drift-parity + all-* describers --


def test_ec2_addresses_and_eip_associations():
    client = _FakeClient(
        direct={
            "describe_addresses": {
                "Addresses": [
                    {
                        "PublicIp": "1.2.3.4",
                        "AllocationId": "eipalloc-1",
                        "AssociationId": "eipassoc-1",
                    },
                    {"PublicIp": "5.6.7.8", "AllocationId": "eipalloc-2"},
                ]
            }
        }
    )
    # AWS::EC2::EIP's CCAPI primaryIdentifier is the composite PublicIp|AllocationId (RC7).
    assert cl.list_ec2_describe_addresses(_one_service("ec2", client)) == [
        "1.2.3.4|eipalloc-1",
        "5.6.7.8|eipalloc-2",
    ]
    assert cl.list_ec2_describe_eip_associations(_one_service("ec2", client)) == ["eipassoc-1"]


def test_ec2_vpc_gateway_attachments_emits_igw_pipe_vpc():
    client = _FakeClient(
        paginated={
            "describe_vpcs": [{"Vpcs": []}],
            "describe_internet_gateways": [
                {"InternetGateways": [{"Attachments": [{"VpcId": "vpc-1"}]}]}
            ],
        }
    )
    assert cl.list_ec2_describe_vpc_gateway_attachments(_one_service("ec2", client)) == [
        "IGW|vpc-1"
    ]


def test_ec2_vpc_gateway_attachments_excludes_default_vpc():
    """The default VPC's IGW attachment is dropped; a scenario VPC's is kept."""
    client = _FakeClient(
        paginated={
            "describe_vpcs": [{"Vpcs": [{"VpcId": "vpc-default"}]}],
            "describe_internet_gateways": [
                {
                    "InternetGateways": [
                        {"Attachments": [{"VpcId": "vpc-default"}]},
                        {"Attachments": [{"VpcId": "vpc-app"}]},
                    ]
                }
            ],
        }
    )
    assert cl.list_ec2_describe_vpc_gateway_attachments(_one_service("ec2", client)) == [
        "IGW|vpc-app"
    ]


def test_ec2_all_security_groups_excludes_default():
    """The default security group (GroupName='default') is excluded; others are kept."""
    client = _FakeClient(
        paginated={
            "describe_security_groups": [
                {
                    "SecurityGroups": [
                        {"GroupId": "sg-default", "GroupName": "default"},
                        {"GroupId": "sg-custom", "GroupName": "my-app-sg"},
                    ]
                }
            ]
        }
    )
    assert cl.list_ec2_describe_all_security_groups(_one_service("ec2", client)) == [
        "sg-custom",
    ]


def test_ec2_all_security_groups_keeps_non_default():
    """Security groups without GroupName='default' are all kept."""
    client = _FakeClient(
        paginated={
            "describe_security_groups": [
                {
                    "SecurityGroups": [
                        {"GroupId": "sg-1", "GroupName": "web-tier"},
                        {"GroupId": "sg-2", "GroupName": "db-tier"},
                    ]
                }
            ]
        }
    )
    assert cl.list_ec2_describe_all_security_groups(_one_service("ec2", client)) == [
        "sg-1",
        "sg-2",
    ]


def test_ec2_all_network_acls_excludes_default():
    """Default network ACLs (IsDefault=True) are excluded; custom ones are kept."""
    client = _FakeClient(
        paginated={
            "describe_network_acls": [
                {
                    "NetworkAcls": [
                        {"NetworkAclId": "acl-default", "IsDefault": True, "VpcId": "vpc-1"},
                        {"NetworkAclId": "acl-custom", "IsDefault": False, "VpcId": "vpc-1"},
                    ]
                }
            ]
        }
    )
    assert cl.list_ec2_describe_all_network_acls(_one_service("ec2", client)) == [
        "acl-custom",
    ]


def test_ec2_all_network_acls_keeps_non_default():
    """Non-default network ACLs are all kept."""
    client = _FakeClient(
        paginated={
            "describe_network_acls": [
                {
                    "NetworkAcls": [
                        {"NetworkAclId": "acl-1", "IsDefault": False, "VpcId": "vpc-1"},
                        {"NetworkAclId": "acl-2", "IsDefault": False, "VpcId": "vpc-2"},
                    ]
                }
            ]
        }
    )
    assert cl.list_ec2_describe_all_network_acls(_one_service("ec2", client)) == [
        "acl-1",
        "acl-2",
    ]


def test_ec2_all_route_tables_excludes_main():
    """Main route tables (Main=True association) are excluded; others are kept."""
    client = _FakeClient(
        paginated={
            "describe_route_tables": [
                {
                    "RouteTables": [
                        {
                            "RouteTableId": "rtb-main",
                            "VpcId": "vpc-1",
                            "Associations": [{"Main": True, "RouteTableAssociationId": "a-1"}],
                        },
                        {
                            "RouteTableId": "rtb-custom",
                            "VpcId": "vpc-1",
                            "Associations": [
                                {"Main": False, "RouteTableAssociationId": "a-2", "SubnetId": "s-1"}
                            ],
                        },
                    ]
                }
            ]
        }
    )
    assert cl.list_ec2_describe_all_route_tables(_one_service("ec2", client)) == [
        "rtb-custom",
    ]


def test_ec2_all_route_tables_keeps_non_main():
    """Route tables without a Main=True association are all kept."""
    client = _FakeClient(
        paginated={
            "describe_route_tables": [
                {
                    "RouteTables": [
                        {
                            "RouteTableId": "rtb-1",
                            "Associations": [{"Main": False}],
                        },
                        {
                            "RouteTableId": "rtb-2",
                            "Associations": [],
                        },
                    ]
                }
            ]
        }
    )
    assert cl.list_ec2_describe_all_route_tables(_one_service("ec2", client)) == [
        "rtb-1",
        "rtb-2",
    ]


def test_ec2_subnet_route_table_associations_all_ids():
    """With no default VPC, every association id is returned (incl. main-table ones)."""
    client = _FakeClient(
        paginated={
            "describe_vpcs": [{"Vpcs": []}],
            "describe_route_tables": [
                {
                    "RouteTables": [
                        {
                            "VpcId": "vpc-app",
                            "Associations": [{"RouteTableAssociationId": "rtbassoc-1"}],
                        }
                    ]
                }
            ],
        }
    )
    assert cl.list_ec2_describe_subnet_route_table_associations(_one_service("ec2", client)) == [
        "rtbassoc-1"
    ]


def test_ec2_subnet_route_table_associations_excludes_default_vpc_main():
    """The default VPC's implicit main association is dropped; everything else is kept.

    Kept: the default VPC's explicit subnet association, and a NON-default VPC's main
    association (a task-created VPC's main table is real drift). Dropped: only the
    default VPC's ``Main`` association, which is AWS-created and undeletable.
    """
    client = _FakeClient(
        paginated={
            "describe_vpcs": [{"Vpcs": [{"VpcId": "vpc-default"}]}],
            "describe_route_tables": [
                {
                    "RouteTables": [
                        {
                            "VpcId": "vpc-default",
                            "Associations": [
                                {"RouteTableAssociationId": "rtbassoc-default-main", "Main": True},
                                {
                                    "RouteTableAssociationId": "rtbassoc-default-subnet",
                                    "Main": False,
                                    "SubnetId": "subnet-d",
                                },
                            ],
                        },
                        {
                            "VpcId": "vpc-app",
                            "Associations": [
                                {"RouteTableAssociationId": "rtbassoc-app-main", "Main": True}
                            ],
                        },
                    ]
                }
            ],
        }
    )
    assert cl.list_ec2_describe_subnet_route_table_associations(_one_service("ec2", client)) == [
        "rtbassoc-default-subnet",
        "rtbassoc-app-main",
    ]


def test_ec2_subnet_nacl_associations_all_ids():
    """With no default VPC, every NACL association id is returned."""
    client = _FakeClient(
        paginated={
            "describe_vpcs": [{"Vpcs": []}],
            "describe_network_acls": [
                {
                    "NetworkAcls": [
                        {
                            "NetworkAclId": "acl-app",
                            "IsDefault": False,
                            "VpcId": "vpc-app",
                            "Associations": [{"NetworkAclAssociationId": "aclassoc-1"}],
                        }
                    ]
                }
            ],
        }
    )
    assert cl.list_ec2_describe_subnet_nacl_associations(_one_service("ec2", client)) == [
        "aclassoc-1"
    ]


def test_ec2_subnet_nacl_associations_excludes_default_vpc():
    """The default VPC's default-NACL associations are dropped; everything else is kept.

    Kept: a custom (non-default) NACL's association in the default VPC, and a NON-default
    VPC's default-NACL association (a task-created VPC's NACL is real drift). Dropped: only
    the default VPC's default-NACL associations, which are AWS-created and undeletable.
    """
    client = _FakeClient(
        paginated={
            "describe_vpcs": [{"Vpcs": [{"VpcId": "vpc-default"}]}],
            "describe_network_acls": [
                {
                    "NetworkAcls": [
                        {
                            "NetworkAclId": "acl-default",
                            "IsDefault": True,
                            "VpcId": "vpc-default",
                            "Associations": [
                                {"NetworkAclAssociationId": "aclassoc-default-1"},
                                {"NetworkAclAssociationId": "aclassoc-default-2"},
                            ],
                        },
                        {
                            "NetworkAclId": "acl-custom",
                            "IsDefault": False,
                            "VpcId": "vpc-default",
                            "Associations": [{"NetworkAclAssociationId": "aclassoc-custom"}],
                        },
                        {
                            "NetworkAclId": "acl-app-default",
                            "IsDefault": True,
                            "VpcId": "vpc-app",
                            "Associations": [{"NetworkAclAssociationId": "aclassoc-app"}],
                        },
                    ]
                }
            ],
        }
    )
    assert cl.list_ec2_describe_subnet_nacl_associations(_one_service("ec2", client)) == [
        "aclassoc-custom",
        "aclassoc-app",
    ]


def test_ec2_all_vpcs_excludes_default():
    """The default VPC (IsDefault=True) is excluded; scenario VPCs are kept."""
    client = _FakeClient(
        paginated={
            "describe_vpcs": [
                {
                    "Vpcs": [
                        {"VpcId": "vpc-default", "IsDefault": True},
                        {"VpcId": "vpc-app", "IsDefault": False},
                    ]
                }
            ]
        }
    )
    assert cl.list_ec2_describe_all_vpcs(_one_service("ec2", client)) == ["vpc-app"]


def test_ec2_all_subnets_excludes_default_for_az():
    """Default subnets (DefaultForAz=True) are excluded; scenario subnets are kept.

    Keyed on the per-subnet DefaultForAz property, NOT default-VPC membership, so
    an agent-created subnet inside the default VPC still surfaces as drift/orphan.
    """
    client = _FakeClient(
        paginated={
            "describe_subnets": [
                {
                    "Subnets": [
                        {"SubnetId": "subnet-default", "DefaultForAz": True},
                        {"SubnetId": "subnet-agent-in-default-vpc", "DefaultForAz": False},
                        {"SubnetId": "subnet-app", "DefaultForAz": False},
                    ]
                }
            ]
        }
    )
    assert cl.list_ec2_describe_all_subnets(_one_service("ec2", client)) == [
        "subnet-agent-in-default-vpc",
        "subnet-app",
    ]


def test_ec2_all_internet_gateways_excludes_default_vpc_igw():
    """The IGW attached to the default VPC is excluded; scenario IGWs are kept.

    Classified by CURRENT attachment (role, not identity): a detached/floating IGW
    is deletable and must still surface, so only default-VPC-attached ones drop.
    """
    client = _FakeClient(
        paginated={
            "describe_vpcs": [{"Vpcs": [{"VpcId": "vpc-default"}]}],
            "describe_internet_gateways": [
                {
                    "InternetGateways": [
                        {
                            "InternetGatewayId": "igw-default",
                            "Attachments": [{"VpcId": "vpc-default"}],
                        },
                        {"InternetGatewayId": "igw-app", "Attachments": [{"VpcId": "vpc-app"}]},
                        {"InternetGatewayId": "igw-floating", "Attachments": []},
                    ]
                }
            ],
        }
    )
    assert cl.list_ec2_describe_all_internet_gateways(_one_service("ec2", client)) == [
        "igw-app",
        "igw-floating",
    ]


def test_ec2_all_internet_gateways_no_default_vpc_keeps_all():
    """With no default VPC in the region, every IGW is returned."""
    client = _FakeClient(
        paginated={
            "describe_vpcs": [{"Vpcs": []}],
            "describe_internet_gateways": [
                {"InternetGateways": [{"InternetGatewayId": "igw-app"}]}
            ],
        }
    )
    assert cl.list_ec2_describe_all_internet_gateways(_one_service("ec2", client)) == ["igw-app"]


def test_ec2_dhcp_options_excludes_default_vpc_associated_set():
    """The DHCP options set associated with the default VPC is excluded; others kept."""
    client = _FakeClient(
        paginated={
            "describe_vpcs": [
                {"Vpcs": [{"VpcId": "vpc-default", "DhcpOptionsId": "dopt-default"}]}
            ],
            "describe_dhcp_options": [
                {
                    "DhcpOptions": [
                        {"DhcpOptionsId": "dopt-default"},
                        {"DhcpOptionsId": "dopt-custom"},
                    ]
                }
            ],
        }
    )
    assert cl.list_ec2_describe_dhcp_options(_one_service("ec2", client)) == ["dopt-custom"]


def test_ec2_dhcp_options_no_default_vpc_keeps_all():
    """With no default VPC, every DHCP options set is returned."""
    client = _FakeClient(
        paginated={
            "describe_vpcs": [{"Vpcs": []}],
            "describe_dhcp_options": [{"DhcpOptions": [{"DhcpOptionsId": "dopt-custom"}]}],
        }
    )
    assert cl.list_ec2_describe_dhcp_options(_one_service("ec2", client)) == ["dopt-custom"]


def test_s3_bucket_policies_keeps_buckets_with_policy_only():
    def get_bucket_policy(Bucket):  # noqa: N803
        if Bucket == "has-policy":
            return {"Policy": "{}"}
        raise ClientError({"Error": {"Code": "NoSuchBucketPolicy"}}, "GetBucketPolicy")

    client = _FakeClient(
        direct={
            "list_buckets": {"Buckets": [{"Name": "has-policy"}, {"Name": "no-policy"}]},
            "get_bucket_policy": get_bucket_policy,
        }
    )
    assert cl.list_s3_bucket_policies(_one_service("s3", client)) == ["has-policy"]


def test_ecs_services_emit_service_arn_pipe_cluster_composite():
    # AWS::ECS::Service CCAPI primaryIdentifier is composite [ServiceArn, Cluster]; the bare service
    # ARN is rejected, so emit serviceArn|cluster (the cluster being paginated).
    client = _FakeClient(
        paginated={
            "list_clusters": [{"clusterArns": ["arn:cluster/a"]}],
            "list_services": [{"serviceArns": ["arn:svc/1", "arn:svc/2"]}],
        }
    )
    assert cl.list_ecs_list_services(_one_service("ecs", client)) == [
        "arn:svc/1|arn:cluster/a",
        "arn:svc/2|arn:cluster/a",
    ]


def test_imagebuilder_images_emit_the_build_version_arn():
    # AWS::ImageBuilder::Image delete_image needs the BUILD-version arn (.../1.0.0/N); list_images
    # returns only the VERSION arn (.../1.0.0), which delete_image rejects. Fan out per image
    # version via list_image_build_versions and emit each build-version arn.
    client = _FakeClient(
        paginated={
            "list_images": [
                {"imageVersionList": [{"arn": "arn:aws:imagebuilder:us-east-1:1:image/gita/1.0.0"}]}
            ],
            "list_image_build_versions": [
                {
                    "imageSummaryList": [
                        {"arn": "arn:aws:imagebuilder:us-east-1:1:image/gita/1.0.0/2"},
                        {"arn": "arn:aws:imagebuilder:us-east-1:1:image/gita/1.0.0/1"},
                    ]
                }
            ],
        }
    )
    assert cl.list_imagebuilder_images(_one_service("imagebuilder", client)) == [
        "arn:aws:imagebuilder:us-east-1:1:image/gita/1.0.0/2",
        "arn:aws:imagebuilder:us-east-1:1:image/gita/1.0.0/1",
    ]


def test_logs_metric_filters_emit_loggroup_pipe_filter_composite():
    # AWS::Logs::MetricFilter primaryIdentifier is composite [LogGroupName, FilterName]; the bare
    # filterName is rejected, so emit logGroupName|filterName.
    client = _FakeClient(
        paginated={
            "describe_metric_filters": [
                {
                    "metricFilters": [
                        {"filterName": "errors", "logGroupName": "/aws/lambda/fn"},
                        {"filterName": "warns", "logGroupName": "/aws/lambda/fn"},
                    ]
                }
            ]
        }
    )
    assert cl.list_logs_metric_filters(_one_service("logs", client)) == [
        "/aws/lambda/fn|errors",
        "/aws/lambda/fn|warns",
    ]


def test_eks_addons_emits_cluster_pipe_addon_across_clusters():
    """Fan out over clusters and emit each add-on as ``clusterName|addonName``."""

    class _ByCluster(_FakeClient):
        def __init__(self):
            super().__init__(paginated={"list_clusters": [{"clusters": ["c1", "c2"]}]})

        def get_paginator(self, op):
            if op != "list_addons":
                return super().get_paginator(op)
            addons = {"c1": ["vpc-cni", "coredns"], "c2": ["kube-proxy"]}
            paginator = MagicMock()
            paginator.paginate.side_effect = lambda **kw: iter(
                [{"addons": addons[kw["clusterName"]]}]
            )
            return paginator

    assert cl.list_eks_addons(_one_service("eks", _ByCluster())) == [
        "c1|vpc-cni",
        "c1|coredns",
        "c2|kube-proxy",
    ]


def test_eks_addons_empty_when_no_clusters():
    client = _FakeClient(paginated={"list_clusters": [{"clusters": []}]})
    assert cl.list_eks_addons(_one_service("eks", client)) == []


def test_redshift_clusters_by_identifier():
    client = _FakeClient(
        paginated={"describe_clusters": [{"Clusters": [{"ClusterIdentifier": "my-cluster"}]}]}
    )
    assert cl.list_redshift_clusters_by_identifier(_one_service("redshift", client)) == [
        "my-cluster"
    ]


def test_logs_log_groups_by_name():
    client = _FakeClient(
        paginated={"describe_log_groups": [{"logGroups": [{"logGroupName": "/aws/lambda/fn"}]}]}
    )
    assert cl.list_logs_log_groups_by_name(_one_service("logs", client)) == ["/aws/lambda/fn"]


def test_imagebuilder_component_build_versions_per_version():
    client = _FakeClient(
        paginated={
            "list_components": [{"componentVersionList": [{"arn": "arn:comp/1.0.0"}]}],
            "list_component_build_versions": [
                {"componentSummaryList": [{"arn": "arn:comp/1.0.0/1"}]}
            ],
        }
    )
    assert cl.list_imagebuilder_component_build_versions(_one_service("imagebuilder", client)) == [
        "arn:comp/1.0.0/1"
    ]


def test_s3_storage_lens_uses_account_id_and_pages():
    sts = _FakeClient(direct={"get_caller_identity": {"Account": "123456789012"}})
    responses = iter(
        [
            {"StorageLensConfigurationList": [{"Id": "lens-1"}], "NextToken": "t2"},
            {"StorageLensConfigurationList": [{"Id": "lens-2"}]},
        ]
    )
    s3control = _FakeClient(
        direct={"list_storage_lens_configurations": lambda **_kw: next(responses)}
    )
    session = _session({"sts": sts, "s3control": s3control})
    assert cl.list_s3_list_storage_lens_configurations(session) == ["lens-1", "lens-2"]


# -- simple collect()-backed + straightforward describers, for coverage of the id paths --


@pytest.mark.parametrize(
    ("fn", "service", "paginated", "expected"),
    [
        (
            # CCAPI primaryIdentifier is /properties/Id (bare id), not the ARN.
            cl.list_cloudfront_list_distributions,
            "cloudfront",
            {
                "list_distributions": [
                    {"DistributionList": {"Items": [{"Id": "E123", "ARN": "arn:d1"}]}}
                ]
            },
            ["E123"],
        ),
        (
            cl.list_amp_list_workspaces,
            "amp",
            {"list_workspaces": [{"workspaces": [{"arn": "arn:ws1"}]}]},
            ["arn:ws1"],
        ),
        (
            # CCAPI primaryIdentifier is /properties/Id (BrokerId), not BrokerArn.
            cl.list_mq_list_brokers,
            "mq",
            {"list_brokers": [{"BrokerSummaries": [{"BrokerId": "b-1", "BrokerArn": "arn:b1"}]}]},
            ["b-1"],
        ),
        (
            cl.list_iam_policies,
            "iam",
            {"list_policies": [{"Policies": [{"Arn": "arn:pol1"}]}]},
            ["arn:pol1"],
        ),
        (
            cl.describe_ec2_snapshots,
            "ec2",
            {"describe_snapshots": [{"Snapshots": [{"SnapshotId": "snap-1"}]}]},
            ["snap-1"],
        ),
        (
            cl.describe_ec2_images,
            "ec2",
            {"describe_images": [{"Images": [{"ImageId": "ami-1"}]}]},
            ["ami-1"],
        ),
        (
            cl.describe_ec2_nat_gateways,
            "ec2",
            {"describe_nat_gateways": [{"NatGateways": [{"NatGatewayId": "nat-1"}]}]},
            ["nat-1"],
        ),
    ],
)
def test_simple_lister_id_extraction(fn, service, paginated, expected):
    client = _FakeClient(paginated=paginated)
    assert fn(_one_service(service, client)) == expected


def test_appstream_private_images():
    client = _FakeClient(direct={"describe_images": {"Images": [{"Arn": "arn:img"}]}})
    assert cl.describe_appstream_images(_one_service("appstream", client)) == ["arn:img"]


def test_listers_tuple_is_complete_and_unique():
    """custom_listers() registers 146 unique-keyed listers, each pinned to service + op.

    The redundant plain ``ec2:DescribeSecurityGroupRules`` lister was dropped (the direction-split
    ``...RulesIngress`` / ``...RulesEgress`` listers already cover every rule, each pinned to its
    exact CFN type), and a custom ``codedeploy:ListOnPremisesInstances`` lister was added that
    passes ``registrationStatus=Registered`` (so it does not enumerate deregistered instances). A
    custom ``ecr:GetRegistryPolicy`` lister surfaces the account's registry policy (an otherwise
    unlistable singleton) so its cleanup handler can delete it. A custom ``eks:ListAddons`` lister
    fans out over clusters, emitting each add-on as the composite id ``clusterName|addonName``. Two
    caller-scoped EC2 listers were added — ``describe_ec2_prefix_lists`` (filters by ``owner-id``
    so AWS-managed lists are excluded) and ``describe_ec2_vpc_endpoint_services`` (uses
    ``describe_vpc_endpoint_service_configurations`` to return only caller-published services
    instead of every AWS-managed service available in the region). Two ``wafv2`` fan-out listers
    (ListWebACLs, ListIPSets) emit the ``Name|Id|Scope`` composite so standalone WAFv2 resources —
    which had no lister at all — are detected and CCAPI-deletable. A ``logs`` metric-filter lister
    emits the composite ``logGroupName|filterName`` (the CCAPI primaryIdentifier a SimpleLister
    cannot express). A custom ``ec2:DescribeDhcpOptions`` lister supersedes the simple row so the
    default VPC's associated (undeletable) DHCP options set is excluded from orphan reporting.
    """
    listers = cl.custom_listers()
    assert len(listers) == 157
    keys = [(lister.service, lister.op) for lister in listers]
    assert len(keys) == len(set(keys))
    assert all(callable(lister.run) for lister in listers)


def test_mq_brokers_exclude_deletion_in_progress():
    # Amazon MQ broker deletion is slow (~15-20 min): a broker in DELETION_IN_PROGRESS is on its
    # way out, not an orphan, so it is dropped; every other state — including CREATION_FAILED and a
    # missing state — still surfaces so a genuinely stuck broker is reported.
    # Emits BrokerId (CCAPI primaryIdentifier), not BrokerArn.
    client = _FakeClient(
        paginated={
            "list_brokers": [
                {
                    "BrokerSummaries": [
                        {"BrokerId": "b-live", "BrokerState": "RUNNING"},
                        {"BrokerId": "b-gone", "BrokerState": "DELETION_IN_PROGRESS"},
                        {"BrokerId": "b-failed", "BrokerState": "CREATION_FAILED"},
                        {"BrokerId": "b-unknown"},
                    ]
                }
            ]
        }
    )
    assert cl.list_mq_list_brokers(_one_service("mq", client)) == [
        "b-live",
        "b-failed",
        "b-unknown",
    ]


def test_redshift_clusters_exclude_deleting_states():
    # A Redshift cluster shutdown is slow: a cluster in deleting (or final-snapshot, the
    # delete-with-snapshot phase) is dropped, while available and failure states still surface.
    client = _FakeClient(
        paginated={
            "describe_clusters": [
                {
                    "Clusters": [
                        {"ClusterIdentifier": "live", "ClusterStatus": "available"},
                        {"ClusterIdentifier": "deleting", "ClusterStatus": "deleting"},
                        {"ClusterIdentifier": "snap", "ClusterStatus": "final-snapshot"},
                        {"ClusterIdentifier": "stuck", "ClusterStatus": "incompatible-network"},
                    ]
                }
            ]
        }
    )
    assert cl.list_redshift_clusters_by_identifier(_one_service("redshift", client)) == [
        "live",
        "stuck",
    ]
