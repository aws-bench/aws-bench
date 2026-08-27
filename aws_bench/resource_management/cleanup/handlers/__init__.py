"""Resource cleanup handlers.

Importing this package registers all resource handlers, pre-delete hooks,
and failed-resource handlers.
"""

from aws_bench.resource_management.cleanup.handlers import (
    acm,  # noqa: F401
    asg,  # noqa: F401
    athena,  # noqa: F401
    batch,  # noqa: F401
    bedrock,  # noqa: F401
    cross_service,  # noqa: F401
    databases,  # noqa: F401
    directory_service,  # noqa: F401
    dms,  # noqa: F401
    dynamodb,  # noqa: F401
    ec2_image,  # noqa: F401
    ecr,  # noqa: F401
    efs,  # noqa: F401
    eks_nodegroup,  # noqa: F401
    eks_pod_identity,  # noqa: F401
    elbv2,  # noqa: F401
    emr,  # noqa: F401
    events,  # noqa: F401
    glue,  # noqa: F401
    iam,  # noqa: F401
    imagebuilder,  # noqa: F401
    iot,  # noqa: F401
    ipam,  # noqa: F401
    lakeformation,  # noqa: F401
    lambda_,  # noqa: F401
    medialive,  # noqa: F401
    route53,  # noqa: F401
    s3,  # noqa: F401
    s3_bucket_policy,  # noqa: F401
    s3express,  # noqa: F401
    s3tables,  # noqa: F401
    sagemaker,  # noqa: F401
    servicecatalog,  # noqa: F401
    vpc,  # noqa: F401
)
