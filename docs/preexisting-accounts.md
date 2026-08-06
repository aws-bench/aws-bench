# Pre-existing AWS accounts

aws-bench normally creates disposable member accounts in an AWS Organization. In
`preexisting` mode, an external platform such as AWS Control Tower owns the account,
OU, SCPs, and account termination; aws-bench receives an explicit scenario-to-account
allowlist and manages only benchmark resources inside those accounts.

Use this mode when the Organization is centrally administered and aws-bench is not
allowed to create or close accounts. It is a new aws-bench backend, not a mode that
the upstream benchmark previously supplied.

## Configuration

Start from
[`examples/preexisting-ec2-multiregion.yaml`](../examples/preexisting-ec2-multiregion.yaml):

```yaml
schema_version: "1.0"
mode: preexisting
name: aws-bench
accounts:
  ec2-multiregion:
    PRIMARY: "111122223333"
runner_role: AWSBenchRunner
```

Pass the file as a global option, before the command:

```bash
uv run aws-bench --account-config ./accounts.yaml env init \
  --env-name aws-bench -d aws-bench-quickstart@0.7.0
```

`AWSBENCH_ACCOUNT_CONFIG=/path/to/accounts.yaml` is equivalent. A config is an
allowlist: missing scenarios, missing account tags, and any account ID not listed in
it fail closed. One account cannot appear under multiple scenarios in the same file,
because each scenario needs its own clean baseline. With one physical account, run
one scenario wave at a time and use a different config for the next wave after a
successful cleanup.

## What `env init` does

In pre-existing mode, `env init` does not create an Organization, OU, account, SCP,
IAM role, or quota request. It validates:

- the scenario/account mapping;
- the ambient identity or configured `runner_role` can enter the account;
- the externally provisioned `cfn-service-execution` role exists;
- current service quotas already meet the scenario requirements.

It then captures the pristine pre-setup baseline. The baseline uses the account's
`awsbench-state-<account-id>` S3 bucket and therefore may create/configure that
benchmark-owned bucket on first use. Resource discovery runs in the process rather
than through the management-account scanner Lambda.

`env terminate` is disabled. Contamination flags are kept in the config's adjacent
`.state.json` file, or in `state_file` when specified; place that file on persistent
storage for SLURM runs.

## Credentials, in plain language

For the managed `aws-bench` account, set `runner_role: AWSBenchRunner`. An interactive
IAM Identity Center `SolutionsAdmin` session can assume it; an unattended workload
identity will need the same trust path. Explicit per-task roles are assumed directly
from that runner identity; aws-bench no longer routes through
`OrganizationAccountAccessRole`, which exists only in accounts created by the original
benchmark flow.

Do not use long-lived access keys. An unattended SLURM run needs a renewable workload
identity that can assume the runner role. An interactive SSO login proves human access,
but its browser session is not a durable 24-hour batch-job identity.
