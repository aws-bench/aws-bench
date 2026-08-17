# Pre-existing AWS accounts

aws-bench normally creates disposable member accounts in an AWS Organization. In
`preexisting` mode, an external platform such as AWS Control Tower owns the account,
OU, SCPs, and account termination; aws-bench receives an explicit scenario-to-account
allowlist and manages only benchmark resources inside those accounts.

Use this mode when the Organization is centrally administered and aws-bench is not
allowed to create or close accounts. It is a new aws-bench backend, not a mode that
the upstream benchmark previously supplied.

This mode is experimental.

## Configuration

Start from `examples/preexisting-ec2-multiregion.yaml` in the repository:

```yaml
schema_version: "1.0"
mode: preexisting
name: acme-benchmark
accounts:
  ec2-multiregion:
    PRIMARY: "111122223333"
runner_role: AWSBenchRunner
cfn_role: cfn-service-execution
```

Pass the file as a global option, before the command:

```bash
uv run aws-bench --account-config ./accounts.yaml env init \
  --env-name acme-benchmark -d aws-bench-quickstart@0.7.0
```

`AWSBENCH_ACCOUNT_CONFIG=/path/to/accounts.yaml` is equivalent. A config is an
allowlist: missing scenarios, missing account tags, and any account ID not listed in
it fail closed. One account cannot appear under multiple scenarios in the same file,
because each scenario needs its own clean baseline. With one physical account, run
one scenario wave at a time and use a different config for the next wave after a
successful cleanup.

`name` must not match an OU name the managed backend uses. One organization can host
both modes, and a shared name makes each mode resolve the other's environment.

## Prerequisites the external platform owns

aws-bench does not create any of these, and `env init` fails if one is missing:

1. The member accounts named in the config.
2. `runner_role` in each account, assumable by the identity running aws-bench.
3. The role named by `cfn_role` in each account, trusting
   `cloudformation.amazonaws.com` with permission to delete the scenario's stacks.
   Cleanup passes it as `RoleARN` on `DeleteStack` so teardown does not depend on the
   CDK bootstrap role. It may be the same role as `runner_role`, provided that role's
   trust policy admits both `cloudformation.amazonaws.com` and whoever assumes the
   runner identity.
4. Service quotas already meeting the scenario's requirements. aws-bench verifies
   quotas in this mode and does not request increases, so an unmet quota fails
   `env init` rather than opening a support case.
5. **A region-restriction SCP** limiting each account to the scenario's declared
   regions.

## What `env init` does

In pre-existing mode, `env init` does not create an Organization, OU, account, SCP,
IAM role, or quota request. It validates:

- the scenario/account mapping;
- the configured `runner_role` can enter the account;
- the role named by `cfn_role` exists;
- current service quotas already meet the scenario requirements.

It then creates the contamination state file and captures the pristine pre-setup
baseline. Resource discovery runs in the process rather than through the
management-account scanner Lambda, which assumes a role and lives in an account that
does not exist in this mode.

`env terminate` is disabled: it refuses before issuing any Organizations call.

## The region guardrail is not verified

**aws-bench does not check that the region-restriction SCP exists or is in force.**
Confirming it is the operator's responsibility.

This matters more than it looks. Cleanup discovers resources only in the regions the
scenario declares, and that set is a boundary only if something actually denies the
others. With no SCP in force, an agent holding broad permissions can create resources
in an unscanned region, and cleanup will not find them — they persist after teardown
reports success.

## Where benchmark state lives

Baseline snapshots and contamination flags are written to `~/.aws-bench/state/`, on
the host running aws-bench. Nothing benchmark-specific is written into an account
under test, so no state sits where a task's agent could read or modify it.

The consequence is that state is host-local. A run must reach `env cleanup` from the
same machine that ran `env setup`; another host sees no baseline. Set `state_file` to
put the contamination marker on shared persistent storage for cluster runs.

Reading contamination flags fails closed: if the state file is missing, or present but
not parseable as the expected schema, aws-bench raises rather than reading either as "no
accounts contaminated". Recording a new flag is the exception — it creates the file when
absent, because adding a flag only ever narrows what aws-bench will reuse.

## Credentials, in plain language

`runner_role` is required. An interactive IAM Identity Center session can assume it; an
unattended workload identity needs the same trust path. Explicit per-task roles are
assumed from that runner identity. A task that names no role runs as the runner role —
aws-bench never widens permissions to those of whoever invoked it.

One case reuses the invoking session directly: when the caller's ambient identity
already *is* the role being targeted, aws-bench reuses that session rather than
attempting a self-assume, which most trust policies reject. Permissions are identical,
but the session is the caller's, so the resulting CloudTrail entries carry the caller's
session name rather than an `aws-bench-*` one, and the credential's lifetime is the
ambient session's. Run from a session that is not itself a task role if you need every
action attributable to aws-bench.

aws-bench does not route through `OrganizationAccountAccessRole` in this mode; that
role exists only in accounts the managed backend created.

Do not use long-lived access keys. An unattended batch run needs a renewable workload
identity that can assume the runner role. An interactive SSO login proves human access,
but its browser session is not a durable 24-hour batch-job identity.
