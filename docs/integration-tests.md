# Integration Tests

`make integration` runs end-to-end acceptance tests against **real AWS
resources** in a dedicated test account. The tests drive the installed
`aws-bench` CLI as a subprocess, exactly as a user would. Each test covers the
happy path of one command or feature. `tests/integration/BACKLOG.md` lists what
is covered today and what comes next.

The tests are opt-in. They are excluded from `make test`, `make check` and CI
(pytest marker `integration`), and the target refuses to run unless you opt in
explicitly.

> **Warning:** a run deploys a scenario into your test account, runs the
> benchmark against it, and then removes everything with `env cleanup`. Resources
> cost money while they exist. Use an account that holds nothing else, and never
> one that could be production.

## What the test does

`test_quickstart_oracle_earns_full_reward` runs the `aws-bench-quickstart`
dataset (9 tasks, scenario `ec2-multiregion`) with the `oracle` agent. The
`oracle` agent replays each task's reference solution instead of calling a
model, so every trial is expected to score `1.0`.

| Step | Command | Timeout |
|---|---|---|
| 1 | `env init` validates the account and records the pristine baseline | 20 min |
| 2 | `env setup` deploys the scenario (CDK) in `us-east-1`, `us-west-1`, `us-west-2` | 45 min |
| 3 | `run -a oracle`, with pre-run environment verification on | 90 min |
| 4 | Asserts every trial completed, without errors, with reward `1.0` | |
| 5 | `env cleanup` always runs, even if an earlier step failed | 30 min |

`scripts/check_job_rewards.py` checks the rewards. It fails if the job has no
trials, if any trial raised an exception, if fewer trials completed than
planned, or if any trial has a missing reward or one below `1.0`. You can also
run it on its own: `python scripts/check_job_rewards.py <jobs-dir>/<job-name>`.

The other tests in `tests/integration/test_acceptance.py` are named placeholders
(skipped) for the commands in the backlog.

## 1. Prepare the test account

The tests use [pre-existing mode](preexisting-accounts.md). aws-bench does not
create or close accounts. You prepare one member account, and `env init` only
validates it. The quickstart scenario needs **one** account (account tag
`PRIMARY`).

### 1.1 Runner role

The role aws-bench assumes in the test account, for example `AWSBenchRunner`. It
bootstraps CDK, deploys the scenario stacks (including IAM roles), and assumes
the per-task agent roles that those stacks create. In a disposable test account,
`AdministratorAccess` is the practical choice.

Trust the identity you run the tests from:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "AWS": "<CALLER_ROLE_ARN>" },
    "Action": "sts:AssumeRole"
  }]
}
```

### 1.2 CloudFormation role

`env cleanup` passes this role as `RoleARN` on `DeleteStack`, for example
`cfn-service-execution`. It must trust `cloudformation.amazonaws.com` and be
able to delete the scenario's stacks:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "cloudformation.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
```

The runner role can serve as both, if its trust policy admits both principals.

### 1.3 Service quota

`ec2-multiregion` requires **Running On-Demand Standard (A, C, D, H, I, M, R, T,
Z) instances**, quota code `L-1216C47A`, to be at least **8 vCPUs in
`us-east-1`**. Pre-existing mode checks quotas and fails `env init` if they are
too low. It never requests an increase. Check, and request one if needed (in the
test account):

```bash
aws service-quotas get-service-quota --region us-east-1 \
  --service-code ec2 --quota-code L-1216C47A --query 'Quota.Value'

aws service-quotas request-service-quota-increase --region us-east-1 \
  --service-code ec2 --quota-code L-1216C47A --desired-value 8
```

Request it early, because approval can take time.

### 1.4 Region-restriction SCP

aws-bench **does not verify** this guardrail, but you need it. Cleanup only
scans the scenario's regions. Attach an SCP to the test account (from the
Organization's management account) that denies everything outside them. Review
the global-service exemptions for your organization:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "DenyOutsideScenarioRegions",
    "Effect": "Deny",
    "NotAction": [
      "iam:*", "sts:*", "organizations:*", "account:*", "support:*",
      "cloudfront:*", "route53:*", "budgets:*", "ce:*", "health:*"
    ],
    "Resource": "*",
    "Condition": {
      "StringNotEquals": { "aws:RequestedRegion": ["us-east-1", "us-west-1", "us-west-2"] }
    }
  }]
}
```

The quickstart tasks still pass with this SCP. The reference solutions enumerate
all regions but skip those that return `AccessDenied`.

### 1.5 Bedrock access for the LLM judge

All nine quickstart tasks are scored by an LLM judge
(`bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0`, from each task's
`tests/judge.toml`). A Bedrock token is needed even with the `oracle` agent.
In the account where you mint the token (section 3), enable model access to
**Claude Sonnet 4.5**.

## 2. Write the account config

Create an account config YAML **outside the repository**. Account IDs are not
secrets, but they don't belong in commits. Start from
`examples/preexisting-ec2-multiregion.yaml`:

```yaml
schema_version: "1.0"
mode: preexisting
name: <ENV_NAME>               # used as --env-name; must not match any OU name in the org
runner_role: <RUNNER_ROLE_NAME>
cfn_role: <CFN_ROLE_NAME>
accounts:
  ec2-multiregion:
    PRIMARY: "<TEST_ACCOUNT_ID>"
```

In pre-existing mode `--env-name` must equal `name`. The test reads it from this
file.

## 3. Prepare the machine

- **Docker** with Compose v2 and buildx 0.17.0 or later.
- **uv**, and the repository installed (`uv sync`).
- **AWS credentials** for an identity that can assume the runner role. Check:

  ```bash
  aws sts assume-role --role-arn arn:aws:iam::<TEST_ACCOUNT_ID>:role/<RUNNER_ROLE_NAME> \
    --role-session-name preflight --query 'AssumedRoleUser.Arn' --output text
  ```

- **Use credentials that refresh themselves.** A full run can take longer than
  one hour. aws-bench refreshes its own sessions into the runner role, but each
  refresh uses your base credentials. Static keys exported into the shell expire
  mid-run, and `env cleanup` can then leave resources behind. Use a
  self-refreshing source instead, such as:
  - a `credential_process` profile
  - IAM Identity Center
  - an instance role

  Unset any static `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and
  `AWS_SESSION_TOKEN`, which would override your profile.
- **Bedrock token** for the judge:

  ```bash
  eval $(uv run aws-bench env creds --eval)   # exports AWS_BEARER_TOKEN_BEDROCK
  ```

  `env creds` mints the token in the account of the credentials you run it with.
  It creates there:
  - an IAM user `bedrock-api-user` with `AmazonBedrockFullAccess`
  - a 30-day service-specific credential (an IAM user holds at most two)
  - a cached copy in SSM Parameter Store, `/bedrock-aws-bench/bedrock-api-key`

  Later calls reuse the cached token. Never commit the token.

## 4. Run

```bash
AWS_BENCH_INTEG=1 \
AWS_BENCH_INTEG_ACCOUNT_CONFIG=/path/to/accounts.yaml \
make integration
```

Without the two opt-in variables, the target prints usage and exits with code 2.

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `AWS_BENCH_INTEG` | yes | Must be `1` to opt in |
| `AWS_BENCH_INTEG_ACCOUNT_CONFIG` | yes | Path to the account config YAML (section 2) |
| `AWS_BEARER_TOKEN_BEDROCK` | yes | Bedrock token for the LLM judge (section 3). The test fails before deploying anything without it. |
| `AWS_PROFILE` (or other standard AWS variables) | yes | Base credentials that can assume the runner role |
| `AWS_BENCH_INTEG_DATASET` | no | Registry dataset (default `aws-bench-quickstart`) |
| `AWS_BENCH_INTEG_DATASET_PATH` | no | Path to a local `aws-bench-datasets` checkout, used instead of the registry. Use it to test unreleased dataset changes. Cannot be combined with `AWS_BENCH_INTEG_DATASET`. |
| `AWS_BENCH_INTEG_SCENARIO` | no | Scenario to use with `AWS_BENCH_INTEG_DATASET_PATH` (default `ec2-multiregion`). The account config must map it. |
| `AWS_BENCH_INTEG_TASKS` | no | Comma-separated `run -i` globs to run a subset of tasks, e.g. `'*list-ec-instances-all-regions'`. Start each glob with `*`, so it matches both directory and registry task names. |
| `AWS_BENCH_INTEG_HOME_LINKS` | no | Extra directories (comma-separated, relative to `HOME`) to link into the test's scratch `HOME`. Use it when your credential helper reads files there (see [How the test isolates state](#how-the-test-isolates-state)). |
| `AWS_BENCH_INTEG_ENV_NAME` | no | If set, must equal the config's `name`. The test fails otherwise. |

`-d <dataset>` always downloads `registry.json` from the repository's `main`
branch on GitHub, not from your checkout. To test a registry change before it is
merged, use `AWS_BENCH_INTEG_DATASET_PATH` with a matching datasets checkout.

### Examples

```bash
# One task only (quicker smoke run)
AWS_BENCH_INTEG=1 AWS_BENCH_INTEG_ACCOUNT_CONFIG=/path/to/accounts.yaml \
AWS_BENCH_INTEG_TASKS='*list-ec-instances-all-regions' \
make integration

# Unreleased dataset changes from a local checkout
AWS_BENCH_INTEG=1 AWS_BENCH_INTEG_ACCOUNT_CONFIG=/path/to/accounts.yaml \
AWS_BENCH_INTEG_DATASET_PATH=/path/to/aws-bench-datasets \
make integration
```

## How the test isolates state

- **Scratch `HOME`.** Each pytest session runs the CLI with `HOME` set to a
  scratch directory (`<pytest-tmp>/integ-home0`). aws-bench state and logs (in
  `~/.aws-bench`) therefore never touch your own environments. `~/.aws` and
  `~/.docker` are symlinked in, and `AWS_BENCH_INTEG_HOME_LINKS` adds more.
- **Scratch working directory.** Job output goes to `<pytest-tmp>/integ-cwd0/jobs/`.
- **No overlapping runs on one account.** The env name is fixed by the config, and
  stacks are per account, so two runs against the same account interfere. Use one
  account per concurrent runner, or run one at a time.
- **State is host-local.** `env cleanup` must run on the machine that ran
  `env setup`, because it needs the baseline recorded by `env init`.

## Watching a run and reading results

Each CLI call's output is captured and printed only on failure. While a run is in
progress, from another terminal:

```bash
H=$(ls -td "$(python3 -c 'import tempfile;print(tempfile.gettempdir())')"/pytest-of-$USER/pytest-*/integ-home0 | head -1)
ls -t "$H/.aws-bench/logs"                                      # one directory per command
tail -f "$(ls -td "$H"/.aws-bench/logs/*/ | head -1)run.log"   # the current command's log
```

The job's `result.json` is in `<pytest-tmp>/integ-cwd0/jobs/integ-<run-id>/`.
pytest keeps the last three `pytest-*` directories.

## Recovering after an interrupted run

If the run was interrupted after `env setup`, or cleanup failed, run cleanup
again with the same scratch `HOME`, so aws-bench finds the recorded baseline:

```bash
HOME="$H" AWSBENCH_ACCOUNT_CONFIG=/path/to/accounts.yaml \
  uv run aws-bench env cleanup --env-name <ENV_NAME> -d aws-bench-quickstart --yes
```

With a local datasets checkout, replace `-d ...` with
`--scenario-path /path/to/aws-bench-datasets/scenarios --include-scenarios ec2-multiregion`.

Afterwards, check that no scenario stacks remain in the three regions. Only
`CDKToolkit` is expected, if a bootstrap ran. Also check for leftover AMIs and
EBS snapshots in `us-east-1`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `make integration` prints usage, exit code 2 | Opt-in variables not set | Set `AWS_BENCH_INTEG=1` and `AWS_BENCH_INTEG_ACCOUNT_CONFIG` |
| `TestEnvironmentNotFoundError: Active pre-existing config names environment ...` | `--env-name` differs from the config's `name` | Unset `AWS_BENCH_INTEG_ENV_NAME`, or make it match |
| `AWS_BEARER_TOKEN_BEDROCK is not set` | No judge credential | `eval $(uv run aws-bench env creds --eval)` |
| `env init` fails on quotas | `L-1216C47A` below 8 in `us-east-1` | Request the increase (section 1.3) |
| Credential helper fails only under the test | It reads files under `HOME`, which the test redirects | List its directories in `AWS_BENCH_INTEG_HOME_LINKS` |
| `ExpiredToken` mid-run, cleanup fails | Base credentials don't refresh | Use a self-refreshing source (section 3), then recover (above) |
| `Verification failed ... New resources detected` before trials start | Something changed the account after `env setup`. If the resource is an EBS snapshot `Created by CreateImage(...)`, the run used an `ec2-multiregion` scenario from before the AMI-wait fix in the datasets repo. | Use a current registry or datasets checkout, then recover and re-run |

## Validation status

The quickstart test has been run live in a pre-existing-mode test account. All
nine `aws-bench-quickstart` tasks completed with reward `1.0`, with pre-run
environment verification on, and `env cleanup` finished successfully. A run
longer than one hour with self-refreshing credentials has not been validated yet.
