# Datasets Development Guide

This guide covers how to contribute to [aws-bench-datasets](https://github.com/aws-bench/aws-bench-datasets) — authoring scenarios and tasks, running them locally, and ensuring quality standards.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Dataset Structure](#dataset-structure)
- [Authoring a Scenario](#authoring-a-scenario)
- [Authoring a Task](#authoring-a-task)
- [Running Your Dataset](#running-your-dataset)
- [Contribution Guidelines](#contribution-guidelines)
- [Common Gotchas](#common-gotchas)
- [Further Reading](#further-reading)

---

## Overview

An aws-bench dataset is a collection of **scenarios** (AWS environments) and **tasks** (benchmark instructions). Scenarios provision infrastructure; tasks instruct an AI agent to interact with that infrastructure. The framework handles account isolation, credential injection, environment reset, and verification.

Tasks fall into two categories:

- **Introspection** (`request_type = "introspection"`) — The agent queries AWS and reports findings. Verified via `tests/ground_truth.json` using an LLM judge.
- **Mutation** (`request_type = "mutation"`) — The agent makes changes to AWS. Verified via `tests/check.py` which programmatically asserts expected state using boto3.

---

## Getting Started

### Prerequisites

- Node.js 20+
- npm
- AWS CDK CLI (`npx cdk`)
- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/)
- Docker
- Access to the [aws-bench](https://github.com/aws-bench/aws-bench) framework
- A configured AWS test environment (see [aws-bench README](https://github.com/aws-bench/aws-bench))

### Setup

```bash
git clone https://github.com/aws-bench/aws-bench-datasets.git && cd aws-bench-datasets
npm ci
```

### Running Checks

```bash
make ready    # Auto-fix Python formatting, then run the full gate (recommended before submitting)
```

Or run individual steps:

```bash
make build    # Compile TypeScript + CDK apps
make test     # Jest tests, linting (Python, shell, Docker), config validation
make check    # build + test combined (full gate without auto-fix)
make fix      # Auto-fix Python lint/format only (mutates files)
```

---

## Dataset Structure

A **scenario** defines a reusable AWS environment (infrastructure provisioned via CDK or other IaC). A **task** is a single benchmark instruction that an agent executes against a deployed scenario.

Scenarios and tasks are stored as top-level directories in the datasets package (`scenarios/` and `tasks/`). Tasks are organized in nested directories by the scenario they correspond to (`tasks/<scenario-id>/<task-name>/`), making it easy to locate and run all tasks for a given scenario.

```
aws-bench-datasets/
├── scenarios/
│   └── <scenario-id>/
│       ├── scenario.toml
│       ├── scenario/
│       │   ├── Dockerfile
│       │   ├── cdk_app/
│       │   └── setup/            # optional: post-deploy setup scripts
│       ├── deploy/
│       │   └── deploy.sh
│       ├── reset/                 # optional: reset to post-setup baseline
│       │   └── reset.sh
│       └── cleanup/               # optional: teardown
│           └── cleanup.sh
├── tasks/
│   └── <scenario-id>/
│       └── <task-name>/
│           ├── task.toml
│           ├── instruction.md
│           ├── environment/
│           │   ├── Dockerfile
│           │   └── docker-compose.yaml
│           ├── tests/
│           │   ├── test.sh
│           │   ├── ground_truth.json   # introspection tasks
│           │   └── check.py            # mutation tasks
│           ├── pre_invoke/             # optional
│           │   └── pre_invoke.sh
│           ├── post_invoke/            # optional
│           │   └── post_invoke.sh
│           └── solution/               # optional
│               └── solve.sh
└── shared/
    ├── judge/                     # shared LLM judge files
    ├── tasks/                     # shared per-task reset scripts
    │   ├── <scenario-id>/
    │   │   └── <task-name>/reset.py
    │   └── scripts/sync.sh
    └── steering/                  # shared agent steering instructions
```

Key relationships:
- Many tasks can/should reference the same scenario (N:1).
- Each task references exactly one scenario via `scenario_id` in `task.toml`.
- Scenarios never share an account at runtime — each gets its own isolated test account.

> **Note:** Tasks must be organized under `tasks/<scenario-id>/` for public contributions. While the framework technically supports any flat or nested layout (as long as each task directory contains a valid `task.toml`), organizing by scenario is required to maintain consistency across the dataset. This convention also lets you quickly run all tasks for a single scenario by passing that path to `--path`. When using path arguments, the framework will not pick up tasks in nested directories — you cannot point the framework to the base path of many directories containing tasks.

---

## Authoring a Scenario

A scenario defines an AWS environment that tasks operate against. The scenario contract is IaC-agnostic — you can use CDK, Terraform, CloudFormation, or plain AWS CLI scripts. In the canonical aws-bench-datasets package, we use **AWS CDK (TypeScript)**, which is the recommended option. See `scenarios/` for working examples.

> **Note on non-CDK IaC:** Scenarios are designed to conceptually support any IaC mechanism (CDK, Terraform, CloudFormation, etc.), but the current framework implementation contains some assumptions on the usage of CDK or CloudFormation (e.g., automatic resource cleanup, drift detection, environment verification). Please [create an issue](https://github.com/aws-bench/aws-bench/issues) if you want to contribute scenarios that use other IaC mechanisms so that we can provide additional support.

**Setup scripts (optional).** Many scenarios need post-deploy mutations that CDK cannot express declaratively — for example, seeding a DynamoDB table with test data, sending messages to SQS to generate logs, or intentionally breaking a resource into a specific error state for troubleshooting tasks. These are handled by **setup scripts**: small Python (or shell) scripts that run *after* `cdk deploy` completes, inside the same container. Setup scripts live in `scenario/setup/` and are orchestrated by `deploy.sh`. They receive the same AWS credentials and environment variables as CDK. Each script should be idempotent — safe to re-run on an already-configured environment.

### Scenario directory layout

```
my-scenario/
├── scenario.toml              # required: metadata + resource requirements
├── scenario/                  # required: Docker build context
│   ├── Dockerfile             # required
│   ├── cdk_app/               # CDK application (recommended)
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   ├── cdk.json
│   │   └── stacks/
│   └── setup/                 # optional: post-deploy setup scripts
│       └── setup_*.py
├── deploy/
│   └── deploy.sh              # required: provisions resources
├── reset/                     # optional
│   └── reset.sh
└── cleanup/                   # optional
    └── cleanup.sh
```

### Writing `scenario.toml`

This is the only file the framework reads directly. Everything else is opaque — your Docker image owns it.

Minimal example:

```toml
schema_version = "1.0"

[scenario]
name = "my-scenario"
description = "One-line summary of what this scenario provisions."
account_tags = ["PRIMARY"]
regions = ["us-east-1"]
```

With all options:

```toml
schema_version = "1.0"

[scenario]
name = "my-scenario"
description = "Lambda function with misconfigured environment variables."
authors = [{ name = "Your Name", email = "you@example.com" }]
keywords = ["lambda", "debugging"]
account_tags = ["PRIMARY"]
regions = ["us-east-1", "us-west-2"]

[environment]
build_timeout_sec = 600.0   # Docker build timeout (default: 600)
cpus = 2                    # container CPUs (default: 1)
memory_mb = 4096            # container memory (default: 2048)

[deploy]
timeout_sec = 1200.0        # deploy script timeout (default: 600)

[verify]
timeout_sec = 300.0         # verify script timeout (default: 120)

[cleanup]
timeout_sec = 600.0         # cleanup script timeout (default: 300)

[[quotas]]
account_tag = "PRIMARY"
region = "us-east-1"
service_code = "ec2"
quota_code = "L-1216C47A"
desired_value = 8
```

Field rules:
- **`account_tags`** — exactly one entry in v1. Becomes the AWS profile name inside the container.
- **`regions`** — must be non-empty. Every `[[quotas]]` region must appear here.
- **`[[quotas]]`** — one entry per (account, region, quota). Each is an idempotent service-quota request submitted by `aws-bench env init`.

### Writing the Dockerfile

Pick a base image that fits your IaC tool. The image needs:
- Your IaC tool (CDK, Terraform, CloudFormation CLI, etc.)
- AWS CLI v2 (if your scripts use `aws` commands)
- Your infrastructure source code

Example for CDK (TypeScript):

```dockerfile
FROM --platform=linux/amd64 public.ecr.aws/docker/library/node:20-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git unzip \
    && rm -rf /var/lib/apt/lists/*

# AWS CLI v2
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip \
    && unzip -q awscliv2.zip && ./aws/install \
    && rm -rf aws awscliv2.zip

RUN npm install -g aws-cdk

COPY cdk_app/ /app/cdk_app/
RUN cd /app/cdk_app && npm install

# Copy setup scripts (if any)
COPY setup/ /app/setup/
```

### Writing `deploy.sh`

Runs inside the container. The framework provides:
1. **One AWS profile per `account_tag`** in `~/.aws/config` (auto-refreshing credentials).
2. **One env var per `account_tag`** with the resolved account ID (e.g., `$PRIMARY=123456789012`).

CDK + setup scripts example:

```bash
#!/bin/bash
set -euo pipefail

cd /app/cdk_app

export CDK_DEFAULT_ACCOUNT="$PRIMARY"

npm run build

npx cdk bootstrap --profile PRIMARY "aws://${PRIMARY}/us-east-1"
npx cdk deploy --profile PRIMARY --all --require-approval never --concurrency 10

# Run post-deploy setup scripts
export AWS_PROFILE=PRIMARY
for script in /app/setup/setup_*.py; do
    [ -f "$script" ] || continue
    echo "--- $(basename "$script") ---"
    python3 "$script" || { echo "Setup failed: $script" >&2; exit 1; }
done
```

Always use `set -euo pipefail`. Exit 0 = success; non-zero = failure.

### Writing `cleanup.sh` (optional)

Use for teardown the framework can't handle automatically (non-CFN resources, versioned S3 buckets, cross-account artifacts). Your script runs first, then the framework deletes remaining CFN stacks.

Scripts must be idempotent and always `exit 0` — cleanup is best-effort; failures must not block the framework.

### Writing `reset/reset.sh` (optional)

Runs during `env reset`. Restores the account to post-setup baseline without a full teardown/redeploy. Faster than the cleanup + setup cycle.

The reset script typically calls per-task `reset.py` scripts from `shared/tasks/` to undo agent mutations. See [Per-Task Reset Scripts](#per-task-reset-scripts) below.

---

## Authoring a Task

A task defines a single benchmark instruction for the agent. Each task references one scenario.

### Task directory layout

```
my-task/
├── task.toml              # required: task metadata + scenario reference
├── instruction.md         # required: what the agent sees
├── environment/           # required: agent container definition
│   ├── Dockerfile
│   └── docker-compose.yaml
├── tests/                 # required: verification
│   ├── test.sh            # required entry point
│   ├── ground_truth.json  # introspection tasks: reference answer
│   └── check.py           # mutation tasks: programmatic verifier
├── pre_invoke/            # optional: runs before the agent
│   └── pre_invoke.sh
├── post_invoke/           # optional: runs after tests (cleanup)
│   └── post_invoke.sh
└── solution/              # optional: reference solution
    └── solve.sh
```

### Writing `solution/solve.sh` (optional)

A reference solution that demonstrates how to complete the task correctly. Used for validation during development — you can run it to confirm the verifier produces a passing score. Not executed during benchmark runs.

### Writing `task.toml`

```toml
schema_version = "1.1"

[task]
name = "my-category/my-task-name"
description = "Short description of what the agent must do"

[scenario]
scenario_id = "my-scenario"
agent_role_name = "ApplicationReadOnlyRole"

[metadata]
id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
category = "my-category"
request_type = "introspection"
aws_services = ["lambda", "cloudwatch"]

[agent]
timeout_sec = 300.0

[verifier]
timeout_sec = 120.0

[verifier.env]
FUNCTION_NAME = "{{my-scenario-Lambda-us-east-1-FunctionName}}"
```

The TOML file also specifies environment variables for different stages (e.g., `[verifier.env]`, `[solution.env]`, `[reset.env]`, `[pre_invoke.env]`, `[post_invoke.env]`).

Key sections:
- **`[scenario]`** — links this task to its scenario. `scenario_id` must match a scenario's `[scenario].name`.
- **`agent_role_name`** — the IAM role the framework assumes on behalf of the agent. This role must be created by your scenario's CDK code.
- **`[verifier.env]`** — env vars passed to the test script. Use `{{placeholder}}` syntax for values resolved from CloudFormation outputs at runtime. Other stages (`[pre_invoke.env]`, `[post_invoke.env]`, `[reset.env]`, `[solution.env]`) also support placeholders.

### Writing `instruction.md`

The prompt the agent receives. Keep it clear and specific. You can use `{{placeholder}}` values that get resolved at runtime.

Introspection task example:

```markdown
You have access to an AWS account with a Lambda function named `{{FunctionName}}`.

The function is failing intermittently. Investigate the root cause.

IMPORTANT: Write your final answer to /logs/agent/agent-output.txt.
```

Mutation task example:

```markdown
You have access to an AWS account. Create an EMR cluster that runs a Spark step
to process the data in s3://{{BucketName}}/input/.

Write the output to /logs/agent/agent-output.json with the following format:

{"clusterId": "<cluster id>", "stepId": "<step id>"}
```

### Writing the verifier (`tests/test.sh`)

The verifier validates whether the agent completed the task correctly. The framework executes `tests/test.sh` as the entry point.

#### Using the shared judge (recommended for introspection tasks)

aws-bench provides a ready-to-use LLM judge in `shared/judge/`. Copy the shared files into your task's `tests/` directory and only maintain `tests/ground_truth.json`:

```bash
bash shared/tasks/scripts/sync.sh
```

Then write `tests/ground_truth.json`:

```json
{
  "instruction": "List all EC2 instance IDs in all regions.",
  "expected_answer": "You have instances {{my-scenario-EC2-us-east-1-InstanceId}} in us-east-1."
}
```

Declare any placeholders in `task.toml`:

```toml
[verifier.env]
my-scenario-EC2-us-east-1-InstanceId = "{{my-scenario-EC2-us-east-1-InstanceId}}"
```

#### Custom verification (mutation tasks)

For mutation tasks with functional checks, write your own `tests/check.py`. The only contract is: the verifier must write a reward to `/logs/verifier/reward.txt`.

### Lifecycle Hooks

#### `pre_invoke/` (optional)

Runs **before** the agent starts. Common uses:
- Seed dynamic state (create resources the agent needs to discover).
- Generate runtime placeholders resolved from live account state.
- Mutate the environment to create the "broken" state the agent must fix.

Credentials are injected as environment variables — use the AWS CLI and SDKs without `--profile`.

#### `post_invoke/` (optional)

Runs **after** the agent finishes (regardless of pass/fail). Common uses:
- Roll back mutations the agent made (restore original state for the next run).
- Clean up resources created during `pre_invoke` that shouldn't persist.

Both hooks:
- Run inside the scenario container with AWS credentials for the scenario account.
- Must be idempotent (may run multiple times across retries).
- Have a configurable timeout. If your task creates resources that take time to delete (e.g., ElastiCache clusters, EKS clusters), ensure the timeout is sufficient and delete resources in dependency order.
- Are copied into the container at runtime — they are isolated from the host filesystem. Any scripts or dependencies they call must be included within the same folder or already present in the container image.
- Shared implementations live in `shared/tasks/` and are synced via `bash shared/tasks/scripts/sync.sh`.

### Per-Task Reset Scripts

Per-task data-plane reset logic lives in `shared/tasks/<scenario-id>/<task-name>/reset.py`. These scripts:
- Handle cleanup of agent-created resources (e.g., EKS clusters, S3 objects, DynamoDB items).
- Are shared across both `pre_invoke` and `post_invoke` — the same `reset.py` is imported by both hooks to ensure consistent cleanup logic.
- Follow a best-effort pattern: return a list of error strings rather than raising exceptions.
- Must be kept in sync with the copies in each task's directory. Run `bash shared/tasks/scripts/sync.sh` to propagate changes from `shared/` into the task directories — the script also verifies that all copies are up to date and will flag drift.

Placeholder environment variables can be defined in the `[reset.env]` section of `task.toml` (e.g., `REGION = "{{my-placeholder}}"`) and will be resolved at runtime for the reset script. All placeholders used must also be declared in `scenario.toml`. Since multiple tasks share a scenario, placeholder names must not collide.

---

## Running Your Dataset

### 1. Initialize the environment

```bash
aws-bench env init \
  --env-name my-test-env \
  --scenario-path /path/to/my-dataset/scenarios \
  --wait-for-quotas
```

### 2. Deploy scenarios

```bash
aws-bench env setup \
  --env-name my-test-env \
  --scenario-path /path/to/my-dataset/scenarios
```

### 3. Run the benchmark

```bash
eval $(uv run aws-bench env creds --eval)

aws-bench run \
  --env-name my-test-env \
  --scenario-path /path/to/my-dataset/scenarios \
  --path /path/to/my-dataset/tasks \
  -a claude-code \
  -m global.anthropic.claude-sonnet-4-6 \
  --ve AWS_BEARER_TOKEN_BEDROCK=$AWS_BEARER_TOKEN_BEDROCK
```

### 4. Run a single scenario's tasks

```bash
aws-bench run \
  --env-name my-test-env \
  --scenario-path /path/to/my-dataset/scenarios/my-scenario \
  --path /path/to/my-dataset/tasks/my-scenario \
  -a claude-code \
  -m global.anthropic.claude-sonnet-4-6 \
  --ve AWS_BEARER_TOKEN_BEDROCK=$AWS_BEARER_TOKEN_BEDROCK
```

### 5. Verify and clean up

```bash
# Check for drift
aws-bench env verify --env-name my-test-env --scenario-path /path/to/my-dataset/scenarios

# Tear down resources
aws-bench env cleanup --env-name my-test-env --scenario-path /path/to/my-dataset/scenarios --yes
```

---

## Contribution Guidelines

### Code Standards

- **TypeScript (CDK)**: Follow existing stack patterns. Many resource types default to `RemovalPolicy.RETAIN`, which prevents stack deletion. Explicitly set `RemovalPolicy.DESTROY` and `autoDeleteObjects: true` for all S3 buckets, DynamoDB tables, and log groups to ensure proper cleanup after benchmark runs.
- **Shell scripts**: Use `set -euo pipefail`. Scripts must be idempotent and best-effort (exit 0).
- **Python (hooks)**: Format with `ruff`. No `.pyc` or `__pycache__` in committed files.
- **Commit messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/) — `feat(scenario-id): description` or `fix(scenario-id): description`.

### Task Quality Guidelines

- **Instruction clarity**: The agent prompt should be unambiguous. An engineer should be able to solve it without guessing intent.
- **Verifier correctness**: `tests/ground_truth.json` must be accurate for the deployed environment. The assumptions or premise on the account presented in ground truth should be true. For mutation tasks, test against a live account to verify that the asserted state via `tests/check.py` is real.
- **Isolation**: Tasks must not depend on side effects from other tasks. Each task runs in its own container against a reset environment.
- **Metadata completeness**: Fill in `difficulty_explanation` and `solution_explanation` in `task.toml`.

---

## Common Gotchas

- **`account_tag` must match exactly** between `scenario.toml` fields — tags are case-sensitive.
- **Every `[[quotas]].region` must appear in `[scenario].regions`.**
- **`scenario_id` in `task.toml` must match the `[scenario].name` in `scenario.toml`** — not the directory name.
- **Placeholder `{{values}}` in task files** must have matching CloudFormation stack outputs from `deploy.sh`.
- **Use `RemovalPolicy.DESTROY`** in CDK — many resource types default to `RETAIN`, which prevents stack deletion during cleanup.
- **Delete resources in dependency order** in `post_invoke` and `reset.py` — e.g., delete an ElastiCache replication group before its subnet group, or stack deletion will fail.
- **Symlinks inside `deploy/`, `verify/`, `cleanup/` are rejected** — copy files directly or include them in the Docker build context.

---

## Further Reading

- Working examples in `scenarios/` and `tasks/`
- `shared/judge/README.md` — shared LLM judge reference
