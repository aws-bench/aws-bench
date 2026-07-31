# aws-bench

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/aws-bench/aws-bench?tab=contributing-ov-file)

An open-source benchmark for evaluating AI coding agents on real-world AWS tasks.

## Overview

aws-bench measures how well AI agents and model combinations (e.g. Claude Code with Sonnet, Codex with GPT) perform on real AWS work — diagnosing misconfigurations, provisioning infrastructure, and operating live cloud environments.

Unlike benchmarks that score against static fixtures, aws-bench runs each agent against **disposable, real AWS environments**:

1. It provisions isolated AWS accounts and deploys a **scenario** (real infrastructure defined as CDK stacks).
2. It runs the agent against a **task** inside a sandboxed container, with scoped AWS credentials.
3. It scores the result with an **automated verifier** — either an LLM judge (for read-only diagnosis tasks) or a programmatic check against live AWS state (for tasks that create or modify resources).

This gives a faithful, reproducible signal of agent performance on the kind of work AWS practitioners actually do.

- **Datasets:** tasks and scenarios live in the companion repo, [aws-bench-datasets](https://github.com/aws-bench/aws-bench-datasets).

> **Built on Harbor.** aws-bench is based on [Harbor](https://github.com/harbor-framework/harbor), an open-source framework for evaluating AI agents and language models. aws-bench extends Harbor with the AWS-specific environment provisioning, scenarios, and verifiers described above.

## Table of Contents

- [How It Works](#how-it-works)
- [Datasets](#datasets)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Configuring AWS Access](#configuring-aws-access)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
- [Security](#security)
- [License](#license)

## How It Works

At a high level, using aws-bench follows three steps:

1. **Provision** the AWS test environment (`aws-bench env init` + `env setup`) — this creates disposable accounts and deploys the scenario's real infrastructure.
2. **Run** the benchmark (`aws-bench run`) — the agent attempts each task, and the verifier scores the results.
3. **Tear down** the environment (`aws-bench env cleanup` + `env terminate`) — this removes the deployed resources and closes the test accounts.

aws-bench is built around a handful of core concepts:

| Concept      | Description                                                                                                                                         |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Scenario** | A real AWS environment (CDK stacks + setup scripts) that a task runs against. Each scenario maps to a dedicated AWS account.                        |
| **Task**     | A single problem an agent must solve against a scenario, with an instruction, a verifier, and a reference solution.                                 |
| **Run**      | One full benchmark execution across a dataset. Each attempt at a single task is a _trial_, and a run spans many trials (tasks × agents × attempts). |
| **Verifier** | The scoring logic. Writes a reward (`1.0` = pass, `0.0` = fail) for each trial.                                                                     |

Tasks come in two families:

- **Introspection** (read-only): the agent diagnoses the environment and writes an answer; an LLM judge compares it against a reference answer.
- **Mutation** (create/modify): the agent changes real AWS resources; a programmatic verifier checks live AWS state, and changes are rolled back afterward.

See the [Getting Started guide](docs/getting-started.md) for a deeper walkthrough of these concepts and a full setup tutorial.

## Datasets

Datasets are versioned collections of tasks and their scenarios. aws-bench ships a small set of curated datasets, and **every scenario is also runnable as its own dedicated dataset** — so you can work with the benchmark incrementally, one environment at a time, instead of running the full suite.

### Curated datasets

| Dataset                | Tasks | Scenarios | Description                                                                                                                                                                                               |
| ---------------------- | ----- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `aws-bench-quickstart` | 9     | 1         | The recommended starting point (used in the [Quickstart](#quickstart)). A tiny dataset — reusing the `ec2-multiregion` tasks — to confirm your installation and AWS environment are configured correctly. |
| `aws-bench-basic`      | 78    | 4         | Entry-level tasks across core AWS services. The fastest way to establish a baseline for an agent.                                                                                                         |
| `aws-bench-advanced`   | 47    | 3         | Full-difficulty tasks exercising multi-service, real-world AWS scenarios.                                                                                                                                 |

### Per-scenario datasets

Each scenario is published as a standalone dataset, so you can run just one environment at a time:

`api-and-observability` · `compute-and-data` · `databases-and-storage` · `ec2-multiregion` · `reference-architectures` · `serverless-apps` · `streaming-and-iot` · `troubleshooting-multiservice`

Run any of them by name, for example:

```bash
uv run aws-bench run -d ec2-multiregion@latest --env-name my-bench -a claude-code -m <model-id>
```

For the complete list of agents, models, and datasets you can pass to `-a` / `-m` / `-d`, see [Supported agents, models, and datasets](docs/getting-started.md#supported-agents-models-and-datasets).

## Requirements

| Requirement           | Details                                                                                                                                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **OS**                | macOS or Linux                                                                                                                                                                                         |
| **Python**            | 3.12+                                                                                                                                                                                                  |
| **Package manager**   | [uv](https://docs.astral.sh/uv/getting-started/installation/)                                                                                                                                          |
| **Container runtime** | Docker with the Compose v2 plugin **and buildx ≥ 0.17.0** (see [Troubleshooting](#troubleshooting))                                                                                                    |
| **AWS**               | An AWS account you control, with permissions to create an AWS Organization and member accounts (aws-bench provisions disposable test accounts). See [Configuring AWS Access](#configuring-aws-access). |

## Installation

```bash
git clone https://github.com/aws-bench/aws-bench.git && cd aws-bench
uv sync
uv run aws-bench --help
```

## Quickstart

> This is the condensed flow. For detailed setup — account creation, quota handling, and platform-specific instructions — see the [Getting Started guide](docs/getting-started.md).

```bash
# 1. Configure AWS credentials for your management account.
#    Use static keys...
aws configure --profile my-aws-bench-profile
#    ...or IAM Identity Center (SSO):
aws configure sso --profile my-aws-bench-profile   # one-time setup
aws sso login --profile my-aws-bench-profile        # refresh the session later
export AWS_PROFILE=my-aws-bench-profile             # any profile with the required permissions

# 2. Initialize the test environment (creates org, accounts, quotas)
uv run aws-bench env init \
  --env-name aws-bench-env \
  -d aws-bench-quickstart \
  --wait-for-quotas

# 3. Deploy scenario resources
uv run aws-bench env setup \
  --env-name aws-bench-env \
  -d aws-bench-quickstart

# 4. Generate a Bedrock bearer token for the verifier and agent,
#    (supposing your agent uses Bedrock for LLM inference).
#    First ensure no stale AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN
#    are set in your shell, as they would override AWS_PROFILE and mint the token
#    against the wrong account.
eval $(uv run aws-bench env creds --eval)

# 5. Run the benchmark
uv run aws-bench run \
  --env-name aws-bench-env \
  -d aws-bench-quickstart \
  -a claude-code \
  -m <model-id> \
  --ve AWS_BEARER_TOKEN_BEDROCK=$AWS_BEARER_TOKEN_BEDROCK \
  --yes
```

Results land in `jobs/<timestamp>/`. Each trial has a `reward.json` (`1.0` = pass, `0.0` = fail).

**Example agent and model IDs** — pass these to `-a` / `-m` (use a model your account can access):

| `-a` (agent)  | `-m` (model)                                                                |
| ------------- | --------------------------------------------------------------------------- |
| `claude-code` | `global.anthropic.claude-sonnet-5`                                          |
| `kiro-cli`    | `global.anthropic.claude-sonnet-5`                                          |
| `oracle`      | _(none — replays the reference solution, useful for validating a scenario)_ |

Model IDs follow the provider's naming (the examples above are Amazon Bedrock model IDs). See [Supported agents, models, and datasets](docs/getting-started.md#supported-agents-models-and-datasets) in the Getting Started guide for the full list of supported agents, how each authenticates to a model provider, and the available datasets.

## Usage

### Environment management

```bash
aws-bench env init       # Provision environment, accounts, submit quota requests
aws-bench env setup      # Deploy scenario resources (CDK stacks)
aws-bench env show       # Display environment state
aws-bench env creds      # Generate a Bedrock bearer token for convenience (use with --eval; optional)
aws-bench env verify     # Check for resource drift
aws-bench env reset      # Reset to post-setup state
aws-bench env cleanup    # Tear down deployed resources
aws-bench env terminate  # Close test accounts
```

### Running benchmarks

```bash
aws-bench run            # Start a benchmark job
```

### Common flags

| Flag              | Purpose                                               |
| ----------------- | ----------------------------------------------------- |
| `--env-name`      | Name of the testing environment                       |
| `-d name@version` | Dataset from the registry (e.g. `aws-bench-basic`)    |
| `--path` / `-p`   | Path to a local tasks directory (alternative to `-d`) |
| `--scenario-path` | Path to a local scenarios directory                   |
| `-a` / `--agent`  | Agent to evaluate (e.g. `claude-code`, `oracle`)      |
| `-m` / `--model`  | Model ID                                              |
| `--yes`           | Skip confirmation prompts                             |
| `--quiet`         | Reduce output verbosity                               |

### Dataset modes

**Registry mode** — reference a published dataset by name:

```bash
uv run aws-bench run -d aws-bench-basic --env-name my-bench -a claude-code -m <model-id>
```

**Local path mode** — point at local directories (useful when developing tasks):

```bash
uv run aws-bench run \
  --path ./tasks/my-scenario \
  --scenario-path ./scenarios \
  --env-name my-bench -a claude-code -m <model-id>
```

## Configuring AWS Access

aws-bench provisions disposable AWS accounts under an AWS Organization, so it needs credentials for a **management account** with permission to create and manage member accounts. It scopes down to least-privilege roles for the agent and verifier at run time.

- **Region:** set `AWS_REGION=us-east-1` before running aws-bench — it expects the `us-east-1` region. Also export `AWS_DEFAULT_REGION=us-east-1` in your shell so your own AWS CLI/SDK calls resolve to the same region.
- **Minimal permissions:** the exact IAM policy required is documented in `docs/getting-started`.
- **Model access:** agents call an LLM provider of your choice — aws-bench does not require any particular one. If you run models on **Amazon Bedrock**, the `aws-bench env creds` command generates a bearer token as a convenience; ensure your account has access to the models you intend to evaluate. Agents using other providers can ignore this command and supply their own credentials.

## Troubleshooting

| Problem                                                                    | Solution                                                                    |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Quota gate blocks `env setup`                                              | Re-run `env init --wait-for-quotas` or check the AWS Service Quotas console |
| `env setup` fails with Docker errors                                       | Ensure the Docker daemon is running; verify `docker ps` works               |
| `docker compose … unknown flag` or `'compose' is not a docker command`     | Install the Docker Compose v2 plugin (see below)                            |
| `compose build requires buildx 0.17.0 or later`                            | Upgrade buildx to ≥ 0.17.0 (see below)                                      |
| Model-provider credentials expired                                         | Re-run `eval $(uv run aws-bench env creds --eval)`                          |
| `AccountContaminatedError` during `aws-bench env setup` or `aws-bench run` | Run `aws-bench env cleanup`                                                 |

### Docker Compose v2 + buildx

aws-bench orchestrates agent environments with `docker compose` and builds images with buildx, so **both plugins must be present and recent**. Docker Desktop bundles both. On native Linux, Docker may be installed without the Compose v2 plugin or with a buildx too old for modern Compose.

```bash
docker compose version    # need v2.x+   (not the legacy `docker-compose` v1)
docker buildx version     # need v0.17.0+
```

If either is missing or too old, install them at the user level (no sudo required):

```bash
mkdir -p ~/.docker/cli-plugins
ARCH=$(uname -m)                                   # x86_64 or aarch64
LARCH=$([ "$ARCH" = aarch64 ] && echo arm64 || echo amd64)

# Docker Compose v2
curl -fSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-${ARCH}" \
  -o ~/.docker/cli-plugins/docker-compose && chmod +x ~/.docker/cli-plugins/docker-compose

# buildx (latest; must be >= 0.17.0)
BX=$(curl -fsSL https://api.github.com/repos/docker/buildx/releases/latest \
     | grep -oE '"tag_name":[[:space:]]*"v[0-9.]+"' | grep -oE 'v[0-9.]+' | head -1)
curl -fSL "https://github.com/docker/buildx/releases/download/${BX}/buildx-${BX}.linux-${LARCH}" \
  -o ~/.docker/cli-plugins/docker-buildx && chmod +x ~/.docker/cli-plugins/docker-buildx
```

## Roadmap

aws-bench is under active development. Near-term priorities:

- **arXiv report** — publish the benchmark paper and reference results.
- **Expanded dataset tiers** — grow the `basic` and `advanced` datasets and add more scenarios.
- **Public leaderboard** — standardized results reporting across agents and models.
- **Broader agent support** — first-class support for more agent frameworks.
- **Expanded mutation-task coverage** — more create/modify tasks with programmatic verifiers.

Have a request? Open an issue or start a discussion.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](https://github.com/aws-bench/aws-bench?tab=contributing-ov-file) for how to set up a development environment, run the tests, and submit changes, and our [Code of Conduct](https://github.com/aws-bench/aws-bench?tab=coc-ov-file) before participating.

- **Bugs and feature requests:** open a [GitHub issue](https://github.com/aws-bench/aws-bench/issues).
- **Datasets:** task and scenario contributions go to [aws-bench-datasets](https://github.com/aws-bench/aws-bench-datasets).
- **Framework development:** see the [Framework Development Guide](docs/framework-development.md) for repo structure, testing expectations, and the full contributor workflow.
- **Datasets development:** see the [Datasets Development Guide](docs/datasets-development.md) for dataset contribution guidelines and expectations.

### Development

```bash
make test       # pytest
make lint       # ruff check
make format     # ruff format
make typecheck  # pyright
make check      # lint + typecheck (no auto-fix)
make ready      # all of the above
```

## Security

Please report security issues responsibly — see [SECURITY.md](https://github.com/aws-bench/aws-bench?tab=security-ov-file). Do not open public issues for security vulnerabilities.

## License

aws-bench is licensed under the [Apache License 2.0](LICENSE). See the [NOTICE](NOTICE) file for attributions.
