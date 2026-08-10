# Getting Started with aws-bench

This guide walks you through installing aws-bench, configuring AWS access, and running your first benchmark end-to-end against the datasets that ship with the project.

It is focused on **running the benchmark with the provided datasets**. Authoring your own scenarios and tasks is a separate topic and is not covered here.

## Contents

- [What is aws-bench?](#what-is-aws-bench)
- [Key concepts](#key-concepts)
- [How it works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Configure AWS access](#configure-aws-access)
- [Install aws-bench](#install-aws-bench)
- [Your first run: the quickstart dataset](#your-first-run-the-quickstart-dataset)
- [Supported agents, models, and datasets](#supported-agents-models-and-datasets)
- [Running the full datasets](#running-the-full-datasets)
- [Command reference](#command-reference)
- [Checking results](#checking-results)
- [Troubleshooting](#troubleshooting)
- [Environment health and stale state](#environment-health-and-stale-state)
- [Appendix: Amazon Bedrock model access](#appendix-amazon-bedrock-model-access)
- [Appendix: Docker Compose v2 + buildx](#appendix-docker-compose-v2-buildx)

## What is aws-bench?

aws-bench is an open-source benchmark for evaluating AI coding agents on real-world AWS tasks. It provisions disposable AWS accounts, deploys real infrastructure (a **scenario**), runs an agent against a **task** inside a sandboxed container, and scores the result with an automated **verifier**. It is built on the [Harbor](https://github.com/harbor-framework/harbor) evaluation framework.

Datasets (tasks + scenarios) live in the companion repository, [aws-bench-datasets](https://github.com/aws-bench/aws-bench-datasets).

## Key concepts

| Term | Meaning |
|------|---------|
| **Management account** | The AWS account where you run the `aws-bench` CLI. It owns an AWS Organization and orchestrates everything. It holds **no** test resources itself. |
| **Test environment** | The collection of test accounts (an Organizational Unit under your management account) that aws-bench creates. Created with `env init`, removed with `env terminate`. |
| **Test account** | A member account in the test environment. Each hosts one scenario's resources; the agent operates here, never on the management account. |
| **Scenario** | The real AWS environment a task runs against (CDK stacks + setup scripts). Maps to a dedicated test account. |
| **Task** | A single problem the agent must solve against a scenario — an instruction, a verifier, and a reference solution. |
| **Dataset** | A versioned bundle of tasks + scenarios, addressed as `name@version` and referenced with `-d`. |
| **Trial** | One agent attempt at one task. Produces a reward: `1.0` (pass) or `0.0` (fail). |
| **Job** | One benchmark run — a collection of trials, with aggregated results. |

## How it works

Using aws-bench follows three steps, plus an optional teardown:

```
aws-bench env init   →   aws-bench env setup   →   aws-bench run   →   aws-bench env cleanup
(provision accounts)     (deploy resources)        (evaluate agent)     (optional teardown)
```

A run does **not** tear anything down. When `aws-bench run` finishes, aws-bench automatically **resets the environment to its clean, post-setup state**, leaving it ready for the next run — so you can simply run `aws-bench run` again without any manual steps in between. Teardown is a deliberate, separate step:

- **`env cleanup`** — run this when you want to stop incurring costs for the deployed resources. It removes the deployed infrastructure but **keeps the test accounts**. To benchmark again later, re-run `env init` followed by `env setup` — a full deploy that takes about as long as the first one.
- **`env terminate`** — optionally, you can fully close the test accounts with this command. Closed accounts are first **suspended for 90 days** (during which no further costs accrue) and then permanently closed at the end of that period, per [AWS Organizations account closure](https://docs.aws.amazon.com/organizations/latest/userguide/orgs_manage_accounts_close.html).

> **Cost & safety:** aws-bench creates real AWS resources in real accounts, which incur charges while they exist. Running `env cleanup` when you finish a benchmarking session is enough to stop those charges — it removes the deployed resources (the emptied accounts are kept, ready for a future session). `env terminate` is only needed if you also want to close the accounts entirely.

| Phase | Command | What happens | Typical duration |
|-------|---------|--------------|------------------|
| **Init** | `aws-bench env init` | Creates the Organization, OU, and test accounts; submits service-quota requests | ≤ 5 min (up to ~60 min if you opt to wait for quota approvals) |
| **Setup** | `aws-bench env setup` | Builds scenario containers and deploys CDK stacks into the test accounts | 10–30 min |
| **Run** | `aws-bench run` | Executes agent trials, runs verifiers, collects rewards | 30 min to ~6 h, depending on the size and complexity of the selected dataset |
| **Cleanup** | `aws-bench env cleanup` | Removes deployed resources (keeps the accounts) | 5–180 min, depending on the resources deployed |
| **Terminate** | `aws-bench env terminate` | Removes resources **and** closes the test accounts | ≤ 10 min |

All env commands are **idempotent** — safe to re-run if interrupted.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **OS** | macOS or Linux |
| **Python** | 3.12+ |
| **Package manager** | [uv](https://docs.astral.sh/uv/getting-started/installation/) |
| **Container runtime** | Docker with the Compose v2 plugin **and** buildx ≥ 0.17.0 (see [appendix](#appendix-docker-compose-v2-buildx)) |
| **AWS account** | An account you control with permission to create an AWS Organization and member accounts. aws-bench provisions **disposable** test accounts under it. |

## Configure AWS access

aws-bench uses your default AWS credential chain. Configure a profile for your **management account** with permission to manage the Organization, then export it:

```bash
# Option A — static access keys
aws configure --profile my-awsbench-profile

# Option B — IAM Identity Center (SSO)
aws configure sso --profile my-awsbench-profile   # one-time setup
aws sso login --profile my-awsbench-profile        # refresh the session later

export AWS_PROFILE=my-awsbench-profile
aws sts get-caller-identity                        # verify you're in the right account
```

## Install aws-bench

```bash
git clone https://github.com/aws-bench/aws-bench.git && cd aws-bench
uv sync
uv run aws-bench --help   # verify the CLI runs
```

## Your first run: the quickstart dataset

The `aws-bench-quickstart` dataset is a tiny, single-scenario dataset (9 tasks) designed to confirm your installation and AWS setup are working before you commit to a longer run. Use it as an end-to-end smoke test.

```bash
# 1. Provision the test environment (Organization, OU, accounts, quotas)
uv run aws-bench env init \
  --env-name awsbench-env \
  -d aws-bench-quickstart \
  --wait-for-quotas

# 2. Deploy the scenario's resources
uv run aws-bench env setup \
  --env-name awsbench-env \
  -d aws-bench-quickstart

# 3. Generate a Bedrock bearer token for the verifier and agent,
#    (if your agent uses Bedrock for LLM inference).
#    First ensure no stale AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN
#    are set in your shell, as they would override AWS_PROFILE and mint the token 
#    against the wrong account (see the Bedrock appendix).
eval $(uv run aws-bench env creds --eval)

# 4. Run the benchmark
uv run aws-bench run \
  --env-name awsbench-env \
  -d aws-bench-quickstart \
  -a claude-code \
  -m global.anthropic.claude-sonnet-5 \
  --ve AWS_BEARER_TOKEN_BEDROCK=$AWS_BEARER_TOKEN_BEDROCK \
  --yes

# 5. Tear down the deployed resources when you're done
uv run aws-bench env cleanup --env-name awsbench-env -d aws-bench-quickstart
```

If step 4 produces per-trial rewards (see [Checking results](#checking-results)), your environment is configured correctly.

> **Note:** `env init` can take a while the first time because service-quota increases may need approval. `--wait-for-quotas` blocks until they're ready; without it, check status later with `aws-bench env show`.

### Example agent and model IDs

Pass these to `-a` / `-m` (use a model your account can access):

| `-a` (agent) | `-m` (model) |
|--------------|--------------|
| `claude-code` | `global.anthropic.claude-sonnet-5` |
| `kiro-cli` | `global.anthropic.claude-sonnet-5` |
| `oracle` | *(none — replays the reference solution; useful for validating a scenario)* |

Model IDs follow the provider's naming (the examples above are Amazon Bedrock model IDs). For the full list of agents, how each authenticates to a model provider, and the available datasets, see [Supported agents, models, and datasets](#supported-agents-models-and-datasets) below.

## Supported agents, models, and datasets

This section enumerates what you can pass to `-a` (agent), `-m` (model), and `-d` (dataset).

### Agents

aws-bench supports agents at two tiers: those with an **aws-bench-specific adapter** (enumerated below) and any **native Harbor agent** resolvable by name (see the note after the table).

The agents below have an aws-bench-specific adapter that integrates them with the benchmark — routing them to a model provider such as Amazon Bedrock, wiring them into Harbor (as with `kiro-cli`), or handling trajectory extraction and per-trial setup. Pass one with `-a <name>`.

| `-a` (agent) | Model provider(s) | How it authenticates | Notes |
|--------------|-------------------|----------------------|-------|
| `claude-code` | Anthropic API or Amazon Bedrock | `ANTHROPIC_API_KEY`, or Bedrock auto-detected from a non-empty `AWS_BEARER_TOKEN_BEDROCK` | Can install Claude Code plugins (each bundling MCP servers + skills) per trial — e.g. `--ak marketplaces='["owner/repo"]' --ak plugins='["name@owner/repo"]'`. |
| `codex` | OpenAI or Amazon Bedrock | `OPENAI_API_KEY`, or Bedrock auto-detected from a non-empty `AWS_BEARER_TOKEN_BEDROCK` | |
| `kiro-cli` | Kiro | `KIRO_API_KEY` (`ksk_…`) exported on the host | |
| `mini-swe-agent` | Any LiteLLM provider (incl. Amazon Bedrock) | Provider-specific; Bedrock uses `bedrock/<model-id>` and `AWS_BEARER_TOKEN_BEDROCK` | |
| `aws-bench-baseline-agent` | Amazon Bedrock (Strands Agent SDK) | `AWS_BEARER_TOKEN_BEDROCK` | aws-bench's baseline evaluation agent (see note below). |
| `oracle` | *(none)* | *(none)* | Replays a task's reference solution (`solution/solve.sh`) instead of calling a model. Reference solutions are provided for **mutation** tasks, so the oracle validates that a mutation scenario and its verifier work end-to-end. |

> **About the baseline agent:** `aws-bench-baseline-agent` is a minimal reference agent — a Strands Agent SDK loop over a Bedrock model, a purpose-built AWS-specific system prompt, three built-in tools (guarded `bash`, `read_file`, `write_file`), and optional MCP servers.

> **Other agents:** aws-bench is built on [Harbor](https://github.com/harbor-framework/harbor), and any agent Harbor ships (e.g. `aider`, `gemini-cli`, `goose`, `opencode`, `cline-cli`) is also resolvable via `-a <name>`. You can also point at your own agent class with `--agent-import-path module.path:ClassName` — see Harbor's [Integrating your own agent](https://www.harborframework.com/docs/agents#integrating-your-own-agent) guide.

### Models

There is **no fixed list of supported models** — `-m` accepts any model ID your chosen agent's provider understands, and each agent routes to its provider differently (see the table above). Two rules of thumb:

- **The model ID follows the provider's own naming.** For Amazon Bedrock, that's an ID like `global.anthropic.claude-sonnet-5` (or `bedrock/us.anthropic.claude-sonnet-5` for LiteLLM-based agents such as `mini-swe-agent`).
- **You must have access to the model.** On Bedrock, enable model access in the console first — see [Appendix: Amazon Bedrock model access](#appendix-amazon-bedrock-model-access).

`-m` can be repeated to evaluate several models in one run (e.g. `-m <model-a> -m <model-b>`).

### Datasets

Reference a dataset with `-d <name>` (or `-d <name>@<version>`). **Omit the version to get the latest** — pinning `@<version>` is only needed to reproduce an older run, and pinned versions go stale as datasets are updated.

**Curated datasets:**

| `-d` (dataset) | Tasks | Scenarios | Description |
|----------------|-------|-----------|-------------|
| `aws-bench-quickstart` | 9 | `ec2-multiregion` | Smoke test (used above). |
| `aws-bench-basic` | 78 | `compute-and-data`, `databases-and-storage`, `serverless-apps`, `streaming-and-iot` | Entry-level tasks across core AWS services. |
| `aws-bench-advanced` | 47 | `api-and-observability`, `reference-architectures`, `troubleshooting-multiservice` | Full-difficulty, multi-service scenarios. |

**Per-scenario datasets** — every scenario is also runnable as its own dataset, so you can work one environment at a time:

`api-and-observability` · `compute-and-data` · `databases-and-storage` · `ec2-multiregion` · `reference-architectures` · `serverless-apps` · `streaming-and-iot` · `troubleshooting-multiservice`

The same `-d` value must be used across `env init`, `env setup`, `run`, and `env cleanup`. See [Running the full datasets](#running-the-full-datasets) for the full workflow.

## Running the full datasets

Once the quickstart works, run one of the larger datasets ([Datasets](#datasets) lists them all) by swapping the `-d` value across `env init`, `env setup`, `run`, and `env cleanup` — for example `aws-bench-basic` (78 tasks) or `aws-bench-advanced` (47 tasks).

Because **every scenario is also runnable as its own dedicated dataset**, you can work incrementally — one environment at a time — instead of provisioning the full suite. For example:

```bash
uv run aws-bench env init  --env-name awsbench-env -d ec2-multiregion --wait-for-quotas
uv run aws-bench env setup --env-name awsbench-env -d ec2-multiregion
uv run aws-bench run       --env-name awsbench-env -d ec2-multiregion -a claude-code -m <model-id> --yes
```

## Command reference

### Environment lifecycle

```bash
aws-bench env init       # Provision the Organization, OU, accounts, and quota requests
aws-bench env setup      # Build scenario containers and deploy CDK stacks
aws-bench env show       # Show current environment state
aws-bench env creds      # Generate a Bedrock bearer token (use with --eval; optional)
aws-bench env verify     # Check deployed resources for drift
aws-bench env reset      # Reset the environment to its post-setup state
aws-bench env cleanup    # Tear down deployed resources (keeps accounts)
aws-bench env terminate  # Tear down resources and close the test accounts
```

- **`env verify`** *(optional)* — confirms the deployed environment still matches what `setup` produced. Useful before a run if an environment has been idle.
- **`env reset`** *(optional)* — manually returns a scenario to its clean, post-setup state. aws-bench already does this automatically at the end of each run, so you rarely need it — reach for it only to force a clean state (e.g. after an interrupted run).
- **`env cleanup` vs `env terminate`** — `cleanup` removes deployed resources but keeps the accounts; to benchmark again you re-run `env init` then `env setup`. `terminate` also closes the accounts; use it when you're finished with the environment entirely.

### Running benchmarks

```bash
uv run aws-bench run \
  --env-name awsbench-env \
  -d <dataset> \
  -a <agent> \
  -m <model-id> \
  --yes
```

Common flags:

| Flag | Purpose |
|------|---------|
| `--env-name` | Name of the test environment |
| `-d name@version` | Dataset from the registry (omit `@version` for the latest) |
| `-a` / `--agent` | Agent to evaluate (e.g. `claude-code`, `oracle`) |
| `-m` / `--model` | Model ID |
| `--ve KEY=VALUE` | Environment variable passed into the verifier/agent container (e.g. `AWS_BEARER_TOKEN_BEDROCK`) |
| `--yes` | Skip confirmation prompts |
| `--quiet` | Reduce output verbosity |

## Checking results

Results land in `jobs/<timestamp>/` under the package directory. For each trial:

| File | Contents |
|------|----------|
| `agent/agent-output.txt` | The agent's response |
| `verifier/test-stdout.txt` | Verifier / LLM-judge reasoning |
| `verifier/reward.json` | Score: `1.0` (pass) or `0.0` (fail) |
| `verifier/reward-details.json` | Verifier reasoning details |

Aggregated results are in `jobs/<timestamp>/result.json`.

> Some reward variance across repeated runs is expected due to LLM non-determinism — this is normal.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Quota gate blocks `env setup` | Re-run `env init --wait-for-quotas`, or check the AWS Service Quotas console for pending requests |
| `env setup` fails with Docker errors | Ensure the Docker daemon is running (`docker ps`) and the Compose v2 + buildx plugins are installed ([appendix](#appendix-docker-compose-v2-buildx)) |
| `compose build requires buildx 0.17.0 or later` | Upgrade buildx to ≥ 0.17.0 ([appendix](#appendix-docker-compose-v2-buildx)) |
| Bedrock `ResourceNotFoundException` | Model access isn't enabled/propagated — see the [Bedrock appendix](#appendix-amazon-bedrock-model-access) |
| Bedrock bearer token expired | Re-run `eval $(uv run aws-bench env creds --eval)` |


## Environment health and stale state

aws-bench includes dedicated mechanisms to keep your test environment in the correct state between runs — after each run, it automatically resets the environment to the post-setup baseline so the next trial starts clean. In rare cases, however, a resource management gap (bug) may leave behind unexpected state that was not fully cleaned up, which can affect subsequent trial results:

- **False positives** — the agent inherits leftover resources from a prior trial and receives unearned credit.
- **False negatives** — leftover state blocks the agent from completing the task (e.g., a resource name collision, a full quota, or a pre-existing configuration that confuses the agent's investigation).

### How to detect

If you suspect stale state is affecting results, run:

```bash
uv run aws-bench env verify --env-name <env-name> -d <dataset>
```

This checks each test account for drift (resources modified since setup) and new resources (created by a prior agent but not cleaned up). A clean result looks like:

```
Verification passed for 1 of 1 account(s).
```

Any failures indicate leftover state that may impact trials.

### What to do

1. **Reset the environment** (fast — restores to post-setup baseline without redeploying):

   ```bash
   uv run aws-bench env reset --env-name <env-name> -d <dataset> --yes
   ```

2. **If reset fails**, tear down and redeploy (slower but guaranteed clean):

   ```bash
   uv run aws-bench env cleanup --env-name <env-name> -d <dataset> --yes
   uv run aws-bench env setup  --env-name <env-name> -d <dataset>
   ```

3. **Re-run the affected trials** on the now-clean environment.

### How to report

If you encounter a reproducible case where `env verify` fails after a run (indicating the reset didn't fully clean up), please [file an issue](https://github.com/aws-bench/aws-bench/issues) with:

- The dataset and task that was running
- The output of `env verify`
- Whether `env reset` resolved it

We treat environment-contamination bugs as high priority and are committed to resolving them quickly so that benchmark results remain reliable.

> **Note:** This is not expected behavior — aws-bench is designed to fully reset the environment between trials. If you see it, it's a bug we want to fix.

## Appendix: Amazon Bedrock model access

Using Bedrock is **optional** — aws-bench works with any LLM provider your agent supports.

### How credentials work

aws-bench uses model and AWS credentials in **three separate places**, each with its own authentication:

| Component | What it does | How it authenticates | You need to… |
|-----------|-------------|---------------------|--------------|
| **Agent** | Runs the model being evaluated (e.g., Claude Sonnet) | Bearer token (`AWS_BEARER_TOKEN_BEDROCK`), only if it runs on Bedrock | Generate a token with `aws-bench env creds` and pass it via `--ve` (skip if the agent uses another provider) |
| **Verifier — validation script** (mutation tasks) | Checks live AWS state in the test account | Framework-provided IAM role (`OrganizationAccountAccessRole` by default), automatic | Nothing, as the framework injects these credentials |
| **Verifier — LLM judge** (read-only / diagnosis tasks) | Scores the agent's answer with an Anthropic model | A model-provider credential you pass via `--ve`, the framework IAM role can't reach a model provider. **Recommended:** Bedrock bearer token (`AWS_BEARER_TOKEN_BEDROCK`); or `ANTHROPIC_API_KEY` | Pass the credential via `--ve` (any LiteLLM-supported provider of the LLM judge model works) |

The bearer token is a long-lived (30-day) service-specific credential for `bedrock.amazonaws.com`. It's cached in SSM Parameter Store; subsequent `env creds` calls reuse it without creating new credentials. Regenerate with `--force` if expired.

> **Before generating the token, ensure no stale `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` are exported in your shell.** `env creds` uses the default AWS credential chain, in which these environment variables **take precedence over `AWS_PROFILE`** — so leftover keys silently generate the token against the wrong account. Run `unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN` (and confirm with `aws sts get-caller-identity`) so the token is minted against your management account.

### Running with Bedrock (agent uses Bedrock models)

Amazon Bedrock hosts models from multiple providers. aws-bench supports evaluating agents against any of them using a single bearer token:

| Agent (`-a`) | Provider | Bedrock model IDs (`-m`) |
|--------------|----------|--------------------------|
| `claude-code` | Anthropic | `global.anthropic.claude-sonnet-5`, `global.anthropic.claude-sonnet-4-6` |
| `codex` | OpenAI | `openai.gpt-5.6-sol`, `openai.gpt-5.6-terra`, `openai.gpt-5.6-luna` |
| `aws-bench-baseline-agent` | Anthropic | `us.anthropic.claude-sonnet-4-6` |

```bash
# Generate the bearer token (one-time, valid ~30 days)
eval $(uv run aws-bench env creds --eval)
```

#### Claude Code (Anthropic models)

```bash
uv run aws-bench run \
  --env-name awsbench-env \
  -d <dataset> \
  -a claude-code \
  -m global.anthropic.claude-sonnet-5 \
  --ve "AWS_BEARER_TOKEN_BEDROCK=$AWS_BEARER_TOKEN_BEDROCK" \
  --yes
```

#### Codex (OpenAI models on Bedrock)

```bash
uv run aws-bench run \
  --env-name awsbench-env \
  -d <dataset> \
  -a codex \
  -m openai.gpt-5.6-terra \
  --ve "AWS_BEARER_TOKEN_BEDROCK=$AWS_BEARER_TOKEN_BEDROCK" \
  --yes
```

The GPT-5.6 model family on Bedrock:
- `openai.gpt-5.6-sol` — frontier reasoning (best quality, higher latency)
- `openai.gpt-5.6-terra` — balanced production (recommended starting point)
- `openai.gpt-5.6-luna` — fast and cost-efficient

Both agents auto-detect Bedrock from the presence of `AWS_BEARER_TOKEN_BEDROCK` in the host environment (set by `eval` above). The `--ve` flag passes the token to the verifier container for the LLM judge.

### Running without Bedrock (agent uses another provider)

Skip `env creds` entirely and omit `--ve AWS_BEARER_TOKEN_BEDROCK`. Supply your agent's credentials through its own mechanism:

```bash
uv run aws-bench run \
  --env-name awsbench-env \
  -d <dataset> \
  -a <your-agent> \
  -m <your-model-id> \
  --yes
```

> **Note:** If your dataset includes read-only / diagnosis tasks, the verifier's LLM judge still needs a model credential for Anthropic models (supported by LiteLLM), as the framework-provided IAM role can't reach a model provider. Pass one via `--ve` (recommended: `AWS_BEARER_TOKEN_BEDROCK` from `env creds`; or `ANTHROPIC_API_KEY`), independent of which provider your agent uses. Mutation-task validation scripts need nothing extra, they run on credentials injected by the framework.

### Overriding the judge model

By default, the LLM judge uses the model specified in each task's `tests/judge.toml`. To override it for a run — for example, to evaluate judge consistency across models or to use a cheaper model during development — pass `REWARDKIT_JUDGE` via `--ve`:

```bash
uv run aws-bench run \
  --env-name awsbench-env \
  -d <dataset> \
  -a claude-code \
  -m global.anthropic.claude-sonnet-5 \
  --ve "AWS_BEARER_TOKEN_BEDROCK=$AWS_BEARER_TOKEN_BEDROCK" \
  --ve "REWARDKIT_JUDGE=bedrock/us.anthropic.claude-sonnet-4-6" \
  --yes
```

`REWARDKIT_JUDGE` accepts any [LiteLLM-supported model identifier](https://docs.litellm.ai/docs/providers). It takes priority over `judge.toml`. Mutation tasks (which use programmatic `check.py` verifiers) ignore this variable.

### Enabling model access

If you haven't already, request access to the models you plan to use in the [Amazon Bedrock console](https://console.aws.amazon.com/bedrock/home#/modelaccess) on your management account. Access typically propagates within minutes.

## Appendix: Docker Compose v2 + buildx

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

---

Ready to go deeper? See the [README](https://github.com/aws-bench/aws-bench#readme) for an overview, or the [aws-bench-datasets](https://github.com/aws-bench/aws-bench-datasets) repository for the tasks and scenarios themselves.
