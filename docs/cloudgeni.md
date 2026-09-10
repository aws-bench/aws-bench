# Running CloudGeni on AWS-Bench

The CloudGeni adapter delegates each AWS-Bench trial to the `aws-bench` CloudGeni
persona while preserving AWS-Bench's account isolation, reset, verifier, and
scoring lifecycle. The generic control is AWS-Bench's built-in Codex agent using
the same Azure OpenAI endpoint, deployment, `gpt-5.6-sol` model, and medium
reasoning effort.

## Prerequisites

- Complete the normal [AWS-Bench quickstart](getting-started.md), including
  `env init` and `env setup` for `aws-bench-quickstart`.
- Run a CloudGeni deployment containing the AWS-Bench credential lease support,
  with `CLOUDGENI_AWS_BENCH_ENABLED=true` and the `aws-bench` AgentConfig seeded.
- Create an organization API key. Its creator must remain an active organization
  and workspace member because that identity owns and can cancel the agent session.
- Make the CloudGeni API reachable from the Harbor trial containers. For a local
  Docker deployment, use `host.docker.internal` and the worktree's published API
  port.
- Set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and the Azure deployment
  name for the generic Codex control. Read-only tasks also need the normal
  verifier model credential described in the AWS-Bench guide.

Never put a CloudGeni API key directly in a committed command, config, or result
artifact. The examples below use shell variables intentionally.

## CloudGeni run

Use the same dataset, task filter, attempt count, timeout multipliers, and verifier
environment for both arms. Start with `-n 1`; raise concurrency only after the
credential lease and session cleanup fields are green in every trial's
`agent/cloudgeni-state.json`.

```bash
uv run aws-bench run \
  --env-name awsbench-env \
  -d aws-bench-quickstart \
  --agent-import-path aws_bench.agents.cloudgeni:CloudGeniAgent \
  --job-name cloudgeni-quickstart \
  --ae "CLOUDGENI_API_URL=$CLOUDGENI_CONTAINER_API_URL" \
  --ae "CLOUDGENI_HOST_API_URL=$CLOUDGENI_HOST_API_URL" \
  --ae "CLOUDGENI_API_KEY=$CLOUDGENI_API_KEY" \
  --ae "CLOUDGENI_ORGANIZATION_ID=$CLOUDGENI_ORGANIZATION_ID" \
  --ae "CLOUDGENI_WORKSPACE_ID=$CLOUDGENI_WORKSPACE_ID" \
  --ae CLOUDGENI_AWS_BENCH_TIMEOUT_SECONDS=270 \
  --ve "AWS_BEARER_TOKEN_BEDROCK=$AWS_BEARER_TOKEN_BEDROCK" \
  -n 1 \
  --yes
```

The adapter reads AWS-Bench's temporary `ASIA...` profile inside the isolated
trial container, verifies the account with STS, creates a time-limited CloudGeni
credential lease, starts the dedicated persona, copies the final response to the
requested `/logs/agent/agent-output.*` file, cancels/deletes the session, and
hard-purges the encrypted credential row. A host-side `finally` repeats cleanup
if Harbor cancels the in-container process.

## Generic direct-agent control

```bash
uv run aws-bench run \
  --env-name awsbench-env \
  -d aws-bench-quickstart \
  -a codex \
  -m "$AZURE_OPENAI_DEPLOYMENT" \
  --ak reasoning_effort=medium \
  --job-name generic-codex-quickstart \
  --ae "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT" \
  --ae "AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API_KEY" \
  --ve "AWS_BEARER_TOKEN_BEDROCK=$AWS_BEARER_TOKEN_BEDROCK" \
  -n 1 \
  --yes
```

Codex is the useful generic control because a text-only one-shot model call cannot
inspect or mutate the isolated AWS account. This control gives the same model a
generic shell/tool loop without CloudGeni's persona, capability broker, session
runtime, or product context.

## Compare and clean up

```bash
uv run python scripts/compare_runs.py \
  jobs/cloudgeni-quickstart \
  jobs/generic-codex-quickstart \
  --output jobs/quickstart-comparison.md \
  --json-output jobs/quickstart-comparison.json

uv run aws-bench env cleanup \
  --env-name awsbench-env \
  -d aws-bench-quickstart \
  --yes
```

Before accepting a result, confirm every CloudGeni trial state says both
`sessionDeleted: true` and `leasePurged: true`, the two jobs contain the same
task/attempt pairs, and neither job has infrastructure exceptions. Compare pass
rate and paired reward first; latency, tokens, and cost are secondary diagnostics.
