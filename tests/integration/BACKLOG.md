# Integration test backlog

End-to-end acceptance tests, organised per command, covering happy paths only
(behaviour and outputs as expected in normal usage). Run manually with
`make integration`; see `docs/integration-tests.md`.

## Prioritised commands

Importance = how much a regression hurts users. Ease = cost/time/account risk of testing it.

| # | Command / feature | Importance | Ease | Status |
|---|---|---|---|---|
| 1 | `env init` + `env setup` + `run -d aws-bench-quickstart -a oracle` + `env cleanup` | High | Medium | Implemented (`test_quickstart_oracle_earns_full_reward`); passed live at `5804afc` with all 9 tasks, verification on (later changes unit-tested only) |
| 2 | `env show`, `env list` (read-only, run against the env from #1) | Medium | High | Placeholder |
| 3 | `env creds` (mints a Bedrock token; uses 1 of 2 IAM slots) | Medium | Medium | Placeholder |
| 4 | `env cleanup` as a standalone assertion (no stacks remain) | High | Medium | Placeholder |
| 5 | `env verify` / `env reset` (drift detection and baseline restore) | High | Medium | Not started |
| 6 | `env snapshot` | Low | High | Not started |
| 7 | `run` with a non-oracle agent (needs model credentials, non-deterministic) | High | Low | Not started |
| 8 | Provisioning mode `env init` / `env terminate` (creates and closes real accounts; 90-day suspension) | High | Low | Placeholder |
| 9 | `job` subcommands, `view`, `metrics` | Low | High | Not started |

Suggested order: 1 is done. Next 2 and 4 (cheap), then 5, 3, 7, 8.

Also pending: PR automation (an approval-gated GitHub workflow), a live re-run of
the current suite, and validating a run longer than one hour with
self-refreshing credentials.

## Isolation approach

- **Unique run IDs.** Each session generates an 8-char ID (`run_id` fixture), used in
  job names. It cannot go in the env name: pre-existing mode requires `--env-name` to
  equal the account config's `name`. `HOME` is redirected for the CLI subprocess (with
  `~/.aws` and `~/.docker` symlinked in), so local state in `~/.aws-bench` never
  collides with a developer's real environments.
- **Pre-existing mode caveat.** The account allowlist maps to fixed AWS accounts, so
  unique env names do not isolate AWS resources: two runs on the same account still
  share stacks. Resource-level isolation therefore needs one of:
  1. one dedicated account (or account set) per concurrent runner, or
  2. serialisation. Locally, only one person should use a given account at a
     time (a shared lock, e.g. an S3 or DynamoDB lock item, is a possible later
     addition). A future CI workflow should queue runs, e.g. with a GitHub
     concurrency group and `cancel-in-progress: false`, because cancelling a run
     would skip cleanup.
- **Always clean up.** The `deployed_env` fixture runs `env cleanup` in a finalizer.
- **No cross-account credentials in the repo.** The account config path and AWS
  credentials come from the caller's environment.

## Known issues

None open.

### Resolved: AMI snapshot race in `ec2-multiregion`

- **Symptom.** `run` failed pre-run verification in us-east-1 at random:
  `New resources detected - AWS::EC2::Snapshot`
  (`Created by CreateImage(i-...) for ami-...`).
- **Cause.** The scenario's `CreateAmi` custom resource calls `ec2:CreateImage`,
  which returns before EC2 creates the AMI's EBS snapshot. The snapshot could
  appear after the post-setup baseline was captured.
- **Fix.** aws-bench/aws-bench-datasets#93 (`1adde7b`): `deploy.sh` waits for
  the AMI to be `available`, and `cleanup.sh` deletes its snapshot.
  aws-bench/aws-bench#152 (`ec7ec56`) re-pinned the registry.
- **Validated.** All 9 quickstart tasks earned full reward with verification on,
  and cleanup finished successfully.
  The temporary `--no-verify-env` default was then removed, so the test always
  verifies the environment.

## Open questions

- How many test accounts are available, and can each PR/runner get its own?
- Should `env terminate` / provisioning mode be tested at all, given the 90-day
  account suspension? Probably only in a disposable organisation.
- Expected wall-clock time and cost per run, and an acceptable budget per PR.
- Should a nightly run on `main` replace per-PR runs once stable?
