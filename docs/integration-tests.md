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
| 2 | `env setup` deploys the scenario (CDK) in `us-east-1`, `us-west-1`, `us-west-2` | scenario `environment.build_timeout_sec` + `deploy.timeout_sec` + 30 min (70 min for `ec2-multiregion`) |
| 3 | `run -a oracle`, with pre-run environment verification on | sum over the selected tasks of `environment.build_timeout_sec` + `agent.timeout_sec` + `verifier.timeout_sec` (+ `pre_invoke`/`post_invoke` `timeout_sec`) + 30 min (340 min for the nine quickstart tasks) |
| 4 | Asserts the job planned the expected number of trials, and every trial completed, without errors, with reward `1.0` | |
| 5 | `env cleanup` is attempted once, also when init, setup or the test failed (the standalone test in `test_env_cleanup.py` runs its own deployment and cleanup, see below) | scenario `environment.build_timeout_sec` + `cleanup.timeout_sec` + 30 min (130 min for `ec2-multiregion`); no deadline if no `scenario.toml` can be read |

The derived timeouts are built from the budgets the scenario and tasks
declare:

- **Setup and cleanup** come from `scenario.toml`, so the test does not stop
  either step while it is inside those budgets. Cleanup includes the build
  budget because `env cleanup` builds the scenario container first. With
  `AWS_BENCH_INTEG_DATASET_PATH` the file is
  `<checkout>/scenarios/<scenario>/scenario.toml`. With a registry dataset it
  is the copy that `env init` caches in the scratch `HOME`. With several
  scenarios the budgets are summed. If no manifest can be read, setup falls
  back to 70 min, and cleanup runs with no deadline from the test at all,
  because any fixed guess could stop a custom scenario's cleanup inside its own
  budget and leave resources behind. The CLI bounds only its container build
  and cleanup-phase script; the rest of `env cleanup` has no declared budget, so
  a hung cleanup needs a manual interrupt.
- **`run`** comes from each selected task's `task.toml`, summed as if the
  trials ran one after another. The phase budgets count when the task declares
  `[pre_invoke]`/`[post_invoke]` or ships that script directory. A task whose
  `task.toml` cannot be read, or has no agent timeout, counts 60 min. With no
  task directories at all the test uses 360 min. The scenario reset that
  follows each mutating trial, and any environment healthcheck, are not
  derived from their budgets; only the 30 min margin covers them.
- The budgets are not scaled: the test passes no timeout multiplier.

The test prints the values it used, including when cleanup has no deadline. On a timeout the CLI gets `SIGTERM`, so
its own shutdown can stop containers and save results, and `SIGKILL` 60 s
later if it is still running.

The test counts the trials it expects with aws-bench's own dataset resolver,
using the same source arguments and `-i` globs that it passes to `run` (one
trial per task; the test passes no `-k`), and fails if the job planned a
different number. Without `AWS_BENCH_INTEG_EXPECTED_TRIALS`, this confirms
that the job ran every task the selection resolves, but it cannot detect a
selection that is itself too small: a glob, dataset or registry that resolves
fewer tasks than intended lowers both counts and still passes. With
`AWS_BENCH_INTEG_EXPECTED_TRIALS=N` it can: the test fails before `run` if
the selection does not resolve exactly `N` trials, and afterwards requires
the job to have planned `N`.

`scripts/check_job_rewards.py` checks the rewards per trial, using the trial
names in `result.json`, and per eval. It fails if the job has no trials or a
different number than expected (`--expected-trials N`), if any trial raised an
exception, if fewer trials completed than planned, if any planned trial has no
reward, if an eval's trial count does not match the trials it lists, or if any
reward value of any trial is not `1.0`. Trials may carry different reward
keys, also within one eval; each trial is checked against the rewards it has,
and a trial with several reward keys counts once. Malformed, negative or empty
statistics also count as a failure, including an `exception_stats` entry that
is not a list. The script's
docstring lists every check, and `tests/scripts/test_check_job_rewards.py` has
failing inputs for them. You can also run it on its own:
`python scripts/check_job_rewards.py <jobs-dir>/<job-name> [--expected-trials N]`.

The test's own failure messages and CLI results mask secrets: the values of
`--ve`/`--ae` assignments, of any `KEY=VALUE` argument whose key looks secret,
of arguments passed as `Secret`, and of environment variables whose names look
secret, wherever they appear in the captured output. Values of secret-named
environment variables shorter than 8 characters are not scrubbed, so that
short flag values such as `1` or `true` do not mangle ordinary output; a real
secret that short would appear unmasked. `tests/test_integration_helpers.py` checks the full pytest report
(also with `-l` and `--full-trace`) for a canary on the non-zero exit,
timeout, unexpected-exception, environment-only-secret and
setup-plus-cleanup-failure paths. This covers the test's output only. What the
CLI itself writes to its logs under the scratch `HOME` is not checked here.

**Account IDs in output.** Account IDs are masked in the suite's own failure
messages, problem lists, status prints and object reprs, and in the pytest
report of the integration tests (listed below). Each ID appears as
`********<last 4 digits>`. Any 12-digit run is masked, wherever it comes from:
a stack, scenario, environment or trial name, a path, an ARN, an error message,
the account config or the CLI's output. Other digit runs are left alone, and
the masked text keeps its length and line layout. Accounts that share their
last four digits render identically, so compare the raw values in a check and
say so in the message when two IDs are compared.

The suite masks what it builds, where it builds it, because pytest repeats what
it is given:
- A CLI failure (non-zero exit, timeout) reports the command line and the last
  4000 characters of stdout and stderr. The whole text is masked first and cut
  afterwards, so a cut cannot expose the last digits of an ID. Secrets are
  scrubbed first, then IDs are masked.
- `CliResult.stdout` and `stderr` keep the raw text, because the `env show`
  parser reads the account ID from the panel title. Messages use
  `CliResult.masked_output()`, the cleanup summary check reads the masked
  stdout, and the `repr` of a `CliResult` is masked. The fields are `MaskedStr`
  values, a `str` whose `repr` masks IDs, as are the argument and the lines the
  `env show` and `env list` parsers work on. pytest cuts a `repr` longer than
  240 characters in the middle, which can leave a few digits of an ID that no
  mask matches; the `MaskedStr` `repr` is masked before that cut.
- The problem lists of the `env show`, `env list`, stack, reward and cleanup
  checks are masked when they are built. The tests compute the list before they
  assert on it (`problems == []`, never `f(raw_output) == []`), so pytest's
  assertion explanation shows only masked text and not the arguments of a call.
- The leak notice of a failed cleanup, the messages and skip reasons of the
  settings checks, a failure to load the account config (a validation error
  quotes the value it rejected), and the `step timeouts:`, `expected trials:`
  and `env show:` lines are masked too.
- Objects that hold raw names or output mask their own `repr`, which is what
  pytest prints for fixtures, frame arguments (`--tb=long`, `-vvv`) and locals
  (`-l`): the parsed `env show` and `env list` results, stack records,
  `IntegSettings`, the timeouts, the CLI argument lists and the account mapping.
- The CLI's environment (`build_cli_env`, `cli_display_env`) is a `SecretDict`,
  the `cli_env` fixture a `SecretEnv`, and `integ_settings` passes the process
  environment to `load_settings` as a `SecretEnv`. Their `repr` shows no
  values, so frame arguments and `-l` locals of these frames show no
  credentials.

What the suite does not build is masked when pytest builds its report. A
`pytest_runtest_makereport` hook in `tests/integration/conftest.py` masks, for
every phase (setup, call, teardown) of the integration tests: the traceback
text, including frame arguments, locals (`-l`), `--full-trace` frames and the
chain of exceptions with their causes and notes; the one-line message of the
short test summary and of the `--junitxml` file; the captured stdout, stderr
and log sections; and skip and xfail reasons. This is what covers an exception
raised by code the tests call, such as the dataset resolver or the credential
provider. The hook renders the traceback to text first, so the terminal shows
it without pytest's colour highlighting. It renders it the way
`report.longreprtext` does, but without the final `strip()`, so the blank line
and the indentation of the first source line stay as pytest prints them (a test
compares the text of a report without an ID, before and after the hook). It
depends on the shape of the report and on pytest internals, including the
private `ReprFileLocation` class, so a pytest upgrade can break it; a test
fails if it stops running.

`tests/test_integration_helpers.py` checks this with a canary ID in a nested
pytest:
- The real acceptance and cleanup test modules run against a fake CLI, with the
  canary in a stack, scenario, environment and trial name, a path, an ARN in the
  CLI's output and the account config, at `--tb=long -vvv -rA`, with and
  without `-l --full-trace`.
- The called code is made to fail: a dataset resolver that raises with the
  canary (also with a cause and a note), a credential provider that raises, a
  log record, a failing fixture (the account config sits in a directory named after the
  canary), settings and config failures, a skip reason, and the other `--tb`
  styles (`short`, `native`, `line`, `no`). In each case neither the terminal
  output nor the junit XML contains the canary or any six consecutive digits of
  it, nor the credentials set in the session's environment (including the
  fixture failure at `-l --full-trace`), and the masked text is in the terminal
  output.
- A frame argument that is a raw CLI output of about 500 characters with the
  canary every 13 characters (so every cut falls inside an ID), passed to the
  real `env show` parser, which then fails: no run of six or more digits from
  the canary is shown, at the default verbosity, at `-v` and with `-l`.
- A probe file checks the setup, call and teardown reports of a failure, a
  setup error, a teardown error, a skip, an xfail, an XPASS and a strict XPASS:
  the traceback, short summary, stdout, stderr and log sections, skip and xfail
  reasons and the junit XML, and the outcome counts of the probe.
- In-process: the terminal text of a report without an ID is unchanged by the
  hook; a report with an ID is masked in its terminal text, its `str` (junit
  XML) and its `reprcrash` message. In a nested session, a `conftest.py` that
  exposes the hook masks the tests below its directory and not those outside
  it.
- `-s` with the suite's own prints shows no canary.
- The nested sessions do not inherit `PYTEST_*` or `AWS_BENCH_INTEG*`
  variables.

Not covered:
- Files the CLI writes: the job directory (`result.json`) and the CLI's logs
  under the scratch `HOME` hold IDs as they are. The CLI's logs are also not
  checked for secrets.
- `scripts/check_job_rewards.py` run on its own prints trial names and paths as
  they are. The suite masks the problems it gets from that script.
- Output that pytest prints outside the report. A probe with an ID-bearing
  value showed these channels unmasked; no test pins them: the warnings summary,
  collection errors, live `log_cli` logging, test node IDs (parametrize IDs,
  also in the junit `testcase` name), and anything that called code writes to
  the terminal with `-s`.
- An ID split by colour codes or inside a longer run of digits.
- Truncation of long raw strings from called code. pytest cuts the `repr` of a
  frame argument or local longer than 240 characters in the middle before the
  hook sees it, so a few digits of an ID can remain when a long raw string that
  the suite did not wrap is a frame argument or local at a verbosity below
  `-vvv`. A probe with a 450-character argument showed leaked fragments at
  some offsets at the default verbosity, `-v`, `-vv` and `-l`, and none at
  `-vvv`. No test pins this. The CLI output in `CliResult` and the parsers'
  inputs are `MaskedStr`, and the case above pins the `env show` parser.
- Serialized reports. A report that is serialized (pytest-xdist, report-log)
  loses `reprcrash`, so pytest has no one-line message for the short summary
  line of that test. The restored text stays masked. An in-process test pins
  this restored behaviour. xdist is not used today.
- Colour. The hook's traceback text has no colour highlighting.
- Other pytest versions. The hook uses pytest's private `ReprFileLocation` and
  `TerminalWriter` classes, and its tests run only against the locked pytest
  version.

If cleanup fails, how it is reported depends on what failed before it:

- **Init or setup failed**: that failure is reported as
  the error at setup of the test, with the cleanup failure appended to its
  message (`Also, env cleanup failed (resources may be left behind): ...`).
- **The test body failed**: pytest reports the test as `FAILED` with its own
  message, and the cleanup failure separately, as an `ERROR at teardown` with
  `env cleanup failed (resources may be left behind): ...`.
- **Nothing else failed**: the cleanup failure is reported as an
  `ERROR at teardown` with the same notice.

`tests/test_integration_helpers.py` unit-tests the `deployed` context manager
behind this with a stubbed CLI, including exceptions raised inside its block.

### `env show` and `env list`

Two read-only tests check the output of these commands, with `COLUMNS=200`.
`env show` runs with a 10 min timeout (an assumed bound for a few accounts and
their enabled regions) and `env list` with a 2 min timeout (a credential check
and a config read in pre-existing mode); neither is derived or validated live
yet. The test removes `FORCE_COLOR` and `TTY_COMPATIBLE` from the CLI's
environment, because either makes Rich treat a pipe as a terminal, and with
`TERM` dumb or unknown Rich then ignores `COLUMNS` and uses 80 columns.
`PY_COLORS` and `CLICOLOR_FORCE` are removed too, defensively (Rich does not
read them). At 200 columns the unit tests parse `ec2-multiregion`
panels without truncation; a very long env or scenario name can still be
truncated, and the parsers then fail instead of passing. Any colour codes left
are stripped before parsing.

- `test_env_show_happy_path` runs `env show --env-name <name>` after the
  quickstart test, on the same deployment: the `deployed_env` fixture is
  module-scoped, so the module still runs `env init`, `env setup` and
  `env cleanup` once. It requires exit code 0, a header titled with the env
  name whose `Scenario accounts` count equals the number of accounts in the
  config, and exactly one `ACTIVE` panel per configured scenario, account tag
  and account ID. In every configured account, at least one stack named
  `<scenario>-*` (the scenario in the panel title) must be in
  `CREATE_COMPLETE`, `UPDATE_COMPLETE` or `IMPORT_COMPLETE`. Other stacks,
  such as `CDKToolkit` or unrelated stacks in the account, do not count.
  Failed or rolled-back stacks are not checked (drift detection is left to
  the `env verify` test, backlog #5), and neither are stack regions or the
  header's OU row.
  - The env name, account count and account triples are compared with the
    same account config the CLI reads, so they verify rendering and plumbing.
    The stack check is the part tied to real AWS state. `ACTIVE` is a rendering
    guard: pre-existing mode always reports it.
  - The per-account stack check assumes `env setup` deployed the scenario's
    stacks into every configured account. aws-bench does not enforce this
    (each scenario's deploy script decides), but it holds when the account
    config maps only the scenario under test: every scenario in
    aws-bench-datasets has a single account tag, and its CDK app names its
    stacks `<scenario>-<id>-<region>`.
- `test_env_list_happy_path` runs `env list`, which in pre-existing mode needs
  only the config and the caller's credentials, not a deployment. It requires
  exit code 0 and exactly one row: the env name, OU ID `preexisting`, and the
  number of accounts in the config.

### `env cleanup` as the step under test

`test_env_cleanup_happy_path` (in `tests/integration/test_env_cleanup.py`)
runs `env cleanup` as the asserted step and then checks the accounts
themselves. Without it, cleanup only runs as the finalizer of the shared
deployment, and nothing looks at what it leaves behind.

The test deploys its own environment (`env init`, `env setup`), because it
consumes the environment it deploys. It uses the same derived timeouts as the
quickstart test. The work it adds to a run is one more `env init`, `env setup`
and asserted `env cleanup`, plus two read-only stack listings per account (one
before cleanup, one after). Each command runs under its own deadline: for
`ec2-multiregion`, `env init` 20 min, `env setup` up to 70 min and `env cleanup`
up to 130 min. These are upper bounds, not measurements. Run time and the cost
of the stacks have not been measured yet.

The module runs after `test_acceptance.py`. pytest tears down that module's
shared deployment (an `env cleanup` attempt) when the module ends, before the
next module starts, and any cleanup failure is reported at that teardown. The
two modules use the same accounts, so the accounts must be dedicated to these
tests and start clean: the test does not check that they are empty before
`env init`, and a teardown failure in the earlier module leaves stacks that
this module then deploys next to. Both use the env name fixed by the account
config, so do not run the suite in parallel or in random order.

The account config is assumed to map only the scenario under test, as in the
`env show` section above. The expected account count in the summary is the
number of accounts in the whole config, and the stack check reports accounts
of any other scenario because it has no manifest for them.

It asserts, in order:

1. Before cleanup: in every configured account, `cloudformation:ListStacks`
   shows at least one stack named `<scenario>-*` in `CREATE_COMPLETE`,
   `UPDATE_COMPLETE` or `IMPORT_COMPLETE`. So a healthy `<scenario>-*` stack
   exists before cleanup, and an empty account afterwards means it was
   removed. The check does not show that `env setup` created it: the deferred
   emptiness check before `env init` (see the backlog) would establish that.
   An account that was not checked, was checked twice, is not configured,
   was queried over other regions than the manifest declares, or had a failed
   query is reported with the same wording as in step 3. If the check fails,
   the test fails with the reason and still runs the cleanup attempt.
2. `env cleanup --yes` exits 0 and prints
   `Cleanup complete: N account(s) cleaned successfully.` with `N` equal to the
   number of configured accounts, and no `Cleanup completed with errors` line.
3. In every configured account and every region the scenario's
   `scenario.toml` declares, `cloudformation:ListStacks` (all pages, no status
   filter) returns only stacks in `DELETE_COMPLETE`. `DELETE_IN_PROGRESS` and
   `DELETE_FAILED` count as left behind.
4. CDK bootstrap stacks get no exemption. On the asserted path (exit 0 and the
   success summary) the CLI deletes the bootstrap stacks too, so a remaining
   `CDKToolkit` or other stack whose name the product treats as CDK bootstrap
   infrastructure (its `INFRA_PREFIXES`) is reported as a leftover. An
   environment stack that survives and was not deferred makes the command exit
   non-zero and skips infrastructure deletion, so the test fails on step 2.
   The manual check under "Recovering after an interrupted run" is for an
   interrupted run, not for this command.

A stack that the product defers deleting (for example one blocked by
requester-managed network interfaces) keeps its `FAILED` status internally but
is left out of `DeletionSummary.failed`. The command then exits 0 with the
success summary, and `CDKToolkit` is deliberately retained
(`_maybe_delete_infra` in `cleanup/stack_deleter.py`). This is the case where
the asserted success summary coexists with a retained bootstrap stack. The test
reports both, and the deferred environment stack is the real cause of the
failure. This is intended, because the stacks are still in the account. The
safety-net cleanup then runs.

Nested stacks (summaries with a `ParentId`) are listed and counted on purpose.
The product's `StackDeleter.list_stacks` never deletes them, so a surviving
nested stack needs manual deletion.

The check fails, instead of passing, when an account was not checked, when
credentials for an account cannot be obtained, when a region cannot be listed,
or when the scenario's regions cannot be read from its manifest. It uses the
CLI's own credential path (`CredentialProvider`, which assumes the configured
`runner_role`), not a second authentication mechanism.

If a check fails, the test still attempts `env cleanup` once more (see
`deployed` above); it skips that attempt only after every check above passed
and the test body ended normally.

**Region scope.** Only the regions the scenario declares are listed. That
matches the command: `ScenarioTrial._run_cleanup` (`scenario/trial.py`) passes
the manifest's declared regions explicitly to the asserted cleanup, and so does
the pre-cleanup residual sweep. The safety-net retry runs the same command, so
it gets the same list. `CleanupManager._limit_to_baseline_regions`
(`cleanup/manager.py`) intersects that list with the regions of the POST_SETUP
snapshot if one exists, and a missing snapshot leaves the list unchanged. All
enabled regions (`_discover_regions`) are used only when the all-regions option
(`cleanup_all_regions`, off by default) is set.

**Not verified.** Only CloudFormation stacks are checked. Other resource types
(for example AMIs, EBS snapshots, IAM roles, S3 buckets and Cloud Control
resources) are not looked at, and neither are regions the scenario does not
declare (the region-restriction SCP is expected to keep resources out of
them). The baseline snapshots are not checked either. Cleanup runs its
pre-cleanup residual sweep (`_run_pre_cleanup_residual_sweep`), the per-region
resource sweeper (`_cleanup_region_in_phases`), the global leftovers sweep
(`_sweep_global_leftovers`) and the orphan scan (`_scan_orphaned_resources`),
but no trial runs, so they only see resources that setup created. Detecting and
deleting task-created residue stays uncovered; running trials would add that
input. The check also does not show that
cleanup removed stacks that were in the account before `env init`: the product
deletes all stacks in the scenario's regions, so use an account that holds
nothing else. The test module body itself is covered only by a live run, which
has not happened yet; the helpers are unit tested.

**IAM actions the check needs.** The list shows only the permissions the check
relies on, all read-only. It is not a list of every call, and it is not limited
to grants beyond what the CLI needs: `sts:AssumeRole` is the CLI's own
assumption.

| Action | Resource | Why |
|---|---|---|
| `sts:AssumeRole` | the configured `runner_role` of each account | the same assumption the CLI makes |
| `cloudformation:ListStacks` | `*` (it does not support resource-level scoping) | list stacks in each declared region |

The credential path also calls `sts:GetCallerIdentity` (once or twice per
account) before it assumes the role. That call needs no grant. The test itself
calls no write APIs and does not need `ec2:DescribeRegions`, because the
regions come from the manifest.

**Output.** Failure messages name accounts as `<scenario>/<tag>
(********<last 4 digits>)` and report API errors by error code only, never by
message, because messages can contain ARNs and account IDs. Every 12-digit
number in the messages is masked, including the tail of the CLI output that an
assertion shows, the stack and scenario names inside the stack problems, and the
CLI's own failure output when `env cleanup` exits non-zero or `env init` or
`env setup` fails (see "Account IDs in output" above).

The parsers and checks are in `tests/integration/helpers.py`.
`tests/test_integration_helpers.py` tests them on output rendered by the CLI's
own display functions, with and without colour, including inputs that must
fail.

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

- **Use credentials that refresh themselves.** The full suite runs longer than
  one hour: with one quickstart task it took about 84 min, because the
  `env cleanup` test runs its own `env init`, `env setup` and `env cleanup`
  after the quickstart, `env show` and `env list` tests. aws-bench refreshes its
  own sessions into the runner role, but each refresh uses your base credentials. Static
  keys exported into the shell expire mid-run, and `env cleanup` can then leave
  resources behind. Use a self-refreshing source instead, such as:
  - a `credential_process` profile
  - IAM Identity Center
  - an instance role

  Unset any static `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and
  `AWS_SESSION_TOKEN`, which would override your profile.
  With static credentials, run one test module at a time, so that each run
  fits in the credentials' lifetime. `make integration` always runs the whole
  directory, so call pytest directly with the same environment variables:
  ```bash
  uv run pytest tests/integration/test_env_cleanup.py -m integration \
    --no-cov --timeout=0 -p no:cacheprovider -v
  ```
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
An explicit `pytest -m <expr>` overrides the default deselection of the
`integration` marker. Nothing is deployed unless `AWS_BENCH_INTEG=1` and the
account config are set, but the result is not all skips: the Bedrock token is
checked first. For example, `pytest -m integration tests/integration` without
the opt-in variables reports `7 skipped, 1 error` (the error is
`AWS_BEARER_TOKEN_BEDROCK is not set`) if no token is set, and `8 skipped` if
one is.

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `AWS_BENCH_INTEG` | yes | Must be `1` to opt in |
| `AWS_BENCH_INTEG_ACCOUNT_CONFIG` | yes | Path to the account config YAML (section 2) |
| `AWS_BEARER_TOKEN_BEDROCK` | yes | Bedrock token for the LLM judge (section 3). The quickstart test, and the tests that share its deployment, fail before deploying anything without it. |
| `AWS_PROFILE` (or other standard AWS variables) | yes | Base credentials that can assume the runner role |
| `AWS_BENCH_INTEG_DATASET` | no | Registry dataset (default `aws-bench-quickstart`) |
| `AWS_BENCH_INTEG_DATASET_PATH` | no | Path to a local `aws-bench-datasets` checkout, used instead of the registry. Use it to test unreleased dataset changes. Cannot be combined with `AWS_BENCH_INTEG_DATASET`. |
| `AWS_BENCH_INTEG_SCENARIO` | no | Scenario to use with `AWS_BENCH_INTEG_DATASET_PATH` (default `ec2-multiregion`). The account config must map it. |
| `AWS_BENCH_INTEG_TASKS` | no | Comma-separated `run -i` globs to run a subset of tasks, e.g. `'*list-ec-instances-all-regions'`. Start each glob with `*`, so it matches both directory and registry task names. |
| `AWS_BENCH_INTEG_EXPECTED_TRIALS` | no | Positive integer: the number of trials the run must plan, e.g. `9` for the full quickstart dataset or `1` with a single-task glob. The test fails before `run` if the selection resolves a different number. Without it, a selection that is itself too small is not detected. Any other value fails the test at startup. |
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

Afterwards, check that no scenario stacks remain in the three regions. After an
interrupted run (not after a clean `env cleanup`, which also deletes the
bootstrap stacks) only `CDKToolkit` is expected, if a bootstrap ran. Also check for leftover
AMIs and EBS snapshots in `us-east-1`.

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

The quickstart test has been run live, in a pre-existing-mode test account, at
commit `5804afc`. All nine `aws-bench-quickstart` tasks completed with reward
`1.0`, with pre-run environment verification on, and `env cleanup` finished
successfully. It was run live again, with one task, at commit `ae7b194`:
reward `1.0`, verification on, and a clean `env cleanup`. The `env show` and
`env list` tests were run live when they were added, together with the quickstart
test and one task: all three tests passed, and `env cleanup` finished
successfully. The whole suite then passed in one run with one task: `4 passed, 4 skipped`
in about 84 min. That includes `test_env_cleanup_happy_path`: before cleanup,
every configured account held a healthy `<scenario>-*` stack, and afterwards
every stack in the scenario's regions was `DELETE_COMPLETE`, the CDK bootstrap
stacks included. The run does not record how long each command took, so the
`env show` and `env list` timeouts are still assumed, not derived. That run,
about 84 min, used a self-refreshing credential source (a `credential_process`
profile), so a run longer than one hour with such a source is validated.
