# Framework Development Guide

How to verify code changes and submit a PR for the [aws-bench](https://github.com/aws-bench/aws-bench) framework.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **OS** | macOS or Linux |
| **Python** | 3.12+ |
| **uv** | [Install guide](https://docs.astral.sh/uv/getting-started/installation/) |

## Install Dependencies

You should have the aws-bench source code checked out locally. If not, please refer to the [Getting Started guide](getting-started.md).

From the repository root:

```bash
uv sync
```

Verify the installation:

```bash
uv run aws-bench --help    # should print the CLI usage
```

## Repository Walkthrough

```
aws_bench/
├── cli/                  # CLI commands (env, run, dataset)
├── resource_management/  # Account lifecycle: deploy, cleanup, reset, snapshot, verify
│   ├── ccapi/            #   CloudControl API fallback deleters
│   ├── cleanup/          #   Stack deletion handlers + orchestration
│   ├── deploy/           #   CDK deployment orchestration
│   ├── fastscan/         #   Fast resource-type scan and enumeration
│   ├── reset/            #   Post-task baseline restoration
│   ├── snapshot/         #   Resource state capture at different stages, used to identify new resources created
│   ├── storage/          #   State ledger (DynamoDB/local)
│   └── verify/           #   Resource state verification
├── scenario/             # Scenario trial orchestration (deploy, reset, cleanup lifecycle)
├── task/                 # Task trial orchestration (agent invocation, verifier, hooks)
├── dataset/              # Registry parsing, dataset resolution
├── account_management/   # OU/account provisioning and assignment
├── agents/               # Agent adapters (claude-code, codex, kiro, strands, etc.)
├── metrics/              # Scoring and result aggregation
├── logging/              # Structured logging
└── utils/                # Shared helpers
scripts/                  # Maintenance scripts (registry updates, metrics upload)
tests/                    # Mirrors aws_bench/ structure 1:1
docs/                     # User and contributor documentation
```

### Testing expectations by area

| Area | What to test | How |
|------|-------------|-----|
| `cli/` | Command parsing, argument validation, output format | Unit tests with mocked internals. Testable locally with `make ready`. |
| `resource_management/` | Deletion handlers, reaper logic, verification, state transitions | Unit tests + **live testing against a sample scenario**. Request maintainer live-test in the PR if you cannot run one yourself. |
| `scenario/` | Lifecycle orchestration (deploy/reset/cleanup sequencing) | Unit tests. Behavioral changes likely need a live run. Request maintainer live-test in the PR if you cannot run one yourself. |
| `task/` | Agent invocation, verifier execution, hook sequencing | Unit tests with mocked agents/verifiers. End-to-end validation requires a live scenario. Request maintainer live-test in the PR if you cannot run one yourself. |
| `dataset/` | Registry parsing, version resolution, descriptor validation | Unit tests (offline, deterministic). |
| `account_management/` | OU assignment, account provisioning | Unit tests with moto. Live testing for Organizations changes — request maintainer run. |
| `agents/` | Agent adapter contracts | Request maintainer to run comprehensive tests across multiple scenarios. |
| `metrics/`, `logging/`, `utils/` | Scoring logic, log formatting, helpers | Unit tests. Minimal regression risk. |

## Development Workflow

1. Create a branch from `main` using the format `<type>/<short-description>` (e.g., `feat/timeout-flag`, `fix/empty-exports`, `docs/dev-guide`)
2. Make your code changes and add corresponding unit tests
3. Run the local gate to verify correctness
4. Open a PR describing the issue and how the change fixes/improves the code. If tested live against a dataset, append data and findings.
5. A maintainer reviews, pulls locally, and runs integration tests against live datasets to verify no regressions. We are actively working on adding automatic integration tests as a GitHub workflow.
6. Approve and merge

> **Why is it not required for contributors to run integration tests?** Integration tests deploy real AWS resources and run benchmark scenarios, which incurs AWS costs. We run these on our end to avoid passing that cost to contributors.

## Running the Local Gate

Before submitting a PR, run:

```bash
make ready    # auto-fix lint/format → run the full gate
```

This is the single pre-submit command. It auto-fixes formatting, then runs the full quality gate (lint, format check, type check, unit tests with coverage).

If you prefer to run steps separately:

```bash
make fix           # auto-fix lint + format (mutates files)
make check         # lint + format-check + typecheck + test (no auto-fix)
```

### Individual targets

```bash
make lint          # ruff check
make format-check  # fail on format drift (no mutations)
make typecheck     # pyright
make test          # pytest --cov
```

The coverage threshold is **85%** — the test run fails if overall coverage drops below this.

## Running Specific Tests

```bash
uv run pytest tests/cli/                          # one directory
uv run pytest tests/cli/test_env.py               # one file
uv run pytest tests/cli/test_env.py::test_init    # one test
uv run pytest -k "registry"                       # by name pattern
```

### Unit test characteristics

- All unit tests run **offline** — no AWS credentials or Docker daemon required
- [moto](https://github.com/getmoto/moto) mocks AWS services; pytest-mock handles patching
- Tests are isolated via autouse fixtures (fresh cwd, isolated ledger, reset global state)
- pytest-timeout (30s) catches stuck tests

### Ensuring coverage for new code

The test suite enforces **85% coverage**. When adding new functionality:

- Add tests in the corresponding `tests/` subdirectory (mirrors `aws_bench/`)
- Run `uv run pytest --cov --cov-report=term-missing` to see which lines lack coverage
- The gate fails if overall coverage drops below the threshold

## Opening a PR

1. Ensure `make ready` passes
2. Commit with a [Conventional Commits](https://www.conventionalcommits.org/) message:
   - `feat(cli): add --timeout flag to env setup`
   - `fix(scenario): handle empty CloudFormation exports`
   - `docs: update getting-started guide`
3. Open a PR against `main` with:
   - **What issue this addresses** — describe the problem or improvement
   - **What the change does** — summary of the approach
   - **How it was verified** — which tests were added/run locally

A maintainer will pull the PR locally, run integration tests against live datasets to check for regressions, and merge once satisfied.

## Maintainer: Registry Updates

The `registry.json` file maps dataset names to pinned versions from the [aws-bench-datasets](https://github.com/aws-bench/aws-bench-datasets) repo. This is a **maintainer-only** workflow used when releasing new dataset versions:

```bash
make registry-bump    # regenerate registry.json from the datasets repo
```

This requires a local checkout of `aws-bench-datasets` at `../aws-bench-datasets` (override with `DATASETS_PATH=...`).

Contributors can run this locally, for example to make sure the framework picks up local dataset changes in a private live run. The release of new datasets versions in the registry will be usually driven by the project maintainers.

The same regeneration runs in GitHub Actions as the **Registry Bump** workflow: on a manual dispatch (with the datasets ref to pin, default `main`) and weekly. When pins change, it opens a pull request with the new `registry.json`, using the `RELEASE_PLEASE_TOKEN` secret described under Releases.

## Maintainer: Releases

Releases are cut by [release-please](https://github.com/googleapis/release-please-action). A maintainer starts a release by running the Release Please workflow by hand (Actions > Release Please > Run workflow). The run opens or updates one release pull request that bumps the version in `pyproject.toml` and adds the next `CHANGELOG.md` section, computed from the conventional commits since the last tag. Merging that pull request creates the tag and the GitHub Release: the workflow run on that push publishes the merged release and never opens pull requests on its own.

- Re-pin the registry to the dataset release first, then run the workflow again to refresh the release pull request with the new commits. New merges to `main` do not refresh an open release pull request on their own, and a refresh dismisses earlier approvals.
- Do not hand-edit `CHANGELOG.md` sections that release-please generates, and do not run `cz bump`; version, changelog, tag and release come from the release pull request.
- `release-please-config.json` and `.release-please-manifest.json` hold the configuration and the current version. A `release-as` entry forces the next version; remove it once that release ships.
- Store a fine-grained personal access token (contents, pull requests and issues: read and write, on both repositories) as the `RELEASE_PLEASE_TOKEN` secret. GitHub starts no workflow for pull requests opened with the default `GITHUB_TOKEN`, so CI would not run on the release pull request and the required status checks would stay missing. The same secret serves the Registry Bump workflow. A GitHub App token cannot be stored as a secret, because it expires after one hour.
- The token owner cannot approve the pull requests opened in their name; another team member reviews the release and registry pull requests.
- The release pull request does not touch `uv.lock`. Run `uv lock` right after the merge, in the same pull request that removes `release-as`. Do not commit to the release branch; release-please force-pushes it on every run.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `make test` fails with import errors | Run `uv sync` to ensure dependencies are installed |
| Coverage below 85% | Add tests for new code; run `pytest --cov --cov-report=term-missing` to find gaps |
| pyright reports errors in third-party code | Verify you're on Python 3.12+ and re-run `uv sync` |
