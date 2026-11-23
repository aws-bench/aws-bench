"""Per-command happy-path acceptance tests against a real test account.

Opt-in only (``make integration``). Implemented: the quickstart + oracle flow
(``env setup`` -> ``run`` -> full reward -> ``env cleanup``), ``env show``
against the same deployment, and ``env list``. ``env cleanup`` as the step
under test is in ``test_env_cleanup.py``, which deploys on its own. The remaining
commands are named placeholders; see ``BACKLOG.md`` for priorities.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from .conftest import REPO_ROOT
from .helpers import (
    INIT_TIMEOUT_SEC,
    IntegSettings,
    MaskedDict,
    Secret,
    SecretEnv,
    StepTimeouts,
    deployed,
    env_list_problems,
    env_show_problems,
    expected_trial_count,
    find_scenario_manifests,
    load_accounts,
    mask_account_ids,
    mask_problems,
    parse_env_list,
    parse_env_show,
    resolve_run_tasks,
    run_timeout_for,
    step_timeouts_for,
)

pytestmark = pytest.mark.integration

# `env show` / `env list` are read-only and have no manifest budget either.
# Assumed, not derived: `env show` collects quotas per account and lists stacks
# region by region (sequentially) in every enabled region; 600 s is sized for
# the supported few accounts x enabled regions. A live run finished within it; the
# actual duration was not recorded.
SHOW_TIMEOUT_SEC = 600.0
# `env list` only checks credentials and reads the config in pre-existing mode.
# Assumed; a live run finished within it, the actual duration was not recorded.
LIST_TIMEOUT_SEC = 120.0


def _load_reward_checker():
    """Load scripts/check_job_rewards.py (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "check_job_rewards", REPO_ROOT / "scripts" / "check_job_rewards.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def bedrock_token() -> Secret:
    """Bearer token for the verifier's LLM judge (every quickstart task uses one).

    Session-scoped so it is checked before ``deployed_env`` deploys anything.
    Wrapped in ``Secret`` so pytest's failure tracebacks (which print each
    frame's arguments) never show it.
    """
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
    if not token:
        pytest.fail(
            "AWS_BEARER_TOKEN_BEDROCK is not set. The quickstart tasks are scored by a "
            "Bedrock LLM judge; run `eval $(uv run aws-bench env creds --eval)` first."
        )
    return Secret(token)


def _step_timeouts(settings: IntegSettings, cli_env: SecretEnv) -> StepTimeouts:
    """Timeouts from the scenario budgets; printed so a run shows what it used."""
    timeouts = step_timeouts_for(
        find_scenario_manifests(
            dataset_path=settings.dataset_path,
            scenario=settings.scenario,
            home=Path(cli_env["HOME"]),
        )
    )
    print(f"step timeouts: {timeouts.describe()}")  # describe() masks account IDs
    return timeouts


@pytest.fixture(scope="module")
def deployed_env(integ_settings: IntegSettings, cli_env: SecretEnv, run_cli, bedrock_token: Secret):
    """``env init`` + ``env setup``, with one ``env cleanup`` attempt on every exit path.

    Module-scoped: every test here that requests it shares one deployment, and
    cleanup runs once, after the last of them. If init or setup fails, pytest
    caches that error and reports it for each such test without deploying again.

    It requests ``bedrock_token`` itself, so every test that shares the
    deployment, not only the ones that use the token, fails before anything is
    deployed when the token is missing.

    Setup and cleanup timeouts come from the scenario manifest (read after
    ``env init``, which caches registry scenarios; re-read for cleanup, since
    the cache may only exist after a failed init). If init or setup fails and
    cleanup fails too, the original failure is reported with the cleanup
    failure attached (see ``helpers.deployed``).
    """
    with deployed(
        run_cli,
        init_args=integ_settings.env_command("init", "--quiet"),
        setup_args=integ_settings.env_command("setup", "--quiet"),
        cleanup_args=integ_settings.env_command("cleanup", "--yes", "--quiet"),
        init_timeout=INIT_TIMEOUT_SEC,
        setup_timeout=lambda: _step_timeouts(integ_settings, cli_env).setup,
        cleanup_timeout=lambda: _step_timeouts(integ_settings, cli_env).cleanup,
    ):
        yield integ_settings


def test_quickstart_oracle_earns_full_reward(
    bedrock_token: Secret, deployed_env: IntegSettings, cli_env: SecretEnv, run_cli
):
    """Quickstart dataset + oracle agent: the expected number of trials, all with reward 1.0."""
    job_name = f"integ-{deployed_env.run_id}"
    jobs_dir = Path(run_cli.cwd) / "jobs"  # type: ignore[attr-defined]
    # Same resolver and cache directory as the CLI `run` below.
    tasks = resolve_run_tasks(
        deployed_env.run_dataset_config(
            download_dir=Path(cli_env["HOME"]) / ".aws-bench" / "cache" / "tasks"
        )
    )
    # Fails here, before `run`, if AWS_BENCH_INTEG_EXPECTED_TRIALS disagrees.
    expected = expected_trial_count(deployed_env, resolved=tasks.expected_trials)
    run_timeout = run_timeout_for(tasks.task_dirs)
    note = f"; {tasks.note}" if tasks.note else ""
    print(mask_account_ids(f"expected trials: {expected}; timeout: {run_timeout.describe()}{note}"))
    run_cli(
        "run",
        *deployed_env.task_filter_args(),
        "--env-name", deployed_env.env_name,
        *deployed_env.run_source_args(),
        "-a", "oracle",
        "--job-name", job_name,
        "--jobs-dir", str(jobs_dir),
        "--ve", Secret(f"AWS_BEARER_TOKEN_BEDROCK={bedrock_token.value}"),
        "--yes", "--quiet",
        timeout=run_timeout.seconds,
    )  # fmt: skip
    checker = _load_reward_checker()
    result = json.loads((jobs_dir / job_name / "result.json").read_text())
    if isinstance(result, dict):
        result = MaskedDict(result)  # a local that pytest may print with -l
    n_total = result.get("n_total_trials") if isinstance(result, dict) else None
    planned = n_total == expected
    assert planned, mask_account_ids(
        f"job planned {n_total!r} trials, expected {expected} "
        f"for globs {list(deployed_env.task_globs) or 'all tasks'}"
    )
    # Trial and eval names come from task names; the checker prints them as they are.
    problems = mask_problems(checker.find_problems(result, expected_trials=expected))
    assert problems == []


def test_env_show_happy_path(deployed_env: IntegSettings, run_cli):
    """`env show --env-name N` exits 0 and shows N's accounts, each with a healthy scenario stack.

    Checked (``env_show_problems``): a header titled N whose account count
    matches the config; exactly one ``ACTIVE`` panel per configured (scenario,
    tag, account ID); and in every configured account a stack named
    ``<scenario>-*`` in ``HEALTHY_STACK_STATUSES``. Failed or rolled-back
    stacks are not checked (left to the `env verify` test), nor are regions or
    the header's OU row. The env name, count and account triples are compared
    with the same account config the CLI reads, so they verify rendering and
    plumbing; the stack check is the part tied to real AWS state. ``ACTIVE`` is
    a rendering guard: pre-existing mode always reports it.

    Needs the deployed env (for the stacks), so it shares the module-scoped
    ``deployed_env`` with the quickstart test: no extra deploy, and it runs
    after that test (file order), before the module's single ``env cleanup``.
    """
    result = run_cli("env", "show", "--env-name", deployed_env.env_name, timeout=SHOW_TIMEOUT_SEC)
    shown = parse_env_show(result.stdout)
    print(
        mask_account_ids(
            f"env show: {len(shown.accounts)} account panel(s), env {shown.env_name!r}"
        )
    )
    accounts = load_accounts(deployed_env.account_config)
    # Not `assert f(raw) == []`: pytest would print the raw output as call arguments.
    problems = env_show_problems(shown, env_name=deployed_env.env_name, accounts=accounts)
    assert problems == [], result.masked_output(4000)


def test_env_list_happy_path(integ_settings: IntegSettings, run_cli):
    """`env list` exits 0 and lists the configured environment with its account count.

    Reads only the account config and the caller identity, so it needs no deploy.
    """
    result = run_cli("env", "list", timeout=LIST_TIMEOUT_SEC)
    rows = parse_env_list(result.stdout)
    accounts = load_accounts(integ_settings.account_config)
    problems = env_list_problems(rows, env_name=integ_settings.env_name, accounts=accounts)
    assert problems == [], result.masked_output(4000)


# --- Placeholders (see BACKLOG.md) -------------------------------------------------


@pytest.mark.skip(reason="backlog: not implemented")
def test_env_init_happy_path():
    """`env init --env-name N -d D` exits 0 and records accounts for N (visible in `env show`)."""


@pytest.mark.skip(reason="backlog: not implemented")
def test_env_setup_happy_path():
    """`env setup` exits 0 and every scenario stack reaches a *_COMPLETE state."""


@pytest.mark.skip(reason="backlog: not implemented")
def test_env_creds_happy_path():
    """`env creds -o FILE` exits 0 and FILE holds a non-empty AWS_BEARER_TOKEN_BEDROCK."""


@pytest.mark.skip(reason="backlog: not implemented")
def test_env_terminate_happy_path():
    """`env terminate --yes` exits 0 and releases the environment (needs a disposable org)."""
