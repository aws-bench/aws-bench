"""Per-command happy-path acceptance tests against a real test account.

Opt-in only (``make integration``). Implemented: the quickstart + oracle flow
(``env setup`` -> ``run`` -> full reward -> ``env cleanup``). The remaining
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
    IntegSettings,
    Secret,
    SecretEnv,
    StepTimeouts,
    deployed,
    expected_trial_count,
    find_scenario_manifests,
    resolve_run_tasks,
    run_timeout_for,
    step_timeouts_for,
)

pytestmark = pytest.mark.integration

# `env init` validates the account and records the baseline; it has no
# manifest budget of its own.
INIT_TIMEOUT_SEC = 1200.0


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
    print(f"step timeouts: {timeouts.describe()}")
    return timeouts


@pytest.fixture(scope="module")
def deployed_env(integ_settings: IntegSettings, cli_env: SecretEnv, run_cli):
    """``env init`` + ``env setup``, with one ``env cleanup`` attempt on every exit path.

    Setup and cleanup timeouts come from the scenario manifest (read after
    ``env init``, which caches registry scenarios; re-read for cleanup, since
    the cache may only exist after a failed init). If init or setup fails and
    cleanup fails too, the original failure is reported with the cleanup
    failure attached (see ``helpers.deployed``).
    """
    env, src = integ_settings.env_name, integ_settings.env_source_args()
    with deployed(
        run_cli,
        init_args=["env", "init", "--env-name", env, *src, "--quiet"],
        setup_args=["env", "setup", "--env-name", env, *src, "--quiet"],
        cleanup_args=["env", "cleanup", "--env-name", env, *src, "--yes", "--quiet"],
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
    print(f"expected trials: {expected}; timeout: {run_timeout.describe()}{note}")
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
    n_total = result.get("n_total_trials") if isinstance(result, dict) else None
    assert n_total == expected, (
        f"job planned {n_total!r} trials, expected {expected} "
        f"for globs {list(deployed_env.task_globs) or 'all tasks'}"
    )
    assert checker.find_problems(result, expected_trials=expected) == []


# --- Placeholders (see BACKLOG.md) -------------------------------------------------


@pytest.mark.skip(reason="backlog: not implemented")
def test_env_init_happy_path():
    """`env init --env-name N -d D` exits 0 and records accounts for N (visible in `env show`)."""


@pytest.mark.skip(reason="backlog: not implemented")
def test_env_setup_happy_path():
    """`env setup` exits 0 and every scenario stack reaches a *_COMPLETE state."""


@pytest.mark.skip(reason="backlog: not implemented")
def test_env_list_happy_path():
    """`env list --env-name N` exits 0 and lists the environment created by this run."""


@pytest.mark.skip(reason="backlog: not implemented")
def test_env_show_happy_path():
    """`env show --env-name N` exits 0 and prints accounts and stack status for N."""


@pytest.mark.skip(reason="backlog: not implemented")
def test_env_creds_happy_path():
    """`env creds -o FILE` exits 0 and FILE holds a non-empty AWS_BEARER_TOKEN_BEDROCK."""


@pytest.mark.skip(reason="backlog: not implemented")
def test_env_cleanup_happy_path():
    """`env cleanup --yes` exits 0 and leaves no scenario stacks in the accounts."""


@pytest.mark.skip(reason="backlog: not implemented")
def test_env_terminate_happy_path():
    """`env terminate --yes` exits 0 and releases the environment (needs a disposable org)."""
