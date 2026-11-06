"""Per-command happy-path acceptance tests against a real test account.

Opt-in only (``make integration``). Implemented: the quickstart + oracle flow
(``env setup`` -> ``run`` -> full reward -> ``env cleanup``). The remaining
commands are named placeholders; see ``BACKLOG.md`` for priorities.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from .conftest import REPO_ROOT, IntegSettings

pytestmark = pytest.mark.integration


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
def bedrock_token() -> str:
    """Bearer token for the verifier's LLM judge (every quickstart task uses one).

    Session-scoped so it is checked before ``deployed_env`` deploys anything.
    """
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
    if not token:
        pytest.fail(
            "AWS_BEARER_TOKEN_BEDROCK is not set. The quickstart tasks are scored by a "
            "Bedrock LLM judge; run `eval $(uv run aws-bench env creds --eval)` first."
        )
    return token


@pytest.fixture(scope="module")
def deployed_env(integ_settings: IntegSettings, run_cli):
    """``env init`` + ``env setup``, with ``env cleanup`` as a finalizer (always runs)."""
    env, src = integ_settings.env_name, integ_settings.env_source_args()
    try:
        run_cli("env", "init", "--env-name", env, *src, "--quiet", timeout=1200)
        run_cli("env", "setup", "--env-name", env, *src, "--quiet", timeout=2700)
        yield integ_settings
    finally:
        # check=False so a failed cleanup never masks the original failure.
        result = run_cli(
            "env", "cleanup", "--env-name", env, *src, "--yes", "--quiet",
            timeout=1800, check=False,
        )  # fmt: skip
        if result.returncode != 0:
            pytest.fail(
                f"env cleanup failed (resources may be left behind):\n{result.output[-4000:]}"
            )


def test_quickstart_oracle_earns_full_reward(
    bedrock_token: str, deployed_env: IntegSettings, run_cli
):
    """Quickstart dataset + oracle agent: every trial completes with reward 1.0."""
    job_name = f"integ-{deployed_env.run_id}"
    jobs_dir = Path(run_cli.cwd) / "jobs"  # type: ignore[attr-defined]
    task_filter = [arg for g in deployed_env.task_globs for arg in ("-i", g)]
    run_cli(
        "run",
        *task_filter,
        "--env-name", deployed_env.env_name,
        *deployed_env.run_source_args(),
        "-a", "oracle",
        "--job-name", job_name,
        "--jobs-dir", str(jobs_dir),
        "--ve", f"AWS_BEARER_TOKEN_BEDROCK={bedrock_token}",
        "--yes", "--quiet",
        timeout=5400,
    )  # fmt: skip
    checker = _load_reward_checker()
    import json

    result = json.loads((jobs_dir / job_name / "result.json").read_text())
    assert checker.find_problems(result) == []


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
