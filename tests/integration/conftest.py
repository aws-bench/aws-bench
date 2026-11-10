"""Fixtures for the real-AWS acceptance tests (opt-in; see ``make integration``).

These tests drive the installed ``aws-bench`` CLI as a **subprocess**, not
Typer's ``CliRunner`` (which the unit tests in ``tests/cli`` use). A subprocess
exercises the real entry point, argument parsing, environment handling and exit
codes, which is what an end-to-end acceptance test should verify. The cost is
no in-process coverage, which is fine: these tests are excluded from the
coverage gate.

The default test run deselects this suite (marker ``integration``). Selecting it
explicitly (``make integration`` or ``pytest -m integration``) still deploys
nothing unless ``AWS_BENCH_INTEG=1`` and the account config variable are set:
the tests then skip, except the quickstart test, which errors early if
``AWS_BEARER_TOKEN_BEDROCK`` is not set. The settings logic lives in
``helpers.py`` so the default suite can unit-test it.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from .helpers import (
    HOME_LINKS_VAR,
    CliResult,
    IntegSettings,
    Secret,
    SecretEnv,
    load_settings,
    run_aws_bench,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Tag everything under tests/integration with the ``integration`` marker.

    Also disables the suite-wide pytest-timeout (30s in pyproject.toml, meant for
    unit tests): real deploys take tens of minutes, and every CLI call here carries
    its own per-command ``subprocess`` timeout instead.
    """
    for item in items:
        if "tests/integration" in item.nodeid.replace(os.sep, "/"):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.timeout(0))


@pytest.fixture(scope="session")
def run_id() -> str:
    """Unique per-session ID, used to namespace env names, job names and state."""
    return uuid.uuid4().hex[:8]


@pytest.fixture(scope="session")
def integ_settings(run_id: str) -> IntegSettings:
    """Skip unless opted in and configured; otherwise return the settings."""
    return load_settings(os.environ, run_id)


@pytest.fixture(scope="session")
def cli_env(integ_settings: IntegSettings, tmp_path_factory: pytest.TempPathFactory) -> SecretEnv:
    """Subprocess environment: real AWS credentials pass through; state is per-run.

    Returned as a ``SecretEnv``, whose ``repr`` hides the values, because pytest
    prints fixture arguments when a fixture fails and this environment carries
    the caller's credentials.

    ``HOME`` is redirected so ``~/.aws-bench`` state/logs of this run never touch
    (or collide with) the developer's own environments. Tool directories are
    symlinked into the scratch home so credentials keep working:

    - ``~/.aws``: profiles, including the IAM Identity Center token cache.
    - ``~/.docker``: Docker CLI plugins such as buildx.
    - Any extra paths in ``AWS_BENCH_INTEG_HOME_LINKS`` (comma-separated, relative
      to ``HOME``), e.g. what a ``credential_process`` helper reads from
      ``HOME``. Runs longer than one hour need refreshable base credentials.
    """
    env = dict(os.environ)
    real_home = Path(env.get("HOME", str(Path.home())))
    scratch_home = tmp_path_factory.mktemp("integ-home")
    extra = [p.strip() for p in env.get(HOME_LINKS_VAR, "").split(",") if p.strip()]
    for name in (".aws", ".docker", *extra):
        if (real_home / name).exists():
            link = scratch_home / name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(real_home / name, target_is_directory=True)
    env["HOME"] = str(scratch_home)
    env["AWSBENCH_ACCOUNT_CONFIG"] = str(integ_settings.account_config)
    return SecretEnv(env)


@pytest.fixture(scope="session")
def run_cli(cli_env: SecretEnv, tmp_path_factory: pytest.TempPathFactory):
    """Return a helper that runs ``aws-bench <args>`` as a subprocess.

    The working directory is a per-session scratch dir so relative output dirs
    (``jobs/``, ``scenario-jobs/``) stay out of the repo. Pass secret arguments
    as ``Secret`` so they never appear as frame arguments; failures and results
    mask secret values (see ``helpers.run_aws_bench``).
    """
    cwd = tmp_path_factory.mktemp("integ-cwd")

    def _run(*args: str | Secret, timeout: float | None = 3600, check: bool = True) -> CliResult:
        return run_aws_bench(args, cwd=cwd, env=cli_env, timeout=timeout, check=check)

    _run.cwd = cwd  # type: ignore[attr-defined]
    return _run
