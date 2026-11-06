"""Fixtures for the real-AWS acceptance tests (opt-in; see ``make integration``).

These tests drive the installed ``aws-bench`` CLI as a **subprocess**, not
Typer's ``CliRunner`` (which the unit tests in ``tests/cli`` use). A subprocess
exercises the real entry point, argument parsing, environment handling and exit
codes, which is what an end-to-end acceptance test should verify. The cost is
no in-process coverage, which is fine: these tests are excluded from the
coverage gate.

Nothing here runs unless ``AWS_BENCH_INTEG=1`` and the account config variable
are set; otherwise every test skips with a message saying what is missing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from aws_bench.account_management.preexisting import load_account_config

REPO_ROOT = Path(__file__).resolve().parents[2]
OPT_IN_VAR = "AWS_BENCH_INTEG"
ACCOUNT_CONFIG_VAR = "AWS_BENCH_INTEG_ACCOUNT_CONFIG"
ENV_NAME_VAR = "AWS_BENCH_INTEG_ENV_NAME"
DATASET_VAR = "AWS_BENCH_INTEG_DATASET"
# Optional comma-separated `run -i` globs, e.g. "*list-ec-instances-all-regions".
# Harbor matches the task directory name (local/git) or "org/name" (registry);
# a leading "*" covers both.
TASKS_VAR = "AWS_BENCH_INTEG_TASKS"
# Optional local aws-bench-datasets checkout (has scenarios/ and tasks/). When
# set, it replaces `-d <dataset>`, so unreleased dataset changes can be tested.
DATASET_PATH_VAR = "AWS_BENCH_INTEG_DATASET_PATH"
# Scenario to use with DATASET_PATH_VAR (ignored otherwise).
SCENARIO_VAR = "AWS_BENCH_INTEG_SCENARIO"
# Extra paths (comma-separated, relative to HOME) to symlink into the scratch
# HOME, e.g. config/cache directories a credential_process helper needs.
HOME_LINKS_VAR = "AWS_BENCH_INTEG_HOME_LINKS"
DEFAULT_DATASET = "aws-bench-quickstart"
DEFAULT_SCENARIO = "ec2-multiregion"


@dataclass(frozen=True)
class IntegSettings:
    """Validated opt-in settings for one integration session."""

    run_id: str
    env_name: str
    dataset: str
    account_config: Path
    task_globs: tuple[str, ...] = ()
    dataset_path: Path | None = None
    scenario: str = DEFAULT_SCENARIO

    def env_source_args(self) -> list[str]:
        """Scenario-source arguments for `env init|setup|cleanup`.

        Local mode: `--scenario-path` must be a directory *containing* scenario
        directories (each immediate subdirectory with a scenario.toml counts), so
        it points at `<checkout>/scenarios`, narrowed by `--include-scenarios`.
        """
        if self.dataset_path is None:
            return ["-d", self.dataset]
        return [
            "--scenario-path", str(self.dataset_path / "scenarios"),
            "--include-scenarios", self.scenario,
        ]  # fmt: skip

    def run_source_args(self) -> list[str]:
        """Dataset-source arguments for `run`.

        Local mode needs both a tasks dir (`-p`) and a scenarios dir. `run` loads
        every scenario under `--scenario-path` but only uses those its tasks
        reference, so `-p <checkout>/tasks/<scenario>` limits it to one.
        """
        if self.dataset_path is None:
            return ["-d", self.dataset]
        return [
            "--scenario-path", str(self.dataset_path / "scenarios"),
            "-p", str(self.dataset_path / "tasks" / self.scenario),
        ]  # fmt: skip


@dataclass(frozen=True)
class CliResult:
    """Completed ``aws-bench`` invocation."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


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
    if os.environ.get(OPT_IN_VAR) != "1":
        pytest.skip(f"integration tests are opt-in: set {OPT_IN_VAR}=1 (see `make integration`)")
    raw = os.environ.get(ACCOUNT_CONFIG_VAR)
    if not raw:
        pytest.skip(f"{ACCOUNT_CONFIG_VAR} must point to a pre-existing account config YAML")
    config = Path(raw).expanduser().resolve()
    if not config.is_file():
        pytest.skip(f"{ACCOUNT_CONFIG_VAR}={raw} is not a file")
    # Pre-existing mode requires --env-name to equal the config's `name`
    # (TestEnvironmentNotFoundError otherwise), so it cannot carry a per-run
    # suffix. Isolation between runs comes from the redirected HOME and unique
    # job names; AWS-level isolation needs one account per concurrent runner.
    env_name = load_account_config(config).name
    override = os.environ.get(ENV_NAME_VAR)
    if override and override != env_name:
        pytest.fail(
            f"{ENV_NAME_VAR}={override!r} does not match the account config's "
            f"name {env_name!r}; pre-existing mode requires them to be equal. "
            f"Unset {ENV_NAME_VAR} or change `name` in {config}."
        )
    dataset_path: Path | None = None
    scenario = os.environ.get(SCENARIO_VAR) or DEFAULT_SCENARIO
    raw_dataset_path = os.environ.get(DATASET_PATH_VAR)
    if raw_dataset_path:
        if os.environ.get(DATASET_VAR):
            pytest.fail(f"Set either {DATASET_VAR} or {DATASET_PATH_VAR}, not both.")
        dataset_path = Path(raw_dataset_path).expanduser().resolve()
        for required in (
            dataset_path / "scenarios" / scenario / "scenario.toml",
            dataset_path / "tasks" / scenario,
        ):
            if not required.exists():
                pytest.fail(
                    f"{DATASET_PATH_VAR}={raw_dataset_path}: missing {required}. Point it at "
                    f"an aws-bench-datasets checkout and set {SCENARIO_VAR} if needed."
                )
    return IntegSettings(
        run_id=run_id,
        env_name=env_name,
        dataset=os.environ.get(DATASET_VAR, DEFAULT_DATASET),
        account_config=config,
        task_globs=tuple(g.strip() for g in os.environ.get(TASKS_VAR, "").split(",") if g.strip()),
        dataset_path=dataset_path,
        scenario=scenario,
    )


@pytest.fixture(scope="session")
def cli_env(
    integ_settings: IntegSettings, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, str]:
    """Subprocess environment: real AWS credentials pass through; state is per-run.

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
    return env


@pytest.fixture(scope="session")
def run_cli(cli_env: dict[str, str], tmp_path_factory: pytest.TempPathFactory):
    """Return a helper that runs ``aws-bench <args>`` as a subprocess.

    The working directory is a per-session scratch dir so relative output dirs
    (``jobs/``, ``scenario-jobs/``) stay out of the repo.
    """
    cwd = tmp_path_factory.mktemp("integ-cwd")

    def _run(*args: str, timeout: float = 3600, check: bool = True) -> CliResult:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "aws_bench.cli.main", *args],
            cwd=cwd,
            env=cli_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result = CliResult(tuple(args), proc.returncode, proc.stdout, proc.stderr)
        if check and proc.returncode != 0:
            pytest.fail(
                f"`aws-bench {' '.join(args)}` exited {proc.returncode}\n"
                f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
            )
        return result

    _run.cwd = cwd  # type: ignore[attr-defined]
    return _run
