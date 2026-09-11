"""Integration: ``aws-bench env setup -d <name>@<version>`` happy path.

Mocks at IO boundaries:
  * ``AwsBenchRegistry.from_path`` → fake registry (registry-mode entry)
  * ``ScenarioJob.create`` → returns a mock job (skips quota checks /
    account resolution / Scenario materialization)
  * ``preflight_*`` and ``CredentialProvider.get`` → no AWS, no Docker.

The test pins the ``@``-split contract: when ``-d test-dataset@1.0.0``
is passed, the resulting ``AwsBenchDatasetConfig`` must have
``name == "test-dataset"`` and ``version == "1.0.0"`` — never the raw
concatenated string, which the registry would silently fail to match.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aws_bench.cli.env import setup as env_setup
from aws_bench.dataset.registry import (
    AwsBenchDatasetSpec,
    AwsBenchRegistry,
    RegistryScenarioId,
)


@pytest.fixture
def fake_registry():
    return AwsBenchRegistry(
        datasets=[
            AwsBenchDatasetSpec(
                name="test-dataset",
                version="1.0.0",
                description="",
                tasks=[],
                scenarios=[
                    RegistryScenarioId(
                        name="ec2-small",
                        git_url="https://github.com/x/y",
                        git_commit_id="abc",
                        path=Path("scenarios/ec2-small"),
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def stub_setup_deps(mocker, fake_registry):
    """Stub every external dep of ``env setup``: registry, AWS, Docker, ScenarioJob."""
    ns = MagicMock()

    mocker.patch(
        "aws_bench.dataset.config.AwsBenchRegistry.from_path",
        return_value=fake_registry,
    )
    mocker.patch("aws_bench.cli.env.preflight_aws_credentials")
    mocker.patch("aws_bench.cli.env.preflight_docker_cli")
    mocker.patch("aws_bench.cli.env.preflight_docker_daemon")
    mocker.patch("aws_bench.cli.env.preflight_docker_plugins")

    mocker.patch(
        "aws_bench.cli.env.CredentialProvider.get",
        classmethod(lambda cls: MagicMock()),
    )

    fake_result = MagicMock()
    fake_result.all_passed = True
    fake_result.n_total = 1
    fake_result.n_succeeded = 1
    fake_result.n_failed = 0

    fake_job = MagicMock()
    fake_job.run = AsyncMock(return_value=fake_result)
    ns.fake_job = fake_job

    ns.scenario_job_create = mocker.patch(
        "aws_bench.cli.env.ScenarioJob.create",
        new=AsyncMock(return_value=fake_job),
    )
    mocker.patch("aws_bench.cli.env.display_setup_summary")

    return ns


def test_env_setup_registry_mode_happy_path(stub_setup_deps, tmp_path):
    fake_registry_path = tmp_path / "registry.json"
    fake_registry_path.write_text("[]")

    # This test drives ``setup`` directly (not via CliRunner), so it supplies the
    # ``ctx`` Typer would otherwise inject. An empty ``meta`` means no ledger is
    # wired, so the beat-2 recording in ``_run_phase_command`` is skipped.
    env_setup(
        ctx=MagicMock(meta={}),
        ou_name="awsbench-ou",
        scenario_path=None,
        dataset="test-dataset@1.0.0",
        registry_url=None,
        registry_path=fake_registry_path,
        include_scenarios=None,
        exclude_scenarios=None,
        job_name=None,
        jobs_dir=None,
        n_concurrent=None,
        timeout_multiplier=None,
        quiet=True,
        max_retries=None,
        retry_include=None,
        retry_exclude=None,
        force_build=None,
        delete=None,
        override_cpus=None,
        override_memory_mb=None,
        override_build_timeout_sec=None,
    )

    # ScenarioJob.create was invoked with a job config whose dataset
    # carries the parsed name + version, not the raw "name@version" string.
    stub_setup_deps.scenario_job_create.assert_called_once()
    job_cfg = stub_setup_deps.scenario_job_create.call_args.args[0]
    assert job_cfg.ou_name == "awsbench-ou"
    assert job_cfg.dataset.name == "test-dataset"
    assert job_cfg.dataset.version == "1.0.0"
    assert job_cfg.dataset.registry_path == fake_registry_path

    # quiet=True takes the bare run path (no progress wrapper).
    stub_setup_deps.fake_job.run.assert_called_once()
