"""End-to-end ledger behavior across the CLI surface.

Exercises the Typer root callback's ledger wiring: every command opens an
entry (beat 1) and the ``call_on_close`` finalizer stamps the outcome (beat 3)
on success, error, interrupt, and clean ``typer.Exit``. The callback runs
before every command, so a regression here breaks the whole CLI.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import typer
from typer.testing import CliRunner

import aws_bench.logging.logger as _logger_mod

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_console():
    root = logging.getLogger("aws_bench")
    saved = root.handlers.copy()
    saved_level = root.level
    root.handlers.clear()
    _logger_mod._configured = False
    yield
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            h.close()
    root.handlers.clear()
    root.handlers.extend(saved)
    root.setLevel(saved_level)
    _logger_mod._configured = False


def _build_app(monkeypatch, tmp_path):
    """A minimal app reusing the real root callback, with LOGS_DIR redirected."""
    monkeypatch.setattr("aws_bench.logging.ledger.LOGS_DIR", tmp_path / "logs")
    from aws_bench.cli.main import _root

    app = typer.Typer()
    app.callback()(_root)

    @app.command()
    def ok():
        logging.getLogger("aws_bench.t").info("ran-ok")

    @app.command()
    def boom():
        raise ValueError("kaboom")

    @app.command()
    def exit_clean():
        raise typer.Exit()

    @app.command()
    def exit_fail():
        raise typer.Exit(code=1)

    @app.command()
    def interrupt():
        raise KeyboardInterrupt

    return app


def _only_entry(tmp_path):
    entries = list((tmp_path / "logs").iterdir())
    assert len(entries) == 1, entries
    return entries[0]


def test_every_command_writes_an_entry(monkeypatch, tmp_path):
    app = _build_app(monkeypatch, tmp_path)
    result = runner.invoke(app, ["ok"])
    assert result.exit_code == 0
    entry = _only_entry(tmp_path)
    data = json.loads((entry / "command.json").read_text())
    # The command label derives from sys.argv (covered by TestCommandLabel);
    # under CliRunner sys.argv is the test runner's, so assert the entry exists
    # and finalized, not its label.
    assert data["exit_status"] == "ok"
    assert "ran-ok" in (entry / "run.log").read_text()


def test_failing_command_finalizes_error(monkeypatch, tmp_path):
    app = _build_app(monkeypatch, tmp_path)
    result = runner.invoke(app, ["boom"])
    assert result.exit_code != 0
    data = json.loads((_only_entry(tmp_path) / "command.json").read_text())
    assert data["exit_status"] == "error"
    assert data["error_type"] == "ValueError"


def test_clean_exit_zero_finalizes_ok(monkeypatch, tmp_path):
    """``typer.Exit(0)`` is a clean exit, not an error.

    Inside ``call_on_close`` the live exception is a ``typer.Exit`` with
    ``exit_code == 0``; the callback must translate that to ``None`` before
    handing it to ``finalize`` so the entry reads ``ok``, not ``error``.
    """
    app = _build_app(monkeypatch, tmp_path)
    result = runner.invoke(app, ["exit-clean"])
    assert result.exit_code == 0
    data = json.loads((_only_entry(tmp_path) / "command.json").read_text())
    assert data["exit_status"] == "ok"
    assert data["error_type"] is None


def test_nonzero_exit_finalizes_error(monkeypatch, tmp_path):
    app = _build_app(monkeypatch, tmp_path)
    result = runner.invoke(app, ["exit-fail"])
    assert result.exit_code == 1
    data = json.loads((_only_entry(tmp_path) / "command.json").read_text())
    assert data["exit_status"] == "error"
    assert data["error_type"] == "Exit"


def test_interrupt_finalizes_interrupted(monkeypatch, tmp_path):
    app = _build_app(monkeypatch, tmp_path)
    result = runner.invoke(app, ["interrupt"])
    assert result.exit_code != 0
    data = json.loads((_only_entry(tmp_path) / "command.json").read_text())
    assert data["exit_status"] == "interrupted"
    assert data["error_type"] == "KeyboardInterrupt"


# ---------------------------------------------------------------------------
# Beat 2 in `run`: job identity + resolved config patched into command.json
# ---------------------------------------------------------------------------


def _make_local_dataset(tmp_path):
    """A minimal local tasks + scenarios pair so ``validate_run`` passes.

    The dirs only need to exist for the local-dataset path through ``start``;
    ``AwsBenchJob.create`` is stubbed, so no task/scenario is actually loaded.
    """
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    return tasks_dir, scenarios_dir


def _stub_run_seams(monkeypatch, *, job_id: str, is_resuming: bool):
    """Stub the AWS/Docker seams of ``run`` and return a stub job from ``create``.

    Patches ``AwsBenchJob.create`` (AsyncMock → stub job), the three preflights,
    ``_confirm_host_env_access``, and the results renderer, all at their
    ``aws_bench.cli.jobs`` call sites. The run reaches beat 2 (a) before
    ``create`` and beat 2 (b) right after it, then renders and exits cleanly.
    """
    from harbor.models.job.result import JobResult, JobStats

    from aws_bench.resource_management.verify.models import VerificationReport

    now = datetime.now(timezone.utc)
    job_result = JobResult(
        id=uuid4(), started_at=now, finished_at=now, n_total_trials=0, stats=JobStats(evals={})
    )

    job = MagicMock()
    job.id = job_id
    job.is_resuming = is_resuming
    job._job_result_path = Path("/tmp/job-result")
    job.run = AsyncMock(return_value=job_result)

    # Mirror real create(): it resolves and assigns config.test_environment on
    # the passed-in config (run's verify gate asserts it is set).
    async def _create(config):
        config.test_environment = MagicMock()
        return job

    monkeypatch.setattr("aws_bench.cli.jobs.AwsBenchJob.create", _create)
    # Default-on verification gate is a run seam too: stub it to a passing report.
    monkeypatch.setattr(
        "aws_bench.cli.jobs.ResourceManager.verify_environment",
        AsyncMock(return_value=VerificationReport(passed=True, env_name="awsbench-ou", results=[])),
    )
    monkeypatch.setattr("aws_bench.cli.jobs._confirm_host_env_access", lambda *a, **k: None)
    monkeypatch.setattr("aws_bench.cli.jobs.print_job_results_tables", lambda *a, **k: None)
    monkeypatch.setattr("aws_bench.cli.jobs.preflight_docker_cli", lambda *a, **k: None)
    monkeypatch.setattr("aws_bench.cli.jobs.preflight_docker_daemon", lambda *a, **k: None)
    monkeypatch.setattr("aws_bench.cli.jobs.preflight_docker_plugins", lambda *a, **k: None)
    monkeypatch.setattr("aws_bench.cli.jobs.preflight_aws_credentials", lambda *a, **k: None)
    return job


def test_run_patches_job_identity_into_entry(monkeypatch, tmp_path):
    """A ``run`` invocation records job_id, is_resuming, and resolved_config.

    Drives the real ``run`` alias through the app so the Typer root callback
    wires the ledger and ``start`` receives a live ``ctx``. ``AwsBenchJob.create``
    is stubbed to return a job with a known ``.id`` / ``.is_resuming``; beat 2 (b)
    must upgrade the entry with that identity, and the entry finalizes ``ok``.
    """
    monkeypatch.setattr("aws_bench.logging.ledger.LOGS_DIR", tmp_path / "logs")
    tasks_dir, scenarios_dir = _make_local_dataset(tmp_path)
    job_id = str(uuid4())
    _stub_run_seams(monkeypatch, job_id=job_id, is_resuming=True)

    from aws_bench.cli.main import app

    result = runner.invoke(
        app,
        [
            "run",
            "--env-name",
            "awsbench-ou",
            "--path",
            str(tasks_dir),
            "--scenario-path",
            str(scenarios_dir),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    entry = _only_entry(tmp_path)
    data = json.loads((entry / "command.json").read_text())
    assert data["job_id"] == job_id
    assert data["is_resuming"] is True
    assert data["resolved_config"] is not None
    assert data["resolved_config"]["dataset"]["path"] == str(tasks_dir)
    assert data["job_dir"] is not None
    assert data["exit_status"] == "ok"


def test_run_records_resolved_config_when_create_fails(monkeypatch, tmp_path):
    """Beat 2 (a) runs before ``create``, so a create failure still leaves the config.

    If ``AwsBenchJob.create`` raises, the run aborts (typer.Exit(1)) and the
    entry finalizes ``error`` — but ``resolved_config`` is already populated from
    the pre-create patch, and ``job_id`` stays ``None`` (never upgraded).
    """
    monkeypatch.setattr("aws_bench.logging.ledger.LOGS_DIR", tmp_path / "logs")
    tasks_dir, scenarios_dir = _make_local_dataset(tmp_path)

    monkeypatch.setattr(
        "aws_bench.cli.jobs.AwsBenchJob.create",
        AsyncMock(side_effect=RuntimeError("create blew up")),
    )
    monkeypatch.setattr("aws_bench.cli.jobs.preflight_docker_cli", lambda *a, **k: None)
    monkeypatch.setattr("aws_bench.cli.jobs.preflight_docker_daemon", lambda *a, **k: None)
    monkeypatch.setattr("aws_bench.cli.jobs.preflight_docker_plugins", lambda *a, **k: None)
    monkeypatch.setattr("aws_bench.cli.jobs.preflight_aws_credentials", lambda *a, **k: None)

    from aws_bench.cli.main import app

    result = runner.invoke(
        app,
        [
            "run",
            "--env-name",
            "awsbench-ou",
            "--path",
            str(tasks_dir),
            "--scenario-path",
            str(scenarios_dir),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--yes",
        ],
    )

    assert result.exit_code == 1, result.output
    data = json.loads((_only_entry(tmp_path) / "command.json").read_text())
    assert data["resolved_config"] is not None
    assert data["resolved_config"]["dataset"]["path"] == str(tasks_dir)
    assert data["job_id"] is None
    assert data["exit_status"] == "error"


# ---------------------------------------------------------------------------
# Beat 2 in env phase commands: scenarios config + job-dir snapshot
# ---------------------------------------------------------------------------


def _stub_env_phase_seams(monkeypatch, *, tmp_path):
    """Stub the AWS/Docker seams of ``_run_phase_command`` (env cleanup path).

    Patches the three preflights, ``CredentialProvider.get``, and
    ``ScenarioJob.create`` + ``run_phase_with_progress`` at their
    ``aws_bench.cli.env`` call sites so the phase reaches beat 2, records into
    the ledger, then renders an empty (all-passed) result and exits cleanly —
    without touching Docker or AWS.
    """
    from aws_bench.scenario.results import ScenarioJobResult

    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_cli", lambda: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_daemon", lambda: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_docker_plugins", lambda plugins: None)
    monkeypatch.setattr("aws_bench.cli.env.preflight_aws_credentials", lambda cred, **kwargs: None)
    monkeypatch.setattr(
        "aws_bench.cli.env.CredentialProvider.get",
        classmethod(lambda cls: MagicMock()),
    )

    result = ScenarioJobResult(
        job_name="test",
        job_dir=tmp_path / "jobs" / "test",
        started_at=datetime.now(timezone.utc),
        n_total=0,
        n_succeeded=0,
        n_failed=0,
        trial_results=[],
    )
    job = MagicMock()
    job.run = AsyncMock(return_value=result)
    monkeypatch.setattr("aws_bench.cli.env.ScenarioJob.create", AsyncMock(return_value=job))

    async def _passthrough(job, phase):
        return await job.run(phase)

    monkeypatch.setattr("aws_bench.cli.env.run_phase_with_progress", _passthrough)


def test_env_phase_records_config_and_registers_snapshot(monkeypatch, tmp_path):
    """An env phase invocation records the scenarios config and its job dir.

    Drives the real ``env cleanup`` through the app so the Typer root callback
    wires the ledger and ``_run_phase_command`` receives a live ``ctx``. The
    Docker/AWS seams are stubbed; beat 2 must patch ``resolved_config`` (the
    ``ScenarioJobConfig`` dump) and a non-null ``job_dir`` into the entry, leave
    ``job_id`` null (phases have no stable job id), and finalize ``ok``.
    """
    monkeypatch.setattr("aws_bench.logging.ledger.LOGS_DIR", tmp_path / "logs")
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir = tmp_path / "jobs"
    _stub_env_phase_seams(monkeypatch, tmp_path=tmp_path)

    from aws_bench.cli.main import app

    result = runner.invoke(
        app,
        [
            "env",
            "cleanup",
            "--env-name",
            "awsbench-ou",
            "--scenario-path",
            str(scenarios_dir),
            "--jobs-dir",
            str(jobs_dir),
            "--job-name",
            "phase-job",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    entry = _only_entry(tmp_path)
    data = json.loads((entry / "command.json").read_text())
    assert data["job_id"] is None
    assert data["is_resuming"] is False
    assert data["resolved_config"] is not None
    assert data["resolved_config"]["dataset"]["scenarios_path"] == str(scenarios_dir)
    assert data["job_dir"] == str(jobs_dir / "phase-job")
    assert data["exit_status"] == "ok"


def test_env_init_records_config_without_snapshot(monkeypatch, tmp_path):
    """``env init`` records its resolved scenario config but no job dir.

    init provisions accounts directly — there is no job directory — so beat 2
    records ``resolved_config`` with ``job_dir`` null and registers nothing to
    snapshot. The provisioning seams are stubbed so it never touches AWS.
    """
    monkeypatch.setattr("aws_bench.logging.ledger.LOGS_DIR", tmp_path / "logs")
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("aws_bench.cli.env.preflight_aws_credentials", lambda cred, **kwargs: None)
    monkeypatch.setattr(
        "aws_bench.cli.env.CredentialProvider.get",
        classmethod(lambda cls: MagicMock()),
    )
    monkeypatch.setattr("aws_bench.cli.env.AccountManager", MagicMock())
    monkeypatch.setattr("aws_bench.cli.env.QuotaManager", MagicMock())
    monkeypatch.setattr("aws_bench.cli.env.display_provisioning_summary", lambda *a, **k: None)
    monkeypatch.setattr(
        "aws_bench.cli.env.provision_scenarios",
        AsyncMock(return_value=MagicMock(timed_out=False)),
    )

    async def _no_scenarios(self):
        return {}

    monkeypatch.setattr(
        "aws_bench.dataset.config.AwsBenchDatasetConfig.get_scenarios", _no_scenarios
    )

    from aws_bench.cli.main import app

    result = runner.invoke(
        app,
        ["env", "init", "--env-name", "awsbench-ou", "--scenario-path", str(scenarios_dir), "-q"],
    )

    assert result.exit_code == 0, result.output
    data = json.loads((_only_entry(tmp_path) / "command.json").read_text())
    assert data["job_id"] is None
    assert data["resolved_config"] is not None
    assert data["resolved_config"]["scenarios_path"] == str(scenarios_dir)
    assert data["job_dir"] is None
    assert data["exit_status"] == "ok"


# ---------------------------------------------------------------------------
# argv → command label (the ledger's command field + dir-name slug)
# ---------------------------------------------------------------------------


class TestCommandLabel:
    def test_top_level_command(self):
        from aws_bench.cli.main import _command_label

        assert _command_label(["aws-bench", "run", "-c", "cfg.yaml"]) == "run"

    def test_group_subcommand_keeps_leaf(self):
        from aws_bench.cli.main import _command_label

        # A group's leaf must survive, not collapse to the group ("env").
        assert _command_label(["aws-bench", "env", "cleanup", "--env-name", "x"]) == "env cleanup"
        assert _command_label(["aws-bench", "env", "show"]) == "env show"
        assert _command_label(["aws-bench", "job", "start"]) == "job start"

    def test_positional_arg_is_not_mistaken_for_a_command(self):
        from aws_bench.cli.main import _command_label

        # view takes a positional folder — it must NOT become "view /some/path",
        # not even when the folder is literally named like a command ("env").
        assert _command_label(["aws-bench", "view", "/some/jobs"]) == "view"
        assert _command_label(["aws-bench", "view", "env"]) == "view"

    def test_leading_global_flag_does_not_shift_the_command(self):
        from aws_bench.cli.main import _command_label

        # Flags are filtered before reading the command, so a global option
        # before the subcommand doesn't break the read.
        assert _command_label(["aws-bench", "--debug", "env", "cleanup"]) == "env cleanup"

    def test_unknown_command_falls_back_to_program_name(self):
        from aws_bench.cli.main import _command_label

        # A typo or unknown invocation labels safely, never writes raw argv.
        assert _command_label(["aws-bench", "rnu"]) == "aws-bench"
        assert _command_label(["aws-bench", "env", "bogus"]) == "aws-bench"

    def test_bare_or_flag_only_invocation(self):
        from aws_bench.cli.main import _command_label

        assert _command_label(["aws-bench"]) == "aws-bench"
        assert _command_label(["aws-bench", "--help"]) == "aws-bench"

    def test_known_commands_matches_registered_app(self):
        """_KNOWN_COMMANDS must equal the app's actual command paths.

        The set is hand-maintained next to the Typer registrations; this guards
        the drift trap — add or rename a command without updating the set and a
        real invocation silently mislabels as ``aws-bench``. Deriving the truth
        from the resolved Click tree makes that drift fail here instead.
        """
        from typer.main import get_command

        from aws_bench.cli.main import _KNOWN_COMMANDS, app

        root = get_command(app)

        def walk(group, prefix=""):
            paths: set[str] = set()
            for name, cmd in group.commands.items():
                full = f"{prefix}{name}"
                subcommands = getattr(cmd, "commands", None)
                if subcommands:
                    paths |= walk(cmd, prefix=f"{full} ")
                else:
                    paths.add(full)
            return paths

        assert walk(root) == set(_KNOWN_COMMANDS)
