"""``aws-bench run`` CLI.

Builds a ``JobConfig`` from CLI flags, fetches scenarios via
``AwsBenchDatasetConfig``, and threads the resolved scenarios + AWS
account map onto the job before invoking ``Job.run()``.
"""

import json
import logging
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
import yaml
from dotenv import load_dotenv
from harbor.cli.jobs import (
    _confirm_host_env_access,
    _format_duration,
    print_job_results_tables,
)
from harbor.cli.utils import load_mcp_servers, parse_env_vars, parse_kwargs, run_async
from harbor.models.trial.config import AgentConfig
from typer import Option, Typer

from aws_bench.agents import _KIRO_CLI_NAME as _  # noqa: F401 — registers kiro-cli agent
from aws_bench.cli.job_config import AwsBenchJobConfig
from aws_bench.cli.metrics import parse_metric
from aws_bench.cli.preflight import (
    preflight_aws_credentials,
    preflight_docker_cli,
    preflight_docker_daemon,
    preflight_docker_plugins,
)
from aws_bench.cli.ui import console
from aws_bench.dataset.config import AwsBenchDatasetConfig
from aws_bench.logging.ledger import current_ledger
from aws_bench.logging.logger import (
    file_logging,
    get_logger,
    set_console_level,
)
from aws_bench.resource_management.exceptions import EnvironmentVerifyError
from aws_bench.resource_management.manager import ResourceManager
from aws_bench.task.job import AwsBenchJob
from aws_bench.utils.credentials_provider import CredentialProvider
from aws_bench.utils.error_display import print_exception

logger = get_logger(__name__)

# Harbor's run-path modules log on this shared logger, which doesn't propagate
# to the aws_bench tree; the run path attaches its file handlers here too.
_HARBOR_RUN_LOGGER_NAME = "harbor.utils.logger"

# Scenarios verified concurrently in the pre-run gate — its own fan-out,
# independent of n_concurrent_trials (which bounds trial execution).
_VERIFY_CONCURRENCY = 8


def _maybe_harbor_into_runlog(run_log: Path | None):
    """Tee Harbor's run-tree logger into the ledger run.log, when there is one."""
    if run_log is None:
        return nullcontext()
    return file_logging(run_log, logger_name=_HARBOR_RUN_LOGGER_NAME)


jobs_app = Typer(no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]})


def start(
    ctx: typer.Context,
    config_path: Annotated[
        Path | None,
        Option(
            "-c",
            "--config",
            help="Job config file (YAML/JSON). Overrides individual options.",
            rich_help_panel="Config",
            show_default=False,
        ),
    ] = None,
    # ── Dataset ──────────────────────────────────────────────────────
    path: Annotated[
        Path | None,
        Option(
            "-p",
            "--path",
            help="Directory containing task directories.",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    dataset_name_version: Annotated[
        str | None,
        Option(
            "-d",
            "--dataset",
            help="Dataset name[@version] from a registry.",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    registry_url: Annotated[
        str | None,
        Option(
            "--registry-url",
            help="Registry URL for remote dataset.",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    registry_path: Annotated[
        Path | None,
        Option(
            "--registry-path",
            help="Path to local registry.",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    include_task_names: Annotated[
        list[str] | None,
        Option(
            "-i",
            "--include-task-name",
            help="Task names to include (glob patterns).",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    exclude_task_names: Annotated[
        list[str] | None,
        Option(
            "-x",
            "--exclude-task-name",
            help="Task names to exclude (glob patterns).",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    n_tasks: Annotated[
        int | None,
        Option(
            "-l",
            "--n-tasks",
            help="Max number of tasks.",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    extra_instruction_paths: Annotated[
        list[Path] | None,
        Option(
            "--extra-instruction-path",
            help=(
                "Path to an extra instruction file to append to the task "
                "instruction. Can be used multiple times."
            ),
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    metric: Annotated[
        list[str] | None,
        Option(
            "--metric",
            help=(
                "Add a metric on top of the dataset's metrics (does not "
                "replace them). Format: 'mean' or 'uv-script:script_path=./m.py'. "
                "Can be used multiple times."
            ),
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    # ── Agent ────────────────────────────────────────────────────────
    agent_name: Annotated[
        str | None,
        Option(
            "-a",
            "--agent",
            help="Agent name.",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    agent_import_path: Annotated[
        str | None,
        Option(
            "--agent-import-path",
            help="Import path for a custom agent class (e.g. my_module.agents:MyAgent).",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    model_names: Annotated[
        list[str] | None,
        Option(
            "-m",
            "--model",
            help="Model name for the agent. Can be repeated for multi-model runs.",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    agent_kwargs: Annotated[
        list[str] | None,
        Option(
            "--ak",
            "--agent-kwarg",
            help="Agent keyword argument in key=value format (JSON/Python literal parsing).",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    agent_env: Annotated[
        list[str] | None,
        Option(
            "--ae",
            "--agent-env",
            help="Environment variable for the agent in KEY=VALUE format.",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    mcp_config: Annotated[
        list[Path] | None,
        Option(
            "--mcp-config",
            help=(
                "Path to a Claude-style .mcp.json or MCP server config "
                "file. Can be used multiple times."
            ),
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    skills: Annotated[
        list[Path] | None,
        Option(
            "--skill",
            "--skills",
            help=(
                "Path to a skill directory, or a root containing skill "
                "directories. Can be used multiple times."
            ),
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    # ── Job Settings ─────────────────────────────────────────────────
    job_name: Annotated[
        str | None,
        Option(
            "--job-name",
            help="Job name (default: timestamp).",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    jobs_dir: Annotated[
        Path | None,
        Option(
            "-o",
            "--jobs-dir",
            help="Results directory (default: jobs).",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    n_attempts: Annotated[
        int | None,
        Option(
            "-k",
            "--n-attempts",
            help="Number of attempts per trial.",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    timeout_multiplier: Annotated[
        float | None,
        Option(
            "--timeout-multiplier",
            help=f"Multiplier for task timeouts (default: {
                AwsBenchJobConfig.model_fields['timeout_multiplier'].default
            }).",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    agent_timeout_multiplier: Annotated[
        float | None,
        Option(
            "--agent-timeout-multiplier",
            help="Multiplier for agent execution timeout (overrides --timeout-multiplier).",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    verifier_timeout_multiplier: Annotated[
        float | None,
        Option(
            "--verifier-timeout-multiplier",
            help="Multiplier for verifier timeout (overrides --timeout-multiplier).",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    agent_setup_timeout_multiplier: Annotated[
        float | None,
        Option(
            "--agent-setup-timeout-multiplier",
            help="Multiplier for agent setup timeout (overrides --timeout-multiplier).",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    environment_build_timeout_multiplier: Annotated[
        float | None,
        Option(
            "--environment-build-timeout-multiplier",
            help="Multiplier for environment build timeout (overrides --timeout-multiplier).",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    n_concurrent: Annotated[
        int | None,
        Option(
            "-n",
            "--n-concurrent",
            help="Number of trials to run at once (default: 4).",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    max_retries: Annotated[
        int | None,
        Option(
            "-r",
            "--max-retries",
            help="Maximum number of retry attempts for failed trials.",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    retry_include: Annotated[
        list[str] | None,
        Option(
            "--retry-include",
            help="Exception type names to retry on (can be repeated).",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    retry_exclude: Annotated[
        list[str] | None,
        Option(
            "--retry-exclude",
            help="Exception type names to NEVER retry on (can be repeated).",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    quiet: Annotated[
        bool,
        Option(
            "-q",
            "--quiet",
            "--silent",
            help="Suppress individual trial progress displays.",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = False,
    debug: Annotated[
        bool,
        Option(
            "--debug",
            help="Enable debug logging.",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = False,
    env_file: Annotated[
        Path | None,
        Option(
            "--env-file",
            help="Path to a .env file to load into the environment.",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    # ── Environment ──────────────────────────────────────────────────
    environment_force_build: Annotated[
        bool | None,
        Option(
            "--force-build/--no-force-build",
            help="Force rebuild the environment image.",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    environment_delete: Annotated[
        bool | None,
        Option(
            "--delete/--no-delete",
            help="Delete the environment after completion.",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    mounts: Annotated[
        str | None,
        Option(
            "--mounts",
            help=(
                "JSON array of volume mounts for the agent environment "
                "container (Docker Compose service volume format). "
                "Useful for docker.sock passthrough or sharing host "
                "directories with the agent."
            ),
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    extra_docker_compose: Annotated[
        list[Path] | None,
        Option(
            "--extra-docker-compose",
            help=(
                "Additional Docker Compose overlay file layered on top of "
                "the task's environment definition. Can be used multiple "
                "times to chain overlays."
            ),
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    # ── Verifier ─────────────────────────────────────────────────────
    verifier_env: Annotated[
        list[str] | None,
        Option(
            "--ve",
            "--verifier-env",
            help="Environment variable for the verifier in KEY=VALUE format.",
            rich_help_panel="Verifier",
            show_default=False,
        ),
    ] = None,
    verifier_import_path: Annotated[
        str | None,
        Option(
            "--verifier-import-path",
            help=(
                "Import path for a custom BaseVerifier subclass "
                "(module.path:ClassName). Replaces the default "
                "tests/test.sh runner for every trial in the run."
            ),
            rich_help_panel="Verifier",
            show_default=False,
        ),
    ] = None,
    verifier_kwargs: Annotated[
        list[str] | None,
        Option(
            "--verifier-kwarg",
            help=(
                "Constructor kwarg for the custom verifier in key=value "
                "format. Requires --verifier-import-path. Repeatable."
            ),
            rich_help_panel="Verifier",
            show_default=False,
        ),
    ] = None,
    # ── Overrides ────────────────────────────────────────────────────
    override_cpus: Annotated[
        int | None,
        Option(
            "--override-cpus",
            help="Override the number of CPUs for the environment.",
            rich_help_panel="Overrides",
            show_default=False,
        ),
    ] = None,
    override_memory_mb: Annotated[
        int | None,
        Option(
            "--override-memory-mb",
            help="Override the memory (MB) for the environment.",
            rich_help_panel="Overrides",
            show_default=False,
        ),
    ] = None,
    override_storage_mb: Annotated[
        int | None,
        Option(
            "--override-storage-mb",
            help="Override the storage (MB) for the environment.",
            rich_help_panel="Overrides",
            show_default=False,
        ),
    ] = None,
    override_gpus: Annotated[
        int | None,
        Option(
            "--override-gpus",
            help="Override the number of GPUs for the environment.",
            rich_help_panel="Overrides",
            show_default=False,
        ),
    ] = None,
    # ── Execution ────────────────────────────────────────────────────
    disable_verification: Annotated[
        bool,
        Option(
            "--disable-verification/--enable-verification",
            help="Skip verification scripts for faster agent debugging.",
            rich_help_panel="Execution",
            show_default=False,
        ),
    ] = False,
    yes: Annotated[
        bool,
        Option(
            "--yes",
            "-y",
            help="Auto-confirm prompts (e.g. env var access).",
            rich_help_panel="Execution",
            show_default=False,
        ),
    ] = False,
    scenario_path: Annotated[
        Path | None,
        Option(
            "--scenario-path",
            help="Directory containing scenario directories.",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    # ── AWS ──────────────────────────────────────────────────────────
    env_name: Annotated[
        str | None,
        Option(
            "--env-name",
            help="Testing environment name for AWS account resolution.",
            rich_help_panel="AWS",
            show_default=False,
        ),
    ] = None,
    verify: Annotated[
        bool | None,
        Option(
            "--verify-env/--no-verify-env",
            help="Verify AWS environment state before running trials (default: on).",
            rich_help_panel="AWS",
            show_default=False,
        ),
    ] = None,
):
    """Run benchmark trials against AWS environments."""
    # -- Load .env file -------------------------------------------------------
    if env_file is not None:
        if not env_file.exists():
            console.print(f"[red]Env file not found: {env_file}[/red]")
            raise typer.Exit(code=1)
        load_dotenv(env_file, override=True)

    # -- Build the job config from config file or CLI args --------------------
    if config_path is not None:
        if not config_path.exists():
            console.print(f"[red]Config file not found: {config_path}[/red]")
            raise typer.Exit(code=1)
        if config_path.suffix in (".yaml", ".yml"):
            config = AwsBenchJobConfig.model_validate(yaml.safe_load(config_path.read_text()))
        elif config_path.suffix == ".json":
            config = AwsBenchJobConfig.model_validate_json(config_path.read_text())
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    else:
        config = AwsBenchJobConfig()

    # Override only when the flag was passed, so a -c value survives. env_name
    # may come from either source; presence is validated below.
    if env_name is not None:
        config.env_name = env_name
    if verify is not None:
        config.verify = verify

    if config.env_name is None:
        console.print(
            "[red]No environment name. Pass --env-name <name> or set 'env_name' in -c.[/red]"
        )
        raise typer.Exit(code=1)

    # Apply CLI overrides
    if job_name is not None:
        config.job_name = job_name
    if jobs_dir is not None:
        config.jobs_dir = jobs_dir
    if n_attempts is not None:
        config.n_attempts = n_attempts
    if timeout_multiplier is not None:
        config.timeout_multiplier = timeout_multiplier
    if agent_timeout_multiplier is not None:
        config.agent_timeout_multiplier = agent_timeout_multiplier
    if verifier_timeout_multiplier is not None:
        config.verifier_timeout_multiplier = verifier_timeout_multiplier
    if agent_setup_timeout_multiplier is not None:
        config.agent_setup_timeout_multiplier = agent_setup_timeout_multiplier
    if environment_build_timeout_multiplier is not None:
        config.environment_build_timeout_multiplier = environment_build_timeout_multiplier
    if n_concurrent is not None:
        config.n_concurrent_trials = n_concurrent
    if quiet:
        config.quiet = quiet
    if debug:
        config.debug = debug
    # Set console level before create() resolves accounts/exports.
    if config.debug:
        set_console_level(logging.DEBUG)
    if max_retries is not None:
        config.retry.max_retries = max_retries
    if retry_include is not None:
        config.retry.include_exceptions = set(retry_include)
    if retry_exclude is not None:
        config.retry.exclude_exceptions = set(retry_exclude)

    # Agent config:
    # - With --agent or --agent-import-path: replace the entire agents list.
    # - Without: merge --ak/--ae into all existing agent configs.
    parsed_mcp_servers = [server for path in mcp_config or [] for server in load_mcp_servers(path)]

    if agent_name is not None or agent_import_path is not None:
        parsed_kwargs = parse_kwargs(agent_kwargs)
        parsed_env = parse_env_vars(agent_env)

        if model_names is not None:
            config.agents = [
                AgentConfig(
                    name=agent_name,
                    import_path=agent_import_path,
                    model_name=model_name,
                    skills=skills or [],
                    kwargs=parsed_kwargs,
                    env=parsed_env,
                    mcp_servers=parsed_mcp_servers,
                )
                for model_name in model_names
            ]
        else:
            config.agents = [
                AgentConfig(
                    name=agent_name,
                    import_path=agent_import_path,
                    skills=skills or [],
                    kwargs=parsed_kwargs,
                    env=parsed_env,
                    mcp_servers=parsed_mcp_servers,
                )
            ]
    else:
        parsed_kwargs = parse_kwargs(agent_kwargs)
        parsed_env = parse_env_vars(agent_env)
        if parsed_kwargs or parsed_env or parsed_mcp_servers or skills:
            for agent in config.agents:
                if parsed_kwargs:
                    agent.kwargs.update(parsed_kwargs)
                if parsed_env:
                    agent.env.update(parsed_env)
                if parsed_mcp_servers:
                    agent.mcp_servers.extend(parsed_mcp_servers)
                if skills:
                    agent.skills.extend(skills)

    # Environment config
    if environment_force_build is not None:
        config.environment.force_build = environment_force_build
    if environment_delete is not None:
        config.environment.delete = environment_delete
    if override_cpus is not None:
        config.environment.override_cpus = override_cpus
    if override_memory_mb is not None:
        config.environment.override_memory_mb = override_memory_mb
    if override_storage_mb is not None:
        config.environment.override_storage_mb = override_storage_mb
    if override_gpus is not None:
        config.environment.override_gpus = override_gpus
    if mounts is not None:
        config.environment.mounts = json.loads(mounts)
    if extra_docker_compose is not None:
        config.environment.extra_docker_compose = list(extra_docker_compose)
    if extra_instruction_paths is not None:
        config.extra_instruction_paths = list(extra_instruction_paths)

    # Verifier config
    if verifier_env is not None:
        config.verifier.env.update(parse_env_vars(verifier_env))
    if disable_verification:
        config.verifier.disable = True
    if verifier_import_path is not None:
        config.verifier.import_path = verifier_import_path
    if verifier_kwargs:
        config.verifier.kwargs.update(parse_kwargs(verifier_kwargs))

    # A CLI dataset source replaces config.dataset; otherwise the -c file's
    # dataset is used as-is.
    cli_dataset_given = (
        path is not None
        or dataset_name_version is not None
        or scenario_path is not None
        or registry_url is not None
        or registry_path is not None
        or include_task_names is not None
        or exclude_task_names is not None
        or n_tasks is not None
    )
    if cli_dataset_given:
        config.dataset = AwsBenchDatasetConfig.from_cli_args(
            scenario_path=scenario_path,
            dataset=dataset_name_version,
            registry_url=registry_url,
            registry_path=registry_path,
            path=path,
            include_task_names=include_task_names,
            exclude_task_names=exclude_task_names,
            n_tasks=n_tasks,
        )

    if metric is not None:
        config.metrics = [parse_metric(m) for m in metric]

    if config.dataset is None:
        console.print(
            "[red]No dataset provided. Pass -p <tasks-dir> + --scenario-path "
            "<scenarios-dir>, -d <name>[@<version>], or declare a dataset in -c.[/red]"
        )
        raise typer.Exit(code=1)

    config.dataset.validate_run()

    # -- Run the job ----------------------------------------------------------
    job_dir = config.jobs_dir / config.job_name
    job_dir.mkdir(parents=True, exist_ok=True)

    # Record config + job dir before any AWS work, so a run that dies inside
    # create() still leaves the config and snapshots the job dir.
    ledger = current_ledger(ctx.meta)
    ledger.set_resolved(
        job_id=None,
        job_dir=job_dir,
        is_resuming=False,
        resolved_config=config.model_dump(mode="json"),
    )
    ledger.register_job_dir(job_dir)

    async def _run_job():
        job = await AwsBenchJob.create(config)

        # Upgrade with the stable job id now that create() produced it.
        ledger.set_resolved(
            job_id=str(job.id),
            job_dir=job_dir,
            is_resuming=job.is_resuming,
            resolved_config=config.model_dump(mode="json"),
        )

        # Persist the resolved run identity before the verification gate, so a
        # verification failure leaves config.json (and verification.json) behind.
        job._job_config_path.write_text(config.model_dump_json(indent=4))

        if config.verify:
            assert config.test_environment is not None  # set by create()
            report = await ResourceManager.verify_environment(
                config.test_environment,
                job.run_scenarios,
                n_concurrent=_VERIFY_CONCURRENCY,
            )
            if not report.passed:
                # Print the verdict first, then persist the report best-effort:
                # a failed artifact write must not mask the verification failure.
                failed = sorted(
                    {(r.environment_id, r.account_id) for r in report.results if not r.success}
                )
                console.print(
                    f"[red]Verification failed for {len(failed)} of "
                    f"{len(report.results)} account(s):[/red]"
                )
                for environment_id, account_id in failed:
                    console.print(f"  - {environment_id} (account {account_id})")
                report_path = job.job_dir / "verification.json"
                try:
                    report_path.write_text(json.dumps(asdict(report), indent=4))
                    console.print(f"See {report_path} for details.")
                except OSError as exc:
                    logger.warning("Could not write %s: %s", report_path, exc)
                console.print(
                    "Run 'aws-bench env reset' to fix, or pass --no-verify-env to bypass."
                )
                raise EnvironmentVerifyError(
                    f"Verification failed; see {job.job_dir / 'verification.json'}"
                )

        _confirm_host_env_access(job, console, skip_confirm=yes)

        # Run
        job_result = await job.run()

        # Results summary
        console.print()
        print_job_results_tables(job_result)
        console.print("[bold]Job Info[/bold]")
        console.print(
            f"Total runtime: {_format_duration(job_result.started_at, job_result.finished_at)}"
        )
        console.print(f"Results written to {job._job_result_path}")

    try:
        # 2nd/3rd handlers route Harbor's run tree to job.log + run.log.
        with (
            file_logging(job_dir / "job.log"),
            file_logging(job_dir / "job.log", logger_name=_HARBOR_RUN_LOGGER_NAME),
            _maybe_harbor_into_runlog(ledger.run_log_path),
        ):
            # Preflight inside the span so the AWS identity line lands in job.log.
            preflight_docker_cli()
            plugins = preflight_docker_daemon()
            preflight_docker_plugins(plugins)
            preflight_aws_credentials(CredentialProvider.get(), ou_name=config.env_name)
            run_async(_run_job())
    except Exception as exc:
        print_exception(console, debug=debug)
        raise typer.Exit(code=1) from exc


jobs_app.command()(start)
