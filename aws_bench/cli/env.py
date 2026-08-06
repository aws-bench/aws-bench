"""CLI commands for testing environment management."""

import asyncio
import logging
import os
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.panel import Panel
from typer import Option

from aws_bench.account_management.constants import SCENARIO_ACCOUNT_TAG_KEY
from aws_bench.account_management.exceptions import (
    AccountResolutionError,
    DuplicateScenarioAccountError,
    TestEnvironmentNotFoundError,
)
from aws_bench.account_management.manager import AccountManager
from aws_bench.account_management.models import ScenarioAccount
from aws_bench.account_management.organizations import OrganizationsClient
from aws_bench.cli.display import (
    display_cleanup_results,
    display_provisioning_summary,
    display_reset_results,
    display_setup_summary,
    display_snapshot_results,
    display_verify_results,
    render_account_state,
    render_env_header,
    render_env_list,
)
from aws_bench.cli.preflight import (
    PreflightError,
    preflight_aws_credentials,
    preflight_docker_cli,
    preflight_docker_daemon,
)
from aws_bench.cli.scenario_progress import (
    provision_scenarios_with_progress,
    run_phase_with_progress,
    snapshot_env_with_progress,
)
from aws_bench.cli.shared_options import (
    DatasetOption,
    DebugOption,
    DeleteOption,
    ExcludeScenariosOption,
    ForceBuildOption,
    IncludeScenariosOption,
    JobNameOption,
    JobsDirOption,
    MaxRetriesOption,
    NConcurrentOption,
    OverrideBuildTimeoutSecOption,
    OverrideCpusOption,
    OverrideMemoryMbOption,
    QuietOption,
    RegistryPathOption,
    RegistryUrlOption,
    RetryExcludeOption,
    RetryIncludeOption,
    ScenarioPathOption,
    TimeoutMultiplierOption,
)
from aws_bench.cli.ui import TeeConsole, console
from aws_bench.constants import OUTPUT_DIR
from aws_bench.dataset.config import AwsBenchDatasetConfig
from aws_bench.logging.ledger import current_ledger
from aws_bench.logging.logger import file_logging, get_logger, log_context, set_console_level
from aws_bench.resource_management.ccapi.models import MAX_WORKERS_ACCOUNT
from aws_bench.resource_management.manager import list_account_stacks
from aws_bench.resource_management.models import QuotaStatus
from aws_bench.resource_management.quota_manager import QuotaManager
from aws_bench.resource_management.snapshot.models import SnapshotResult
from aws_bench.scenario.events import ScenarioPhase
from aws_bench.scenario.exceptions import InsufficientQuotaError
from aws_bench.scenario.job import ScenarioJob
from aws_bench.scenario.job_config import ScenarioJobConfig
from aws_bench.scenario.provisioning import ProvisioningSummary, provision_scenarios
from aws_bench.scenario.results import ScenarioJobResult
from aws_bench.scenario.trial_paths import ScenarioJobPaths
from aws_bench.utils.bedrock_credentials import BedrockCredentialError, generate_bearer_token
from aws_bench.utils.concurrent import interruptible_executor
from aws_bench.utils.credentials_provider import CredentialProvider
from aws_bench.utils.error_display import print_exception

logger = get_logger(__name__)

env_app = typer.Typer(no_args_is_help=True)

QUOTA_STATUS_POLLING_INTERVAL = 60
DEFAULT_QUOTA_STATUS_POLLING_TIMEOUT = 1800


def _apply_debug(debug: bool) -> None:
    """Raise the console log level to DEBUG for ``--debug`` (run.log is always DEBUG)."""
    if debug:
        set_console_level(logging.DEBUG)


def _describe_org_account_quota(quota_manager: QuotaManager) -> str:
    """Return a colorized one-line summary of the org account-limit increase request.

    Answers "is an increase pending?" for ``env show`` by reading the Service
    Quotas request history (organizations / L-E619E033) via
    ``diagnose_org_account_quota``. Resilient: any lookup failure renders as
    ``unknown`` so ``env show`` never breaks on it.
    """
    try:
        status, reason = quota_manager.diagnose_org_account_quota()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read org account-quota status: %s", exc)
        return "[dim]unknown[/dim]"
    # is_pending: REQUESTED / ALREADY_PENDING -> in review.
    # is_success: ALREADY_MET / APPROVED / CASE_CLOSED -> granted.
    # DENIED: rejected (surfaced distinctly, not hidden like "no request").
    # FAILED: no request on file, or the history read failed.
    if status.is_pending:
        return f"[yellow]{reason}[/yellow]"
    if status.is_success:
        return f"[green]{reason}[/green]"
    if status == QuotaStatus.DENIED:
        return f"[red]{reason}[/red]"
    return f"[dim]{reason}[/dim]"


async def _run_job_phase(
    job_cfg: ScenarioJobConfig,
    phase: ScenarioPhase,
    cred_provider: CredentialProvider,
    *,
    quiet: bool,
) -> ScenarioJobResult:
    """Run a scenario job phase with optional progress display.

    Helper for CLI commands that run phases (setup, reset, verify, cleanup).
    The caller owns the ``job.log`` span (opened at command entry so preflight
    and dataset validation are captured too); this just creates + runs the job.
    """
    job = await ScenarioJob.create(job_cfg, cred_provider)
    if quiet:
        return await job.run(phase)
    return await run_phase_with_progress(job, phase)


def _build_env_dataset_config(
    scenario_path: Path | None,
    dataset: str | None,
    registry_url: str | None,
    registry_path: Path | None,
    include_scenarios: list[str] | None,
    exclude_scenarios: list[str] | None,
    *,
    debug: bool = False,
) -> AwsBenchDatasetConfig:
    """Build + validate the scenarios-only dataset config for an env command.

    Construction (the local-xor-registry source rule) and ``validate_env`` (no
    tasks dir allowed for env) are usage errors, so they are rendered and exit
    here — before the ``job.log`` span opens — rather than inside it. Shared by
    every env command that takes a dataset.
    """
    try:
        cfg = AwsBenchDatasetConfig.from_cli_args(
            scenario_path,
            dataset,
            registry_url,
            registry_path,
            include_scenarios,
            exclude_scenarios,
        )
        cfg.validate_env()
    except Exception as exc:
        print_exception(console, debug=debug)
        raise typer.Exit(code=1) from exc
    return cfg


def _run_phase_command(
    job_cfg: ScenarioJobConfig,
    phase: ScenarioPhase,
    ctx: typer.Context,
    *,
    quiet: bool,
) -> ScenarioJobResult:
    """Run one phase command end to end: log span, preflight, job.

    The shared scaffold for ``setup`` / ``verify`` / ``reset`` / ``cleanup``.
    All four run scenario containers, so all four fail fast on the same
    preconditions (Docker available + reachable, AWS credentials valid) and
    open ``job.log`` from the start — the span covers preflight (incl. the AWS
    identity line) and ``create`` + ``run``. The dataset config is built and
    validated by the caller before the span opens.

    Exceptions propagate to the caller, which renders them (via
    ``print_exception``) and maps them to an exit code; setup additionally
    attaches a next-step hint to quota / account-resolution failures. Returns
    the job result; the caller sets the exit code from its pass/fail state.
    """
    paths = ScenarioJobPaths(job_dir=job_cfg.jobs_dir / job_cfg.job_name)
    paths.mkdir()

    # Beat 2: record config + job dir before any AWS work, so a phase that dies
    # inside create()/run() still leaves the config and snapshots the job dir.
    # Phases have no stable job id.
    ledger = current_ledger(ctx.meta)
    ledger.set_resolved(
        job_id=None,
        job_dir=paths.job_dir,
        is_resuming=False,
        resolved_config=job_cfg.model_dump(mode="json"),
    )
    ledger.register_job_dir(paths.job_dir)

    with file_logging(paths.log_path):
        preflight_docker_cli()
        preflight_docker_daemon()
        preflight_aws_credentials(CredentialProvider.get(), ou_name=job_cfg.ou_name)
        return asyncio.run(_run_job_phase(job_cfg, phase, CredentialProvider.get(), quiet=quiet))


@env_app.command()
def init(
    ctx: typer.Context,
    ou_name: Annotated[
        str,
        Option(
            "--env-name",
            help="Name of the testing environment.",
            rich_help_panel="Scenario Source",
        ),
    ],
    # Scenario source
    scenario_path: ScenarioPathOption = None,
    dataset: DatasetOption = None,
    registry_url: RegistryUrlOption = None,
    registry_path: RegistryPathOption = None,
    include_scenarios: IncludeScenariosOption = None,
    exclude_scenarios: ExcludeScenariosOption = None,
    n_concurrent: Annotated[
        int,
        Option(
            "-n",
            "--n-concurrent",
            help="Number of accounts to provision in parallel.",
            rich_help_panel="Init Settings",
        ),
    ] = 4,
    quiet: Annotated[
        bool,
        Option(
            "-q",
            "--quiet",
            "--silent",
            help="Suppress the rich live progress display; falls back to log output.",
            rich_help_panel="Init Settings",
            show_default=False,
        ),
    ] = False,
    debug: DebugOption = False,
    wait_for_quotas: Annotated[
        bool,
        Option(
            "--wait-for-quotas",
            help=(
                "Poll until every requested quota's current effective value "
                "meets the scenario's desired value. Exits non-zero on timeout."
            ),
            rich_help_panel="Init Settings",
            show_default=False,
        ),
    ] = False,
    quota_timeout: Annotated[
        int,
        Option(
            "--quota-timeout",
            help=(
                "Max time (in SECONDS) to wait for quota approval before exiting "
                "non-zero. Used with --wait-for-quotas."
            ),
            rich_help_panel="Init Settings",
        ),
    ] = DEFAULT_QUOTA_STATUS_POLLING_TIMEOUT,
    poll_interval: Annotated[
        int,
        Option(
            "--poll-interval",
            help=("Time (in SECONDS) between quota-status polls. Used with --wait-for-quotas."),
            rich_help_panel="Init Settings",
        ),
    ] = QUOTA_STATUS_POLLING_INTERVAL,
) -> None:
    """Initialize the testing environment: provision accounts and submit quotas."""
    _apply_debug(debug)
    dataset_cfg = _build_env_dataset_config(
        scenario_path,
        dataset,
        registry_url,
        registry_path,
        include_scenarios,
        exclude_scenarios,
        debug=debug,
    )

    # init provisions accounts directly — no job dir. After provisioning it
    # captures the pristine PRE_SETUP baseline (below); record the resolved
    # scenario source as the invocation's config.
    current_ledger(ctx.meta).set_resolved(
        job_id=None,
        job_dir=None,
        is_resuming=False,
        resolved_config=dataset_cfg.model_dump(mode="json"),
    )

    cred_provider = CredentialProvider.get()
    account_manager = AccountManager()

    async def _run() -> ProvisioningSummary:
        preflight_aws_credentials(cred_provider, ou_name=ou_name)
        account_manager.init_organization(ou_name)
        scenarios = list((await dataset_cfg.get_scenarios()).values())
        # provision_scenarios captures each account's PRE_SETUP baseline as its
        # final step, so the summary already reflects snapshot outcomes.
        if quiet:
            return await provision_scenarios(
                scenarios,
                ou_name,
                account_manager=account_manager,
                quota_manager=QuotaManager(cred_provider),
                cred_provider=cred_provider,
                n_concurrent=n_concurrent,
                wait_for_quotas=wait_for_quotas,
                quota_timeout=quota_timeout,
                poll_interval=poll_interval,
            )
        return await provision_scenarios_with_progress(
            scenarios,
            ou_name,
            account_manager=account_manager,
            quota_manager=QuotaManager(cred_provider),
            n_concurrent=n_concurrent,
            wait_for_quotas=wait_for_quotas,
            quota_timeout=quota_timeout,
            poll_interval=poll_interval,
            cred_provider=cred_provider,
        )

    try:
        summary = asyncio.run(_run())
    except Exception as exc:
        print_exception(console, debug=debug)
        raise typer.Exit(code=1) from exc

    display_provisioning_summary(summary)
    if not summary.all_succeeded:
        raise typer.Exit(code=1)


@env_app.command()
def setup(
    ctx: typer.Context,
    ou_name: Annotated[
        str,
        Option(
            "--env-name",
            help="Name of the testing environment.",
            rich_help_panel="Scenario Source",
        ),
    ],
    # Scenario source
    scenario_path: ScenarioPathOption = None,
    dataset: DatasetOption = None,
    registry_url: RegistryUrlOption = None,
    registry_path: RegistryPathOption = None,
    include_scenarios: IncludeScenariosOption = None,
    exclude_scenarios: ExcludeScenariosOption = None,
    # Job settings
    job_name: JobNameOption = None,
    jobs_dir: JobsDirOption = None,
    n_concurrent: NConcurrentOption = None,
    timeout_multiplier: TimeoutMultiplierOption = None,
    quiet: QuietOption = False,
    debug: DebugOption = False,
    max_retries: MaxRetriesOption = None,
    retry_include: RetryIncludeOption = None,
    retry_exclude: RetryExcludeOption = None,
    # Environment
    force_build: ForceBuildOption = None,
    delete: DeleteOption = None,
    override_cpus: OverrideCpusOption = None,
    override_memory_mb: OverrideMemoryMbOption = None,
    override_build_timeout_sec: OverrideBuildTimeoutSecOption = None,
    # Setup-specific
    mounts: Annotated[
        str | None,
        Option(
            "--mounts",
            help=(
                "JSON array of mounts to apply to the scenario container, "
                "REPLACING the manifest's mounts. "
                'E.g. \'[{"type":"bind","source":"/host/path","target":"/container/path"}]\''
            ),
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Deploy scenario environments to provisioned AWS accounts."""
    _apply_debug(debug)
    dataset_cfg = _build_env_dataset_config(
        scenario_path,
        dataset,
        registry_url,
        registry_path,
        include_scenarios,
        exclude_scenarios,
        debug=debug,
    )

    job_cfg = ScenarioJobConfig.from_cli_args(
        ou_name,
        dataset_cfg,
        job_name,
        jobs_dir,
        n_concurrent,
        timeout_multiplier,
        force_build,
        delete,
        override_cpus,
        override_memory_mb,
        override_build_timeout_sec,
        mounts,
        max_retries,
        retry_include,
        retry_exclude,
    )

    # Setup adds remediation hints: quota / account-resolution failures point the
    # operator at `env init`; everything else falls through to the generic render.
    try:
        result = _run_phase_command(job_cfg, ScenarioPhase.DEPLOY, ctx=ctx, quiet=quiet)
    except InsufficientQuotaError as exc:
        source = (
            f"--scenario-path {scenario_path}"
            if scenario_path is not None
            else f"--dataset {dataset}"
        )
        print_exception(console, debug=debug)
        console.print(
            f"Run `aws-bench env init --env-name {ou_name} {source} --wait-for-quotas` "
            f"and wait for AWS approval."
        )
        raise typer.Exit(code=1) from exc
    except (AccountResolutionError, DuplicateScenarioAccountError) as exc:
        print_exception(console, debug=debug)
        console.print(
            f"Run `aws-bench env init --env-name {ou_name}` to provision the missing accounts."
        )
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        print_exception(console, debug=debug)
        raise typer.Exit(code=1) from exc

    display_setup_summary(result)
    if not result.all_passed:
        raise typer.Exit(code=1)


@env_app.command()
def show(
    name: str = typer.Option(..., "--env-name", help="Name of the testing environment."),
    debug: DebugOption = False,
) -> None:
    """Show the current testing environment state including accounts, quotas, and stack status."""
    _apply_debug(debug)
    account_manager = AccountManager()
    if account_manager.is_preexisting is True:
        environment = account_manager.resolve_test_environment(name)
        org_info = environment.org
        ou_id = environment.ou_id
    else:
        org_info = account_manager._org.get_org_info()
        ou_id = account_manager._require_ou(org_info, name)
    cred_provider = CredentialProvider.get()
    quota_manager = QuotaManager(cred_provider)

    # DEBUG detail lands in run.log (console is INFO) so the invocation is on record.
    logger.debug("Showing environment '%s' (%s) in org %s", name, ou_id, org_info.org_id)

    # Org-level: surface whether an account-limit increase request is pending, so
    # operators can tell why account creation is (or would be) blocked.
    account_quota_line = (
        "[dim]externally managed[/dim]"
        if account_manager.is_preexisting is True
        else _describe_org_account_quota(quota_manager)
    )

    accounts = account_manager.list_scenario_accounts(name)
    if not accounts:
        logger.debug("No scenario accounts found for environment '%s'", name)
        render_env_header(
            console,
            org_info.management_account_id,
            org_info.org_id,
            name,
            ou_id,
            account_quota=account_quota_line,
        )
        console.print("  No scenario accounts found.")
        return

    logger.debug("Found %d scenario account(s) in '%s'", len(accounts), name)

    def _account_panel(acct: ScenarioAccount) -> Panel:
        # log_context rides a contextvar into the workers, tagging every line
        # beneath (incl. the collectors' per-region context) [account][region].
        with log_context(f"{acct.scenario_name}/{acct.account_tag} ({acct.account_id})"):
            # Quotas (service-quotas) and stacks (cloudformation) are independent
            # services, so run them concurrently.
            with interruptible_executor(max_workers=2) as inner:
                quotas_future = inner.submit(
                    quota_manager.collect_requested_quotas, acct.account_id
                )
                stacks_future = inner.submit(list_account_stacks, acct.account_id, cred_provider)
                quota_entries, quota_error = quotas_future.result()
                stacks = stacks_future.result()

            unmet = sum(1 for e in quota_entries if not e.is_met)
            if quota_error:
                logger.debug("quota lookup failed: %s", quota_error)
            logger.debug(
                "%d quota entr(ies) (%d unmet), %d stack(s)", len(quota_entries), unmet, len(stacks)
            )
            for e in quota_entries:
                logger.debug(
                    "quota %s [%s] %s requested=%s current=%s %s",
                    e.quota_id,
                    e.region,
                    e.name or "?",
                    e.requested,
                    e.current,
                    "met" if e.is_met else "UNMET",
                )
            if stacks:
                logger.debug("stacks: %s", ", ".join(s.get("name", "?") for s in stacks))
            return render_account_state(acct, quota_entries, quota_error, stacks)

    # Fan out (accounts are independent throttle buckets), collecting all panels
    # before printing so --debug log lines never split the rendered block.
    with interruptible_executor(max_workers=min(len(accounts), MAX_WORKERS_ACCOUNT)) as executor:
        panels = list(executor.map(_account_panel, accounts))

    render_env_header(
        console,
        org_info.management_account_id,
        org_info.org_id,
        name,
        ou_id,
        len(accounts),
        account_quota=account_quota_line,
    )
    for panel in panels:
        console.print(panel)


@env_app.command("list")
def list_envs(
    debug: DebugOption = False,
) -> None:
    """List all test environments in the organization."""
    _apply_debug(debug)
    cred_provider = CredentialProvider.get()
    try:
        preflight_aws_credentials(cred_provider)
    except PreflightError as exc:
        print_exception(console, debug=debug)
        raise typer.Exit(code=1) from exc

    account_manager = AccountManager()
    if account_manager.is_preexisting is True:
        assert account_manager._preexisting is not None
        configured = account_manager._preexisting
        render_env_list(
            console,
            "preexisting",
            cred_provider.get_caller_account_id(),
            [
                {
                    "name": configured.name,
                    "ou_id": "preexisting",
                    "account_count": sum(len(tags) for tags in configured.accounts.values()),
                }
            ],
        )
        return

    org_client = OrganizationsClient()
    try:
        org_info = org_client.get_org_info()
        ous = org_client.list_ous(org_info.root_id)
    except Exception as exc:
        print_exception(console, debug=debug)
        raise typer.Exit(code=1) from exc

    if not ous:
        console.print("No test environments found.")
        return

    def _count_active(ou: dict) -> dict:
        try:
            accounts = org_client.list_accounts_in_ou(ou["Id"])
            count = sum(1 for a in accounts if a["Status"] == "ACTIVE")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to list accounts for OU %s: %s", ou["Name"], exc)
            count = None
        return {"name": ou["Name"], "ou_id": ou["Id"], "account_count": count}

    with interruptible_executor(max_workers=min(len(ous), MAX_WORKERS_ACCOUNT)) as executor:
        environments = list(executor.map(_count_active, ous))

    render_env_list(console, org_info.org_id, org_info.management_account_id, environments)


@env_app.command()
def snapshot(
    ou_name: Annotated[
        str,
        Option(
            "--env-name",
            help="Name of the testing environment.",
        ),
    ],
    output: Annotated[
        Path | None,
        Option(
            "--output",
            "-o",
            help=(
                "Directory to write the snapshot JSON files to (one per account, "
                "grouped into a <scenario-name>/ subdirectory per scenario in the "
                "environment). Defaults to ~/.aws-bench/snapshots."
            ),
        ),
    ] = None,
    # Scenario source
    scenario_path: ScenarioPathOption = None,
    dataset: DatasetOption = None,
    registry_url: RegistryUrlOption = None,
    registry_path: RegistryPathOption = None,
    include_scenarios: IncludeScenariosOption = None,
    exclude_scenarios: ExcludeScenariosOption = None,
    n_concurrent: NConcurrentOption = None,
    debug: DebugOption = False,
) -> None:
    """Capture an on-demand observability snapshot of scenario accounts.

    Captures the current resource state of each account for inspection/debugging
    and writes it to a JSON file under the output directory (one file per
    account). This is purely observational — it has no lifecycle preconditions, is
    not stored in the S3 state bucket, and is not consumed by
    setup/verify/reset/cleanup.

    Accounts are resolved from the scenario source: each source scenario's declared
    account_tags are matched to the OU's tagged accounts, and each account is scanned
    only across that scenario's declared regions (not every enabled region) — the
    region-guardrail SCP denies CloudFormation outside those regions, so scanning
    them would waste calls and be denied. A scenario source (``--scenario-path`` or
    ``-d``) is therefore required, matching the other env commands; a source scenario
    with no provisioned account fails loud.
    """
    _apply_debug(debug)
    cred_provider = CredentialProvider.get()
    try:
        preflight_aws_credentials(cred_provider)
    except PreflightError as exc:
        print_exception(console, debug=debug)
        raise typer.Exit(code=1) from exc

    dataset_cfg = _build_env_dataset_config(
        scenario_path,
        dataset,
        registry_url,
        registry_path,
        include_scenarios,
        exclude_scenarios,
        debug=debug,
    )

    output_dir = output if output is not None else OUTPUT_DIR / "snapshots"
    resolved_n_concurrent = (
        n_concurrent
        if n_concurrent is not None
        else int(ScenarioJobConfig.model_fields["n_concurrent"].default)
    )

    async def _capture() -> dict[str, list[SnapshotResult]]:
        scenarios = await dataset_cfg.get_scenarios()
        return await snapshot_env_with_progress(
            list(scenarios.values()),
            ou_name,
            output_dir,
            n_concurrent=resolved_n_concurrent,
        )

    try:
        results = asyncio.run(_capture())
    except Exception as exc:
        print_exception(console, debug=debug)
        raise typer.Exit(code=1) from exc

    if not results:
        console.print("No scenario accounts found.")
        raise typer.Exit(code=0)

    if display_snapshot_results(results):
        raise typer.Exit(code=1)


@env_app.command()
def cleanup(
    ctx: typer.Context,
    name: str = typer.Option(..., "--env-name", help="Name of the testing environment."),
    # Scenario source
    scenario_path: ScenarioPathOption = None,
    dataset: DatasetOption = None,
    registry_url: RegistryUrlOption = None,
    registry_path: RegistryPathOption = None,
    include_scenarios: IncludeScenariosOption = None,
    exclude_scenarios: ExcludeScenariosOption = None,
    # Job settings
    job_name: JobNameOption = None,
    jobs_dir: JobsDirOption = None,
    n_concurrent: NConcurrentOption = None,
    timeout_multiplier: TimeoutMultiplierOption = None,
    quiet: QuietOption = False,
    max_retries: MaxRetriesOption = None,
    retry_include: RetryIncludeOption = None,
    retry_exclude: RetryExcludeOption = None,
    debug: DebugOption = False,
    # Environment
    force_build: ForceBuildOption = None,
    delete: DeleteOption = None,
    override_cpus: OverrideCpusOption = None,
    override_memory_mb: OverrideMemoryMbOption = None,
    override_build_timeout_sec: OverrideBuildTimeoutSecOption = None,
    # Cleanup-specific
    yes: Annotated[
        bool,
        Option(
            "-y",
            "--yes",
            help="Skip confirmation prompt before cleanup.",
            show_default=False,
        ),
    ] = False,
) -> None:
    """Clean up all CloudFormation stacks across scenario accounts and delete their snapshots."""
    _apply_debug(debug)
    # Confirmation prompt (interactive, before the job.log span).
    if not yes:
        prompt = "About to delete all resources. This is irreversible. Continue?"
        if not typer.confirm(prompt, default=False):
            console.print("Cleanup cancelled.")
            raise typer.Exit(code=0)

    dataset_cfg = _build_env_dataset_config(
        scenario_path,
        dataset,
        registry_url,
        registry_path,
        include_scenarios,
        exclude_scenarios,
        debug=debug,
    )
    job_cfg = ScenarioJobConfig.from_cli_args(
        name,
        dataset_cfg,
        job_name,
        jobs_dir,
        n_concurrent,
        timeout_multiplier,
        force_build,
        delete,
        override_cpus,
        override_memory_mb,
        override_build_timeout_sec,
        None,  # mounts (setup-only)
        max_retries,
        retry_include,
        retry_exclude,
    )

    try:
        result = _run_phase_command(job_cfg, ScenarioPhase.CLEANUP, ctx=ctx, quiet=quiet)
    except Exception as exc:
        print_exception(console, debug=debug)
        raise typer.Exit(code=1) from exc

    has_failures = display_cleanup_results(result)
    if has_failures or not result.all_passed:
        raise typer.Exit(code=1)


@env_app.command()
def verify(
    ctx: typer.Context,
    name: str = typer.Option(..., "--env-name", help="Name of the testing environment."),
    # Scenario source
    scenario_path: ScenarioPathOption = None,
    dataset: DatasetOption = None,
    registry_url: RegistryUrlOption = None,
    registry_path: RegistryPathOption = None,
    include_scenarios: IncludeScenariosOption = None,
    exclude_scenarios: ExcludeScenariosOption = None,
    # Job settings
    job_name: JobNameOption = None,
    jobs_dir: JobsDirOption = None,
    n_concurrent: NConcurrentOption = None,
    timeout_multiplier: TimeoutMultiplierOption = None,
    quiet: QuietOption = False,
    max_retries: MaxRetriesOption = None,
    retry_include: RetryIncludeOption = None,
    retry_exclude: RetryExcludeOption = None,
    debug: DebugOption = False,
    # Environment
    force_build: ForceBuildOption = None,
    delete: DeleteOption = None,
    override_cpus: OverrideCpusOption = None,
    override_memory_mb: OverrideMemoryMbOption = None,
    override_build_timeout_sec: OverrideBuildTimeoutSecOption = None,
) -> None:
    """Verify that accounts match post-setup baseline state."""
    _apply_debug(debug)
    dataset_cfg = _build_env_dataset_config(
        scenario_path,
        dataset,
        registry_url,
        registry_path,
        include_scenarios,
        exclude_scenarios,
        debug=debug,
    )
    job_cfg = ScenarioJobConfig.from_cli_args(
        name,
        dataset_cfg,
        job_name,
        jobs_dir,
        n_concurrent,
        timeout_multiplier,
        force_build,
        delete,
        override_cpus,
        override_memory_mb,
        override_build_timeout_sec,
        None,  # mounts (setup-only)
        max_retries,
        retry_include,
        retry_exclude,
    )

    try:
        result = _run_phase_command(job_cfg, ScenarioPhase.VERIFY, ctx=ctx, quiet=quiet)
    except Exception as exc:
        print_exception(console, debug=debug)
        raise typer.Exit(code=1) from exc

    verify_results = result.get_verify_results()
    has_failures = display_verify_results(verify_results)
    if has_failures or not result.all_passed:
        raise typer.Exit(code=1)


@env_app.command()
def reset(
    ctx: typer.Context,
    name: str = typer.Option(..., "--env-name", help="Name of the testing environment."),
    # Scenario source
    scenario_path: ScenarioPathOption = None,
    dataset: DatasetOption = None,
    registry_url: RegistryUrlOption = None,
    registry_path: RegistryPathOption = None,
    include_scenarios: IncludeScenariosOption = None,
    exclude_scenarios: ExcludeScenariosOption = None,
    # Job settings
    job_name: JobNameOption = None,
    jobs_dir: JobsDirOption = None,
    n_concurrent: NConcurrentOption = None,
    timeout_multiplier: TimeoutMultiplierOption = None,
    quiet: QuietOption = False,
    max_retries: MaxRetriesOption = None,
    retry_include: RetryIncludeOption = None,
    retry_exclude: RetryExcludeOption = None,
    debug: DebugOption = False,
    # Environment
    force_build: ForceBuildOption = None,
    delete: DeleteOption = None,
    override_cpus: OverrideCpusOption = None,
    override_memory_mb: OverrideMemoryMbOption = None,
    override_build_timeout_sec: OverrideBuildTimeoutSecOption = None,
    # Reset-specific
    yes: Annotated[
        bool,
        Option(
            "-y",
            "--yes",
            help="Skip confirmation prompt before reset.",
            show_default=False,
        ),
    ] = False,
) -> None:
    """Reset accounts to post-setup baseline state."""
    _apply_debug(debug)
    # Build + validate the dataset config first, so a bad --scenario-path / -d
    # fails before the (interactive) confirmation prompt.
    dataset_cfg = _build_env_dataset_config(
        scenario_path,
        dataset,
        registry_url,
        registry_path,
        include_scenarios,
        exclude_scenarios,
        debug=debug,
    )

    # Confirmation prompt (interactive, before the job.log span).
    if not yes:
        if not typer.confirm(
            "About to reset accounts. This will:\n"
            "  - Delete resources created after setup\n"
            "  - Revert stack drift to baseline\n"
            "  - Automatically re-deploy any stack that can't be reverted\n"
            "Continue?",
            default=False,
        ):
            console.print("Reset cancelled.")
            raise typer.Exit(code=0)

    job_cfg = ScenarioJobConfig.from_cli_args(
        name,
        dataset_cfg,
        job_name,
        jobs_dir,
        n_concurrent,
        timeout_multiplier,
        force_build,
        delete,
        override_cpus,
        override_memory_mb,
        override_build_timeout_sec,
        None,  # mounts (setup-only)
        max_retries,
        retry_include,
        retry_exclude,
    )

    # Redeploy is handled automatically within the reset trial.
    try:
        result = _run_phase_command(job_cfg, ScenarioPhase.RESET, ctx=ctx, quiet=quiet)
    except Exception as exc:
        print_exception(console, debug=debug)
        raise typer.Exit(code=1) from exc

    reset_results = result.get_reset_results()
    has_failures = display_reset_results(reset_results)
    if has_failures or not result.all_passed:
        raise typer.Exit(code=1)


@env_app.command()
def terminate(
    name: str = typer.Option(
        ..., "--env-name", help="Name of the testing environment to terminate."
    ),
    include_scenarios: IncludeScenariosOption = None,
    exclude_scenarios: ExcludeScenariosOption = None,
    no_close: Annotated[
        bool,
        Option(
            "--no-close",
            help="Remove scenario tags and move accounts to root, but don't close them.",
            show_default=False,
        ),
    ] = False,
    debug: DebugOption = False,
) -> None:
    """Terminate the testing environment: close all accounts and clean up.

    Accounts enter a 90-day suspension period. This action is irreversible
    after the suspension period expires.
    """
    _apply_debug(debug)
    account_manager = AccountManager()

    try:
        org_info = account_manager._org.get_org_info()
        ou_id = account_manager._require_ou(org_info, name)
    except TestEnvironmentNotFoundError:
        console.print(f"Testing environment '{name}' not found.")
        raise typer.Exit(code=1)

    accounts = account_manager._org.list_accounts_in_ou(ou_id)
    active_accounts = [a for a in accounts if a["Status"] == "ACTIVE"]

    # Apply include/exclude scenario filters
    if include_scenarios or exclude_scenarios:
        filtered: list[dict] = []
        for acct in active_accounts:
            tags = account_manager._org.get_tags(acct["Id"])
            tag_value = tags.get(SCENARIO_ACCOUNT_TAG_KEY, "")
            scenario_name = tag_value.split("/", 1)[0] if "/" in tag_value else tag_value
            if include_scenarios and not any(fnmatch(scenario_name, p) for p in include_scenarios):
                continue
            if exclude_scenarios and any(fnmatch(scenario_name, p) for p in exclude_scenarios):
                continue
            filtered.append(acct)
        active_accounts = filtered

    if not active_accounts:
        console.print(f"No active accounts in environment '{name}'. Cleaning up...")
        account_manager._org.detach_all_scps(ou_id)
        try:
            account_manager._org.delete_organizational_unit(ou_id)
            console.print(f"Deleted environment '{name}' ({ou_id}).")
        except Exception as e:
            console.print(f"Could not delete environment: {escape(str(e))}")
            raise typer.Exit(code=1)
        return

    console.print(f"Environment: {name} ({ou_id})")
    console.print(f"Accounts to {'untag' if no_close else 'close'} ({len(active_accounts)}):")
    for acct in active_accounts:
        console.print(f"  {acct['Id']}  {acct.get('Name', '')}")

    action = "UNTAG (no-close)" if no_close else "CLOSE"
    console.print(
        f"\n⚠️  This will {action} {len(active_accounts)} account(s) and DELETE the environment.\n"
        "Accounts enter a 90-day suspension period. This is irreversible after that period.\n"
        "Note: Run 'aws-bench env cleanup' first to remove deployed resources before terminating."
    )
    if not typer.confirm("Are you sure you want to terminate?", default=False):
        console.print("Terminate cancelled.")
        return

    account_filter = [a["Id"] for a in active_accounts]
    try:
        results = account_manager.terminate_environment(
            name, no_close=no_close, account_filter=account_filter
        )
    except Exception as exc:
        print_exception(console, debug=debug)
        raise typer.Exit(code=1) from exc

    console.print("\nResults:")
    for account_id, status in results.items():
        # status embeds a botocore error on FAILED — escape its brackets.
        console.print(f"  {account_id}: {escape(status)}")

    failed = sum(1 for s in results.values() if "FAILED" in s)
    if failed:
        console.print(f"\n{failed} account(s) failed to close.")
        raise typer.Exit(code=1)
    else:
        console.print(f"\nTerminated environment '{name}' successfully.")


@env_app.command("creds")
def creds(
    output: Annotated[
        Path | None,
        Option("--output", "-o", help="Write to file (dotenv format) instead of stdout"),
    ] = None,
    force: Annotated[bool, Option(help="Regenerate even if cached creds are valid")] = False,
    no_verify: Annotated[bool, Option("--no-verify", help="Skip credential verification")] = False,
    days: Annotated[int, Option(help="Credential lifetime in days")] = 30,
    eval_mode: Annotated[
        bool, Option("--eval", help="Output eval-friendly format (no comments)")
    ] = False,
) -> None:
    """Generate Bedrock bearer token for agent environments."""
    # Status → stderr (TeeConsole → run.log too); stdout carries only the export
    # line for ``eval $(...)``. Under --eval, redirect stdout log handlers too.
    err_console = TeeConsole(stderr=True)
    if eval_mode:
        for handler in logging.getLogger("aws_bench").handlers:
            if isinstance(handler, RichHandler):
                handler.console = Console(stderr=True)
            elif isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
                handler.stream = sys.stderr

    err_console.print("[bold]Generating Bedrock bearer token...[/bold]")
    try:
        token = generate_bearer_token(force=force, no_verify=no_verify, days=days)
    except BedrockCredentialError as e:
        err_console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(code=1) from e

    env_vars = {"AWS_BEARER_TOKEN_BEDROCK": token}
    if output:
        lines = [f"{k}={v}" for k, v in env_vars.items()]
        fd = os.open(str(output), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines) + "\n")
        err_console.print(f"[green]✓[/green] Credentials written to {output}")
        return
    err_console.print()
    # Raw print: the token is a secret (never teed) and must reach the shell
    # verbatim (Rich would soft-wrap and corrupt a long token under a pipe).
    for k, v in env_vars.items():
        print(f'export {k}="{v}"')
    if not eval_mode:
        err_console.print()
        err_console.print(
            "[dim]Load into your shell with:[/dim] eval $(aws-bench env creds --eval)"
        )
    err_console.print("[green]✓[/green] Bearer token ready.")
