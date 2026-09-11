"""Shared CLI option definitions for env commands.

These Type-annotated option definitions can be reused across verify, reset, and cleanup
commands to avoid duplication. Import and use them in command function signatures.
"""

from pathlib import Path
from typing import Annotated

from typer import Option

from aws_bench.scenario.job_config import ScenarioJobConfig

# Scenario Source Options (shared across verify, reset, cleanup)

ScenarioPathOption = Annotated[
    Path | None,
    Option(
        "--scenario-path",
        help=(
            "Path to a directory containing scenario directories (each "
            "immediate subdirectory is treated as one scenario). Mutually "
            "exclusive with --dataset."
        ),
        rich_help_panel="Scenario Source",
        show_default=False,
    ),
]

DatasetOption = Annotated[
    str | None,
    Option(
        "-d",
        "--dataset",
        help=(
            "Registry dataset name@version (e.g. 'aws-bench-quickstart@1.0.0'). "
            "Mutually exclusive with --scenario-path."
        ),
        rich_help_panel="Scenario Source",
        show_default=False,
    ),
]

RegistryUrlOption = Annotated[
    str | None,
    Option(
        "--registry-url",
        help="Override the default registry URL.",
        rich_help_panel="Scenario Source",
        show_default=False,
    ),
]

RegistryPathOption = Annotated[
    Path | None,
    Option(
        "--registry-path",
        help="Path to a local registry (offline / dev).",
        rich_help_panel="Scenario Source",
        show_default=False,
    ),
]

IncludeScenariosOption = Annotated[
    list[str] | None,
    Option(
        "--include-scenarios",
        help="Scenario name pattern(s) to include (fnmatch glob; can be repeated).",
        rich_help_panel="Scenario Source",
        show_default=False,
    ),
]

ExcludeScenariosOption = Annotated[
    list[str] | None,
    Option(
        "--exclude-scenarios",
        help=(
            "Scenario name pattern(s) to exclude (fnmatch glob; can be repeated). "
            "Applied after --include-scenarios."
        ),
        rich_help_panel="Scenario Source",
        show_default=False,
    ),
]

# Job Settings Options (shared across verify, reset, cleanup)

JobNameOption = Annotated[
    str | None,
    Option(
        "--job-name",
        help="Name of the job (default: timestamp).",
        rich_help_panel="Job Settings",
        show_default=False,
    ),
]

JobsDirOption = Annotated[
    Path | None,
    Option(
        "-o",
        "--jobs-dir",
        help=(
            "Directory to store job results "
            f"(default: {ScenarioJobConfig.model_fields['jobs_dir'].default})."
        ),
        rich_help_panel="Job Settings",
        show_default=False,
    ),
]

NConcurrentOption = Annotated[
    int | None,
    Option(
        "-n",
        "--n-concurrent",
        help=(
            "Number of concurrent scenario trials "
            f"(default: {ScenarioJobConfig.model_fields['n_concurrent'].default})."
        ),
        rich_help_panel="Job Settings",
        show_default=False,
    ),
]

TimeoutMultiplierOption = Annotated[
    float | None,
    Option(
        "--timeout-multiplier",
        help=(
            "Multiplier applied to all phase timeouts "
            f"(default: {ScenarioJobConfig.model_fields['timeout_multiplier'].default})."
        ),
        rich_help_panel="Job Settings",
        show_default=False,
    ),
]

QuietOption = Annotated[
    bool,
    Option(
        "-q",
        "--quiet",
        "--silent",
        help="Suppress the rich live progress display; falls back to log output.",
        rich_help_panel="Job Settings",
        show_default=False,
    ),
]

MaxRetriesOption = Annotated[
    int | None,
    Option(
        "-r",
        "--max-retries",
        help="Maximum number of retry attempts per trial (default: 0).",
        rich_help_panel="Job Settings",
        show_default=False,
    ),
]

RetryIncludeOption = Annotated[
    list[str] | None,
    Option(
        "--retry-include",
        help="Exception type names to retry on (can be repeated).",
        rich_help_panel="Job Settings",
        show_default=False,
    ),
]

RetryExcludeOption = Annotated[
    list[str] | None,
    Option(
        "--retry-exclude",
        help="Exception type names to NEVER retry on (can be repeated).",
        rich_help_panel="Job Settings",
        show_default=False,
    ),
]

DebugOption = Annotated[
    bool,
    Option(
        "--debug",
        help="Enable debug-level logging.",
        rich_help_panel="Job Settings",
        show_default=False,
    ),
]

# Environment Options (shared across verify, reset, cleanup)

ForceBuildOption = Annotated[
    bool | None,
    Option(
        "--force-build/--no-force-build",
        help="Force a fresh Docker image build (no cache).",
        rich_help_panel="Environment",
        show_default=False,
    ),
]

DeleteOption = Annotated[
    bool | None,
    Option(
        "--delete/--no-delete",
        help="Delete the scenario container after the phase completes.",
        rich_help_panel="Environment",
        show_default=False,
    ),
]

OverrideCpusOption = Annotated[
    int | None,
    Option(
        "--override-cpus",
        help="Override the scenario's authored CPU count.",
        rich_help_panel="Environment",
        show_default=False,
    ),
]

OverrideMemoryMbOption = Annotated[
    int | None,
    Option(
        "--override-memory-mb",
        help="Override the scenario's authored memory limit (MB).",
        rich_help_panel="Environment",
        show_default=False,
    ),
]

OverrideBuildTimeoutSecOption = Annotated[
    float | None,
    Option(
        "--override-build-timeout-sec",
        help="Override the scenario's authored Docker build timeout (sec).",
        rich_help_panel="Environment",
        show_default=False,
    ),
]
