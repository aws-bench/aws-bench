"""Global constants for aws-bench."""

from pathlib import Path

DEFAULT_REGION = "us-east-1"
OUTPUT_DIR = Path.home() / ".aws-bench"

# Per-invocation command ledger (one entry dir per aws-bench CLI call).
LOGS_DIR = OUTPUT_DIR / "logs"

# Host-local benchmark state: baseline snapshots and contamination flags.
STATE_DIR = OUTPUT_DIR / "state"

# Cache for fetched scenarios. Layout: ``<SCENARIO_CACHE_DIR>/<uuid>/<name>/``,
# where ``<uuid>`` is content-addressed by ``(git_url, git_commit_id, path)``.
CACHE_DIR = OUTPUT_DIR / "cache"
SCENARIO_CACHE_DIR = CACHE_DIR / "scenarios"

# Cache for fetched tasks. Same layout as scenarios. Plumbed into the
# inherited ``DatasetConfig.download_dir`` so all aws-bench-managed cache lives
# under ``~/.aws-bench/cache/`` rather than the library default
# ``~/.cache/harbor/``.
TASK_CACHE_DIR = CACHE_DIR / "tasks"

# Cache for fetched registry-referenced metric scripts and extra-instruction
# files. Same commit-addressed layout as scenarios/tasks; one subdir per type.
METRIC_CACHE_DIR = CACHE_DIR / "metrics"
INSTRUCTION_CACHE_DIR = CACHE_DIR / "extra_instructions"

# Regenerated CCAPI skip-types list — kept here, not in the package, whose
# PyPI-install dir is read-only and wiped on upgrade.
CCAPI_CACHE_DIR = CACHE_DIR / "ccapi"

# Redirect the library cache location so all artifacts live under
# ~/.aws-bench/cache/. ``GitTaskId.get_local_path()`` reads
# ``harbor.constants.TASK_CACHE_DIR``; patch both the constants module and the
# importing module so the override takes effect regardless of import order.
import harbor.constants  # noqa: E402
import harbor.models.task.id  # noqa: E402

harbor.constants.TASK_CACHE_DIR = TASK_CACHE_DIR
harbor.models.task.id.TASK_CACHE_DIR = TASK_CACHE_DIR

# Default ``registry.json`` URL: the registry is hosted in source and fetched
# at runtime. Override at the CLI with ``--registry-url`` or
# ``--registry-path``.
DEFAULT_REGISTRY_URL = "https://raw.githubusercontent.com/aws-bench/aws-bench/main/registry.json"

REGISTRY_CACHE_DIR = CACHE_DIR / "registry"
