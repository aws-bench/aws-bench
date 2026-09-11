"""Pre-execution checks that fail fast with clear messages.

Run before any non-trivial work (Docker build, AWS API calls). Each
check is a single round-trip (<1s) so operator sees configuration
errors immediately instead of after a 30s docker build.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Final

from aws_bench.exceptions import AWSBenchError
from aws_bench.logging.logger import get_logger
from aws_bench.utils.credentials_provider import CredentialProvider

logger = get_logger(__name__)

# Minimum buildx the modern ``docker compose build`` path needs. Documented as a
# hard requirement in README.md ("buildx >= 0.17.0") and docs/getting-started.md.
_BUILDX_MIN_VERSION: Final = (0, 17, 0)
_BUILDX_MIN_VERSION_STR: Final = "0.17.0"

# Go-template that makes ``docker info`` emit its already-collected CLI plugin
# inventory as ``name version`` lines, so the daemon check can hand the plugins
# to preflight_docker_plugins without a second ``docker info`` round-trip.
_PLUGINS_TEMPLATE: Final = "{{range .ClientInfo.Plugins}}{{.Name}} {{.Version}}\n{{end}}"

# Both plugins install sudo-free at the user level; see the "Docker Compose v2 +
# buildx" section in README.md and the appendix in docs/getting-started.md.
_PLUGIN_REMEDY: Final = (
    "See the 'Docker Compose v2 + buildx' section of the README "
    "(or the appendix in docs/getting-started.md) for sudo-free, user-level "
    "install commands."
)


class PreflightError(AWSBenchError):
    """A preflight check failed; the CLI should exit immediately."""


def preflight_docker_cli() -> None:
    """Verify the docker binary is on $PATH."""
    if shutil.which("docker") is None:
        raise PreflightError(
            "docker binary not found on PATH. Install Docker Desktop "
            "(macOS) or docker-engine (Linux) and ensure 'docker' is in "
            "your shell's PATH."
        )


def preflight_docker_daemon() -> dict[str, str]:
    """Verify the docker daemon is reachable and return its CLI plugin inventory.

    ``docker info`` exits 0 only when the daemon responds. On macOS
    with Docker Desktop stopped this exits 1 with 'Cannot connect to
    the Docker daemon'.

    The ``--format`` template additionally emits the CLI plugin inventory that
    ``docker info`` already collects, so :func:`preflight_docker_plugins` can
    validate Compose and buildx from this same call without a second round-trip
    (preserving this module's <1s-per-check property). Returns a mapping of
    plugin name to reported version string (e.g. ``{"buildx": "v0.36.1"}``).
    """
    proc = subprocess.run(
        ["docker", "info", "--format", _PLUGINS_TEMPLATE],
        capture_output=True,
        timeout=10,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        raise PreflightError(
            f"docker daemon not reachable. {stderr}\n"
            "Ensure Docker Desktop is running (macOS) or the docker "
            "service is started (Linux)."
        )
    return _parse_plugins(proc.stdout.decode(errors="replace"))


def preflight_docker_plugins(plugins: dict[str, str]) -> None:
    """Verify the Compose v2 and buildx CLI plugins are present and recent enough.

    ``plugins`` is the inventory returned by :func:`preflight_docker_daemon`,
    reusing its single ``docker info`` call. aws-bench builds agent environments
    with ``docker compose`` and images with buildx, so a missing Compose plugin
    or a buildx below ``0.17.0`` breaks every trial's ``compose build`` — the
    exact failures the README/docs troubleshooting table calls out.

    A present-but-unparseable buildx version is a pass-with-warning rather than a
    hard fail: the plugin exists (so the observed missing-plugin and known-old
    failures are still caught), and refusing to run on a version string we merely
    cannot parse would risk false negatives on legitimate dev/patched builds.
    """
    if "compose" not in plugins:
        raise PreflightError(
            "Docker Compose v2 plugin not found. aws-bench builds agent "
            "environments with 'docker compose'; without it plain 'docker' "
            f"parses the subcommand's flags and fails.\n{_PLUGIN_REMEDY}"
        )

    if "buildx" not in plugins:
        raise PreflightError(
            "Docker buildx plugin not found. aws-bench builds images with "
            f"buildx (>= {_BUILDX_MIN_VERSION_STR} required).\n{_PLUGIN_REMEDY}"
        )

    raw_version = plugins["buildx"]
    version = _parse_version(raw_version)
    if version is None:
        logger.warning(
            "Could not parse buildx version %r; skipping the >= %s check. "
            "If image builds fail, verify 'docker buildx version'.",
            raw_version,
            _BUILDX_MIN_VERSION_STR,
        )
        return

    if not _at_least(version, _BUILDX_MIN_VERSION):
        raise PreflightError(
            f"Docker buildx {raw_version} is below the required "
            f">= {_BUILDX_MIN_VERSION_STR}. Modern 'docker compose build' needs "
            f"buildx {_BUILDX_MIN_VERSION_STR} or later.\n{_PLUGIN_REMEDY}"
        )


def _parse_plugins(info_output: str) -> dict[str, str]:
    """Parse ``name version`` lines from the docker info plugin template.

    A plugin that reports no version yields an empty-string value rather than
    being dropped, so callers can distinguish "buildx present, version unknown"
    from "buildx absent".
    """
    plugins: dict[str, str] = {}
    for line in info_output.splitlines():
        parts = line.split()
        if not parts:
            continue
        plugins[parts[0]] = parts[1] if len(parts) > 1 else ""
    return plugins


def _parse_version(raw: str) -> tuple[int, ...] | None:
    """Parse a ``v``-prefixed dotted version into an int tuple for numeric compare.

    Strips a leading ``v`` and any pre-release/build suffix (``-``/``+``), then
    converts the dotted core to ints. Numeric comparison is required because
    versions arrive ``v``-prefixed and lexical order is wrong: ``"v0.9.0"`` sorts
    after ``"v0.17.0"`` as a string. Returns ``None`` when no numeric version can
    be recovered so the caller can warn instead of crashing preflight.
    """
    core = raw.strip().lstrip("vV").split("-", 1)[0].split("+", 1)[0]
    if not core:
        return None
    try:
        return tuple(int(part) for part in core.split("."))
    except ValueError:
        return None


def _at_least(version: tuple[int, ...], floor: tuple[int, ...]) -> bool:
    """Return whether ``version`` is >= ``floor``, zero-padding to equal length."""
    width = max(len(version), len(floor))
    padded_version = version + (0,) * (width - len(version))
    padded_floor = floor + (0,) * (width - len(floor))
    return padded_version >= padded_floor


def preflight_aws_credentials(
    cred_provider: CredentialProvider, *, ou_name: str | None = None
) -> str:
    """Verify AWS credentials and log the caller identity the command will use.

    Logs the account and caller ARN (and OU when given) so the operator sees,
    before any work, which account the command is acting on — the common cause
    of running ``init`` and ``setup`` against different accounts unawares.
    Returns the caller account id.
    """
    try:
        identity = cred_provider.session.client("sts").get_caller_identity()
    except Exception as exc:  # noqa: BLE001 — boto3 surfaces a wide tree
        raise PreflightError(
            f"AWS credentials are missing or expired: {exc}\nRefresh credentials and re-run."
        ) from exc

    account_id = identity["Account"]
    suffix = f", OU '{ou_name}'" if ou_name else ""
    logger.info("Using AWS account %s (%s)%s", account_id, identity["Arn"], suffix)
    return account_id
