"""Point an installed plugin's MCP config at another endpoint.

A plugin's bundled MCP config names a fixed endpoint, and the agent CLI's plugin install
writes it into a plugin cache under its config directory. To run against a different
endpoint (a pre-production stage, a self-hosted proxy) without republishing the plugin,
the Claude Code and Codex adapters replace the endpoint in the cached MCP configs right
after the install. Both CLIs use the same layout: ``<config dir>/plugins/cache/...`` with
the MCP config in ``.mcp.json`` or ``mcp.json``.

The rewrite fails the setup if the old endpoint is still configured afterwards, or if no
config names the new one, so a trial never silently runs against the original endpoint.
"""

from __future__ import annotations

import json
import shlex

# Plugin MCP config file names the agent CLIs read from an installed plugin.
PLUGIN_MCP_FILES = (".mcp.json", "mcp.json")


def _sed_pattern(literal: str) -> str:
    """Escape a literal for a sed basic regular expression delimited by ``|``."""
    return "".join("\\" + c if c in "\\.*[]^$|" else c for c in literal)


def _sed_replacement(literal: str) -> str:
    """Escape a literal for a sed replacement delimited by ``|``."""
    return "".join("\\" + c if c in "\\&|" else c for c in literal)


def validate_endpoint_override(
    endpoint_from: str | None,
    endpoint_to: str | None,
    endpoint_args: list[str] | None,
    has_plugins: bool,
) -> list[str]:
    """Check the endpoint-override agent kwargs and return the extra arguments.

    Args:
        endpoint_from: Endpoint URL to replace.
        endpoint_to: Replacement endpoint URL.
        endpoint_args: Extra arguments inserted after the replacement URL.
        has_plugins: Whether the agent installs any plugin.

    Returns:
        A copy of ``endpoint_args`` (empty when none were given).

    Raises:
        ValueError: If the combination is incomplete or a value cannot be embedded in
            the ``sed`` expression.
    """
    args = list(endpoint_args) if endpoint_args else []
    if bool(endpoint_from) != bool(endpoint_to):
        raise ValueError("mcp_endpoint_from and mcp_endpoint_to must be given together")
    if args and not endpoint_to:
        raise ValueError("mcp_endpoint_args needs mcp_endpoint_from and mcp_endpoint_to")
    if endpoint_to and not has_plugins:
        raise ValueError("an MCP endpoint rewrite needs plugins to rewrite")
    for value in (endpoint_from, endpoint_to, *args):
        if value is not None and ("|" in value or "\n" in value):
            raise ValueError(f"MCP endpoint values must not contain '|' or newlines: {value!r}")
    return args


def build_endpoint_rewrite_command(
    endpoint_from: str, endpoint_to: str, endpoint_args: list[str], config_dir: str
) -> str:
    """Rewrite the endpoint in every installed plugin MCP config, then verify.

    The URL is matched as a whole JSON string, so the replacement (the new URL followed by
    ``endpoint_args``) keeps the config valid JSON. The check reads the plugin cache: no
    config may still name the old endpoint, and at least one must name the new one.

    Args:
        endpoint_from: Endpoint URL to replace.
        endpoint_to: Replacement endpoint URL.
        endpoint_args: Extra arguments inserted after the replacement URL.
        config_dir: The agent CLI's config directory, as a shell fragment that is valid
            inside double quotes (for example ``${CLAUDE_CONFIG_DIR:-$HOME/.claude}``).

    Returns:
        A shell command that exits non-zero when the rewrite did not take effect.
    """
    old = json.dumps(endpoint_from)
    new = ", ".join(json.dumps(v) for v in (endpoint_to, *endpoint_args))
    expr = f"s|{_sed_pattern(old)}|{_sed_replacement(new)}|g"
    stale_msg = shlex.quote(f"MCP endpoint rewrite failed: {endpoint_from} still configured")
    missing_msg = shlex.quote(f"MCP endpoint rewrite failed: no plugin config names {endpoint_to}")
    includes = " ".join(f"--include={shlex.quote(f)}" for f in PLUGIN_MCP_FILES)
    src, dst = shlex.quote(endpoint_from), shlex.quote(endpoint_to)
    return (
        f'( d="{config_dir}/plugins"; '
        f'for f in $(grep -rlF {src} "$d" {includes}); do '
        f'sed -i {shlex.quote(expr)} "$f" || exit 1; done; '
        f'if grep -rqF {src} "$d/cache" {includes}; then '
        f"echo {stale_msg} >&2; exit 1; fi; "
        f'grep -rqF {dst} "$d/cache" {includes} || '
        f"{{ echo {missing_msg} >&2; exit 1; }}; "
        f"echo {shlex.quote('MCP_ENDPOINT_OK ' + endpoint_to)} )"
    )
