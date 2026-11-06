"""Point an installed plugin's MCP config at another endpoint.

A plugin's bundled MCP config names a fixed endpoint, and the agent CLI's plugin install
writes it into ``<config dir>/plugins/cache`` (``.mcp.json`` or ``mcp.json``). To run against
a different endpoint without republishing the plugin, the Claude Code and Codex adapters
replace the endpoint in those cached configs right after the install.
"""

from __future__ import annotations

import json
import shlex

# jq program: values equal to $old become $new. In an array (a command's args) the extra
# arguments follow the new value; an object value (a `url` field) has no room for them.
_JQ_REWRITE = """walk(
  if type == "array" then map(if . == $old then ($new, $extra[]) else . end)
  elif type == "object" then map_values(
    if . != $old then .
    elif ($extra | length) > 0 then error("mcp_endpoint_args need an args list, not a field")
    else $new end)
  else . end)"""


def validate_endpoint_override(
    endpoint_from: str | None,
    endpoint_to: str | None,
    endpoint_args: list[str] | None,
    has_plugins: bool,
) -> list[str]:
    """Check the endpoint-override agent kwargs and return the extra arguments.

    Raises:
        ValueError: If the combination is incomplete.
    """
    args = list(endpoint_args) if endpoint_args else []
    if bool(endpoint_from) != bool(endpoint_to):
        raise ValueError("mcp_endpoint_from and mcp_endpoint_to must be given together")
    if args and not endpoint_to:
        raise ValueError("mcp_endpoint_args needs mcp_endpoint_from and mcp_endpoint_to")
    if endpoint_to and not has_plugins:
        raise ValueError("an MCP endpoint rewrite needs plugins to rewrite")
    return args


def build_endpoint_rewrite_command(
    endpoint_from: str, endpoint_to: str, endpoint_args: list[str], config_dir: str
) -> str:
    """Build the shell command that rewrites the cached plugin configs.

    Args:
        endpoint_from: Endpoint URL to replace.
        endpoint_to: Replacement endpoint URL.
        endpoint_args: Extra arguments inserted after the replacement URL in an args list.
        config_dir: The agent CLI's config directory, as a shell fragment that is valid
            inside double quotes (for example ``${CLAUDE_CONFIG_DIR:-$HOME/.claude}``).

    Returns:
        A shell command that exits non-zero when no config was rewritten.
    """
    old, new = shlex.quote(endpoint_from), shlex.quote(endpoint_to)
    extra = shlex.quote(json.dumps(endpoint_args))
    return (
        f'( d="{config_dir}/plugins/cache"; found=0; '
        'command -v jq >/dev/null || { echo "jq not found" >&2; exit 1; }; '
        'for f in $(find "$d" \\( -name .mcp.json -o -name mcp.json \\) 2>/dev/null); do '
        f"if jq -e --arg old {old} 'any(..; . == $old)' \"$f\" >/dev/null 2>&1; then "
        f"jq --arg old {old} --arg new {new} --argjson extra {extra} {shlex.quote(_JQ_REWRITE)} "
        '"$f" > "$f.tmp" && mv "$f.tmp" "$f" || exit 1; found=1; fi; done; '
        '[ "$found" = 1 ] || '
        f'{{ echo "MCP endpoint rewrite failed: no plugin config names "{old} >&2; exit 1; }}; '
        f"echo MCP_ENDPOINT_OK {new} )"
    )
