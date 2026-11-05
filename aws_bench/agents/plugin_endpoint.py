"""Point an installed plugin's MCP config at another endpoint.

A plugin's bundled MCP config names a fixed endpoint, and the agent CLI's plugin install
writes it into ``<config dir>/plugins/cache`` (``.mcp.json`` or ``mcp.json``). To run against
a different endpoint without republishing the plugin, the Claude Code and Codex adapters
replace the endpoint in those cached configs right after the install.
"""

from __future__ import annotations

import json
import shlex

# Runs in the trial container, where aws_bench is not installed, so it uses only the standard
# library. Arguments: cache dir, old endpoint, new endpoint, JSON list of extra arguments.
# The configs are edited as parsed JSON and only values equal to the old endpoint change.
_REWRITE_SCRIPT = """
import json, os, sys

cache, old, new, extra = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
replaced = 0


def rewrite(node):
    global replaced
    if isinstance(node, list):
        out = []
        for item in node:
            if item == old:
                out += [new, *extra]
                replaced += 1
            else:
                out.append(rewrite(item))
        return out
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if value == old and extra:
                sys.exit(f"mcp_endpoint_args need an args list, but '{key}' holds the endpoint")
            if value == old:
                out[key] = new
                replaced += 1
            else:
                out[key] = rewrite(value)
        return out
    return node


for root, _, names in os.walk(cache):
    for name in names:
        if name in (".mcp.json", "mcp.json"):
            path = os.path.join(root, name)
            with open(path) as handle:
                config = rewrite(json.load(handle))
            with open(path, "w") as handle:
                json.dump(config, handle, indent=2)
if not replaced:
    sys.exit(f"no plugin config under {cache} names {old}")
print(f"MCP_ENDPOINT_OK {new}")
"""


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
    return (
        "( py=$(command -v python3 || command -v python) "
        '|| { echo "python not found" >&2; exit 1; }; '
        f'"$py" -c {shlex.quote(_REWRITE_SCRIPT)} "{config_dir}/plugins/cache" '
        f"{shlex.quote(endpoint_from)} {shlex.quote(endpoint_to)} "
        f"{shlex.quote(json.dumps(endpoint_args))} )"
    )
