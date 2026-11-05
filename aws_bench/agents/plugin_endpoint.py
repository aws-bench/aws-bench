"""Point an installed plugin's MCP config at another endpoint.

A plugin's bundled MCP config names a fixed endpoint, and the agent CLI's plugin install
writes it into a plugin cache under its config directory. To run against a different
endpoint (a pre-production stage, a self-hosted proxy) without republishing the plugin,
the Claude Code and Codex adapters replace the endpoint in the cached MCP configs right
after the install. Both CLIs use the same layout: ``<config dir>/plugins/cache/...`` with
the MCP config in ``.mcp.json`` or ``mcp.json``.

The configs are edited as parsed JSON, so the result is always valid JSON. The rewrite fails
the setup if a config still names the old endpoint afterwards, or if no config names the new
one, so a trial never silently runs against the original endpoint.
"""

from __future__ import annotations

import inspect
import json
import shlex


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


# The two functions below run inside the trial container, where aws_bench is not installed,
# so they use only the standard library and `build_endpoint_rewrite_command` ships their
# source with `python -c`. Tests call them directly.


def rewrite_plugin_endpoints(
    cache_dir: str, endpoint_from: str, endpoint_to: str, endpoint_args: list[str]
) -> list[str]:
    """Rewrite ``endpoint_from`` to ``endpoint_to`` in every cached plugin MCP config.

    Only values equal to ``endpoint_from`` are replaced. When the value is an element of a
    list (a command's ``args``), ``endpoint_args`` are inserted right after the new URL;
    when it is an object value (a ``url`` field) there is nowhere to put them, so that is
    an error. Nothing is written unless every config passes the checks.

    Args:
        cache_dir: The agent CLI's ``plugins/cache`` directory.
        endpoint_from: Endpoint URL to replace.
        endpoint_to: Replacement endpoint URL.
        endpoint_args: Extra arguments inserted after the replacement URL in a list.

    Returns:
        Error messages; empty when the rewrite took effect.
    """
    import json
    import os

    errors: list[str] = []
    pending: list[tuple[str, object]] = []
    names_new = False

    def strings(node: object):
        if isinstance(node, str):
            yield node
        elif isinstance(node, list):
            for item in node:
                yield from strings(item)
        elif isinstance(node, dict):
            for item in node.values():
                yield from strings(item)

    def rewrite(node: object, where: str) -> object:
        if isinstance(node, list):
            out: list[object] = []
            for item in node:
                if item == endpoint_from:
                    out.append(endpoint_to)
                    out.extend(endpoint_args)
                else:
                    out.append(rewrite(item, where))
            return out
        if isinstance(node, dict):
            result: dict[str, object] = {}
            for key, value in node.items():
                if value == endpoint_from:
                    if endpoint_args:
                        errors.append(
                            f"{where}: '{key}' holds the endpoint, so mcp_endpoint_args "
                            "(command arguments) cannot be inserted"
                        )
                    result[key] = endpoint_to
                else:
                    result[key] = rewrite(value, where)
            return result
        return node

    for root, _dirs, files in os.walk(cache_dir):
        for name in sorted(files):
            if name not in (".mcp.json", "mcp.json"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, encoding="utf-8") as handle:
                    original = json.load(handle)
            except (OSError, ValueError) as exc:
                errors.append(f"{path}: cannot be read as JSON ({exc})")
                continue
            updated = rewrite(original, path)
            for text in strings(updated):
                if endpoint_from in text and endpoint_to not in text:
                    errors.append(f"{path}: still names {endpoint_from} in {text!r}")
                if endpoint_to in text:
                    names_new = True
            if updated != original:
                pending.append((path, updated))

    if not names_new:
        errors.append(f"no plugin config under {cache_dir} names {endpoint_to}")
    if not errors:
        for path, updated in pending:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(updated, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
    return errors


def rewrite_main(argv: list[str]) -> int:
    """Command-line entry: ``cache_dir endpoint_from endpoint_to <JSON list of extra args>``."""
    import json
    import sys

    errors = rewrite_plugin_endpoints(argv[1], argv[2], argv[3], json.loads(argv[4]))
    for error in errors:
        print(f"MCP endpoint rewrite failed: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"MCP_ENDPOINT_OK {argv[3]}")
    return 0


def build_endpoint_rewrite_command(
    endpoint_from: str, endpoint_to: str, endpoint_args: list[str], config_dir: str
) -> str:
    """Build the shell command that rewrites the cached plugin configs and verifies it.

    Args:
        endpoint_from: Endpoint URL to replace.
        endpoint_to: Replacement endpoint URL.
        endpoint_args: Extra arguments inserted after the replacement URL.
        config_dir: The agent CLI's config directory, as a shell fragment that is valid
            inside double quotes (for example ``${CLAUDE_CONFIG_DIR:-$HOME/.claude}``).

    Returns:
        A shell command that exits non-zero when the rewrite did not take effect.
    """
    program = "\n".join(
        [
            "from __future__ import annotations",
            "import sys",
            inspect.getsource(rewrite_plugin_endpoints),
            inspect.getsource(rewrite_main),
            "sys.exit(rewrite_main(sys.argv))",
        ]
    )
    return (
        "( py=$(command -v python3 || command -v python) "
        '|| { echo "MCP endpoint rewrite failed: python not found" >&2; exit 1; }; '
        f'"$py" -c {shlex.quote(program)} "{config_dir}/plugins/cache" '
        f"{shlex.quote(endpoint_from)} {shlex.quote(endpoint_to)} "
        f"{shlex.quote(json.dumps(endpoint_args))} )"
    )
