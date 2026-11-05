"""Plugin-capable Claude Code agent for aws-bench.

Extends Harbor's ``claude-code`` agent so a headless ``claude --print`` run can
install Claude Code plugins into its per-trial config directory before the task
runs. A plugin bundles MCP servers and skills together, so installing one makes
both available to the agent.

Marketplaces and plugins are supplied as constructor kwargs, so they flow from
``--ak`` on the run CLI::

    -a claude-code \
      --ak marketplaces='["anthropics/claude-plugins-official"]' \
      --ak plugins='["aws-core@claude-plugins-official"]'

With no ``plugins``, no plugin install runs.

The installed plugins' MCP configs can be pointed at another endpoint, for
example a pre-production stage, without republishing the plugin::

    --ak mcp_endpoint_from=https://prod.example/mcp \
    --ak mcp_endpoint_to=https://preprod.example/mcp \
    --ak mcp_endpoint_args='["--region", "us-east-1"]'

Setup fails if any installed plugin config still names ``mcp_endpoint_from``
afterwards, so a trial never silently runs against the original endpoint.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from harbor.agents.installed.claude_code import ClaudeCode as _HarborClaudeCode
from harbor.environments.base import BaseEnvironment

from aws_bench.agents.plugin_endpoint import (
    build_endpoint_rewrite_command,
    validate_endpoint_override,
)

# claude installs to ~/.local/bin, which is off the PATH for the non-interactive
# exec shell; prepend it so `claude plugin install` resolves.
_PATH_RESTORE = 'export PATH="$HOME/.local/bin:$PATH"'


class ClaudeCode(_HarborClaudeCode):
    """Claude Code that installs plugins from marketplaces per trial."""

    def __init__(
        self,
        logs_dir: Path,
        marketplaces: list[str] | None = None,
        plugins: list[str] | None = None,
        mcp_endpoint_from: str | None = None,
        mcp_endpoint_to: str | None = None,
        mcp_endpoint_args: list[str] | None = None,
        *args,
        **kwargs,
    ):
        """Store the marketplaces to add and plugins to install.

        Args:
            logs_dir: Trial logs directory, forwarded to the base agent.
            marketplaces: Marketplace git sources to add, each an ``owner/repo``
                slug or a clone URL.
            plugins: Plugin specs to install, each ``<plugin>@<marketplace>``.
            mcp_endpoint_from: MCP endpoint URL to replace in the installed
                plugins' MCP configs.
            mcp_endpoint_to: Replacement endpoint URL.
            mcp_endpoint_args: Extra arguments inserted after the replacement
                URL, for a proxy that cannot infer them from the new hostname.
            *args: Forwarded to the base agent.
            **kwargs: Forwarded to the base agent.
        """
        self._marketplaces = list(marketplaces) if marketplaces else []
        self._plugins = list(plugins) if plugins else []
        if self._marketplaces and not self._plugins:
            raise ValueError(
                "marketplaces were given without plugins; a marketplace is only "
                "added when a plugin from it is installed"
            )
        self._mcp_endpoint_from = mcp_endpoint_from
        self._mcp_endpoint_to = mcp_endpoint_to
        self._mcp_endpoint_args = validate_endpoint_override(
            mcp_endpoint_from, mcp_endpoint_to, mcp_endpoint_args, bool(self._plugins)
        )
        super().__init__(logs_dir, *args, **kwargs)

    async def install(self, environment: BaseEnvironment) -> None:
        """Install Claude Code, plus git when plugins are requested.

        Claude Code shells out to ``git`` to clone a plugin marketplace, but the
        agent base image does not ship it. Install it during agent setup using
        whichever package manager the image provides. The container has no GitHub
        SSH key, so GitHub SSH remotes are rewritten to anonymous HTTPS for the
        public marketplace clone.
        """
        await super().install(environment)

        if not self._plugins:
            return

        await self.exec_as_root(
            environment,
            command=(
                "if ! command -v git >/dev/null 2>&1; then "
                "if command -v apt-get >/dev/null 2>&1; then "
                "apt-get update >/dev/null 2>&1 && apt-get install -y git >/dev/null 2>&1; "
                "elif command -v apk >/dev/null 2>&1; then "
                "apk add --no-cache git >/dev/null 2>&1; "
                "elif command -v yum >/dev/null 2>&1; then "
                "yum install -y git >/dev/null 2>&1; "
                "fi; fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_agent(
            environment,
            command=('git config --global url."https://github.com/".insteadOf "git@github.com:"'),
        )

    def _build_register_mcp_servers_command(self) -> str | None:
        """Chain plugin installation onto the per-trial config setup command.

        Harbor sets ``CLAUDE_CONFIG_DIR`` for this command, so the marketplace,
        plugin cache, and ``enabledPlugins`` entry land in the per-trial config
        directory. A failed ``plugin install`` aborts setup; the task does not
        run without its plugin.
        """
        base_cmd = super()._build_register_mcp_servers_command()

        if not self._plugins:
            return base_cmd

        parts = [_PATH_RESTORE]
        for source in self._marketplaces:
            parts.append(f"claude plugin marketplace add {shlex.quote(source)} 2>/dev/null || true")
        for spec in self._plugins:
            parts.append(f"claude plugin install {shlex.quote(spec)} --scope user")
        if self._mcp_endpoint_to:
            # Every install must succeed before the rewrite, or a missing plugin config would go
            # unnoticed; each marketplace add already ends in `|| true`.
            parts.append(self._build_endpoint_rewrite_command())
            plugin_cmd = " && ".join(parts)
        else:
            plugin_cmd = "; ".join(parts)

        if base_cmd:
            return f"{base_cmd} && {plugin_cmd}"
        return plugin_cmd

    def _build_endpoint_rewrite_command(self) -> str:
        """Rewrite the endpoint in the plugins ``claude plugin install`` just cached."""
        assert self._mcp_endpoint_from and self._mcp_endpoint_to
        return build_endpoint_rewrite_command(
            self._mcp_endpoint_from,
            self._mcp_endpoint_to,
            self._mcp_endpoint_args,
            "${CLAUDE_CONFIG_DIR:-$HOME/.claude}",
        )
