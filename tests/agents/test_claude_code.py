"""Tests for the plugin-capable Claude Code agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from harbor.agents.factory import AgentFactory
from harbor.models.agent.name import AgentName

from aws_bench.agents.claude_code import ClaudeCode

_MARKETPLACE = "anthropics/claude-plugins-official"
_PLUGIN = "aws-core@claude-plugins-official"


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs"


@pytest.fixture
def plugin_agent(logs_dir: Path) -> ClaudeCode:
    return ClaudeCode(logs_dir=logs_dir, marketplaces=[_MARKETPLACE], plugins=[_PLUGIN])


def _fresh_environment() -> MagicMock:
    environment = MagicMock()
    environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
    return environment


def _commands(environment: MagicMock) -> list[str]:
    return [c.kwargs.get("command", "") for c in environment.exec.call_args_list]


# ── name ──


def test_name_is_claude_code():
    """Keeps the built-in name so it overrides Harbor's claude-code."""
    assert ClaudeCode.name() == "claude-code"


# ── factory override ──


def test_factory_resolves_claude_code_to_subclass():
    """Importing aws_bench.agents maps `claude-code` to this subclass."""
    import aws_bench.agents  # noqa: F401

    assert AgentFactory.get_agent_class(AgentName("claude-code")) is ClaudeCode


def test_subclass_replaces_builtin_in_agents_list():
    """The subclass replaces Harbor's claude-code entry in the factory map."""
    import aws_bench.agents  # noqa: F401

    assert AgentFactory._AGENT_MAP[AgentName("claude-code")] == (
        f"{ClaudeCode.__module__}:{ClaudeCode.__qualname__}"
    )


# ── init ──


def test_defaults_to_no_plugins(logs_dir: Path):
    """Defaults to no marketplaces or plugins."""
    agent = ClaudeCode(logs_dir=logs_dir)
    assert agent._marketplaces == []
    assert agent._plugins == []


def test_stores_marketplaces_and_plugins(plugin_agent: ClaudeCode):
    """Stores the supplied marketplaces and plugins."""
    assert plugin_agent._marketplaces == [_MARKETPLACE]
    assert plugin_agent._plugins == [_PLUGIN]


def test_none_normalizes_to_empty_list(logs_dir: Path):
    """Normalizes explicit None to empty lists."""
    agent = ClaudeCode(logs_dir=logs_dir, marketplaces=None, plugins=None)
    assert agent._marketplaces == []
    assert agent._plugins == []


def test_marketplaces_without_plugins_raises(logs_dir: Path):
    """A marketplace with no plugin to install is a no-op, so reject it loudly."""
    with pytest.raises(ValueError, match="marketplaces were given without plugins"):
        ClaudeCode(logs_dir=logs_dir, marketplaces=[_MARKETPLACE], plugins=None)


# ── MCP / plugin setup command ──


def test_no_plugins_passes_through_to_base(logs_dir: Path):
    """Returns the base command (None here) when no plugins are requested."""
    agent = ClaudeCode(logs_dir=logs_dir)
    assert agent._build_register_mcp_servers_command() is None


def test_installs_each_plugin(plugin_agent: ClaudeCode):
    """Installs each requested plugin user-scoped."""
    cmd = plugin_agent._build_register_mcp_servers_command()
    assert cmd is not None
    assert f"claude plugin install {_PLUGIN} --scope user" in cmd


def test_adds_each_marketplace(plugin_agent: ClaudeCode):
    """Adds each requested marketplace."""
    cmd = plugin_agent._build_register_mcp_servers_command()
    assert cmd is not None
    assert f"claude plugin marketplace add {_MARKETPLACE}" in cmd


def test_restores_path(plugin_agent: ClaudeCode):
    """Restores PATH so claude resolves under the minimal exec PATH."""
    cmd = plugin_agent._build_register_mcp_servers_command()
    assert cmd is not None
    assert "export PATH=" in cmd


def test_adds_marketplace_before_installing_plugin(plugin_agent: ClaudeCode):
    """Adds the marketplace before installing the plugin that needs it."""
    cmd = plugin_agent._build_register_mcp_servers_command()
    assert cmd is not None
    assert cmd.index("marketplace add") < cmd.index("plugin install")


def test_supports_multiple_plugins(logs_dir: Path):
    """Adds every marketplace and installs every plugin."""
    agent = ClaudeCode(
        logs_dir=logs_dir,
        marketplaces=["a/one", "b/two"],
        plugins=["p1@one", "p2@two"],
    )
    cmd = agent._build_register_mcp_servers_command()
    assert cmd is not None
    assert "claude plugin install p1@one --scope user" in cmd
    assert "claude plugin install p2@two --scope user" in cmd
    assert "marketplace add a/one" in cmd
    assert "marketplace add b/two" in cmd


def test_chains_after_base_mcp_servers(logs_dir: Path):
    """Writes task MCP servers first, then chains plugin install with &&."""
    stdio = MagicMock(transport="stdio", command="run", args=[])
    stdio.name = "srv"
    agent = ClaudeCode(logs_dir=logs_dir, plugins=[_PLUGIN], mcp_servers=[stdio])
    cmd = agent._build_register_mcp_servers_command()
    assert cmd is not None
    assert ".claude.json" in cmd
    assert cmd.index(".claude.json") < cmd.index("plugin install")
    assert " && " in cmd


# ── install ──


@pytest.mark.asyncio
async def test_no_plugins_skips_git_install(logs_dir: Path):
    """Skips git install when no plugins are requested."""
    agent = ClaudeCode(logs_dir=logs_dir)
    environment = _fresh_environment()

    await agent.install(environment)

    assert not any("command -v git" in c for c in _commands(environment))


@pytest.mark.asyncio
async def test_installs_git_when_plugins_requested(plugin_agent: ClaudeCode):
    """Installs git during setup when plugins are requested."""
    environment = _fresh_environment()

    await plugin_agent.install(environment)

    assert any("command -v git" in c for c in _commands(environment))


@pytest.mark.asyncio
async def test_rewrites_github_ssh_to_https(plugin_agent: ClaudeCode):
    """Rewrites GitHub SSH remotes to anonymous HTTPS for the clone."""
    environment = _fresh_environment()

    await plugin_agent.install(environment)

    assert any("insteadOf" in c and "git@github.com:" in c for c in _commands(environment))
