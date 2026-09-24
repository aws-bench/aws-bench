"""Tests for the Bedrock-aware Codex agent's MCP server registration.

These pin the corrected ``config.toml`` rendering: a stdio MCP server must emit
``command`` and ``args`` as separate TOML keys (Harbor's base flattens them into
one ``command`` string, so codex never starts the server), and every value must
be escaped so it produces valid TOML.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
from harbor.models.task.config import MCPServerConfig

from aws_bench.agents.codex import Codex


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs"


@pytest.fixture
def agent(logs_dir: Path) -> Codex:
    return Codex(logs_dir=logs_dir)


def _stdio_server(name: str, command: str, args: list[str]) -> MCPServerConfig:
    server = MagicMock()
    # `name` is a reserved MagicMock kwarg, so it must be set after construction.
    server.name = name
    server.transport = "stdio"
    server.command = command
    server.args = args
    return cast(MCPServerConfig, server)


def _url_server(name: str, url: str, transport: str = "streamable-http") -> MCPServerConfig:
    server = MagicMock()
    server.name = name
    server.transport = transport
    server.url = url
    server.command = None
    server.args = []
    return cast(MCPServerConfig, server)


def _parse_written_toml(command: str) -> dict:
    """Extract and parse the TOML body the shell command appends to config.toml.

    The command is ``echo <shell-quoted-toml> >> "$CODEX_HOME/config.toml"``; the
    TOML payload is the single-quoted argument to ``echo``.
    """
    assert command.startswith("echo ")
    assert command.endswith('>> "$CODEX_HOME/config.toml"')
    start = command.index("'")
    end = command.rindex("'")
    payload = command[start + 1 : end]
    return tomllib.loads(payload)


class TestCodexMcpServersEmpty:
    def test_no_servers_returns_none(self, agent: Codex):
        assert agent._build_register_mcp_servers_command() is None


class TestCodexStdioRendering:
    def test_stdio_renders_command_and_args_separately(self, agent: Codex):
        # A non-proxy stdio server renders command/args separately (the base
        # bug fix) and is not given any aws-mcp hardening.
        agent.mcp_servers = [_stdio_server("plain", "some-server", ["--flag", "value", "sub"])]

        command = agent._build_register_mcp_servers_command()
        assert command is not None
        parsed = _parse_written_toml(command)

        assert parsed["mcp_servers"]["plain"] == {
            "command": "some-server",
            "args": ["--flag", "value", "sub"],
        }
        # The global startup grace is emitted once, top-level.
        assert parsed["mcp_optional_startup_grace_ms"] == 0

    def test_command_has_no_space_for_multi_arg_server(self, agent: Codex):
        # The precise failure being fixed: the base flattened command+args into
        # a single `command` string, so `command` contained spaces.
        agent.mcp_servers = [
            _stdio_server("aws-mcp", "uvx", ["mcp-proxy-for-aws-cli@latest", "--skip-auth"])
        ]

        command = agent._build_register_mcp_servers_command()
        assert command is not None
        parsed = _parse_written_toml(command)

        rendered_command = parsed["mcp_servers"]["aws-mcp"]["command"]
        assert " " not in rendered_command
        assert rendered_command == "uvx"

    def test_empty_args_still_renders_args_key(self, agent: Codex):
        agent.mcp_servers = [_stdio_server("solo", "run-server", [])]

        command = agent._build_register_mcp_servers_command()
        assert command is not None
        parsed = _parse_written_toml(command)

        assert parsed["mcp_servers"]["solo"] == {"command": "run-server", "args": []}


class TestCodexTomlEscaping:
    def test_values_with_quote_and_backslash_stay_parseable(self, agent: Codex):
        agent.mcp_servers = [
            _stdio_server(
                "tricky",
                "runner",
                ['arg-with-"quote"', "path\\with\\backslash", 'both "\\'],
            )
        ]

        command = agent._build_register_mcp_servers_command()
        assert command is not None
        # The assertion is that this parses at all — invalid TOML raises here.
        parsed = _parse_written_toml(command)

        assert parsed["mcp_servers"]["tricky"]["args"] == [
            'arg-with-"quote"',
            "path\\with\\backslash",
            'both "\\',
        ]

    def test_control_characters_are_escaped(self, agent: Codex):
        agent.mcp_servers = [_stdio_server("ctrl", "runner", ["line1\nline2\ttab"])]

        command = agent._build_register_mcp_servers_command()
        assert command is not None
        parsed = _parse_written_toml(command)

        assert parsed["mcp_servers"]["ctrl"]["args"] == ["line1\nline2\ttab"]


class TestCodexUrlRendering:
    def test_url_server_renders_url_and_no_args(self, agent: Codex):
        agent.mcp_servers = [_url_server("remote", "https://mcp.example.test/sse", "sse")]

        command = agent._build_register_mcp_servers_command()
        assert command is not None
        parsed = _parse_written_toml(command)

        assert parsed["mcp_servers"]["remote"] == {"url": "https://mcp.example.test/sse"}
        assert "args" not in parsed["mcp_servers"]["remote"]
        assert "command" not in parsed["mcp_servers"]["remote"]

    def test_url_with_quote_stays_parseable(self, agent: Codex):
        agent.mcp_servers = [_url_server("remote", 'https://mcp.example.test/"weird"')]

        command = agent._build_register_mcp_servers_command()
        assert command is not None
        parsed = _parse_written_toml(command)

        assert parsed["mcp_servers"]["remote"]["url"] == 'https://mcp.example.test/"weird"'


class TestCodexMcpServersMixed:
    def test_multiple_servers_each_render_correctly(self, agent: Codex):
        agent.mcp_servers = [
            _stdio_server("aws-mcp", "uvx", ["proxy@latest", "--skip-auth"]),
            _url_server("remote", "https://mcp.example.test/mcp"),
        ]

        command = agent._build_register_mcp_servers_command()
        assert command is not None
        parsed = _parse_written_toml(command)

        assert parsed["mcp_servers"]["aws-mcp"] == {
            "command": "uvx",
            "args": ["proxy@latest", "--skip-auth"],
        }
        assert parsed["mcp_servers"]["remote"] == {"url": "https://mcp.example.test/mcp"}


class TestCodexAwsMcpHardening:
    """Pinning and hardening of the aws-mcp (Agent Toolkit) proxy server.

    Codex keeps and uses the server thanks to a top-level startup grace, a
    per-server cold-start timeout, forwarded AWS env, and an exact proxy pin.
    """

    @staticmethod
    def _aws_mcp_server() -> MCPServerConfig:
        return _stdio_server(
            "aws-mcp",
            "uvx",
            [
                "mcp-proxy-for-aws-cli@latest",
                "aws-mcp.us-east-1.api.aws/mcp",
                "--skip-auth",
                "--metadata",
                "INSTALL_SOURCE=agent-toolkit-core",
            ],
        )

    def test_aws_mcp_proxy_pinned_and_hardened(self, agent: Codex):
        agent.mcp_servers = [self._aws_mcp_server()]

        command = agent._build_register_mcp_servers_command()
        assert command is not None
        parsed = _parse_written_toml(command)

        # Bug 1 (drop): global grace disables the early drop of a slow server.
        assert parsed["mcp_optional_startup_grace_ms"] == 0

        block = parsed["mcp_servers"]["aws-mcp"]
        # Reproducibility: proxy pinned to the exact version the original codex
        # ATK lanes resolved from @latest, not the drifting @latest itself.
        assert block["args"][0] == "mcp-proxy-for-aws-cli==1.7.0"
        # Only the package spec is rewritten; endpoint + flags are untouched.
        assert block["args"][1:] == [
            "aws-mcp.us-east-1.api.aws/mcp",
            "--skip-auth",
            "--metadata",
            "INSTALL_SOURCE=agent-toolkit-core",
        ]
        # Bug 1 (timeout): generous cold-start window for uvx (2-4s).
        assert block["startup_timeout_sec"] == 120
        # Bug 2 (env): forward the AWS account/profile env into the subprocess
        # so the proxy's boto3 does not fall back to the runner instance role.
        assert block["env_vars"] == [
            "AWS_PROFILE",
            "AWS_DEFAULT_PROFILE",
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
            "AWS_CONFIG_FILE",
            "AWS_SHARED_CREDENTIALS_FILE",
        ]

    def test_grace_key_precedes_server_tables(self, agent: Codex):
        # The top-level grace key MUST appear before any [mcp_servers.*] header,
        # else TOML parses it as part of the last table. tomllib parsing (in
        # _parse_written_toml) would raise if it were misplaced.
        agent.mcp_servers = [self._aws_mcp_server()]
        command = agent._build_register_mcp_servers_command()
        assert command is not None
        parsed = _parse_written_toml(command)
        assert parsed["mcp_optional_startup_grace_ms"] == 0
        payload = command[command.index("'") + 1 : command.rindex("'")]
        assert payload.index("mcp_optional_startup_grace_ms") < payload.index("[mcp_servers.")

    def test_already_pinned_proxy_is_repinned(self, agent: Codex):
        # A proxy spec already carrying a (stale) == pin is re-pinned exactly.
        agent.mcp_servers = [
            _stdio_server("aws-mcp", "uvx", ["mcp-proxy-for-aws-cli==1.6.6", "--skip-auth"])
        ]
        parsed = _parse_written_toml(agent._build_register_mcp_servers_command() or "")
        assert parsed["mcp_servers"]["aws-mcp"]["args"][0] == "mcp-proxy-for-aws-cli==1.7.0"

    def test_non_proxy_stdio_server_not_hardened(self, agent: Codex):
        # A stdio server that is not the uvx aws proxy keeps plain command/args
        # (no pin, no timeout, no env_vars) even if it happens to be named
        # "aws-mcp" — hardening keys off the proxy package spec, not the name.
        agent.mcp_servers = [_stdio_server("aws-mcp", "uvx", ["some-other-proxy@latest"])]
        command = agent._build_register_mcp_servers_command()
        assert command is not None
        parsed = _parse_written_toml(command)
        assert parsed["mcp_servers"]["aws-mcp"] == {
            "command": "uvx",
            "args": ["some-other-proxy@latest"],
        }
