"""Tests for the Bedrock-aware Codex agent's MCP server registration.

These pin the corrected ``config.toml`` rendering: a stdio MCP server must emit
``command`` and ``args`` as separate TOML keys (Harbor's base flattens them into
one ``command`` string, so codex never starts the server), and every value must
be escaped so it produces valid TOML.
"""

from __future__ import annotations

import shlex
import tomllib
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from harbor.models.task.config import MCPServerConfig

from aws_bench.agents.codex import Codex

AWS_ENV_VARS = [
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_CONFIG_FILE",
    "AWS_SHARED_CREDENTIALS_FILE",
]


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
        # bug fix) and gets env_vars, but no proxy timeout or global grace.
        agent.mcp_servers = [_stdio_server("plain", "some-server", ["--flag", "value", "sub"])]

        command = agent._build_register_mcp_servers_command()
        assert command is not None
        parsed = _parse_written_toml(command)

        assert parsed["mcp_servers"]["plain"] == {
            "command": "some-server",
            "args": ["--flag", "value", "sub"],
            "env_vars": AWS_ENV_VARS,
        }
        assert "mcp_optional_startup_grace_ms" not in parsed

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

        assert parsed["mcp_servers"]["solo"] == {
            "command": "run-server",
            "args": [],
            "env_vars": AWS_ENV_VARS,
        }


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
        assert "mcp_optional_startup_grace_ms" not in parsed

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
            "env_vars": AWS_ENV_VARS,
        }
        assert parsed["mcp_servers"]["remote"] == {"url": "https://mcp.example.test/mcp"}
        assert "mcp_optional_startup_grace_ms" not in parsed


class TestCodexAwsMcpHardening:
    """Codex MCP startup workarounds.

    Every stdio server gets the AWS selector ``env_vars``. When an AWS MCP
    proxy is configured, a bounded top-level startup grace and a per-proxy
    cold-start timeout are added. The proxy's version spec is never rewritten.
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

    def test_aws_mcp_proxy_hardened(self, agent: Codex):
        agent.mcp_servers = [self._aws_mcp_server()]

        command = agent._build_register_mcp_servers_command()
        assert command is not None
        parsed = _parse_written_toml(command)

        # Bounded grace so the proxy's 2-4s uvx cold start is waited for.
        assert parsed["mcp_optional_startup_grace_ms"] == 30_000

        block = parsed["mcp_servers"]["aws-mcp"]
        # The user-supplied args (including the version spec) are untouched.
        assert block["args"] == [
            "mcp-proxy-for-aws-cli@latest",
            "aws-mcp.us-east-1.api.aws/mcp",
            "--skip-auth",
            "--metadata",
            "INSTALL_SOURCE=agent-toolkit-core",
        ]
        assert block["startup_timeout_sec"] == 120
        assert block["env_vars"] == AWS_ENV_VARS

    def test_grace_key_precedes_server_tables(self, agent: Codex):
        # The top-level grace key MUST appear before any [mcp_servers.*] header,
        # else TOML parses it as part of the last table.
        agent.mcp_servers = [
            _stdio_server("first", "run-server", []),
            self._aws_mcp_server(),
        ]
        command = agent._build_register_mcp_servers_command()
        assert command is not None
        parsed = _parse_written_toml(command)
        assert parsed["mcp_optional_startup_grace_ms"] == 30_000
        payload = command[command.index("'") + 1 : command.rindex("'")]
        assert payload.index("mcp_optional_startup_grace_ms") < payload.index("[mcp_servers.")

    @pytest.mark.parametrize(
        ("server_command", "args"),
        [
            ("uvx", ["mcp-proxy-for-aws@latest", "https://aws-mcp.us-east-1.api.aws/mcp"]),
            ("uvx", ["mcp-proxy-for-aws-cli==1.6.6", "--skip-auth"]),
            ("uvx", ["mcp-proxy-for-aws-cli>=1.7"]),
            ("uvx", ["mcp-proxy-for-aws-cli~=1.7"]),
            ("uvx", ["mcp-proxy-for-aws-cli"]),
            ("uvx", ["--from", "mcp-proxy-for-aws-cli", "mcp-proxy-for-aws"]),
            ("mcp-proxy-for-aws-cli", ["https://aws-mcp.us-east-1.api.aws/mcp"]),
            ("/opt/venv/bin/mcp-proxy-for-aws", ["https://aws-mcp.us-east-1.api.aws/mcp"]),
        ],
    )
    def test_proxy_variants_detected_and_args_untouched(
        self, agent: Codex, server_command: str, args: list[str]
    ):
        agent.mcp_servers = [_stdio_server("aws-mcp", server_command, args)]
        parsed = _parse_written_toml(agent._build_register_mcp_servers_command() or "")
        block = parsed["mcp_servers"]["aws-mcp"]
        assert parsed["mcp_optional_startup_grace_ms"] == 30_000
        assert block["command"] == server_command
        assert block["args"] == args
        assert block["startup_timeout_sec"] == 120
        assert block["env_vars"] == AWS_ENV_VARS

    @pytest.mark.parametrize(
        "args",
        [
            ["some-other-proxy@latest"],
            ["mcp-proxy-for-aws-extras"],
            ["awslabs.aws-api-mcp-server@latest"],
        ],
    )
    def test_non_proxy_stdio_server_gets_env_only(self, agent: Codex, args: list[str]):
        # Any boto3-based stdio server needs the AWS env, so env_vars is emitted
        # for every stdio server. The proxy timeout and the global grace key off
        # the proxy package name, not the server name.
        agent.mcp_servers = [_stdio_server("aws-mcp", "uvx", args)]
        command = agent._build_register_mcp_servers_command()
        assert command is not None
        parsed = _parse_written_toml(command)
        assert parsed["mcp_servers"]["aws-mcp"] == {
            "command": "uvx",
            "args": args,
            "env_vars": AWS_ENV_VARS,
        }
        assert "mcp_optional_startup_grace_ms" not in parsed

    def test_url_server_gets_no_env_vars(self, agent: Codex):
        agent.mcp_servers = [self._aws_mcp_server(), _url_server("remote", "https://x.test/mcp")]
        parsed = _parse_written_toml(agent._build_register_mcp_servers_command() or "")
        assert parsed["mcp_servers"]["remote"] == {"url": "https://x.test/mcp"}


async def _provider_toml(agent: Codex) -> str:
    """Run the Bedrock provider-config step and return the TOML it appends."""
    with patch.object(agent, "exec_as_agent", new_callable=AsyncMock) as exec_as_agent:
        await agent._write_bedrock_provider_config(MagicMock())
    command = exec_as_agent.call_args.kwargs["command"]
    return shlex.split(command.split(" && ", 1)[1])[1]


async def _written_provider_config(agent: Codex) -> dict:
    return tomllib.loads(await _provider_toml(agent))


@pytest.mark.asyncio
class TestCodexBedrockRegion:
    async def test_agent_env_region_survives_trial_override(
        self, logs_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("AWS_REGION", raising=False)
        agent = Codex(logs_dir=logs_dir, extra_env={"AWS_REGION": "us-east-2"})
        # An aws-bench trial sets the agent's AWS_REGION to the task's Region.
        agent._extra_env["AWS_REGION"] = "us-west-2"

        parsed = await _written_provider_config(agent)

        assert parsed["model_provider"] == "amazon-bedrock"
        assert parsed["model_providers"]["amazon-bedrock"]["aws"]["region"] == "us-east-2"

    async def test_host_region_is_the_fallback(
        self, logs_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        parsed = await _written_provider_config(Codex(logs_dir=logs_dir))

        assert parsed["model_providers"]["amazon-bedrock"]["aws"]["region"] == "us-east-1"

    async def test_default_region(self, logs_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("AWS_REGION", raising=False)
        parsed = await _written_provider_config(Codex(logs_dir=logs_dir))

        assert parsed["model_providers"]["amazon-bedrock"]["aws"]["region"] == "us-east-2"

    async def test_region_pin_stays_above_appended_tables(self, logs_dir: Path):
        agent = Codex(logs_dir=logs_dir)
        agent.mcp_servers = [_stdio_server("aws-mcp", "uvx", ["proxy@latest"])]
        mcp_command = agent._build_register_mcp_servers_command()
        assert mcp_command is not None
        mcp_toml = mcp_command[mcp_command.index("'") + 1 : mcp_command.rindex("'")]

        # config.toml is the provider block, then the tables harbor appends.
        parsed = tomllib.loads(await _provider_toml(agent) + mcp_toml)

        assert parsed["model_providers"]["amazon-bedrock"]["aws"]["region"]
        assert "model_providers" not in parsed["mcp_servers"]["aws-mcp"]
