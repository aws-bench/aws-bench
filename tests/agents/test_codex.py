"""Tests for the Bedrock-aware Codex agent.

The MCP tests pin the corrected ``config.toml`` rendering: a stdio MCP server must
emit ``command`` and ``args`` as separate TOML keys, and every value must be
escaped so it produces valid TOML. The Bedrock tests pin what reaches the
container in Bedrock mode: the bearer token and Region on the ``codex exec``
process env, and ``model_provider`` in the ``config.toml`` harbor uploads.
"""

from __future__ import annotations

import contextlib
import tomllib
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

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
        agent.mcp_servers = [
            _stdio_server(
                "aws-mcp",
                "uvx",
                ["mcp-proxy-for-aws-cli@latest", "https://example.test/mcp", "--skip-auth"],
            )
        ]

        command = agent._build_register_mcp_servers_command()
        assert command is not None
        parsed = _parse_written_toml(command)

        assert parsed == {
            "mcp_servers": {
                "aws-mcp": {
                    "command": "uvx",
                    "args": [
                        "mcp-proxy-for-aws-cli@latest",
                        "https://example.test/mcp",
                        "--skip-auth",
                    ],
                }
            }
        }

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


# ── Bedrock mode: run() ──


_BEARER = "bedrock-bearer-token-xyz"
_BEDROCK_MODEL = "global.anthropic.claude-sonnet-5"


@pytest.fixture(autouse=True)
def _clear_bedrock_env(monkeypatch: pytest.MonkeyPatch):
    """Start every test from a clean provider environment."""
    for var in (
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_REGION",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "CODEX_AUTH_JSON_PATH",
        "CODEX_FORCE_AUTH_JSON",
        "CODEX_HOME",
    ):
        monkeypatch.delenv(var, raising=False)


def _fresh_environment() -> MagicMock:
    """A fake environment that applies ``scoped_exec_env`` overlays like harbor's.

    harbor 0.22.0 delivers agent env through ``environment.scoped_exec_env`` and
    its ``_merge_env`` applies the overlay over each exec's own ``env``
    (per-exec env < scoped env). Uploaded files are recorded by remote path with
    the content they had at upload time, since harbor deletes the local copy.
    """
    environment = MagicMock()
    environment.default_user = None
    overlays: list[dict] = []
    calls: list[tuple[str, dict]] = []
    uploads: dict[str, str] = {}

    async def _exec(**kwargs):
        merged: dict = dict(kwargs.get("env") or {})
        for overlay in overlays:
            merged.update(overlay)
        calls.append((kwargs.get("command", ""), merged))
        return MagicMock(return_code=0, stdout="", stderr="")

    async def _upload_file(source_path, target_path):
        uploads[str(target_path)] = Path(source_path).read_text()

    @contextlib.contextmanager
    def _scoped_exec_env(env: dict):
        overlays.append(dict(env))
        try:
            yield
        finally:
            overlays.pop()

    environment.exec = AsyncMock(side_effect=_exec)
    environment.upload_file = AsyncMock(side_effect=_upload_file)
    environment.scoped_exec_env = _scoped_exec_env
    environment._recorded_calls = calls
    environment._uploads = uploads
    return environment


def _codex_exec_env(environment: MagicMock) -> dict:
    """Return the merged env of the final ``codex exec`` command."""
    for command, env in environment._recorded_calls:
        if "codex exec" in command:
            return env
    raise AssertionError("no `codex exec` command was executed")


def _uploaded_config(environment: MagicMock) -> dict | None:
    """Parse the ``config.toml`` harbor uploaded, or None when it uploaded none."""
    for remote_path, content in environment._uploads.items():
        if remote_path.endswith("config.toml"):
            return tomllib.loads(content)
    return None


def _uploaded_config_path(environment: MagicMock) -> str | None:
    """The remote path the ``config.toml`` was uploaded to, or None when none was."""
    for remote_path in environment._uploads:
        if remote_path.endswith("config.toml"):
            return remote_path
    return None


@pytest.mark.asyncio
async def test_run_forwards_bearer_token_into_codex_exec(
    logs_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """In Bedrock mode the bearer token and Region reach the `codex exec` process env."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", _BEARER)
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    agent = Codex(logs_dir=logs_dir, model_name=_BEDROCK_MODEL)
    environment = _fresh_environment()

    await agent.run("do the task", environment, MagicMock())

    env = _codex_exec_env(environment)
    assert env.get("AWS_BEARER_TOKEN_BEDROCK") == _BEARER
    assert env.get("AWS_REGION") == "us-west-2"


@pytest.mark.asyncio
async def test_run_bedrock_puts_model_provider_in_uploaded_config(
    logs_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The provider rides harbor's own config upload, which replaces config.toml whole."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", _BEARER)
    agent = Codex(logs_dir=logs_dir, model_name=_BEDROCK_MODEL)
    environment = _fresh_environment()

    await agent.run("do the task", environment, MagicMock())

    config = _uploaded_config(environment)
    assert config is not None
    assert config["model_provider"] == "amazon-bedrock"


@pytest.mark.asyncio
async def test_run_without_token_configures_no_provider(logs_dir: Path):
    """Without a bearer token the adapter is a no-op over harbor's Codex."""
    agent = Codex(logs_dir=logs_dir, model_name=_BEDROCK_MODEL)
    environment = _fresh_environment()

    await agent.run("do the task", environment, MagicMock())

    assert "AWS_BEARER_TOKEN_BEDROCK" not in _codex_exec_env(environment)
    config = _uploaded_config(environment)
    assert config is None or "model_provider" not in config


# ── Bedrock mode: config.toml upload (escaping and CODEX_HOME) ──

_DEFAULT_CONFIG_PATH = "/tmp/codex-home/config.toml"


async def _run_bedrock_with_args(logs_dir: Path, args: list[str], **agent_kwargs) -> MagicMock:
    """Run a Bedrock-mode trial with one stdio MCP server carrying ``args``."""
    agent = Codex(logs_dir=logs_dir, model_name=_BEDROCK_MODEL, **agent_kwargs)
    agent.mcp_servers = [_stdio_server("aws-mcp", "uvx", args)]
    environment = _fresh_environment()
    await agent.run("do the task", environment, MagicMock())
    return environment


@pytest.mark.asyncio
async def test_upload_keeps_a_literal_backslash_x_sequence(
    logs_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """``toml.dumps`` turns the four characters backslash-x-4-1 into ``A``; this keeps them."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", _BEARER)
    literal = "\\x41"

    environment = await _run_bedrock_with_args(logs_dir, [literal])

    config = _uploaded_config(environment)
    assert config is not None
    assert config["mcp_servers"]["aws-mcp"]["args"] == [literal]


@pytest.mark.asyncio
async def test_upload_survives_a_backspace_character(
    logs_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """``toml.dumps`` raises ``IndexError`` on a leading backspace and mangles a later one."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", _BEARER)
    args = ["\bkey", "tab\bkey"]

    environment = await _run_bedrock_with_args(logs_dir, args)

    config = _uploaded_config(environment)
    assert config is not None
    assert config["mcp_servers"]["aws-mcp"]["args"] == args


@pytest.mark.asyncio
async def test_upload_round_trips_quote_and_backslash(
    logs_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", _BEARER)
    args = ['arg-with-"quote"', "path\\with\\backslash", 'both "\\']

    environment = await _run_bedrock_with_args(logs_dir, args)

    config = _uploaded_config(environment)
    assert config is not None
    assert config["mcp_servers"]["aws-mcp"]["args"] == args


@pytest.mark.asyncio
async def test_upload_lands_in_the_agent_env_codex_home(
    logs_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A ``-ae CODEX_HOME`` overlay moves where codex reads, so the upload moves with it."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", _BEARER)

    environment = await _run_bedrock_with_args(
        logs_dir, ["--skip-auth"], extra_env={"CODEX_HOME": "/tmp/custom-codex"}
    )

    assert _uploaded_config_path(environment) == "/tmp/custom-codex/config.toml"
    assert _DEFAULT_CONFIG_PATH not in environment._uploads


@pytest.mark.asyncio
async def test_host_codex_home_does_not_relocate_the_upload(
    logs_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Only the agent env reaches the container, so a host ``CODEX_HOME`` must not move the file."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", _BEARER)
    monkeypatch.setenv("CODEX_HOME", "/somewhere/on/host")

    environment = await _run_bedrock_with_args(logs_dir, ["--skip-auth"])

    assert _uploaded_config_path(environment) == _DEFAULT_CONFIG_PATH


@pytest.mark.asyncio
async def test_empty_agent_env_codex_home_keeps_the_default_path(
    logs_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", _BEARER)

    environment = await _run_bedrock_with_args(
        logs_dir, ["--skip-auth"], extra_env={"CODEX_HOME": ""}
    )

    assert _uploaded_config_path(environment) == _DEFAULT_CONFIG_PATH


@pytest.mark.asyncio
async def test_openai_mode_without_servers_or_base_url_uploads_nothing(logs_dir: Path):
    """The empty-dict early return is preserved: nothing to render means no upload."""
    agent = Codex(logs_dir=logs_dir, model_name="gpt-5")
    environment = _fresh_environment()

    await agent.run("do the task", environment, MagicMock())

    assert _uploaded_config_path(environment) is None
