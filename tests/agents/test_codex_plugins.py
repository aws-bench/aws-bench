"""Per-trial plugin setup for Codex (no network or real AWS calls)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig

from aws_bench.agents.codex import Codex


@pytest.mark.parametrize("options", [{}, {"marketplaces": None, "plugins": None}])
def test_no_plugins(tmp_path: Path, options: dict):
    agent = Codex(logs_dir=tmp_path / "logs", **options)
    assert agent._marketplaces == []
    assert agent._plugins == []
    assert agent._build_register_mcp_servers_command() is None


def test_copies_plugin_options(tmp_path: Path):
    marketplaces = ["owner/repo"]
    plugins = ["sample@market"]
    agent = Codex(logs_dir=tmp_path / "logs", marketplaces=marketplaces, plugins=plugins)
    marketplaces.clear()
    plugins.clear()
    assert agent._marketplaces == ["owner/repo"]
    assert agent._plugins == ["sample@market"]


def test_marketplace_without_plugin_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="marketplaces were given without plugins"):
        Codex(logs_dir=tmp_path / "logs", marketplaces=["owner/repo"])


@pytest.fixture
def shell_env(tmp_path: Path) -> dict[str, str]:
    """A fake Codex executable records argv and can fail any selected step."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "codex"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(42 if sys.argv[-1] == os.environ.get('FAIL_SPEC') else 0)\n"
    )
    executable.chmod(0o755)
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    return {
        **os.environ,
        "HOME": str(tmp_path),
        "CODEX_HOME": str(codex_home),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CALLS": str(tmp_path / "calls.jsonl"),
        "FAIL_SPEC": "",
    }


def _run_setup(agent: Codex, env: dict[str, str]) -> tuple[int, list[list[str]]]:
    command = agent._build_register_mcp_servers_command()
    assert command is not None
    result = subprocess.run(
        ["bash", "-c", command], env=env, cwd=env["HOME"], capture_output=True, text=True
    )
    calls = Path(env["CALLS"])
    return result.returncode, [json.loads(line) for line in calls.read_text().splitlines()]


def test_plugins_without_explicit_marketplaces(tmp_path: Path, shell_env: dict[str, str]):
    agent = Codex(logs_dir=tmp_path / "logs", plugins=["sample@market"])
    code, calls = _run_setup(agent, shell_env)
    assert code == 0
    assert calls == [["plugin", "add", "sample@market"]]


@pytest.mark.parametrize("failed_step", range(4))
def test_any_failure_stops_setup(tmp_path: Path, shell_env: dict[str, str], failed_step: int):
    specs = ["owner/one", "owner/two", "first@one", "second@two"]
    agent = Codex(logs_dir=tmp_path / "logs", marketplaces=specs[:2], plugins=specs[2:])
    shell_env["FAIL_SPEC"] = specs[failed_step]
    code, calls = _run_setup(agent, shell_env)
    assert code == 42
    assert [call[-1] for call in calls] == specs[: failed_step + 1]


def test_plugins_preserve_bedrock_and_mcp_config(tmp_path: Path, shell_env: dict[str, str]):
    config = Path(shell_env["CODEX_HOME"]) / "config.toml"
    config.write_text('model_provider = "amazon-bedrock"\n')
    agent = Codex(
        logs_dir=tmp_path / "logs",
        plugins=["sample@market"],
        mcp_servers=[MCPServerConfig(name="tools", transport="stdio", command="uvx", args=["run"])],
    )
    command = agent._build_register_mcp_servers_command()
    assert command is not None
    assert command.index("config.toml") < command.index("codex plugin add")
    code, _ = _run_setup(agent, shell_env)
    assert code == 0
    assert tomllib.loads(config.read_text()) == {
        "model_provider": "amazon-bedrock",
        "mcp_servers": {"tools": {"command": "uvx", "args": ["run"]}},
    }


@pytest.mark.asyncio
async def test_setup_failure_prevents_task_execution(tmp_path: Path, monkeypatch):
    for key in ("CODEX_AUTH_JSON_PATH", "CODEX_FORCE_AUTH_JSON", "AWS_BEARER_TOKEN_BEDROCK"):
        monkeypatch.delenv(key, raising=False)
    agent = Codex(
        logs_dir=tmp_path / "logs", model_name="openai/test-model", plugins=["sample@market"]
    )
    with patch.object(
        agent, "exec_as_agent", new_callable=AsyncMock, side_effect=[None, RuntimeError("failed")]
    ) as execute:
        with pytest.raises(RuntimeError, match="failed"):
            await agent.run("test instruction", MagicMock(), AgentContext())
    assert not any("codex exec " in c.kwargs["command"] for c in execute.call_args_list)
