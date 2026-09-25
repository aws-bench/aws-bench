"""Tests for the Kiro CLI agent."""

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aws_bench.agents.kiro_cli import KiroCli
from aws_bench.cli.preflight import PreflightError


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs"


@pytest.fixture
def agent(logs_dir: Path) -> KiroCli:
    return KiroCli(logs_dir=logs_dir)


class TestKiroCliName:
    def test_name(self):
        assert KiroCli.name() == "kiro-cli"


class TestKiroCliInit:
    def test_default_init(self, agent: KiroCli):
        assert agent.model_name is None

    def test_init_with_model(self, logs_dir: Path):
        agent = KiroCli(logs_dir=logs_dir, model_name="auto")
        assert agent.model_name == "auto"


class TestKiroCliVersionCommand:
    def test_get_version_command(self, agent: KiroCli):
        cmd = agent.get_version_command()
        assert cmd is not None
        assert "kiro-cli" in cmd
        assert "--version" in cmd


class TestKiroCliInstall:
    @pytest.mark.asyncio
    async def test_install_calls_exec(self, agent: KiroCli):
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))

        await agent.install(environment)

        assert environment.exec.call_count >= 2

    @pytest.mark.asyncio
    async def test_install_disables_greeting(self, agent: KiroCli):
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))

        await agent.install(environment)

        calls = environment.exec.call_args_list
        install_call = calls[1]
        command = install_call.kwargs.get("command", "")
        assert "chat.greeting.enabled false" in command


class TestKiroCliSetup:
    @pytest.mark.asyncio
    async def test_setup_raises_when_kiro_api_key_missing(self, agent: KiroCli):
        environment = MagicMock()

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(PreflightError, match="KIRO_API_KEY"):
                await agent.setup(environment)

        environment.exec.assert_not_called()


class TestKiroCliRun:
    @staticmethod
    def _run_cmds(calls):
        return [
            c.kwargs.get("command", "")
            for c in calls
            if "kiro-cli chat" in c.kwargs.get("command", "")
        ]

    @pytest.mark.asyncio
    async def test_run_basic(self, agent: KiroCli):
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        context = MagicMock()

        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test"}, clear=True):
            await agent.run("Do the task", environment, context)

        calls = environment.exec.call_args_list
        run_cmds = self._run_cmds(calls)
        assert run_cmds
        command = run_cmds[0]
        assert "--no-interactive" in command
        assert "--trust-all-tools" in command
        assert "Do the task" in command
        assert "/logs/agent/kiro-cli.txt" in command

    @pytest.mark.asyncio
    async def test_run_passes_kiro_api_key(self, agent: KiroCli):
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        context = MagicMock()

        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test123"}, clear=True):
            await agent.run("Do the task", environment, context)

        calls = environment.exec.call_args_list
        # The main run call should have the env
        run_call = [c for c in calls if "kiro-cli chat" in c.kwargs.get("command", "")][0]
        env = run_call.kwargs.get("env", {})
        assert env.get("KIRO_API_KEY") == "ksk_test123"

    @pytest.mark.asyncio
    async def test_run_with_model_flag(self, logs_dir: Path):
        agent = KiroCli(logs_dir=logs_dir, model_name="sonnet-4")
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        context = MagicMock()

        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test"}, clear=True):
            await agent.run("Do the task", environment, context)

        run_cmds = self._run_cmds(environment.exec.call_args_list)
        assert "--model" in run_cmds[0]
        assert "sonnet-4" in run_cmds[0]

    @pytest.mark.asyncio
    async def test_run_with_effort(self, logs_dir: Path):
        agent = KiroCli(logs_dir=logs_dir, effort="high")
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        context = MagicMock()

        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test"}, clear=True):
            await agent.run("Do the task", environment, context)

        run_cmds = self._run_cmds(environment.exec.call_args_list)
        assert "--effort high" in run_cmds[0]

    @pytest.mark.asyncio
    async def test_run_with_mcp_servers(self, logs_dir: Path):
        agent = KiroCli(logs_dir=logs_dir)
        mock_server = MagicMock()
        mock_server.name = "test-server"
        mock_server.transport = "stdio"
        mock_server.command = "node"
        mock_server.args = ["server.js"]
        agent.mcp_servers = cast(list, [mock_server])

        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        context = MagicMock()

        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test"}, clear=True):
            await agent.run("Do the task", environment, context)

        calls = environment.exec.call_args_list
        mcp_call = calls[0]
        mcp_command = mcp_call.kwargs.get("command", "")
        assert "mcp.json" in mcp_command
        assert "test-server" in mcp_command

    @pytest.mark.asyncio
    async def test_run_with_skills_dir(self, logs_dir: Path):
        agent = KiroCli(logs_dir=logs_dir, skills_dir="/harbor/skills")
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        context = MagicMock()

        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test"}, clear=True):
            await agent.run("Do the task", environment, context)

        calls = environment.exec.call_args_list
        skills_cmds = [
            c.kwargs.get("command", "") for c in calls if "cp -r" in c.kwargs.get("command", "")
        ]
        assert skills_cmds
        assert "/harbor/skills" in skills_cmds[0]
        assert "~/.kiro/skills/" in skills_cmds[0]


class TestKiroCliBuildMcpJson:
    def test_returns_none_for_empty(self):
        assert KiroCli._build_mcp_json([]) is None

    def test_builds_stdio_server(self):
        server = MagicMock()
        server.name = "my-server"
        server.transport = "stdio"
        server.command = "node"
        server.args = ["index.js"]

        result = KiroCli._build_mcp_json([server])
        assert result == {"my-server": {"command": "node", "args": ["index.js"]}}

    def test_builds_http_server(self):
        server = MagicMock()
        server.name = "remote"
        server.transport = "streamable-http"
        server.url = "http://localhost:3000"

        result = KiroCli._build_mcp_json([server])
        assert result == {"remote": {"url": "http://localhost:3000"}}


class TestAgentEngineFlag:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("engine", ["v1", "v2", "v3"])
    async def test_run_with_agent_engine(self, logs_dir: Path, engine: str):
        agent = KiroCli(logs_dir=logs_dir, agent_engine=engine)
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test"}, clear=True):
            await agent.run("Do the task", environment, MagicMock())
        run_cmd = [
            c.kwargs.get("command", "")
            for c in environment.exec.call_args_list
            if "kiro-cli chat" in c.kwargs.get("command", "")
        ][0]
        assert f"--agent-engine {engine}" in run_cmd

    def test_rejects_unknown_engine(self, logs_dir: Path):
        with pytest.raises(ValueError):
            KiroCli(logs_dir=logs_dir, agent_engine="v9")


class TestAgentEngineDefault:
    def _run_cmd(self, environment) -> str:
        return [
            c.kwargs.get("command", "")
            for c in environment.exec.call_args_list
            if "kiro-cli chat" in c.kwargs.get("command", "")
        ][0]

    @pytest.mark.asyncio
    async def test_defaults_to_v3_and_warns(self, logs_dir: Path):
        logger = MagicMock()
        agent = KiroCli(logs_dir=logs_dir, logger=logger)
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test"}, clear=True):
            await agent.run("Do the task", environment, MagicMock())
        assert "--agent-engine v3" in self._run_cmd(environment)
        warning = " ".join(str(a) for a in logger.getChild.return_value.warning.call_args[0])
        assert "v3" in warning and "agent_engine" in warning

    @pytest.mark.asyncio
    async def test_explicit_engine_does_not_warn(self, logs_dir: Path):
        logger = MagicMock()
        agent = KiroCli(logs_dir=logs_dir, logger=logger, agent_engine="v1")
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test"}, clear=True):
            await agent.run("Do the task", environment, MagicMock())
        assert "--agent-engine v1" in self._run_cmd(environment)
        logger.getChild.return_value.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_env_fallback_does_not_warn(self, logs_dir: Path):
        logger = MagicMock()
        with patch.dict(
            "os.environ", {"KIRO_API_KEY": "ksk_test", "KIRO_CLI_AGENT_ENGINE": "v2"}, clear=True
        ):
            agent = KiroCli(logs_dir=logs_dir, logger=logger)
            environment = MagicMock()
            environment.exec = AsyncMock(
                return_value=MagicMock(return_code=0, stdout="", stderr="")
            )
            await agent.run("Do the task", environment, MagicMock())
        assert "--agent-engine v2" in self._run_cmd(environment)
        logger.getChild.return_value.warning.assert_not_called()
