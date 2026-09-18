"""Tests for the Bedrock-capable MiniSwe agent.

The install tests execute the recorded reinstall command in a real shell, with
``HOME`` and ``PATH`` built from scratch, so the ``uv`` that runs is always a stub
recording its argv and never the developer's or CI runner's real ``uv``.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from harbor.agents.installed.mini_swe_agent import MiniSweAgent as HarborMiniSweAgent

from aws_bench.agents.mini_swe_agent import MiniSweAgent

_BEDROCK_MODEL = "bedrock/us.anthropic.claude-sonnet-4-6"
# The trial blanks the raw credential variables on the agent's env so the
# container falls back to AWS_PROFILE; this is the entry that shadows the key.
_EMPTY_GUARD = {"AWS_ACCESS_KEY_ID": ""}


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs"


@pytest.fixture
def agent(logs_dir: Path) -> MiniSweAgent:
    return MiniSweAgent(logs_dir=logs_dir, model_name=_BEDROCK_MODEL)


@pytest.fixture(autouse=True)
def _clear_model_env(monkeypatch: pytest.MonkeyPatch):
    """Start every test from a host environment with no model or AWS variables."""
    for var in (
        "MSWEA_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def _fresh_environment() -> MagicMock:
    """A fake environment recording ``(command, env)`` per exec, every exec succeeding."""
    environment = MagicMock()
    environment.default_user = None
    calls: list[tuple[str, dict]] = []

    async def _exec(**kwargs):
        calls.append((kwargs.get("command", ""), dict(kwargs.get("env") or {})))
        return MagicMock(return_code=0, stdout="", stderr="")

    environment.exec = AsyncMock(side_effect=_exec)
    environment._recorded_calls = calls
    return environment


# ── install ──


async def _reinstall_command(agent: MiniSweAgent) -> str:
    """Run ``install`` against a fake environment and return the boto3 reinstall command."""
    environment = _fresh_environment()
    await agent.install(environment)
    return environment._recorded_calls[-1][0]


def _write_stub_uv(directory: Path, argv_log: Path) -> None:
    """Place an executable ``uv`` in ``directory`` that appends its argv to ``argv_log``."""
    directory.mkdir(parents=True, exist_ok=True)
    stub = directory / "uv"
    stub.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> {shlex.quote(str(argv_log))}\n')
    stub.chmod(0o755)


def _run_in_shell(
    command: str, *, home: Path, path_dirs: list[Path]
) -> subprocess.CompletedProcess:
    """Execute ``command`` under ``/bin/bash -c`` with an environment built from scratch.

    Only ``HOME`` and ``PATH`` are set. ``PATH`` holds nothing but ``path_dirs``, and
    every other word in the command is a shell builtin, so no ``UV_*`` or ``XDG_*``
    variable and no ``uv`` from the host, ``/usr/bin`` included, can reach it.
    """
    env = {"HOME": str(home), "PATH": ":".join(str(d) for d in path_dirs)}
    return subprocess.run(["/bin/bash", "-c", command], env=env, capture_output=True, text=True)


@pytest.mark.asyncio
async def test_reinstall_guards_the_env_file_and_keeps_the_boto3_install(agent: MiniSweAgent):
    """The env-file source is conditional, PATH is exported, and the install tail is intact."""
    command = await _reinstall_command(agent)

    assert 'if [ -f "$HOME/.local/bin/env" ]; then source "$HOME/.local/bin/env"; fi' in command
    assert 'export PATH="$HOME/.local/bin:$PATH"' in command
    assert "uv tool install mini-swe-agent --with boto3 --with 'litellm<=1.91.3' --force" in command


@pytest.mark.asyncio
async def test_reinstall_runs_with_preinstalled_uv_and_no_env_file(
    agent: MiniSweAgent, tmp_path: Path
):
    """On harbor 0.22.0 a preinstalled ``uv`` skips the bootstrap, so no env file is written."""
    command = await _reinstall_command(agent)
    home = tmp_path / "home"
    home.mkdir()
    argv_log = tmp_path / "uv-argv.log"
    _write_stub_uv(tmp_path / "bin", argv_log)

    result = _run_in_shell(command, home=home, path_dirs=[tmp_path / "bin"])

    assert result.returncode == 0, result.stderr
    assert "tool install mini-swe-agent --with boto3 --with litellm<=1.91.3 --force" in (
        argv_log.read_text()
    )


@pytest.mark.asyncio
async def test_reinstall_sources_the_env_file_when_it_is_the_only_route_to_uv(
    agent: MiniSweAgent, tmp_path: Path
):
    """After a fresh uv bootstrap the env file is what puts ``uv`` on PATH."""
    command = await _reinstall_command(agent)
    home = tmp_path / "home"
    argv_log = tmp_path / "uv-argv.log"
    _write_stub_uv(tmp_path / "elsewhere", argv_log)
    env_file = home / ".local" / "bin" / "env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(f'export PATH="{tmp_path / "elsewhere"}:$PATH"\n')
    empty = tmp_path / "empty"
    empty.mkdir()

    result = _run_in_shell(command, home=home, path_dirs=[empty])

    assert result.returncode == 0, result.stderr
    assert "tool install mini-swe-agent" in argv_log.read_text()


@pytest.mark.asyncio
async def test_reinstall_fails_loudly_without_uv_or_env_file(agent: MiniSweAgent, tmp_path: Path):
    """Neither route to ``uv`` present: the command exits non-zero instead of silently passing."""
    command = await _reinstall_command(agent)
    home = tmp_path / "home"
    home.mkdir()
    empty = tmp_path / "empty"
    empty.mkdir()

    result = _run_in_shell(command, home=home, path_dirs=[empty])

    assert result.returncode != 0


@pytest.mark.asyncio
async def test_reinstall_carries_the_pinned_version(logs_dir: Path, tmp_path: Path):
    """An ``--agent-version`` pin survives the ``--force`` reinstall."""
    agent = MiniSweAgent(logs_dir=logs_dir, model_name=_BEDROCK_MODEL, version="1.2.3")
    command = await _reinstall_command(agent)
    home = tmp_path / "home"
    home.mkdir()
    argv_log = tmp_path / "uv-argv.log"
    _write_stub_uv(tmp_path / "bin", argv_log)

    result = _run_in_shell(command, home=home, path_dirs=[tmp_path / "bin"])

    assert result.returncode == 0, result.stderr
    assert "tool install mini-swe-agent==1.2.3 " in argv_log.read_text()


# ── model_connection on Bedrock ──


def _mini_swe_exec_env(environment: MagicMock) -> dict:
    """Return the env of the ``mini-swe-agent`` command ``run()`` executed."""
    for command, env in environment._recorded_calls:
        if "mini-swe-agent --yolo" in command:
            return env
    raise AssertionError("no `mini-swe-agent` command was executed")


def test_bedrock_restores_the_host_generic_key_over_the_empty_guard(
    logs_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """On harbor 0.22.0 the agent env resolves first, so the empty guard shadows a host key."""
    monkeypatch.setenv("MSWEA_API_KEY", "model-key")
    agent = MiniSweAgent(logs_dir=logs_dir, model_name=_BEDROCK_MODEL, extra_env=_EMPTY_GUARD)

    access = agent.model_connection

    assert access.api_key == "model-key"
    assert access.env["MSWEA_API_KEY"] == "model-key"


def test_bedrock_explicit_generic_key_beats_the_host_value(
    logs_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MSWEA_API_KEY", "host-key")
    agent = MiniSweAgent(
        logs_dir=logs_dir,
        model_name=_BEDROCK_MODEL,
        extra_env={**_EMPTY_GUARD, "MSWEA_API_KEY": "explicit"},
    )

    access = agent.model_connection

    assert access.api_key == "explicit"
    assert access.env["MSWEA_API_KEY"] == "explicit"


def test_bedrock_explicitly_empty_generic_key_is_left_as_harbor_resolves_it(
    logs_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """An explicit ``-ae MSWEA_API_KEY=`` resolves first; harbor already writes the empty value."""
    monkeypatch.setenv("MSWEA_API_KEY", "host-key")
    extra_env = {**_EMPTY_GUARD, "MSWEA_API_KEY": ""}
    agent = MiniSweAgent(logs_dir=logs_dir, model_name=_BEDROCK_MODEL, extra_env=extra_env)
    harbor_agent = HarborMiniSweAgent(
        logs_dir=logs_dir, model_name=_BEDROCK_MODEL, extra_env=extra_env
    )

    access = agent.model_connection

    assert access.api_key == ""
    assert access.env["MSWEA_API_KEY"] == ""
    assert access == harbor_agent.model_connection


def test_bedrock_without_any_generic_key_is_unchanged_from_harbor(logs_dir: Path):
    agent = MiniSweAgent(logs_dir=logs_dir, model_name=_BEDROCK_MODEL, extra_env=_EMPTY_GUARD)
    harbor_agent = HarborMiniSweAgent(
        logs_dir=logs_dir, model_name=_BEDROCK_MODEL, extra_env=_EMPTY_GUARD
    )

    access = agent.model_connection

    assert access.api_key == ""
    assert access == harbor_agent.model_connection


def test_non_bedrock_provider_is_unchanged_from_harbor(
    logs_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The restore is narrow: an OpenAI model with the same guard resolves as harbor does."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    agent = MiniSweAgent(logs_dir=logs_dir, model_name="openai/gpt-5", extra_env=_EMPTY_GUARD)
    harbor_agent = HarborMiniSweAgent(
        logs_dir=logs_dir, model_name="openai/gpt-5", extra_env=_EMPTY_GUARD
    )

    access = agent.model_connection

    assert access.api_key == "openai-key"
    assert access == harbor_agent.model_connection


@pytest.mark.asyncio
async def test_run_passes_the_restored_key_to_mini_swe_agent(
    logs_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Through ``run()``, the launched mini-swe-agent sees the host key despite the guard."""
    monkeypatch.setenv("MSWEA_API_KEY", "model-key")
    agent = MiniSweAgent(logs_dir=logs_dir, model_name=_BEDROCK_MODEL, extra_env=_EMPTY_GUARD)
    environment = _fresh_environment()

    await agent.run("do the task", environment, MagicMock())

    assert _mini_swe_exec_env(environment)["MSWEA_API_KEY"] == "model-key"
