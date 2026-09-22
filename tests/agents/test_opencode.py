"""Tests for the Bedrock-aware OpenCode agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from harbor.agents.factory import AgentFactory
from harbor.models.agent.name import AgentName

from aws_bench.agents.opencode import _DEFAULT_AWS_REGION, OpenCode

_BEARER = "bedrock-bearer-token-xyz"
_MODEL = "amazon-bedrock/global.anthropic.claude-sonnet-5"
# Test-account SigV4 chain, as injected into a trial so the agent can act on
# resources under test. Must NOT be treated as Bedrock inference credentials.
_TEST_ACCESS_KEY = "AKIA_TEST_ACCOUNT"
_TEST_SECRET_KEY = "test-account-secret"


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs"


@pytest.fixture
def agent(logs_dir: Path) -> OpenCode:
    return OpenCode(logs_dir=logs_dir, model_name=_MODEL)


def _fresh_environment() -> MagicMock:
    environment = MagicMock()
    environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
    environment.default_user = None
    return environment


def _exec_calls(environment: MagicMock) -> list[tuple[str, dict]]:
    """Return (command, env) for each environment.exec call."""
    calls = []
    for c in environment.exec.call_args_list:
        calls.append((c.kwargs.get("command", ""), c.kwargs.get("env") or {}))
    return calls


def _opencode_run_env(environment: MagicMock) -> dict:
    """Return the env of the final `opencode run` command."""
    for command, env in _exec_calls(environment):
        if "opencode --model=" in command:
            return env
    raise AssertionError("no `opencode run` command was executed")


@pytest.fixture(autouse=True)
def _clear_bedrock_env(monkeypatch: pytest.MonkeyPatch):
    """Start every test from a clean AWS environment."""
    for var in (
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)


# -- name / factory registration --


def test_name_is_opencode():
    """Keeps the built-in name so it overrides Harbor's opencode."""
    assert OpenCode.name() == "opencode"


def test_factory_resolves_opencode_to_subclass():
    """Importing aws_bench.agents maps `opencode` to this subclass."""
    import aws_bench.agents  # noqa: F401

    assert AgentFactory._AGENT_MAP[AgentName("opencode")] is OpenCode


def test_subclass_replaces_builtin_in_agents_list():
    """The subclass replaces Harbor's opencode in the factory list."""
    import aws_bench.agents  # noqa: F401

    entries = [a for a in AgentFactory._AGENTS if a.name() == "opencode"]
    assert entries == [OpenCode]


# -- bedrock-mode detection --


def test_bedrock_mode_true_when_token_set(agent: OpenCode, monkeypatch: pytest.MonkeyPatch):
    """A non-empty bearer token turns on Bedrock mode."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", _BEARER)
    assert agent._is_bedrock_mode() is True


def test_bedrock_mode_false_when_token_absent(agent: OpenCode):
    """No bearer token means no Bedrock mode."""
    assert agent._is_bedrock_mode() is False


def test_bedrock_mode_false_when_token_blank(agent: OpenCode, monkeypatch: pytest.MonkeyPatch):
    """A whitespace-only token does not count as Bedrock mode."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "   ")
    assert agent._is_bedrock_mode() is False


def test_bedrock_mode_true_when_token_in_extra_env(logs_dir: Path):
    """A token supplied via extra_env (-ae) counts, not just os.environ.

    --ae values land in the agent's _extra_env, never the host environment, so
    detection must go through Harbor's _get_env (extra_env first, then host).
    """
    agent = OpenCode(
        logs_dir=logs_dir,
        model_name=_MODEL,
        extra_env={"AWS_BEARER_TOKEN_BEDROCK": _BEARER},
    )
    assert agent._is_bedrock_mode() is True


def test_inject_reads_token_from_extra_env(logs_dir: Path):
    """Injection also resolves the token via _get_env, so the -ae path is complete."""
    agent = OpenCode(
        logs_dir=logs_dir,
        model_name=_MODEL,
        extra_env={"AWS_BEARER_TOKEN_BEDROCK": _BEARER},
    )
    agent._inject_bedrock_env()
    assert agent._extra_env["AWS_BEARER_TOKEN_BEDROCK"] == _BEARER
    assert agent._extra_env["AWS_REGION"] == _DEFAULT_AWS_REGION


# -- env injection --


def test_inject_forwards_token(agent: OpenCode, monkeypatch: pytest.MonkeyPatch):
    """The bearer token is placed into _extra_env for every exec to inherit."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", _BEARER)
    agent._inject_bedrock_env()
    assert agent._extra_env["AWS_BEARER_TOKEN_BEDROCK"] == _BEARER


def test_inject_defaults_region_when_host_unset(agent: OpenCode, monkeypatch: pytest.MonkeyPatch):
    """Region falls back to the token's mint Region when the host has none."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", _BEARER)
    agent._inject_bedrock_env()
    assert agent._extra_env["AWS_REGION"] == _DEFAULT_AWS_REGION


def test_inject_honors_host_region(agent: OpenCode, monkeypatch: pytest.MonkeyPatch):
    """A host AWS_REGION is honored over the default."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", _BEARER)
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    agent._inject_bedrock_env()
    assert agent._extra_env["AWS_REGION"] == "eu-west-1"


def test_inject_does_not_override_explicit_extra_env(
    logs_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Values supplied via extra_env (e.g. -ae) win over the injected ones."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", _BEARER)
    agent = OpenCode(
        logs_dir=logs_dir,
        model_name=_MODEL,
        extra_env={"AWS_REGION": "ap-south-1"},
    )
    agent._inject_bedrock_env()
    assert agent._extra_env["AWS_REGION"] == "ap-south-1"


# -- run() integration: what actually reaches `opencode run` --


@pytest.mark.asyncio
async def test_run_forwards_bearer_token_into_opencode_run(
    agent: OpenCode, monkeypatch: pytest.MonkeyPatch
):
    """In Bedrock mode the bearer token reaches the `opencode run` process env."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", _BEARER)
    environment = _fresh_environment()

    await agent.run("do the task", environment, MagicMock())

    env = _opencode_run_env(environment)
    assert env.get("AWS_BEARER_TOKEN_BEDROCK") == _BEARER
    assert env.get("AWS_REGION") == _DEFAULT_AWS_REGION


@pytest.mark.asyncio
async def test_run_pins_region_from_host(agent: OpenCode, monkeypatch: pytest.MonkeyPatch):
    """A host AWS_REGION present in the trial is carried into the run env."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", _BEARER)
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    environment = _fresh_environment()

    await agent.run("do the task", environment, MagicMock())

    assert _opencode_run_env(environment).get("AWS_REGION") == "us-west-2"


@pytest.mark.asyncio
async def test_run_leaves_test_account_sigv4_in_place(
    agent: OpenCode, monkeypatch: pytest.MonkeyPatch
):
    """The test-account SigV4 chain is preserved for the agent's own tools."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", _BEARER)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", _TEST_ACCESS_KEY)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", _TEST_SECRET_KEY)
    environment = _fresh_environment()

    await agent.run("do the task", environment, MagicMock())

    env = _opencode_run_env(environment)
    # Both credential types coexist: bearer for Bedrock, SigV4 for the tools.
    assert env.get("AWS_BEARER_TOKEN_BEDROCK") == _BEARER
    assert env.get("AWS_ACCESS_KEY_ID") == _TEST_ACCESS_KEY
    assert env.get("AWS_SECRET_ACCESS_KEY") == _TEST_SECRET_KEY


@pytest.mark.asyncio
async def test_run_without_token_injects_nothing(agent: OpenCode, monkeypatch: pytest.MonkeyPatch):
    """Without a bearer token the adapter is a no-op over Harbor's OpenCode."""
    environment = _fresh_environment()

    await agent.run("do the task", environment, MagicMock())

    assert agent._extra_env == {}
    assert "AWS_BEARER_TOKEN_BEDROCK" not in _opencode_run_env(environment)
