"""Tests for the provider-aware Codex agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from harbor.agents.installed.codex import Codex as HarborCodex

from aws_bench.agents.codex import Codex


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs"


def test_azure_settings_accept_complete_extra_env(logs_dir: Path) -> None:
    agent = Codex(
        logs_dir=logs_dir,
        extra_env={
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "azure-secret",
        },
    )

    assert agent._azure_settings() == (
        "https://example.openai.azure.com",
        "azure-secret",
    )


def test_azure_settings_reject_partial_configuration(logs_dir: Path) -> None:
    agent = Codex(
        logs_dir=logs_dir,
        extra_env={"AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com"},
    )

    with pytest.raises(ValueError, match="requires both"):
        agent._azure_settings()


@pytest.mark.asyncio
async def test_run_writes_azure_provider_before_harbor_run(logs_dir: Path) -> None:
    agent = Codex(
        logs_dir=logs_dir,
        model_name="gpt-5.6-sol",
        extra_env={
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "azure-secret",
        },
    )
    environment = MagicMock()
    agent.exec_as_agent = AsyncMock()

    with (
        patch.object(Codex, "_is_bedrock_mode", return_value=False),
        patch.object(HarborCodex, "run", new_callable=AsyncMock) as harbor_run,
    ):
        await agent.run("Do the task", environment, MagicMock())

    provider_command = agent.exec_as_agent.call_args.kwargs["command"]
    assert "azure_openai" in provider_command
    assert "https://example.openai.azure.com/openai/v1" in provider_command
    assert "azure-secret" not in provider_command
    assert agent._extra_env["AZURE_OPENAI_API_KEY"] == "azure-secret"
    harbor_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_rejects_multiple_non_openai_providers(logs_dir: Path) -> None:
    agent = Codex(
        logs_dir=logs_dir,
        model_name="gpt-5.6-sol",
        extra_env={
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "azure-secret",
        },
    )

    with patch.object(Codex, "_is_bedrock_mode", return_value=True):
        with pytest.raises(ValueError, match="only one Codex provider"):
            await agent.run("Do the task", MagicMock(), MagicMock())
