"""Tests for the CloudGeni Harbor adapter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from harbor.agents.installed.base import NonZeroAgentExitCodeError

from aws_bench.agents.cloudgeni import CloudGeniAgent


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs"


@pytest.fixture
def configured_agent(logs_dir: Path) -> CloudGeniAgent:
    return CloudGeniAgent(
        logs_dir=logs_dir,
        extra_env={
            "CLOUDGENI_API_URL": "https://api.example.com",
            "CLOUDGENI_API_KEY": "cgk_test_secret",
            "CLOUDGENI_ORGANIZATION_ID": "org_test",
            "CLOUDGENI_WORKSPACE_ID": "ws_test",
        },
    )


def _environment(return_codes: list[int] | None = None) -> MagicMock:
    codes = iter(return_codes or [0, 0, 0])
    environment = MagicMock()
    environment.exec = AsyncMock(
        side_effect=lambda **_kwargs: MagicMock(return_code=next(codes), stdout="", stderr="")
    )
    return environment


def test_name() -> None:
    assert CloudGeniAgent.name() == "cloudgeni"


@pytest.mark.asyncio
async def test_setup_uploads_standard_library_runner(configured_agent: CloudGeniAgent) -> None:
    environment = _environment([0, 0])

    await configured_agent.setup(environment)

    commands = [call.kwargs["command"] for call in environment.exec.call_args_list]
    assert "command -v python3" in commands[0]
    assert "cloudgeni_runner.py" in commands[1]
    assert "cgk_test_secret" not in "".join(commands)


@pytest.mark.asyncio
async def test_run_requires_configuration(logs_dir: Path) -> None:
    agent = CloudGeniAgent(logs_dir=logs_dir)

    with pytest.raises(ValueError, match="CLOUDGENI_API_KEY"):
        await agent.run("Do the task", _environment(), MagicMock())


@pytest.mark.asyncio
async def test_run_passes_secrets_only_as_environment(
    configured_agent: CloudGeniAgent,
) -> None:
    environment = _environment([0, 0])

    with patch.object(configured_agent, "_host_cleanup"):
        await configured_agent.run("Do the task", environment, MagicMock())

    run_call = environment.exec.call_args_list[1]
    assert run_call.kwargs["command"] == "python3 /installed-agent/cloudgeni_runner.py"
    assert run_call.kwargs["env"]["CLOUDGENI_API_KEY"] == "cgk_test_secret"
    assert "cgk_test_secret" not in run_call.kwargs["command"]


@pytest.mark.asyncio
async def test_run_cleans_up_after_bridge_failure(configured_agent: CloudGeniAgent) -> None:
    environment = _environment([0, 1])

    with patch.object(configured_agent, "_host_cleanup") as cleanup:
        with pytest.raises(NonZeroAgentExitCodeError):
            await configured_agent.run("Do the task", environment, MagicMock())

    cleanup.assert_called_once()


def test_host_cleanup_cancels_session_before_purging_lease(
    configured_agent: CloudGeniAgent, logs_dir: Path
) -> None:
    logs_dir.mkdir(parents=True)
    (logs_dir / "cloudgeni-state.json").write_text(
        json.dumps({"sessionId": "sess_1", "integrationId": "int_1"})
    )

    with patch.object(configured_agent, "_host_request", return_value=200) as request:
        configured_agent._host_cleanup()

    assert request.call_args_list[0].args == (
        "DELETE",
        "/organizations/org_test/workspaces/ws_test/agent-sessions/sess_1",
    )
    assert request.call_args_list[1].args == (
        "DELETE",
        "/organizations/org_test/workspaces/ws_test/integrations/int_1/aws-bench-credential-lease",
    )
    state = json.loads((logs_dir / "cloudgeni-state.json").read_text())
    assert state["cleanup"] == {"sessionDeleted": True, "leasePurged": True}


def test_populate_context_reads_usage(configured_agent: CloudGeniAgent, logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True)
    (logs_dir / "cloudgeni-state.json").write_text(
        json.dumps(
            {
                "sessionId": "sess_1",
                "cloudgeniRunId": "run_1",
                "status": "COMPLETED",
                "cleanup": {"sessionDeleted": True, "leasePurged": True},
                "usage": {"inputTokens": 123, "outputTokens": 45},
            }
        )
    )
    context = MagicMock(n_input_tokens=None, n_output_tokens=None, metadata=None)

    configured_agent.populate_context_post_run(context)

    assert context.n_input_tokens == 123
    assert context.n_output_tokens == 45
    assert context.metadata["cloudgeniStatus"] == "COMPLETED"
