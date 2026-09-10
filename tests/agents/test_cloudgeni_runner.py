"""Unit tests for the standard-library CloudGeni container bridge."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aws_bench.agents import cloudgeni_runner as runner


@pytest.fixture
def bridge_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    logs = tmp_path / "logs"
    monkeypatch.setattr(runner, "_PROMPT_PATH", tmp_path / "prompt.txt")
    monkeypatch.setattr(runner, "_LOGS_DIR", logs)
    monkeypatch.setattr(runner, "_OUTPUT_TEXT_PATH", logs / "agent-output.txt")
    monkeypatch.setattr(runner, "_OUTPUT_JSON_PATH", logs / "agent-output.json")
    monkeypatch.setattr(runner, "_STATE_PATH", logs / "cloudgeni-state.json")
    runner._PROMPT_PATH.write_text("Return JSON to /logs/agent/agent-output.json")
    return logs


@pytest.fixture(autouse=True)
def bridge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "CLOUDGENI_API_URL": "https://api.example.com",
        "CLOUDGENI_API_KEY": "cgk_test_secret",
        "CLOUDGENI_ORGANIZATION_ID": "org_test",
        "CLOUDGENI_WORKSPACE_ID": "ws_test",
        "AWS_REGION": "us-west-2",
        "AWS_PROFILE": "trial",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_api_root_normalizes_version() -> None:
    assert runner._api_root("https://api.example.com/") == "https://api.example.com/api/v1"
    assert runner._api_root("https://api.example.com/api/v1") == ("https://api.example.com/api/v1")


def test_credentials_reads_temporary_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credentials_path = tmp_path / "credentials"
    credentials_path.write_text(
        "[trial]\n"
        "aws_access_key_id=ASIAABCDEFGHIJKLMNOP\n"
        "aws_secret_access_key=abcdefghijklmnopqrstuvwxyz0123456789ABCD\n"
        "aws_session_token=temporary-session-token\n"
    )
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials_path))

    values = runner._credentials()

    assert values["accessKeyId"] == "ASIAABCDEFGHIJKLMNOP"
    assert values["sessionToken"] == "temporary-session-token"


def test_credentials_rejects_static_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    credentials_path = tmp_path / "credentials"
    credentials_path.write_text(
        "[trial]\n"
        "aws_access_key_id=AKIAABCDEFGHIJKLMNOP\n"
        "aws_secret_access_key=abcdefghijklmnopqrstuvwxyz0123456789ABCD\n"
        "aws_session_token=temporary-session-token\n"
    )
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials_path))

    with pytest.raises(runner.BridgeError, match="temporary ASIA"):
        runner._credentials()


def test_caller_account_id_uses_scoped_profile() -> None:
    completed = MagicMock(stdout='{"Account":"123456789012"}')

    with patch.object(runner.subprocess, "run", return_value=completed) as run:
        assert runner._caller_account_id() == "123456789012"

    assert run.call_args.args[0][0:4] == ["aws", "sts", "get-caller-identity", "--profile"]
    assert "trial" in run.call_args.args[0]


def test_write_output_extracts_fenced_json(bridge_paths: Path) -> None:
    runner._write_output(
        "Write /logs/agent/agent-output.json",
        'Result:\n```json\n{"resourceId":"r-1"}\n```',
    )

    assert json.loads((bridge_paths / "agent-output.json").read_text()) == {"resourceId": "r-1"}
    assert "Result:" in (bridge_paths / "agent-output.txt").read_text()


def test_final_assistant_text_reads_projected_message() -> None:
    response = {
        "data": {
            "messages": [
                {"role": "user", "parts": [{"type": "text", "text": "prompt"}]},
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "text": "final answer"}],
                },
            ]
        }
    }

    with patch.object(runner, "_request", return_value=(200, response)):
        assert runner._final_assistant_text("sess_1") == "final answer"


def test_final_assistant_text_prefers_native_turn_result() -> None:
    session = {
        "latestRun": {
            "status": "COMPLETED",
            "result": {"output": "native final answer"},
        }
    }

    with patch.object(runner, "_request") as request:
        assert runner._final_assistant_text("sess_1", session) == "native final answer"

    request.assert_not_called()


def test_main_runs_session_writes_output_and_cleans_up(bridge_paths: Path) -> None:
    session_state = {
        "latestRun": {"status": "COMPLETED", "result": {"output": '{"ok":true}'}},
        "totalInputTokens": 100,
        "totalOutputTokens": 25,
        "modelTokenUsage": 125,
    }
    cleanup = {"sessionDeleted": True, "leasePurged": True}

    with (
        patch.object(
            runner,
            "_credentials",
            return_value={
                "accessKeyId": "ASIAABCDEFGHIJKLMNOP",
                "secretAccessKey": "s" * 40,
                "sessionToken": "token",
            },
        ),
        patch.object(runner, "_caller_account_id", return_value="123456789012"),
        patch.object(runner, "_create_lease", return_value="int_1"),
        patch.object(runner, "_create_session", return_value=("sess_1", "run_1")),
        patch.object(runner, "_wait_for_session", return_value=session_state),
        patch.object(runner, "_final_assistant_text", return_value='{"ok":true}'),
        patch.object(runner, "_cleanup", return_value=cleanup) as cleanup_call,
    ):
        assert runner.main() == 0

    state = json.loads((bridge_paths / "cloudgeni-state.json").read_text())
    assert state["status"] == "COMPLETED"
    assert state["cleanup"] == cleanup
    assert json.loads((bridge_paths / "agent-output.json").read_text()) == {"ok": True}
    cleanup_call.assert_called_once_with("sess_1", "int_1")


def test_main_records_safe_failure_and_still_cleans_up(bridge_paths: Path) -> None:
    with (
        patch.object(runner, "_credentials", side_effect=runner.BridgeError("bad profile")),
        patch.object(
            runner,
            "_cleanup",
            return_value={"sessionDeleted": True, "leasePurged": True},
        ) as cleanup,
    ):
        assert runner.main() == 1

    state = json.loads((bridge_paths / "cloudgeni-state.json").read_text())
    assert state["status"] == "BRIDGE_FAILED"
    assert state["error"] == "bad profile"
    assert "bad profile" in (bridge_paths / "agent-output.txt").read_text()
    cleanup.assert_called_once_with(None, None)
