"""Tests for Kiro CLI ATIF trajectory support (SQLite extraction approach)."""

import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from harbor.models.trajectories import FinalMetrics, ObservationResult, Step, ToolCall, Trajectory

from aws_bench.agents.kiro_cli import _DB_FILENAME, _SESSIONS_DIRNAME, KiroCli


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "logs"
    d.mkdir()
    return d


@pytest.fixture
def agent(logs_dir: Path) -> KiroCli:
    return KiroCli(logs_dir=logs_dir)


def _create_db(logs_dir: Path, conversation: dict) -> Path:
    """Helper: create a SQLite DB with a conversation_v2 row."""
    db_path = logs_dir / _DB_FILENAME
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE conversations_v2 (key TEXT, value TEXT, created_at TEXT)")
    conn.execute(
        "INSERT INTO conversations_v2 (key, value, created_at) VALUES (?, ?, ?)",
        ("/app", json.dumps(conversation), "2025-01-01T00:00:00Z"),
    )
    conn.commit()
    conn.close()
    return db_path


class TestSupportsAtif:
    def test_supports_atif_flag(self):
        assert KiroCli.SUPPORTS_ATIF is True


class TestExtractConversation:
    def test_missing_db_returns_none(self, agent: KiroCli, logs_dir: Path):
        result = agent._extract_conversation(logs_dir / "nonexistent.db")
        assert result is None

    def test_empty_db_returns_none(self, agent: KiroCli, logs_dir: Path):
        db_path = logs_dir / _DB_FILENAME
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE conversations_v2 (key TEXT, value TEXT, created_at TEXT)")
        conn.commit()
        conn.close()
        result = agent._extract_conversation(db_path)
        assert result is None

    def test_extracts_conversation(self, agent: KiroCli, logs_dir: Path):
        conversation = {"history": [{"user": {"content": {"Prompt": {"prompt": "hi"}}}}]}
        db_path = _create_db(logs_dir, conversation)
        result = agent._extract_conversation(db_path)
        assert result == conversation

    def test_returns_most_recent_conversation(self, agent: KiroCli, logs_dir: Path):
        db_path = logs_dir / _DB_FILENAME
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE conversations_v2 (key TEXT, value TEXT, created_at TEXT)")
        conn.execute(
            "INSERT INTO conversations_v2 (key, value, created_at) VALUES (?, ?, ?)",
            ("/app", json.dumps({"history": [], "old": True}), "2025-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO conversations_v2 (key, value, created_at) VALUES (?, ?, ?)",
            ("/app", json.dumps({"history": [], "latest": True}), "2025-01-02T00:00:00Z"),
        )
        conn.commit()
        conn.close()
        result = agent._extract_conversation(db_path)
        assert result is not None
        assert result.get("latest") is True


class TestConvertConversationToTrajectory:
    def test_empty_history_returns_none(self, agent: KiroCli):
        result = agent._convert_conversation_to_trajectory({"history": []})
        assert result is None

    def test_user_prompt(self, agent: KiroCli):
        conversation = {
            "history": [
                {
                    "user": {"content": {"Prompt": {"prompt": "Create a file"}}},
                    "assistant": {"Response": {"content": "Done"}},
                }
            ],
        }
        traj = agent._convert_conversation_to_trajectory(conversation)
        assert traj is not None
        # user prompt + assistant response = 2 steps
        user_steps = [s for s in traj.steps if s.source == "user"]
        assert len(user_steps) == 1
        assert user_steps[0].message == "Create a file"

    def test_tool_use_step(self, agent: KiroCli):
        conversation = {
            "history": [
                {
                    "user": {"content": {"Prompt": {"prompt": "Write hello.txt"}}},
                    "assistant": {
                        "ToolUse": {
                            "content": "I'll write the file",
                            "tool_uses": [
                                {"id": "tu_1", "name": "fs_write", "args": {"path": "hello.txt"}}
                            ],
                        }
                    },
                },
                {
                    "user": {
                        "content": {
                            "ToolUseResults": {
                                "tool_use_results": [
                                    {
                                        "tool_use_id": "tu_1",
                                        "content": [{"Text": "File written"}],
                                        "status": "success",
                                    }
                                ]
                            }
                        }
                    },
                    "assistant": {"Response": {"content": "Done!"}},
                },
            ],
        }
        traj = agent._convert_conversation_to_trajectory(conversation)
        assert traj is not None
        agent_steps = [s for s in traj.steps if s.source == "agent"]
        # First agent step should have tool_calls and observation
        tool_step = agent_steps[0]
        assert tool_step.tool_calls is not None
        assert tool_step.tool_calls[0].function_name == "fs_write"
        assert tool_step.observation is not None
        assert tool_step.observation.results[0].content == "File written"

    def test_full_sequence_step_ids_are_sequential(self, agent: KiroCli):
        conversation = {
            "conversation_id": "conv-123",
            "history": [
                {
                    "user": {"content": {"Prompt": {"prompt": "hi"}}},
                    "assistant": {"Response": {"content": "hello"}},
                }
            ],
        }
        traj = agent._convert_conversation_to_trajectory(conversation)
        assert traj is not None
        for i, step in enumerate(traj.steps):
            assert step.step_id == i + 1

    def test_session_id_from_conversation(self, agent: KiroCli):
        conversation = {
            "conversation_id": "conv-abc",
            "history": [
                {
                    "user": {"content": {"Prompt": {"prompt": "hi"}}},
                    "assistant": {"Response": {"content": "hey"}},
                }
            ],
        }
        traj = agent._convert_conversation_to_trajectory(conversation)
        assert traj is not None
        assert traj.session_id == "conv-abc"

    def test_credits_from_usage_info(self, agent: KiroCli):
        conversation = {
            "history": [
                {
                    "user": {"content": {"Prompt": {"prompt": "hi"}}},
                    "assistant": {"Response": {"content": "hey"}},
                }
            ],
            "user_turn_metadata": {"usage_info": [{"value": 0.5}, {"value": 1.2}]},
        }
        traj = agent._convert_conversation_to_trajectory(conversation)
        assert traj is not None
        assert traj.final_metrics is not None
        assert traj.final_metrics.total_cost_usd == pytest.approx(1.7)

    def test_none_request_metadata_does_not_crash(self, agent: KiroCli):
        """Regression: request_metadata=None should not raise AttributeError."""
        conversation = {
            "history": [
                {
                    "user": {"content": {"Prompt": {"prompt": "hi"}}},
                    "assistant": {
                        "ToolUse": {"tool_uses": [{"id": "t1", "name": "x", "args": {}}]}
                    },
                    "request_metadata": {"stream_end_timestamp_ms": 1000},
                },
                {
                    "user": {
                        "content": {
                            "ToolUseResults": {
                                "tool_use_results": [
                                    {
                                        "tool_use_id": "t1",
                                        "content": "ok",
                                        "status": "success",
                                    }
                                ]
                            }
                        }
                    },
                    "assistant": {"Response": {"content": "done"}},
                    "request_metadata": None,
                },
            ],
        }
        traj = agent._convert_conversation_to_trajectory(conversation)
        assert traj is not None
        assert len(traj.steps) >= 2

    def test_none_user_and_assistant_fields_do_not_crash(self, agent: KiroCli):
        """Regression: user=None or assistant=None in a turn should not crash."""
        conversation = {
            "history": [
                {
                    "user": {"content": {"Prompt": {"prompt": "hi"}}},
                    "assistant": None,
                    "request_metadata": None,
                },
                {
                    "user": None,
                    "assistant": {"Response": {"content": "hello"}},
                    "request_metadata": {"stream_end_timestamp_ms": 2000},
                },
            ],
        }
        traj = agent._convert_conversation_to_trajectory(conversation)
        assert traj is not None

    def test_tool_call_counts(self, agent: KiroCli):
        conversation = {
            "history": [
                {
                    "user": {"content": {"Prompt": {"prompt": "do it"}}},
                    "assistant": {
                        "ToolUse": {
                            "tool_uses": [
                                {"id": "tu_1", "name": "read", "args": {}},
                                {"id": "tu_2", "name": "write", "args": {}},
                            ]
                        }
                    },
                },
                {
                    "user": {
                        "content": {
                            "ToolUseResults": {
                                "tool_use_results": [
                                    {"tool_use_id": "tu_1", "content": "ok", "status": "success"},
                                ]
                            }
                        }
                    },
                    "assistant": {"Response": {"content": "done"}},
                },
            ],
        }
        traj = agent._convert_conversation_to_trajectory(conversation)
        assert traj is not None
        assert traj.final_metrics is not None
        # 2 tool calls total, 1 rejected (tu_2 not answered)
        assert traj.final_metrics.extra is not None
        assert traj.final_metrics.extra["total_tool_calls"] == 2
        assert traj.final_metrics.extra["total_tool_calls_rejected"] == 1


class TestPopulateContextPostRun:
    def test_no_db_file(self, agent: KiroCli, logs_dir: Path):
        context = MagicMock()
        agent.populate_context_post_run(context)
        assert not (logs_dir / "trajectory.json").exists()

    def test_writes_trajectory_json(self, agent: KiroCli, logs_dir: Path):
        conversation = {
            "history": [
                {
                    "user": {"content": {"Prompt": {"prompt": "hi"}}},
                    "assistant": {"Response": {"content": "hello"}},
                }
            ],
        }
        _create_db(logs_dir, conversation)
        context = MagicMock()
        agent.populate_context_post_run(context)
        traj_path = logs_dir / "trajectory.json"
        assert traj_path.exists()
        data = json.loads(traj_path.read_text())
        assert data["schema_version"] == "ATIF-v1.7"
        assert len(data["steps"]) == 2  # user prompt + assistant response

    def test_sets_cost_from_credits(self, agent: KiroCli, logs_dir: Path):
        conversation = {
            "history": [
                {
                    "user": {"content": {"Prompt": {"prompt": "hi"}}},
                    "assistant": {"Response": {"content": "hey"}},
                }
            ],
            "user_turn_metadata": {"usage_info": [{"value": 2.5}]},
        }
        _create_db(logs_dir, conversation)
        context = MagicMock()
        agent.populate_context_post_run(context)
        assert context.cost_usd == pytest.approx(2.5)


class TestRunCopiesDb:
    @pytest.mark.asyncio
    async def test_run_copies_db_in_finally(self, agent: KiroCli):
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        context = MagicMock()

        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test"}, clear=True):
            await agent.run("Do the task", environment, context)

        calls = environment.exec.call_args_list
        # Last call should be the DB copy
        last_cmd = calls[-1].kwargs.get("command", "")
        assert "data.sqlite3" in last_cmd
        assert f"/logs/agent/{_DB_FILENAME}" in last_cmd

    @pytest.mark.asyncio
    async def test_run_no_agent_flag(self, agent: KiroCli):
        """Verify --agent flag is NOT passed (hook code removed)."""
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        context = MagicMock()

        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test"}, clear=True):
            await agent.run("Do the task", environment, context)

        calls = environment.exec.call_args_list
        commands = [c.kwargs.get("command", "") for c in calls]
        # The main run command should NOT have --agent
        run_cmds = [c for c in commands if "kiro-cli chat" in c]
        assert run_cmds
        assert "--agent " not in run_cmds[0]


# --- typed accessors: narrow the Optional ATIF fields for pyright ---


def _calls(step: Step) -> list[ToolCall]:
    assert step.tool_calls is not None
    return step.tool_calls


def _obs(step: Step) -> ObservationResult:
    assert step.observation is not None
    return step.observation.results[0]


def _obs_text(step: Step) -> str:
    content = _obs(step).content
    assert isinstance(content, str)
    return content


def _metrics(traj: Trajectory | None) -> FinalMetrics:
    assert traj is not None and traj.final_metrics is not None
    return traj.final_metrics


def _extra(traj: Trajectory | None) -> dict[str, Any]:
    extra = _metrics(traj).extra
    assert extra is not None
    return extra


# --- V3 engine: ~/.kiro/sessions/cli/<id>.json + .jsonl (primary source) ---

_SESSION_ID = "373dafde-0d5f-4dab-9224-fc3b580e753e"
_TOOL_USE_ID = "toolu_bdrk_01C5N6BXVgamvWtyczhy1f3t"


def _session_meta(session_id: str = _SESSION_ID, updated_at: str = "2026-09-23T15:08:00Z") -> dict:
    return {
        "session_id": session_id,
        "cwd": "/app",
        "created_at": "2026-09-23T15:07:24Z",
        "updated_at": updated_at,
        "session_state": {
            "version": "v1",
            "conversation_metadata": {
                "user_turn_metadatas": [
                    {
                        "model": "claude-sonnet-4-6",
                        "input_token_count": 10,
                        "output_token_count": 3,
                        "metering_usage": [
                            {"value": 1.25, "unit": "credit"},
                            {"value": 0.75, "unit": "credit"},
                        ],
                    }
                ]
            },
            "rts_model_state": {"model_info": {"model_id": "claude-sonnet-4-6"}},
        },
    }


def _session_entries() -> list[dict]:
    return [
        {
            "version": "v1",
            "kind": "Prompt",
            "data": {
                "content": [{"kind": "text", "data": "Run sleep 25 then reply pong"}],
                "meta": {"timestamp": 1790176047},
            },
        },
        {
            "version": "v1",
            "kind": "AssistantMessage",
            "data": {
                "content": [
                    {"kind": "text", "data": ""},
                    {
                        "kind": "toolUse",
                        "data": {
                            "toolUseId": _TOOL_USE_ID,
                            "name": "shell",
                            "input": {"command": "sleep 25"},
                        },
                    },
                ]
            },
        },
        {
            "version": "v1",
            "kind": "ToolResults",
            "data": {
                "content": [
                    {
                        "kind": "toolResult",
                        "data": {
                            "toolUseId": _TOOL_USE_ID,
                            "content": [
                                {"kind": "json", "data": {"exit_status": "exit status: 0"}}
                            ],
                            "status": "success",
                        },
                    }
                ]
            },
        },
        {
            "version": "v1",
            "kind": "AssistantMessage",
            "data": {"content": [{"kind": "text", "data": "pong"}]},
        },
    ]


def _write_session(logs_dir: Path, meta: dict, entries: list[dict]) -> Path:
    sessions = logs_dir / _SESSIONS_DIRNAME / "cli"
    sessions.mkdir(parents=True, exist_ok=True)
    (sessions / f"{meta['session_id']}.json").write_text(json.dumps(meta))
    (sessions / f"{meta['session_id']}.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in entries)
    )
    return sessions


class TestConvertSessionToTrajectory:
    def test_steps_and_tool_observation(self, agent: KiroCli):
        traj = agent._convert_session_to_trajectory(_session_meta(), _session_entries())
        assert traj is not None
        assert traj.session_id == _SESSION_ID
        assert [s.source for s in traj.steps] == ["user", "agent", "agent"]
        assert traj.steps[0].message == "Run sleep 25 then reply pong"
        assert traj.steps[0].timestamp is not None
        tool_step = traj.steps[1]
        assert _calls(tool_step)[0].tool_call_id == _TOOL_USE_ID
        assert _calls(tool_step)[0].function_name == "shell"
        assert _calls(tool_step)[0].arguments == {"command": "sleep 25"}
        assert _obs(tool_step).source_call_id == _TOOL_USE_ID
        assert "exit status: 0" in _obs_text(tool_step)
        assert traj.steps[2].message == "pong"
        assert traj.steps[2].model_name == "claude-sonnet-4-6"

    def test_metrics_from_turn_metadata(self, agent: KiroCli):
        traj = agent._convert_session_to_trajectory(_session_meta(), _session_entries())
        assert traj is not None
        m = _metrics(traj)
        assert m.total_cost_usd == pytest.approx(2.0)
        assert m.total_prompt_tokens == 10
        assert m.total_completion_tokens == 3
        assert _extra(traj)["total_tool_calls"] == 1
        assert _extra(traj)["total_tool_calls_errors"] == 0

    def test_empty_entries_returns_none(self, agent: KiroCli):
        assert agent._convert_session_to_trajectory(_session_meta(), []) is None


class TestSessionsDirSource:
    def test_picks_most_recently_updated_session(self, agent: KiroCli, logs_dir: Path):
        _write_session(logs_dir, _session_meta("old", "2026-09-23T10:00:00Z"), _session_entries())
        _write_session(logs_dir, _session_meta("new", "2026-09-23T11:00:00Z"), _session_entries())
        traj = agent._trajectory_from_sessions(logs_dir / _SESSIONS_DIRNAME)
        assert traj is not None
        assert traj.session_id == "new"

    def test_missing_dir_returns_none(self, agent: KiroCli, logs_dir: Path):
        assert agent._trajectory_from_sessions(logs_dir / _SESSIONS_DIRNAME) is None


class TestPopulateContextPrefersSessions:
    def test_session_files_win_over_sqlite(self, agent: KiroCli, logs_dir: Path):
        _create_db(
            logs_dir,
            {
                "conversation_id": "from-sqlite",
                "history": [
                    {
                        "user": {"content": {"Prompt": {"prompt": "hi"}}},
                        "assistant": {"Response": {"content": "hello"}},
                    }
                ],
            },
        )
        _write_session(logs_dir, _session_meta(), _session_entries())
        context = MagicMock()
        agent.populate_context_post_run(context)
        data = json.loads((logs_dir / "trajectory.json").read_text())
        assert data["session_id"] == _SESSION_ID
        assert context.cost_usd == pytest.approx(2.0)

    def test_empty_sessions_dir_falls_back_to_sqlite(self, agent: KiroCli, logs_dir: Path):
        (logs_dir / _SESSIONS_DIRNAME / "cli").mkdir(parents=True)
        _create_db(
            logs_dir,
            {
                "conversation_id": "from-sqlite",
                "history": [
                    {
                        "user": {"content": {"Prompt": {"prompt": "hi"}}},
                        "assistant": {"Response": {"content": "hello"}},
                    }
                ],
            },
        )
        agent.populate_context_post_run(MagicMock())
        data = json.loads((logs_dir / "trajectory.json").read_text())
        assert data["session_id"] == "from-sqlite"


class TestRunCopiesSessions:
    @pytest.mark.asyncio
    async def test_run_copies_sessions_dir_in_finally(self, agent: KiroCli):
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test"}, clear=True):
            await agent.run("Do the task", environment, MagicMock())
        last_cmd = environment.exec.call_args_list[-1].kwargs.get("command", "")
        assert ".kiro/sessions " in last_cmd
        assert f"/logs/agent/{_SESSIONS_DIRNAME}" in last_cmd


# --- V3 (KAS) engine: ~/.kiro/sessions/<workspace>/sess_<id>/{session.json,messages.jsonl} ---

_KAS_ID = "sess_be1a3667-798d-4ee7-80af-0546ba8c6339"
_KAS_TOOL_ID = "run_command_toolu_bdrk_01Hn46f6McTKjurQuHPbGFhg"


def _kas_meta(session_id: str = _KAS_ID, last_modified: str = "2026-09-23T16:03:03.100Z") -> dict:
    return {
        "schemaVersion": "1.0.0",
        "id": session_id,
        "createdAt": "2026-09-23T16:02:55.602Z",
        "lastModifiedAt": last_modified,
        "modelId": "claude-sonnet-4-6",
        "status": "idle",
    }


def _kas_events() -> list[dict]:
    def ev(payload: dict, ts: str = "2026-09-23T16:02:58.992Z") -> dict:
        return {"id": "x", "timestamp": ts, "payload": payload}

    return [
        ev({"type": "user", "content": "Run echo hi then reply pong"}, "2026-09-23T16:02:55.692Z"),
        ev({"type": "turn_start", "executionId": "e1"}),
        ev({"type": "steering_inclusion", "documents": []}),
        ev({"type": "assistant", "content": "...", "operationType": "Reasoning"}),
        ev({"type": "pending_interaction", "toolCallId": _KAS_TOOL_ID}),
        ev({"type": "interaction_resolved", "toolCallId": _KAS_TOOL_ID}),
        ev(
            {
                "type": "tool_call",
                "toolCallId": _KAS_TOOL_ID,
                "toolName": "execute_bash",
                "args": {"command": "echo hi"},
                "status": "completed",
            }
        ),
        ev(
            {
                "type": "tool_result",
                "toolCallId": _KAS_TOOL_ID,
                "content": "Output:\nhi\n\nExit Code: 0",
                "success": True,
            }
        ),
        ev(
            {"type": "assistant", "content": "pong", "operationType": "Say"},
            "2026-09-23T16:03:03.073Z",
        ),
        ev(
            {
                "type": "usage_summary",
                "promptTurnSummaries": [{"unit": "credit", "usage": 0.35}],
                "elapsedTime": 7098,
                "status": "success",
            }
        ),
        ev({"type": "turn_end", "stopReason": "end_turn"}),
        ev({"type": "session_start", "agentType": "vibe", "content": "You are Kiro..."}),
    ]


def _write_kas_session(logs_dir: Path, meta: dict, events: list[dict]) -> Path:
    d = logs_dir / _SESSIONS_DIRNAME / "da5ade4fbec1ce65" / meta["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "session.json").write_text(json.dumps(meta))
    (d / "messages.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (d / "publish.cursor").write_text("0")
    return d


class TestConvertKasSessionToTrajectory:
    def test_steps_and_tool_observation(self, agent: KiroCli):
        traj = agent._convert_kas_session_to_trajectory(_kas_meta(), _kas_events())
        assert traj is not None
        assert traj.session_id == _KAS_ID
        assert [s.source for s in traj.steps] == ["user", "agent", "agent"]
        assert traj.steps[0].message == "Run echo hi then reply pong"
        assert traj.steps[0].timestamp == "2026-09-23T16:02:55.692Z"
        tool_step = traj.steps[1]
        assert _calls(tool_step)[0].tool_call_id == _KAS_TOOL_ID
        assert _calls(tool_step)[0].function_name == "execute_bash"
        assert _calls(tool_step)[0].arguments == {"command": "echo hi"}
        assert _obs(tool_step).source_call_id == _KAS_TOOL_ID
        assert "Exit Code: 0" in _obs_text(tool_step)
        assert traj.steps[2].message == "pong"
        assert traj.steps[2].model_name == "claude-sonnet-4-6"

    def test_metrics_from_usage_summary(self, agent: KiroCli):
        traj = agent._convert_kas_session_to_trajectory(_kas_meta(), _kas_events())
        assert traj is not None
        m = _metrics(traj)
        assert m.total_cost_usd == pytest.approx(0.35)
        assert _extra(traj)["total_tool_calls"] == 1
        assert _extra(traj)["total_tool_calls_errors"] == 0

    def test_failed_tool_result_counts_as_error(self, agent: KiroCli):
        events = _kas_events()
        next(e for e in events if e["payload"]["type"] == "tool_result")["payload"]["success"] = (
            False
        )
        traj = agent._convert_kas_session_to_trajectory(_kas_meta(), events)
        assert traj is not None
        assert _extra(traj)["total_tool_calls_errors"] == 1

    def test_empty_events_returns_none(self, agent: KiroCli):
        assert agent._convert_kas_session_to_trajectory(_kas_meta(), []) is None


class TestSessionsDirKasSource:
    def test_reads_kas_session(self, agent: KiroCli, logs_dir: Path):
        _write_kas_session(logs_dir, _kas_meta(), _kas_events())
        traj = agent._trajectory_from_sessions(logs_dir / _SESSIONS_DIRNAME)
        assert traj is not None
        assert traj.session_id == _KAS_ID

    def test_picks_newest_across_v2_and_v3_layouts(self, agent: KiroCli, logs_dir: Path):
        _write_session(
            logs_dir, _session_meta("v2-old", "2026-09-23T10:00:00Z"), _session_entries()
        )
        _write_kas_session(logs_dir, _kas_meta("sess_new", "2026-09-23T11:00:00Z"), _kas_events())
        traj = agent._trajectory_from_sessions(logs_dir / _SESSIONS_DIRNAME)
        assert traj is not None
        assert traj.session_id == "sess_new"

    def test_kas_session_wins_over_sqlite_in_post_run(self, agent: KiroCli, logs_dir: Path):
        _create_db(
            logs_dir,
            {
                "conversation_id": "from-sqlite",
                "history": [
                    {
                        "user": {"content": {"Prompt": {"prompt": "hi"}}},
                        "assistant": {"Response": {"content": "hello"}},
                    }
                ],
            },
        )
        _write_kas_session(logs_dir, _kas_meta(), _kas_events())
        context = MagicMock()
        agent.populate_context_post_run(context)
        data = json.loads((logs_dir / "trajectory.json").read_text())
        assert data["session_id"] == _KAS_ID
        assert context.cost_usd == pytest.approx(0.35)


class TestToolResultText:
    def test_v2_text_result_part_is_verbatim(self, agent: KiroCli):
        entries = _session_entries()
        tool_results = next(e for e in entries if e["kind"] == "ToolResults")
        tool_results["data"]["content"][0]["data"]["content"] = [
            {"kind": "text", "data": "hello world"}
        ]
        traj = agent._convert_session_to_trajectory(_session_meta(), entries)
        assert traj is not None
        assert _obs_text(traj.steps[1]) == "hello world"

    def test_v1_text_result_part_is_verbatim(self, agent: KiroCli):
        conversation = {
            "history": [
                {
                    "user": {"content": {"Prompt": {"prompt": "hi"}}},
                    "assistant": {
                        "ToolUse": {"content": "", "tool_uses": [{"id": "t1", "name": "x"}]}
                    },
                },
                {
                    "user": {
                        "content": {
                            "ToolUseResults": {
                                "tool_use_results": [
                                    {"tool_use_id": "t1", "content": [{"Text": "hello world"}]}
                                ]
                            }
                        }
                    },
                    "assistant": {"Response": {"content": "done"}},
                },
            ]
        }
        traj = agent._convert_conversation_to_trajectory(conversation)
        assert traj is not None
        assert _obs_text(traj.steps[1]) == "hello world"
