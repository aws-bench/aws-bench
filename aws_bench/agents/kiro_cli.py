"""Kiro CLI agent for aws-bench.

Installs Kiro CLI into the trial environment, configures auth and MCP servers,
then runs it non-interactively against the task instruction. Auth uses
``$KIRO_API_KEY`` from the host shell.

Trajectory data is rebuilt after the run from Kiro CLI's own session store.
Kiro CLI has three agent engines (``--agent-engine v1|v2|v3``) with different
stores, and picks one per session unless pinned:

* v1 (legacy): ``conversations_v2`` table in ``~/.local/share/kiro-cli/data.sqlite3``
* v2: ``~/.kiro/sessions/cli/<session_id>.json`` + ``.jsonl``
* v3 (KAS): ``~/.kiro/sessions/<workspace>/sess_<id>/session.json`` + ``messages.jsonl``

Both stores are copied out; the ``~/.kiro/sessions`` tree is preferred and
SQLite is the fallback.
"""

from __future__ import annotations

import json
import os
import shlex
import sqlite3
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)

from aws_bench.cli.preflight import PreflightError

_OUTPUT_FILENAME = "kiro-cli.txt"
_DB_FILENAME = "kiro-cli-data.sqlite3"
_CONTAINER_DB_PATH = "~/.local/share/kiro-cli/data.sqlite3"
_SESSIONS_DIRNAME = "kiro-cli-sessions"
_DEFAULT_AGENT_ENGINE = "v3"
_CONTAINER_SESSIONS_DIR = "~/.kiro/sessions"
# kiro-cli installs into $HOME/.local/bin which is not on PATH for non-login
# shells spawned by environment.exec.
_PATH_PREFIX = 'export PATH="$HOME/.local/bin:$PATH"; '


class KiroCli(BaseInstalledAgent):
    """Kiro CLI agent — runs tasks in headless mode via kiro-cli chat."""

    SUPPORTS_ATIF: bool = True

    CLI_FLAGS = [
        CliFlag(
            "effort",
            cli="--effort",
            type="enum",
            choices=["low", "medium", "high", "xhigh", "max"],
            env_fallback="KIRO_CLI_EFFORT_LEVEL",
        ),
        # v1 = legacy (SQLite); v2 = sessions/cli files; v3 = KAS sessions/<ws>/sess_* files.
        # Pinned to v3 by default, run() warns when the default is applied.
        CliFlag(
            "agent_engine",
            cli="--agent-engine",
            type="enum",
            choices=["v1", "v2", "v3"],
            default=_DEFAULT_AGENT_ENGINE,
            env_fallback="KIRO_CLI_AGENT_ENGINE",
        ),
    ]

    @staticmethod
    def name() -> str:
        """Return the agent name identifier."""
        return "kiro-cli"

    def get_version_command(self) -> str | None:
        """Return the shell command to detect the installed kiro-cli version."""
        return f"{_PATH_PREFIX}kiro-cli --version"

    def parse_version(self, stdout: str) -> str:
        """Parse semver from kiro-cli --version output."""
        import re

        text = stdout.strip()
        match = re.search(r"(\d+\.\d+\.\d+)", text)
        return match.group(1) if match else text

    @staticmethod
    def _build_mcp_json(
        servers: list[MCPServerConfig],
    ) -> dict[str, dict[str, Any]] | None:
        """Build Kiro CLI MCP server config dict from Harbor's MCPServerConfig list."""
        if not servers:
            return None
        mcp_servers: dict[str, dict[str, Any]] = {}
        for server in servers:
            if server.transport == "stdio":
                entry: dict[str, Any] = {
                    "command": server.command,
                    "args": server.args,
                }
            else:
                entry = {"url": server.url}
            mcp_servers[server.name] = entry
        return mcp_servers

    def _kiro_env(self) -> dict[str, str]:
        """Collect Kiro CLI env vars from the host."""
        return {"KIRO_API_KEY": os.environ.get("KIRO_API_KEY", "")}

    async def setup(self, environment: BaseEnvironment) -> None:
        """Validate KIRO_API_KEY before spending time on install."""
        if not os.environ.get("KIRO_API_KEY", ""):
            raise PreflightError(
                "KIRO_API_KEY is not set (or is empty), but the agent is kiro-cli. "
                "Export it before running: export KIRO_API_KEY=ksk_xxxxxxxx"
            )
        await super().setup(environment)

    async def install(self, environment: BaseEnvironment) -> None:
        """Install kiro-cli binary in the agent environment."""
        await self.exec_as_root(
            environment,
            command=(
                "if command -v apt-get &> /dev/null; then"
                "  apt-get update && apt-get install -y curl unzip libasound2;"
                " elif command -v yum &> /dev/null; then"
                "  yum install -y curl unzip alsa-lib;"
                " elif command -v apk &> /dev/null; then"
                "  apk add --no-cache curl bash unzip alsa-lib;"
                " else"
                '  echo "Warning: no known package manager found" >&2;'
                " fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "ARCH=$(uname -m); "
                "case $ARCH in x86_64) KIRO_ARCH=x86_64;; aarch64|arm64) KIRO_ARCH=aarch64;; "
                '*) echo "Unsupported architecture: $ARCH"; exit 1;; esac; '
                "curl --proto '=https' --tlsv1.2 -sSf "
                '"https://desktop-release.q.us-east-1.amazonaws.com/latest/'
                'kirocli-${KIRO_ARCH}-linux.zip" '
                "-o /tmp/kirocli.zip && "
                "cd /tmp && unzip -q kirocli.zip && "
                "./kirocli/install.sh --force --no-confirm && "
                "rm -rf /tmp/kirocli.zip /tmp/kirocli && "
                f"{_PATH_PREFIX}"
                "kiro-cli --version && "
                "mkdir -p ~/.kiro/settings && "
                "kiro-cli settings chat.greeting.enabled false"
            ),
        )

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Execute a task using kiro-cli in headless mode."""
        env = self._kiro_env()

        # Write MCP server config if any servers are configured
        mcp_servers = self._build_mcp_json(self.mcp_servers)
        if mcp_servers:
            mcp_json_str = json.dumps({"mcpServers": mcp_servers}, indent=2)
            escaped_mcp = shlex.quote(mcp_json_str)
            await self.exec_as_agent(
                environment,
                command=(
                    f"mkdir -p ~/.kiro/settings && echo {escaped_mcp} > ~/.kiro/settings/mcp.json"
                ),
                env=env or None,
            )

        if "agent_engine" not in self._flag_kwargs and "KIRO_CLI_AGENT_ENGINE" not in os.environ:
            self.logger.warning(
                "kiro-cli agent engine not set; defaulting to --agent-engine %s. "
                "Override with --ak agent_engine=v1|v2|v3 or KIRO_CLI_AGENT_ENGINE.",
                _DEFAULT_AGENT_ENGINE,
            )

        escaped_instruction = shlex.quote(instruction)
        model_flag = f"--model {shlex.quote(self.model_name)} " if self.model_name else ""
        cli_flags = self.build_cli_flags()
        extra_flags = (cli_flags + " ") if cli_flags else ""

        # Copy skills into ~/.kiro/skills/. kiro-cli's default agent loads skills
        # from ~/.kiro/skills/ (global) at chat start, including in headless mode:
        # it lists each skill's name/description and reads the full SKILL.md when
        # a request matches. Verified with kiro-cli 2.21.2 that a skill present
        # only in ~/.kiro/skills/ is discovered and used.
        #
        # Write the resolved home and copied-skill count to /logs/agent/ (which is
        # collected with the run) rather than stdout: environment.exec discards
        # command stdout, so an echoed marker would never surface in the logs.
        # This makes a run that staged no skills easy to spot (global=0 means the
        # copy found nothing at skills_dir).
        if self.skills_dir:
            src = shlex.quote(self.skills_dir)
            await self.exec_as_agent(
                environment,
                command=(
                    f"mkdir -p ~/.kiro/skills /logs/agent && "
                    f"cp -r {src}/* ~/.kiro/skills/ 2>/dev/null || true; "
                    f'{{ echo "kiro-skills: HOME=$HOME src={self.skills_dir}"; '
                    f'echo "kiro-skills: global=$(ls ~/.kiro/skills 2>/dev/null | wc -l)"; }} '
                    f"| tee /logs/agent/kiro-skills.log"
                ),
                env=env or None,
            )

        run_command = (
            f"{_PATH_PREFIX}"
            f"kiro-cli chat --trust-all-tools --no-interactive "
            f"{model_flag}"
            f"{extra_flags}"
            f"{escaped_instruction} 2>&1 </dev/null | "
            f"tee /logs/agent/{_OUTPUT_FILENAME}"
        )
        try:
            await self.exec_as_agent(environment, command=run_command, env=env or None)
        finally:
            # Copy both session stores to logs so populate_context_post_run can read them
            try:
                await self.exec_as_agent(
                    environment,
                    command=(
                        f"cp -r {_CONTAINER_SESSIONS_DIR} /logs/agent/{_SESSIONS_DIRNAME} "
                        f"2>/dev/null; "
                        f"cp {_CONTAINER_DB_PATH} /logs/agent/{_DB_FILENAME} 2>/dev/null || true"
                    ),
                )
            except Exception:
                pass

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Build the ATIF trajectory from session files (v2/v3), falling back to SQLite (v1)."""
        trajectory = self._trajectory_from_sessions(self.logs_dir / _SESSIONS_DIRNAME)
        if trajectory is None:
            trajectory = self._trajectory_from_db(self.logs_dir / _DB_FILENAME)
        if trajectory is None:
            self.logger.warning(
                "No Kiro CLI session found in %s or %s; no trajectory written",
                _SESSIONS_DIRNAME,
                _DB_FILENAME,
            )
            return

        trajectory_path = self.logs_dir / "trajectory.json"
        try:
            with open(trajectory_path, "w", encoding="utf-8") as f:
                json.dump(trajectory.to_json_dict(), f, indent=2, ensure_ascii=False)
            self.logger.debug("Wrote Kiro CLI trajectory to %s", trajectory_path)
        except OSError as exc:
            self.logger.debug("Failed to write trajectory: %s", exc)
            return

        if trajectory.final_metrics:
            context.cost_usd = trajectory.final_metrics.total_cost_usd

    def _trajectory_from_sessions(self, sessions_dir: Path) -> Trajectory | None:
        """Convert the most recently updated session under the copied ``~/.kiro/sessions``.

        Handles both the v2 layout (``cli/<id>.json`` + ``cli/<id>.jsonl``) and the
        v3/KAS layout (``<workspace>/sess_<id>/session.json`` + ``messages.jsonl``).
        """
        if not sessions_dir.is_dir():
            return None
        # (last-updated ISO timestamp, metadata, transcript path, converter)
        candidates: list[tuple[str, dict[str, Any], Path, Any]] = []
        for meta_path in sessions_dir.glob("cli/*.json"):
            meta = self._read_json(meta_path)
            if meta is not None:
                candidates.append(
                    (
                        str(meta.get("updated_at", "")),
                        meta,
                        meta_path.with_suffix(".jsonl"),
                        self._convert_session_to_trajectory,
                    )
                )
        for meta_path in sessions_dir.glob("*/sess_*/session.json"):
            meta = self._read_json(meta_path)
            if meta is not None:
                candidates.append(
                    (
                        str(meta.get("lastModifiedAt", "")),
                        meta,
                        meta_path.with_name("messages.jsonl"),
                        self._convert_kas_session_to_trajectory,
                    )
                )
        if not candidates:
            return None
        _, meta, transcript_path, convert = max(candidates, key=lambda c: c[0])
        entries: list[dict[str, Any]] = []
        try:
            for line in transcript_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entries.append(json.loads(line))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.logger.debug("Failed to read Kiro CLI session transcript: %s", exc)
            return None
        try:
            return convert(meta, entries)
        except Exception as exc:
            self.logger.debug("Failed to convert Kiro CLI session to trajectory: %s", exc)
            return None

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.logger.debug("Skipping unreadable Kiro CLI session %s: %s", path, exc)
            return None
        return data if isinstance(data, dict) else None

    # ------------------------------------------------------------------
    # Store parsers: each turns one engine's session into a flat event list
    # (see _build_trajectory), so the ATIF assembly lives in one place.
    # ------------------------------------------------------------------

    def _convert_kas_session_to_trajectory(
        self, meta: dict[str, Any], events: list[dict[str, Any]]
    ) -> Trajectory | None:
        """v3/KAS: ``session.json`` + ``messages.jsonl`` events."""
        flat: list[dict[str, Any]] = []
        for e in events:
            ts = e.get("timestamp")
            p = e.get("payload") or {}
            kind = p.get("type")
            if kind == "user":
                flat.append(_user(str(p.get("content", "")), ts))
            elif kind == "assistant" and p.get("operationType") == "Say":
                flat.append(_agent(str(p.get("content", "")), ts))
            elif kind == "tool_call":
                call = (p.get("toolCallId", ""), p.get("toolName", "unknown"), p.get("args") or {})
                flat.append(_agent("", ts, [call]))
            elif kind == "tool_result":
                flat.append(
                    _result(
                        p.get("toolCallId", ""),
                        str(p.get("content", "")),
                        p.get("success") is not False,
                    )
                )
            elif kind == "usage_summary":
                flat.append(
                    _usage(
                        sum(float(t.get("usage", 0)) for t in p.get("promptTurnSummaries") or [])
                    )
                )
        model_id = meta.get("modelId") or self.model_name
        return self._build_trajectory(meta.get("id", "unknown"), model_id, flat)

    def _convert_session_to_trajectory(
        self, meta: dict[str, Any], entries: list[dict[str, Any]]
    ) -> Trajectory | None:
        """v2: ``<id>.json`` metadata + ``<id>.jsonl`` Prompt/AssistantMessage/ToolResults lines."""
        flat: list[dict[str, Any]] = []
        for entry in entries:
            kind = entry.get("kind")
            data = entry.get("data") or {}
            content = data.get("content") or []
            text = "".join(c.get("data", "") for c in content if c.get("kind") == "text")
            if kind == "Prompt":
                ts = (data.get("meta") or {}).get("timestamp")
                flat.append(_user(text, self._ms_to_iso(ts * 1000) if ts else None))
            elif kind == "AssistantMessage":
                calls = [
                    (
                        c["data"].get("toolUseId", ""),
                        c["data"].get("name", "unknown"),
                        c["data"].get("input", {}),
                    )
                    for c in content
                    if c.get("kind") == "toolUse" and isinstance(c.get("data"), dict)
                ]
                flat.append(_agent(text, None, calls))
            elif kind == "ToolResults":
                for part in content:
                    if part.get("kind") != "toolResult":
                        continue
                    tr = part.get("data") or {}
                    flat.append(
                        _result(
                            tr.get("toolUseId", ""),
                            _join_parts(tr.get("content") or []),
                            str(tr.get("status", "success")).lower() == "success",
                        )
                    )
        state = meta.get("session_state") or {}
        turns = (state.get("conversation_metadata") or {}).get("user_turn_metadatas") or []
        credits = [u.get("value", 0) for t in turns for u in t.get("metering_usage") or []]
        flat.append(
            _usage(
                sum(credits) if credits else None,
                sum(t.get("input_token_count") or 0 for t in turns) or None,
                sum(t.get("output_token_count") or 0 for t in turns) or None,
            )
        )
        model_id = (
            ((state.get("rts_model_state") or {}).get("model_info") or {}).get("model_id")
            or next((t.get("model") for t in turns if t.get("model")), None)
            or self.model_name
        )
        return self._build_trajectory(meta.get("session_id", "unknown"), model_id, flat)

    def _trajectory_from_db(self, db_path: Path) -> Trajectory | None:
        """Legacy V1 engine: convert the newest ``conversations_v2`` row."""
        if not db_path.exists():
            return None
        conversation = self._extract_conversation(db_path)
        if not conversation:
            self.logger.debug("No conversation data found in Kiro CLI database")
            return None
        try:
            return self._convert_conversation_to_trajectory(conversation)
        except Exception as exc:
            self.logger.debug("Failed to convert Kiro CLI conversation to trajectory: %s", exc)
            return None

    def _extract_conversation(self, db_path: Path) -> dict[str, Any] | None:
        """Read the most recent conversation from the kiro-cli SQLite database."""
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.execute(
                "SELECT value FROM conversations_v2 ORDER BY created_at DESC LIMIT 1;"
            )
            row = cursor.fetchone()
            conn.close()
        except (sqlite3.Error, OSError) as exc:
            self.logger.debug("SQLite read failed: %s", exc)
            return None

        if not row:
            return None

        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return None

    def _convert_conversation_to_trajectory(
        self, conversation: dict[str, Any]
    ) -> Trajectory | None:
        """v1: a ``conversations_v2`` row (``history`` turns + ``user_turn_metadata``)."""
        flat: list[dict[str, Any]] = []
        for turn in conversation.get("history") or []:
            user = turn.get("user") or {}
            content = user.get("content") or {}
            if "Prompt" in content:
                flat.append(_user(content["Prompt"].get("prompt", ""), user.get("timestamp")))
            if "ToolUseResults" in content:
                for tr in content["ToolUseResults"].get("tool_use_results", []):
                    flat.append(
                        _result(
                            tr.get("tool_use_id", ""),
                            _join_parts(tr.get("content", "")),
                            str(tr.get("status", "success")).lower() != "error",
                        )
                    )
            assistant = turn.get("assistant") or {}
            if not isinstance(assistant, dict):
                continue
            ts = self._ms_to_iso(
                (turn.get("request_metadata") or {}).get("stream_end_timestamp_ms")
            )
            if "ToolUse" in assistant:
                tu = assistant["ToolUse"]
                calls = [
                    (t.get("id", ""), t.get("name", "unknown"), t.get("args", {}))
                    for t in tu.get("tool_uses", [])
                ]
                flat.append(_agent(tu.get("content", ""), ts, calls))
            elif "Response" in assistant:
                flat.append(_agent(assistant["Response"].get("content", ""), ts))
        usage_info = (conversation.get("user_turn_metadata") or {}).get("usage_info") or []
        if usage_info:
            flat.append(_usage(sum(u.get("value", 0) for u in usage_info)))
        model_id = (conversation.get("model_info") or {}).get("model_id") or self.model_name
        return self._build_trajectory(
            conversation.get("conversation_id", "unknown"), model_id, flat
        )

    def _build_trajectory(
        self, session_id: str, model_id: str | None, events: list[dict[str, Any]]
    ) -> Trajectory | None:
        """Assemble an ATIF trajectory from the flat events produced by the store parsers."""
        results = {
            e["call_id"]: ObservationResult(source_call_id=e["call_id"], content=e["content"])
            for e in events
            if e["kind"] == "result"
        }
        steps: list[Step] = []
        n_calls = 0
        for e in events:
            if e["kind"] == "user":
                steps.append(
                    Step(
                        step_id=len(steps) + 1, source="user", message=e["text"], timestamp=e["ts"]
                    )
                )
            elif e["kind"] == "agent":
                calls = [
                    ToolCall(tool_call_id=cid, function_name=name, arguments=args)
                    for cid, name, args in e["calls"]
                ]
                n_calls += len(calls)
                observed = [results[c.tool_call_id] for c in calls if c.tool_call_id in results]
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        source="agent",
                        message=e["text"],
                        tool_calls=calls or None,
                        observation=Observation(results=observed) if observed else None,
                        timestamp=e["ts"],
                        model_name=model_id,
                    )
                )
        if not steps:
            return None

        usage = [e for e in events if e["kind"] == "usage"]
        credits = [u["credits"] for u in usage if u["credits"] is not None]
        metrics = FinalMetrics(
            total_steps=len(steps),
            total_prompt_tokens=sum(u["prompt_tokens"] or 0 for u in usage) or None,
            total_completion_tokens=sum(u["completion_tokens"] or 0 for u in usage) or None,
            total_cost_usd=sum(credits) if credits else None,
            extra={
                "total_tool_calls": n_calls,
                "total_tool_calls_errors": sum(
                    1 for e in events if e["kind"] == "result" and not e["ok"]
                ),
                "total_tool_calls_rejected": n_calls - len(results),
            },
        )
        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id,
            agent=Agent(name="kiro-cli", version=self._version or "unknown", model_name=model_id),
            steps=steps,
            final_metrics=metrics,
        )

    @staticmethod
    def _ms_to_iso(ms: int | None) -> str | None:
        """Convert millisecond epoch timestamp to ISO 8601 string."""
        if ms is None:
            return None
        from datetime import datetime, timezone

        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


# Flat event constructors shared by the three store parsers.
def _user(text: str, ts: str | None) -> dict[str, Any]:
    return {"kind": "user", "text": text, "ts": ts}


def _agent(
    text: str, ts: str | None, calls: list[tuple[str, str, Any]] | None = None
) -> dict[str, Any]:
    return {"kind": "agent", "text": text, "ts": ts, "calls": calls or []}


def _result(call_id: str, content: str, ok: bool) -> dict[str, Any]:
    return {"kind": "result", "call_id": call_id, "content": content, "ok": ok}


def _usage(
    credits: float | None, prompt_tokens: int | None = None, completion_tokens: int | None = None
) -> dict[str, Any]:
    return {
        "kind": "usage",
        "credits": credits,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def _join_parts(parts: Any) -> str:
    """Flatten a tool-result content list into text.

    v1 items look like ``{"Text": str}`` / ``{"Json": obj}``; v2 items like
    ``{"kind": "text"|"json", "data": ...}``. Text is kept verbatim, the rest is JSON.
    """
    if not isinstance(parts, list):
        return str(parts)
    out = []
    for item in parts:
        if not isinstance(item, dict):
            out.append(str(item))
        elif "Text" in item:
            out.append(str(item["Text"]))
        elif item.get("kind") == "text":
            out.append(str(item.get("data", "")))
        else:
            out.append(json.dumps(item.get("Json", item.get("data", item))))
    return " ".join(out)
