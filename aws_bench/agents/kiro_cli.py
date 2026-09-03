"""Kiro CLI agent for aws-bench.

Installs Kiro CLI into the trial environment, configures auth and MCP servers,
then runs it non-interactively against the task instruction. Auth uses
``$KIRO_API_KEY`` from the host shell.

Trajectory data is extracted from Kiro CLI's native SQLite database after the
run completes.
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

        escaped_instruction = shlex.quote(instruction)
        model_flag = f"--model {shlex.quote(self.model_name)} " if self.model_name else ""
        cli_flags = self.build_cli_flags()
        extra_flags = (cli_flags + " ") if cli_flags else ""

        # Stage skills where kiro-cli's default agent looks. The default agent
        # scans both `.kiro/skills/*/SKILL.md` (workspace, relative to the run's
        # cwd) and `~/.kiro/skills/*/SKILL.md` (global) and surfaces each skill's
        # name/description in context (headless included), reading a full
        # SKILL.md on demand. Copying only to ~/.kiro/skills has left the dir
        # empty at chat time when the copy step's $HOME/cwd differed from the
        # chat process, so we stage to BOTH locations. The echo/ls line records
        # the resolved paths and file counts in the run log for debugging.
        if self.skills_dir:
            src = shlex.quote(self.skills_dir)
            await self.exec_as_agent(
                environment,
                command=(
                    f'echo "kiro-skills: HOME=$HOME PWD=$PWD src={self.skills_dir}"; '
                    f"mkdir -p ~/.kiro/skills .kiro/skills && "
                    f"cp -r {src}/* ~/.kiro/skills/ 2>/dev/null || true; "
                    f"cp -r {src}/* .kiro/skills/ 2>/dev/null || true; "
                    f'echo "kiro-skills: global=$(ls ~/.kiro/skills 2>/dev/null | wc -l) '
                    f'workspace=$(ls .kiro/skills 2>/dev/null | wc -l)"'
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
            # Copy the SQLite DB to logs so populate_context_post_run can read it
            try:
                await self.exec_as_agent(
                    environment,
                    command=(
                        f"cp {_CONTAINER_DB_PATH} /logs/agent/{_DB_FILENAME} 2>/dev/null || true"
                    ),
                )
            except Exception:
                pass

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Extract conversation from SQLite DB and build ATIF trajectory."""
        db_path = self.logs_dir / _DB_FILENAME
        if not db_path.exists():
            self.logger.debug("No Kiro CLI database found at %s", db_path)
            return

        conversation = self._extract_conversation(db_path)
        if not conversation:
            self.logger.debug("No conversation data found in Kiro CLI database")
            return

        try:
            trajectory = self._convert_conversation_to_trajectory(conversation)
        except Exception as exc:
            self.logger.debug("Failed to convert Kiro CLI conversation to trajectory: %s", exc)
            return
        if not trajectory:
            self.logger.debug("Failed to convert conversation to trajectory")
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
        """Convert a kiro-cli conversation dict to an ATIF Trajectory."""
        history = conversation.get("history", [])
        if not history:
            return None

        # Build set of answered tool_use_ids
        answered_ids: set[str] = set()
        for turn in history:
            user_content = (turn.get("user") or {}).get("content") or {}
            if "ToolUseResults" in user_content:
                for tr in user_content["ToolUseResults"].get("tool_use_results", []):
                    answered_ids.add(tr.get("tool_use_id", ""))

        steps: list[Step] = []
        model_id = (conversation.get("model_info") or {}).get("model_id") or self.model_name

        for turn in history:
            user_content = (turn.get("user") or {}).get("content") or {}
            assistant = turn.get("assistant") or {}
            meta = turn.get("request_metadata") or {}

            # Timestamps: user prompt has user.timestamp, agent response uses
            # request_start for the LLM call and stream_end for completion.
            user_ts = (turn.get("user") or {}).get("timestamp")
            agent_ts = self._ms_to_iso(meta.get("stream_end_timestamp_ms"))

            # User prompt
            if "Prompt" in user_content:
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        source="user",
                        message=user_content["Prompt"].get("prompt", ""),
                        timestamp=user_ts,
                    )
                )

            # Assistant response
            if isinstance(assistant, dict):
                if "ToolUse" in assistant:
                    tool_use_data = assistant["ToolUse"]
                    tool_calls = []
                    for tu in tool_use_data.get("tool_uses", []):
                        tool_calls.append(
                            ToolCall(
                                tool_call_id=tu.get("id", ""),
                                function_name=tu.get("name", "unknown"),
                                arguments=tu.get("args", {}),
                            )
                        )
                    # Build observation from matching tool results in next turn
                    observation = self._find_observation_for_tools(
                        tool_use_data.get("tool_uses", []), history, turn
                    )
                    msg = tool_use_data.get("content", "")
                    steps.append(
                        Step(
                            step_id=len(steps) + 1,
                            source="agent",
                            message=msg,
                            tool_calls=tool_calls if tool_calls else None,
                            observation=observation,
                            timestamp=agent_ts,
                            model_name=model_id,
                        )
                    )
                elif "Response" in assistant:
                    steps.append(
                        Step(
                            step_id=len(steps) + 1,
                            source="agent",
                            message=assistant["Response"].get("content", ""),
                            timestamp=agent_ts,
                            model_name=model_id,
                        )
                    )

        if not steps:
            return None

        # Extract native metrics (no token estimates)
        metrics = self._extract_native_metrics(conversation, history, answered_ids)

        model_info = conversation.get("model_info", {})
        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=conversation.get("conversation_id", "unknown"),
            agent=Agent(
                name="kiro-cli",
                version=self._version or "unknown",
                model_name=model_info.get("model_id") or self.model_name,
            ),
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

    @staticmethod
    def _find_observation_for_tools(
        tool_uses: list[dict[str, Any]],
        history: list[dict[str, Any]],
        current_turn: dict[str, Any],
    ) -> Observation | None:
        """Find tool results for the given tool_uses from subsequent turns."""
        # Look for tool results in the turn after the current one
        found_current = False
        for turn in history:
            if turn is current_turn:
                found_current = True
                continue
            if found_current:
                user_content = (turn.get("user") or {}).get("content") or {}
                if "ToolUseResults" in user_content:
                    results = []
                    for tr in user_content["ToolUseResults"].get("tool_use_results", []):
                        content_parts = tr.get("content", "")
                        if isinstance(content_parts, list):
                            text = " ".join(
                                (
                                    item.get("Text", json.dumps(item.get("Json", item)))
                                    if isinstance(item, dict)
                                    else str(item)
                                )
                                for item in content_parts
                            )
                        else:
                            text = str(content_parts)
                        results.append(
                            ObservationResult(
                                source_call_id=tr.get("tool_use_id", ""),
                                content=text,
                            )
                        )
                    if results:
                        return Observation(results=results)
                break
        return None

    @staticmethod
    def _extract_native_metrics(
        conversation: dict[str, Any],
        history: list[dict[str, Any]],
        answered_ids: set[str],
    ) -> FinalMetrics:
        """Extract only natively-provided metrics from the conversation."""
        n_tool_calls = 0
        n_tool_calls_errors = 0
        n_tool_calls_rejected = 0

        for turn in history:
            assistant = turn.get("assistant") or {}
            if isinstance(assistant, dict) and "ToolUse" in assistant:
                for tu in assistant["ToolUse"].get("tool_uses", []):
                    n_tool_calls += 1
                    if tu.get("id", "") not in answered_ids:
                        n_tool_calls_rejected += 1

            user_content = (turn.get("user") or {}).get("content") or {}
            if "ToolUseResults" in user_content:
                for tr in user_content["ToolUseResults"].get("tool_use_results", []):
                    if tr.get("status", "success").lower() == "error":
                        n_tool_calls_errors += 1

        # Credits from usage_info (native cost metric from kiro-cli)
        usage_info = (conversation.get("user_turn_metadata") or {}).get("usage_info", [])
        total_credits = sum(entry.get("value", 0) for entry in usage_info) if usage_info else None

        return FinalMetrics(
            total_steps=len(history),
            total_cost_usd=total_credits,
            extra={
                "total_tool_calls": n_tool_calls,
                "total_tool_calls_errors": n_tool_calls_errors,
                "total_tool_calls_rejected": n_tool_calls_rejected,
            },
        )
