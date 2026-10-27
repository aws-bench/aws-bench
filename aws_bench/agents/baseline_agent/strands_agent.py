"""Harbor driver for the aws-bench baseline agent.

Installs and runs the Strands Agent SDK inside the trial container with
a baseline system prompt, bash/file tools, and MCP server connectivity.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig

from aws_bench.agents.baseline_agent.system_prompt import build_system_prompt

_OUTPUT_FILENAME = "strands-agent.txt"
_DEFAULT_AWS_REGION = "us-east-1"


class StrandsAgent(BaseInstalledAgent):
    """aws-bench baseline agent — Strands SDK with Bedrock and MCP tools."""

    SUPPORTS_ATIF: bool = True

    @staticmethod
    def name() -> str:
        """Return the agent name identifier."""
        return "aws-bench-baseline-agent"

    def get_version_command(self) -> str | None:
        """Return version detection command (not applicable)."""
        return None

    @staticmethod
    def _build_mcp_json(servers: list[MCPServerConfig]) -> dict[str, Any] | None:
        """Convert Harbor MCPServerConfig list to mcp.json format."""
        if not servers:
            return None
        mcp_servers: dict[str, dict[str, Any]] = {}
        for server in servers:
            if server.transport == "stdio":
                mcp_servers[server.name] = {
                    "command": server.command,
                    "args": server.args,
                }
            else:
                mcp_servers[server.name] = {
                    "url": server.url,
                    "transport": server.transport,
                }
        return {"mcpServers": mcp_servers}

    def _agent_env(self) -> dict[str, str]:
        """Collect environment variables to forward into the container."""
        env: dict[str, str] = {}
        region = os.environ.get("AWS_REGION", _DEFAULT_AWS_REGION)
        env["AWS_REGION"] = region
        env["AWS_DEFAULT_REGION"] = region

        if bearer := os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip():
            env["AWS_BEARER_TOKEN_BEDROCK"] = bearer

        if model := self.model_name:
            env["MODEL_NAME"] = model

        return env

    async def install(self, environment: BaseEnvironment) -> None:
        """Ensure uv is available in the container."""
        await self.exec_as_agent(
            environment,
            command=(
                "if ! command -v uv &> /dev/null; then "
                "pip install uv && "
                'export PATH="$HOME/.local/bin:$PATH"; fi'
            ),
        )

    async def setup(self, environment: BaseEnvironment) -> None:
        """Upload agent code, system prompt, and MCP config."""
        await super().setup(environment)

        # Upload main.py via base64 to handle arbitrary content safely
        main_py_path = Path(__file__).parent / "main.py"
        main_py_b64 = base64.b64encode(main_py_path.read_bytes()).decode()
        await self.exec_as_agent(
            environment,
            command=f"echo '{main_py_b64}' | base64 -d > /installed-agent/main.py",
        )

        # Upload system prompt
        prompt_text = build_system_prompt()
        escaped_prompt = shlex.quote(prompt_text)
        await self.exec_as_agent(
            environment,
            command=f"printf '%s' {escaped_prompt} > /installed-agent/system_prompt.txt",
        )

        # Upload MCP config if servers are configured
        mcp_config = self._build_mcp_json(self.mcp_servers)
        if mcp_config:
            mcp_json_str = json.dumps(mcp_config, indent=2)
            escaped_mcp = shlex.quote(mcp_json_str)
            await self.exec_as_agent(
                environment,
                command=f"printf '%s' {escaped_mcp} > /installed-agent/mcp.json",
            )

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Execute the task using the Strands agent."""
        env = self._agent_env()

        # Write the prompt to a file — avoids shell escaping issues with long prompts
        escaped_instruction = shlex.quote(instruction)
        await self.exec_as_agent(
            environment,
            command=f"printf '%s' {escaped_instruction} > /installed-agent/prompt.txt",
        )

        run_command = (
            'export PATH="$HOME/.local/bin:$PATH"; '
            "uv run --python 3.13 "
            "--with strands-agents --with strands-agents-tools "
            "--with boto3 --with botocore --with 'mcp<2' "
            f"/installed-agent/main.py "
            f"2>&1 | tee /logs/agent/{_OUTPUT_FILENAME}"
        )
        await self.exec_as_agent(environment, command=run_command, env=env or None)

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Validate trajectory was exported by the agent runtime."""
        trajectory_path = self.logs_dir / "trajectory.json"
        if not trajectory_path.exists():
            self.logger.debug("No trajectory file found at %s", trajectory_path)
            return
        self.logger.debug("Trajectory available at %s", trajectory_path)
