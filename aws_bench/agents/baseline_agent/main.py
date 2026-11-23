#!/usr/bin/env python3
"""aws-bench Baseline Agent (Strands SDK) for Harbor Benchmark Framework."""

import json
import logging
import os
import re
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config as BotocoreConfig
from mcp import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent, tool
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient

os.makedirs("/logs/agent", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "us.anthropic.claude-sonnet-5-5"
MCP_CONFIG_PATH = Path("/installed-agent/mcp.json")
SYSTEM_PROMPT_PATH = Path("/installed-agent/system_prompt.txt")
PROMPT_PATH = Path("/installed-agent/prompt.txt")
TIMEOUT_SECS = 120
MAX_READ_SIZE_MB = 10.0

# --- Security rules for bash execution ---

DENIED_COMMANDS = [
    r"\brm\s+.*-[rf]",
    r"\brm\s+.*(/|~|\$HOME)",
    r"\bsudo\s+(rm|dd|mkfs|fdisk|mount|umount|iptables|systemctl|service)",
    r"\bsu\s+",
    r"\bchmod\s+.*777",
    r"\bchown\s+root",
    r"\bdd\s+.*of=/dev",
    r"\bmkfs\b",
    r"\bfdisk\b",
    r"\bmount\s+.*(/dev|/proc|/sys)",
    r"\biptables\s+.*-F",
    r"\bsystemctl\s+(stop|disable|mask)",
    r"\bkill\s+-9\s+1\b",
    r"\b(apt|yum)\s+remove\s+.*essential",
    r"\bpip\s+uninstall\s+.*pip",
    r"\|\s*(sh|bash)\b",
    r"\bcurl\s+.*\|\s*(sh|bash)",
    r"\bwget\s+.*\|\s*(sh|bash)",
    r"\bunset\s+PATH",
    r"\bexport\s+PATH=\s*$",
]


def _split_commands(command: str) -> list:
    """Split a multi-line command string into individual commands for security scanning."""
    commands = []
    for line in command.split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            for cmd in re.split(r"[;&]+|&&|\|\|", line):
                cmd = cmd.strip()
                if cmd:
                    commands.append(cmd)
    return commands


def _is_command_allowed(command: str) -> tuple:
    """Check whether a command passes security rules."""
    for cmd in _split_commands(command):
        for pattern in DENIED_COMMANDS:
            if re.search(pattern, cmd):
                return False, f"Command blocked by security rule: {pattern} (matched: {cmd})"
    return True, ""


# --- Built-in tools ---


@tool
def execute_bash(command: str, cwd: str = "") -> dict:
    """Execute bash commands with safety checks.

    Args:
        command: The bash command to execute.
        cwd: Optional working directory. Defaults to current directory.

    Returns:
        Dict with status, stdout, stderr, and exit_code.
    """
    allowed, reason = _is_command_allowed(command)
    if not allowed:
        return {
            "status": "ERROR",
            "stdout": "",
            "stderr": f"SECURITY VIOLATION: {reason}",
            "exit_code": 1,
        }

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd or os.getcwd(),
            timeout=TIMEOUT_SECS,
        )
        return {
            "status": "SUCCESS" if result.returncode == 0 else "ERROR",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "ERROR",
            "stdout": "",
            "stderr": f"Command timed out after {TIMEOUT_SECS} seconds",
            "exit_code": -1,
        }
    except Exception as e:
        return {"status": "ERROR", "stdout": "", "stderr": str(e), "exit_code": -1}


@tool
def write_file(file_path: str, content: str, overwrite: bool = False) -> dict:
    """Write content to a file under /logs/agent/.

    Args:
        file_path: Absolute file path (must be under /logs/agent/).
        content: Content to write.
        overwrite: Whether to overwrite an existing file.

    Returns:
        Dict with status and details.
    """
    path = Path(file_path).resolve()

    if not path.is_absolute():
        return {"status": "ERROR", "error": "Only absolute paths under /logs/agent/ are allowed."}

    if not str(path).startswith("/logs/agent/"):
        return {"status": "ERROR", "error": "Writes are only allowed under /logs/agent/."}

    if path.exists() and not overwrite:
        return {"status": "ERROR", "error": f"File exists and overwrite=False: {file_path}"}

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "status": "SUCCESS",
            "bytes_written": len(content.encode("utf-8")),
            "file_path": str(path),
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


@tool
def read_file(file_path: str) -> dict:
    """Read content from a file.

    Args:
        file_path: Path to the file to read.

    Returns:
        Dict with status and file content or error.
    """
    path = Path(file_path)

    if not path.exists():
        return {"status": "ERROR", "error": f"File does not exist: {file_path}"}

    if not path.is_file():
        return {"status": "ERROR", "error": f"Path is not a file: {file_path}"}

    file_size = path.stat().st_size
    max_bytes = int(MAX_READ_SIZE_MB * 1024 * 1024)
    if file_size > max_bytes:
        return {
            "status": "ERROR",
            "error": f"File too large: {file_size} bytes (max: {max_bytes} bytes)",
        }

    try:
        content = path.read_text(encoding="utf-8")
        return {"status": "SUCCESS", "content": content, "file_size": file_size}
    except UnicodeDecodeError as e:
        return {"status": "ERROR", "error": f"Encoding error: {e}"}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


# --- MCP config loading ---


def load_mcp_config() -> list:
    """Load MCP server configurations from mcp.json."""
    if not MCP_CONFIG_PATH.exists():
        logger.info("No MCP config found at %s", MCP_CONFIG_PATH)
        return []

    with open(MCP_CONFIG_PATH) as f:
        config = json.load(f)

    if "mcpServers" not in config:
        raise RuntimeError(f"Invalid MCP config — missing 'mcpServers' key in {MCP_CONFIG_PATH}")

    server_configs = []
    for name, cfg in config["mcpServers"].items():
        if cfg.get("disabled", False):
            logger.info("MCP server '%s' is disabled, skipping", name)
            continue
        server_configs.append(cfg)
        logger.info("Loaded MCP server config: %s", name)

    return server_configs


def create_mcp_clients(server_configs: list) -> list:
    """Create MCPClient instances for stdio, SSE, and HTTP transports."""
    mcp_clients = []

    for cfg in server_configs:
        try:
            if "url" in cfg:
                url = cfg["url"]
                headers = cfg.get("headers", {})
                transport = cfg.get("transport", "")
                if not transport:
                    path = urlparse(url).path
                    transport = "sse" if path.rstrip("/").endswith("/sse") else "streamable-http"
                if transport == "sse":
                    logger.info("Creating SSE client: %s", url)
                    client = sse_client(url, headers=headers)
                else:
                    logger.info("Creating HTTP client: %s", url)
                    client = streamablehttp_client(url, headers=headers)
                mcp_clients.append(MCPClient(lambda c=client: c))
            else:
                command = cfg.get("command")
                if not command:
                    logger.warning("No command for stdio server config, skipping")
                    continue

                args = cfg.get("args", [])
                env = cfg.get("env", {})
                server_env = os.environ.copy()
                server_env.update(env)

                logger.info("Creating stdio client: %s %s", command, args)
                params = StdioServerParameters(command=command, args=args, env=server_env)
                client = stdio_client(params)
                mcp_clients.append(MCPClient(lambda c=client: c))
        except Exception as e:
            raise RuntimeError(f"Failed to create MCP client: {e}") from e

    logger.info("Created %d MCP client(s)", len(mcp_clients))
    return mcp_clients


# --- Metrics ---


def emit_metrics(result, output_path="/logs/agent/metrics.json"):
    """Emit metrics to JSON file."""
    try:
        metrics_summary = result.metrics.get_summary()
        metrics_data = {
            "stop_reason": result.stop_reason,
            "execution_summary": {
                "total_cycles": metrics_summary.get("total_cycles", 0),
                "total_duration_seconds": metrics_summary.get("total_duration", 0),
                "average_cycle_time_seconds": metrics_summary.get("average_cycle_time", 0),
            },
            "token_usage": metrics_summary.get("accumulated_usage", {}),
            "latency": metrics_summary.get("accumulated_metrics", {}),
            "tool_usage": metrics_summary.get("tool_usage", {}),
        }
        with open(output_path, "w") as f:
            json.dump(metrics_data, f, indent=2, default=str)
        logger.info("Metrics written to %s", output_path)
    except Exception as e:
        logger.error("Failed to emit metrics: %s", e, exc_info=True)


# --- ATIF Trajectory Export ---


def emit_trajectory(agent_instance, result, model_id, output_path="/logs/agent/trajectory.json"):
    """Export the conversation history as an ATIF trajectory."""
    try:
        messages = agent_instance.messages
        if not messages:
            logger.warning("No messages to export as trajectory")
            return

        steps = []
        step_id = 1

        for msg in messages:
            role = msg.get("role", "")
            content_blocks = msg.get("content", [])

            if role == "user":
                # Check if this is a tool result message
                tool_results = [b for b in content_blocks if "toolResult" in b]
                if tool_results:
                    # Tool results are attached as observations on the preceding agent step
                    # They were already handled inline below
                    continue

                # Regular user message
                text_parts = [b.get("text", "") for b in content_blocks if "text" in b]
                message_text = "\n".join(text_parts)
                if message_text:
                    steps.append(
                        {
                            "step_id": step_id,
                            "source": "user",
                            "message": message_text,
                        }
                    )
                    step_id += 1

            elif role == "assistant":
                text_parts = []
                tool_calls = []

                for block in content_blocks:
                    if "text" in block:
                        text_parts.append(block["text"])
                    elif "toolUse" in block:
                        tu = block["toolUse"]
                        tool_calls.append(
                            {
                                "tool_call_id": tu.get("toolUseId", ""),
                                "function_name": tu.get("name", "unknown"),
                                "arguments": tu.get("input", {}),
                            }
                        )

                message_text = "\n".join(text_parts) if text_parts else ""

                # Look ahead for tool results in the next user message
                observation = _find_tool_results(messages, msg, tool_calls)

                step = {
                    "step_id": step_id,
                    "source": "agent",
                    "message": message_text,
                    "model_name": model_id,
                }
                if tool_calls:
                    step["tool_calls"] = tool_calls
                if observation:
                    step["observation"] = observation

                steps.append(step)
                step_id += 1

        if not steps:
            logger.warning("No steps generated for trajectory")
            return

        # Build final metrics from result
        metrics_summary = result.metrics.get_summary()
        accumulated_usage = metrics_summary.get("accumulated_usage", {})
        final_metrics = {
            "total_prompt_tokens": accumulated_usage.get("inputTokens"),
            "total_completion_tokens": accumulated_usage.get("outputTokens"),
            "total_steps": len(steps),
        }

        trajectory = {
            "schema_version": "ATIF-v1.7",
            "agent": {
                "name": "aws-bench-baseline-agent",
                "version": "1.0.0",
                "model_name": model_id,
            },
            "steps": steps,
            "final_metrics": final_metrics,
        }

        with open(output_path, "w") as f:
            json.dump(trajectory, f, indent=2, default=str)
        logger.info("Trajectory written to %s (%d steps)", output_path, len(steps))
    except Exception as e:
        logger.error("Failed to emit trajectory: %s", e, exc_info=True)


def _find_tool_results(messages, current_msg, tool_calls):
    """Find tool results that follow the current assistant message."""
    if not tool_calls:
        return None

    # Find the index of current message
    try:
        idx = messages.index(current_msg)
    except ValueError:
        return None

    # Look at the next message for tool results
    if idx + 1 >= len(messages):
        return None

    next_msg = messages[idx + 1]
    if next_msg.get("role") != "user":
        return None

    results = []
    for block in next_msg.get("content", []):
        if "toolResult" not in block:
            continue
        tr = block["toolResult"]
        content_parts = tr.get("content", [])
        result_text = "\n".join(p.get("text", "") for p in content_parts if "text" in p)
        results.append(
            {
                "source_call_id": tr.get("toolUseId", ""),
                "content": result_text,
            }
        )

    if results:
        return {"results": results}
    return None


# --- Main ---


def main():
    """Run the aws-bench baseline agent."""
    model_id = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()

    logger.info("Model: %s", model_id)
    logger.info("Prompt: %s", prompt[:200])

    # Load system prompt
    if SYSTEM_PROMPT_PATH.exists():
        system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
        logger.info("Loaded system prompt from %s", SYSTEM_PROMPT_PATH)
    else:
        system_prompt = ""
        logger.warning("No system prompt found at %s", SYSTEM_PROMPT_PATH)

    try:
        boto_config = BotocoreConfig(read_timeout=3600)
        # Clear any profile env vars — inside the container they may be set to
        # empty strings by the credential staging framework, which boto3 treats
        # as "look up profile named ''" and fails with ProfileNotFound.
        for var in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
            if not os.environ.get(var):
                os.environ.pop(var, None)

        # Use bearer token auth when available — allows LLM inference via a
        # management/Bedrock account independent of the child account profile.
        bearer_token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip()
        model = BedrockModel(
            model_id=model_id,
            boto_session=boto3.Session(),
            boto_client_config=boto_config,
        )
        if bearer_token:
            model.client._request_signer._auth_handler = None  # noqa: SLF001
            model.client.meta.events.register(
                "before-sign.bedrock-runtime.*",
                lambda request, **kwargs: request.headers.__setitem__(
                    "Authorization", f"Bearer {bearer_token}"
                ),
            )
            logger.info("Using bearer token authentication for Bedrock")
        else:
            logger.info("Using default credential chain for Bedrock")

        server_configs = load_mcp_config()
        mcp_clients = create_mcp_clients(server_configs)

        builtin_tools = [execute_bash, write_file, read_file]

        if not mcp_clients:
            logger.info("Running with built-in tools only")
            agent = Agent(model=model, system_prompt=system_prompt, tools=builtin_tools)
            response = agent(prompt)
        else:
            logger.info("Running with %d MCP client(s) + built-in tools", len(mcp_clients))
            with ExitStack() as stack:
                tools = list(builtin_tools)
                for mcp_client in mcp_clients:
                    stack.enter_context(mcp_client)
                    tools.extend(mcp_client.list_tools_sync())
                logger.info("Total tools loaded: %d", len(tools))
                agent = Agent(model=model, system_prompt=system_prompt, tools=tools)
                response = agent(prompt)

        emit_metrics(response)
        emit_trajectory(agent, response, model_id)
        print(response)

    except Exception as e:
        logger.error("Agent failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
