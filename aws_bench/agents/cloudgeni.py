"""Harbor installed-agent adapter for running AWS-Bench through CloudGeni."""

from __future__ import annotations

import asyncio
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from harbor.agents.installed.base import (
    BaseInstalledAgent,
    NonZeroAgentExitCodeError,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_RUNNER_PATH = Path(__file__).with_name("cloudgeni_runner.py")
_REMOTE_RUNNER_PATH = "/installed-agent/cloudgeni_runner.py"
_REMOTE_PROMPT_PATH = "/installed-agent/cloudgeni-prompt.txt"
_STATE_FILENAME = "cloudgeni-state.json"
_REQUIRED_ENV = (
    "CLOUDGENI_API_URL",
    "CLOUDGENI_API_KEY",
    "CLOUDGENI_ORGANIZATION_ID",
    "CLOUDGENI_WORKSPACE_ID",
)


class CloudGeniAgent(BaseInstalledAgent):
    """Delegate one isolated AWS-Bench trial to CloudGeni's OpenGeni runtime."""

    SUPPORTS_ATIF: bool = False

    @staticmethod
    def name() -> str:
        """Return the custom agent identifier."""
        return "cloudgeni"

    def get_version_command(self) -> str | None:
        """CloudGeni is a remote service, so there is no container binary version."""
        return None

    async def install(self, environment: BaseEnvironment) -> None:
        """Ensure the standard-library bridge has a Python interpreter."""
        result = await environment.exec(
            command=(
                "command -v python3 >/dev/null || "
                "(apt-get update -qq && apt-get install -y -qq python3)"
            ),
            user="root",
        )
        if result.return_code != 0:
            raise NonZeroAgentExitCodeError("Could not install Python for the CloudGeni bridge")

    async def setup(self, environment: BaseEnvironment) -> None:
        """Upload the bridge without placing CloudGeni credentials on a command line."""
        await self.install(environment)
        runner = base64.b64encode(_RUNNER_PATH.read_bytes()).decode()
        result = await environment.exec(
            command=(
                f"mkdir -p /installed-agent && echo '{runner}' | base64 -d > {_REMOTE_RUNNER_PATH}"
            ),
            user="root",
        )
        if result.return_code != 0:
            raise NonZeroAgentExitCodeError("Could not upload the CloudGeni bridge")

    def _require_configuration(self) -> None:
        missing = [name for name in _REQUIRED_ENV if not (self._get_env(name) or "").strip()]
        if missing:
            raise ValueError(
                "CloudGeni agent requires --ae values for: " + ", ".join(sorted(missing))
            )

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Run the bridge, then enforce host-side cancellation and lease cleanup."""
        self._require_configuration()
        encoded_prompt = base64.b64encode(instruction.encode()).decode()
        prompt_result = await environment.exec(
            command=f"echo '{encoded_prompt}' | base64 -d > {_REMOTE_PROMPT_PATH}"
        )
        if prompt_result.return_code != 0:
            raise NonZeroAgentExitCodeError("Could not upload the CloudGeni trial prompt")

        try:
            result = await environment.exec(
                command=f"python3 {_REMOTE_RUNNER_PATH}",
                env=dict(self._extra_env),
            )
            if result.return_code != 0:
                raise NonZeroAgentExitCodeError(
                    f"CloudGeni bridge failed (exit {result.return_code}); "
                    f"see /logs/agent/{_STATE_FILENAME}"
                )
        finally:
            await asyncio.to_thread(self._host_cleanup)

    def _host_api_url(self) -> str:
        configured = self._get_env("CLOUDGENI_HOST_API_URL") or self._get_env("CLOUDGENI_API_URL")
        if not configured:
            return ""
        return configured.replace("host.docker.internal", "localhost").rstrip("/")

    def _host_request(self, method: str, path: str) -> int:
        root = self._host_api_url()
        if not root.endswith("/api/v1"):
            root = f"{root}/api/v1"
        request = urllib.request.Request(
            f"{root}{path}",
            method=method,
            headers={
                "Accept": "application/json",
                "X-CLOUDGENI-API-KEY": self._get_env("CLOUDGENI_API_KEY") or "",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code
        except urllib.error.URLError:
            return 0

    def _host_cleanup(self) -> None:
        """Best-effort cleanup that still runs when Harbor cancels container execution."""
        state_path = self.logs_dir / _STATE_FILENAME
        try:
            state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        organization_id = self._get_env("CLOUDGENI_ORGANIZATION_ID")
        workspace_id = self._get_env("CLOUDGENI_WORKSPACE_ID")
        if not organization_id or not workspace_id or not self._host_api_url():
            return
        prefix = f"/organizations/{organization_id}/workspaces/{workspace_id}"
        cleanup = state.get("cleanup") if isinstance(state.get("cleanup"), dict) else {}
        session_id = state.get("sessionId")
        if isinstance(session_id, str) and session_id:
            cleanup["sessionDeleted"] = self._host_request(
                "DELETE", f"{prefix}/agent-sessions/{session_id}"
            ) in {200, 404}
        integration_id = state.get("integrationId")
        if isinstance(integration_id, str) and integration_id:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                status = self._host_request(
                    "DELETE",
                    f"{prefix}/integrations/{integration_id}/aws-bench-credential-lease",
                )
                if status in {200, 404}:
                    cleanup["leasePurged"] = True
                    break
                if status != 409:
                    break
                time.sleep(2)
        state["cleanup"] = cleanup
        try:
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        except OSError:
            pass

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Expose CloudGeni run identity and usage in Harbor's result metadata."""
        state_path = self.logs_dir / _STATE_FILENAME
        try:
            state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        usage = state.get("usage") if isinstance(state.get("usage"), dict) else {}
        input_tokens = usage.get("inputTokens")
        output_tokens = usage.get("outputTokens")
        if isinstance(input_tokens, int):
            context.n_input_tokens = input_tokens
        if isinstance(output_tokens, int):
            context.n_output_tokens = output_tokens
        context.metadata = {
            **(context.metadata or {}),
            "cloudgeniSessionId": state.get("sessionId"),
            "cloudgeniRunId": state.get("cloudgeniRunId"),
            "cloudgeniStatus": state.get("status"),
            "cloudgeniCleanup": state.get("cleanup"),
        }
