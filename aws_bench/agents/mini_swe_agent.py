"""Mini SWE Agent with boto3 for Bedrock support.

Subclasses the harbor MiniSweAgent to add boto3 to the install,
which is required by LiteLLM for Bedrock model calls.

Usage:
    aws-bench run --agent-import-path "aws_bench.agents.mini_swe_bedrock:MiniSweBedrock" \
        -m "bedrock/us.anthropic.claude-sonnet-5-5" ...
"""

from __future__ import annotations

from harbor.agents.installed.mini_swe_agent import MiniSweAgent as _HarborMiniSweAgent
from harbor.environments.base import BaseEnvironment


class MiniSweAgent(_HarborMiniSweAgent):
    """MiniSweAgent with boto3 injected for Bedrock LLM calls."""

    async def install(self, environment: BaseEnvironment) -> None:
        """Install mini-swe-agent with boto3 for Bedrock API calls."""
        await super().install(environment)
        # uv has no incremental inject, and the base fuses uv-bootstrap with the
        # tool install in one shell chain, so there's no seam to pass --with through.
        # Re-run the install once with boto3, mirroring the base's version spec so an
        # --agent-version pin survives the --force reinstall.
        #
        # litellm<=1.91.3: litellm 1.92.0 imports the proxy-only fastapi dependency
        # on completion(..., tools=[...]) calls, which mini-swe-agent always makes.
        # Installs without the [proxy] extra (like this tool venv) then crash with
        # ModuleNotFoundError: No module named 'fastapi'. mini-swe-agent's own spec
        # leaves litellm's upper bound open, so we cap it here. Remove once the
        # upstream fix (https://github.com/BerriAI/litellm/issues/32993) is released
        # and verified.
        version_spec = f"=={self._version}" if self._version else ""
        await self.exec_as_agent(
            environment,
            command=(
                'source "$HOME/.local/bin/env" && '
                f"uv tool install mini-swe-agent{version_spec} "
                "--with boto3 --with 'litellm<=1.91.3' --force"
            ),
        )
