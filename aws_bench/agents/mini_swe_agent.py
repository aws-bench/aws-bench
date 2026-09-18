"""Mini SWE Agent with boto3 for Bedrock support.

Subclasses the harbor MiniSweAgent for two Bedrock needs. boto3 is added to the
install, which LiteLLM requires for Bedrock model calls. The generic
``MSWEA_API_KEY`` is restored when the trial's empty ``AWS_ACCESS_KEY_ID`` guard
would otherwise be taken as the Bedrock API key.

Usage:
    aws-bench run --agent-import-path "aws_bench.agents.mini_swe_bedrock:MiniSweBedrock" \
        -m "bedrock/us.anthropic.claude-sonnet-4-6" ...
"""

from __future__ import annotations

from dataclasses import replace
from typing import override

from harbor.agents.installed.mini_swe_agent import MiniSweAgent as _HarborMiniSweAgent
from harbor.agents.model_connection import ResolvedModelConnection
from harbor.environments.base import BaseEnvironment


class MiniSweAgent(_HarborMiniSweAgent):
    """MiniSweAgent with boto3 injected for Bedrock LLM calls."""

    @property
    @override
    def model_connection(self) -> ResolvedModelConnection:
        """Restore the generic ``MSWEA_API_KEY`` the trial's empty Bedrock guard shadows.

        Harbor 0.22.0 resolves the agent's env before the host env
        (``BaseInstalledAgent._env_sources``), so the trial's ``AWS_ACCESS_KEY_ID=""``
        matches as the Bedrock API key first and a host ``MSWEA_API_KEY`` never
        reaches mini-swe-agent. An explicit ``-ae`` value for the generic key
        resolves ahead of the guard, so it is left as given.
        """
        access = super().model_connection
        if access.provider != "amazon-bedrock" or access.api_key != "":
            return access
        generic = self._get_env("MSWEA_API_KEY")
        if generic is None:
            return access
        return replace(access, api_key=generic, env={**access.env, "MSWEA_API_KEY": generic})

    async def install(self, environment: BaseEnvironment) -> None:
        """Install mini-swe-agent with boto3 for Bedrock API calls.

        The env-file source is guarded the way harbor's own install guards it:
        Harbor 0.22.0 skips the uv installer when ``uv`` is already on PATH, so
        ``$HOME/.local/bin/env`` exists only after a fresh bootstrap.
        """
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
                'if [ -f "$HOME/.local/bin/env" ]; then source "$HOME/.local/bin/env"; fi && '
                'export PATH="$HOME/.local/bin:$PATH" && '
                f"uv tool install mini-swe-agent{version_spec} "
                "--with boto3 --with 'litellm<=1.91.3' --force"
            ),
        )
