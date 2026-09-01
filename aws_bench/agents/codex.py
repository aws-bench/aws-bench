"""Bedrock- and Azure-aware Codex agent for aws-bench.

Harbor's built-in ``Codex`` agent only knows how to talk to OpenAI: it writes
``OPENAI_API_KEY`` into the container and runs ``codex exec`` with no provider
configured, so codex falls back to its default OpenAI provider. To drive Codex
against Amazon Bedrock, two things are missing, and this subclass fills both:

1. ``model_provider = "amazon-bedrock"`` must be written into the *container's*
   ``$CODEX_HOME/config.toml``. Without it codex never routes to Bedrock,
   regardless of which env vars are set. This cannot be supplied via ``-ae``.
2. The Bedrock auth env (``AWS_BEARER_TOKEN_BEDROCK`` + ``AWS_REGION``) must be
   forwarded from the host into the codex subprocess. Harbor's Codex does not.

Bedrock mode is auto-detected from a non-empty ``AWS_BEARER_TOKEN_BEDROCK``.
Azure mode is selected when both ``AZURE_OPENAI_ENDPOINT`` and
``AZURE_OPENAI_API_KEY`` are present. When neither is configured this behaves
exactly like harbor's Codex (OpenAI auth).

Region: pass it with ``-ae AWS_REGION=us-east-2`` (API-key auth requires a
Region). The host ``AWS_REGION`` is auto-forwarded as a fallback.
"""

from __future__ import annotations

import os
import shlex
from urllib.parse import urlsplit

from harbor.agents.installed.codex import Codex as _HarborCodex
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_DEFAULT_AWS_REGION = "us-east-2"


class Codex(_HarborCodex):
    """Codex agent that can target Azure OpenAI or Bedrock in addition to OpenAI."""

    @staticmethod
    def _is_bedrock_mode() -> bool:
        """Detect Bedrock mode from the environment.

        Triggered solely by a non-empty ``AWS_BEARER_TOKEN_BEDROCK``. That token
        is an unambiguous Bedrock signal — it has no other purpose — so no
        separate opt-in flag is needed. (The standard AWS credential-chain vars
        cannot serve as a trigger: they are present in any AWS shell regardless
        of whether Bedrock is intended.) When the token is absent, this behaves
        exactly like harbor's Codex against OpenAI.
        """
        return bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip())

    def _inject_bedrock_env(self) -> None:
        """Forward Bedrock auth env into ``_extra_env`` so every exec inherits it.

        ``BaseInstalledAgent._exec`` merges ``_extra_env`` into the environment of
        every command (including the final ``codex exec``), so populating it here
        is sufficient. Values already supplied via ``-ae`` take priority and are
        never overwritten.

        Only the bearer token is forwarded for Bedrock auth — never the SigV4
        credential chain. In an aws-bench trial those AWS_* credentials belong to
        the *test account* (injected by the hook system so the agent can act on
        the resources under test); Bedrock inference is authorized against the
        separate management/Bedrock account that issued the bearer token, so the
        chain would be the wrong credentials anyway. ``AWS_REGION`` is required.
        """
        token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip()
        self._extra_env.setdefault("AWS_BEARER_TOKEN_BEDROCK", token)
        # Bedrock requires a Region. Honor -ae / host AWS_REGION, else default.
        self._extra_env.setdefault("AWS_REGION", os.environ.get("AWS_REGION", _DEFAULT_AWS_REGION))

    def _azure_settings(self) -> tuple[str, str] | None:
        """Resolve an Azure OpenAI endpoint/key pair from agent or host env."""
        endpoint = (self._get_env("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
        api_key = (self._get_env("AZURE_OPENAI_API_KEY") or "").strip()
        if not endpoint and not api_key:
            return None
        if not endpoint or not api_key:
            raise ValueError(
                "Azure Codex requires both AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
            )
        parsed = urlsplit(endpoint)
        if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("AZURE_OPENAI_ENDPOINT must be an HTTPS Azure resource endpoint")
        return endpoint, api_key

    def _inject_azure_env(self, api_key: str) -> None:
        """Forward the Azure API key only through the process environment."""
        self._extra_env.setdefault("AZURE_OPENAI_API_KEY", api_key)

    async def _write_azure_provider_config(
        self, environment: BaseEnvironment, endpoint: str
    ) -> None:
        """Configure Codex's Responses provider for Azure OpenAI's v1 API."""
        remote_codex_home = self._REMOTE_CODEX_HOME.as_posix()
        provider_block = (
            'model_provider = "azure_openai"\n'
            "[model_providers.azure_openai]\n"
            'name = "Azure OpenAI"\n'
            f'base_url = "{endpoint}/openai/v1"\n'
            'wire_api = "responses"\n'
            'env_http_headers = { "api-key" = "AZURE_OPENAI_API_KEY" }\n'
        )
        await self.exec_as_agent(
            environment,
            command=(
                f'mkdir -p "$CODEX_HOME" && '
                f'echo {shlex.quote(provider_block)} >> "$CODEX_HOME/config.toml"'
            ),
            env={"CODEX_HOME": remote_codex_home},
        )

    async def _write_bedrock_provider_config(self, environment: BaseEnvironment) -> None:
        """Write ``model_provider = "amazon-bedrock"`` into the container config.

        Runs as its own step before harbor's setup. ``model_provider`` is a
        top-level TOML key and harbor only ever *appends* (``>>``) to
        ``config.toml`` (base_url, ``[mcp_servers.*]`` tables), so writing it
        first guarantees it stays above every table header — a bare key written
        after a table header would be mis-parsed as belonging to that table.

        We ``mkdir -p`` the home ourselves so this does not depend on harbor's
        own mkdir having run yet (it is idempotent with harbor's).
        """
        remote_codex_home = self._REMOTE_CODEX_HOME.as_posix()
        provider_block = 'model_provider = "amazon-bedrock"\n'
        await self.exec_as_agent(
            environment,
            command=(
                f'mkdir -p "$CODEX_HOME" && '
                f'echo {shlex.quote(provider_block)} >> "$CODEX_HOME/config.toml"'
            ),
            env={"CODEX_HOME": remote_codex_home},
        )

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Run the task, configuring a selected non-OpenAI provider first.

        Provider config is written before harbor's run, which appends the rest
        of config.toml. Azure and Bedrock are mutually exclusive.
        """
        azure = self._azure_settings()
        bedrock = self._is_bedrock_mode()
        if azure and bedrock:
            raise ValueError("Configure only one Codex provider: Azure OpenAI or Amazon Bedrock")
        if azure:
            endpoint, api_key = azure
            self._inject_azure_env(api_key)
            await self._write_azure_provider_config(environment, endpoint)
        elif bedrock:
            self._inject_bedrock_env()
            await self._write_bedrock_provider_config(environment)
        # super().run is decorated with @with_prompt_template; do not re-decorate.
        await super().run(instruction, environment, context)
