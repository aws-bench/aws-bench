"""Bedrock-aware Codex agent for aws-bench.

Harbor's built-in ``Codex`` agent only knows how to talk to OpenAI: it writes
``OPENAI_API_KEY`` into the container and runs ``codex exec`` with no provider
configured, so codex falls back to its default OpenAI provider. To drive Codex
against Amazon Bedrock, three things are missing, and this subclass fills all three:

1. ``model_provider = "amazon-bedrock"`` must be present in the *container's*
   ``$CODEX_HOME/config.toml``, which harbor renders and uploads from the
   effective config dict. Without it codex never routes to Bedrock, regardless
   of which env vars are set. This cannot be supplied via ``-ae``.
2. The Bedrock auth env (``AWS_BEARER_TOKEN_BEDROCK`` + ``AWS_REGION``) must be
   forwarded from the host into the codex subprocess. Harbor's Codex does not.
3. The uploaded ``config.toml`` must be valid TOML in the directory codex reads.
   Harbor renders it with ``toml.dumps``, which rewrites backslash-x sequences and
   control characters in string values. It also uploads the file to a fixed path
   even when a ``-ae CODEX_HOME`` overlay makes codex read elsewhere. The upload
   override escapes every string and follows the agent env's ``CODEX_HOME``.

Bedrock mode is auto-detected from a non-empty ``AWS_BEARER_TOKEN_BEDROCK`` in
the environment. When that token is absent this behaves exactly like harbor's
Codex (OpenAI auth).

Region: pass it with ``-ae AWS_REGION=us-east-2`` (API-key auth requires a
Region). The host ``AWS_REGION`` is auto-forwarded as a fallback.
"""

from __future__ import annotations

import os
import shlex
from pathlib import PurePosixPath
from typing import Any

import toml
from harbor.agents.installed.codex import Codex as _HarborCodex
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_DEFAULT_AWS_REGION = "us-east-2"

# TOML basic strings must escape these two characters; every other printable
# character is emitted verbatim. Control characters get their own escapes below.
_TOML_SIMPLE_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _toml_basic_string(value: str) -> str:
    r"""Render ``value`` as a quoted, escaped TOML basic string.

    Codex's ``config.toml`` is TOML, so any value interpolated into it must be a
    valid TOML string. This escapes ``\\`` and ``"`` and the control characters
    TOML forbids in a basic string, so arbitrary command tokens, argument
    values, or URLs cannot produce invalid TOML.

    Args:
        value: The raw string to encode.

    Returns:
        The value wrapped in double quotes with TOML escapes applied.
    """
    encoded = []
    for char in value:
        simple = _TOML_SIMPLE_ESCAPES.get(char)
        if simple is not None:
            encoded.append(simple)
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            encoded.append(f"\\u{ord(char):04X}")
        else:
            encoded.append(char)
    return '"' + "".join(encoded) + '"'


class Codex(_HarborCodex):
    """Codex agent that can target Amazon Bedrock in addition to OpenAI."""

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

    def _inject_bedrock_env(self) -> dict[str, str]:
        """Resolve the Bedrock auth env and return it for ``run`` to overlay.

        Harbor's ``Trial`` snapshots ``extra_env`` before ``run()`` and ``_exec``
        no longer merges it, so values added here reach ``codex exec`` only
        through the returned dict, which ``run`` applies with
        ``environment.scoped_exec_env``. ``setdefault`` keeps values supplied via
        ``-ae`` authoritative.

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
        return {k: self._extra_env[k] for k in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION")}

    def _build_effective_config(self, openai_base_url: str | None = None) -> dict[str, Any]:
        """Add ``model_provider = "amazon-bedrock"`` to the config harbor uploads.

        Harbor renders ``$CODEX_HOME/config.toml`` from this dict and uploads it
        whole, so the provider has to be in the dict; a line appended to the file
        beforehand would be overwritten. Harbor skips the upload for an empty
        dict, and this key makes it non-empty in Bedrock mode.
        """
        config = super()._build_effective_config(openai_base_url)
        if self._is_bedrock_mode():
            config.setdefault("model_provider", "amazon-bedrock")
        return config

    async def _upload_effective_config(
        self, environment: BaseEnvironment, config: dict[str, Any], remote_path: str
    ) -> None:
        r"""Upload ``config.toml`` with every string escaped, to the home codex reads.

        Harbor 0.22.0 renders the file with ``toml.dumps``, whose ``_dump_str`` turns
        a literal ``\x41`` into ``A`` and raises ``IndexError`` on a leading backspace.
        It also uploads to the fixed ``_REMOTE_CODEX_HOME`` while the trial applies
        ``extra_env`` as the container's exec overlay, so under ``-ae CODEX_HOME``
        every ``$CODEX_HOME`` reference in the container resolves elsewhere. The host
        environment never reaches the container, so only ``extra_env`` decides the path.
        """
        if not config:
            return
        codex_home = self.extra_env.get("CODEX_HOME")
        if codex_home:
            remote_path = (PurePosixPath(codex_home) / "config.toml").as_posix()
        encoder = toml.TomlEncoder()
        encoder.dump_funcs[str] = _toml_basic_string
        await self._upload_config_text(
            environment,
            content=toml.dumps(config, encoder=encoder),
            remote_path=remote_path,
            filename="config.toml",
        )

    def _build_register_mcp_servers_command(self) -> str | None:
        r"""Write MCP server config to ``$CODEX_HOME/config.toml`` with correct keys.

        Harbor's base implementation collapses a stdio server's ``command`` and
        ``args`` into a single ``command`` string (via ``shlex.join``) and emits
        no ``args`` key, so codex tries to exec a binary whose filename is the
        entire joined command line. The exec fails, the MCP server never starts,
        and the agent silently loses its tools. Codex's ``config.toml`` schema
        requires ``command`` (the executable) and ``args`` (an array) as separate
        keys for a stdio server. The base implementation also interpolates raw
        values into ``"..."`` without escaping, producing invalid TOML for any
        value containing ``"`` or ``\\``.

        This override renders ``command`` and ``args`` as separate keys and
        escapes every emitted value as a TOML basic string. ``MCPServerConfig``
        exposes only ``name``/``transport``/``url``/``command``/``args`` — no env
        or timeout field — so there is nothing further to render.

        Returns:
            A shell command appending the config, or ``None`` when no MCP servers
            are configured.
        """
        if not self.mcp_servers:
            return None
        lines: list[str] = []
        for server in self.mcp_servers:
            lines.append(f"[mcp_servers.{server.name}]")
            if server.transport == "stdio":
                lines.append(f"command = {_toml_basic_string(server.command or '')}")
                rendered_args = ", ".join(_toml_basic_string(arg) for arg in server.args)
                lines.append(f"args = [{rendered_args}]")
            else:
                lines.append(f"url = {_toml_basic_string(server.url or '')}")
            lines.append("")
        escaped_config = shlex.quote("\n".join(lines))
        return f'echo {escaped_config} >> "$CODEX_HOME/config.toml"'

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Run the task, overlaying the Bedrock auth env when in Bedrock mode.

        The provider itself reaches ``config.toml`` through
        ``_build_effective_config``, which harbor's run calls.
        """
        if self._is_bedrock_mode():
            bedrock_env = self._inject_bedrock_env()
            with environment.scoped_exec_env(bedrock_env):
                await super().run(instruction, environment, context)
            return
        # super().run is decorated with @with_prompt_template; do not re-decorate.
        await super().run(instruction, environment, context)
