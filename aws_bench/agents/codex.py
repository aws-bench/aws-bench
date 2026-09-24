"""Bedrock-aware Codex agent for aws-bench.

Harbor's built-in ``Codex`` agent only knows how to talk to OpenAI: it writes
``OPENAI_API_KEY`` into the container and runs ``codex exec`` with no provider
configured, so codex falls back to its default OpenAI provider. To drive Codex
against Amazon Bedrock, two things are missing, and this subclass fills both:

1. ``model_provider = "amazon-bedrock"`` must be written into the *container's*
   ``$CODEX_HOME/config.toml``. Without it codex never routes to Bedrock,
   regardless of which env vars are set. This cannot be supplied via ``-ae``.
2. The Bedrock auth env (``AWS_BEARER_TOKEN_BEDROCK`` + ``AWS_REGION``) must be
   forwarded from the host into the codex subprocess. Harbor's Codex does not.

Bedrock mode is auto-detected from a non-empty ``AWS_BEARER_TOKEN_BEDROCK`` in
the environment. When that token is absent this behaves exactly like harbor's
Codex (OpenAI auth).

Region: pass it with ``-ae AWS_REGION=us-east-2`` (API-key auth requires a
Region). The host ``AWS_REGION`` is auto-forwarded as a fallback.
"""

from __future__ import annotations

import os
import shlex

from harbor.agents.installed.codex import Codex as _HarborCodex
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_DEFAULT_AWS_REGION = "us-east-2"

# --- aws-mcp (Agent Toolkit) proxy hardening -------------------------------
#
# The aws-mcp server is launched by the Agent Toolkit's MCP config as
# ``uvx mcp-proxy-for-aws-cli@latest <endpoint> --skip-auth ...``. Left as
# emitted by the base path (``command``/``args`` only), two harness bugs keep
# Codex from ever using the toolkit:
#
#   1. Codex silently drops an *optional* MCP server whose startup exceeds a
#      small default grace. ``uvx`` cold start for the proxy is 2-4s, so the
#      server is frequently dropped before it registers and the tools never
#      appear. ``mcp_optional_startup_grace_ms = 0`` (a top-level key, which
#      MUST sit above every ``[mcp_servers.*]`` table) disables that early
#      drop, and a generous per-server ``startup_timeout_sec`` gives the cold
#      start room to finish.
#   2. Codex starts MCP servers with a minimal environment, so the proxy's
#      boto3 does not inherit the shell's target-account creds/profile and
#      falls back to IMDS -> the runner instance role -> ``AccessDenied``.
#      ``env_vars`` forwards the AWS account/profile settings into the
#      subprocess so ``run_script`` acts against the task's target account.
#
# The proxy package is additionally pinned to an exact version (rather than the
# drifting ``@latest``) so a re-run reproduces the exact toolkit the original
# codex ATK lanes ran. ``1.7.0`` is the version ``@latest`` resolved to for
# every codex ATK lane on framework pin 8748fd26 (all ran 2026-09-15, after
# 1.7.0's 11:20 UTC release, and 1.7.0 remains the newest published release).
_AWS_MCP_PROXY_PACKAGE = "mcp-proxy-for-aws-cli"
_AWS_MCP_PROXY_PIN = f"{_AWS_MCP_PROXY_PACKAGE}==1.7.0"
_MCP_OPTIONAL_STARTUP_GRACE_MS = 0
_AWS_MCP_STARTUP_TIMEOUT_SEC = 120
_AWS_MCP_ENV_VARS = (
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_CONFIG_FILE",
    "AWS_SHARED_CREDENTIALS_FILE",
)

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

    @staticmethod
    def _pin_aws_mcp_proxy(args: list[str]) -> tuple[list[str], bool]:
        """Pin the ``mcp-proxy-for-aws-cli`` package spec in ``args`` to an exact version.

        The aws-mcp server's args reference the proxy package, but the version
        drifts because the Agent Toolkit's MCP config uses ``@latest``. This
        rewrites that spec to :data:`_AWS_MCP_PROXY_PIN`, matching the package
        whether it is unpinned (``mcp-proxy-for-aws-cli``), tagged
        (``...@latest``), or already ``==`` pinned. Only the package spec token
        is rewritten; the endpoint URL and every flag are left untouched.

        Args:
            args: The stdio server's argument vector.

        Returns:
            ``(rewritten_args, found)`` where ``found`` is ``True`` iff a proxy
            spec was present (i.e. this is the uvx aws-mcp proxy server).
        """
        rewritten: list[str] = []
        found = False
        for arg in args:
            if (
                arg == _AWS_MCP_PROXY_PACKAGE
                or arg.startswith(f"{_AWS_MCP_PROXY_PACKAGE}@")
                or arg.startswith(f"{_AWS_MCP_PROXY_PACKAGE}==")
            ):
                rewritten.append(_AWS_MCP_PROXY_PIN)
                found = True
            else:
                rewritten.append(arg)
        return rewritten, found

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
        escapes every emitted value as a TOML basic string. It additionally
        hardens the Agent Toolkit's aws-mcp proxy server so Codex actually keeps
        and uses it (see the module-level notes on the two harness bugs):

        * ``mcp_optional_startup_grace_ms = 0`` is emitted once as a top-level
          key, *above* every ``[mcp_servers.*]`` table, so Codex does not drop a
          server that is slow to start (uvx cold start is 2-4s).
        * For the uvx aws-mcp proxy server (detected by its
          ``mcp-proxy-for-aws-cli`` arg), the proxy is pinned to an exact
          version, and ``startup_timeout_sec`` + ``env_vars`` are emitted so the
          cold start has time to finish and the subprocess inherits the AWS
          account/profile settings (``MCPServerConfig`` itself carries no
          timeout/env field, so these are emitted here rather than modeled).

        Returns:
            A shell command appending the config, or ``None`` when no MCP servers
            are configured.
        """
        if not self.mcp_servers:
            return None
        lines: list[str] = []
        # Global key: must precede every [mcp_servers.*] table header below.
        lines.append(f"mcp_optional_startup_grace_ms = {_MCP_OPTIONAL_STARTUP_GRACE_MS}")
        lines.append("")
        for server in self.mcp_servers:
            lines.append(f"[mcp_servers.{server.name}]")
            if server.transport == "stdio":
                lines.append(f"command = {_toml_basic_string(server.command or '')}")
                args, is_aws_mcp_proxy = self._pin_aws_mcp_proxy(list(server.args))
                rendered_args = ", ".join(_toml_basic_string(arg) for arg in args)
                lines.append(f"args = [{rendered_args}]")
                if is_aws_mcp_proxy:
                    lines.append(f"startup_timeout_sec = {_AWS_MCP_STARTUP_TIMEOUT_SEC}")
                    rendered_env = ", ".join(_toml_basic_string(v) for v in _AWS_MCP_ENV_VARS)
                    lines.append(f"env_vars = [{rendered_env}]")
            else:
                lines.append(f"url = {_toml_basic_string(server.url or '')}")
            lines.append("")
        escaped_config = shlex.quote("\n".join(lines))
        return f'echo {escaped_config} >> "$CODEX_HOME/config.toml"'

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Run the task, configuring Bedrock first when in Bedrock mode.

        In Bedrock mode, forward auth env and write the provider config, then
        defer to harbor's run (which appends the rest of config.toml).
        """
        if self._is_bedrock_mode():
            self._inject_bedrock_env()
            await self._write_bedrock_provider_config(environment)
        # super().run is decorated with @with_prompt_template; do not re-decorate.
        await super().run(instruction, environment, context)
