"""Bedrock-aware OpenCode agent for aws-bench.

Harbor's built-in ``OpenCode`` agent can target Amazon Bedrock (model form
``amazon-bedrock/<model_id>``, registered in ``~/.config/opencode/opencode.json``
by the base agent), but for ``provider == "amazon-bedrock"`` it forwards only the
**SigV4 credential chain** (``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` /
``AWS_REGION``) into the container. That is the wrong credential for aws-bench:

1. AWSBench authorizes Bedrock inference with a **bearer token**
   (``AWS_BEARER_TOKEN_BEDROCK``, minted per management account by ``env creds``),
   which Harbor never forwards -- so it never reaches opencode.
2. The SigV4 vars that *are* present in a trial belong to the **test (member)
   account** -- injected by the hook system so the agent can act on the resources
   under test -- while Bedrock inference is authorized against the separate
   **management/Bedrock account** that issued the bearer token. Forwarding the
   chain therefore authenticates Bedrock against an account with no Bedrock access.

This subclass forwards the bearer token and the Bedrock Region into
``_extra_env``. ``BaseInstalledAgent._exec`` merges ``_extra_env`` into every
command it runs (including the final ``opencode run``) *and* lets it override the
Region Harbor forwards from the host -- so the token reaches opencode and the
Region is pinned to the token's account.

The test-account SigV4 chain is deliberately left in place: the agent's own tools
(bash, aws CLI, MCP AWS servers) need it to act on the resources under test. This
means both credential types are present in the ``opencode run`` process, so
correctness relies on opencode's Bedrock provider preferring the bearer token
when both are available. Whether the opencode binary does so is the open
validation gate -- confirm on a probe lane before scaling (see the runbook).

Bedrock mode is auto-detected from a non-empty ``AWS_BEARER_TOKEN_BEDROCK`` -- an
unambiguous Bedrock signal with no other purpose, so no separate opt-in flag is
needed. When the token is absent this behaves exactly like Harbor's OpenCode.

Model IDs use opencode's ``provider/model_id`` form; for Bedrock that is
``amazon-bedrock/<bedrock-inference-profile-id>``, e.g.
``amazon-bedrock/global.anthropic.claude-sonnet-5``. Because the base agent
registers the model in ``opencode.json`` for any ``amazon-bedrock`` model, ids
outside opencode's built-in registry are accepted -- so, unlike kiro-cli, there is
no model-ID dialect to satisfy.
"""

from __future__ import annotations

from harbor.agents.installed.opencode import OpenCode as _HarborOpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# Region the per-MA bearer token is authorized against. The token is minted in
# us-east-1 (SSM ``/bedrock-aws-bench/bedrock-api-key``), matching the baseline
# Strands agent's default; ``global.*`` inference profiles are cross-region.
_DEFAULT_AWS_REGION = "us-east-1"


class OpenCode(_HarborOpenCode):
    """OpenCode agent that can target Amazon Bedrock via a bearer token."""

    def _is_bedrock_mode(self) -> bool:
        """Detect Bedrock mode from the environment.

        Triggered solely by a non-empty ``AWS_BEARER_TOKEN_BEDROCK``. That token
        is an unambiguous Bedrock signal -- it has no other purpose -- so no
        separate opt-in flag is needed. (The standard AWS credential-chain vars
        cannot serve as a trigger: they are present in any AWS shell regardless
        of whether Bedrock is intended.) When the token is absent this behaves
        exactly like Harbor's OpenCode.

        Read via ``_get_env`` so the token counts however it was supplied:
        ``--ae`` values land in ``_extra_env``, not ``os.environ``.
        """
        return bool((self._get_env("AWS_BEARER_TOKEN_BEDROCK") or "").strip())

    def _inject_bedrock_env(self) -> None:
        """Forward Bedrock auth env into ``_extra_env`` so every exec inherits it.

        ``BaseInstalledAgent._exec`` merges ``_extra_env`` over the per-command
        environment of every command (including the final ``opencode run``), so
        populating it here is sufficient and also overrides the host
        ``AWS_REGION`` Harbor forwards for ``amazon-bedrock``. Values already
        supplied via ``extra_env`` / ``-ae`` take priority and are never
        overwritten.

        Only the bearer token is added for Bedrock auth -- the SigV4 chain Harbor
        forwards is left untouched, because the agent's own tools need those
        (test-account) credentials to act on the resources under test.
        ``AWS_REGION`` is required for Bedrock auth.
        """
        token = (self._get_env("AWS_BEARER_TOKEN_BEDROCK") or "").strip()
        self._extra_env.setdefault("AWS_BEARER_TOKEN_BEDROCK", token)
        # Bedrock requires a Region. Honor extra_env / host AWS_REGION, else default.
        self._extra_env.setdefault("AWS_REGION", self._get_env("AWS_REGION") or _DEFAULT_AWS_REGION)

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Run the task, forwarding Bedrock auth env first when in Bedrock mode.

        In Bedrock mode, inject the bearer token + Region into ``_extra_env`` and
        then defer to Harbor's run (which registers the model in opencode.json and
        executes ``opencode run``).
        """
        if self._is_bedrock_mode():
            self._inject_bedrock_env()
        # super().run is decorated with @with_prompt_template; do not re-decorate.
        await super().run(instruction, environment, context)
