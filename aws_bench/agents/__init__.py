"""Custom agents for aws-bench."""

from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.models.agent.name import AgentName

from aws_bench.agents.baseline_agent import StrandsAgent
from aws_bench.agents.claude_code import ClaudeCode
from aws_bench.agents.codex import Codex
from aws_bench.agents.kiro_cli import KiroCli
from aws_bench.agents.mini_swe_agent import MiniSweAgent
from aws_bench.agents.opencode import OpenCode


def _override_builtin(agent_cls: type[BaseAgent]) -> None:
    """Swap a built-in Harbor agent for an aws-bench subclass of the same name.

    The name is already a valid AgentName, so only the factory mapping changes;
    `-a <name>` then resolves to the subclass.
    """
    name = AgentName(agent_cls.name())
    AgentFactory._AGENT_MAP[name] = agent_cls
    AgentFactory._AGENTS = [
        agent_cls if agent.name() == agent_cls.name() else agent for agent in AgentFactory._AGENTS
    ]


# Route `-a codex` to the Bedrock-capable subclass, `-a claude-code` to the
# plugin-capable subclass, `-a opencode` to the Bedrock-capable subclass, and
# `-a mini-swe-agent` to the Bedrock-capable subclass.
_override_builtin(Codex)
_override_builtin(ClaudeCode)
_override_builtin(OpenCode)
_override_builtin(MiniSweAgent)

# Register kiro-cli so `-a kiro-cli` works without --agent-import-path, by
# patching Harbor's AgentName enum + AgentFactory at import time.
_KIRO_CLI_NAME = "kiro-cli"
if _KIRO_CLI_NAME not in AgentName._value2member_map_:
    _member = str.__new__(AgentName, _KIRO_CLI_NAME)
    _member._name_ = "KIRO_CLI"
    _member._value_ = _KIRO_CLI_NAME
    AgentName._member_map_["KIRO_CLI"] = _member
    AgentName._value2member_map_[_KIRO_CLI_NAME] = _member
    AgentName._member_names_.append("KIRO_CLI")
    AgentFactory._AGENTS.append(KiroCli)
    AgentFactory._AGENT_MAP[AgentName(_KIRO_CLI_NAME)] = KiroCli

# Register aws-bench-baseline-agent so `-a aws-bench-baseline-agent` works.
_BASELINE_AGENT_NAME = "aws-bench-baseline-agent"
if _BASELINE_AGENT_NAME not in AgentName._value2member_map_:
    _member = str.__new__(AgentName, _BASELINE_AGENT_NAME)
    _member._name_ = "AWS_BENCH_BASELINE_AGENT"
    _member._value_ = _BASELINE_AGENT_NAME
    AgentName._member_map_["AWS_BENCH_BASELINE_AGENT"] = _member
    AgentName._value2member_map_[_BASELINE_AGENT_NAME] = _member
    AgentName._member_names_.append("AWS_BENCH_BASELINE_AGENT")
    AgentFactory._AGENTS.append(StrandsAgent)
    AgentFactory._AGENT_MAP[AgentName(_BASELINE_AGENT_NAME)] = StrandsAgent
