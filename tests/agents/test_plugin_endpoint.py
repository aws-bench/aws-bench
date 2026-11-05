"""The shared plugin MCP endpoint rewrite used by the Claude Code and Codex adapters."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from aws_bench.agents.plugin_endpoint import (
    build_endpoint_rewrite_command,
    validate_endpoint_override,
)

_FROM = "https://prod.example/mcp"
_TO = "https://preprod.example/mcp"


# ── validation ──


def test_valid_override_returns_a_copy_of_the_extra_arguments():
    extra = ["--region", "us-east-1"]
    got = validate_endpoint_override(_FROM, _TO, extra, has_plugins=True)
    assert got == extra
    assert got is not extra


def test_no_override_is_valid_without_plugins():
    assert validate_endpoint_override(None, None, None, has_plugins=False) == []


@pytest.mark.parametrize(
    ("endpoint_from", "endpoint_to", "endpoint_args"),
    [
        (_FROM, None, None),
        (None, _TO, None),
        (None, None, ["--region", "us-east-1"]),
    ],
)
def test_override_arguments_must_be_complete(endpoint_from, endpoint_to, endpoint_args):
    """`from` and `to` come together, and extra arguments need both."""
    with pytest.raises(ValueError):
        validate_endpoint_override(endpoint_from, endpoint_to, endpoint_args, has_plugins=True)


def test_override_needs_plugins():
    with pytest.raises(ValueError, match="plugins"):
        validate_endpoint_override(_FROM, _TO, None, has_plugins=False)


@pytest.mark.parametrize("value", ["a|b", "line\nbreak"])
def test_override_rejects_sed_delimiters(value: str):
    """`|` delimits the sed expression and newlines would split it, so both are refused."""
    with pytest.raises(ValueError):
        validate_endpoint_override(_FROM, value, None, has_plugins=True)
    with pytest.raises(ValueError):
        validate_endpoint_override(_FROM, _TO, [value], has_plugins=True)


# ── the rewrite command ──


def _gnu_sed() -> bool:
    return subprocess.run(["sed", "--version"], capture_output=True).returncode == 0


def _plugin_config(config_dir: Path, text: str) -> Path:
    """Write ``text`` where both CLIs cache an installed plugin's MCP config."""
    path = config_dir / "plugins" / "cache" / "market" / "sample" / "1.0.0" / ".mcp.json"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    return path


def _run(config_dir: Path, args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    command = build_endpoint_rewrite_command(_FROM, _TO, args or [], str(config_dir))
    return subprocess.run(["bash", "-c", command], capture_output=True, text=True)


def test_command_is_valid_shell(tmp_path: Path):
    command = build_endpoint_rewrite_command(_FROM, _TO, ["--region", "us-east-1"], str(tmp_path))
    assert subprocess.run(["bash", "-n", "-c", command]).returncode == 0


@pytest.mark.skipif(not _gnu_sed(), reason="the rewrite runs in a Linux container with GNU sed")
class TestRewriteBehavior:
    def test_rewrites_the_endpoint_and_inserts_extra_arguments(self, tmp_path: Path):
        config = _plugin_config(tmp_path, f'{{"args": ["proxy", "{_FROM}", "--skip-auth"]}}')

        result = _run(tmp_path, ["--region", "us-east-1"])

        assert result.returncode == 0, result.stderr
        assert f"MCP_ENDPOINT_OK {_TO}" in result.stdout
        assert config.read_text() == (
            f'{{"args": ["proxy", "{_TO}", "--region", "us-east-1", "--skip-auth"]}}'
        )

    def test_fails_when_the_old_endpoint_remains(self, tmp_path: Path):
        """A trial must never run silently against the original endpoint."""
        # Not a whole JSON string, so the rewrite leaves it and the check must catch it.
        _plugin_config(tmp_path, f'{{"url": "{_FROM}/v2"}}')

        result = _run(tmp_path)

        assert result.returncode != 0
        assert "still configured" in result.stderr

    def test_fails_when_no_config_names_the_new_endpoint(self, tmp_path: Path):
        _plugin_config(tmp_path, '{"args": ["proxy", "https://unrelated.example/mcp"]}')

        result = _run(tmp_path)

        assert result.returncode != 0
        assert "no plugin config names" in result.stderr
