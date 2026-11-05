"""The plugin MCP endpoint rewrite shared by the Claude Code and Codex adapters."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from aws_bench.agents.plugin_endpoint import (
    build_endpoint_rewrite_command,
    validate_endpoint_override,
)

_FROM = "https://prod.example/mcp"
_TO = "https://preprod.example/mcp"


@pytest.mark.parametrize(
    ("endpoint_from", "endpoint_to", "endpoint_args", "has_plugins"),
    [
        (_FROM, None, None, True),  # `from` and `to` come together
        (None, _TO, None, True),
        (None, None, ["--region", "us-east-1"], True),  # extra arguments need both
        (_FROM, _TO, None, False),  # nothing to rewrite without a plugin
    ],
)
def test_invalid_combinations_are_rejected(endpoint_from, endpoint_to, endpoint_args, has_plugins):
    with pytest.raises(ValueError):
        validate_endpoint_override(endpoint_from, endpoint_to, endpoint_args, has_plugins)


def _config(root: Path, where: str, data: object) -> Path:
    path = root / where
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def _cached(root: Path, data: object, version: str = "1.0.0") -> Path:
    return _config(root, f"plugins/cache/market/sample/{version}/.mcp.json", data)


def _run(root: Path, endpoint_to: str = _TO, args: list[str] | None = None):
    command = build_endpoint_rewrite_command(_FROM, endpoint_to, args or [], str(root))
    return subprocess.run(["bash", "-c", command], capture_output=True, text=True)


def _args_config(endpoint: str = _FROM) -> dict:
    return {"mcpServers": {"aws": {"command": "uvx", "args": ["proxy", endpoint, "--skip-auth"]}}}


def test_replaces_the_endpoint_and_inserts_extra_arguments_after_it(tmp_path: Path):
    path = _cached(tmp_path, _args_config())

    result = _run(tmp_path, args=["--region", "us-east-1"])

    assert result.returncode == 0, result.stderr
    assert f"MCP_ENDPOINT_OK {_TO}" in result.stdout
    assert json.loads(path.read_text())["mcpServers"]["aws"]["args"] == [
        "proxy",
        _TO,
        "--region",
        "us-east-1",
        "--skip-auth",
    ]


def test_replaces_a_url_field_and_keeps_the_json_valid(tmp_path: Path):
    path = _cached(tmp_path, {"mcpServers": {"aws": {"type": "http", "url": _FROM}}})

    assert _run(tmp_path).returncode == 0

    assert json.loads(path.read_text())["mcpServers"]["aws"]["url"] == _TO


def test_extra_arguments_cannot_go_into_a_url_field(tmp_path: Path):
    original = {"mcpServers": {"aws": {"type": "http", "url": _FROM}}}
    path = _cached(tmp_path, original)

    result = _run(tmp_path, args=["--region", "us-east-1"])

    assert result.returncode != 0
    assert json.loads(path.read_text()) == original


def test_only_the_plugin_cache_is_rewritten(tmp_path: Path):
    _cached(tmp_path, _args_config())
    source = _config(tmp_path, "plugins/marketplaces/market/sample/.mcp.json", _args_config())

    assert _run(tmp_path).returncode == 0

    assert json.loads(source.read_text()) == _args_config()


def test_a_new_endpoint_that_extends_the_old_one_is_accepted(tmp_path: Path):
    """Values are compared as parsed JSON, so `.../mcp` -> `.../mcp/v2` is not a leftover."""
    path = _cached(tmp_path, _args_config())

    assert _run(tmp_path, endpoint_to=f"{_FROM}/v2").returncode == 0

    assert json.loads(path.read_text())["mcpServers"]["aws"]["args"][1] == f"{_FROM}/v2"


def test_fails_when_no_config_names_the_endpoint(tmp_path: Path):
    _cached(tmp_path, _args_config("https://unrelated.example/mcp"))

    result = _run(tmp_path)

    assert result.returncode != 0
    assert "names" in result.stderr


def test_fails_when_there_is_no_plugin_cache(tmp_path: Path):
    assert _run(tmp_path).returncode != 0


def test_values_are_passed_without_shell_interpretation(tmp_path: Path):
    """URLs and arguments go in as argv, so quotes, `$` and `&` are just characters."""
    tricky = "https://preprod.example/mcp?a=1&b='2'$HOME"
    path = _cached(tmp_path, _args_config())

    result = _run(tmp_path, endpoint_to=tricky, args=["--x", 'a "b" $c'])

    assert result.returncode == 0, result.stderr
    assert json.loads(path.read_text())["mcpServers"]["aws"]["args"][1:4] == [
        tricky,
        "--x",
        'a "b" $c',
    ]
