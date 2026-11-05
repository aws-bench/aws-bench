"""The shared plugin MCP endpoint rewrite used by the Claude Code and Codex adapters."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from aws_bench.agents.plugin_endpoint import (
    build_endpoint_rewrite_command,
    rewrite_plugin_endpoints,
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


# ── the rewrite ──


def _config(root: Path, where: str, data: object) -> Path:
    """Write an MCP config under ``root`` and return its path."""
    path = root / where
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data, indent=2))
    return path


def _cache_config(
    root: Path, data: object, name: str = ".mcp.json", version: str = "1.0.0"
) -> Path:
    return _config(root, f"plugins/cache/market/sample/{version}/{name}", data)


def _rewrite(root: Path, args: list[str] | None = None, endpoint_to: str = _TO) -> list[str]:
    return rewrite_plugin_endpoints(str(root / "plugins" / "cache"), _FROM, endpoint_to, args or [])


def _args_config(endpoint: str = _FROM) -> dict:
    return {"mcpServers": {"aws": {"command": "uvx", "args": ["proxy", endpoint, "--skip-auth"]}}}


def test_replaces_the_endpoint_and_inserts_extra_arguments_after_it(tmp_path: Path):
    path = _cache_config(tmp_path, _args_config())

    assert _rewrite(tmp_path, ["--region", "us-east-1"]) == []

    args = json.loads(path.read_text())["mcpServers"]["aws"]["args"]
    assert args == ["proxy", _TO, "--region", "us-east-1", "--skip-auth"]


def test_replaces_a_url_field_and_keeps_the_json_valid(tmp_path: Path):
    """An HTTP server holds the endpoint in a field, not in a command's arguments."""
    path = _cache_config(tmp_path, {"mcpServers": {"aws": {"type": "http", "url": _FROM}}})

    assert _rewrite(tmp_path) == []

    assert json.loads(path.read_text())["mcpServers"]["aws"]["url"] == _TO


def test_extra_arguments_cannot_go_into_a_url_field(tmp_path: Path):
    """There is nowhere to put command arguments, so it fails instead of writing bad JSON."""
    original = {"mcpServers": {"aws": {"type": "http", "url": _FROM}}}
    path = _cache_config(tmp_path, original)

    errors = _rewrite(tmp_path, ["--region", "us-east-1"])

    assert any("cannot be inserted" in e for e in errors)
    assert json.loads(path.read_text()) == original


def test_only_the_plugin_cache_is_rewritten(tmp_path: Path):
    """Marketplace sources under plugins/marketplaces are not installed configs."""
    _cache_config(tmp_path, _args_config())
    source = _config(tmp_path, "plugins/marketplaces/market/sample/.mcp.json", _args_config())

    assert _rewrite(tmp_path) == []

    assert json.loads(source.read_text()) == _args_config()


def test_a_new_endpoint_that_extends_the_old_one_is_not_stale(tmp_path: Path):
    """Comparing parsed values, not text: `.../mcp` is a prefix of `.../mcp/v2`."""
    path = _cache_config(tmp_path, _args_config())

    assert _rewrite(tmp_path, endpoint_to=f"{_FROM}/v2") == []

    assert json.loads(path.read_text())["mcpServers"]["aws"]["args"][1] == f"{_FROM}/v2"


def test_running_it_twice_is_a_no_op(tmp_path: Path):
    _cache_config(tmp_path, _args_config())
    assert _rewrite(tmp_path) == []
    assert _rewrite(tmp_path) == []


def test_every_cached_version_is_rewritten(tmp_path: Path):
    paths = [
        _cache_config(tmp_path, _args_config(), version="1.0.0"),
        _cache_config(tmp_path, _args_config(), name="mcp.json", version="1.1.0"),
    ]

    assert _rewrite(tmp_path) == []

    for path in paths:
        assert json.loads(path.read_text())["mcpServers"]["aws"]["args"][1] == _TO


def test_a_leftover_mention_of_the_old_endpoint_fails(tmp_path: Path):
    """An endpoint embedded in a longer value is not a whole-value match, so it must fail."""
    _cache_config(tmp_path, {"args": [_FROM], "env": {"URL": f"--endpoint={_FROM}/other"}})

    errors = _rewrite(tmp_path)

    assert any("still names" in e for e in errors)


def test_fails_when_no_config_names_the_new_endpoint(tmp_path: Path):
    _cache_config(tmp_path, {"args": ["proxy", "https://unrelated.example/mcp"]})

    errors = _rewrite(tmp_path)

    assert any("names" in e and _TO in e for e in errors)


def test_fails_on_an_unreadable_config_and_writes_nothing(tmp_path: Path):
    good = _cache_config(tmp_path, _args_config(), version="1.0.0")
    _cache_config(tmp_path, "{not json", name="mcp.json", version="1.1.0")

    errors = _rewrite(tmp_path)

    assert any("cannot be read as JSON" in e for e in errors)
    assert json.loads(good.read_text()) == _args_config()  # all or nothing


def test_fails_when_there_is_no_cache(tmp_path: Path):
    assert _rewrite(tmp_path) != []


# ── the command run in the container ──


def _run(root: Path, args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    command = build_endpoint_rewrite_command(_FROM, _TO, args or [], str(root))
    return subprocess.run(["bash", "-c", command], capture_output=True, text=True)


def test_command_is_valid_shell(tmp_path: Path):
    command = build_endpoint_rewrite_command(_FROM, _TO, ["--region", "us-east-1"], str(tmp_path))
    assert subprocess.run(["bash", "-n", "-c", command]).returncode == 0


def test_command_rewrites_the_cache_and_reports_success(tmp_path: Path):
    path = _cache_config(tmp_path, _args_config())

    result = _run(tmp_path, ["--region", "us-east-1"])

    assert result.returncode == 0, result.stderr
    assert f"MCP_ENDPOINT_OK {_TO}" in result.stdout
    assert json.loads(path.read_text())["mcpServers"]["aws"]["args"][1:4] == [
        _TO,
        "--region",
        "us-east-1",
    ]


def test_command_exits_non_zero_when_the_rewrite_cannot_take_effect(tmp_path: Path):
    _cache_config(tmp_path, {"args": ["proxy", "https://unrelated.example/mcp"]})

    result = _run(tmp_path)

    assert result.returncode != 0
    assert "MCP endpoint rewrite failed" in result.stderr


def test_command_passes_values_without_shell_interpretation(tmp_path: Path):
    """URLs and arguments go in as argv, so quotes, `$` and `&` are just characters."""
    tricky = "https://preprod.example/mcp?a=1&b='2'$HOME"
    path = _cache_config(tmp_path, _args_config())
    command = build_endpoint_rewrite_command(_FROM, tricky, ["--x", 'a "b" $c'], str(tmp_path))

    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert json.loads(path.read_text())["mcpServers"]["aws"]["args"][1:4] == [
        tricky,
        "--x",
        'a "b" $c',
    ]
