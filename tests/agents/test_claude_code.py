"""Tests for the plugin-capable Claude Code agent."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from harbor.agents.factory import AgentFactory
from harbor.models.agent.name import AgentName

from aws_bench.agents.claude_code import ClaudeCode

_MARKETPLACE = "anthropics/claude-plugins-official"
_PLUGIN = "aws-core@claude-plugins-official"
_SOURCES = "/installed-agent/claude-plugin-sources"
_URL = "https://example.com/builds/aws-core.zip?sig=a&b"
# Paths in the trial container. The adapter copies a path that exists on the machine that
# runs it, so these must not exist where the tests run.
_CONTAINER = "/only-in-the-container"
_CONTAINER_MARKETPLACE = f"{_CONTAINER}/agent-toolkit"
_CONTAINER_PLUGIN = f"{_CONTAINER}/aws-core"


@pytest.fixture(autouse=True)
def _no_flags_from_the_host_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Harbor fills an unset CLI flag from a host variable such as CLAUDE_CODE_EFFORT_LEVEL."""
    for descriptor in [*ClaudeCode.CLI_FLAGS, *ClaudeCode.ENV_VARS]:
        if descriptor.env_fallback:
            monkeypatch.delenv(descriptor.env_fallback, raising=False)


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs"


@pytest.fixture
def plugin_agent(logs_dir: Path) -> ClaudeCode:
    return ClaudeCode(logs_dir=logs_dir, marketplaces=[_MARKETPLACE], plugins=[_PLUGIN])


def _plugin_folder(folder: Path, name: str) -> Path:
    """Make ``folder`` a plugin root whose manifest has ``name``."""
    manifest = folder / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": name}))
    return folder


@pytest.fixture
def plugin_dir(tmp_path: Path) -> Path:
    """A plugin root on the host."""
    return _plugin_folder(tmp_path / "aws-core", "aws-core")


@pytest.fixture
def marketplace_dir(tmp_path: Path) -> Path:
    """A marketplace root on the host."""
    manifest = tmp_path / "toolkit" / ".claude-plugin" / "marketplace.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"name": "toolkit", "plugins": []}')
    return manifest.parent.parent


def _fresh_environment() -> MagicMock:
    environment = MagicMock()
    environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
    environment.upload_dir = AsyncMock()
    environment.upload_file = AsyncMock()
    return environment


def _commands(environment: MagicMock) -> list[str]:
    return [c.kwargs.get("command", "") for c in environment.exec.call_args_list]


# ── name ──


def test_name_is_claude_code():
    """Keeps the built-in name so it overrides Harbor's claude-code."""
    assert ClaudeCode.name() == "claude-code"


# ── factory override ──


def test_factory_resolves_claude_code_to_subclass():
    """Importing aws_bench.agents maps `claude-code` to this subclass."""
    import aws_bench.agents  # noqa: F401

    assert AgentFactory._AGENT_MAP[AgentName("claude-code")] is ClaudeCode


def test_subclass_replaces_builtin_in_agents_list():
    """The subclass replaces Harbor's claude-code in the factory list."""
    import aws_bench.agents  # noqa: F401

    claude_entries = [a for a in AgentFactory._AGENTS if a.name() == "claude-code"]
    assert claude_entries == [ClaudeCode]


# ── init ──


def test_defaults_to_no_plugins(logs_dir: Path):
    """Defaults to no marketplaces or plugins."""
    agent = ClaudeCode(logs_dir=logs_dir)
    assert agent._marketplaces == []
    assert agent._plugins == []


def test_stores_marketplaces_and_plugins(plugin_agent: ClaudeCode):
    """Stores the supplied marketplaces and plugins."""
    assert plugin_agent._marketplaces == [_MARKETPLACE]
    assert plugin_agent._plugins == [_PLUGIN]


def test_none_normalizes_to_empty_list(logs_dir: Path):
    """Normalizes explicit None to empty lists."""
    agent = ClaudeCode(logs_dir=logs_dir, marketplaces=None, plugins=None)
    assert agent._marketplaces == []
    assert agent._plugins == []


def test_marketplaces_without_plugins_raises(logs_dir: Path):
    """A marketplace with no plugin to install is a no-op, so reject it loudly."""
    with pytest.raises(ValueError, match="marketplaces were given without plugins"):
        ClaudeCode(logs_dir=logs_dir, marketplaces=[_MARKETPLACE], plugins=None)


# ── MCP / plugin setup command ──


def test_no_plugins_passes_through_to_base(logs_dir: Path):
    """Returns the base command (None here) when no plugins are requested."""
    agent = ClaudeCode(logs_dir=logs_dir)
    assert agent._build_register_mcp_servers_command() is None


def test_installs_each_plugin(plugin_agent: ClaudeCode):
    """Installs each requested plugin user-scoped."""
    cmd = plugin_agent._build_register_mcp_servers_command()
    assert cmd is not None
    assert f"claude plugin install {_PLUGIN} --scope user" in cmd


def test_adds_each_marketplace(plugin_agent: ClaudeCode):
    """Adds each requested marketplace."""
    cmd = plugin_agent._build_register_mcp_servers_command()
    assert cmd is not None
    assert f"claude plugin marketplace add {_MARKETPLACE}" in cmd


def test_restores_path(plugin_agent: ClaudeCode):
    """Restores PATH so claude resolves under the minimal exec PATH."""
    cmd = plugin_agent._build_register_mcp_servers_command()
    assert cmd is not None
    assert "export PATH=" in cmd


def test_adds_marketplace_before_installing_plugin(plugin_agent: ClaudeCode):
    """Adds the marketplace before installing the plugin that needs it."""
    cmd = plugin_agent._build_register_mcp_servers_command()
    assert cmd is not None
    assert cmd.index("marketplace add") < cmd.index("plugin install")


def test_supports_multiple_plugins(logs_dir: Path):
    """Adds every marketplace and installs every plugin."""
    agent = ClaudeCode(
        logs_dir=logs_dir,
        marketplaces=["a/one", "b/two"],
        plugins=["p1@one", "p2@two"],
    )
    cmd = agent._build_register_mcp_servers_command()
    assert cmd is not None
    assert "claude plugin install p1@one --scope user" in cmd
    assert "claude plugin install p2@two --scope user" in cmd
    assert "marketplace add a/one" in cmd
    assert "marketplace add b/two" in cmd


def test_chains_after_base_mcp_servers(logs_dir: Path):
    """Writes task MCP servers first, then chains plugin install with &&."""
    stdio = MagicMock(transport="stdio", command="run", args=[])
    stdio.name = "srv"
    agent = ClaudeCode(logs_dir=logs_dir, plugins=[_PLUGIN], mcp_servers=[stdio])
    cmd = agent._build_register_mcp_servers_command()
    assert cmd is not None
    assert ".claude.json" in cmd
    assert cmd.index(".claude.json") < cmd.index("plugin install")
    assert " && " in cmd


# ── install ──


@pytest.mark.asyncio
async def test_no_plugins_skips_git_install(logs_dir: Path):
    """Skips git install when no plugins are requested."""
    agent = ClaudeCode(logs_dir=logs_dir)
    environment = _fresh_environment()

    await agent.install(environment)

    assert not any("command -v git" in c for c in _commands(environment))


@pytest.mark.asyncio
async def test_installs_git_when_plugins_requested(plugin_agent: ClaudeCode):
    """Installs git during setup when plugins are requested."""
    environment = _fresh_environment()

    await plugin_agent.install(environment)

    assert any("command -v git" in c for c in _commands(environment))


@pytest.mark.asyncio
async def test_rewrites_github_ssh_to_https(plugin_agent: ClaudeCode):
    """Rewrites GitHub SSH remotes to anonymous HTTPS for the clone."""
    environment = _fresh_environment()

    await plugin_agent.install(environment)

    assert any("insteadOf" in c and "git@github.com:" in c for c in _commands(environment))


# ── MCP endpoint override ──

_FROM = "https://prod.example/mcp"
_TO = "https://preprod.example/mcp"


def _override_agent(logs_dir: Path) -> ClaudeCode:
    return ClaudeCode(
        logs_dir=logs_dir,
        marketplaces=[_MARKETPLACE],
        plugins=[_PLUGIN],
        mcp_endpoint_from=_FROM,
        mcp_endpoint_to=_TO,
    )


def test_endpoint_override_is_off_by_default(plugin_agent: ClaudeCode):
    """Without the arguments, no rewrite is appended to the plugin install."""
    cmd = plugin_agent._build_register_mcp_servers_command()
    assert cmd is not None
    assert "MCP_ENDPOINT_OK" not in cmd


def test_endpoint_override_runs_after_the_plugin_install(logs_dir: Path):
    """The rewrite must follow `claude plugin install`, which is what writes the config."""
    cmd = _override_agent(logs_dir)._build_register_mcp_servers_command()
    assert cmd is not None
    assert cmd.rindex("claude plugin install") < cmd.index("MCP_ENDPOINT_OK")
    assert "CLAUDE_CONFIG_DIR" in cmd
    assert subprocess.run(["bash", "-n", "-c", cmd]).returncode == 0


def test_endpoint_override_validates_its_arguments(logs_dir: Path):
    """The shared validation applies; its cases are covered in test_plugin_endpoint."""
    with pytest.raises(ValueError, match="given together"):
        ClaudeCode(logs_dir=logs_dir, plugins=[_PLUGIN], mcp_endpoint_from=_FROM)


@pytest.mark.parametrize(
    "uncached_source",
    [
        {"plugin_dirs": [_CONTAINER_PLUGIN]},
        {"plugin_urls": [_URL]},
        {"marketplaces": [_MARKETPLACE, _CONTAINER_MARKETPLACE]},
    ],
)
def test_endpoint_override_rejects_a_plugin_that_is_not_in_the_cache(
    logs_dir: Path, uncached_source: dict[str, list[str]]
):
    """The rewrite changes the plugin cache, which Claude Code does not read for these sources."""
    options = {"marketplaces": [_MARKETPLACE], "plugins": [_PLUGIN], **uncached_source}
    with pytest.raises(ValueError, match="plugin cache"):
        ClaudeCode(logs_dir=logs_dir, mcp_endpoint_from=_FROM, mcp_endpoint_to=_TO, **options)


# ── list arguments ──


def test_a_bare_string_is_one_item(logs_dir: Path):
    """`--ak plugins=name@market` arrives as a str and must not split into characters."""
    agent = ClaudeCode(
        logs_dir=logs_dir,
        marketplaces=_MARKETPLACE,
        plugins=_PLUGIN,
        plugin_dirs=_CONTAINER_PLUGIN,
        plugin_urls=_URL,
    )
    assert agent._marketplaces == [_MARKETPLACE]
    assert agent._plugins == [_PLUGIN]
    assert agent._container_plugin_dirs == [_CONTAINER_PLUGIN, f"{_SOURCES}/plugins/1"]


# ── plugin directories, archives and URLs ──


def test_no_plugin_dirs_leaves_the_cli_flags_unchanged(plugin_agent: ClaudeCode):
    assert plugin_agent.build_cli_flags() == ""


def test_a_path_that_is_not_on_the_host_is_a_container_path(logs_dir: Path):
    """For example a plugin that the trial container already has, or one under a bind mount."""
    agent = ClaudeCode(logs_dir=logs_dir, plugin_dirs=[f"{_CONTAINER}/my plugin"])
    assert agent.build_cli_flags() == f"--plugin-dir '{_CONTAINER}/my plugin'"


def test_a_plugin_url_that_does_not_start_with_a_scheme_is_refused(logs_dir: Path):
    """Curl reads ``aws-core.zip`` as the host name, and ``.zip`` is a top-level domain."""
    with pytest.raises(ValueError, match="plugin_urls item .* has no scheme"):
        ClaudeCode(logs_dir=logs_dir, plugin_urls=["aws-core.zip?from=https://example.com"])


@pytest.mark.parametrize("kwarg", ["marketplaces", "plugins", "plugin_dirs", "plugin_urls"])
@pytest.mark.parametrize("value", ["", ["a/b", ""]])
def test_an_empty_item_is_refused(logs_dir: Path, kwarg: str, value: Any):
    """An unset shell variable in `--ak name=$VALUE` gives an empty item."""
    with pytest.raises(ValueError, match=f"{kwarg} has an empty item"):
        ClaudeCode(logs_dir=logs_dir, **{kwarg: value})


@pytest.mark.parametrize("kwarg", ["marketplaces", "plugins", "plugin_dirs", "plugin_urls"])
@pytest.mark.parametrize("value", [[None], ["a/b", 1], 5, False, {"a/b": 1}])
def test_a_list_kwarg_must_hold_strings(logs_dir: Path, kwarg: str, value: Any):
    """JSON from `--ak` can give other types, which would fail later with an unrelated error."""
    with pytest.raises(ValueError, match=f"{kwarg} must be a string or a list of strings"):
        ClaudeCode(logs_dir=logs_dir, **{kwarg: value})


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read every folder")
def test_a_path_that_this_user_cannot_read_is_a_container_path(logs_dir: Path, tmp_path: Path):
    """For example `/root/plugin` or `~other-user/plugin`, which exist only in the container."""
    locked = tmp_path / "locked"
    locked.mkdir(mode=0)
    try:
        agent = ClaudeCode(
            logs_dir=logs_dir, plugin_dirs=[f"{locked}/plugin", "~no-such-user/plugin"]
        )
    finally:
        locked.chmod(0o700)
    assert agent._container_plugin_dirs == [f"{locked}/plugin", "~no-such-user/plugin"]


def test_a_host_plugin_dir_is_loaded_from_its_container_copy(logs_dir: Path, plugin_dir: Path):
    """The flag follows the flags of the base agent."""
    agent = ClaudeCode(logs_dir=logs_dir, plugin_dirs=str(plugin_dir), max_turns=5)
    assert agent.build_cli_flags() == f"--max-turns 5 --plugin-dir {_SOURCES}/plugins/0"


# ── local marketplaces ──


def test_a_host_marketplace_is_added_by_its_container_copy(logs_dir: Path, marketplace_dir: Path):
    agent = ClaudeCode(
        logs_dir=logs_dir, marketplaces=[str(marketplace_dir)], plugins=["aws-core@toolkit"]
    )
    cmd = agent._build_register_mcp_servers_command()
    assert cmd is not None
    assert f"claude plugin marketplace add {_SOURCES}/marketplaces/0 " in cmd
    assert str(marketplace_dir) not in cmd


def test_a_marketplace_path_that_is_not_on_the_host_is_passed_through(logs_dir: Path):
    """A bind-mounted marketplace is named by its container path."""
    agent = ClaudeCode(
        logs_dir=logs_dir, marketplaces=[_CONTAINER_MARKETPLACE], plugins=["aws-core@toolkit"]
    )
    cmd = agent._build_register_mcp_servers_command()
    assert cmd is not None
    assert f"claude plugin marketplace add {_CONTAINER_MARKETPLACE} " in cmd


@pytest.mark.asyncio
async def test_an_owner_repo_slug_is_never_a_host_path(
    logs_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Claude Code reads `owner/repo` as GitHub, also when a folder of that name exists."""
    (tmp_path / "owner" / "repo").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    agent = ClaudeCode(logs_dir=logs_dir, marketplaces=["owner/repo"], plugins=["p@repo"])
    environment = _fresh_environment()

    await agent.install(environment)

    environment.upload_dir.assert_not_awaited()
    cmd = agent._build_register_mcp_servers_command()
    assert cmd is not None
    assert "claude plugin marketplace add owner/repo " in cmd


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spelling",
    [
        "./",
        "../toolkit",
        "~/toolkit",
        "./.claude-plugin/marketplace.json",
        "~/toolkit/.claude-plugin//marketplace.json/",  # a doubled and a trailing slash
    ],
)
async def test_a_host_marketplace_is_copied_however_its_path_is_written(
    logs_dir: Path, marketplace_dir: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
):
    """Each spelling is found on the host, as the folder or as the path of its manifest."""
    monkeypatch.chdir(marketplace_dir)
    monkeypatch.setenv("HOME", str(marketplace_dir.parent))
    agent = ClaudeCode(logs_dir=logs_dir, marketplaces=[spelling], plugins=["aws-core@toolkit"])
    environment = _fresh_environment()

    await agent.install(environment)

    environment.upload_dir.assert_awaited_once_with(
        source_dir=marketplace_dir.resolve(), target_dir=f"{_SOURCES}/marketplaces/0"
    )


@pytest.mark.asyncio
async def test_a_host_marketplace_can_be_named_by_a_manifest_that_is_a_link(
    logs_dir: Path, marketplace_dir: Path
):
    """A manifest that is a relative link inside the marketplace stays valid in the copy."""
    manifest = marketplace_dir / ".claude-plugin" / "marketplace.json"
    manifest.rename(marketplace_dir / "catalog.json")
    manifest.symlink_to("../catalog.json")
    agent = ClaudeCode(logs_dir=logs_dir, marketplaces=[str(manifest)], plugins=["a@toolkit"])
    environment = _fresh_environment()

    await agent.install(environment)

    environment.upload_dir.assert_awaited_once_with(
        source_dir=marketplace_dir.resolve(), target_dir=f"{_SOURCES}/marketplaces/0"
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read every folder")
def test_a_host_marketplace_that_this_user_cannot_read_is_refused(
    logs_dir: Path, marketplace_dir: Path
):
    manifest_folder = marketplace_dir / ".claude-plugin"
    manifest_folder.chmod(0)
    try:
        with pytest.raises(ValueError, match="must be a folder that holds"):
            ClaudeCode(logs_dir=logs_dir, marketplaces=[str(marketplace_dir)], plugins=["a@b"])
    finally:
        manifest_folder.chmod(0o700)


@pytest.mark.parametrize("not_a_marketplace", [".claude-plugin", "marketplace.json"])
def test_a_host_path_that_is_not_a_marketplace_is_refused(
    logs_dir: Path, marketplace_dir: Path, not_a_marketplace: str
):
    """A host path must be the marketplace folder or the manifest in its `.claude-plugin`."""
    (marketplace_dir / "marketplace.json").touch()
    source = str(marketplace_dir / not_a_marketplace)
    with pytest.raises(ValueError, match="must be a folder that holds a readable"):
        ClaudeCode(logs_dir=logs_dir, marketplaces=[source], plugins=["aws-core@toolkit"])


# ── staging during install ──


@pytest.mark.asyncio
async def test_install_copies_host_sources_into_the_container(
    logs_dir: Path, tmp_path: Path, plugin_dir: Path, marketplace_dir: Path
):
    archive = tmp_path / "build.zip"
    archive.touch()
    agent = ClaudeCode(
        logs_dir=logs_dir,
        marketplaces=[_MARKETPLACE, str(marketplace_dir)],
        plugins=[_PLUGIN, "aws-core@toolkit"],
        plugin_dirs=[str(plugin_dir), str(archive)],
    )
    environment = _fresh_environment()

    await agent.install(environment)

    assert environment.upload_dir.await_args_list == [
        ((), {"source_dir": marketplace_dir, "target_dir": f"{_SOURCES}/marketplaces/1"}),
        ((), {"source_dir": plugin_dir, "target_dir": f"{_SOURCES}/plugins/0"}),
    ]
    environment.upload_file.assert_awaited_once_with(
        source_path=archive, target_path=f"{_SOURCES}/plugins/1.zip"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("spelling", ["./aws-core", "aws-core", "~/aws-core"])
async def test_install_resolves_a_relative_host_path(
    logs_dir: Path, plugin_dir: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
):
    """The upload runs later, possibly from another working directory."""
    monkeypatch.chdir(plugin_dir.parent)
    monkeypatch.setenv("HOME", str(plugin_dir.parent))
    agent = ClaudeCode(logs_dir=logs_dir, plugin_dirs=[spelling])
    environment = _fresh_environment()

    await agent.install(environment)

    environment.upload_dir.assert_awaited_once_with(
        source_dir=plugin_dir.resolve(), target_dir=f"{_SOURCES}/plugins/0"
    )


@pytest.mark.asyncio
async def test_install_makes_the_parent_folders_before_it_copies(logs_dir: Path, plugin_dir: Path):
    """Harbor's `docker compose cp` upload does not create missing parent folders."""
    agent = ClaudeCode(logs_dir=logs_dir, plugin_dirs=[str(plugin_dir)])
    environment = _fresh_environment()

    await agent.install(environment)

    steps = [
        "upload" if name == "upload_dir" else kwargs["command"]
        for name, _, kwargs in environment.mock_calls
    ]
    parents = f"mkdir -p {_SOURCES}/marketplaces {_SOURCES}/plugins"
    mkdir = next(i for i, step in enumerate(steps) if parents in step)
    readable = next(i for i, step in enumerate(steps) if f"chmod -R a+rX {_SOURCES}" in step)
    assert mkdir < steps.index("upload") < readable


@pytest.mark.asyncio
async def test_install_downloads_and_unpacks_each_archive(logs_dir: Path, tmp_path: Path):
    """A host, a container and a URL archive each become a folder that is read in place."""
    archive = tmp_path / "build.zip"
    archive.touch()
    agent = ClaudeCode(
        logs_dir=logs_dir,
        plugin_dirs=[str(archive), f"{_CONTAINER}/aws core[1].zip"],
        plugin_urls=[_URL],
    )
    environment = _fresh_environment()

    await agent.install(environment)

    assert agent.build_cli_flags() == " ".join(
        f"--plugin-dir {_SOURCES}/plugins/{index}" for index in range(3)
    )
    commands = [c.removeprefix("set -o pipefail; ") for c in _commands(environment)]
    curl = f"curl -fsSL -o {_SOURCES}/plugins/2.zip --globoff '{_URL}'"
    assert [c for c in commands if c.startswith("curl")] == [curl]
    assert [c for c in commands if "unzip -q" in c] == [
        f"unzip -q {_SOURCES}/plugins/0.zip -d {_SOURCES}/plugins/0",
        # The unzip tool would read `[1]` as a pattern, so it gets a copy with a plain name.
        f"cp '{_CONTAINER}/aws core[1].zip' {_SOURCES}/plugins/1.zip"
        f" && unzip -q {_SOURCES}/plugins/1.zip -d {_SOURCES}/plugins/1",
        f"unzip -q {_SOURCES}/plugins/2.zip -d {_SOURCES}/plugins/2",
    ]
    unpack = commands.index(f"unzip -q {_SOURCES}/plugins/2.zip -d {_SOURCES}/plugins/2")
    readable = next(i for i, c in enumerate(commands) if c.startswith("chmod"))
    assert commands.index(curl) < unpack < readable


@pytest.mark.asyncio
@pytest.mark.parametrize(("step", "code"), [("aws-core.zip", 22), ("unzip", 9)])
async def test_a_failed_download_or_unpack_fails_install(logs_dir: Path, step: str, code: int):
    """A URL has nothing to upload; its download and its unpack must still run and stop install."""
    agent = ClaudeCode(logs_dir=logs_dir, plugin_urls=[_URL])
    environment = _fresh_environment()
    environment.exec.side_effect = lambda command, **_: MagicMock(
        return_code=code if step in command else 0, stdout="", stderr=""
    )

    with pytest.raises(RuntimeError, match=f"exit {code}"):
        await agent.install(environment)


@pytest.mark.asyncio
async def test_install_stages_nothing_without_local_sources(logs_dir: Path):
    """A git marketplace and a container path need no copy."""
    agent = ClaudeCode(
        logs_dir=logs_dir,
        marketplaces=[_MARKETPLACE, _CONTAINER_MARKETPLACE],
        plugins=[_PLUGIN],
        plugin_dirs=[_CONTAINER_PLUGIN],
    )
    environment = _fresh_environment()

    await agent.install(environment)

    environment.upload_dir.assert_not_awaited()
    environment.upload_file.assert_not_awaited()
    assert not any(_SOURCES in c for c in _commands(environment))


# ── the setup command, run with a fake claude ──


@pytest.fixture
def fake_claude(tmp_path: Path) -> dict[str, str]:
    """Shell environment with a fake `claude`, and only `bash` and `jq` on the PATH.

    The fake records its argv, fails on $FAIL_ARG and lists $PLUGIN_LIST. When $PLUGIN_LIST_BY_DIRS
    has an entry for its `--plugin-dir` values, joined with a space, it lists that entry.
    """
    home = tmp_path / "home"
    executable = home / ".local" / "bin" / "claude"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if os.environ.get('FAIL_ARG') in sys.argv:\n"
        "    sys.exit(42)\n"
        "if 'list' in sys.argv:\n"
        "    lists = json.loads(os.environ.get('PLUGIN_LIST_BY_DIRS', '{}'))\n"
        "    args = sys.argv\n"
        "    dirs = ' '.join(args[i + 1] for i, a in enumerate(args) if a == '--plugin-dir')\n"
        "    print(json.dumps(lists[dirs]) if dirs in lists else os.environ['PLUGIN_LIST'])\n"
    )
    executable.chmod(0o755)
    tools = tmp_path / "tools"
    tools.mkdir()
    for tool in ("bash", "jq"):
        (tools / tool).symlink_to(str(shutil.which(tool)))
    calls = tmp_path / "calls.jsonl"
    calls.touch()
    return {
        **os.environ,
        "PATH": str(tools),
        "HOME": str(home),
        "CALLS": str(calls),
        "PLUGIN_LIST": "[]",
    }


def _run(agent: ClaudeCode, env: dict[str, str], **extra: str) -> subprocess.CompletedProcess[str]:
    command = agent._build_register_mcp_servers_command()
    assert command is not None
    return subprocess.run(
        ["bash", "-c", command], env={**env, **extra}, capture_output=True, text=True
    )


def _calls(env: dict[str, str]) -> list[list[str]]:
    return [json.loads(line) for line in Path(env["CALLS"]).read_text().splitlines()]


def test_a_failed_plugin_install_stops_setup(logs_dir: Path, fake_claude: dict[str, str]):
    """Every install must succeed; the second one must not hide a failure of the first."""
    agent = ClaudeCode(logs_dir=logs_dir, marketplaces=["a/one"], plugins=["p1@one", "p2@one"])

    result = _run(agent, fake_claude, FAIL_ARG="p1@one")

    assert result.returncode != 0
    assert ["plugin", "install", "p2@one", "--scope", "user"] not in _calls(fake_claude)


def _installed(**fields: object) -> dict[str, object]:
    """One entry of `claude plugin list --json` for a plugin installed from a marketplace."""
    return {"id": _PLUGIN, "scope": "user", "enabled": True, **fields}


def test_a_marketplace_url_with_a_query_string_is_one_argument(
    logs_dir: Path, fake_claude: dict[str, str]
):
    """Unquoted, the `&` of a signed URL would put the command in the background."""
    url = "https://example.com/marketplace.json?sig=a&b=2"
    agent = ClaudeCode(logs_dir=logs_dir, marketplaces=[url], plugins=[_PLUGIN])

    result = _run(agent, fake_claude, PLUGIN_LIST=json.dumps([_installed()]))

    assert result.returncode == 0, result.stderr
    assert _calls(fake_claude)[0] == ["plugin", "marketplace", "add", url]


def test_a_failed_marketplace_add_leaves_the_decision_to_the_install(
    plugin_agent: ClaudeCode, fake_claude: dict[str, str]
):
    listed = json.dumps([_installed()])

    result = _run(plugin_agent, fake_claude, FAIL_ARG=_MARKETPLACE, PLUGIN_LIST=listed)

    assert result.returncode == 0, result.stderr
    assert ["plugin", "install", _PLUGIN, "--scope", "user"] in _calls(fake_claude)


def test_setup_passes_when_the_installed_plugin_loads(
    plugin_agent: ClaudeCode, fake_claude: dict[str, str]
):
    result = _run(plugin_agent, fake_claude, PLUGIN_LIST=json.dumps([_installed()]))

    assert result.returncode == 0, result.stderr
    assert _calls(fake_claude)[-1] == ["plugin", "list", "--json"]


@pytest.mark.parametrize(
    ("listed", "message"),
    [
        ([], "did not load"),
        ([_installed(enabled=False)], "did not load"),
        # `plugin install` exits 0 for both: the first plugin loads without its hooks, the
        # second does not load at all.
        ([_installed(errors=["Hook load failed: JSON Parse error"])], "Hook load failed"),
        ([_installed(), _installed(errors=["failed to load: cache-miss"])], "cache-miss"),
    ],
)
def test_setup_fails_when_an_installed_plugin_does_not_load(
    plugin_agent: ClaudeCode,
    fake_claude: dict[str, str],
    listed: list[dict[str, object]],
    message: str,
):
    result = _run(plugin_agent, fake_claude, PLUGIN_LIST=json.dumps(listed))

    assert result.returncode != 0
    assert "A plugin did not load" in result.stderr
    assert message in result.stderr


def test_a_failure_message_starts_with_the_plugin_that_did_not_load(
    plugin_agent: ClaudeCode, fake_claude: dict[str, str]
):
    """Harbor keeps the first 1000 characters of the output, and one full entry can be longer."""
    servers = {"aws-mcp": {"args": ["x" * 1000]}}
    failed = _installed(errors=["Hook load failed"], mcpServers=servers)
    listed = [_installed(id="first@market", mcpServers=servers), failed]

    result = _run(plugin_agent, fake_claude, PLUGIN_LIST=json.dumps(listed))

    assert result.returncode != 0
    assert result.stderr.startswith(
        'A plugin did not load: [{"id":"aws-core@claude-plugins-official","scope":"user",'
        '"enabled":true,"errors":["Hook load failed"]},{"id":"first@market",'
    )
    assert len(result.stderr) < 300


def test_a_failure_message_shows_a_plugin_list_that_is_not_json(
    plugin_agent: ClaudeCode, fake_claude: dict[str, str]
):
    result = _run(plugin_agent, fake_claude, PLUGIN_LIST="Error: no settings")

    assert result.returncode != 0
    assert "A plugin did not load: Error: no settings" in result.stderr


def _from_plugin_dir(install_path: Path, **fields: object) -> dict[str, object]:
    """One entry of `claude plugin list --json` for a plugin loaded with --plugin-dir."""
    return {
        "id": "aws-core@inline",
        "scope": "session",
        "enabled": True,
        "installPath": str(install_path),
        **fields,
    }


def _load_check(
    logs_dir: Path, env: dict[str, str], listed: list[dict[str, object]]
) -> subprocess.CompletedProcess[str]:
    agent = ClaudeCode(logs_dir=logs_dir, plugin_dirs=[_CONTAINER_PLUGIN])
    return _run(agent, env, PLUGIN_LIST=json.dumps(listed))


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([], "did not load"),  # a folder with no plugin in it: no entry and no error
        ([{"enabled": False}], "did not load"),
        ([{"enabled": False, "errors": ["Path not found: /opt/x"]}], "Path not found: /opt/x"),
        ([{"errors": ["Path not found: hooks.json (hooks)"]}], "hooks.json"),  # loaded with errors
        ([{}, {"enabled": False}], "did not load"),  # a folder of plugins with one broken child
        ([{"scope": "user"}], "did not load"),  # an installed plugin is not from the plugin dir
    ],
)
def test_setup_fails_when_the_plugin_dir_does_not_load(
    logs_dir: Path,
    fake_claude: dict[str, str],
    plugin_dir: Path,
    entries: list[dict[str, object]],
    message: str,
):
    listed = [_from_plugin_dir(plugin_dir, **fields) for fields in entries]

    result = _load_check(logs_dir, fake_claude, listed)

    assert result.returncode != 0
    assert _CONTAINER_PLUGIN in result.stderr
    assert message in result.stderr


def test_setup_fails_when_a_loaded_folder_has_no_plugin_manifest(
    logs_dir: Path, fake_claude: dict[str, str], plugin_dir: Path, marketplace_dir: Path
):
    """Claude Code loads a marketplace root as a plugin with no skills and reports no error."""
    listed = [_from_plugin_dir(plugin_dir), _from_plugin_dir(marketplace_dir)]

    result = _load_check(logs_dir, fake_claude, listed)

    assert result.returncode != 0
    assert f"no .claude-plugin/plugin.json in {marketplace_dir}" in result.stderr


def test_the_manifest_rule_is_for_plugin_dirs_only(
    logs_dir: Path, fake_claude: dict[str, str], plugin_dir: Path, tmp_path: Path
):
    listed = [
        _installed(id="other@market", installPath=str(tmp_path)),
        _from_plugin_dir(plugin_dir),
    ]

    result = _load_check(logs_dir, fake_claude, listed)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("others", "name"),
    [
        ([_installed()], "aws-core"),  # Claude Code loads only the plugin of the plugin dir
        # Claude Code runs the hooks of only one of the two.
        ([_installed(id="shared@one"), _installed(id="shared@two")], "shared"),
    ],
)
def test_setup_fails_when_two_plugins_have_the_same_name(
    logs_dir: Path,
    fake_claude: dict[str, str],
    plugin_dir: Path,
    others: list[dict[str, object]],
    name: str,
):
    """Claude Code then leaves out one of them or its hooks, and reports no error."""
    result = _load_check(logs_dir, fake_claude, [_from_plugin_dir(plugin_dir), *others])

    assert result.returncode != 0
    assert f"More than one plugin has the name {name}: [" in result.stderr


def test_setup_fails_when_a_plugin_name_cannot_be_read(
    logs_dir: Path, fake_claude: dict[str, str], plugin_dir: Path
):
    """A list entry without an id must fail the name check, not let it pass silently."""
    listed = [_from_plugin_dir(plugin_dir), {"scope": "user", "enabled": True}]

    result = _load_check(logs_dir, fake_claude, listed)

    assert result.returncode != 0
    assert "The plugin names cannot be read: [" in result.stderr


def test_an_installed_plugin_is_named_by_the_manifest_in_the_folder_claude_code_reads(
    logs_dir: Path, fake_claude: dict[str, str], plugin_dir: Path
):
    """`readFromFolder` wins over `installPath`, and the id can differ from the manifest name."""
    entry = _installed(id="entry@market", installPath="/not-read", readFromFolder=str(plugin_dir))
    result = _load_check(logs_dir, fake_claude, [_from_plugin_dir(plugin_dir), entry])

    assert result.returncode != 0
    assert "More than one plugin has the name aws-core: [" in result.stderr


def test_a_plugin_is_named_by_its_manifest_and_not_by_its_id(
    logs_dir: Path, fake_claude: dict[str, str], plugin_dir: Path, tmp_path: Path
):
    """An installed plugin with another manifest name is another plugin, whatever its id says."""
    other = _plugin_folder(tmp_path / "other", "other")
    listed = [
        _installed(id="aws-core@market", installPath=str(other)),
        _from_plugin_dir(plugin_dir),
    ]

    result = _load_check(logs_dir, fake_claude, listed)

    assert result.returncode == 0, result.stderr


def test_two_plugin_dirs_are_also_listed_together(
    logs_dir: Path, fake_claude: dict[str, str], plugin_dir: Path
):
    """Only the list of all plugins shows that two plugin dirs give plugins of one name."""
    first, second = f"{_CONTAINER}/a", f"{_CONTAINER}/b"
    agent = ClaudeCode(logs_dir=logs_dir, plugin_dirs=[first, second])
    lists = {f"{first} {second}": [_from_plugin_dir(plugin_dir), _from_plugin_dir(plugin_dir)]}

    result = _run(
        agent,
        fake_claude,
        PLUGIN_LIST_BY_DIRS=json.dumps(lists),
        PLUGIN_LIST=json.dumps([_from_plugin_dir(plugin_dir)]),
    )

    assert result.returncode != 0
    assert "More than one plugin has the name aws-core" in result.stderr


def test_each_plugin_dir_is_checked_alone(
    logs_dir: Path, fake_claude: dict[str, str], plugin_dir: Path
):
    """One call per plugin dir, so one that loads nothing cannot hide, then one call for all."""
    first, second = f"{_CONTAINER}/a", f"{_CONTAINER}/b c"
    agent = ClaudeCode(logs_dir=logs_dir, plugin_dirs=[first, second])

    result = _run(agent, fake_claude, PLUGIN_LIST=json.dumps([_from_plugin_dir(plugin_dir)]))

    assert result.returncode == 0, result.stderr
    assert _calls(fake_claude) == [
        ["--plugin-dir", first, "plugin", "list", "--json"],
        ["--plugin-dir", second, "plugin", "list", "--json"],
        ["--plugin-dir", first, "--plugin-dir", second, "plugin", "list", "--json"],
    ]


def test_a_good_plugin_dir_does_not_hide_one_without_a_manifest(
    logs_dir: Path, fake_claude: dict[str, str], plugin_dir: Path, marketplace_dir: Path
):
    """The manifest check ends in a pipeline, so its `exit` alone does not stop the setup."""
    first, second = f"{_CONTAINER}/a", f"{_CONTAINER}/b"
    agent = ClaudeCode(logs_dir=logs_dir, plugin_dirs=[first, second])
    lists = {first: [_from_plugin_dir(marketplace_dir)], second: [_from_plugin_dir(plugin_dir)]}

    result = _run(agent, fake_claude, PLUGIN_LIST_BY_DIRS=json.dumps(lists))

    assert result.returncode != 0
    assert f"Plugin dir {first} has no .claude-plugin/plugin.json" in result.stderr
    assert _calls(fake_claude) == [["--plugin-dir", first, "plugin", "list", "--json"]]


def test_the_load_check_runs_after_the_plugin_installs(
    logs_dir: Path, fake_claude: dict[str, str], plugin_dir: Path
):
    agent = ClaudeCode(
        logs_dir=logs_dir,
        marketplaces=[_MARKETPLACE],
        plugins=[_PLUGIN],
        plugin_dirs=[_CONTAINER_PLUGIN],
    )

    result = _run(agent, fake_claude, PLUGIN_LIST=json.dumps([_from_plugin_dir(plugin_dir)]))

    assert result.returncode == 0, result.stderr
    assert [call[:2] for call in _calls(fake_claude)] == [
        ["plugin", "marketplace"],
        ["plugin", "install"],
        ["--plugin-dir", _CONTAINER_PLUGIN],
        ["--plugin-dir", _CONTAINER_PLUGIN],
    ]
