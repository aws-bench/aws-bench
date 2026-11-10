"""Plugin-capable Claude Code agent for aws-bench.

Extends Harbor's ``claude-code`` agent so a headless ``claude --print`` run can load
Claude Code plugins in its trial container before the task runs. A plugin bundles MCP
servers and skills together, so loading one makes both available to the agent.

Plugin sources are constructor kwargs, so they flow from ``--ak`` on the run CLI. With no
plugin source, no plugin setup runs.

From a marketplace, which is a git repository, a hosted ``marketplace.json`` or a directory::

    -a claude-code \
      --ak marketplaces='["anthropics/claude-plugins-official"]' \
      --ak plugins='["aws-core@claude-plugins-official"]'

Without a marketplace, from a plugin directory, a ``.zip`` archive of one, or the URL of
such an archive::

    --ak plugin_dirs='["~/work/my-plugin"]' \
    --ak plugin_urls='["https://example.com/builds/my-plugin.zip"]'

A marketplace directory or a ``plugin_dirs`` path that exists on the machine that runs
aws-bench is copied into each trial container. Any other path is used as a path in the
container, for example under a bind mount. Write a marketplace directory as a path
(``/dir``, ``./dir``, ``../dir`` or ``~/dir``), because Claude Code reads ``owner/repo`` as
a GitHub repository. A URL archive is downloaded once per trial, and each archive is
unpacked with ``unzip`` before the task runs. The trial container must have ``jq``, and
``unzip`` when an archive is used. Setup commands go to the trial log, so do not put a
long-lived secret in a URL.

Setup fails when a plugin does not install, download, unpack or load. Setup also fails in
two cases where Claude Code reports no error. A ``plugin_dirs`` or ``plugin_urls`` plugin
must have a manifest (``.claude-plugin/plugin.json``): Claude Code loads some folders
without one, such as a marketplace root, as an empty plugin. Two plugins must not have the
same manifest name, because Claude Code then leaves out one of them or its hooks.

The MCP configs of plugins installed from a git or hosted marketplace can be pointed at
another endpoint, for example a pre-production stage, without republishing the plugin::

    --ak mcp_endpoint_from=https://prod.example/mcp \
    --ak mcp_endpoint_to=https://preprod.example/mcp \
    --ak mcp_endpoint_args='["--region", "us-east-1"]'

Setup fails if any installed plugin config still names ``mcp_endpoint_from``
afterwards, so a trial never silently runs against the original endpoint. The rewrite
changes the plugin cache. Claude Code does not read the cache for a ``plugin_dirs`` or
``plugin_urls`` plugin, nor for a plugin that a marketplace directory holds in a subfolder,
so the rewrite is refused together with ``plugin_dirs``, ``plugin_urls`` or a marketplace
directory.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

from harbor.agents.installed.claude_code import ClaudeCode as _HarborClaudeCode
from harbor.environments.base import BaseEnvironment

from aws_bench.agents.plugin_endpoint import (
    build_endpoint_rewrite_command,
    validate_endpoint_override,
)

# claude installs to ~/.local/bin, which is off the PATH for the non-interactive
# exec shell; prepend it so `claude plugin` commands resolve.
_PATH_RESTORE = 'export PATH="$HOME/.local/bin:$PATH"'

# Container folder for the plugin sources that install() copies or downloads.
_SOURCES_DIR = "/installed-agent/claude-plugin-sources"

# jq test on one entry of `claude plugin list --json`: it is enabled and has no load error.
_JQ_ENTRY_LOADED = ".enabled and ((.errors // []) | length == 0)"
# jq test on the list: there is at least one plugin, and every one loaded. The command exits 0
# also when a plugin fails to load.
_JQ_LOADED = f"length > 0 and all({_JQ_ENTRY_LOADED})"
# The same test on the plugins of one plugin dir; the list also has the installed ones.
_JQ_PLUGIN_DIR_LOADED = f'map(select(.scope == "session")) | {_JQ_LOADED}'
# jq program: for each plugin, the folder that Claude Code reads it from, then the name in its id.
_JQ_FOLDERS_AND_ID_NAMES = '.[] | .readFromFolder // .installPath, (.id | split("@")[0])'
# jq program on lines of text: the lines that occur more than once.
_JQ_REPEATED_LINES = (
    'split("\\n") | group_by(.) | map(select(length > 1 and .[0] != "")[0]) | join(", ")'
)
# What a failure message shows of the list, with the entries that did not load first. Harbor
# keeps the first 1000 characters of a failed command's output, and one full entry can be
# longer.
_JQ_SUMMARY = f"map({{id, scope, enabled, errors}}) | sort_by({_JQ_ENTRY_LOADED})"


def _as_list(kwarg: str, value: list[str] | str | None) -> list[str]:
    """Copy the value of a list kwarg; a bare ``--ak kwarg=value`` arrives as one ``str``.

    Raises:
        ValueError: If an item is not a string, or is empty, as an unset shell variable makes it.
    """
    if value is None:
        return []
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise ValueError(f"{kwarg} must be a string or a list of strings")
    if "" in items:
        raise ValueError(f"{kwarg} has an empty item")
    return list(items)


def _is_path(source: str) -> bool:
    """Whether Claude Code reads a marketplace source as a path, not as a slug or a URL."""
    return source.startswith(("/", "./", "../", "~"))


def _host_path(source: str) -> Path | None:
    """Return the absolute path that ``source`` names on this machine, or None."""
    # Unlike pathlib, os.path does not raise for an unreadable folder or an unknown `~user`.
    path = Path(os.path.expanduser(source))
    return path.resolve() if os.path.exists(path) else None


def _plugin_list_step(*dirs: str) -> str:
    """Build shell that puts ``claude plugin list --json`` in the variable ``out``.

    Each of ``dirs`` goes after ``--plugin-dir`` unchanged, so quote it for the shell first.
    """
    flags = "".join(f"--plugin-dir {word} " for word in dirs)
    return (
        'command -v jq >/dev/null || { echo "jq not found" >&2; exit 1; }; '
        f"out=$(claude {flags}plugin list --json) || exit 1"
    )


def _fail_step(message: str) -> str:
    """Build shell that prints ``message`` and a summary of the list in ``out``, then exits 1."""
    return (
        f"{{ entries=$(printf '%s' \"$out\" | jq -c {shlex.quote(_JQ_SUMMARY)} 2>/dev/null); "
        f'echo "{message}: ${{entries:-$out}}" >&2; exit 1; }}'
    )


def _require_step(jq_test: str, message: str) -> str:
    """Build shell that exits 1 with ``message`` unless the list in ``out`` passes ``jq_test``."""
    return (
        f"printf '%s' \"$out\" | jq -e {shlex.quote(jq_test)} >/dev/null || {_fail_step(message)}"
    )


def _distinct_names_step() -> str:
    """Build shell that exits 1 if the list in ``out`` has a repeated or unreadable name.

    Claude Code names a plugin by its manifest, or by the name in its id when it has no
    manifest. Of an installed plugin and a ``--plugin-dir`` plugin with one name it loads only
    the ``--plugin-dir`` one. Of two installed or two ``--plugin-dir`` plugins with one name it
    loads both but runs the hooks of only one. It reports no error.
    """
    return (
        "repeated=$(set -o pipefail; "
        f"printf '%s' \"$out\" | jq -r {shlex.quote(_JQ_FOLDERS_AND_ID_NAMES)} | "
        "while IFS= read -r plugin && IFS= read -r name; do "
        'jq -r --arg name "$name" \'.name // $name\' "$plugin/.claude-plugin/plugin.json" '
        '2>/dev/null || echo "$name"; done | '
        f"jq -Rsr {shlex.quote(_JQ_REPEATED_LINES)}) || "
        f"{_fail_step('The plugin names cannot be read')}; "
        f'[ -z "$repeated" ] || {_fail_step("More than one plugin has the name $repeated")}'
    )


class ClaudeCode(_HarborClaudeCode):
    """Claude Code that loads plugins from marketplaces, folders, archives and URLs."""

    def __init__(
        self,
        logs_dir: Path,
        marketplaces: list[str] | str | None = None,
        plugins: list[str] | str | None = None,
        plugin_dirs: list[str] | str | None = None,
        plugin_urls: list[str] | str | None = None,
        mcp_endpoint_from: str | None = None,
        mcp_endpoint_to: str | None = None,
        mcp_endpoint_args: list[str] | None = None,
        *args,
        **kwargs,
    ):
        """Store the plugin sources and decide where the container finds each one.

        Args:
            logs_dir: Trial logs directory, forwarded to the base agent.
            marketplaces: Marketplace sources to add: an ``owner/repo`` slug, a git clone
                URL, the URL of a hosted ``marketplace.json``, or a directory written as a
                path that starts with ``/``, ``./``, ``../`` or ``~``.
            plugins: Plugin specs to install, each ``<plugin>@<marketplace>``.
            plugin_dirs: Plugins to load without a marketplace: a plugin directory, a
                ``.zip`` archive of one, or a folder of plugin directories. Only the
                directories with a manifest load from such a folder, and setup does not
                fail for the others.
            plugin_urls: URLs of plugin ``.zip`` archives to load without a marketplace.
            mcp_endpoint_from: MCP endpoint URL to replace in the installed
                plugins' MCP configs.
            mcp_endpoint_to: Replacement endpoint URL.
            mcp_endpoint_args: Extra arguments inserted after the replacement
                URL, for a proxy that cannot infer them from the new hostname.
            *args: Forwarded to the base agent.
            **kwargs: Forwarded to the base agent.
        """
        self._marketplaces = _as_list("marketplaces", marketplaces)
        self._plugins = _as_list("plugins", plugins)
        if self._marketplaces and not self._plugins:
            raise ValueError(
                "marketplaces were given without plugins; a marketplace is only "
                "added when a plugin from it is installed"
            )
        # (host path, container path) of each source that install() copies into the container.
        self._uploads: list[tuple[Path, str]] = []
        # Shell that install() runs after the copies, to download and unpack the archives. Claude
        # Code reads a folder in place, but unpacks an archive into a new temporary folder for
        # every command, so the load check would inspect a different copy than the session.
        self._stage_commands: list[str] = []
        self._marketplaces_to_add = [
            self._plan_marketplace(index, source) for index, source in enumerate(self._marketplaces)
        ]
        # Container folders for `--plugin-dir`: the plugin_dirs, then the plugin_urls.
        dirs = _as_list("plugin_dirs", plugin_dirs)
        self._container_plugin_dirs = [
            self._plan_plugin_dir(index, source) for index, source in enumerate(dirs)
        ]
        for index, url in enumerate(_as_list("plugin_urls", plugin_urls), start=len(dirs)):
            # curl reads a value that does not start with a scheme, such as `my-plugin.zip`, as
            # a host name.
            if not re.match(r"[A-Za-z][A-Za-z0-9+.-]*://", url):
                raise ValueError(
                    f"plugin_urls item {url} has no scheme such as https://; "
                    "give a path in plugin_dirs"
                )
            folder = f"{_SOURCES_DIR}/plugins/{index}"
            # Without --globoff, curl reads `[]` and `{}` in a URL as a set of URLs.
            self._stage_commands.append(f"curl -fsSL -o {folder}.zip --globoff {shlex.quote(url)}")
            self._stage_commands.append(f"unzip -q {folder}.zip -d {folder}")
            self._container_plugin_dirs.append(folder)

        if mcp_endpoint_to and (
            self._container_plugin_dirs or any(map(_is_path, self._marketplaces))
        ):
            raise ValueError(
                "the MCP endpoint rewrite changes the plugin cache, which Claude Code does not "
                "read for plugin_dirs, plugin_urls or the plugins inside a marketplace directory; "
                "change the endpoint in that plugin's .mcp.json instead"
            )
        self._mcp_endpoint_from = mcp_endpoint_from
        self._mcp_endpoint_to = mcp_endpoint_to
        self._mcp_endpoint_args = validate_endpoint_override(
            mcp_endpoint_from, mcp_endpoint_to, mcp_endpoint_args, bool(self._plugins)
        )
        super().__init__(logs_dir, *args, **kwargs)

    def _plan_marketplace(self, index: int, source: str) -> str:
        """Return the source for ``marketplace add``; plan a copy of a host directory."""
        host = _host_path(source) if _is_path(source) else None
        if host is None:
            return source
        # Claude Code also takes the path of the manifest; the marketplace is the folder around it.
        path = Path(os.path.expanduser(source))
        if path.match(".claude-plugin/marketplace.json"):
            host = path.parent.parent.resolve()
        # Fail early: the setup command hides the error of a `marketplace add` that fails.
        if not os.path.isfile(host / ".claude-plugin" / "marketplace.json"):
            raise ValueError(
                f"marketplace {source} must be a folder that holds a readable "
                ".claude-plugin/marketplace.json, or the path of that file"
            )
        target = f"{_SOURCES_DIR}/marketplaces/{index}"
        self._uploads.append((host, target))
        return target

    def _plan_plugin_dir(self, index: int, source: str) -> str:
        """Return the folder for ``--plugin-dir``; plan the copy and the unpack it needs."""
        folder = f"{_SOURCES_DIR}/plugins/{index}"
        unzip = f"unzip -q {folder}.zip -d {folder}"
        host = _host_path(source)
        if host and host.is_dir():
            self._uploads.append((host, folder))
        elif host:
            self._uploads.append((host, f"{folder}.zip"))
            self._stage_commands.append(unzip)
        elif source.endswith(".zip"):
            # unzip can read the name of an archive as a pattern and unpack another file, so a
            # container archive is unpacked from a copy with a plain name.
            self._stage_commands.append(f"cp {shlex.quote(source)} {folder}.zip && {unzip}")
        else:
            return source
        return folder

    async def install(self, environment: BaseEnvironment) -> None:
        """Install Claude Code and the plugin sources, plus git when plugins are requested.

        Claude Code shells out to ``git`` to clone a plugin marketplace, but the
        agent base image does not ship it. Install it during agent setup using
        whichever package manager the image provides. The container has no GitHub
        SSH key, so GitHub SSH remotes are rewritten to anonymous HTTPS for the
        public marketplace clone.
        """
        await super().install(environment)
        await self._stage_plugin_sources(environment)

        if not self._plugins:
            return

        await self.exec_as_root(
            environment,
            command=(
                "if ! command -v git >/dev/null 2>&1; then "
                "if command -v apt-get >/dev/null 2>&1; then "
                "apt-get update >/dev/null 2>&1 && apt-get install -y git >/dev/null 2>&1; "
                "elif command -v apk >/dev/null 2>&1; then "
                "apk add --no-cache git >/dev/null 2>&1; "
                "elif command -v yum >/dev/null 2>&1; then "
                "yum install -y git >/dev/null 2>&1; "
                "fi; fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_agent(
            environment,
            command=('git config --global url."https://github.com/".insteadOf "git@github.com:"'),
        )

    async def _stage_plugin_sources(self, environment: BaseEnvironment) -> None:
        """Copy host plugin sources into the container, then download and unpack archives.

        A step that fails stops agent setup. The files are made readable for an agent that
        does not run as root.
        """
        if not (self._uploads or self._stage_commands):
            return
        # Harbor's `docker compose cp` upload does not create missing parent folders.
        await self.exec_as_root(
            environment, command=f"mkdir -p {_SOURCES_DIR}/marketplaces {_SOURCES_DIR}/plugins"
        )
        for host, target in self._uploads:
            if host.is_dir():
                await environment.upload_dir(source_dir=host, target_dir=target)
            else:
                await environment.upload_file(source_path=host, target_path=target)
        for command in self._stage_commands:
            await self.exec_as_root(environment, command=command)
        await self.exec_as_root(environment, command=f"chmod -R a+rX {_SOURCES_DIR}")

    def build_cli_flags(self) -> str:
        """Add one ``--plugin-dir`` for each plugin directory and archive."""
        flags = [f"--plugin-dir {shlex.quote(path)}" for path in self._container_plugin_dirs]
        return " ".join(filter(None, [super().build_cli_flags(), *flags]))

    def _build_register_mcp_servers_command(self) -> str | None:
        """Chain plugin setup onto the per-trial config setup command.

        Harbor sets ``CLAUDE_CONFIG_DIR`` for this command, so the marketplace, plugin
        cache, and ``enabledPlugins`` entry land in the per-trial config directory. The
        steps are joined with ``&&``: a plugin that does not install or load aborts setup,
        and the task does not run without it.
        """
        steps = [super()._build_register_mcp_servers_command()]
        if self._plugins or self._container_plugin_dirs:
            steps.append(_PATH_RESTORE)
        steps += [
            f"(claude plugin marketplace add {shlex.quote(source)} 2>/dev/null || true)"
            for source in self._marketplaces_to_add
        ]
        steps += [
            f"claude plugin install {shlex.quote(spec)} --scope user" for spec in self._plugins
        ]
        if self._mcp_endpoint_to:
            steps.append(self._build_endpoint_rewrite_command())
        if self._container_plugin_dirs:
            steps.append(self._build_plugin_dir_check_command())
        if self._plugins or self._container_plugin_dirs:
            steps.append(self._build_plugin_list_check_command())
        return " && ".join(filter(None, steps)) or None

    def _build_endpoint_rewrite_command(self) -> str:
        """Rewrite the endpoint in the plugins ``claude plugin install`` just cached."""
        assert self._mcp_endpoint_from and self._mcp_endpoint_to
        return build_endpoint_rewrite_command(
            self._mcp_endpoint_from,
            self._mcp_endpoint_to,
            self._mcp_endpoint_args,
            "${CLAUDE_CONFIG_DIR:-$HOME/.claude}",
        )

    def _build_plugin_list_check_command(self) -> str:
        """Fail unless every plugin of the trial loaded and no two have the same name.

        ``plugin install`` also exits 0 for a plugin that then loads with errors or not at
        all. For example, a plugin with a hooks file that is not valid JSON loads without
        its hooks.
        """
        list_step = _plugin_list_step(*map(shlex.quote, self._container_plugin_dirs))
        loaded_step = _require_step(_JQ_LOADED, "A plugin did not load")
        return f"( {list_step}; {loaded_step}; {_distinct_names_step()} )"

    def _build_plugin_dir_check_command(self) -> str:
        """Fail unless every plugin dir loads without error and each of its plugins has a manifest.

        Each plugin dir is loaded alone, so one that loads nothing is not hidden by
        another one.
        """
        dirs = " ".join(shlex.quote(path) for path in self._container_plugin_dirs)
        steps = [
            _plugin_list_step('"$dir"'),
            _require_step(_JQ_PLUGIN_DIR_LOADED, "Plugin dir $dir did not load"),
            # The `exit` in the loop ends only the subshell of the pipeline; `|| exit 1` passes
            # the failure on.
            "printf '%s' \"$out\" | jq -r '.[] | select(.scope == \"session\") | .installPath' | "
            'while IFS= read -r plugin; do [ -f "$plugin/.claude-plugin/plugin.json" ] || '
            '{ echo "Plugin dir $dir has no .claude-plugin/plugin.json in $plugin" >&2; exit 1; }; '
            "done || exit 1",
        ]
        return f"( for dir in {dirs}; do {'; '.join(steps)}; done )"
