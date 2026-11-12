"""Tests for the artifact file selection of run directories."""

from __future__ import annotations

from pathlib import Path

from aws_bench.metrics.run_data import iter_artifact_files


def _touch(root: Path, *paths: str) -> None:
    for rel in paths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")


def _uploaded(run_dir: Path) -> set[tuple[str, str | None]]:
    return {
        (path.relative_to(run_dir).as_posix(), artifact_dir)
        for path, artifact_dir in iter_artifact_files(run_dir)
    }


def test_skips_claude_code_plugin_cache_and_marketplaces(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        "result.json",
        "trial__a/result.json",
        "trial__a/agent/trajectory.json",
        "trial__a/agent/sessions/projects/-app/session.jsonl",
        "trial__a/agent/sessions/plugins/installed_plugins.json",
        "trial__a/agent/sessions/plugins/cache/official/aws-core/1.1.0/plugin.json",
        "trial__a/agent/sessions/plugins/marketplaces/official/.git/HEAD",
        "trial__a/agent/sessions/plugins/marketplaces/official/README.md",
    )

    assert _uploaded(tmp_path) == {
        ("result.json", None),
        ("trial__a/result.json", "trial__a"),
        ("trial__a/agent/trajectory.json", "trial__a/agent"),
        (
            "trial__a/agent/sessions/projects/-app/session.jsonl",
            "trial__a/agent/sessions/projects/-app",
        ),
        (
            "trial__a/agent/sessions/plugins/installed_plugins.json",
            "trial__a/agent/sessions/plugins",
        ),
    }


def test_keeps_same_named_directories_elsewhere(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        "agent/sessions/plugins/cache/run-level.txt",
        "trial__a/artifacts/agent/sessions/plugins/cache/file.txt",
        "trial__a/agent/cache/file.txt",
    )

    assert {path for path, _ in _uploaded(tmp_path)} == {
        "agent/sessions/plugins/cache/run-level.txt",
        "trial__a/artifacts/agent/sessions/plugins/cache/file.txt",
        "trial__a/agent/cache/file.txt",
    }
