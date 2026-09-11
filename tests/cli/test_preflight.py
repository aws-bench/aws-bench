"""Tests for cli/preflight.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aws_bench.cli.preflight import (
    PreflightError,
    preflight_aws_credentials,
    preflight_docker_cli,
    preflight_docker_daemon,
    preflight_docker_plugins,
)


def _daemon_proc(stdout: bytes) -> MagicMock:
    return MagicMock(returncode=0, stdout=stdout, stderr=b"")


def test_preflight_docker_cli_succeeds_when_docker_on_path():
    with patch("aws_bench.cli.preflight.shutil.which", return_value="/usr/local/bin/docker"):
        preflight_docker_cli()  # no raise


def test_preflight_docker_cli_raises_when_docker_missing():
    with patch("aws_bench.cli.preflight.shutil.which", return_value=None):
        with pytest.raises(PreflightError, match="docker"):
            preflight_docker_cli()


def test_preflight_docker_daemon_succeeds_on_zero_exit():
    proc = _daemon_proc(b"buildx v0.36.1\ncompose v5.5.0\n")
    with patch("aws_bench.cli.preflight.subprocess.run", return_value=proc):
        plugins = preflight_docker_daemon()
    assert plugins == {"buildx": "v0.36.1", "compose": "v5.5.0"}


def test_preflight_docker_daemon_raises_on_nonzero_exit():
    proc = MagicMock(returncode=1, stderr=b"Cannot connect to the Docker daemon")
    with patch("aws_bench.cli.preflight.subprocess.run", return_value=proc):
        with pytest.raises(PreflightError, match="daemon"):
            preflight_docker_daemon()


def test_preflight_docker_plugins_passes_when_both_present_and_current():
    preflight_docker_plugins({"buildx": "v0.36.1", "compose": "v5.5.0"})  # no raise


def test_preflight_docker_plugins_raises_when_compose_missing():
    with pytest.raises(PreflightError, match="Compose"):
        preflight_docker_plugins({"buildx": "v0.36.1"})


def test_preflight_docker_plugins_raises_when_buildx_missing():
    with pytest.raises(PreflightError, match="buildx"):
        preflight_docker_plugins({"compose": "v5.5.0"})


def test_preflight_docker_plugins_raises_when_buildx_below_floor():
    with pytest.raises(PreflightError, match="0.17.0"):
        preflight_docker_plugins({"buildx": "v0.12.1", "compose": "v5.5.0"})


def test_preflight_docker_plugins_buildx_v090_below_v0170_is_numeric_not_lexical():
    # Regression guard: as strings "v0.9.0" > "v0.17.0", so a lexical compare
    # would wrongly pass. The floor must be enforced numerically.
    with pytest.raises(PreflightError, match="0.17.0"):
        preflight_docker_plugins({"buildx": "v0.9.0", "compose": "v5.5.0"})


def test_preflight_docker_plugins_buildx_exactly_at_floor_passes():
    preflight_docker_plugins({"buildx": "v0.17.0", "compose": "v5.5.0"})  # no raise


def test_preflight_docker_plugins_unparseable_buildx_warns_and_passes(caplog):
    with caplog.at_level("WARNING"):
        preflight_docker_plugins({"buildx": "buildx-dev", "compose": "v5.5.0"})  # no raise
    assert "buildx" in caplog.text


def test_preflight_docker_plugins_missing_buildx_version_warns_and_passes(caplog):
    with caplog.at_level("WARNING"):
        preflight_docker_plugins({"buildx": "", "compose": "v5.5.0"})  # no raise
    assert "buildx" in caplog.text


def test_preflight_aws_credentials_succeeds_on_get_caller_identity():
    cred = MagicMock()
    cred.session.client.return_value.get_caller_identity.return_value = {
        "Account": "111",
        "Arn": "arn:aws:sts::111:assumed-role/Admin/me",
    }
    assert preflight_aws_credentials(cred) == "111"


def test_preflight_aws_credentials_logs_identity_with_ou(caplog):
    cred = MagicMock()
    cred.session.client.return_value.get_caller_identity.return_value = {
        "Account": "111",
        "Arn": "arn:aws:sts::111:assumed-role/Admin/me",
    }
    with caplog.at_level("INFO"):
        preflight_aws_credentials(cred, ou_name="my-ou")
    assert "111" in caplog.text
    assert "my-ou" in caplog.text


def test_preflight_aws_credentials_raises_on_sts_error():
    cred = MagicMock()
    cred.session.client.return_value.get_caller_identity.side_effect = Exception("expired")
    with pytest.raises(PreflightError, match="credentials"):
        preflight_aws_credentials(cred)
