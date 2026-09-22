"""Scheduling and native credential-process checks with synthetic credentials."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from threading import Event
from unittest.mock import AsyncMock, MagicMock

import boto3
import botocore.session
import pytest
from botocore.client import BaseClient
from botocore.credentials import RefreshableCredentials

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.account_management.preexisting import ACCOUNT_CONFIG_ENV_VAR
from aws_bench.exceptions import CredentialError
from aws_bench.utils import credentials_provider
from aws_bench.utils.credentials_provider import (
    CREDENTIAL_ENV_VARS,
    CREDS_DIR,
    CredentialProvider,
    build_aws_config,
    credential_command,
    credential_env,
    credential_refresh_delay,
    mint_credentials,
    refresh_credentials_loop,
)


def test_refresh_timing_constants():
    assert credentials_provider._CRED_REFRESH_SKEW_SEC == 900
    assert credentials_provider.CRED_REFRESH_MIN_SLEEP_SEC == 30
    assert credentials_provider._CRED_REFRESH_RETRY_SEC == 60


@pytest.mark.parametrize(
    "remaining, expected", [(3600, 2700), (931, 31), (930, 30), (900, 30), (60, 30), (30.5, 30)]
)
@pytest.mark.parametrize("offset", [-7, 0, 5.5])
def test_refresh_delay_uses_remaining_lifetime_and_minimum(remaining, expected, offset):
    zone = timezone(timedelta(hours=offset))
    expiry = datetime.now(zone) + timedelta(seconds=remaining)
    delay = credential_refresh_delay(expiry)
    assert isinstance(delay, float)
    assert delay == pytest.approx(expected, abs=0.1)


@pytest.mark.parametrize("remaining", [-3600, -1, 0, 1, 29, 30])
def test_refresh_delay_rejects_expired_and_too_short_lifetimes(remaining):
    expiry = datetime.now(timezone.utc) + timedelta(seconds=remaining)
    with pytest.raises(CredentialError):
        credential_refresh_delay(expiry)


@pytest.mark.parametrize("expiry", [None, "2099-01-01T00:00:00Z", datetime(2099, 1, 1)])
def test_refresh_delay_rejects_missing_invalid_and_naive_expiry(expiry):
    with pytest.raises(CredentialError):
        credential_refresh_delay(expiry)


@pytest.mark.asyncio
async def test_refresh_loop_retries_without_an_extra_normal_delay(monkeypatch):
    initial = datetime.now(timezone.utc) + timedelta(hours=1)
    renewed = initial + timedelta(hours=1)
    sleeps = []

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(credentials_provider.asyncio, "sleep", sleep)
    publish = AsyncMock(
        side_effect=[
            RuntimeError("SYNTHETIC_SECRET"),
            RuntimeError("SYNTHETIC_SECRET"),
            renewed,
            asyncio.CancelledError(),
        ]
    )
    log = MagicMock()
    with pytest.raises(asyncio.CancelledError):
        await refresh_credentials_loop(initial, publish, log)
    assert sleeps == pytest.approx([2700, 60, 60, 6300], abs=0.1)
    assert publish.await_count == 4
    assert log.warning.call_count == 2
    assert "SYNTHETIC_SECRET" not in str(log.warning.call_args_list)


@pytest.mark.asyncio
async def test_refresh_loop_waits_for_successful_publication_before_rescheduling(monkeypatch):
    initial = datetime.now(timezone.utc) + timedelta(hours=1)
    renewed = initial + timedelta(hours=1)
    sleeps = []
    published = [initial]
    calls = 0

    async def sleep(delay):
        sleeps.append((delay, published[-1]))

    async def publish():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("publication failed")
        if calls == 2:
            published.append(renewed)
            return renewed
        raise asyncio.CancelledError()

    monkeypatch.setattr(credentials_provider.asyncio, "sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        await refresh_credentials_loop(initial, publish, MagicMock())
    assert [delay for delay, _ in sleeps] == pytest.approx([2700, 60, 6300], abs=0.1)
    assert [expiry for _, expiry in sleeps] == [initial, initial, renewed]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [None, datetime(2099, 1, 1), "not-an-expiry", "short"])
async def test_refresh_loop_retries_invalid_publication_expiry(monkeypatch, invalid):
    initial = datetime.now(timezone.utc) + timedelta(hours=1)
    if invalid == "short":
        invalid = datetime.now(timezone.utc) + timedelta(seconds=30)
    sleeps = []

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(credentials_provider.asyncio, "sleep", sleep)
    publish = AsyncMock(side_effect=[invalid, asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await refresh_credentials_loop(initial, publish, MagicMock())
    assert sleeps == pytest.approx([2700, 60], abs=0.1)


@pytest.mark.asyncio
async def test_refresh_loop_rejects_initial_expiry_before_sleep_or_publication(monkeypatch):
    sleep = AsyncMock()
    publish = AsyncMock()
    monkeypatch.setattr(credentials_provider.asyncio, "sleep", sleep)
    with pytest.raises(CredentialError):
        await refresh_credentials_loop(datetime.now(timezone.utc), publish, MagicMock())
    sleep.assert_not_awaited()
    publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_loop_cancellation_during_sleep_stops_without_publication(monkeypatch):
    sleeping = asyncio.Event()
    blocked = asyncio.Event()

    async def sleep(delay):
        sleeping.set()
        await blocked.wait()

    monkeypatch.setattr(credentials_provider.asyncio, "sleep", sleep)
    publish = AsyncMock()
    log = MagicMock()
    task = asyncio.create_task(
        refresh_credentials_loop(datetime.now(timezone.utc) + timedelta(hours=1), publish, log)
    )
    await asyncio.wait_for(sleeping.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    publish.assert_not_awaited()
    log.warning.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_loop_cancellation_during_publication_propagates(monkeypatch):
    started = asyncio.Event()
    cancelled = asyncio.Event()
    monkeypatch.setattr(credentials_provider, "credential_refresh_delay", lambda _: 0)

    async def publish():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return datetime.now(timezone.utc) + timedelta(hours=1)

    log = MagicMock()
    task = asyncio.create_task(refresh_credentials_loop(datetime.now(timezone.utc), publish, log))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    log.warning.assert_not_called()


@pytest.fixture
def synthetic_cloud(tmp_path, monkeypatch):
    """Keep credentials local and fail every unexpected cloud operation."""
    for key in (*CREDENTIAL_ENV_VARS, ACCOUNT_CONFIG_ENV_VAR):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "unused-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "unused-credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    state = {"calls": 0, "expiry": expiry}

    def call(client, operation, params):
        assert operation == "AssumeRole"
        assert params == {
            "RoleArn": "arn:aws:iam::111122223333:role/OrganizationAccountAccessRole",
            "RoleSessionName": "app-session",
        }
        state["calls"] += 1
        generation = state["calls"]
        return {
            "Credentials": {
                "AccessKeyId": f"SYNTHETIC_{generation}_KEY",
                "SecretAccessKey": f"SYNTHETIC_{generation}_SECRET",
                "SessionToken": f"SYNTHETIC_{generation}_TOKEN",
                "Expiration": state["expiry"],
            }
        }

    monkeypatch.setattr(BaseClient, "_make_api_call", call)
    host = boto3.Session(
        aws_access_key_id="SYNTHETIC_HOST_KEY",
        aws_secret_access_key="SYNTHETIC_HOST_SECRET",
        region_name="us-east-1",
    )
    return CredentialProvider(host), state


@pytest.mark.asyncio
async def test_cancelled_mint_cannot_publish_after_cleanup(synthetic_cloud, tmp_path, monkeypatch):
    provider, _ = synthetic_cloud
    started = Event()
    release = Event()
    finished = Event()
    target = tmp_path / "PRIMARY.json"
    monkeypatch.setattr(credentials_provider, "credential_refresh_delay", lambda _: 0)

    def mint():
        started.set()
        try:
            assert release.wait(5)
            return mint_credentials(
                provider, {"PRIMARY": "111122223333"}, ORG_ACCESS_ROLE, "app-session"
            )
        finally:
            finished.set()

    async def publish():
        files, expiry = await asyncio.to_thread(mint)
        target.write_text(files["PRIMARY.json"])
        return expiry

    task = asyncio.create_task(
        refresh_credentials_loop(datetime.now(timezone.utc), publish, MagicMock())
    )
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        target.unlink(missing_ok=True)
    finally:
        release.set()
    assert await asyncio.to_thread(finished.wait, 5)
    assert not target.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_cli", [False, True], ids=["sdk", "sdk-and-cli"])
async def test_shared_refresher_renews_a_persistent_native_process_consumer(
    synthetic_cloud, tmp_path, monkeypatch, use_cli
):
    aws = shutil.which("aws") if use_cli else None
    if use_cli:
        if aws is None:
            pytest.skip("AWS CLI v2 is not installed")
        version = subprocess.run([aws, "--version"], capture_output=True, text=True, timeout=10)
        if not version.stdout.startswith("aws-cli/2."):
            pytest.skip("AWS CLI v2 is required for configure export-credentials")
    provider, state = synthetic_cloud
    if use_cli:
        monkeypatch.setattr(credentials_provider, "CRED_REFRESH_MIN_SLEEP_SEC", 0.01)
        state["expiry"] = datetime.now(timezone.utc) + timedelta(seconds=10)
    home = tmp_path / "runner's home $literal"
    directory = home / CREDS_DIR
    directory.mkdir(parents=True, mode=0o700)
    (home / ".aws/config").write_text(build_aws_config(["PRIMARY"]))
    accounts = {"PRIMARY": "111122223333"}
    files, initial_expiry = mint_credentials(provider, accounts, ORG_ACCESS_ROLE, "app-session")

    def publish_files(contents):
        for name, body in contents.items():
            temporary = directory / f".{name}.tmp"
            temporary.write_text(body)
            temporary.chmod(0o600)
            temporary.replace(directory / name)

    publish_files(files)
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("sh", "cat"):
        executable = shutil.which(name)
        assert executable is not None
        (tools / name).symlink_to(executable)
    # The process command has only sh and cat available, with no application Python.
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(tools))
    core = botocore.session.Session()
    core.set_config_variable("config_file", str(home / ".aws/config"))
    core.set_config_variable("credentials_file", str(home / ".aws/credentials"))
    consumer = boto3.Session(botocore_session=core, profile_name="PRIMARY", region_name="us-east-1")
    credentials = consumer.get_credentials()
    assert isinstance(credentials, RefreshableCredentials)
    assert credentials.get_frozen_credentials().access_key == "SYNTHETIC_1_KEY"

    def cli_credentials():
        assert aws is not None
        command = credential_command(
            shlex.join([aws, "configure", "export-credentials", "--format", "process"]), "PRIMARY"
        )
        result = subprocess.run(
            [str(tools / "sh"), "-c", command],
            env=credential_env(
                "PRIMARY",
                {"HOME": str(home), "PATH": str(tools), "AWS_PAGER": ""},
            ),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return json.loads(result.stdout)

    if use_cli:
        assert cli_credentials()["AccessKeyId"] == "SYNTHETIC_1_KEY"
    assert not (home / ".aws/credentials").exists()
    state["expiry"] = initial_expiry + timedelta(hours=2)

    # Schedule a real loop tick promptly without changing the credential lifetime.
    monkeypatch.setattr(credentials_provider, "_CRED_REFRESH_SKEW_SEC", 3600)
    monkeypatch.setattr(credentials_provider, "CRED_REFRESH_MIN_SLEEP_SEC", 0.01)
    published = asyncio.Event()

    async def refresh_once():
        new_files, expiry = await asyncio.to_thread(
            mint_credentials, provider, accounts, ORG_ACCESS_ROLE, "app-session"
        )
        publish_files(new_files)
        published.set()
        return expiry

    task = asyncio.create_task(
        refresh_credentials_loop(initial_expiry, refresh_once, logging.getLogger(__name__))
    )
    try:
        await asyncio.wait_for(published.wait(), timeout=5)
        # This exact SDK object outlives its first generation, then re-runs the process.
        if use_cli:
            remaining = (initial_expiry - datetime.now(timezone.utc)).total_seconds()
            await asyncio.sleep(max(remaining, 0) + 0.02)
            assert datetime.now(timezone.utc) > initial_expiry
        else:
            credentials._time_fetcher = lambda: initial_expiry + timedelta(seconds=1)
        frozen = await asyncio.to_thread(credentials.get_frozen_credentials)
        assert frozen.access_key == "SYNTHETIC_2_KEY"
        assert frozen.secret_key == "SYNTHETIC_2_SECRET"
        assert frozen.token == "SYNTHETIC_2_TOKEN"
        assert consumer.get_credentials() is credentials
        assert credentials._expiry_time == state["expiry"]
        assert (
            json.loads((directory / "PRIMARY.json").read_text())["AccessKeyId"] == frozen.access_key
        )
        if use_cli:
            assert cli_credentials()["AccessKeyId"] == "SYNTHETIC_2_KEY"
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert state["calls"] == 2
    assert sorted(path.name for path in directory.iterdir()) == ["PRIMARY.json"]
