"""Tests for AwsBenchTrial — AWS credential injection + placeholder substitution.

Exercises the AwsBenchTrial lifecycle overrides: pre-invoke runs once after the
container starts, placeholders substitute into the agent instruction, agent creds
inject, verifier creds land at the precedence the verifier reads (then restore),
and post-invoke runs before the environment stops. The agent / environment / task
are faked down to what the overrides touch.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shlex
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.oracle import OracleAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.trial.errors import AgentTimeoutError
from harbor.trial.single_step import SingleStepTrial
from harbor.trial.trial import Trial

from aws_bench.dataset.models import RoleType, ScriptType
from aws_bench.dataset.task_config import AwsBenchTask
from aws_bench.exceptions import AccountContaminatedError, OperationCancelled
from aws_bench.task import aws_trial
from aws_bench.task.aws_trial import AwsBenchSingleStepTrial, AwsBenchTrial
from aws_bench.utils import credentials_provider
from aws_bench.utils.credentials_provider import CREDENTIAL_ENV_VARS, CredentialError

TASK_NAME = "org/my-task"
CREDS_PATH = PurePosixPath("/home/runner/.aws/creds")


@pytest.fixture(autouse=True)
def no_contamination(mocker):
    """Default the _prepare contamination gate to clean so tests make no AWS call.

    _prepare()'s gate calls AccountManager().get_contaminated_accounts, which hits
    real Organizations tagging. Return [] by default; the dedicated gate tests
    re-patch AccountManager to assert the blocked/allowed paths.
    """
    acct = mocker.MagicMock()
    acct.get_contaminated_accounts.return_value = []
    return mocker.patch("aws_bench.task.aws_trial.AccountManager", return_value=acct)


@pytest.fixture
def fake_creds(mocker):
    """Mint synthetic process credentials without making AWS calls."""
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    mocker.patch.object(aws_trial.CredentialProvider, "get", return_value=MagicMock())
    return mocker.patch.object(
        aws_trial,
        "mint_credentials",
        return_value=(
            {
                "PRIMARY.json": json.dumps(
                    {
                        "Version": 1,
                        "AccessKeyId": "AKIA",
                        "SecretAccessKey": "secret",
                        "SessionToken": "token",
                        "Expiration": expiry.isoformat(),
                    }
                )
            },
            expiry,
        ),
    )


def _scenario_ref(**roles):
    """A ScenarioRef-like stand-in exposing role_name(role_type)."""
    role_map = {
        RoleType.AGENT: roles.get("agent"),
        RoleType.VERIFIER: roles.get("verifier"),
        RoleType.PRE_INVOKE: roles.get("pre_invoke"),
        RoleType.POST_INVOKE: roles.get("post_invoke"),
    }
    return SimpleNamespace(role_name=lambda rt: role_map[rt])


def _phase(timeout_sec=None, env=None):
    return SimpleNamespace(timeout_sec=timeout_sec, env=env or {})


def _exec_calls(trial):
    """The recorded calls to the trial's mocked agent_environment.exec."""
    return trial.agent_environment.exec.call_args_list


def _environment():
    """Record Harbor operations and uploaded files, without a Docker dependency."""
    environment = MagicMock()
    environment.default_user = None
    environment.files = {}
    environment.uploads = []
    environment.publications = []
    environment.cleared_dirs = []
    environment.private_uploads = 0
    environment.mountinfo = "1 0 0:1 / / rw - overlay overlay rw\n"

    async def execute(*, command, env=None, user=None, **kwargs):
        stdout = ""
        if "pwd -P && id -u && id -g" in command:
            home = (env or {}).get("HOME", "/home/runner")
            effective_user = user if user is not None else environment.default_user
            uid = {"root": "0", "verifier": "1001"}.get(effective_user, "1000")
            stdout = f"{home}\n{uid}\n{uid}\n"
        elif "cat /proc/self/mountinfo" in command:
            stdout = environment.mountinfo
        elif "mktemp -d " in command:
            environment.private_uploads += 1
            stdout = str(
                aws_trial._CREDENTIAL_UPLOAD_ROOT / f".aws-creds.{environment.private_uploads:08d}"
            )
        elif "cat " in command and "; fi" in command:
            path = shlex.split(command.rsplit("cat ", 1)[1].split("; fi")[0])[0]
            stdout = environment.files.get(path, "")
        elif "mv -fT -- " in command:
            source, destination = shlex.split(command.rsplit("mv -fT -- ", 1)[1])
            tokens = shlex.split(command)
            directories = [
                tokens[index + 3]
                for index in range(len(tokens) - 3)
                if tokens[index : index + 3] == ["cd", "-P", "--"]
            ]
            source = str(PurePosixPath(directories[0]) / PurePosixPath(source).name)
            destination = str(PurePosixPath(directories[-1]) / destination)
            environment.files[destination] = environment.files.pop(source)
            environment.publications.append((destination, environment.files[destination]))
        elif "rm -rf -- " in command:
            directory = PurePosixPath(shlex.split(command.rsplit("rm -rf -- ", 1)[1])[0])
            environment.files = {
                path: body
                for path, body in environment.files.items()
                if not PurePosixPath(path).is_relative_to(directory)
            }
        return ExecResult(return_code=0, stdout=stdout, stderr="")

    async def upload(source_path, target_path):
        source = Path(source_path)
        body = source.read_text()
        environment.uploads.append((target_path, body, source.stat().st_mode & 0o777))
        environment.files[target_path] = body

    async def empty(directories, *, chmod=True):
        parent = aws_trial._CREDENTIAL_CWD.get()
        directories = [parent / path if parent is not None else path for path in directories]
        environment.cleared_dirs.extend(directories)
        for path in list(environment.files):
            if any(PurePosixPath(path).is_relative_to(directory) for directory in directories):
                del environment.files[path]
        return ExecResult(return_code=0, stdout="", stderr="")

    environment.with_default_user = MethodType(BaseEnvironment.with_default_user, environment)
    environment.exec = AsyncMock(side_effect=execute)
    environment.upload_file = AsyncMock(side_effect=upload)
    environment.empty_dirs = AsyncMock(side_effect=empty)
    return environment


def _make_trial(
    tmp_path,
    *,
    exports=None,
    pre_invoke=None,
    post_invoke=None,
    has_pre_script=False,
    has_post_script=False,
) -> Any:
    """Build an AwsBenchTrial with only the attributes the overrides touch.

    Bypasses __init__ (no Docker/agent factory) — sets the fields the AWS
    overrides read: self.config (account_mapping/exports/verifier/job_id/trial_name),
    self.task (name + config.{scenario,pre_invoke,post_invoke} + has_phase_script),
    self.agent (with _extra_env), self.agent_environment, self.paths, self.logger.

    ``exports`` is tag-keyed (``tag -> {name -> value}``), matching config.exports.
    """
    trial = AwsBenchSingleStepTrial.__new__(AwsBenchSingleStepTrial)

    task_config = SimpleNamespace(
        scenario=_scenario_ref(agent="AgentRole", verifier="VerifierRole"),
        pre_invoke=pre_invoke,
        post_invoke=post_invoke,
        verifier=SimpleNamespace(env={"REGION": "us-east-1"}, user=None),
        agent=SimpleNamespace(user=None),
    )
    task = SimpleNamespace(
        name=TASK_NAME,
        config=task_config,
        paths=SimpleNamespace(task_dir=tmp_path / "task"),
        has_phase_script=lambda st: (
            (st == ScriptType.PRE_INVOKE and has_pre_script)
            or (st == ScriptType.POST_INVOKE and has_post_script)
        ),
    )

    trial.config = SimpleNamespace(  # type: ignore[assignment]
        account_mapping={"PRIMARY": "123456789012"},
        regions=["us-east-1"],
        exports=exports or {},
        verifier=SimpleNamespace(env={"REGION": "us-east-1"}),
        job_id=None,
        trial_name="trial-0",
        verify_env=True,
    )
    trial.task = task  # type: ignore[assignment]
    # The real agents (oracle / installed) carry _extra_env; the cred-injection
    # override refuses an agent without it, so the double must expose one.
    trial.agent = SimpleNamespace(_extra_env={})  # type: ignore[assignment]
    trial.agent_environment = _environment()
    trial.paths = SimpleNamespace(trial_dir=tmp_path / "trial")  # type: ignore[assignment]
    trial.logger = MagicMock()
    # __init__ is bypassed here; set the per-trial state it would establish.
    trial._aws_placeholders = {tag: dict(v) for tag, v in (exports or {}).items()}
    trial._aws_post_invoke_done = False
    trial._credential_dir = None
    trial._credential_operation_task = None
    trial._credential_failed = False
    # __init__ builds one AccountManager and reuses it for the _prepare gate.
    # Bypassed here, so establish it too — resolves to the autouse no_contamination
    # mock (clean by default); the gate tests inject their own onto the instance.
    trial._account_manager = aws_trial.AccountManager()
    # Default to "container started" so post-invoke tests exercise the script;
    # the skip-when-not-started case sets this False explicitly.
    trial._agent_container_started = True
    return trial


# --- create dispatch -------------------------------------------------------


@pytest.mark.asyncio
async def test_create_dispatches_single_step(mocker):
    task = SimpleNamespace(has_steps=False)
    mocker.patch.object(AwsBenchTask, "from_config", AsyncMock(return_value=task))
    # Patch the base init out to skip the heavy Docker / agent-factory chain.
    mocker.patch.object(SingleStepTrial, "__init__", lambda self, config, _task=None: None)
    trial = await AwsBenchTrial.create(MagicMock())
    assert isinstance(trial, AwsBenchSingleStepTrial)


@pytest.mark.asyncio
async def test_create_multi_step_raises_not_implemented(mocker):
    task = SimpleNamespace(has_steps=True)
    mocker.patch.object(AwsBenchTask, "from_config", AsyncMock(return_value=task))
    with pytest.raises(NotImplementedError, match="multi-step"):
        await AwsBenchTrial.create(MagicMock())


# --- placeholder substitution into the instruction ------------------------


@pytest.mark.asyncio
async def test_run_agent_phase_substitutes_instruction(tmp_path, fake_creds, mocker):
    trial = _make_trial(tmp_path, exports={"PRIMARY": {"BucketName": "my-bucket"}})
    # Single account tag, so a bare {{BucketName}} resolves against the sole tag.
    trial._aws_placeholders = {"PRIMARY": {"BucketName": "my-bucket"}}

    seen = {}

    async def fake_super_phase(self, *, instruction, **kw):
        seen["instruction"] = instruction

    mocker.patch.object(Trial, "_run_agent_phase", fake_super_phase)
    await trial._run_agent_phase(
        target=MagicMock(), instruction="deploy to {{BucketName}}", timeout_sec=None, user=None
    )
    assert seen["instruction"] == "deploy to my-bucket"


async def _extra_env_during_run(trial, mocker) -> dict[str, str]:
    """Run the agent phase and return the agent env seen during the (mocked) run."""
    seen: dict[str, str] = {}

    async def fake_super_phase(self, *, instruction, **kw):
        seen.update(self.agent._extra_env)

    mocker.patch.object(Trial, "_run_agent_phase", fake_super_phase)
    await trial._run_agent_phase(target=MagicMock(), instruction="x", timeout_sec=None, user=None)
    return seen


def _as_oracle(trial) -> None:
    """Swap in a real OracleAgent whose _task.config is separate, as harbor's is."""
    oracle = OracleAgent.__new__(OracleAgent)
    oracle._extra_env = {}
    oracle._task = SimpleNamespace(  # type: ignore[assignment]
        config=SimpleNamespace(solution=SimpleNamespace(env=dict(trial.task.config.solution.env)))
    )
    trial.agent = oracle


@pytest.mark.asyncio
async def test_run_agent_phase_injects_solution_env_for_oracle(tmp_path, fake_creds, mocker):
    """The oracle receives [solution.env] with {{placeholder}} tokens resolved."""
    trial = _make_trial(tmp_path, exports={"PRIMARY": {"BucketName": "my-bucket"}})
    trial._aws_placeholders = {"PRIMARY": {"BucketName": "my-bucket"}}
    trial.task.config.solution = SimpleNamespace(  # type: ignore[assignment]
        env={"BUCKET_NAME": "{{BucketName}}"}
    )
    _as_oracle(trial)
    seen = await _extra_env_during_run(trial, mocker)
    assert seen["BUCKET_NAME"] == "my-bucket"


@pytest.mark.asyncio
async def test_run_agent_phase_blanks_oracle_solution_env_during_run(tmp_path, fake_creds, mocker):
    """The config object harbor reads is blanked during the run, then restored."""
    trial = _make_trial(tmp_path, exports={"PRIMARY": {"BucketName": "my-bucket"}})
    trial._aws_placeholders = {"PRIMARY": {"BucketName": "my-bucket"}}
    trial.task.config.solution = SimpleNamespace(  # type: ignore[assignment]
        env={"BUCKET_NAME": "{{BucketName}}"}
    )
    _as_oracle(trial)
    original = dict(trial.agent._task.config.solution.env)  # type: ignore[attr-defined]

    seen_oracle_env: dict[str, str] = {}

    async def fake_super_phase(self, *, instruction, **kw):
        # Read the SAME object harbor's OracleAgent.run reads.
        seen_oracle_env.update(self.agent._task.config.solution.env)  # type: ignore[attr-defined]

    mocker.patch.object(Trial, "_run_agent_phase", fake_super_phase)
    await trial._run_agent_phase(target=MagicMock(), instruction="x", timeout_sec=None, user=None)

    assert seen_oracle_env == {}  # oracle's own copy blanked during the run
    assert (
        trial.agent._task.config.solution.env == original  # type: ignore[attr-defined]
    )  # restored after


@pytest.mark.asyncio
async def test_run_agent_phase_creds_win_over_solution_env(tmp_path, fake_creds, mocker):
    """A [solution.env] key colliding with a cred var loses: the cred value wins."""
    trial = _make_trial(tmp_path, exports={"PRIMARY": {"BucketName": "my-bucket"}})
    trial._aws_placeholders = {"PRIMARY": {"BucketName": "my-bucket"}}
    trial.task.config.solution = SimpleNamespace(  # type: ignore[assignment]
        env={"AWS_SESSION_TOKEN": "{{BucketName}}"}
    )
    _as_oracle(trial)
    seen = await _extra_env_during_run(trial, mocker)
    # Staged creds empty the raw token; solution.env must not resurrect a stale one.
    assert seen["AWS_SESSION_TOKEN"] == ""


@pytest.mark.asyncio
async def test_run_agent_phase_no_solution_env_for_non_oracle(tmp_path, fake_creds, mocker):
    """A non-oracle agent never receives [solution.env] (would leak the answer)."""
    trial = _make_trial(tmp_path, exports={"PRIMARY": {"BucketName": "my-bucket"}})
    trial._aws_placeholders = {"PRIMARY": {"BucketName": "my-bucket"}}
    trial.task.config.solution = SimpleNamespace(  # type: ignore[assignment]
        env={"BUCKET_NAME": "{{BucketName}}"}
    )
    # Default fixture agent is a SimpleNamespace, not an OracleAgent.
    seen = await _extra_env_during_run(trial, mocker)
    assert "BUCKET_NAME" not in seen


@pytest.mark.asyncio
async def test_run_agent_phase_writes_creds_file_and_empties_raw_creds(
    tmp_path, fake_creds, mocker
):
    """Agent gets a creds file in-container; raw cred vars are emptied during the run."""
    trial = _make_trial(tmp_path)
    trial._aws_placeholders = {}

    seen: dict[str, str] = {}

    async def fake_super_phase(self, *, instruction, **kw):
        # The injected env is live only during the agent run; capture it here.
        seen.update(self.agent._extra_env)

    mocker.patch.object(Trial, "_run_agent_phase", fake_super_phase)
    await trial._run_agent_phase(target=MagicMock(), instruction="x", timeout_sec=None, user=None)

    # Raw creds are emptied so a host-forwarded set cannot outrank the file.
    assert seen["AWS_ACCESS_KEY_ID"] == ""
    assert seen["AWS_SESSION_TOKEN"] == ""
    # Secret contents travel through upload_file, never through shell arguments.
    write_cmds = [c.kwargs.get("command", "") for c in _exec_calls(trial)]
    assert all("AKIA" not in cmd and "secret" not in cmd for cmd in write_cmds)
    uploads = trial.agent_environment.uploads
    assert any(
        "PRIMARY.json" in path and '"AccessKeyId": "AKIA"' in body for path, body, _ in uploads
    )
    assert all(mode == 0o600 for _, _, mode in uploads)


@pytest.mark.asyncio
async def test_run_agent_phase_sets_aws_profile_to_first_tag(tmp_path, fake_creds, mocker):
    """AWS_PROFILE defaults to the first tag so an ambient-creds task is unchanged."""
    trial = _make_trial(tmp_path)
    trial._aws_placeholders = {}

    seen: dict[str, str] = {}

    async def fake_super_phase(self, *, instruction, **kw):
        seen.update(self.agent._extra_env)

    mocker.patch.object(Trial, "_run_agent_phase", fake_super_phase)
    await trial._run_agent_phase(target=MagicMock(), instruction="x", timeout_sec=None, user=None)
    assert seen["AWS_PROFILE"] == "PRIMARY"


@pytest.mark.asyncio
async def test_run_agent_phase_pins_region_to_first_scenario_region(tmp_path, fake_creds, mocker):
    """AWS_REGION/AWS_DEFAULT_REGION are pinned to the scenario's first region."""
    trial = _make_trial(tmp_path)
    trial.config.regions = ["eu-west-1", "us-east-1"]  # type: ignore[attr-defined]
    trial._aws_placeholders = {}

    seen: dict[str, str] = {}

    async def fake_super_phase(self, *, instruction, **kw):
        seen.update(self.agent._extra_env)

    mocker.patch.object(Trial, "_run_agent_phase", fake_super_phase)
    await trial._run_agent_phase(target=MagicMock(), instruction="x", timeout_sec=None, user=None)
    assert seen["AWS_REGION"] == "eu-west-1"
    assert seen["AWS_DEFAULT_REGION"] == "eu-west-1"


@pytest.mark.asyncio
async def test_run_agent_phase_restores_extra_env_after_run(tmp_path, fake_creds, mocker):
    """The injected cred env is removed from _extra_env once the agent run returns."""
    trial = _make_trial(tmp_path)
    trial._aws_placeholders = {}
    trial.agent._extra_env["PRESET"] = "keep"  # type: ignore[attr-defined]

    async def fake_super_phase(self, *, instruction, **kw):
        pass

    mocker.patch.object(Trial, "_run_agent_phase", fake_super_phase)
    await trial._run_agent_phase(target=MagicMock(), instruction="x", timeout_sec=None, user=None)

    # Pre-existing entries survive; the transient cred env does not.
    assert trial.agent._extra_env == {"PRESET": "keep"}  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_run_agent_phase_clears_creds_directory_after_run(tmp_path, fake_creds, mocker):
    """Both cleanup points clear the whole directory without Harbor's chmod."""
    trial = _make_trial(tmp_path)
    trial._aws_placeholders = {}

    async def fake_super_phase(self, *, instruction, **kw):
        pass

    mocker.patch.object(Trial, "_run_agent_phase", fake_super_phase)
    await trial._run_agent_phase(target=MagicMock(), instruction="x", timeout_sec=None, user=None)
    assert trial.agent_environment.cleared_dirs == [CREDS_PATH, CREDS_PATH]
    assert all(
        invocation.kwargs == {"chmod": False}
        for invocation in trial.agent_environment.empty_dirs.await_args_list
    )
    assert trial._credential_dir is None
    assert not any("/creds/" in path for path in trial.agent_environment.files)


@pytest.mark.asyncio
async def test_staged_credentials_raises_on_empty_account_mapping(tmp_path, fake_creds):
    """An empty account mapping cannot produce credentials, so fail before writing."""
    trial = _make_trial(tmp_path)
    trial.config.account_mapping = {}  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="empty account_mapping"):
        async with trial._staged_credentials(RoleType.AGENT):
            pass

    # Nothing was written when the mapping is empty.
    assert _exec_calls(trial) == []


@pytest.mark.asyncio
async def test_staged_credentials_raises_when_write_fails(tmp_path, fake_creds):
    """An upload error stops the phase and does not expose transport output."""
    trial = _make_trial(tmp_path)
    trial.agent_environment.upload_file.side_effect = RuntimeError("disk full: secret")

    with pytest.raises(RuntimeError, match="publish credential file") as error:
        async with trial._staged_credentials(RoleType.AGENT):
            pytest.fail("Consumer must not run after a failed upload")
    assert "secret" not in str(error.value)
    assert trial._credential_dir is None


@pytest.mark.asyncio
async def test_staged_credentials_cleanup_failure_does_not_mask_body_error(tmp_path, fake_creds):
    """A failed cleanup is logged, never replacing the body's exception."""
    trial = _make_trial(tmp_path)

    trial.agent_environment.empty_dirs.side_effect = [
        ExecResult(return_code=0),
        ExecResult(return_code=1, stderr="container gone"),
    ]

    with pytest.raises(ValueError, match="body blew up"):
        async with trial._staged_credentials(RoleType.AGENT):
            raise ValueError("body blew up")

    # The cleanup ran (and failed) but the body's error surfaced, not the rm error.
    trial.logger.warning.assert_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_staging_preserves_unrelated_config_and_clears_all_credential_files(
    tmp_path, fake_creds
):
    trial = _make_trial(tmp_path)
    environment = trial.agent_environment
    original_config = "[profile unrelated]\nregion = eu-west-2\n"
    original_static = "[unrelated]\naws_access_key_id = unrelated-key\n"
    environment.files.update(
        {
            "/home/runner/.aws/config": original_config,
            "/home/runner/.aws/credentials": original_static,
            str(CREDS_PATH / "OLD.json"): "old",
            str(CREDS_PATH / ".interrupted-upload"): "old",
            str(CREDS_PATH / "nested/old.json"): "old",
        }
    )

    async with trial._staged_credentials(RoleType.AGENT):
        config = environment.files["/home/runner/.aws/config"]
        assert config.startswith(original_config)
        assert "[profile PRIMARY]\ncredential_process = sh -c" in config
        assert environment.files["/home/runner/.aws/credentials"] == original_static
        assert {
            path for path in environment.files if PurePosixPath(path).is_relative_to(CREDS_PATH)
        } == {str(CREDS_PATH / "PRIMARY.json")}
        environment.files[str(CREDS_PATH / ".consumer-created")] = "discard"

    assert environment.files == {
        "/home/runner/.aws/config": config,
        "/home/runner/.aws/credentials": original_static,
    }
    assert not trial.logger.warning.called  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "contents"),
    [
        ("credentials", "[PRIMARY]\naws_access_key_id = other-key\n"),
        ("config", "[profile PRIMARY]\nrole_arn = arn:aws:iam::111122223333:role/Other\n"),
        ("config", "[profile PRIMARY]\ncredential_process = echo wrong\n"),
        ("config", "[profile PRIMARY\ninvalid = secret\n"),
    ],
)
async def test_staging_rejects_conflicting_config_before_mint(
    tmp_path, fake_creds, filename, contents
):
    trial = _make_trial(tmp_path)
    path = f"/home/runner/.aws/{filename}"
    trial.agent_environment.files[path] = contents

    with pytest.raises(CredentialError):
        async with trial._staged_credentials(RoleType.AGENT):
            pytest.fail("Conflicting configuration must stop the consumer")

    fake_creds.assert_not_called()
    assert trial.agent_environment.files[path] == contents


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mountpoint",
    [
        "/home",
        "/home/runner/.aws",
        "/home/runner/.aws/creds",
        "/home/runner/.aws/creds/nested",
        "/home/runner/.aws/config",
    ],
)
async def test_staging_refuses_mounts_that_overlap_credential_writes(
    tmp_path, fake_creds, mountpoint
):
    trial = _make_trial(tmp_path)
    trial.agent_environment.mountinfo += f"2 1 0:2 / {mountpoint} rw - tmpfs tmpfs rw\n"

    with pytest.raises(RuntimeError, match="mount overlaps"):
        async with trial._staged_credentials(RoleType.AGENT):
            pytest.fail("A mount conflict must stop the consumer")

    trial.agent_environment.empty_dirs.assert_not_awaited()
    trial.agent_environment.upload_file.assert_not_awaited()
    fake_creds.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("probe", ["", "/home/runner\n", "relative\n1\n1\n", "/x\nuid\n1\n"])
async def test_staging_rejects_unresolved_home_before_cleanup(tmp_path, fake_creds, probe):
    trial = _make_trial(tmp_path)
    trial.agent_environment.exec.side_effect = None
    trial.agent_environment.exec.return_value = ExecResult(return_code=0, stdout=probe)

    with pytest.raises(RuntimeError, match="home and file ownership"):
        async with trial._staged_credentials(RoleType.AGENT):
            pytest.fail("An invalid home must stop the consumer")

    trial.agent_environment.empty_dirs.assert_not_awaited()
    fake_creds.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [None, ExecResult(return_code=1)])
async def test_failed_entry_cleanup_prevents_mint_and_consumer(tmp_path, fake_creds, result):
    trial = _make_trial(tmp_path)
    trial.agent_environment.empty_dirs.return_value = result
    trial.agent_environment.empty_dirs.side_effect = None

    with pytest.raises(RuntimeError, match="clear the credential directory"):
        async with trial._staged_credentials(RoleType.AGENT):
            pytest.fail("Failed cleanup must stop the consumer")

    fake_creds.assert_not_called()
    trial.agent_environment.upload_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_exit_cleanup_retries_before_a_phase_with_a_different_home(tmp_path, fake_creds):
    trial = _make_trial(tmp_path)
    environment = trial.agent_environment
    empty = environment.empty_dirs.side_effect
    count = 0

    async def fail_first_exit(directories, *, chmod):
        nonlocal count
        count += 1
        if count == 2:
            return ExecResult(return_code=1)
        return await empty(directories, chmod=chmod)

    environment.empty_dirs.side_effect = fail_first_exit
    async with trial._staged_credentials(RoleType.AGENT):
        pass
    assert trial._credential_dir == CREDS_PATH

    verifier_path = PurePosixPath("/home/verifier/.aws/creds")
    async with trial._staged_credentials(
        RoleType.VERIFIER, user="verifier", env={"HOME": "/home/verifier"}
    ):
        assert str(CREDS_PATH / "PRIMARY.json") not in environment.files
        assert str(verifier_path / "PRIMARY.json") in environment.files
        assert environment.default_user == "verifier"

    assert environment.cleared_dirs == [CREDS_PATH, CREDS_PATH, verifier_path, verifier_path]
    assert environment.empty_dirs.await_count == 5
    assert environment.default_user is None
    assert trial._credential_dir is None
    commands = [c.kwargs["command"] for c in _exec_calls(trial)]
    assert any(
        "cd -P -- /home/verifier/.aws/creds" in command and "chmod 700 ." in command
        for command in commands
    )
    assert any("chmod 600 " in command and "chown 1001:1001 " in command for command in commands)


@pytest.mark.asyncio
async def test_config_owner_follows_users_who_share_a_home(tmp_path, fake_creds):
    import re

    trial = _make_trial(tmp_path)
    environment = trial.agent_environment
    execute = environment.exec.side_effect
    config_owner = None

    async def permission_checked_exec(*, command, user=None, **kwargs):
        nonlocal config_owner
        if "chown " in command and "/config" in command:
            match = re.search(r"chown ([0-9]+):[0-9]+ ", command)
            assert match is not None and user == "root"
            config_owner = int(match[1])
        if "cat /home/runner/.aws/config; fi" in command and config_owner is not None:
            effective_user = user if user is not None else environment.default_user
            uid = {"root": 0, "runner": 1000, "verifier": 1001}[effective_user]
            if uid != config_owner:
                return ExecResult(return_code=1, stderr="Permission denied")
        return await execute(command=command, user=user, **kwargs)

    environment.exec.side_effect = permission_checked_exec
    for role, user, uid in (
        (RoleType.PRE_INVOKE, "root", 0),
        (RoleType.AGENT, "runner", 1000),
        (RoleType.VERIFIER, "verifier", 1001),
    ):
        async with trial._staged_credentials(role, user=user):
            assert config_owner == uid
            assert str(CREDS_PATH / "PRIMARY.json") in environment.files

    configs = [path for path, _, _ in environment.uploads if "/.config." in path]
    assert len(configs) == 1
    assert environment.default_user is None


@pytest.mark.asyncio
@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="Linux container file descriptors")
@pytest.mark.parametrize("refresh", [False, True])
async def test_delayed_remote_rename_preserves_old_credentials(
    tmp_path, fake_creds, monkeypatch, refresh
):
    import os
    import shutil
    import subprocess

    trial = _make_trial(tmp_path)
    monkeypatch.setattr(aws_trial, "_CREDENTIAL_UPLOAD_ROOT", PurePosixPath(tmp_path))
    directory = tmp_path / "creds"
    directory.mkdir(mode=0o700)
    destination = directory / "PRIMARY.json"
    destination.write_text("OLD_KEY")
    trial._credential_dir = PurePosixPath(directory)
    expiration = datetime.now(timezone.utc) + timedelta(seconds=31)
    payload_expiration = expiration + timedelta(hours=1) if refresh else expiration
    valid_until = expiration if refresh else None
    remote_now = int(expiration.timestamp()) - 60

    async def upload(source, target):
        shutil.copyfile(source, target)

    async def execute(*, command, **kwargs):
        # Control the clock at the final rename to model transport delay.
        result = subprocess.run(
            ["sh", "-c", f"date() {{ printf '%s\\n' {remote_now}; }}\n{command}"],
            text=True,
            capture_output=True,
            check=False,
        )
        return ExecResult(return_code=result.returncode, stdout=result.stdout, stderr=result.stderr)

    trial.agent_environment.upload_file.side_effect = upload
    trial.agent_environment.exec.side_effect = execute
    owner = f"{os.getuid()}:{os.getgid()}"
    await trial._publish_credential_file(
        PurePosixPath(destination),
        "VALID_KEY",
        owner,
        payload_expiration,
        valid_until=valid_until,
    )
    assert destination.read_text() == "VALID_KEY"

    remote_now = int(expiration.timestamp()) + (1 if refresh else -29)
    with pytest.raises(RuntimeError, match="publish credential file"):
        await trial._publish_credential_file(
            PurePosixPath(destination),
            "NEAR_EXPIRY_KEY",
            owner,
            payload_expiration,
            valid_until=valid_until,
        )
    assert destination.read_text() == "VALID_KEY"


def _local_credential_environment(home, monkeypatch):
    """Execute credential shell operations on temporary files through Harbor's cleanup."""
    import os
    import shutil
    import subprocess

    from harbor.models.task.config import TaskOS

    if not Path("/proc/self/fd").is_dir():
        pytest.skip("Linux container file descriptors")
    upload_root = home / "private-uploads"
    upload_root.mkdir(mode=0o700, exist_ok=True)
    monkeypatch.setattr(aws_trial, "_CREDENTIAL_UPLOAD_ROOT", PurePosixPath(upload_root))
    environment = _environment()
    environment.os = TaskOS.LINUX
    environment._empty_dirs_command = MethodType(BaseEnvironment._empty_dirs_command, environment)
    environment._reset_dirs_user = lambda: "root"

    async def execute(command, *, env=None, **kwargs):
        if "cat /proc/self/mountinfo" in command:
            return ExecResult(return_code=0, stdout="1 0 0:1 / / rw - overlay overlay rw\n")
        result = subprocess.run(
            ["sh", "-c", command],
            env={**os.environ, **(env or {}), "HOME": str(home)},
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return ExecResult(return_code=result.returncode, stdout=result.stdout, stderr=result.stderr)

    async def upload(source, destination):
        shutil.copyfile(source, destination)

    async def empty(directories, *, chmod):
        return await BaseEnvironment.empty_dirs(environment, directories, chmod=chmod)

    environment.exec.side_effect = execute
    environment.upload_file.side_effect = upload
    environment.empty_dirs.side_effect = empty
    return environment


@pytest.mark.asyncio
async def test_native_credential_staging_preserves_other_files(tmp_path, fake_creds, monkeypatch):
    home = tmp_path / "home with spaces"
    home.mkdir()
    unrelated = home / "keep"
    unrelated.write_text("unrelated")
    trial = _make_trial(tmp_path)
    trial.agent_environment = _local_credential_environment(home, monkeypatch)
    directory = home / ".aws" / "creds"

    async with trial._staged_credentials(RoleType.AGENT):
        assert json.loads((directory / "PRIMARY.json").read_text())["AccessKeyId"] == "AKIA"
        assert directory.stat().st_mode & 0o777 == 0o700
        assert (directory / "PRIMARY.json").stat().st_mode & 0o777 == 0o600
        assert (directory.parent / "config").stat().st_mode & 0o777 == 0o600
    assert list(directory.iterdir()) == []
    assert list((home / "private-uploads").iterdir()) == []
    assert unrelated.read_text() == "unrelated"


@pytest.mark.asyncio
async def test_writable_upload_root_is_rejected_before_transfer(tmp_path, fake_creds, monkeypatch):
    import os

    directory = tmp_path / "creds"
    directory.mkdir()
    trial = _make_trial(tmp_path)
    environment = trial.agent_environment = _local_credential_environment(tmp_path, monkeypatch)
    trial._credential_dir = PurePosixPath(directory)
    (tmp_path / "private-uploads").chmod(0o777)
    with pytest.raises(RuntimeError, match="publish credential file"):
        await trial._publish_credential_file(
            PurePosixPath(directory / "PRIMARY.json"), "SYNTHETIC", f"{os.getuid()}:{os.getgid()}"
        )
    assert list(directory.iterdir()) == []
    assert list((tmp_path / "private-uploads").iterdir()) == []
    environment.upload_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_directory_swap_cannot_redirect_permission_changes(tmp_path, fake_creds, monkeypatch):
    """Changing the path after validation must not change another directory's mode."""
    home = tmp_path / "home"
    home.mkdir()
    directory = home / ".aws" / "creds"
    displaced = tmp_path / "original-creds"
    other = tmp_path / "other-directory"
    other.mkdir(mode=0o750)
    trial = _make_trial(tmp_path)
    environment = trial.agent_environment = _local_credential_environment(home, monkeypatch)
    execute = environment.exec.side_effect
    swapped = False

    async def swap_before_chmod(command, **kwargs):
        nonlocal swapped
        if not swapped and "chmod 700" in command:
            swapped = True
            command = (
                "chmod() { "
                f"command mv -- {shlex.quote(str(directory))} {shlex.quote(str(displaced))}; "
                f"command ln -s -- {shlex.quote(str(other))} {shlex.quote(str(directory))}; "
                'command chmod "$@"; }\n' + command
            )
        return await execute(command, **kwargs)

    environment.exec.side_effect = swap_before_chmod
    with contextlib.suppress(RuntimeError):
        async with trial._staged_credentials(RoleType.AGENT):
            pass
    assert swapped
    assert other.stat().st_mode & 0o777 == 0o750


@pytest.mark.asyncio
async def test_publication_swap_cannot_redirect_permission_changes(
    tmp_path, fake_creds, monkeypatch
):
    """The uploaded file's open descriptor must survive replacement of its path."""
    import os

    directory = tmp_path / "creds"
    directory.mkdir()
    displaced = tmp_path / "original-upload"
    other = tmp_path / "other-file"
    other.write_text("unrelated")
    other.chmod(0o640)
    trial = _make_trial(tmp_path)
    environment = trial.agent_environment = _local_credential_environment(tmp_path, monkeypatch)
    trial._credential_dir = PurePosixPath(directory)
    execute = environment.exec.side_effect
    upload = environment.upload_file.side_effect
    uploaded: Path | None = None

    async def remember_upload(source, destination):
        nonlocal uploaded
        uploaded = Path(destination)
        await upload(source, destination)

    async def swap_before_chmod(command, **kwargs):
        if uploaded is not None and "chmod 600" in command:
            command = (
                "chmod() { "
                f"command mv -- {shlex.quote(str(uploaded))} {shlex.quote(str(displaced))}; "
                f"command ln -s -- {shlex.quote(str(other))} {shlex.quote(str(uploaded))}; "
                'command chmod "$@"; }\n' + command
            )
        return await execute(command, **kwargs)

    environment.upload_file.side_effect = remember_upload
    environment.exec.side_effect = swap_before_chmod
    with contextlib.suppress(RuntimeError):
        await trial._publish_credential_file(
            PurePosixPath(directory / "PRIMARY.json"), "SYNTHETIC", f"{os.getuid()}:{os.getgid()}"
        )
    assert uploaded is not None
    assert other.read_text() == "unrelated"
    assert other.stat().st_mode & 0o777 == 0o640


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
async def test_publication_refuses_nonprivate_uploaded_file(
    tmp_path, fake_creds, monkeypatch, kind
):
    import os

    directory = tmp_path / "creds"
    directory.mkdir()
    other = tmp_path / "other-file"
    other.write_text("unrelated")
    other.chmod(0o640)
    trial = _make_trial(tmp_path)
    environment = trial.agent_environment = _local_credential_environment(tmp_path, monkeypatch)
    trial._credential_dir = PurePosixPath(directory)

    async def substitute_upload(source, destination):
        path = Path(destination)
        if kind == "symlink":
            path.symlink_to(other)
        elif kind == "hardlink":
            path.hardlink_to(other)
        else:
            os.mkfifo(path)

    environment.upload_file.side_effect = substitute_upload
    destination = directory / "PRIMARY.json"
    with pytest.raises(RuntimeError, match="publish credential file"):
        await trial._publish_credential_file(
            PurePosixPath(destination), "SYNTHETIC", f"{os.getuid()}:{os.getgid()}"
        )
    assert not destination.exists()
    assert other.read_text() == "unrelated"
    assert other.stat().st_mode & 0o777 == 0o640


@pytest.mark.asyncio
async def test_cleanup_parent_swap_preserves_unrelated_files(tmp_path, fake_creds, monkeypatch):
    """Harbor cleanup must use the checked parent, not follow a replaced ancestor."""
    home = tmp_path / "home"
    aws_dir = home / ".aws"
    aws_dir.mkdir(parents=True)
    displaced = tmp_path / "original-aws"
    other = tmp_path / "other-directory"
    (other / "creds").mkdir(parents=True)
    sentinel = other / "creds" / "sentinel"
    sentinel.write_text("unrelated")
    trial = _make_trial(tmp_path)
    environment = trial.agent_environment = _local_credential_environment(home, monkeypatch)
    empty = environment.empty_dirs.side_effect
    swapped = False

    async def swap_before_empty(directories, *, chmod):
        nonlocal swapped
        if not swapped:
            swapped = True
            aws_dir.rename(displaced)
            aws_dir.symlink_to(other, target_is_directory=True)
        return await empty(directories, chmod=chmod)

    environment.empty_dirs.side_effect = swap_before_empty
    with contextlib.suppress(RuntimeError):
        async with trial._staged_credentials(RoleType.AGENT):
            pass
    assert swapped
    assert sentinel.read_text() == "unrelated"


@pytest.mark.asyncio
async def test_cleanup_child_swap_preserves_unrelated_files(tmp_path, fake_creds, monkeypatch):
    """Deletion must resolve ./children from the opened credential directory."""
    import os
    import shutil

    home = tmp_path / "home"
    directory = home / ".aws" / "creds"
    directory.mkdir(parents=True)
    (directory / "sentinel").write_text("credential")
    displaced = tmp_path / "original-creds"
    other = tmp_path / "other-directory"
    other.mkdir()
    sentinel = other / "sentinel"
    sentinel.write_text("unrelated")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    real_rm = shutil.which("rm")
    assert real_rm is not None
    shim = binaries / "rm"
    shim.write_text(
        "#!/bin/sh\n"
        'case "$3" in\n'
        "  *sentinel)\n"
        f"    mv -- {shlex.quote(str(directory))} {shlex.quote(str(displaced))}\n"
        f"    ln -s -- {shlex.quote(str(other))} {shlex.quote(str(directory))}\n"
        "    ;;\n"
        "esac\n"
        f'exec {shlex.quote(real_rm)} "$@"\n'
    )
    shim.chmod(0o700)
    trial = _make_trial(tmp_path)
    trial.agent_environment = _local_credential_environment(home, monkeypatch)
    with trial._credential_commands("PRIMARY", None, {"PATH": f"{binaries}:{os.defpath}"}):
        await trial._clear_credentials(PurePosixPath(directory))
    assert displaced.exists()
    assert sentinel.read_text() == "unrelated"


@pytest.mark.asyncio
async def test_upload_parent_swap_cannot_write_outside_credentials(
    tmp_path, fake_creds, monkeypatch
):
    import os

    directory = tmp_path / "creds"
    directory.mkdir()
    displaced = tmp_path / "original-creds"
    other = tmp_path / "other-directory"
    other.mkdir()
    trial = _make_trial(tmp_path)
    environment = trial.agent_environment = _local_credential_environment(tmp_path, monkeypatch)
    trial._credential_dir = PurePosixPath(directory)
    upload = environment.upload_file.side_effect

    async def redirect_before_upload(source, destination):
        directory.rename(displaced)
        directory.symlink_to(other, target_is_directory=True)
        await upload(source, destination)

    environment.upload_file.side_effect = redirect_before_upload
    with pytest.raises(RuntimeError, match="publish credential file"):
        await trial._publish_credential_file(
            PurePosixPath(directory / "PRIMARY.json"), "SYNTHETIC", f"{os.getuid()}:{os.getgid()}"
        )
    assert list(other.iterdir()) == []


@pytest.mark.asyncio
async def test_final_cleanup_failure_warns_and_still_stops_container(tmp_path, fake_creds, mocker):
    trial = _make_trial(tmp_path)
    trial._credential_dir = CREDS_PATH
    trial.agent_environment.empty_dirs.side_effect = RuntimeError("container unavailable")
    stopped = mocker.patch.object(Trial, "_stop_agent_environment", AsyncMock())
    recorded = mocker.patch.object(trial, "_record_exception")

    await trial._stop_agent_environment()

    trial.agent_environment.empty_dirs.assert_awaited_once_with([PurePosixPath(".")], chmod=False)
    stopped.assert_awaited_once()
    recorded.assert_not_called()
    trial.logger.warning.assert_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_cancelled_mint_cannot_publish_after_cleanup(tmp_path, fake_creds):
    import threading

    trial = _make_trial(tmp_path)
    started, release, finished = (threading.Event() for _ in range(3))
    credentials = fake_creds.return_value

    def blocked_mint(*args):
        started.set()
        assert release.wait(timeout=5)
        finished.set()
        return credentials

    fake_creds.side_effect = blocked_mint

    async def phase():
        async with trial._staged_credentials(RoleType.AGENT):
            pytest.fail("A cancelled mint must not start the consumer")

    task = asyncio.create_task(phase())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert trial._credential_dir is None
        trial.agent_environment.upload_file.assert_not_awaited()
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)

    await asyncio.sleep(0)
    trial.agent_environment.upload_file.assert_not_awaited()
    assert trial.agent_environment.files == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["config", "PRIMARY.json"])
async def test_cancelled_upload_finishes_before_cleanup(tmp_path, fake_creds, filename):
    trial = _make_trial(tmp_path)
    environment = trial.agent_environment
    upload = environment.upload_file.side_effect
    started, release = asyncio.Event(), asyncio.Event()
    local_source: Path | None = None

    async def blocked_upload(source, destination):
        nonlocal local_source
        if f"/.{filename}." in destination:
            local_source = Path(source)
            started.set()
            await release.wait()
            assert local_source.exists()
        await upload(source, destination)

    environment.upload_file.side_effect = blocked_upload

    async def phase():
        async with trial._staged_credentials(RoleType.AGENT):
            pytest.fail("A cancelled upload must not start the consumer")

    task = asyncio.create_task(phase())
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert local_source is not None and local_source.exists()
    assert environment.empty_dirs.await_count == 1
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert environment.empty_dirs.await_count == 2
    assert not local_source.exists()
    assert trial._credential_dir is None
    assert not any("/creds/" in path for path in environment.files)
    assert not any(PurePosixPath(path).name == filename for path, _ in environment.publications)


@pytest.mark.asyncio
async def test_unsettled_publication_blocks_later_roles_and_still_stops(
    tmp_path, fake_creds, mocker
):
    trial = _make_trial(tmp_path, post_invoke=_phase(), has_post_script=True)
    release = asyncio.Event()
    upload = trial.agent_environment.upload_file.side_effect

    async def blocked_upload(source, destination):
        await release.wait()
        await upload(source, destination)

    trial.agent_environment.upload_file.side_effect = blocked_upload
    mocker.patch.object(aws_trial, "_CREDENTIAL_OPERATION_TIMEOUT_SEC", 0.01)

    with pytest.raises(TimeoutError, match="Credential file operation"):
        async with trial._staged_credentials(RoleType.AGENT):
            pytest.fail("An unsettled publication must stop the consumer")

    operation = trial._credential_operation_task
    assert operation is not None and not operation.done()
    assert trial._credential_failed
    with pytest.raises(RuntimeError, match="cannot start another phase"):
        async with trial._staged_credentials(RoleType.VERIFIER):
            pytest.fail("A verifier must not overlap an unresolved agent publication")

    post = mocker.patch.object(trial, "_run_phase_script", AsyncMock())
    stopped = mocker.patch.object(Trial, "_stop_agent_environment", AsyncMock())
    await trial._stop_agent_environment()
    post.assert_not_awaited()
    stopped.assert_awaited_once()
    release.set()
    await operation
    await trial._stop_agent_environment()
    assert trial._credential_dir is None


@pytest.mark.asyncio
async def test_cleanup_timeout_does_not_fail_a_completed_phase(tmp_path, fake_creds, mocker):
    trial = _make_trial(tmp_path)
    release = asyncio.Event()
    empty = trial.agent_environment.empty_dirs.side_effect
    count = 0

    async def blocked_exit(directories, *, chmod):
        nonlocal count
        count += 1
        if count == 2:
            await release.wait()
        return await empty(directories, chmod=chmod)

    trial.agent_environment.empty_dirs.side_effect = blocked_exit
    mocker.patch.object(aws_trial, "_CREDENTIAL_OPERATION_TIMEOUT_SEC", 0.01)
    async with trial._staged_credentials(RoleType.AGENT):
        pass

    assert not trial._credential_failed
    operation = trial._credential_operation_task
    assert operation is not None and not operation.done()
    stopped = mocker.patch.object(Trial, "_stop_agent_environment", AsyncMock())
    await trial._stop_agent_environment()
    stopped.assert_awaited_once()
    trial.logger.warning.assert_called()  # type: ignore[attr-defined]
    release.set()
    await operation
    assert trial._credential_dir is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        asyncio.CancelledError(),
        AgentTimeoutError("agent timeout"),
        TimeoutError("script timeout"),
        OperationCancelled(),
    ],
)
async def test_interrupted_consumer_cannot_receive_later_role_credentials(
    tmp_path, fake_creds, mocker, failure
):
    trial = _make_trial(tmp_path, post_invoke=_phase(), has_post_script=True)
    with pytest.raises(type(failure)) as error:
        async with trial._staged_credentials(RoleType.AGENT):
            raise failure
    assert error.value is failure
    assert trial._credential_failed

    with pytest.raises(RuntimeError, match="cannot start another phase"):
        async with trial._staged_credentials(RoleType.VERIFIER):
            pytest.fail("The old command may still be alive; do not publish another role")

    post = mocker.patch.object(trial, "_run_phase_script", AsyncMock())
    stopped = mocker.patch.object(Trial, "_stop_agent_environment", AsyncMock())
    await trial._stop_agent_environment()
    post.assert_not_awaited()
    stopped.assert_awaited_once()
    assert fake_creds.call_count == 1
    assert not any("/creds/" in path for path in trial.agent_environment.files)


@pytest.mark.asyncio
async def test_wrapped_docker_timeout_blocks_later_roles(tmp_path, fake_creds, mocker):
    trial = _make_trial(tmp_path, post_invoke=_phase(), has_post_script=True)
    with pytest.raises(RuntimeError, match="Command timed out after 1 seconds") as error:
        async with trial._staged_credentials(RoleType.PRE_INVOKE):
            try:
                raise TimeoutError()
            except TimeoutError:
                # Match Harbor DockerEnvironment's exception conversion.
                raise RuntimeError("Command timed out after 1 seconds")

    assert isinstance(error.value.__context__, TimeoutError)
    assert trial._credential_failed
    with pytest.raises(RuntimeError, match="cannot start another phase"):
        async with trial._staged_credentials(RoleType.AGENT):
            pytest.fail("A timed-out pre-invoke command must not receive agent credentials")
    post = mocker.patch.object(trial, "_run_phase_script", AsyncMock())
    stopped = mocker.patch.object(Trial, "_stop_agent_environment", AsyncMock())
    await trial._stop_agent_environment()
    post.assert_not_awaited()
    stopped.assert_awaited_once()
    assert fake_creds.call_count == 1


@pytest.mark.asyncio
async def test_exited_agent_can_still_receive_a_verifier_result(tmp_path, fake_creds):
    trial = _make_trial(tmp_path)
    with pytest.raises(NonZeroAgentExitCodeError):
        async with trial._staged_credentials(RoleType.AGENT):
            raise NonZeroAgentExitCodeError("agent exited with code 1")

    assert not trial._credential_failed
    async with trial._staged_credentials(RoleType.VERIFIER):
        assert str(CREDS_PATH / "PRIMARY.json") in trial.agent_environment.files
    assert fake_creds.call_count == 2


@pytest.mark.asyncio
async def test_unsettled_refresh_aborts_the_active_consumer(tmp_path, fake_creds, mocker):
    trial = _make_trial(tmp_path)
    release = asyncio.Event()
    upload = trial.agent_environment.upload_file.side_effect
    files, expiry = fake_creds.return_value
    fake_creds.side_effect = [
        (files, expiry),
        ({"PRIMARY.json": files["PRIMARY.json"].replace("AKIA", "RENEWED")}, expiry),
    ]
    publications = 0

    async def blocked_renewal(source, destination):
        nonlocal publications
        if ".PRIMARY.json." in destination:
            publications += 1
            if publications == 2:
                await release.wait()
        await upload(source, destination)

    trial.agent_environment.upload_file.side_effect = blocked_renewal
    mocker.patch.object(aws_trial, "_CREDENTIAL_OPERATION_TIMEOUT_SEC", 0.01)
    mocker.patch.object(credentials_provider, "CRED_REFRESH_MIN_SLEEP_SEC", 0.005)
    mocker.patch.object(credentials_provider, "_CRED_REFRESH_SKEW_SEC", 3600)

    async def phase():
        async with trial._staged_credentials(RoleType.AGENT):
            await asyncio.sleep(1000)

    task = asyncio.create_task(phase())
    with pytest.raises(RuntimeError, match="publication did not settle"):
        await asyncio.wait_for(task, timeout=2)
    assert task.cancelling() == 0
    assert trial._credential_failed
    operation = trial._credential_operation_task
    assert operation is not None
    original = trial.agent_environment.files[str(CREDS_PATH / "PRIMARY.json")]
    release.set()
    await operation
    assert trial.agent_environment.files[str(CREDS_PATH / "PRIMARY.json")] == original
    mocker.patch.object(Trial, "_stop_agent_environment", AsyncMock())
    await trial._stop_agent_environment()
    assert trial._credential_dir is None


@pytest.mark.asyncio
async def test_normal_phase_exit_discards_a_pending_refresh(tmp_path, fake_creds, mocker):
    trial = _make_trial(tmp_path)
    environment = trial.agent_environment
    files, expiry = fake_creds.return_value
    fake_creds.side_effect = [
        (files, expiry),
        ({"PRIMARY.json": files["PRIMARY.json"].replace("AKIA", "RENEWED")}, expiry),
    ]
    started, release, consumer_finished = (asyncio.Event() for _ in range(3))
    upload = environment.upload_file.side_effect

    async def delayed_upload(source, destination):
        if "RENEWED" in Path(source).read_text():
            started.set()
            await release.wait()
        await upload(source, destination)

    environment.upload_file.side_effect = delayed_upload
    mocker.patch.object(credentials_provider, "CRED_REFRESH_MIN_SLEEP_SEC", 0.005)
    mocker.patch.object(credentials_provider, "_CRED_REFRESH_SKEW_SEC", 3600)

    async def phase():
        async with trial._staged_credentials(RoleType.AGENT):
            await started.wait()
            consumer_finished.set()

    task = asyncio.create_task(phase())
    try:
        await asyncio.wait_for(consumer_finished.wait(), timeout=2)
        assert not task.done()
        release.set()
        await asyncio.wait_for(task, timeout=2)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert not trial._credential_failed
    assert not any("RENEWED" in body for _, body in environment.publications)
    assert trial._credential_dir is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["mint", "publication", "stalled_mint"])
@pytest.mark.parametrize("suppress_cancel", [False, True])
async def test_expired_credentials_abort_consumer_and_block_later_roles(
    tmp_path, fake_creds, mocker, failure, suppress_cancel
):
    """Neither repeated errors nor a pending mint may keep an expired phase active."""
    import threading

    trial = _make_trial(tmp_path, post_invoke=_phase(), has_post_script=True)
    files, _ = fake_creds.return_value
    release = threading.Event()
    finished = threading.Event()
    calls = 0

    def mint(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            expiry = datetime.now(timezone.utc) + timedelta(seconds=0.25)
            payload = {**json.loads(files["PRIMARY.json"]), "Expiration": expiry.isoformat()}
            return {"PRIMARY.json": json.dumps(payload)}, expiry
        if failure == "stalled_mint":
            try:
                assert release.wait(timeout=5)
            finally:
                finished.set()
        if failure != "publication":
            raise RuntimeError("synthetic refresh failure")
        return (
            {"PRIMARY.json": files["PRIMARY.json"].replace("AKIA", "RENEWED")},
            datetime.now(timezone.utc) + timedelta(hours=1),
        )

    fake_creds.side_effect = mint
    upload = trial.agent_environment.upload_file.side_effect

    async def fail_renewal_upload(source, destination):
        if failure == "publication" and calls > 1:
            raise RuntimeError("synthetic publication failure")
        await upload(source, destination)

    trial.agent_environment.upload_file.side_effect = fail_renewal_upload
    mocker.patch.object(credentials_provider, "CRED_REFRESH_MIN_SLEEP_SEC", 0.01)
    mocker.patch.object(credentials_provider, "_CRED_REFRESH_RETRY_SEC", 0.01)

    async def phase():
        async with trial._staged_credentials(RoleType.AGENT):
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                if not suppress_cancel:
                    raise

    try:
        with pytest.raises(CredentialError, match="expired|refresh"):
            await asyncio.wait_for(phase(), timeout=2)
    finally:
        release.set()
        if failure == "stalled_mint" and calls > 1:
            assert await asyncio.to_thread(finished.wait, 5)
    assert calls > 1
    assert trial._credential_failed
    assert trial._credential_operation_task is None
    assert not any("/creds/" in path for path in trial.agent_environment.files)
    with pytest.raises(RuntimeError, match="cannot start another phase"):
        async with trial._staged_credentials(RoleType.VERIFIER):
            pytest.fail("Expired phase must not grant credentials for another role")


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_kind", ["return", "nonzero", "cancel"])
async def test_phase_exit_checks_elapsed_expiry_before_disarming_timer(
    tmp_path, fake_creds, mocker, exit_kind
):
    """Final synchronous work must not outrun the timeout callback and admit a verifier."""
    import time

    trial = _make_trial(tmp_path)
    record = json.loads(fake_creds.return_value[0]["PRIMARY.json"])

    def mint(*args):
        expiry = datetime.now(timezone.utc) + timedelta(seconds=0.15)
        return {"PRIMARY.json": json.dumps({**record, "Expiration": expiry.isoformat()})}, expiry

    fake_creds.side_effect = mint
    mocker.patch.object(credentials_provider, "CRED_REFRESH_MIN_SLEEP_SEC", 0.001)
    expected = asyncio.CancelledError if exit_kind == "cancel" else CredentialError
    with pytest.raises(expected):
        async with trial._staged_credentials(RoleType.AGENT):
            time.sleep(0.2)
            if exit_kind == "nonzero":
                raise NonZeroAgentExitCodeError("agent exited")
            if exit_kind == "cancel":
                raise asyncio.CancelledError
    assert trial._credential_failed
    with pytest.raises(RuntimeError, match="cannot start another phase"):
        async with trial._staged_credentials(RoleType.VERIFIER):
            pytest.fail("Elapsed expiration must prevent a later role")


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_cause", [False, True])
async def test_wrapped_expiry_cancellation_remains_a_credential_error(
    tmp_path, fake_creds, mocker, explicit_cause
):
    trial = _make_trial(tmp_path)
    record = json.loads(fake_creds.return_value[0]["PRIMARY.json"])
    minted = False

    def mint(*args):
        nonlocal minted
        if minted:
            raise RuntimeError("synthetic renewal failure")
        minted = True
        expiry = datetime.now(timezone.utc) + timedelta(seconds=0.15)
        return {"PRIMARY.json": json.dumps({**record, "Expiration": expiry.isoformat()})}, expiry

    fake_creds.side_effect = mint
    mocker.patch.object(credentials_provider, "CRED_REFRESH_MIN_SLEEP_SEC", 0.001)
    with pytest.raises(CredentialError):
        async with trial._staged_credentials(RoleType.AGENT):
            try:
                await asyncio.Future()
            except asyncio.CancelledError as exc:
                if explicit_cause:
                    raise NonZeroAgentExitCodeError("agent exited") from exc
                raise NonZeroAgentExitCodeError("agent exited")
    assert trial._credential_failed
    with pytest.raises(RuntimeError, match="cannot start another phase"):
        async with trial._staged_credentials(RoleType.VERIFIER):
            pytest.fail("An error wrapper must not allow later-role credentials")


@pytest.mark.asyncio
async def test_successful_publication_extends_consumer_deadline(tmp_path, fake_creds, mocker):
    trial = _make_trial(tmp_path)
    record = json.loads(fake_creds.return_value[0]["PRIMARY.json"])
    initial_expiry = datetime.now(timezone.utc) + timedelta(seconds=0.25)
    renewed_expiry = initial_expiry + timedelta(seconds=5)
    initial = {"PRIMARY.json": json.dumps({**record, "Expiration": initial_expiry.isoformat()})}
    renewed = {
        "PRIMARY.json": json.dumps(
            {**record, "AccessKeyId": "RENEWED", "Expiration": renewed_expiry.isoformat()}
        )
    }
    fake_creds.side_effect = [(initial, initial_expiry)]
    published = asyncio.Event()
    upload = trial.agent_environment.upload_file.side_effect

    async def observe_publication(source, destination):
        await upload(source, destination)
        if "RENEWED" in Path(source).read_text():
            published.set()

    trial.agent_environment.upload_file.side_effect = observe_publication
    mocker.patch.object(credentials_provider, "CRED_REFRESH_MIN_SLEEP_SEC", 0.01)
    async with trial._staged_credentials(RoleType.AGENT):
        fake_creds.side_effect = None
        fake_creds.return_value = renewed, renewed_expiry
        await asyncio.wait_for(published.wait(), timeout=2)
        await asyncio.sleep(0.3)
        assert datetime.now(timezone.utc) > initial_expiry
        assert (
            json.loads(trial.agent_environment.files[str(CREDS_PATH / "PRIMARY.json")])[
                "AccessKeyId"
            ]
            == "RENEWED"
        )
    assert not trial._credential_failed


@pytest.mark.asyncio
async def test_expiration_aborts_consumer_before_pending_upload_settles(
    tmp_path, fake_creds, mocker
):
    trial = _make_trial(tmp_path)
    record = json.loads(fake_creds.return_value[0]["PRIMARY.json"])
    expiry = datetime.now(timezone.utc) + timedelta(seconds=0.25)
    initial = {"PRIMARY.json": json.dumps({**record, "Expiration": expiry.isoformat()})}
    renewed_expiry = expiry + timedelta(hours=1)
    renewed = {
        "PRIMARY.json": json.dumps(
            {**record, "AccessKeyId": "RENEWED", "Expiration": renewed_expiry.isoformat()}
        )
    }
    fake_creds.side_effect = [(initial, expiry), (renewed, renewed_expiry)]
    started, release, consumer_cancelled = (asyncio.Event() for _ in range(3))
    upload = trial.agent_environment.upload_file.side_effect

    async def delayed_upload(source, destination):
        if "RENEWED" in Path(source).read_text():
            started.set()
            await release.wait()
        await upload(source, destination)

    trial.agent_environment.upload_file.side_effect = delayed_upload
    mocker.patch.object(credentials_provider, "CRED_REFRESH_MIN_SLEEP_SEC", 0.01)

    async def phase():
        async with trial._staged_credentials(RoleType.AGENT):
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                consumer_cancelled.set()
                raise

    task = asyncio.create_task(phase())
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        await asyncio.wait_for(consumer_cancelled.wait(), timeout=2)
        assert not release.is_set()
        release.set()
        with pytest.raises(CredentialError, match="expired|refresh"):
            await asyncio.wait_for(task, timeout=2)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert trial._credential_failed
    assert trial._credential_operation_task is None
    assert not any("/creds/" in path for path in trial.agent_environment.files)


@pytest.mark.asyncio
async def test_unexpected_refresher_exit_aborts_active_consumer(tmp_path, fake_creds, mocker):
    trial = _make_trial(tmp_path)

    async def stopped_refresh(*args):
        return

    mocker.patch.object(aws_trial, "refresh_credentials_loop", stopped_refresh)

    async def phase():
        async with trial._staged_credentials(RoleType.AGENT):
            await asyncio.Future()

    with pytest.raises(CredentialError, match="refresh"):
        await asyncio.wait_for(phase(), timeout=2)
    assert trial._credential_failed
    assert not any("/creds/" in path for path in trial.agent_environment.files)


@pytest.mark.asyncio
async def test_refresh_freezes_account_role_and_reuses_unchanged_files(
    tmp_path, fake_creds, mocker
):
    trial = _make_trial(tmp_path)
    ready: asyncio.Future[Callable[[], Awaitable[datetime]]] = (
        asyncio.get_running_loop().create_future()
    )

    async def capture_refresh(expires_at, refresh_once, log):
        ready.set_result(refresh_once)
        await asyncio.Future()

    mocker.patch.object(aws_trial, "refresh_credentials_loop", capture_refresh)
    async with trial._staged_credentials(RoleType.AGENT):
        refresh = await asyncio.wait_for(ready, timeout=5)
        uploads = trial.agent_environment.upload_file.await_count
        trial.config.account_mapping = {"OTHER": "999900001111"}  # type: ignore[attr-defined]
        trial.task.config.scenario = _scenario_ref(agent="OtherRole")
        assert await refresh() == fake_creds.return_value[1]
        assert trial.agent_environment.upload_file.await_count == uploads

    assert fake_creds.call_count == 2
    for minted in fake_creds.call_args_list:
        assert minted.args[1:] == ({"PRIMARY": "123456789012"}, "AgentRole", "app-session")


@pytest.mark.asyncio
async def test_failed_refresh_keeps_the_last_complete_file(tmp_path, fake_creds, mocker):
    trial = _make_trial(tmp_path)
    ready: asyncio.Future[Callable[[], Awaitable[datetime]]] = (
        asyncio.get_running_loop().create_future()
    )

    async def capture_refresh(expires_at, refresh_once, log):
        ready.set_result(refresh_once)
        await asyncio.Future()

    mocker.patch.object(aws_trial, "refresh_credentials_loop", capture_refresh)
    async with trial._staged_credentials(RoleType.AGENT):
        refresh = await asyncio.wait_for(ready, timeout=5)
        old_file = trial.agent_environment.files[str(CREDS_PATH / "PRIMARY.json")]
        files, expiry = fake_creds.return_value
        fake_creds.return_value = (
            {"PRIMARY.json": files["PRIMARY.json"].replace("AKIA", "RENEWED")},
            expiry,
        )
        upload = trial.agent_environment.upload_file.side_effect

        async def partial_upload(source, destination):
            await upload(source, destination)
            raise RuntimeError("upload failed with secret contents")

        trial.agent_environment.upload_file.side_effect = partial_upload
        with pytest.raises(RuntimeError, match="publish credential file") as error:
            await refresh()
        assert "secret contents" not in str(error.value)
        assert trial.agent_environment.files[str(CREDS_PATH / "PRIMARY.json")] == old_file
        assert not any(".PRIMARY.json." in path for path in trial.agent_environment.files)

    assert not any("/creds/" in path for path in trial.agent_environment.files)


@pytest.mark.asyncio
async def test_environment_wrapper_removes_image_and_transport_credentials(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys

    trial = _make_trial(tmp_path)
    inherited = {name: f"image-{name}" for name in CREDENTIAL_ENV_VARS}
    inherited["AWS_BEARER_TOKEN_BEDROCK"] = "model-token"
    inherited["AWS_REGION"] = "eu-west-1"
    persistent = dict.fromkeys(CREDENTIAL_ENV_VARS, "persistent-secret")
    captured: dict[str, str] = {}
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "host-key")

    async def execute(*, command, env, **kwargs):
        captured.update(env)
        process = subprocess.run(
            ["sh", "-c", command],
            env={**os.environ, **inherited, **persistent, **env},
            capture_output=True,
            text=True,
            check=True,
        )
        return ExecResult(return_code=process.returncode, stdout=process.stdout)

    trial.agent_environment.exec = AsyncMock(side_effect=execute)
    original_exec = trial.agent_environment.exec
    names = [
        *CREDENTIAL_ENV_VARS,
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_REGION",
        "AWS_EC2_METADATA_DISABLED",
    ]
    script = (
        f"import json, os; "
        f"print(json.dumps({{k: os.environ[k] for k in {names!r} if k in os.environ}}))"
    )
    with trial._credential_commands("PRIMARY", "runner"):
        result = await trial.agent_environment.exec(
            command=f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}",
            env={**dict.fromkeys(CREDENTIAL_ENV_VARS, "command-secret"), "AWS_CONFIG_FILE": ""},
        )
        assert trial.agent_environment.default_user == "runner"

    assert json.loads(result.stdout) == {
        "AWS_PROFILE": "PRIMARY",
        "AWS_BEARER_TOKEN_BEDROCK": "model-token",
        "AWS_REGION": "eu-west-1",
        "AWS_EC2_METADATA_DISABLED": "true",
    }
    assert "command-secret" not in captured.values()
    assert "persistent-secret" not in captured.values()
    assert os.environ["AWS_ACCESS_KEY_ID"] == "host-key"
    assert trial.agent_environment.exec is original_exec
    assert trial.agent_environment.default_user is None


# --- verifier creds at the precedence the verifier reads ------------------


@pytest.mark.asyncio
async def test_shared_verifier_overlays_cred_env_on_config_verifier_env(
    tmp_path, fake_creds, mocker
):
    """Verifier cred env is transiently overlaid on self.config.verifier.env.

    The overlay (AWS_PROFILE + emptied raw creds, task env preserved) is present
    DURING the super call (the override_env, HIGHEST-precedence layer), then
    restored so the persisted config never carries live creds.
    """
    trial = _make_trial(tmp_path)
    trial._aws_placeholders = {}
    original_env = trial.config.verifier.env

    seen: dict[str, str] = {}

    async def fake_super(self, **k):
        # The base reads self.config.verifier.env as override_env; capture it here.
        seen.update(self.config.verifier.env)
        return MagicMock()

    mocker.patch.object(Trial, "_run_shared_verifier", fake_super)
    await trial._run_shared_verifier(timeout_sec=None, user=None)

    # The cred env was present for the verifier call: profile set, raw creds
    # emptied, task env preserved alongside.
    assert seen["AWS_PROFILE"] == "PRIMARY"
    assert seen["AWS_ACCESS_KEY_ID"] == ""
    assert seen["REGION"] == "us-east-1"
    # ...and the original env is restored afterward (no live overlay persists).
    assert trial.config.verifier.env is original_env
    assert "AWS_PROFILE" not in trial.config.verifier.env


@pytest.mark.asyncio
async def test_verifier_env_restored_when_super_raises(tmp_path, fake_creds, mocker):
    """If the verifier call raises, the original env is still restored (no live creds persist)."""
    trial = _make_trial(tmp_path)
    trial._aws_placeholders = {}
    original_env = trial.config.verifier.env

    async def fake_super(self, **k):
        raise RuntimeError("verifier blew up")

    mocker.patch.object(Trial, "_run_shared_verifier", fake_super)

    with pytest.raises(RuntimeError, match="verifier blew up"):
        await trial._run_shared_verifier(timeout_sec=None, user=None)

    assert trial.config.verifier.env is original_env
    assert "AWS_ACCESS_KEY_ID" not in trial.config.verifier.env


@pytest.mark.asyncio
async def test_verifier_uses_distinct_config_layers_and_its_actual_home(
    tmp_path, fake_creds, mocker
):
    trial = _make_trial(tmp_path)
    task_env = {
        "HOME": "/home/task",
        "AWS_PROFILE": "OLD",
        "TASK_ONLY": "keep",
        "RESOURCE": "{{BucketName}}",
    }
    override_env = {
        "HOME": "/home/verifier",
        "AWS_PROFILE": "OVERRIDE",
        "AWS_DEFAULT_PROFILE": "OTHER",
        "AWS_REGION": "ap-south-1",
        "AWS_WEB_IDENTITY_TOKEN_FILE": "${MISSING_UNUSED_TOKEN}",
    }
    trial._aws_placeholders = {"PRIMARY": {"BucketName": "bucket-123"}}
    trial.task.config.verifier.env = task_env
    trial.config.verifier.env = override_env
    command_env = {"AWS_PROFILE": "COMMAND", "HOME": "/home/command"}

    async def verify(self, **kwargs):
        assert self.agent_environment.default_user == "verifier"
        for layer in (self.task.config.verifier.env, self.config.verifier.env):
            assert layer["AWS_PROFILE"] == "PRIMARY"
            assert layer["AWS_DEFAULT_PROFILE"] == ""
            assert layer["AWS_WEB_IDENTITY_TOKEN_FILE"] == ""
        assert self.task.config.verifier.env["RESOURCE"] == "bucket-123"
        assert self.config.verifier.env["AWS_REGION"] == "ap-south-1"
        assert "/home/verifier/.aws/creds/PRIMARY.json" in self.agent_environment.files
        assert kwargs["env"] is command_env

    mocker.patch.object(Trial, "_run_shared_verifier", verify)
    await trial._run_shared_verifier(user="verifier", timeout_sec=None, env=command_env)
    assert trial.task.config.verifier.env is task_env
    assert trial.config.verifier.env is override_env
    assert command_env["AWS_PROFILE"] == "COMMAND"
    assert trial.agent_environment.default_user is None


@pytest.mark.asyncio
async def test_verifier_does_not_resolve_shadowed_values(tmp_path, fake_creds, mocker, monkeypatch):
    from harbor.utils.env import resolve_env_vars

    trial = _make_trial(tmp_path)
    original_task = {"HOME": "{{UNUSED_HOME}}", "VALUE": "{{UNUSED_VALUE}}"}
    original_override = {"HOME": "/home/verifier", "VALUE": "${LIVE_VALUE}"}
    trial.task.config.verifier.env = original_task
    trial.config.verifier.env = original_override
    monkeypatch.setenv("LIVE_VALUE", "${LITERAL_VALUE}")
    monkeypatch.delenv("LITERAL_VALUE", raising=False)

    async def verify(self, **kwargs):
        merged = {
            **self.task.config.verifier.env,
            **(kwargs["env"] or {}),
            **self.config.verifier.env,
        }
        resolved = resolve_env_vars(merged)
        assert resolved["HOME"] == "/home/verifier"
        assert resolved["VALUE"] == "${LITERAL_VALUE}"
        assert "/home/verifier/.aws/creds/PRIMARY.json" in self.agent_environment.files

    mocker.patch.object(Trial, "_run_shared_verifier", verify)
    await trial._run_shared_verifier(
        user="verifier", timeout_sec=None, env={"VALUE": "{{ALSO_UNUSED}}"}
    )
    assert trial.task.config.verifier.env is original_task
    assert trial.config.verifier.env is original_override


@pytest.mark.asyncio
async def test_agent_home_override_matches_credential_owner(tmp_path, fake_creds, mocker):
    trial = _make_trial(tmp_path)
    trial.agent._extra_env = {"HOME": "/home/agent"}  # type: ignore[attr-defined]

    async def run(self, **kwargs):
        assert self.agent_environment.default_user == "runner"
        assert "/home/agent/.aws/creds/PRIMARY.json" in self.agent_environment.files
        assert self.agent._extra_env["HOME"] == "/home/agent"
        assert kwargs["user"] == "runner"

    mocker.patch.object(Trial, "_run_agent_phase", run)
    await trial._run_agent_phase(
        instruction="x", target=MagicMock(), timeout_sec=None, user="runner"
    )
    assert trial.agent._extra_env == {"HOME": "/home/agent"}  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_recover_outputs_salvages_without_stopping_env(tmp_path, fake_creds, mocker):
    """_recover_outputs syncs output + collects artifacts but does NOT stop the env.

    Leaving the stop to _finalize is what lets the cancellation signal emit before
    the long post-invoke runs, so the override must not call the base stop here.
    """
    trial = _make_trial(tmp_path)
    trial._result = MagicMock()  # _recover_outputs reads self.result (property over _result)
    sync = mocker.patch.object(trial, "_sync_agent_output", AsyncMock())
    collect = mocker.patch.object(trial, "_collect_artifacts", AsyncMock())
    stop = mocker.patch.object(trial, "_stop_agent_environment", AsyncMock())

    await trial._recover_outputs()

    sync.assert_awaited_once()
    collect.assert_awaited_once()
    stop.assert_not_awaited()  # the stop (and its post-invoke) is deferred to _finalize


# --- pre-invoke runs once in _prepare -------------------------------------


@pytest.mark.asyncio
async def test_prepare_runs_pre_invoke_once_and_seeds_placeholders(tmp_path, fake_creds, mocker):
    trial = _make_trial(
        tmp_path, exports={"PRIMARY": {"Seed": "v"}}, pre_invoke=_phase(), has_pre_script=True
    )
    runner = mocker.patch.object(aws_trial, "ScriptRunner", autospec=True)
    runner.return_value.run = AsyncMock(return_value={"BucketName": "from-pre-invoke"})

    mocker.patch.object(Trial, "_prepare", AsyncMock())
    await trial._prepare()

    runner.return_value.run.assert_awaited_once()
    # Single account tag; pre-invoke's flat output merges under the sole tag.
    assert trial._aws_placeholders["PRIMARY"]["BucketName"] == "from-pre-invoke"
    assert trial._aws_placeholders["PRIMARY"]["Seed"] == "v"  # seed preserved


@pytest.mark.asyncio
async def test_pre_invoke_region_not_overridden_by_scenario_pin(tmp_path, fake_creds, mocker):
    """The region pin is agent-only; a pre-invoke task.toml AWS_REGION survives."""
    trial = _make_trial(
        tmp_path, pre_invoke=_phase(env={"AWS_REGION": "us-west-2"}), has_pre_script=True
    )
    trial.config.regions = ["us-east-1", "us-west-2"]  # type: ignore[attr-defined]
    runner = mocker.patch.object(aws_trial, "ScriptRunner", autospec=True)
    runner.return_value.run = AsyncMock(return_value={})
    mocker.patch.object(Trial, "_prepare", AsyncMock())

    await trial._prepare()

    assert runner.call_args.kwargs["override_env"]["AWS_REGION"] == "us-west-2"


@pytest.mark.asyncio
async def test_pre_invoke_placeholder_merges_under_sole_tag(tmp_path, fake_creds, mocker):
    """A bare-key placeholder.json output merges under the sole account tag.

    Locks the pre-invoke merge: with one account tag, ScriptRunner reads a flat
    placeholder.json ({name: value}) and each bare key folds under that tag,
    landing alongside the seeded exports rather than at the top level.
    """
    trial = _make_trial(
        tmp_path, exports={"PRIMARY": {"Static": "s"}}, pre_invoke=_phase(), has_pre_script=True
    )
    runner = mocker.patch.object(aws_trial, "ScriptRunner", autospec=True)
    # _prepare reads the pre-invoke output from placeholder.json (flat {name: value}).
    run = AsyncMock(return_value={"Computed": "c"})
    runner.return_value.run = run
    mocker.patch.object(Trial, "_prepare", AsyncMock())

    await trial._prepare()

    run.assert_awaited_once_with(output_file_name=aws_trial.PLACEHOLDER_OUTPUT_FILE_NAME)
    assert trial._aws_placeholders == {"PRIMARY": {"Static": "s", "Computed": "c"}}


@pytest.mark.asyncio
async def test_pre_invoke_placeholder_override_fails_loud(tmp_path, fake_creds, mocker):
    """A pre-invoke key colliding with a seeded export raises rather than overwriting.

    The merge uses raise_on_override=True (default), so a script that redefines an
    existing export fails _prepare instead of silently shadowing the seeded value.
    """
    from aws_bench.utils.placeholders import PlaceholderOverrideError

    trial = _make_trial(
        tmp_path,
        exports={"PRIMARY": {"BucketName": "seed"}},
        pre_invoke=_phase(),
        has_pre_script=True,
    )
    runner = mocker.patch.object(aws_trial, "ScriptRunner", autospec=True)
    runner.return_value.run = AsyncMock(return_value={"BucketName": "override"})
    mocker.patch.object(Trial, "_prepare", AsyncMock())

    with pytest.raises(PlaceholderOverrideError):
        await trial._prepare()


@pytest.mark.asyncio
async def test_pre_invoke_qualified_key_merges_under_named_tag(tmp_path, fake_creds, mocker):
    """A TAG::name pre-invoke key folds under that named tag (multi-account seam).

    Dormant under single-account, but the merge must route a qualified key to the
    named tag rather than the sole/first one.
    """
    trial = _make_trial(
        tmp_path,
        exports={"PRIMARY": {"Static": "s"}, "SECONDARY": {}},
        pre_invoke=_phase(),
        has_pre_script=True,
    )
    runner = mocker.patch.object(aws_trial, "ScriptRunner", autospec=True)
    runner.return_value.run = AsyncMock(return_value={"SECONDARY::Computed": "c"})
    mocker.patch.object(Trial, "_prepare", AsyncMock())

    await trial._prepare()

    assert trial._aws_placeholders == {
        "PRIMARY": {"Static": "s"},
        "SECONDARY": {"Computed": "c"},
    }


@pytest.mark.asyncio
async def test_prepare_logs_resolved_placeholders(tmp_path, fake_creds, mocker):
    """Pre-invoke placeholders are logged in {{KEY}}=value form (non-secret).

    This is the only record of what {{...}} values reach instruction/verifier
    scripts, so an unresolved placeholder is visible in the log instead of
    surfacing as an opaque downstream error.
    """
    trial = _make_trial(
        tmp_path, exports={"PRIMARY": {"Seed": "v"}}, pre_invoke=_phase(), has_pre_script=True
    )
    runner = mocker.patch.object(aws_trial, "ScriptRunner", autospec=True)
    runner.return_value.run = AsyncMock(return_value={"BucketName": "from-pre-invoke"})
    mocker.patch.object(Trial, "_prepare", AsyncMock())

    await trial._prepare()

    # self.logger is a MagicMock in this harness; assert the rendered message
    # (the %-args are formatted lazily, so reconstruct from the call).
    logged = "\n".join(
        call.args[0] % call.args[1:]
        for call in trial.logger.debug.call_args_list  # type: ignore[attr-defined]
    )
    assert "{{BucketName}}=from-pre-invoke" in logged


@pytest.mark.asyncio
async def test_prepare_runs_pre_invoke_after_super_starts_container(tmp_path, fake_creds, mocker):
    """Pre-invoke runs AFTER super()._prepare() (which starts the container).

    Pre-invoke uploads and executes a script inside the container, so it must
    not run before the container is up.
    """
    trial = _make_trial(tmp_path, pre_invoke=_phase(), has_pre_script=True)

    order: list[str] = []
    runner = mocker.patch.object(aws_trial, "ScriptRunner", autospec=True)

    async def _run(*a, **k):
        order.append("pre_invoke")
        return {}

    runner.return_value.run = AsyncMock(side_effect=_run)

    async def fake_super_prepare(self):
        order.append("super_prepare")

    mocker.patch.object(Trial, "_prepare", fake_super_prepare)
    await trial._prepare()

    assert order == ["super_prepare", "pre_invoke"]


@pytest.mark.asyncio
async def test_prepare_per_trial_placeholder_isolation(tmp_path, fake_creds, mocker):
    """Two trials with the same exports dict don't leak pre-invoke output; source unchanged."""
    shared_exports = {"PRIMARY": {"Seed": "v"}}
    t1 = _make_trial(
        tmp_path / "a", exports=shared_exports, pre_invoke=_phase(), has_pre_script=True
    )
    t2 = _make_trial(
        tmp_path / "b", exports=shared_exports, pre_invoke=_phase(), has_pre_script=True
    )

    runner = mocker.patch.object(aws_trial, "ScriptRunner", autospec=True)
    runner.return_value.run = AsyncMock(return_value={"Out": "t1only"})
    mocker.patch.object(Trial, "_prepare", AsyncMock())

    await t1._prepare()
    runner.return_value.run = AsyncMock(return_value={})
    await t2._prepare()

    assert "Out" in t1._aws_placeholders["PRIMARY"]
    assert "Out" not in t2._aws_placeholders["PRIMARY"]
    assert shared_exports == {"PRIMARY": {"Seed": "v"}}  # source dict untouched


# --- contamination gate in _prepare ---------------------------------------


@pytest.mark.asyncio
async def test_prepare_blocks_on_contaminated_account(tmp_path, mocker):
    """A mapped account carrying the contamination tag fails _prepare before any work."""
    trial = _make_trial(tmp_path)
    trial.config.scenario_id = "scn-a"  # type: ignore[attr-defined]
    acct = mocker.MagicMock()
    acct.get_contaminated_accounts.return_value = ["123456789012"]
    # __init__ (and _make_trial mirroring it) builds the manager at construction,
    # before this test's mock exists; inject it onto the already-built instance.
    trial._account_manager = acct
    # Base _prepare must never run: the gate short-circuits before container work.
    base_prepare = mocker.patch.object(Trial, "_prepare", AsyncMock())

    with pytest.raises(AccountContaminatedError):
        await trial._prepare()

    base_prepare.assert_not_awaited()


@pytest.mark.asyncio
async def test_prepare_proceeds_when_clean(tmp_path, mocker):
    """A clean account passes the gate and _prepare proceeds to the base setup."""
    trial = _make_trial(tmp_path)
    trial.config.scenario_id = "scn-a"  # type: ignore[attr-defined]
    acct = mocker.MagicMock()
    acct.get_contaminated_accounts.return_value = []
    # __init__ (and _make_trial mirroring it) builds the manager at construction,
    # before this test's mock exists; inject it onto the already-built instance.
    trial._account_manager = acct
    # Stop the base _prepare from doing real container/agent work.
    base_prepare = mocker.patch.object(Trial, "_prepare", AsyncMock())

    await trial._prepare()  # no raise

    acct.get_contaminated_accounts.assert_called_once_with(["123456789012"])
    base_prepare.assert_awaited_once()


@pytest.mark.asyncio
async def test_prepare_skips_contamination_when_verify_env_false(tmp_path, mocker):
    """With verify_env=False (--no-verify-env), _prepare skips the contamination gate."""
    trial = _make_trial(tmp_path)
    trial.config.verify_env = False  # type: ignore[attr-defined]
    trial.config.scenario_id = "scn-a"  # type: ignore[attr-defined]
    acct = mocker.MagicMock()
    acct.get_contaminated_accounts.return_value = ["123456789012"]
    trial._account_manager = acct
    base_prepare = mocker.patch.object(Trial, "_prepare", AsyncMock())

    # Despite a contaminated account, _prepare does NOT raise.
    await trial._prepare()

    acct.get_contaminated_accounts.assert_not_called()
    base_prepare.assert_awaited_once()


# --- post-invoke runs once before the environment stops -------------------


@pytest.mark.asyncio
async def test_stop_runs_post_invoke_before_super_stop(tmp_path, fake_creds, mocker):
    trial = _make_trial(tmp_path, post_invoke=_phase(), has_post_script=True)
    trial._aws_placeholders = {}

    order: list[str] = []
    runner = mocker.patch.object(aws_trial, "ScriptRunner", autospec=True)

    async def _run(*a, **k):
        order.append("post_invoke")
        return {}

    runner.return_value.run = AsyncMock(side_effect=_run)

    async def fake_super_stop(self):
        order.append("stop_env")

    mocker.patch.object(Trial, "_stop_agent_environment", fake_super_stop)
    await trial._stop_agent_environment()
    assert order == ["post_invoke", "stop_env"]


@pytest.mark.asyncio
async def test_stop_runs_post_invoke_only_once(tmp_path, fake_creds, mocker):
    trial = _make_trial(tmp_path, post_invoke=_phase(), has_post_script=True)
    trial._aws_placeholders = {}
    calls: list[str] = []
    runner = mocker.patch.object(aws_trial, "ScriptRunner", autospec=True)
    runner.return_value.run = AsyncMock(side_effect=lambda *a, **k: calls.append("x") or {})

    mocker.patch.object(Trial, "_stop_agent_environment", AsyncMock())
    await trial._stop_agent_environment()
    await trial._stop_agent_environment()
    assert calls == ["x"]


@pytest.mark.asyncio
async def test_stop_skips_post_invoke_when_env_never_started(tmp_path, fake_creds, mocker):
    """No post-invoke when the container never started (e.g. cancelled mid-build).

    Post-invoke runs a script in the agent container; without one it fails with
    "no container found". The base teardown still runs.
    """
    trial = _make_trial(tmp_path, post_invoke=_phase(), has_post_script=True)
    trial._aws_placeholders = {}
    trial._agent_container_started = False  # build never produced a running container
    runner = mocker.patch.object(aws_trial, "ScriptRunner", autospec=True)
    stopped = mocker.patch.object(Trial, "_stop_agent_environment", AsyncMock())

    await trial._stop_agent_environment()

    runner.return_value.run.assert_not_awaited()  # post-invoke skipped
    stopped.assert_awaited_once()  # base teardown still ran


@pytest.mark.asyncio
async def test_setup_agent_environment_marks_started(tmp_path, fake_creds, mocker):
    """_agent_container_started flips True only after the base start returns."""
    trial = _make_trial(tmp_path)
    trial._agent_container_started = False

    started_when_super_ran: list[bool] = []

    async def fake_super(self):
        # The flag must still be False during the start, set only after it returns.
        started_when_super_ran.append(trial._agent_container_started)

    mocker.patch.object(Trial, "_setup_agent_environment", fake_super)
    await trial._setup_agent_environment()

    assert started_when_super_ran == [False]  # not set before super completed
    assert trial._agent_container_started is True  # set after


@pytest.mark.asyncio
async def test_post_invoke_failure_records_exception_and_still_stops(tmp_path, fake_creds, mocker):
    """A failed post-invoke (account reset) is recorded on the result, and teardown proceeds.

    Post-invoke must not block the container teardown, but the failure cannot be
    swallowed silently — the account is left dirty, so the trial records it.
    """
    trial = _make_trial(tmp_path, post_invoke=_phase(), has_post_script=True)
    trial._aws_placeholders = {}
    boom = RuntimeError("reset failed")
    runner = mocker.patch.object(aws_trial, "ScriptRunner", autospec=True)
    runner.return_value.run = AsyncMock(side_effect=boom)
    recorded = mocker.patch.object(trial, "_record_exception")

    stopped = mocker.patch.object(Trial, "_stop_agent_environment", AsyncMock())
    await trial._stop_agent_environment()

    recorded.assert_called_once_with(boom)  # failure surfaced on the result
    stopped.assert_awaited_once()  # teardown still ran


@pytest.mark.asyncio
async def test_post_invoke_cancellation_records_without_reraising(tmp_path, fake_creds, mocker):
    """A cancelled post-invoke is recorded but not re-raised, so finalization continues.

    Re-raising here would propagate out of Harbor's _finalize and skip the
    result.json write and END emit. Trial.run re-raises the originating
    cancellation itself, so the run still stops.
    """
    import asyncio

    trial = _make_trial(tmp_path, post_invoke=_phase(), has_post_script=True)
    trial._aws_placeholders = {}
    runner = mocker.patch.object(aws_trial, "ScriptRunner", autospec=True)
    runner.return_value.run = AsyncMock(side_effect=asyncio.CancelledError())
    recorded = mocker.patch.object(trial, "_record_exception")
    stopped = mocker.patch.object(Trial, "_stop_agent_environment", AsyncMock())

    await trial._stop_agent_environment()  # must NOT raise

    recorded.assert_called_once()
    assert isinstance(recorded.call_args.args[0], asyncio.CancelledError)
    stopped.assert_awaited_once()  # base teardown still ran


def test_init_sets_aws_attrs_before_base_init(tmp_path, mocker):
    """The AWS attrs are set before super().__init__.

    A base-init failure still leaves them readable on the half-built trial, which the
    run()-finally teardown path reads.
    """

    def raising_init(self, config, *, _task=None):
        raise RuntimeError("base init blew up")

    mocker.patch.object(SingleStepTrial, "__init__", raising_init)

    # Construct via __new__ + explicit __init__ so the partially-built instance is
    # still in hand after the base init raises.
    trial = AwsBenchSingleStepTrial.__new__(AwsBenchSingleStepTrial)
    with pytest.raises(RuntimeError, match="base init blew up"):
        trial.__init__(MagicMock(), _task=MagicMock())

    # The subclass set these before delegating to the (now-failed) base init.
    assert trial._aws_placeholders == {}
    assert trial._aws_post_invoke_done is False


# --- post-trial reset -------------------------------------------------------


def _reset_ready_trial(tmp_path, *, mode):
    """A trial with just the fields _reset_scenario_account / run read."""
    from aws_bench.scenario.locator import ScenarioConfig

    trial = AwsBenchSingleStepTrial.__new__(AwsBenchSingleStepTrial)
    trial.config = SimpleNamespace(  # type: ignore[assignment]
        scenario=ScenarioConfig(name="scn-a", path=tmp_path / "scenarios/scn-a"),
        scenario_id="scn-a",
        concurrency_mode=mode,
        account_mapping={"PRIMARY": "111111111111"},
        timeout_multiplier=1.0,
        trial_name="trial-0",
    )
    trial.paths = SimpleNamespace(trial_dir=tmp_path / "trial")  # type: ignore[assignment]
    trial.logger = MagicMock()
    return trial


@pytest.mark.asyncio
async def test_run_triggers_reset_for_mutating(tmp_path, mocker):
    """A MUTATING trial runs reset after super().run() returns."""
    from aws_bench.dataset.task_config import ConcurrencyMode

    trial = _reset_ready_trial(tmp_path, mode=ConcurrencyMode.MUTATING)
    bench_result = SimpleNamespace(exception_info=None)
    mocker.patch.object(SingleStepTrial, "run", AsyncMock(return_value=bench_result))
    reset = mocker.patch.object(trial, "_reset_scenario_account", AsyncMock())

    result = await trial.run()

    assert result is bench_result
    reset.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_run_skips_reset_for_read_only(tmp_path, mocker):
    """A READ_ONLY trial never resets."""
    from aws_bench.dataset.task_config import ConcurrencyMode

    trial = _reset_ready_trial(tmp_path, mode=ConcurrencyMode.READ_ONLY)
    mocker.patch.object(SingleStepTrial, "run", AsyncMock(return_value=SimpleNamespace()))
    reset = mocker.patch.object(trial, "_reset_scenario_account", AsyncMock())

    await trial.run()

    reset.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_skips_reset_on_cancel(tmp_path, mocker):
    """Cancellation from super().run() propagates before reset runs."""
    from aws_bench.dataset.task_config import ConcurrencyMode

    trial = _reset_ready_trial(tmp_path, mode=ConcurrencyMode.MUTATING)
    mocker.patch.object(SingleStepTrial, "run", AsyncMock(side_effect=asyncio.CancelledError()))
    reset = mocker.patch.object(trial, "_reset_scenario_account", AsyncMock())

    with pytest.raises(asyncio.CancelledError):
        await trial.run()
    reset.assert_not_awaited()


@pytest.mark.asyncio
async def test_reset_scenario_account_builds_config_and_runs_reset(tmp_path, mocker):
    from aws_bench.dataset.task_config import ConcurrencyMode
    from aws_bench.scenario.events import ScenarioPhase

    trial = _reset_ready_trial(tmp_path, mode=ConcurrencyMode.MUTATING)
    scenario_trial = MagicMock()
    scenario_trial.run = AsyncMock(return_value=SimpleNamespace(success=True))
    create = mocker.patch(
        "aws_bench.task.aws_trial.ScenarioTrial.create",
        AsyncMock(return_value=scenario_trial),
    )
    mocker.patch("aws_bench.task.aws_trial.CredentialProvider.get", return_value="CREDS")

    await trial._reset_scenario_account()

    reset_cfg = create.call_args.args[0]
    assert reset_cfg.trial_name == "scenario-reset-trial-0"
    assert reset_cfg.labels == {"awsbench.role": "scenario-reset"}
    assert reset_cfg.output_dir == trial.paths.trial_dir
    assert reset_cfg.scenario is trial.config.scenario
    assert reset_cfg.account_mapping == {"PRIMARY": "111111111111"}
    assert create.call_args.args[1] == "CREDS"
    scenario_trial.run.assert_awaited_once_with(ScenarioPhase.RESET)


@pytest.mark.asyncio
async def test_concurrent_resets_from_different_trials_get_distinct_names(tmp_path, mocker):
    """Two overlapping resets of different trials must not share a container name.

    Regression guard for the fixed-name collision (F3): the reset trial name is
    suffixed with the invoking trial name, so the derived scenario container name
    (``awsbench-<trial_name>``) is unique per reset and overlapping resets on one
    Docker daemon no longer force-remove each other on start.
    """
    from aws_bench.dataset.task_config import ConcurrencyMode
    from aws_bench.scenario.container import sanitize_container_name

    trial_a = _reset_ready_trial(tmp_path / "a", mode=ConcurrencyMode.MUTATING)
    trial_a.config.trial_name = "task-alpha__AAAAAAA"
    trial_b = _reset_ready_trial(tmp_path / "b", mode=ConcurrencyMode.MUTATING)
    trial_b.config.trial_name = "task-beta__BBBBBBB"

    scenario_trial = MagicMock()
    scenario_trial.run = AsyncMock(return_value=SimpleNamespace(success=True))
    create = mocker.patch(
        "aws_bench.task.aws_trial.ScenarioTrial.create",
        AsyncMock(return_value=scenario_trial),
    )
    mocker.patch("aws_bench.task.aws_trial.CredentialProvider.get", return_value="CREDS")

    await asyncio.gather(
        trial_a._reset_scenario_account(),
        trial_b._reset_scenario_account(),
    )

    captured = [call.args[0] for call in create.call_args_list]
    assert len(captured) == 2
    # Each reset trial name derives from its invoking trial, so the two differ.
    assert {c.trial_name for c in captured} == {
        f"scenario-reset-{trial_a.config.trial_name}",
        f"scenario-reset-{trial_b.config.trial_name}",
    }
    # The Docker container names ScenarioTrial derives from these are distinct.
    names = {sanitize_container_name(f"awsbench-{c.trial_name}") for c in captured}
    assert len(names) == 2
    # Both stay within Docker's 128-char container-name cap.
    assert all(len(n) <= 128 for n in names)
    # Both still carry the role label so ops tooling can match by role, not name.
    assert all(c.labels == {"awsbench.role": "scenario-reset"} for c in captured)


@pytest.mark.asyncio
async def test_reset_scenario_account_logs_on_failure(tmp_path, mocker):
    from aws_bench.dataset.task_config import ConcurrencyMode

    trial = _reset_ready_trial(tmp_path, mode=ConcurrencyMode.MUTATING)
    scenario_trial = MagicMock()
    scenario_trial.run = AsyncMock(return_value=SimpleNamespace(success=False))
    mocker.patch(
        "aws_bench.task.aws_trial.ScenarioTrial.create",
        AsyncMock(return_value=scenario_trial),
    )
    mocker.patch("aws_bench.task.aws_trial.CredentialProvider.get", return_value="CREDS")

    await trial._reset_scenario_account()  # no raise

    assert trial.logger.error.called  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_reset_scenario_account_swallows_exception(tmp_path, mocker):
    from aws_bench.dataset.task_config import ConcurrencyMode

    trial = _reset_ready_trial(tmp_path, mode=ConcurrencyMode.MUTATING)
    mocker.patch(
        "aws_bench.task.aws_trial.ScenarioTrial.create",
        AsyncMock(side_effect=RuntimeError("boom")),
    )
    mocker.patch("aws_bench.task.aws_trial.CredentialProvider.get", return_value="CREDS")

    await trial._reset_scenario_account()  # no raise
    assert trial.logger.error.called  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_reset_scenario_account_reraises_cancel(tmp_path, mocker):
    from aws_bench.dataset.task_config import ConcurrencyMode

    trial = _reset_ready_trial(tmp_path, mode=ConcurrencyMode.MUTATING)
    mocker.patch(
        "aws_bench.task.aws_trial.ScenarioTrial.create",
        AsyncMock(side_effect=asyncio.CancelledError()),
    )
    mocker.patch("aws_bench.task.aws_trial.CredentialProvider.get", return_value="CREDS")

    with pytest.raises(asyncio.CancelledError):
        await trial._reset_scenario_account()
