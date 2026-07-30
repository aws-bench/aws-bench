"""Tests for AwsBenchTrial — AWS credential injection + placeholder substitution.

Exercises the AwsBenchTrial lifecycle overrides: pre-invoke runs once after the
container starts, placeholders substitute into the agent instruction, agent creds
inject, verifier creds land at the precedence the verifier reads (then restore),
and post-invoke runs before the environment stops. The agent / environment / task
are faked down to what the overrides touch.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from harbor.agents.oracle import OracleAgent
from harbor.trial.single_step import SingleStepTrial
from harbor.trial.trial import Trial

from aws_bench.dataset.models import RoleType, ScriptType
from aws_bench.dataset.task_config import AwsBenchTask
from aws_bench.exceptions import AccountContaminatedError
from aws_bench.task import aws_trial
from aws_bench.task.aws_trial import AwsBenchSingleStepTrial, AwsBenchTrial

TASK_NAME = "org/my-task"


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
    """Stub the cred helper so no STS calls happen; creds are a fixed dict."""
    return mocker.patch.object(
        aws_trial,
        "assume_role_for_script",
        return_value={
            "AWS_ACCESS_KEY_ID": "AKIA",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "AWS_SESSION_TOKEN": "token",
        },
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


def _make_trial(
    tmp_path,
    *,
    exports=None,
    pre_invoke=None,
    post_invoke=None,
    has_pre_script=False,
    has_post_script=False,
):
    """Build an AwsBenchTrial with only the attributes the overrides touch.

    Bypasses __init__ (no Docker/agent factory) — sets the fields the AWS
    overrides read: self.config (account_mapping/exports/verifier/job_id/trial_name),
    self.task (name + config.{scenario,pre_invoke,post_invoke} + has_phase_script),
    self.agent (with _extra_env), self.agent_environment, self.paths, self.logger.

    ``exports`` is tag-keyed (``tag -> {name -> value}``), matching config.exports.
    """
    trial = AwsBenchSingleStepTrial.__new__(AwsBenchSingleStepTrial)

    verifier = SimpleNamespace(env={"REGION": "us-east-1"})
    task_config = SimpleNamespace(
        scenario=_scenario_ref(agent="AgentRole", verifier="VerifierRole"),
        pre_invoke=pre_invoke,
        post_invoke=post_invoke,
        verifier=verifier,
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
        verifier=verifier,
        job_id=None,
        trial_name="trial-0",
        verify_env=True,
    )
    trial.task = task  # type: ignore[assignment]
    # The real agents (oracle / installed) carry _extra_env; the cred-injection
    # override refuses an agent without it, so the double must expose one.
    trial.agent = SimpleNamespace(_extra_env={})  # type: ignore[assignment]
    # The staged-credentials helper execs (write + remove the creds file) on the
    # agent environment, so exec must be awaitable.
    trial.agent_environment = MagicMock()
    trial.agent_environment.exec = AsyncMock(
        return_value=MagicMock(return_code=0, stdout="", stderr="")
    )
    trial.paths = SimpleNamespace(trial_dir=tmp_path / "trial")  # type: ignore[assignment]
    trial.logger = MagicMock()
    # __init__ is bypassed here; set the per-trial state it would establish.
    trial._aws_placeholders = {tag: dict(v) for tag, v in (exports or {}).items()}
    trial._aws_post_invoke_done = False
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
    # The credentials file is written into the container (secrets land on disk).
    write_cmds = [c.kwargs.get("command", "") for c in _exec_calls(trial)]
    assert any("/.aws/credentials" in cmd and "AKIA" in cmd for cmd in write_cmds)


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
async def test_run_agent_phase_removes_creds_file_after_run(tmp_path, fake_creds, mocker):
    """The creds file is removed after the agent run so a later stage cannot read it."""
    trial = _make_trial(tmp_path)
    trial._aws_placeholders = {}

    async def fake_super_phase(self, *, instruction, **kw):
        pass

    mocker.patch.object(Trial, "_run_agent_phase", fake_super_phase)
    await trial._run_agent_phase(target=MagicMock(), instruction="x", timeout_sec=None, user=None)
    cmds = [c.kwargs.get("command", "") for c in _exec_calls(trial)]
    assert any(cmd.startswith("rm -f") and "/.aws/credentials" in cmd for cmd in cmds)


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
    """A non-zero write exit fails the stage instead of running without creds."""
    trial = _make_trial(tmp_path)
    trial.agent_environment.exec = AsyncMock(
        return_value=MagicMock(return_code=1, stdout="", stderr="disk full")
    )

    with pytest.raises(RuntimeError, match="write credentials file"):
        async with trial._staged_credentials(RoleType.AGENT):
            pass


@pytest.mark.asyncio
async def test_staged_credentials_cleanup_failure_does_not_mask_body_error(tmp_path, fake_creds):
    """A failed cleanup is logged, never replacing the body's exception."""
    trial = _make_trial(tmp_path)

    call_count = {"n": 0}

    async def exec_write_ok_then_rm_fails(*, command, user=None, **kw):
        call_count["n"] += 1
        # First call (write) succeeds; the cleanup rm raises.
        if command.startswith("rm -f"):
            raise RuntimeError("container gone")
        return MagicMock(return_code=0, stdout="", stderr="")

    trial.agent_environment.exec = AsyncMock(side_effect=exec_write_ok_then_rm_fails)

    with pytest.raises(ValueError, match="body blew up"):
        async with trial._staged_credentials(RoleType.AGENT):
            raise ValueError("body blew up")

    # The cleanup ran (and failed) but the body's error surfaced, not the rm error.
    trial.logger.warning.assert_called()  # type: ignore[attr-defined]


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
    assert reset_cfg.trial_name == "scenario-reset"
    assert reset_cfg.output_dir == trial.paths.trial_dir
    assert reset_cfg.scenario is trial.config.scenario
    assert reset_cfg.account_mapping == {"PRIMARY": "111111111111"}
    assert create.call_args.args[1] == "CREDS"
    scenario_trial.run.assert_awaited_once_with(ScenarioPhase.RESET)


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
