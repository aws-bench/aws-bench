"""``env cleanup`` as the step under test, checked against the accounts themselves.

Opt-in only (``make integration``). The shared ``deployed_env`` in
``test_acceptance.py`` runs cleanup as its finalizer and nobody looks at what it
leaves behind. This test deploys its own environment, runs ``env cleanup`` as the
asserted step. It lists CloudFormation stacks in every region the scenario
declares, with credentials obtained the way the CLI obtains them, both before
cleanup (a healthy stack must exist) and after it (none may remain).

The module body can only be exercised by a live run; the helpers it calls are
covered by ``tests/test_integration_helpers.py``. It assumes the account config
maps only the scenario under test, as the ``env show`` test does.

Ordering: this module deploys into the same accounts as ``test_acceptance.py``
(the env name is fixed by the account config), so the two must not overlap. They
do not: pytest finalizes a module's fixtures, including that module's
``env cleanup``, before it starts the next module. Do not run the suite in
parallel or in random order.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.account_management.preexisting import ACCOUNT_CONFIG_ENV_VAR
from aws_bench.utils.credentials_provider import CredentialProvider, build_session_name

from .helpers import (
    INIT_TIMEOUT_SEC,
    IntegSettings,
    SecretEnv,
    cleanup_output_problems,
    collect_account_stacks,
    deployed,
    deployed_stack_problems,
    find_scenario_manifests,
    load_accounts,
    mask_account_ids,
    scenario_regions,
    stack_problems,
    step_timeouts_for,
)

pytestmark = pytest.mark.integration


def test_env_cleanup_happy_path(
    integ_settings: IntegSettings, cli_env: SecretEnv, run_cli, monkeypatch: pytest.MonkeyPatch
):
    """`env cleanup --yes` exits 0, prints its success summary, and leaves no stacks behind.

    Own deployment (`env init`, `env setup`). Before cleanup, a read-only
    ``cloudformation:ListStacks`` in every region the scenario declares must show
    a healthy ``<scenario>-*`` stack in every configured account. Then cleanup
    runs as the asserted step, and the same listing must show only
    ``DELETE_COMPLETE`` stacks, bootstrap stacks included (see
    ``helpers.stack_problems``). A region that cannot be listed fails the test.
    Other resource types are not checked.

    ``deployed`` still guarantees a cleanup attempt on every exit path: it is
    skipped only after every check above passed (``mark_clean``).
    """
    cleanup_args = integ_settings.env_command("cleanup", "--yes", "--quiet")

    def manifests() -> list[Path]:
        return find_scenario_manifests(
            dataset_path=integ_settings.dataset_path,
            scenario=integ_settings.scenario,
            home=Path(cli_env["HOME"]),
        )

    def timeouts():
        found = step_timeouts_for(manifests())
        print(f"step timeouts: {found.describe()}")  # describe() masks account IDs
        return found

    with deployed(
        run_cli,
        init_args=integ_settings.env_command("init", "--quiet"),
        setup_args=integ_settings.env_command("setup", "--quiet"),
        cleanup_args=cleanup_args,
        init_timeout=INIT_TIMEOUT_SEC,
        setup_timeout=lambda: timeouts().setup,
        cleanup_timeout=lambda: step_timeouts_for(manifests()).cleanup,
    ) as deployment:
        accounts = load_accounts(integ_settings.account_config)
        regions = scenario_regions(manifests())

        # Read-only checks use the CLI's own credential path.
        monkeypatch.setenv(ACCOUNT_CONFIG_ENV_VAR, str(integ_settings.account_config))
        provider = CredentialProvider()

        def session_for(account_id: str):
            return provider.get_session_for_account(
                account_id, ORG_ACCESS_ROLE, build_session_name("session")
            )

        def list_all():
            return [
                collect_account_stacks(
                    scenario=scenario,
                    tag=tag,
                    account_id=account_id,
                    regions=regions.get(scenario, []),
                    session_for=session_for,
                )
                for scenario, tags in accounts.items()
                for tag, account_id in tags.items()
            ]

        # Lower bound: a healthy <scenario>-* stack exists before cleanup, so an empty account
        # afterwards means it was removed. It does not show that env setup created it.
        before = deployed_stack_problems(list_all(), accounts=accounts, regions_by_scenario=regions)
        assert before == [], "no deployment to clean up before env cleanup:\n" + "\n".join(before)

        # The step under test.
        result = run_cli(*cleanup_args, timeout=step_timeouts_for(manifests()).cleanup)
        n_accounts = sum(len(tags) for tags in accounts.values())
        # Not `assert f(raw) == []`: pytest would print the raw output as call arguments.
        stdout = mask_account_ids(result.stdout)
        summary_problems = cleanup_output_problems(stdout, accounts=n_accounts)
        assert summary_problems == [], stdout[-4000:]

        problems = stack_problems(list_all(), accounts=accounts, regions_by_scenario=regions)
        assert problems == [], "env cleanup left stacks behind:\n" + "\n".join(problems)
        deployment.mark_clean()
