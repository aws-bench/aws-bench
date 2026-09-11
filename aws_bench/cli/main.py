"""aws-bench CLI — standalone Typer app for AWS benchmark evaluation."""

import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer
from typer import Option, Typer

from aws_bench.account_management.preexisting import activate_account_config
from aws_bench.cli.env import env_app
from aws_bench.cli.jobs import jobs_app, start
from aws_bench.cli.view import view
from aws_bench.logging.ledger import LEDGER_META_KEY, open_entry
from aws_bench.utils.concurrent import request_shutdown

app = Typer(
    name="aws-bench",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="aws-bench is an evaluation framework for benchmarking AI agents on real-world AWS tasks. "
    "It provisions disposable AWS test accounts, deploys scenario resources via CDK, "
    "runs an agent against tasks, and scores results.",
)


# Known commands matched against argv to label an invocation; an unrecognized
# one labels as "aws-bench" rather than writing arbitrary argv into command.json
# and the entry dir name. Kept in sync with the registrations by a test.
_KNOWN_COMMANDS = frozenset(
    {
        "run",
        "view",
        "job start",
        "env init",
        "env setup",
        "env show",
        "env list",
        "env snapshot",
        "env cleanup",
        "env verify",
        "env reset",
        "env terminate",
        "env creds",
    }
)
_DEFAULT_LABEL = "aws-bench"


def _command_label(argv: list[str]) -> str:
    """Resolve argv to a known command label, e.g. ``env cleanup`` / ``run``.

    Flags are dropped first, so the read doesn't depend on a fixed argv index.
    Only the first two positionals are considered, so a positional value (``view
    <folder>``) can't be mistaken for the command; an unknown candidate falls
    back to ``aws-bench``.
    """
    positional = [tok for tok in argv[1:] if not tok.startswith("-")]
    if len(positional) >= 2:
        group_leaf = f"{positional[0]} {positional[1]}"
        if group_leaf in _KNOWN_COMMANDS:
            return group_leaf
    if positional and positional[0] in _KNOWN_COMMANDS:
        return positional[0]
    return _DEFAULT_LABEL


def _handle_shutdown(signum, frame):
    """Flag cooperative cancellation, then re-raise as ``KeyboardInterrupt``.

    ``request_shutdown`` lets worker-thread polling loops unwind (a signal only
    reaches the main thread, so they can't be interrupted directly); the
    ``KeyboardInterrupt`` unwinds the async stack so ``finally`` blocks
    (container stop, result persistence) still run.
    """
    request_shutdown()
    raise KeyboardInterrupt


@app.callback()
def _root(
    ctx: typer.Context,
    account_config: Annotated[
        Path | None,
        Option(
            "--account-config",
            envvar="AWSBENCH_ACCOUNT_CONFIG",
            help=(
                "YAML/JSON pre-existing account allowlist. External IaC retains "
                "account, OU, SCP, and termination ownership."
            ),
        ),
    ] = None,
) -> None:
    """Install shutdown handlers and open the command ledger for every command.

    Runs on the main thread before any subcommand and any ``asyncio.run``, so
    SIGINT (Ctrl-C) and SIGTERM (orchestrator/``timeout``) share one handler.
    Opening beat 1 here and registering the beat-3 finalizer on ``call_on_close``
    means every invocation leaves a record, including ones that fail before doing
    any work.
    """
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    if account_config is not None:
        activate_account_config(account_config)

    argv = list(sys.argv)
    entry = open_entry(_command_label(argv), argv, now=datetime.now(timezone.utc))
    ctx.meta[LEDGER_META_KEY] = entry

    def _finalize() -> None:
        # Typer leaves the live exception in sys.exc_info during call_on_close;
        # a clean Exit(0) is a success, everything else classifies in finalize().
        exc = sys.exc_info()[1]
        if isinstance(exc, typer.Exit) and exc.exit_code == 0:
            exc = None
        entry.finalize(exc)

    ctx.call_on_close(_finalize)


app.add_typer(jobs_app, name="job", help="Manage jobs.")
app.add_typer(env_app, name="env", help="Manage the testing environment.")

app.command(name="run", help="Run benchmark trials. Alias for `aws-bench job start`.")(start)
app.command(name="view", help="Start web server to browse trajectory files.")(view)

if __name__ == "__main__":
    app()
