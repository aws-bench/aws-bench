"""Export collection: gather CFN exports and SSM parameters across accounts/regions."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from concurrent.futures import as_completed

from botocore.exceptions import BotoCoreError, ClientError

from aws_bench.account_management.constants import ORG_ACCESS_ROLE
from aws_bench.exceptions import AWSBenchError
from aws_bench.logging.logger import get_logger
from aws_bench.utils.concurrent import build_client, interruptible_executor
from aws_bench.utils.credentials_provider import (
    CredentialProvider,
    build_session_name,
    env_credentials_dict_to_session,
    session_to_env_credentials,
)

logger = get_logger(__name__)

# Cap concurrent (account, region) pulls. boto3 clients are thread-safe
# per-client but the underlying STS / CFN / SSM endpoints throttle, so
# unbounded fan-out hurts latency past ~16-32.
_DEFAULT_MAX_WORKERS = 16


class ExportCollectionError(AWSBenchError):
    """Raised when export collection fails for any (account, region) pair.

    Carries the per-pair failures for diagnostic display; the CLI prints
    them and exits non-zero rather than letting placeholder substitution
    silently use an incomplete map.
    """

    def __init__(self, failures: list[tuple[str, str, str]]) -> None:
        """Build the error from per-pair ``(account_id, region, message)`` tuples."""
        self.failures = failures
        lines = "\n".join(f"  {a}/{r}: {msg}" for a, r, msg in failures)
        super().__init__(
            f"Failed to collect exports from {len(failures)} (account, region) pair(s):\n{lines}"
        )


class ExportCollisionError(AWSBenchError):
    """Raised when two exports resolve to the same name with different values.

    CFN exports and SSM ``/exports`` params share one flat per-account namespace
    that spans every region collected for the account, so a duplicate name would
    make ``{{name}}`` resolve to an arbitrary source. The collector folds this into
    ``ExportCollectionError`` alongside any fetch failures.
    """


def _put_export(dest: dict[str, str], name: str, value: str) -> None:
    """Add ``name -> value`` to ``dest``, failing loud on a conflicting duplicate.

    A repeat with the same value is idempotent (unambiguous); a repeat with a
    different value raises ``ExportCollisionError``.
    """
    if name in dest and dest[name] != value:
        raise ExportCollisionError(
            f"Export name '{name}' is defined more than once with different values."
        )
    dest[name] = value


def _flatten_json_exports(exports: dict[str, str]) -> dict[str, str]:
    """Flatten JSON dict values into {key}-{subkey}: subvalue entries.

    Also preserves the original key with its raw JSON string value so
    placeholders referencing the un-flattened key still resolve. A synthetic
    ``{key}-{subkey}`` that collides with a literal export raises
    ``ExportCollisionError``.
    """
    flat: dict[str, str] = {}
    for key, value in exports.items():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                _put_export(flat, key, value)
                for sub_key, sub_value in parsed.items():
                    _put_export(flat, f"{key}-{sub_key}", str(sub_value))
                continue
        except (json.JSONDecodeError, TypeError):
            pass
        _put_export(flat, key, value)
    return flat


def _collect_one(creds: dict[str, str], region: str) -> dict[str, str]:
    """Read CFN exports + SSM /exports from a single (account, region).

    Builds a thread-local boto3 Session from the frozen credential dict (boto3
    Sessions are not safe to share across threads), then reads CloudFormation
    exports and SSM ``/exports`` String parameters. SSM is read WITHOUT
    decryption: ``/exports`` is a non-secret placeholder namespace, and the values
    flow into the agent instruction and persist in the trial config, so a
    SecureString must never be pulled in here.

    Raises ``ClientError`` / ``BotoCoreError`` directly — the caller aggregates
    per-pair failures into ``ExportCollectionError``.
    """
    session = env_credentials_dict_to_session(creds)
    out: dict[str, str] = {}

    cfn = build_client(session, "cloudformation", region_name=region)
    cfn_paginator = cfn.get_paginator("list_exports")
    for page in cfn_paginator.paginate():
        for export in page.get("Exports", []):
            _put_export(out, export["Name"], export["Value"])

    ssm = build_client(session, "ssm", region_name=region)
    ssm_paginator = ssm.get_paginator("get_parameters_by_path")
    for page in ssm_paginator.paginate(Path="/exports", Recursive=True, WithDecryption=False):
        for param in page.get("Parameters", []):
            _put_export(out, param["Name"].split("/")[-1], param["Value"])
    return out


def collect_account_exports(
    targets: Mapping[str, Iterable[str]],
    role_name: str = ORG_ACCESS_ROLE,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> dict[str, dict[str, str]]:
    """Collect CloudFormation exports + SSM ``/exports`` params from each target.

    Pulls every (account, region) pair concurrently via ``interruptible_executor``.
    Any role-assume or per-pair failure is fatal: ``ExportCollectionError`` is
    raised after every fetch is attempted, listing every failure. The caller
    catches it, prints the failures, and exits non-zero rather than letting
    placeholder substitution proceed with an incomplete map.

    Args:
        targets: Map of ``account_id`` → iterable of regions to query in
            that account, built by joining the scenario→account map with each
            scenario's manifest-declared regions.
        role_name: IAM role to assume in each member account. Defaults to
            ``ORG_ACCESS_ROLE``.
        max_workers: Max concurrent (account, region) reads. Default 16
            balances throttling pressure against latency.

    Raises:
        ExportCollectionError: One or more (account, region) pairs failed to
            assume role, fetch exports, or defined a duplicate export name with
            conflicting values. Includes every failure.

    Returns:
        An account-keyed ``{account_id: {export_name: value}}`` map. JSON-object
        exports are flattened (one entry per sub-key, named ``<export>-<sub-key>``).
        Keying by account namespaces the exports so same-named exports in different
        accounts don't collide, and each trial gets only its account's slice.
    """
    work: list[tuple[str, str]] = sorted(
        {(account_id, region) for account_id, regions in targets.items() for region in regions}
    )
    if not work:
        logger.warning("No (account, region) pairs to query for exports.")
        return {}

    failures: list[tuple[str, str, str]] = []

    # Snapshot frozen credentials once per account (assume-role is cheap and
    # sequential avoids STS throttling). Worker threads each build their own
    # thread-local session from the frozen dict, since boto3 Sessions are not
    # safe to share across threads.
    cred_provider = CredentialProvider.get()
    account_creds: dict[str, dict[str, str]] = {}
    for account_id in sorted({a for a, _ in work}):
        try:
            session = cred_provider.get_session_for_account(
                account_id,
                role_name,
                build_session_name("session", account_id[-6:]),
            )
            account_creds[account_id] = session_to_env_credentials(session)
        except (ClientError, BotoCoreError) as exc:
            failures.append((account_id, "<assume-role>", str(exc)))

    fetchable = [(a, r) for a, r in work if a in account_creds]

    logger.info(
        "Collecting exports across %d (account, region) pair(s) with up to %d workers",
        len(fetchable),
        max_workers,
    )

    per_account: dict[str, dict[str, str]] = {}
    with interruptible_executor(
        max_workers=min(max_workers, max(1, len(fetchable))),
    ) as pool:
        futures = {
            pool.submit(_collect_one, account_creds[account_id], region): (account_id, region)
            for account_id, region in fetchable
        }
        for fut in as_completed(futures):
            account_id, region = futures[fut]
            try:
                dest = per_account.setdefault(account_id, {})
                for name, value in _flatten_json_exports(fut.result()).items():
                    # A repeated name — across CFN/SSM, across regions, or from JSON
                    # flattening — is a silent order-dependent winner without this.
                    _put_export(dest, name, value)
            except (ClientError, BotoCoreError, ExportCollisionError) as exc:
                failures.append((account_id, region, str(exc)))
            except Exception as exc:  # noqa: BLE001
                # Unexpected error — bubble up as a failure with type info.
                failures.append((account_id, region, f"{type(exc).__name__}: {exc}"))

    if failures:
        raise ExportCollectionError(failures)

    return per_account
