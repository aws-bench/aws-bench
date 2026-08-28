"""Tests for the Glue Database cleanup handler."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from aws_bench.account_management.preexisting import ACCOUNT_CONFIG_ENV_VAR
from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import PREPARE_REGISTRY
from aws_bench.resource_management.cleanup.handlers import glue as glue_handler
from aws_bench.resource_management.cleanup.models import HandlerStatus

_TYPE = "AWS::Glue::Database"
_OPS_ROLE = "arn:aws:iam::111122223333:role/cfn-service-execution"
_CALLER_ROLE = "arn:aws:iam::111122223333:role/some-path/OrganizationAccountAccessRole"
_CALLER_STS = "arn:aws:sts::111122223333:assumed-role/OrganizationAccountAccessRole/sess"


@pytest.fixture(autouse=True)
def _no_account_config(monkeypatch):
    """Pin effective_cfn_role() away from the ambient account config.

    It resolves through the account-config env var, so a shell in preexisting mode would
    name a different ops role and fail every assertion here.
    """
    monkeypatch.delenv(ACCOUNT_CONFIG_ENV_VAR, raising=False)


def _resource(name: str = "analytics_db_111122223333_us-east-1") -> Resource:
    return Resource(type=_TYPE, identifier=name)


def _session(lf: MagicMock, caller_arn: str = _CALLER_STS, caller_role: str = _CALLER_ROLE):
    """A session whose lakeformation/sts/iam clients are all controllable."""
    sts = MagicMock()
    sts.get_caller_identity.return_value = {"Account": "111122223333", "Arn": caller_arn}
    iam = MagicMock()
    iam.get_role.return_value = {"Role": {"Arn": caller_role}}
    clients = {"lakeformation": lf, "sts": sts, "iam": iam}
    session = MagicMock()
    session.client.side_effect = lambda service, **_kw: clients[service]
    return session


def _grant_calls(lf: MagicMock) -> set[tuple[str, str]]:
    """(principal, resource-kind) pairs the handler attempted a grant for."""
    return {
        (
            call.kwargs["Principal"]["DataLakePrincipalIdentifier"],
            next(iter(call.kwargs["Resource"])),
        )
        for call in lf.grant_permissions.call_args_list
    }


def _denying(*codes: str, target_kind: str | None = None):
    """A grant_permissions side effect raising `codes[0]`, optionally for one target kind."""

    def _side_effect(**kwargs):
        if target_kind is None or target_kind in kwargs["Resource"]:
            raise ClientError(
                {"Error": {"Code": codes[0], "Message": "denied"}}, "GrantPermissions"
            )
        return {}

    return _side_effect


# -- registry wiring ----------------------------------------------------------


def test_handler_is_registered_as_a_prepare_handler():
    """PREPARE_REGISTRY maps AWS::Glue::Database to this handler."""
    assert PREPARE_REGISTRY[_TYPE] is glue_handler._prepare


def test_registered_handler_grants_when_invoked_through_the_registry():
    lf = MagicMock()
    result = PREPARE_REGISTRY[_TYPE](_resource(), _session(lf))
    assert result.status == HandlerStatus.SUCCESS
    assert lf.grant_permissions.called


# -- grant shape --------------------------------------------------------------


def test_prepare_grants_drop_to_ops_role_and_caller():
    """DROP is granted on the database and its tables, to both deleting principals.

    CloudFormation performs the delete as the ops role and a direct sweep performs it as
    the caller, so both need the grant.
    """
    lf = MagicMock()
    result = glue_handler._prepare(_resource(), _session(lf))

    assert result.status == HandlerStatus.SUCCESS
    assert _grant_calls(lf) == {
        (_OPS_ROLE, "Database"),
        (_OPS_ROLE, "Table"),
        (_CALLER_ROLE, "Database"),
        (_CALLER_ROLE, "Table"),
    }
    assert {tuple(c.kwargs["Permissions"]) for c in lf.grant_permissions.call_args_list} == {
        ("DROP",)
    }


def test_prepare_grant_targets_are_exact():
    """The database and table-wildcard specs are asserted in full.

    A wrong inner parameter name is accepted by a mock but rejected by botocore, so the
    outer key alone is not enough to pin the request shape.
    """
    lf = MagicMock()
    glue_handler._prepare(_resource("mydb"), _session(lf))

    specs = [c.kwargs["Resource"] for c in lf.grant_permissions.call_args_list]
    assert {"Database": {"Name": "mydb"}} in specs
    assert {"Table": {"DatabaseName": "mydb", "TableWildcard": {}}} in specs
    calls = lf.grant_permissions.call_args_list
    assert all(c.kwargs["PermissionsWithGrantOption"] == [] for c in calls)


def test_prepare_dedupes_when_caller_is_the_ops_role():
    """The ops role and caller resolving to one principal yields one grant per target."""
    lf = MagicMock()
    session = _session(
        lf,
        caller_arn="arn:aws:sts::111122223333:assumed-role/cfn-service-execution/sess",
        caller_role=_OPS_ROLE,
    )
    glue_handler._prepare(_resource(), session)

    assert _grant_calls(lf) == {(_OPS_ROLE, "Database"), (_OPS_ROLE, "Table")}


# -- caller principal resolution ---------------------------------------------


def test_prepare_resolves_the_caller_role_through_iam():
    """The caller principal comes from IAM, not from rewriting the sts ARN.

    An assumed-role ARN omits the role's IAM path, so a rebuilt ARN names a role that does
    not exist and Lake Formation rejects the grant.
    """
    lf = MagicMock()
    session = _session(lf, caller_role="arn:aws:iam::111122223333:role/deep/path/Admin")
    glue_handler._prepare(_resource(), session)

    principals = {p for p, _ in _grant_calls(lf)}
    assert "arn:aws:iam::111122223333:role/deep/path/Admin" in principals


def test_prepare_passes_a_user_caller_through_unchanged():
    """A user ARN is already a valid Lake Formation principal."""
    lf = MagicMock()
    user = "arn:aws:iam::111122223333:user/alice"
    glue_handler._prepare(_resource(), _session(lf, caller_arn=user))

    assert user in {p for p, _ in _grant_calls(lf)}


def test_prepare_grants_ops_role_only_when_the_caller_has_no_principal():
    """An identity that is neither a role nor a user is dropped, not guessed at."""
    lf = MagicMock()
    root = "arn:aws:iam::111122223333:root"
    result = glue_handler._prepare(_resource(), _session(lf, caller_arn=root))

    assert result.status == HandlerStatus.SUCCESS
    assert {p for p, _ in _grant_calls(lf)} == {_OPS_ROLE}


def test_prepare_fails_when_the_account_is_unavailable():
    """A caller identity with no Account yields FAILED rather than raising."""
    lf = MagicMock()
    session = _session(lf)
    session.client("sts").get_caller_identity.return_value = {"Arn": _CALLER_STS}

    result = glue_handler._prepare(_resource(), session)
    assert result.status == HandlerStatus.FAILED
    assert not lf.grant_permissions.called


# -- classification -----------------------------------------------------------


def test_prepare_is_skipped_when_only_the_table_grant_lands():
    """Only the table-wildcard grant landing yields SKIPPED, not SUCCESS."""
    lf = MagicMock()
    lf.grant_permissions.side_effect = _denying("AccessDeniedException", target_kind="Database")

    result = glue_handler._prepare(_resource(), _session(lf))
    assert result.status == HandlerStatus.SKIPPED


def test_prepare_continues_after_one_principal_is_rejected():
    """A grant rejected for one principal must not skip the other.

    Some accounts have no ops role, and Lake Formation rejects a grant to a principal that
    does not exist. The caller's grant is the one the direct sweep path needs.
    """
    lf = MagicMock()

    def _side_effect(**kwargs):
        if kwargs["Principal"]["DataLakePrincipalIdentifier"] == _OPS_ROLE:
            raise ClientError(
                {"Error": {"Code": "EntityNotFoundException", "Message": "no principal"}},
                "GrantPermissions",
            )
        return {}

    lf.grant_permissions.side_effect = _side_effect
    result = glue_handler._prepare(_resource(), _session(lf))

    assert result.status == HandlerStatus.SUCCESS
    assert (_CALLER_ROLE, "Database") in _grant_calls(lf)


def test_prepare_database_already_gone_is_skipped():
    """EntityNotFoundException on every target means there is nothing to grant on."""
    lf = MagicMock()
    lf.grant_permissions.side_effect = _denying("EntityNotFoundException")
    assert glue_handler._prepare(_resource(), _session(lf)).status == HandlerStatus.SKIPPED


def test_prepare_without_admin_rights_is_skipped_not_failed():
    """A caller that is not a data lake administrator cannot grant; the delete still runs.

    The caller may already hold DROP, so a denied grant is not a handler failure.
    """
    lf = MagicMock()
    lf.grant_permissions.side_effect = _denying("AccessDeniedException")
    assert glue_handler._prepare(_resource(), _session(lf)).status == HandlerStatus.SKIPPED


def test_prepare_unexpected_error_is_failed_not_skipped():
    """An error outside the benign set is FAILED, so it is not mistaken for a clean skip."""
    lf = MagicMock()
    lf.grant_permissions.side_effect = _denying("InternalServiceException")
    assert glue_handler._prepare(_resource(), _session(lf)).status == HandlerStatus.FAILED


def test_prepare_mixed_errors_are_failed():
    """A benign code alongside an unexpected one is FAILED.

    Classifying on the benign member alone would report a real malfunction as an
    intentional skip.
    """
    lf = MagicMock()
    codes = iter(
        [
            "AccessDeniedException",
            "InternalServiceException",
            "AccessDeniedException",
            "AccessDeniedException",
        ]
    )

    def _side_effect(**_kwargs):
        raise ClientError({"Error": {"Code": next(codes), "Message": "x"}}, "GrantPermissions")

    lf.grant_permissions.side_effect = _side_effect
    assert glue_handler._prepare(_resource(), _session(lf)).status == HandlerStatus.FAILED


def test_prepare_connection_error_is_failed():
    """A Region without a Lake Formation endpoint surfaces as FAILED, never as a raise."""
    lf = MagicMock()
    lf.grant_permissions.side_effect = EndpointConnectionError(endpoint_url="https://lf")
    assert glue_handler._prepare(_resource(), _session(lf)).status == HandlerStatus.FAILED


# -- retry --------------------------------------------------------------------


def test_prepare_retries_transient_contention(monkeypatch):
    """Contention on the permission table is retried, not reported as a failure."""
    monkeypatch.setattr(glue_handler.time, "sleep", lambda _s: None)
    lf = MagicMock()
    attempts = {"n": 0}

    def _side_effect(**_kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ClientError(
                {"Error": {"Code": "ConcurrentModificationException", "Message": "busy"}},
                "GrantPermissions",
            )
        return {}

    lf.grant_permissions.side_effect = _side_effect
    result = glue_handler._prepare(_resource(), _session(lf))

    assert result.status == HandlerStatus.SUCCESS
    assert attempts["n"] == 5  # one retried call plus the remaining three targets


def test_prepare_gives_up_on_persistent_contention(monkeypatch):
    """Contention that never clears is reported, not retried forever."""
    monkeypatch.setattr(glue_handler.time, "sleep", lambda _s: None)
    lf = MagicMock()
    lf.grant_permissions.side_effect = _denying("ConcurrentModificationException")

    result = glue_handler._prepare(_resource(), _session(lf))
    assert result.status == HandlerStatus.FAILED
    assert lf.grant_permissions.call_count == glue_handler._RETRY_ATTEMPTS * 4
