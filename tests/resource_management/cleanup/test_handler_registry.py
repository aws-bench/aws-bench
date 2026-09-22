"""Tests for aws_bench.resource_management.cleanup.handler_registry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import (
    CUSTOM_DELETION_REGISTRY,
    FAILED_RESOURCE_HANDLERS,
    PRE_DELETE_HOOKS,
    PREPARE_REGISTRY,
    resource_handler,
)
from aws_bench.resource_management.cleanup.models import HandlerResult, HandlerStatus


@pytest.fixture(autouse=True)
def _cleanup_registries():
    """Snapshot registries before each test and restore after."""
    prep_snapshot = dict(PREPARE_REGISTRY)
    del_snapshot = dict(CUSTOM_DELETION_REGISTRY)
    hook_snapshot = dict(PRE_DELETE_HOOKS)
    failed_snapshot = list(FAILED_RESOURCE_HANDLERS)
    yield
    PREPARE_REGISTRY.clear()
    PREPARE_REGISTRY.update(prep_snapshot)
    CUSTOM_DELETION_REGISTRY.clear()
    CUSTOM_DELETION_REGISTRY.update(del_snapshot)
    PRE_DELETE_HOOKS.clear()
    PRE_DELETE_HOOKS.update(hook_snapshot)
    FAILED_RESOURCE_HANDLERS.clear()
    FAILED_RESOURCE_HANDLERS.extend(failed_snapshot)


# -- register --


def test_registers_prepare_handler():
    @resource_handler("AWS::Test::PrepOnly", role="prepare")
    def _prep(resource, session):
        return HandlerResult(resource.identifier, resource.type, "prepare", HandlerStatus.SUCCESS)

    assert "AWS::Test::PrepOnly" in PREPARE_REGISTRY
    assert "AWS::Test::PrepOnly" not in CUSTOM_DELETION_REGISTRY


def test_registers_both_handlers():
    @resource_handler("AWS::Test::Both", role="prepare")
    def _prep(resource, session):
        return HandlerResult(resource.identifier, resource.type, "prepare", HandlerStatus.SUCCESS)

    @resource_handler("AWS::Test::Both", role="delete")
    def _del(resource, session):
        return HandlerResult(resource.identifier, resource.type, "delete", HandlerStatus.SUCCESS)

    assert "AWS::Test::Both" in PREPARE_REGISTRY
    assert "AWS::Test::Both" in CUSTOM_DELETION_REGISTRY


def test_full_delete_calls_prepare_then_delete():
    prepare_fn = MagicMock(
        return_value=HandlerResult(
            resource_id="r1",
            resource_type="AWS::Test::Order",
            action="prepare",
            status=HandlerStatus.SUCCESS,
        )
    )
    delete_fn = MagicMock()

    @resource_handler("AWS::Test::Order", role="prepare")
    def _prep(resource, session):
        return prepare_fn(resource, session)

    @resource_handler("AWS::Test::Order", role="delete")
    def _del(resource, session):
        return delete_fn(resource, session)

    resource = Resource(type="AWS::Test::Order", identifier="r1")
    session = MagicMock()
    CUSTOM_DELETION_REGISTRY["AWS::Test::Order"](resource, session)

    prepare_fn.assert_called_once_with(resource, session)
    delete_fn.assert_called_once_with(resource, session)


def test_full_delete_skips_delete_on_prepare_failure():
    failed_result = HandlerResult(
        resource_id="r1",
        resource_type="AWS::Test::Fail",
        action="prepare",
        status=HandlerStatus.FAILED,
        message="instances could not be deleted",
    )

    @resource_handler("AWS::Test::Fail", role="prepare")
    def _prep(resource, session):
        return failed_result

    delete_fn = MagicMock()

    @resource_handler("AWS::Test::Fail", role="delete")
    def _del(resource, session):
        return delete_fn(resource, session)

    resource = Resource(type="AWS::Test::Fail", identifier="r1")
    session = MagicMock()
    result = CUSTOM_DELETION_REGISTRY["AWS::Test::Fail"](resource, session)

    delete_fn.assert_not_called()
    assert result.status == HandlerStatus.FAILED


def test_full_delete_skips_delete_on_prepare_skipped():
    skipped_result = HandlerResult(
        resource_id="r1",
        resource_type="AWS::Test::Skip",
        action="prepare",
        status=HandlerStatus.SKIPPED,
        message="Resource not found",
    )

    @resource_handler("AWS::Test::Skip", role="prepare")
    def _prep(resource, session):
        return skipped_result

    delete_fn = MagicMock()

    @resource_handler("AWS::Test::Skip", role="delete")
    def _del(resource, session):
        return delete_fn(resource, session)

    resource = Resource(type="AWS::Test::Skip", identifier="r1")
    session = MagicMock()
    result = CUSTOM_DELETION_REGISTRY["AWS::Test::Skip"](resource, session)

    delete_fn.assert_not_called()
    assert result.status == HandlerStatus.SKIPPED


def test_resource_handler_invalid_role_raises():
    """Test that resource_handler raises ValueError for invalid role."""
    with pytest.raises(ValueError, match="Unknown role 'invalid'"):

        @resource_handler("AWS::Test::Invalid", role="invalid")  # type: ignore[call-overload]
        def _handler(resource, session):
            pass


# -- package wiring: every handler module must be imported by __init__ --


def test_all_handler_modules_are_imported_by_init():
    """Every handler module in the package must be imported by ``handlers/__init__``.

    Regression guard for a silent-orphan bug: a handler registers its handlers as
    an *import side-effect* (the ``@resource_handler`` decorator runs at import
    time). A module that exists on disk but is not imported by ``__init__.py``
    never registers — so its resource type falls through to the CCAPI fallback,
    which cannot delete non-CCAPI types (e.g. ``AWS::LakeFormation::Resource``,
    ``AWS::IoT::ThingGroup``), leaving them as orphans that fail cleanup.

    Parses ``__init__.py`` statically (rather than checking ``sys.modules``) so
    the assertion is immune to import side-effects from other tests in the run.
    """
    import ast
    from pathlib import Path

    import aws_bench.resource_management.cleanup.handlers as handlers_pkg

    handlers_dir = Path(handlers_pkg.__file__).parent

    # Names imported via ``from <package> import (a, b, ...)`` in __init__.py.
    tree = ast.parse((handlers_dir / "__init__.py").read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == handlers_pkg.__name__
        for alias in node.names
    }

    # Handler modules on disk (exclude dunder + private helpers like _service_delete).
    on_disk = {p.stem for p in handlers_dir.glob("*.py") if not p.stem.startswith("_")}

    missing = sorted(on_disk - imported)
    assert not missing, (
        f"Handler module(s) {missing} exist but are not imported by handlers/__init__.py. "
        "Their @resource_handler decorators never run, so those resource types fall "
        "through to the CCAPI fallback and leak as orphans. Add them to __init__.py."
    )


def test_previously_unwired_handler_types_are_registered():
    """The iot and lakeformation handler types must be registered (import side-effect).

    These two modules were historically present on disk but omitted from
    ``handlers/__init__.py``, so their delete handlers never registered and the
    resources leaked as orphans. Import the package and assert the types are live.
    """
    import aws_bench.resource_management.cleanup.handlers  # noqa: F401

    assert "AWS::LakeFormation::Resource" in CUSTOM_DELETION_REGISTRY
    assert "AWS::IoT::ThingGroup" in CUSTOM_DELETION_REGISTRY


def test_delete_before_prepare_types_all_have_custom_delete_handler():
    """Every delete-before-prepare barrier type must have a custom delete handler.

    The barrier fails closed: a type in ``DELETE_BEFORE_PREPARE_TYPES`` with no
    ``CUSTOM_DELETION_REGISTRY`` entry comes back as an unattempted failure and
    blocks the whole cleanup wave's prepare/CCAPI pipeline every run. Registration is an
    import side-effect, so import the handlers package before asserting.
    """
    import aws_bench.resource_management.cleanup.handlers  # noqa: F401
    from aws_bench.resource_management.cleanup.resource_cleaner import (
        DELETE_BEFORE_PREPARE_TYPES,
    )

    missing = sorted(DELETE_BEFORE_PREPARE_TYPES - set(CUSTOM_DELETION_REGISTRY))
    assert not missing, (
        f"Barrier type(s) {missing} are in DELETE_BEFORE_PREPARE_TYPES but have no "
        "custom delete handler. The delete-before-prepare barrier would fail closed "
        "on them forever. Register a delete handler or drop them from the barrier set."
    )
