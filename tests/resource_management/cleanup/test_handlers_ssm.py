"""Tests for the SSM Patch Baseline cleanup handler.

CCAPI cannot delete AWS::SSM::PatchBaseline (InvalidRequestException), so the
custom handler deletes via the SSM service API. AWS-predefined baselines (an ARN
under a different account) must never be touched.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from aws_bench.resource_management.ccapi.models import Resource
from aws_bench.resource_management.cleanup.handler_registry import (
    CUSTOM_DELETION_REGISTRY,
    PREPARE_REGISTRY,
)
from aws_bench.resource_management.cleanup.handlers.ssm import (
    _delete_patch_baseline,
    _prepare_patch_baseline,
)
from aws_bench.resource_management.cleanup.models import HandlerStatus

_TYPE = "AWS::SSM::PatchBaseline"
_CALLER_ACCOUNT = "111122223333"
_AWS_OWNED_ACCOUNT = "733109147000"


def _resource(baseline_id: str) -> Resource:
    return Resource(type=_TYPE, identifier=baseline_id)


def _session(ssm: MagicMock, *, caller_account: str = _CALLER_ACCOUNT) -> MagicMock:
    sts = MagicMock()
    sts.get_caller_identity.return_value = {"Account": caller_account}
    session = MagicMock()
    session.client.side_effect = lambda name, **_kw: {"ssm": ssm, "sts": sts}[name]
    return session


def _patch_group_paginator(ssm: MagicMock, mappings: list[dict]) -> None:
    """Wire ssm.get_paginator('describe_patch_groups') to yield one page of mappings."""
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Mappings": mappings}]
    ssm.get_paginator.return_value = paginator


def test_handler_is_registered():
    assert _TYPE in CUSTOM_DELETION_REGISTRY
    assert _TYPE in PREPARE_REGISTRY


def test_bare_id_is_deleted_without_ownership_lookup():
    ssm = MagicMock()
    # A bare pb-... id is caller-owned by definition; no STS call needed.
    session = MagicMock()
    session.client.return_value = ssm
    result = _delete_patch_baseline(_resource("pb-0123456789abcdef0"), session)
    assert result.status is HandlerStatus.SUCCESS
    ssm.delete_patch_baseline.assert_called_once_with(BaselineId="pb-0123456789abcdef0")


def test_caller_owned_arn_is_deleted():
    ssm = MagicMock()
    arn = f"arn:aws:ssm:us-east-1:{_CALLER_ACCOUNT}:patchbaseline/pb-0123456789abcdef0"
    result = _delete_patch_baseline(_resource(arn), _session(ssm))
    assert result.status is HandlerStatus.SUCCESS
    ssm.delete_patch_baseline.assert_called_once_with(BaselineId=arn)


def test_aws_owned_default_baseline_is_skipped():
    ssm = MagicMock()
    arn = f"arn:aws:ssm:us-east-1:{_AWS_OWNED_ACCOUNT}:patchbaseline/pb-020d361a05defe4ed"
    result = _delete_patch_baseline(_resource(arn), _session(ssm))
    assert result.status is HandlerStatus.SKIPPED
    ssm.delete_patch_baseline.assert_not_called()


def test_already_gone_is_success():
    ssm = MagicMock()
    ssm.delete_patch_baseline.side_effect = ClientError(
        {"Error": {"Code": "DoesNotExistException", "Message": "gone"}}, "DeletePatchBaseline"
    )
    result = _delete_patch_baseline(_resource("pb-0123456789abcdef0"), _session(ssm))
    assert result.status is HandlerStatus.SUCCESS


def test_delete_still_fails_closed_if_in_use_slips_through():
    """Defense in depth: if prepare didn't clear a registration, delete fails closed."""
    ssm = MagicMock()
    ssm.delete_patch_baseline.side_effect = ClientError(
        {"Error": {"Code": "ResourceInUseException", "Message": "registered for a patch group"}},
        "DeletePatchBaseline",
    )
    result = _delete_patch_baseline(_resource("pb-0123456789abcdef0"), _session(ssm))
    assert result.status is HandlerStatus.FAILED


# --- prepare handler: deregister patch groups before delete ---


def test_prepare_deregisters_matching_patch_groups():
    ssm = MagicMock()
    _patch_group_paginator(
        ssm,
        [
            {"PatchGroup": "prod", "BaselineIdentity": {"BaselineId": "pb-0123456789abcdef0"}},
            {"PatchGroup": "web", "BaselineIdentity": {"BaselineId": "pb-0123456789abcdef0"}},
            # A different baseline's registration must be left alone.
            {"PatchGroup": "other", "BaselineIdentity": {"BaselineId": "pb-ffffffffffffffff0"}},
        ],
    )
    result = _prepare_patch_baseline(_resource("pb-0123456789abcdef0"), _session(ssm))
    assert result.status is HandlerStatus.SUCCESS
    groups = {
        c.kwargs["PatchGroup"] for c in ssm.deregister_patch_baseline_for_patch_group.call_args_list
    }
    assert groups == {"prod", "web"}


def test_prepare_matches_arn_identifier_against_bare_pb_id():
    """Incoming ARN identifier matches describe_patch_groups' bare pb-... id."""
    ssm = MagicMock()
    _patch_group_paginator(
        ssm, [{"PatchGroup": "prod", "BaselineIdentity": {"BaselineId": "pb-0123456789abcdef0"}}]
    )
    arn = f"arn:aws:ssm:us-east-1:{_CALLER_ACCOUNT}:patchbaseline/pb-0123456789abcdef0"
    result = _prepare_patch_baseline(_resource(arn), _session(ssm))
    assert result.status is HandlerStatus.SUCCESS
    ssm.deregister_patch_baseline_for_patch_group.assert_called_once_with(
        BaselineId=arn, PatchGroup="prod"
    )


def test_prepare_no_registrations_is_clean_noop():
    ssm = MagicMock()
    _patch_group_paginator(ssm, [])
    result = _prepare_patch_baseline(_resource("pb-0123456789abcdef0"), _session(ssm))
    assert result.status is HandlerStatus.SUCCESS
    ssm.deregister_patch_baseline_for_patch_group.assert_not_called()


def test_prepare_skips_aws_owned_baseline():
    ssm = MagicMock()
    arn = f"arn:aws:ssm:us-east-1:{_AWS_OWNED_ACCOUNT}:patchbaseline/pb-020d361a05defe4ed"
    result = _prepare_patch_baseline(_resource(arn), _session(ssm))
    assert result.status is HandlerStatus.SKIPPED
    ssm.get_paginator.assert_not_called()


def test_prepare_deregister_failure_fails_closed():
    ssm = MagicMock()
    _patch_group_paginator(
        ssm, [{"PatchGroup": "prod", "BaselineIdentity": {"BaselineId": "pb-0123456789abcdef0"}}]
    )
    ssm.deregister_patch_baseline_for_patch_group.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
        "DeregisterPatchBaselineForPatchGroup",
    )
    result = _prepare_patch_baseline(_resource("pb-0123456789abcdef0"), _session(ssm))
    assert result.status is HandlerStatus.FAILED


def test_full_delete_via_registry_prepares_then_deletes():
    """The registered delete chains prepare (deregister groups) then delete."""
    ssm = MagicMock()
    _patch_group_paginator(
        ssm, [{"PatchGroup": "prod", "BaselineIdentity": {"BaselineId": "pb-0123456789abcdef0"}}]
    )
    handler = CUSTOM_DELETION_REGISTRY[_TYPE]
    result = handler(_resource("pb-0123456789abcdef0"), _session(ssm))
    assert result.status is HandlerStatus.SUCCESS
    ssm.deregister_patch_baseline_for_patch_group.assert_called_once_with(
        BaselineId="pb-0123456789abcdef0", PatchGroup="prod"
    )
    ssm.delete_patch_baseline.assert_called_once_with(BaselineId="pb-0123456789abcdef0")
