"""Tests for export collection in ``aws_bench.resource_management.export_collector``."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aws_bench.resource_management.export_collector import (
    ExportCollectionError,
    ExportCollisionError,
    _put_export,
    collect_account_exports,
)


def test_put_export_raises_on_conflicting_duplicate():
    dest = {"Name": "first"}
    with pytest.raises(ExportCollisionError, match="more than once"):
        _put_export(dest, "Name", "second")


def test_put_export_idempotent_on_same_value():
    dest = {"Name": "v"}
    _put_export(dest, "Name", "v")
    assert dest == {"Name": "v"}


def _fake_cfn_client(exports: list[tuple[str, str]]) -> MagicMock:
    """Build a mock CFN client whose ``list_exports`` paginator yields exports."""
    client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Exports": [{"Name": n, "Value": v} for n, v in exports]}]
    client.get_paginator.return_value = paginator
    return client


def _fake_ssm_client(params: list[tuple[str, str]]) -> MagicMock:
    """Build a mock SSM client whose paginator yields ``/exports`` params."""
    client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Parameters": [{"Name": n, "Value": v} for n, v in params]}]
    client.get_paginator.return_value = paginator
    return client


def _session_returning(per_service_clients: dict[str, MagicMock]) -> MagicMock:
    """Build a mock boto session whose ``.client(svc, region_name=...)`` returns mocks."""
    sess = MagicMock()
    sess.client.side_effect = lambda svc, region_name=None: per_service_clients[svc]
    return sess


def _wire_session(mocker, worker_session: MagicMock) -> MagicMock:
    """Stub the cred chain so worker threads build ``worker_session``.

    ``collect_account_exports`` snapshots frozen creds per account (sequentially),
    then each worker thread reconstructs its own session from that dict via
    ``env_credentials_dict_to_session``. The per-account session that gets
    snapshotted is irrelevant to the assertions; the worker session is what does
    the CFN/SSM reads, so point ``env_credentials_dict_to_session`` at it.

    Returns the patched ``CredentialProvider`` for assert-on-assume-role tests.
    """
    cred = mocker.patch("aws_bench.resource_management.export_collector.CredentialProvider")
    mocker.patch(
        "aws_bench.resource_management.export_collector.session_to_env_credentials",
        return_value={
            "AWS_ACCESS_KEY_ID": "AKIA",
            "AWS_SECRET_ACCESS_KEY": "s",
            "AWS_SESSION_TOKEN": "t",
        },
    )
    mocker.patch(
        "aws_bench.resource_management.export_collector.env_credentials_dict_to_session",
        return_value=worker_session,
    )
    return cred


class TestCollectAccountExports:
    def test_no_work_returns_empty(self, mocker):
        cred = mocker.patch("aws_bench.resource_management.export_collector.CredentialProvider")
        # No regions for the account → no work → empty result.
        result = collect_account_exports(targets={"111111111111": []})
        assert result == {}
        cred.get.assert_not_called()

    def test_collects_cfn_and_ssm_exports_per_account_region(self, mocker):
        cfn = _fake_cfn_client([("CfnExport1", "value1"), ("CfnExport2", "value2")])
        ssm = _fake_ssm_client([("/exports/SsmExport1", "value3")])
        session = _session_returning({"cloudformation": cfn, "ssm": ssm})

        cred = _wire_session(mocker, session)

        result = collect_account_exports(targets={"111111111111": ["us-east-1"]})

        # Account-keyed: each account's exports are namespaced under its id.
        assert result == {
            "111111111111": {
                "CfnExport1": "value1",
                "CfnExport2": "value2",
                "SsmExport1": "value3",
            }
        }
        # The per-account session is snapshotted once for the single account.
        cred.get.return_value.get_session_for_account.assert_called_once()

    def test_dedupes_account_region_pairs(self, mocker):
        cfn = _fake_cfn_client([("E1", "v")])
        ssm = _fake_ssm_client([])
        session = _session_returning({"cloudformation": cfn, "ssm": ssm})

        _wire_session(mocker, session)

        # Same (account, region) appearing twice in the input is deduped
        # (set semantics in the helper).
        collect_account_exports(targets={"111111111111": ["us-east-1", "us-east-1"]})

        # CFN client constructed only once for the dedupe'd pair.
        assert cfn.get_paginator.call_count == 1
        assert ssm.get_paginator.call_count == 1

    def test_raises_when_role_assumption_fails(self, mocker):
        from botocore.exceptions import ClientError

        cred = mocker.patch("aws_bench.resource_management.export_collector.CredentialProvider")
        cred.get.return_value.get_session_for_account.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "nope"}}, "AssumeRole"
        )

        with pytest.raises(ExportCollectionError) as exc_info:
            collect_account_exports(targets={"111111111111": ["us-east-1"]})
        assert exc_info.value.failures
        # Failure carries the assume-role marker.
        assert "<assume-role>" in str(exc_info.value)

    def test_raises_when_pair_fetch_fails(self, mocker):
        from botocore.exceptions import ClientError

        # Session resolution succeeds but CFN.list_exports throws.
        cfn = MagicMock()
        cfn.get_paginator.side_effect = ClientError(
            {"Error": {"Code": "Throttling", "Message": "rate"}}, "ListExports"
        )
        ssm = _fake_ssm_client([])
        session = _session_returning({"cloudformation": cfn, "ssm": ssm})

        _wire_session(mocker, session)

        with pytest.raises(ExportCollectionError) as exc_info:
            collect_account_exports(targets={"111111111111": ["us-east-1"]})
        assert "us-east-1" in str(exc_info.value)
        assert "Throttling" in str(exc_info.value)

    def test_same_name_different_value_within_region_raises(self, mocker):
        # CFN export and SSM /exports param share a name with conflicting values.
        cfn = _fake_cfn_client([("Shared", "from-cfn")])
        ssm = _fake_ssm_client([("/exports/Shared", "from-ssm")])
        session = _session_returning({"cloudformation": cfn, "ssm": ssm})
        _wire_session(mocker, session)

        with pytest.raises(ExportCollectionError) as exc_info:
            collect_account_exports(targets={"111111111111": ["us-east-1"]})
        # The collision is folded into ExportCollectionError with its message.
        assert "Shared" in str(exc_info.value)
        assert "more than once" in str(exc_info.value)

    def test_same_name_same_value_within_region_is_idempotent(self, mocker):
        # Identical value from both sources is unambiguous — no error, single entry.
        cfn = _fake_cfn_client([("Shared", "same")])
        ssm = _fake_ssm_client([("/exports/Shared", "same")])
        session = _session_returning({"cloudformation": cfn, "ssm": ssm})
        _wire_session(mocker, session)

        result = collect_account_exports(targets={"111111111111": ["us-east-1"]})
        assert result == {"111111111111": {"Shared": "same"}}

    def test_same_name_conflicts_across_regions_raises(self, mocker):
        # Same export name in two regions of one account, different values.
        def region_session(region_name=None):
            value = "east" if region_name == "us-east-1" else "west"
            return _session_returning(
                {
                    "cloudformation": _fake_cfn_client([("Dup", value)]),
                    "ssm": _fake_ssm_client([]),
                }
            )

        sess = MagicMock()
        sess.client.side_effect = lambda svc, region_name=None: region_session(region_name).client(
            svc, region_name=region_name
        )
        _wire_session(mocker, sess)

        with pytest.raises(ExportCollectionError) as exc_info:
            collect_account_exports(targets={"111111111111": ["us-east-1", "us-west-2"]})
        assert "Dup" in str(exc_info.value)

    def test_flattens_json_object_exports(self, mocker):
        # CFN export whose value is a JSON object → flattened into
        # <export>-<sub-key> entries alongside the original raw JSON.
        cfn = _fake_cfn_client([("ConfigBundle", '{"DistributionId": "E1", "BucketName": "b"}')])
        ssm = _fake_ssm_client([])
        session = _session_returning({"cloudformation": cfn, "ssm": ssm})

        _wire_session(mocker, session)

        result = collect_account_exports(targets={"111111111111": ["us-east-1"]})
        assert result == {
            "111111111111": {
                "ConfigBundle": '{"DistributionId": "E1", "BucketName": "b"}',
                "ConfigBundle-DistributionId": "E1",
                "ConfigBundle-BucketName": "b",
            }
        }
