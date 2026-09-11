"""Tests for the `env terminate` command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from aws_bench.account_management.models import OrgInfo
from aws_bench.cli.main import app

ENV = "aws_bench.cli.env"
runner = CliRunner()


def _make_org_info():
    return OrgInfo(
        org_id="o-abc",
        root_id="r-root1",
        management_account_id="111111111111",
        management_account_email="mgmt@example.com",
    )


class TestEnvTerminate:
    def test_terminate_closes_accounts_and_deletes_ou(self):
        """Terminate closes active accounts and deletes the OU."""
        accounts = [
            {"Id": "222222222222", "Name": "test-acct", "Status": "ACTIVE"},
        ]

        with patch(f"{ENV}.AccountManager") as mock_acct_cls:
            mock_acct = MagicMock()
            mock_acct_cls.return_value = mock_acct
            mock_acct.is_preexisting = False
            mock_acct._org.get_org_info.return_value = _make_org_info()
            mock_acct._require_ou.return_value = "ou-test1"
            mock_acct._org.list_accounts_in_ou.return_value = accounts
            mock_acct.terminate_environment.return_value = {"222222222222": "CLOSED"}

            result = runner.invoke(app, ["env", "terminate", "--env-name", "test-env"], input="y\n")

        assert result.exit_code == 0
        assert "222222222222" in result.stdout
        assert "CLOSED" in result.stdout
        assert "successfully" in result.stdout
        mock_acct.terminate_environment.assert_called_once_with(
            "test-env", no_close=False, account_filter=["222222222222"]
        )

    def test_terminate_cancelled_by_user(self):
        """Terminate aborts when user declines confirmation."""
        accounts = [
            {"Id": "222222222222", "Name": "test-acct", "Status": "ACTIVE"},
        ]

        with patch(f"{ENV}.AccountManager") as mock_acct_cls:
            mock_acct = MagicMock()
            mock_acct_cls.return_value = mock_acct
            mock_acct.is_preexisting = False
            mock_acct._org.get_org_info.return_value = _make_org_info()
            mock_acct._require_ou.return_value = "ou-test1"
            mock_acct._org.list_accounts_in_ou.return_value = accounts

            result = runner.invoke(app, ["env", "terminate", "--env-name", "test-env"], input="n\n")

        assert result.exit_code == 0
        assert "cancelled" in result.stdout.lower()
        mock_acct.terminate_environment.assert_not_called()

    def test_terminate_no_active_accounts_deletes_ou(self):
        """Terminate with no active accounts just deletes the OU."""
        with patch(f"{ENV}.AccountManager") as mock_acct_cls:
            mock_acct = MagicMock()
            mock_acct_cls.return_value = mock_acct
            mock_acct.is_preexisting = False
            mock_acct._org.get_org_info.return_value = _make_org_info()
            mock_acct._require_ou.return_value = "ou-test1"
            mock_acct._org.list_accounts_in_ou.return_value = []

            result = runner.invoke(app, ["env", "terminate", "--env-name", "test-env"])

        assert result.exit_code == 0
        assert "Deleted environment" in result.stdout
        mock_acct._org.detach_all_scps.assert_called_once_with("ou-test1")
        mock_acct._org.delete_organizational_unit.assert_called_once_with("ou-test1")

    def test_terminate_in_preexisting_mode_issues_no_organizations_calls(self):
        """The refusal lands before any mutation: no SCP detach, no OU delete, no lookup."""
        with patch(f"{ENV}.AccountManager") as mock_acct_cls:
            mock_acct = MagicMock()
            mock_acct_cls.return_value = mock_acct
            mock_acct.is_preexisting = True

            result = runner.invoke(app, ["env", "terminate", "--env-name", "acme-benchmark"])

        assert result.exit_code == 1
        assert "not available in pre-existing account mode" in result.stdout
        mock_acct._org.assert_not_called()
        assert not mock_acct._org.method_calls
        mock_acct._require_ou.assert_not_called()
        mock_acct.terminate_environment.assert_not_called()

    def test_terminate_ou_not_found(self):
        """Terminate with nonexistent OU exits with error."""
        from aws_bench.account_management.exceptions import TestEnvironmentNotFoundError

        with patch(f"{ENV}.AccountManager") as mock_acct_cls:
            mock_acct = MagicMock()
            mock_acct_cls.return_value = mock_acct
            mock_acct.is_preexisting = False
            mock_acct._org.get_org_info.return_value = _make_org_info()
            mock_acct._require_ou.side_effect = TestEnvironmentNotFoundError("not found")

            result = runner.invoke(app, ["env", "terminate", "--env-name", "nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()
