"""Exceptions for the account management module."""

from aws_bench.exceptions import AWSBenchError


class AccountManagementError(AWSBenchError):
    """Base exception for account management errors."""


class NotManagementAccountError(AccountManagementError):
    """Caller is not the management account of the organization."""


class OrganizationNotReadyError(AccountManagementError):
    """Organization exists but has no root."""


class TestEnvironmentNotFoundError(AccountManagementError):
    """Testing environment Organizational Unit (OU) does not exist."""


class AccountCreationError(AccountManagementError):
    """Failed to create a member account."""


class AccountCreationTimeoutError(AccountCreationError):
    """Account creation did not complete within the timeout."""


class EnvironmentNotProvisionedError(AccountManagementError):
    """Accounts have not been provisioned for one or more environment IDs."""


class AccountResolutionError(AccountManagementError):
    """No accounts in the OU match the requested scenario."""


class DuplicateScenarioAccountError(AccountManagementError):
    """One (scenario, account_tag) pair resolved to more than one account."""


class ContaminationStateMissingError(AccountManagementError):
    """The local contamination state file does not exist."""
