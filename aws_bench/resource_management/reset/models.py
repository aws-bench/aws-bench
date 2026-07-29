"""Data models for reset operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


@dataclass
class ResetResult:
    """Result of account reset operation.

    ``needs_redeploy`` is set when a stack could not be reverted in place and
    was deleted. The scenario trial automatically handles redeployment by
    running the DEPLOY phase after reset completes. ``redeploy_succeeded``
    records the outcome — ``None`` when no redeploy was attempted,
    otherwise the boolean result.

    A non-empty ``unresolved_orphans`` maps a resource type to identifiers that
    survived reset (or a baseline type it could not enumerate): the account is
    NOT clean and must never be recaptured as a baseline.
    """

    success: bool
    reason: str
    account_id: str = ""
    scenario_name: str = ""
    details: dict[str, Any] | None = None
    suggestion: str | None = None
    needs_redeploy: bool = False
    redeploy_succeeded: bool | None = None
    unresolved_orphans: dict[str, list[dict]] | None = None


class ResetFailure(Exception):
    """Raised when reset operation fails."""

    def __init__(self, reason: str, details: Any = None, suggestion: str | None = None):
        """Initialize reset failure exception.

        Args:
            reason: Human-readable failure reason
            details: Structured data about the failure
            suggestion: Actionable next step
        """
        self.reason = reason
        self.details = details
        self.suggestion = suggestion

        message = reason
        if suggestion:
            message = f"{reason}\nSuggestion: {suggestion}"

        super().__init__(message)


class ChangesetResult(Enum):
    """Result of changeset creation and validation."""

    READY_TO_EXECUTE = auto()  # Changeset created successfully, ready to execute
    ALREADY_BASELINE = auto()  # No changes needed, already at baseline
    FAILED = auto()  # Changeset creation or validation failed


class RestoreOutcome(Enum):
    """Outcome of restoring a single drifted stack."""

    RESTORED = auto()  # Drift reverted in place; stack matches baseline
    DELETED_NEEDS_REDEPLOY = auto()  # Revert impossible; stack deleted, redeploy via setup
    FAILED = auto()  # Could not restore or delete the stack


@dataclass
class ResetupDeletion:
    """Outcome of deleting one stack for re-setup, plus resources cleanup abandoned.

    ``abandoned`` maps resource type -> identifier dicts (``{"Identifier": ...}``)
    that FORCE_DELETE_STACK left live. Reset surfaces these as unresolved
    orphans so a still-present resource is never absorbed into a fresh baseline.
    """

    outcome: RestoreOutcome
    abandoned: dict[str, list[dict]] = field(default_factory=dict)


@dataclass
class StackResetOutcome:
    """What a stack-remediation phase deleted for re-setup, and what it left behind.

    ``deleted_stacks`` are stacks removed so ``env setup`` recreates them.
    ``abandoned`` is the merged force-delete orphans across the phase.
    """

    deleted_stacks: list[str] = field(default_factory=list)
    abandoned: dict[str, list[dict]] = field(default_factory=dict)
