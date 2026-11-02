"""Exceptions for the dataset module."""

from aws_bench.exceptions import AWSBenchError


class DatasetError(AWSBenchError):
    """Base exception for dataset-related errors."""


class TaskConfigNotFoundError(DatasetError):
    """task.toml does not exist in the expected task directory."""


class TaskConfigInvalidError(DatasetError):
    """task.toml is missing required fields."""


class RegistryValidationError(DatasetError):
    """Raised when registry.json fails strict validation.

    Used for:
    - Schema/JSON parse errors (wraps pydantic/json errors).
    - Uniqueness violations inside a dataset.
    - Missing dataset / version lookups.

    Distinct from ``ScenarioReferenceError``, which fires when a task's
    ``scenario_id`` doesn't match any scenario's manifest name (a
    content-level mismatch, not a registry-shape problem).
    """


class ScenarioReferenceError(DatasetError):
    """A task's scenario_id does not match any scenario's manifest name.

    Also raised when a task's ``agent_account_tags`` names a tag its scenario's
    ``account_tags`` does not declare.

    Distinct from RegistryValidationError, which means the registry
    JSON is structurally broken or the dataset@version doesn't exist.
    Operator action for this error is to fix either task.toml's
    [scenario] scenario_id or scenario.toml's [scenario] name. Raised
    in both registry mode and local mode.
    """


class ScenarioFetchError(DatasetError):
    """Raised when a scenario cannot be fetched from its git source.

    Wraps the underlying git/subprocess errors with an aws-bench-specific
    message that names the scenario and suggests the ``--scenario-path``
    override for offline-dev workflows.
    """


class MetricFetchError(DatasetError):
    """A registry-referenced metric script could not be fetched or resolved.

    Raised when a uv-script git fetch fails, or the named metric file is
    absent from the fetched directory.
    """


class ExtraInstructionFetchError(DatasetError):
    """A registry-referenced extra-instruction file could not be fetched.

    Raised when the git fetch fails or the named file is absent from the
    fetched directory.
    """
