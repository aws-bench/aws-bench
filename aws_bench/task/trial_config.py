"""AwsBenchTrialConfig — Harbor TrialConfig + the per-trial AWS context."""

from __future__ import annotations

from harbor.models.trial.config import TrialConfig
from pydantic import Field

from aws_bench.dataset.task_config import ConcurrencyMode
from aws_bench.scenario.locator import ScenarioConfig


class AwsBenchTrialConfig(TrialConfig):
    """Harbor TrialConfig plus the per-trial AWS context the run resolves.

    ``scenario_id``, ``concurrency_mode``, ``account_mapping``, and ``exports``
    are the run-resolved per-trial data; everything else (roles, scripts) is
    static task content read off ``self.task.config``. The trial reads these off
    ``self.config`` at runtime (credential assumption, placeholder seed).

    ``scenario_id`` picks the queue's scenario-admission gate; ``concurrency_mode``
    decides how it runs (READ_ONLY co-runs, MUTATING is exclusive); ``account_mapping``
    is ``{account_tag: account_id}``, one assumed role and credentials profile per tag.
    ``scenario`` is the source descriptor (path + optional git provenance) the
    post-trial reset re-materializes. These serialize, so they ride the inherited
    ``TrialConfig.__eq__``: on resume a trial whose scenario, account, or scenario
    source changed is re-run. ``exports`` is tag-keyed (``tag -> {export name -> value}``,
    per account tag of the trial's scenario) and ``exclude=True``, so it rides neither
    serialization nor that identity — a value change alone never forces a re-run, and
    the resolved (possibly sensitive) values stay off disk.
    """

    scenario: ScenarioConfig
    scenario_id: str
    concurrency_mode: ConcurrencyMode
    account_mapping: dict[str, str]
    # The scenario manifest's declared regions (validated non-empty). regions[0]
    # pins AWS_REGION/AWS_DEFAULT_REGION for the agent phase.
    regions: list[str]
    # When False (--no-verify-env), skip the per-trial AccountContaminated tag
    # check so operators can force a run against a flagged account.
    verify_env: bool = True
    # exclude=True keeps re-resolved (and possibly sensitive) values out of
    # config.json, the snapshot, and the resume identity.
    exports: dict[str, dict[str, str]] = Field(default_factory=dict, exclude=True)
