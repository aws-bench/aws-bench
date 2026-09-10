"""Create a paired CloudGeni-versus-baseline AWS-Bench comparison.

Usage:
    uv run python scripts/compare_runs.py jobs/cloudgeni jobs/generic-codex \
      --output jobs/comparison.md --json-output jobs/comparison.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from aws_bench.metrics.aggregation import aggregate_basic
from aws_bench.metrics.run_data import RunData, TrialData


def _load(path: Path) -> RunData:
    run = RunData.load(path)
    if run is None or not run.trials:
        raise ValueError(f"No completed AWS-Bench run found at {path}")
    return run


def _attempts(run: RunData) -> dict[tuple[str, int], TrialData]:
    counts: dict[str, int] = defaultdict(int)
    result: dict[tuple[str, int], TrialData] = {}
    for trial in run.trials:
        counts[trial.task_name] += 1
        result[(trial.task_name, counts[trial.task_name])] = trial
    return result


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _summary(run: RunData, *, fallback_agent: str) -> dict[str, Any]:
    basic = aggregate_basic(run)
    latencies = [trial.latency_sec for trial in run.trials if trial.latency_sec is not None]
    rewards = [trial.reward for trial in run.trials if trial.reward is not None]
    return {
        "runDirectory": str(run.run_dir),
        "agent": run.agent_name or fallback_agent,
        "model": run.model_name,
        "provider": run.model_provider,
        "trials": len(run.trials),
        "scoredTrials": len(rewards),
        "passRate": (
            sum(1 for reward in rewards if reward >= 1.0) / len(rewards) if rewards else None
        ),
        "meanReward": _mean(rewards),
        "meanAgentLatencySeconds": _mean(latencies),
        "metrics": basic,
    }


def compare(cloudgeni: RunData, baseline: RunData) -> dict[str, Any]:
    """Return run summaries plus paired per-task outcomes."""
    cloudgeni_attempts = _attempts(cloudgeni)
    baseline_attempts = _attempts(baseline)
    keys = sorted(set(cloudgeni_attempts) | set(baseline_attempts))
    paired = []
    for task_name, attempt in keys:
        cloudgeni_trial = cloudgeni_attempts.get((task_name, attempt))
        baseline_trial = baseline_attempts.get((task_name, attempt))
        cloudgeni_reward = cloudgeni_trial.reward if cloudgeni_trial else None
        baseline_reward = baseline_trial.reward if baseline_trial else None
        paired.append(
            {
                "task": task_name,
                "attempt": attempt,
                "cloudgeniReward": cloudgeni_reward,
                "baselineReward": baseline_reward,
                "rewardDelta": (
                    cloudgeni_reward - baseline_reward
                    if cloudgeni_reward is not None and baseline_reward is not None
                    else None
                ),
                "cloudgeniLatencySeconds": (
                    cloudgeni_trial.latency_sec if cloudgeni_trial else None
                ),
                "baselineLatencySeconds": baseline_trial.latency_sec if baseline_trial else None,
                "cloudgeniException": (
                    cloudgeni_trial.exception_type if cloudgeni_trial else "missing trial"
                ),
                "baselineException": (
                    baseline_trial.exception_type if baseline_trial else "missing trial"
                ),
            }
        )
    return {
        "schemaVersion": 1,
        "cloudgeni": _summary(cloudgeni, fallback_agent="CloudGeni"),
        "genericBaseline": _summary(baseline, fallback_agent="Generic Codex"),
        "pairedTrials": paired,
    }


def _display(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.1%}" if percent else f"{value:.3f}"
    return str(value)


def _markdown(comparison: dict[str, Any]) -> str:
    cloudgeni = comparison["cloudgeni"]
    baseline = comparison["genericBaseline"]
    lines = [
        "# AWS-Bench comparison",
        "",
        "| Metric | CloudGeni | Generic baseline |",
        "| --- | ---: | ---: |",
        f"| Agent | {cloudgeni['agent']} | {baseline['agent']} |",
        f"| Model | {cloudgeni['model'] or 'CloudGeni deployment'} | {baseline['model']} |",
        f"| Trials | {cloudgeni['trials']} | {baseline['trials']} |",
        (
            f"| Pass rate | {_display(cloudgeni['passRate'], percent=True)} | "
            f"{_display(baseline['passRate'], percent=True)} |"
        ),
        (
            f"| Mean reward | {_display(cloudgeni['meanReward'])} | "
            f"{_display(baseline['meanReward'])} |"
        ),
        (
            f"| Mean agent latency (s) | {_display(cloudgeni['meanAgentLatencySeconds'])} | "
            f"{_display(baseline['meanAgentLatencySeconds'])} |"
        ),
        "",
        "## Paired trials",
        "",
        "| Task | Attempt | CloudGeni | Baseline | Delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for trial in comparison["pairedTrials"]:
        lines.append(
            f"| {trial['task']} | {trial['attempt']} | "
            f"{_display(trial['cloudgeniReward'])} | {_display(trial['baselineReward'])} | "
            f"{_display(trial['rewardDelta'])} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build the comparison CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cloudgeni_run", type=Path)
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("aws-bench-comparison.md"))
    parser.add_argument("--json-output", type=Path, default=Path("aws-bench-comparison.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load two completed jobs and write paired reports."""
    args = build_parser().parse_args(argv)
    try:
        comparison = compare(_load(args.cloudgeni_run), _load(args.baseline_run))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_markdown(comparison))
    args.json_output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output} and {args.json_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
