#!/usr/bin/env python3
"""Fail unless every trial in an aws-bench job earned full reward.

Usage:
    python scripts/check_job_rewards.py jobs/<job-name>

Reads ``<job-dir>/result.json`` (a Harbor ``JobResult``) and exits non-zero when:

* the file is missing or unparseable,
* any trial raised an exception (``stats.n_errored_trials`` > 0),
* fewer trials were completed than ``n_total_trials``,
* any trial lacks a verifier reward, or any reward value is not 1.0.
"""

import json
import sys
from pathlib import Path
from typing import Any

FULL_REWARD = 1.0


def find_problems(result: dict[str, Any]) -> list[str]:
    """Return human-readable problems found in a parsed ``result.json``."""
    problems: list[str] = []
    stats = result.get("stats") or {}
    n_total = int(result.get("n_total_trials", 0))
    n_completed = int(stats.get("n_completed_trials", 0))
    n_errored = int(stats.get("n_errored_trials", 0))

    if n_total <= 0:
        problems.append("job contains no trials (n_total_trials == 0)")
    if n_completed < n_total:
        problems.append(f"only {n_completed} of {n_total} trials completed")
    if n_errored:
        problems.append(f"{n_errored} trial(s) raised an exception")

    n_rewarded = 0
    for key, ev in (stats.get("evals") or {}).items():
        for reward_key, by_value in (ev.get("reward_stats") or {}).items():
            for value, trials in by_value.items():
                n_rewarded += len(trials)
                if float(value) != FULL_REWARD:
                    problems.append(
                        f"{key}: {reward_key}={value} for {len(trials)} trial(s): "
                        + ", ".join(sorted(trials))
                    )
    if n_total > 0 and n_rewarded < n_total:
        problems.append(f"only {n_rewarded} of {n_total} trials produced a reward")
    return problems


def main(argv: list[str]) -> int:
    """Check the job directory given in ``argv[1]``."""
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(argv[1]) / "result.json"
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        print(f"ERROR: cannot read {path}: {e}", file=sys.stderr)
        return 1
    problems = find_problems(result)
    if problems:
        print(f"FAIL: {path}")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK: all {result['n_total_trials']} trials earned full reward ({path})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
