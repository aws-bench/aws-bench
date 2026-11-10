#!/usr/bin/env python3
"""Fail unless every trial in an aws-bench job earned full reward.

Usage:
    python scripts/check_job_rewards.py jobs/<job-name> [--expected-trials N]

Reads ``<job-dir>/result.json`` (a Harbor ``JobResult``). Harbor records, per
eval, each rewarded trial's name under ``reward_stats[<reward-key>][<value>]``
and counts it in that eval's ``n_trials``, so the check works per trial and
per eval. It exits 1 when:

* the file is missing, unparseable, or not shaped like a ``JobResult``
  (missing ``stats``, no ``evals``, non-integer or negative counters,
  ``exception_stats``/``reward_stats`` that are not objects, an
  ``exception_stats`` entry that is not a list, non-list or non-string
  ``reward_stats`` trial lists);
* the job has no trials (``n_total_trials`` <= 0), or ``--expected-trials`` is
  given and differs from ``n_total_trials``;
* ``stats.n_completed_trials`` differs from ``n_total_trials``;
* any trial raised an exception (``stats.n_errored_trials`` > 0, any eval's
  ``n_errors`` > 0, or any non-empty ``exception_stats`` entry);
* an eval's ``n_trials`` differs from the number of distinct trials it lists
  under ``reward_stats``, or the evals' ``n_trials`` do not add up to
  ``n_total_trials``;
* fewer distinct trials have a reward than ``n_total_trials``, or more than
  ``n_completed_trials`` (a trial with several reward keys counts once), so
  every planned trial must have at least one reward;
* a trial appears more than once under the same reward key, or in more than
  one eval;
* any reward value of any trial is non-numeric or not 1.0.

Trials may carry different reward keys, also within one eval: each trial is
checked only against the rewards it has.

It exits 2 on wrong usage and 0 only if none of the above applies.
"""

import json
import sys
from pathlib import Path
from typing import Any

FULL_REWARD = 1.0


def _count(data: dict[str, Any], key: str, where: str, problems: list[str]) -> int:
    """Read a non-negative integer counter (missing means 0); report anything else."""
    value = data.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        problems.append(f"{where}{key} is not an integer: {value!r}")
        return 0
    if value < 0:
        problems.append(f"{where}{key} is negative: {value}")
        return 0
    return value


def _check_value(trial: str, reward_key: str, value: str) -> str | None:
    """Return a problem unless ``value`` (a JSON object key) is the full reward."""
    try:
        number = float(value)
    except ValueError:
        return f"trial {trial}: {reward_key}={value!r} is not a number"
    if number != FULL_REWARD:
        return f"trial {trial}: {reward_key}={value} (expected {FULL_REWARD})"
    return None


def _object(ev: dict[str, Any], key: str, where: str, problems: list[str]) -> dict[str, Any] | None:
    """Return ``ev[key]`` (missing means empty) if it is an object; report it otherwise."""
    value = ev.get(key, {})
    if not isinstance(value, dict):
        problems.append(f"{where}malformed {key}: {value!r}")
        return None
    return value


def _check_eval(eval_key: str, ev: dict[str, Any], problems: list[str]) -> tuple[int, set[str]]:
    """Check one eval; return its ``n_trials`` and the names of its rewarded trials."""
    where = f"{eval_key}: "
    n_trials = _count(ev, "n_trials", where, problems)
    n_errors = _count(ev, "n_errors", where, problems)
    if n_errors:
        problems.append(f"{where}n_errors={n_errors}")
    exception_stats = _object(ev, "exception_stats", where, problems)
    for exc_type, names in (exception_stats or {}).items():
        if not isinstance(names, list):
            problems.append(f"{where}malformed exception_stats[{exc_type!r}]: {names!r}")
        elif names:
            listed = ", ".join(sorted(map(str, names)))
            problems.append(f"{where}{exc_type} raised in trial(s): {listed}")
    reward_stats = _object(ev, "reward_stats", where, problems)
    if reward_stats is None:
        return n_trials, set()

    # trial name -> {reward key: reward value (as written in result.json)}
    rewards: dict[str, dict[str, str]] = {}
    for reward_key, by_value in reward_stats.items():
        if not isinstance(by_value, dict):
            problems.append(f"{where}malformed reward_stats[{reward_key!r}]: {by_value!r}")
            continue
        for value, names in by_value.items():
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                problems.append(f"{where}malformed trial list for {reward_key}={value}: {names!r}")
                continue
            for name in names:
                trial_rewards = rewards.setdefault(name, {})
                if reward_key in trial_rewards:
                    problems.append(f"trial {name} has a {reward_key} reward more than once")
                    continue
                trial_rewards[reward_key] = value

    if n_trials != len(rewards):
        problems.append(
            f"{where}n_trials={n_trials}, but {len(rewards)} distinct trial(s) have a reward"
        )
    for trial in sorted(rewards):
        for reward_key, value in sorted(rewards[trial].items()):
            problem = _check_value(trial, reward_key, value)
            if problem:
                problems.append(problem)
    return n_trials, set(rewards)


def find_problems(result: Any, expected_trials: int | None = None) -> list[str]:
    """Return human-readable problems found in a parsed ``result.json``.

    An empty list means the job planned ``expected_trials`` trials (if given),
    every trial completed without errors and every reward of every trial is
    1.0. Malformed input is reported, never raised.
    """
    if not isinstance(result, dict):
        return [f"result.json is not a JSON object: {type(result).__name__}"]
    problems: list[str] = []
    n_total = _count(result, "n_total_trials", "", problems)
    stats = result.get("stats")
    if not isinstance(stats, dict):
        problems.append(f"missing or malformed stats: {stats!r}")
        stats = {}
    n_completed = _count(stats, "n_completed_trials", "stats.", problems)
    n_errored = _count(stats, "n_errored_trials", "stats.", problems)

    if n_total <= 0:
        problems.append("job contains no trials (n_total_trials <= 0)")
    if expected_trials is not None and n_total != expected_trials:
        problems.append(f"n_total_trials is {n_total}, expected {expected_trials}")
    if n_completed < n_total:
        problems.append(f"only {n_completed} of {n_total} trials completed")
    elif n_completed > n_total:
        problems.append(f"{n_completed} trials completed but n_total_trials is {n_total}")
    if n_errored:
        problems.append(f"{n_errored} trial(s) raised an exception")

    evals = stats.get("evals")
    if not isinstance(evals, dict) or not evals:
        problems.append(f"stats has no evals: {evals!r}")
        evals = {}

    rewarded: set[str] = set()
    sum_n_trials = 0
    for eval_key, ev in evals.items():
        if not isinstance(ev, dict):
            problems.append(f"{eval_key}: malformed eval stats: {ev!r}")
            continue
        n_trials, names = _check_eval(eval_key, ev, problems)
        sum_n_trials += n_trials
        for name in sorted(names & rewarded):
            problems.append(f"trial {name} is listed in more than one eval")
        rewarded |= names

    if sum_n_trials != n_total:
        problems.append(
            f"evals report n_trials={sum_n_trials} rewarded trial(s), "
            f"but n_total_trials is {n_total}"
        )
    n_rewarded = len(rewarded)
    if n_total > 0 and n_rewarded < n_total:
        problems.append(f"only {n_rewarded} of {n_total} trials produced a reward")
    if n_rewarded > n_completed:
        problems.append(f"{n_rewarded} trial(s) produced a reward but only {n_completed} completed")
    return problems


def _parse_args(argv: list[str]) -> tuple[Path, int | None] | None:
    """Return (job dir, expected trials) from ``argv``, or None on wrong usage."""
    args = argv[1:]
    expected: int | None = None
    if len(args) == 3 and args[1] == "--expected-trials":
        try:
            expected = int(args[2])
        except ValueError:
            return None
        if expected < 1:
            return None
        args = args[:1]
    if len(args) != 1 or args[0].startswith("-"):
        return None
    return Path(args[0]), expected


def main(argv: list[str]) -> int:
    """Check the job directory given in ``argv[1]`` (optionally ``--expected-trials N``)."""
    parsed = _parse_args(argv)
    if parsed is None:
        print(__doc__, file=sys.stderr)
        return 2
    job_dir, expected = parsed
    path = job_dir / "result.json"
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        print(f"ERROR: cannot read {path}: {e}", file=sys.stderr)
        return 1
    problems = find_problems(result, expected_trials=expected)
    if problems:
        print(f"FAIL: {path}")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK: all {result['n_total_trials']} trials earned full reward ({path})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
