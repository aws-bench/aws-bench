"""Unit tests for ``scripts/check_job_rewards.py``.

The script decides pass/fail for the integration suite, so most cases here are
bad inputs that must be reported as problems. Results use the real Harbor
``result.json`` shape: top-level ``n_total_trials`` and
``stats{n_completed_trials, n_errored_trials, evals{<key>: {n_trials, n_errors,
reward_stats{<reward-key>: {"<value>": [trial names]}}, exception_stats}}}``.
"""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_job_rewards.py"
EVAL_KEY = "oracle__aws-bench-quickstart"


@pytest.fixture(scope="module")
def cjr():
    """Import the standalone script module by path."""
    spec = importlib.util.spec_from_file_location("check_job_rewards", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _trials(n: int) -> list[str]:
    return [f"trial-{i}" for i in range(n)]


def _result(
    reward_stats: dict[str, dict[str, list[str]]],
    *,
    n_total: int,
    n_completed: int | None = None,
    n_errored: int = 0,
    n_trials: int | None = None,
    n_errors: int = 0,
    exception_stats: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    return {
        "n_total_trials": n_total,
        "stats": {
            "n_completed_trials": n_total if n_completed is None else n_completed,
            "n_errored_trials": n_errored,
            "evals": {
                EVAL_KEY: {
                    "n_trials": n_total if n_trials is None else n_trials,
                    "n_errors": n_errors,
                    "reward_stats": reward_stats,
                    "exception_stats": exception_stats or {},
                }
            },
        },
    }


# --------------------------------------------------------------------------- #
# passing results
# --------------------------------------------------------------------------- #


def test_all_trials_full_reward_passes(cjr):
    result = _result({"reward": {"1.0": _trials(9)}}, n_total=9)
    assert cjr.find_problems(result) == []


def test_multiple_reward_keys_all_full_passes(cjr):
    names = _trials(3)
    result = _result({"reward": {"1.0": names}, "accuracy": {"1": names}}, n_total=3)
    assert cjr.find_problems(result) == []


# --------------------------------------------------------------------------- #
# failing results
# --------------------------------------------------------------------------- #


def test_multiple_keys_cannot_mask_a_trial_without_reward(cjr):
    """2 trials, but only trial-A has rewards (under 2 keys)."""
    result = _result(
        {"reward": {"1.0": ["trial-A"]}, "accuracy": {"1.0": ["trial-A"]}},
        n_total=2,
        n_trials=1,
    )
    problems = cjr.find_problems(result)
    assert any("1 of 2 trials produced a reward" in p for p in problems)


def test_multiple_keys_mask_detected_even_if_n_trials_is_wrong(cjr):
    """The rewarded-trial set is checked independently of evals[*].n_trials."""
    result = _result(
        {"reward": {"1.0": ["trial-A"]}, "accuracy": {"1.0": ["trial-A"]}},
        n_total=2,
        n_trials=2,
    )
    problems = cjr.find_problems(result)
    assert any("1 of 2 trials produced a reward" in p for p in problems)


def test_one_trial_missing_reward_fails(cjr):
    result = _result({"reward": {"1.0": _trials(8)}}, n_total=9, n_trials=8)
    problems = cjr.find_problems(result)
    assert any("8 of 9 trials produced a reward" in p for p in problems)
    assert any("n_trials" in p for p in problems)


def test_partial_reward_fails(cjr):
    result = _result({"reward": {"1.0": _trials(2), "0.5": ["trial-2"]}}, n_total=3)
    problems = cjr.find_problems(result)
    assert any("trial-2" in p and "reward=0.5" in p for p in problems)


def test_one_key_full_and_another_zero_for_same_trial_fails(cjr):
    names = _trials(2)
    result = _result(
        {"reward": {"1.0": names}, "accuracy": {"1.0": ["trial-0"], "0.0": ["trial-1"]}},
        n_total=2,
    )
    problems = cjr.find_problems(result)
    assert any("trial-1" in p and "accuracy=0.0" in p for p in problems)


def test_heterogeneous_reward_keys_within_one_eval_all_full_pass(cjr):
    """Tasks of one eval may emit different reward keys; each trial needs only full rewards."""
    result = _result(
        {"reward": {"1.0": ["trial-0", "trial-1"]}, "coverage": {"1.0": ["trial-2"]},
         "accuracy": {"1.0": ["trial-0"]}},
        n_total=3,
    )  # fmt: skip
    assert cjr.find_problems(result) == []


def test_heterogeneous_reward_keys_with_one_zero_value_fail(cjr):
    result = _result(
        {"reward": {"1.0": ["trial-0"]}, "coverage": {"0.0": ["trial-1"]}},
        n_total=2,
    )
    assert cjr.find_problems(result) == ["trial trial-1: coverage=0.0 (expected 1.0)"]


def test_errored_trial_fails(cjr):
    result = _result(
        {"reward": {"1.0": _trials(2)}},
        n_total=2,
        n_errored=1,
        n_errors=1,
        exception_stats={"RuntimeError": ["trial-1"]},
    )
    problems = cjr.find_problems(result)
    assert any("raised an exception" in p for p in problems)
    assert any("RuntimeError" in p and "trial-1" in p for p in problems)


def test_exception_stats_alone_fail(cjr):
    """Exceptions recorded per eval fail even if the job-level counter says 0."""
    result = _result(
        {"reward": {"1.0": _trials(2)}},
        n_total=2,
        exception_stats={"RuntimeError": ["trial-1"]},
    )
    assert any("RuntimeError" in p for p in cjr.find_problems(result))


def test_n_errors_alone_fails(cjr):
    result = _result({"reward": {"1.0": _trials(2)}}, n_total=2, n_errors=1)
    assert any("n_errors" in p for p in cjr.find_problems(result))


def test_fewer_completed_than_total_fails(cjr):
    result = _result({"reward": {"1.0": _trials(2)}}, n_total=2, n_completed=1)
    assert any("only 1 of 2 trials completed" in p for p in cjr.find_problems(result))


def test_more_rewarded_than_completed_fails(cjr):
    result = _result({"reward": {"1.0": _trials(3)}}, n_total=2, n_trials=3)
    assert any("3 trial(s) produced a reward" in p for p in cjr.find_problems(result))


def test_duplicate_trial_under_one_key_fails(cjr):
    result = _result({"reward": {"1.0": ["trial-0", "trial-0"]}}, n_total=1)
    assert any("trial-0" in p and "more than once" in p for p in cjr.find_problems(result))


def test_empty_evals_fails(cjr):
    result = {"n_total_trials": 2, "stats": {"n_completed_trials": 2, "evals": {}}}
    problems = cjr.find_problems(result)
    assert any("no evals" in p for p in problems)
    assert any("0 of 2 trials produced a reward" in p for p in problems)


def test_empty_reward_stats_fails(cjr):
    result = _result({}, n_total=2)
    assert any("0 of 2 trials produced a reward" in p for p in cjr.find_problems(result))


def test_missing_stats_fails(cjr):
    assert cjr.find_problems({"n_total_trials": 2}) != []


def test_empty_result_fails(cjr):
    assert cjr.find_problems({}) != []


def test_zero_trials_fails(cjr):
    result = _result({}, n_total=0)
    assert any("no trials" in p for p in cjr.find_problems(result))


@pytest.mark.parametrize("bad_value", ["not-a-number", "", "nan"])
def test_malformed_reward_value_fails(cjr, bad_value):
    result = _result({"reward": {bad_value: ["trial-0"]}}, n_total=1)
    problems = cjr.find_problems(result)
    assert any("trial-0" in p for p in problems)


@pytest.mark.parametrize(
    "result",
    [
        {"n_total_trials": "x", "stats": {}},
        {"n_total_trials": 1, "stats": []},
        {"n_total_trials": 1, "stats": {"evals": {"e": "oops"}}},
        {"n_total_trials": 1, "stats": {"evals": {"e": {"reward_stats": {"r": ["t"]}}}}},
        {"n_total_trials": 1, "stats": {"evals": {"e": {"reward_stats": {"r": {"1.0": "t"}}}}}},
        [],
    ],
)
def test_malformed_structure_is_a_problem_not_a_crash(cjr, result):
    assert cjr.find_problems(result) != []


def _multi_eval(evals: dict[str, dict[str, Any]], *, n_total: int) -> dict[str, Any]:
    """A result with several evals; each eval's n_trials defaults to its listed trials."""
    built = {}
    for key, ev in evals.items():
        by_values = ev["reward_stats"].values()
        names = {n for by_value in by_values for ns in by_value.values() for n in ns}
        built[key] = {
            "n_trials": ev.get("n_trials", len(names)),
            "n_errors": 0,
            "reward_stats": ev["reward_stats"],
            "exception_stats": {},
        }
    return {
        "n_total_trials": n_total,
        "stats": {"n_completed_trials": n_total, "n_errored_trials": 0, "evals": built},
    }


# --------------------------------------------------------------------------- #
# several evals
# --------------------------------------------------------------------------- #


def test_evals_with_different_reward_keys_pass(cjr):
    result = _multi_eval(
        {
            "a": {"reward_stats": {"reward": {"1.0": ["trial-a"]}}},
            "b": {"reward_stats": {"accuracy": {"1.0": ["trial-b"]}}},
        },
        n_total=2,
    )
    assert cjr.find_problems(result) == []


def test_per_eval_n_trials_mismatch_cannot_cancel_out(cjr):
    """Eval a claims 2 but lists 1, eval b claims 0 but lists 1: sums still match."""
    result = _multi_eval(
        {
            "a": {"n_trials": 2, "reward_stats": {"reward": {"1.0": ["trial-a"]}}},
            "b": {"n_trials": 0, "reward_stats": {"reward": {"1.0": ["trial-b"]}}},
        },
        n_total=2,
    )
    problems = cjr.find_problems(result)
    assert any(p.startswith("a: n_trials=2, but 1 distinct") for p in problems)
    assert any(p.startswith("b: n_trials=0, but 1 distinct") for p in problems)


def test_negative_counter_cannot_offset_another_eval(cjr):
    result = _multi_eval(
        {
            "a": {"n_trials": 3, "reward_stats": {"reward": {"1.0": ["trial-a"]}}},
            "b": {"n_trials": -2, "reward_stats": {}},
        },
        n_total=1,
    )
    problems = cjr.find_problems(result)
    assert any("b: n_trials is negative: -2" in p for p in problems)
    assert any("a: n_trials=3" in p for p in problems)


@pytest.mark.parametrize("key", ["n_total_trials"])
def test_negative_total_fails(cjr, key):
    result = _result({"reward": {"1.0": _trials(1)}}, n_total=1)
    result[key] = -1
    assert any("n_total_trials is negative" in p for p in cjr.find_problems(result))


def test_trial_listed_in_two_evals_fails(cjr):
    result = _multi_eval(
        {
            "a": {"reward_stats": {"reward": {"1.0": ["trial-0"]}}},
            "b": {"reward_stats": {"reward": {"1.0": ["trial-0"]}}},
        },
        n_total=2,
    )
    problems = cjr.find_problems(result)
    assert any("trial-0 is listed in more than one eval" in p for p in problems)
    assert any("only 1 of 2 trials produced a reward" in p for p in problems)


# --------------------------------------------------------------------------- #
# malformed per-eval containers and counters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [[], "", None, ["RuntimeError"], "oops"])
def test_malformed_exception_stats_fails(cjr, bad):
    result = _result({"reward": {"1.0": _trials(1)}}, n_total=1)
    result["stats"]["evals"][EVAL_KEY]["exception_stats"] = bad
    problems = cjr.find_problems(result)
    assert any("malformed exception_stats" in p for p in problems)


@pytest.mark.parametrize("bad", ["", 0, None, False, {}, "RuntimeError", 1, {"t": 1}])
def test_non_list_exception_stats_entry_fails(cjr, bad):
    """An entry must be a list of trial names; a falsey non-list is malformed, not empty."""
    result = _result({"reward": {"1.0": _trials(1)}}, n_total=1)
    result["stats"]["evals"][EVAL_KEY]["exception_stats"] = {"RuntimeError": bad}
    problems = cjr.find_problems(result)
    assert problems == [f"{EVAL_KEY}: malformed exception_stats['RuntimeError']: {bad!r}"]


def test_empty_exception_stats_entry_passes(cjr):
    """Harbor leaves an empty list behind after removing a trial's exception."""
    result = _result({"reward": {"1.0": _trials(1)}}, n_total=1)
    result["stats"]["evals"][EVAL_KEY]["exception_stats"] = {"RuntimeError": []}
    assert cjr.find_problems(result) == []


@pytest.mark.parametrize("bad", [[], "", None, ["trial-0"], 1])
def test_malformed_reward_stats_fails_and_stops_that_eval(cjr, bad):
    result = _result({}, n_total=1)
    result["stats"]["evals"][EVAL_KEY]["reward_stats"] = bad
    problems = cjr.find_problems(result)
    assert any(f"{EVAL_KEY}: malformed reward_stats" in p for p in problems)
    # Early return: no per-trial checks for that eval, but job-level checks still run.
    assert not any("distinct trial(s) have a reward" in p for p in problems)
    assert any("only 0 of 1 trials produced a reward" in p for p in problems)


def test_more_completed_than_total_fails(cjr):
    result = _result({"reward": {"1.0": _trials(2)}}, n_total=2, n_completed=3)
    assert any("3 trials completed but n_total_trials is 2" in p for p in cjr.find_problems(result))


# --------------------------------------------------------------------------- #
# expected trial count
# --------------------------------------------------------------------------- #


def test_expected_trials_match_passes(cjr):
    result = _result({"reward": {"1.0": _trials(9)}}, n_total=9)
    assert cjr.find_problems(result, expected_trials=9) == []


def test_fewer_trials_than_expected_fails_even_if_all_full(cjr):
    """A run whose globs resolved 1 task is internally consistent but not what was asked."""
    result = _result({"reward": {"1.0": _trials(1)}}, n_total=1)
    assert cjr.find_problems(result) == []
    problems = cjr.find_problems(result, expected_trials=9)
    assert problems == ["n_total_trials is 1, expected 9"]


# --------------------------------------------------------------------------- #
# main()
# --------------------------------------------------------------------------- #


def _write_job(tmp_path: Path, result: Any) -> Path:
    job = tmp_path / "job"
    job.mkdir()
    (job / "result.json").write_text(json.dumps(result))
    return job


def test_main_ok_returns_0(cjr, tmp_path, capsys):
    job = _write_job(tmp_path, _result({"reward": {"1.0": _trials(2)}}, n_total=2))
    assert cjr.main(["prog", str(job)]) == 0
    assert "OK: all 2 trials" in capsys.readouterr().out


def test_main_problems_return_1(cjr, tmp_path, capsys):
    job = _write_job(tmp_path, _result({"reward": {"0.0": _trials(2)}}, n_total=2))
    assert cjr.main(["prog", str(job)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_missing_file_returns_1(cjr, tmp_path, capsys):
    assert cjr.main(["prog", str(tmp_path)]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_main_invalid_json_returns_1(cjr, tmp_path):
    (tmp_path / "result.json").write_text("{not json")
    assert cjr.main(["prog", str(tmp_path)]) == 1


def test_main_non_object_json_returns_1(cjr, tmp_path):
    job = _write_job(tmp_path, [1, 2])
    assert cjr.main(["prog", str(job)]) == 1


def test_main_expected_trials_ok_returns_0(cjr, tmp_path):
    job = _write_job(tmp_path, _result({"reward": {"1.0": _trials(2)}}, n_total=2))
    assert cjr.main(["prog", str(job), "--expected-trials", "2"]) == 0


def test_main_expected_trials_mismatch_returns_1(cjr, tmp_path, capsys):
    job = _write_job(tmp_path, _result({"reward": {"1.0": _trials(2)}}, n_total=2))
    assert cjr.main(["prog", str(job), "--expected-trials", "9"]) == 1
    assert "n_total_trials is 2, expected 9" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["prog"],
        ["prog", "a", "b"],
        ["prog", "a", "--expected-trials"],
        ["prog", "a", "--expected-trials", "x"],
        ["prog", "a", "--expected-trials", "0"],
        ["prog", "a", "--expected-trials", "-1"],
        ["prog", "a", "--other", "2"],
        ["prog", "--expected-trials", "2", "a"],
    ],
)
def test_main_usage_returns_2(cjr, argv):
    assert cjr.main(argv) == 2
