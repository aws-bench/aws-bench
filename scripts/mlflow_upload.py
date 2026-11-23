"""Upload aws-bench run results to MLflow.

Reads completed run directories and logs them as MLflow experiments with
traces (one per trial). Uses Harbor's native Pydantic models for robust
parsing with automatic legacy format migration.

Authentication is handled externally — set MLFLOW_TRACKING_AWS_SIGV4=true
and provide AWS credentials via environment, profile, or instance role.

Usage:
    python scripts/mlflow_upload.py <run-dir>
    python scripts/mlflow_upload.py <run-dir> --experiment-name my-experiment
    python scripts/mlflow_upload.py --upload-all <logs-dir>
    python scripts/mlflow_upload.py --upload-all <logs-dir> --experiment-name my-experiment

Environment variables:
    MLFLOW_TRACKING_URI          MLflow tracking server URL (required)
    MLFLOW_TRACKING_AWS_SIGV4    Set to 'true' for SigV4-authenticated servers
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from harbor.models.job.config import JobConfig
from harbor.utils.pass_at_k import compute_pass_at_k_by_evals

from aws_bench.metrics.aggregation import aggregate_basic
from aws_bench.metrics.run_data import (
    RunData,
    TrialData,
    find_run_dirs,
    iter_artifact_files,
    load_json,
    load_model,
    parse_datetime_ns,
    read_verifier_rationale,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logging.getLogger("botocore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

try:
    import mlflow
    from mlflow.entities import AssessmentSource, AssessmentSourceType, SpanEvent
except ImportError:
    logger.error("mlflow package not installed. Install with: pip install mlflow-skinny")
    sys.exit(1)


def _flatten_metrics(data: dict, prefix: str = "") -> dict[str, float]:
    """Flatten a nested dict into dot-separated keys, keeping only numeric values."""
    result: dict[str, float] = {}
    for key, value in data.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_metrics(value, f"{full_key}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[full_key] = float(value)
    return result


def _exception_span_event(trial: TrialData, timestamp_ns: int) -> Optional[SpanEvent]:
    """Build an OTel-style 'exception' SpanEvent from the trial's exception_info."""
    info = trial.result.exception_info
    if not info:
        return None
    return SpanEvent(
        name="exception",
        timestamp=timestamp_ns,
        attributes={
            "exception.type": info.exception_type,
            "exception.message": info.exception_message,
            "exception.stacktrace": (info.exception_traceback or "")[:8000],
        },
    )


def _check_already_uploaded(tracking_uri: str, job_id: str) -> bool:
    """Check if a run with this job_id was already uploaded."""
    try:
        client = mlflow.MlflowClient(tracking_uri)
        runs = client.search_runs(
            experiment_ids=[],
            filter_string=f"tags.`awsbench.job_id` = '{job_id}'",
            max_results=1,
            search_all_experiments=True,
        )
        return len(runs) > 0
    except Exception:
        return False


def upload_single_run(
    run_dir: Path,
    experiment_name: str,
    tracking_uri: str,
    force: bool = False,
    extra_params: Optional[dict[str, str]] = None,
    extra_metrics: Optional[dict[str, float]] = None,
    ui_uri: Optional[str] = None,
) -> bool:
    """Upload a single run directory to MLflow. Returns True on success."""
    run = RunData.load(run_dir)
    if not run:
        logger.error(f"No valid result.json in {run_dir}")
        return False

    stats = run.job_result.stats
    if stats.n_completed_trials == 0 and stats.n_running_trials > 0:
        logger.warning(f"Run {run.run_dir.name} appears to still be in progress, skipping")
        return False

    run_failed = stats.n_completed_trials == 0 and stats.n_errored_trials > 0
    job_id = run.job_id

    if not force and _check_already_uploaded(tracking_uri, job_id):
        logger.info(f"Run {job_id} already uploaded, skipping (use --force to re-upload)")
        return True

    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = _get_or_create_experiment(experiment_name)

    existing_run_id = os.environ.get("MLFLOW_RUN_ID")
    mlflow_run = mlflow.start_run(
        run_id=existing_run_id,
        experiment_id=experiment_id,
        run_name=job_id,
    )

    # --- Params ---
    params = {
        "job_id": job_id,
        "agent_name": run.agent_name or "unknown",
        "model_name": run.model_name or "unknown",
        "model_provider": run.model_provider or "unknown",
        "dataset_source": run.dataset_source or "unknown",
        "n_total_trials": str(run.job_result.n_total_trials),
        "n_concurrent_trials": str(run.n_concurrent_trials or 1),
        "environment_type": run.environment_type or "unknown",
    }
    if run.harbor_version:
        params["harbor_version"] = run.harbor_version
    if run.n_attempts is not None:
        params["n_attempts"] = str(run.n_attempts)
    if extra_params:
        mlflow.log_params(extra_params)
    mlflow.log_params(params)
    if extra_metrics:
        mlflow.log_metrics(extra_metrics)

    # --- Tags ---
    tags = {
        "awsbench.job_id": job_id,
        "awsbench.run_dir": str(run.run_dir),
        "awsbench.status": "completed" if not run_failed else "failed",
    }
    if run.agent_name:
        tags["awsbench.agent"] = run.agent_name
    if run.model_name:
        tags["awsbench.model"] = run.model_name
    if run.model_provider:
        tags["awsbench.provider"] = run.model_provider
    if run.dataset_source:
        tags["awsbench.dataset"] = run.dataset_source
    mlflow.set_tags(tags)

    # --- Aggregate metrics ---
    mlflow.log_metrics(aggregate_basic(run))

    # --- Artifacts: upload the run directory ---
    files = list(iter_artifact_files(run.run_dir))
    logger.info(f"Uploading {len(files)} artifact file(s) from {run.run_dir}")
    artifact_client = mlflow.MlflowClient()
    for path, artifact_dir in files:
        artifact_client.log_artifact(mlflow_run.info.run_id, str(path), artifact_dir)

    mlflow.end_run(status="FAILED" if run_failed else "FINISHED")

    # --- Upload traces for each trial ---
    logger.info(f"Uploading traces for {len(run.trials)} trial(s)")
    attempts = run.task_attempt_numbers()

    trace_pairs: list[tuple[str, TrialData]] = []
    for trial, attempt in zip(run.trials, attempts):
        trace_id = _upload_trial_trace(trial, experiment_id, mlflow_run.info.run_id, attempt)
        if trace_id:
            trace_pairs.append((trace_id, trial))

    if trace_pairs:
        client = mlflow.MlflowClient()
        client.link_traces_to_run(
            run_id=mlflow_run.info.run_id,
            trace_ids=[tid for tid, _ in trace_pairs],
        )

    # Flush traces to DB before logging assessments (avoids FK constraint errors)
    mlflow.flush_trace_async_logging()

    for trace_id, trial in trace_pairs:
        _log_equivalence_assessment(trace_id, trial)

    base_url = (ui_uri or tracking_uri).rstrip("/")
    run_url = f"{base_url}/#/experiments/{experiment_id}/runs/{mlflow_run.info.run_id}"
    logger.info(
        f"Uploaded run {job_id} to experiment '{experiment_name}' "
        f"(traces: {len(trace_pairs)})\n"
        f"MLFlow tracking URL: {run_url}"
    )
    return True


def _log_equivalence_assessment(trace_id: str, trial: TrialData) -> None:
    """Log an LLM judge equivalence assessment on the trace."""
    reward = trial.reward
    if reward is None:
        return

    rationale, judge_model = read_verifier_rationale(trial.trial_dir)

    mlflow.log_feedback(
        trace_id=trace_id,
        name="equivalence",
        value=reward >= 1.0,
        rationale=rationale,
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id=judge_model or "harbor-verifier",
        ),
    )


def _upload_trial_trace(
    trial: TrialData,
    experiment_id: str,
    run_id: str,
    attempt: int = 1,
) -> Optional[str]:
    """Upload a trial as a trace linked to the parent run. Returns trace_id or None."""
    # Try trajectory-based trace first
    if trial.has_trajectory:
        trace_id = _upload_trajectory_as_trace(trial, experiment_id, run_id, attempt)
        if trace_id:
            logger.info(
                f"  Uploaded trace: {trial.trial_name} (reward={trial.reward}, attempt={attempt})"
            )
            return trace_id

    # Fallback: minimal trace from result fields only
    client = mlflow.MlflowClient()
    start_ts = _datetime_to_ns(trial.result.started_at)
    end_ts = _datetime_to_ns(trial.result.finished_at)
    if not start_ts or not end_ts:
        return None

    trace_tags = {
        "awsbench.task_name": trial.task_name,
        "awsbench.trial_name": trial.trial_name,
        "awsbench.attempt": str(attempt),
        "awsbench.run_id": run_id,
    }
    if trial.reward is not None:
        trace_tags["awsbench.reward"] = str(trial.reward)
    if trial.exception_type:
        trace_tags["awsbench.exception_type"] = trial.exception_type

    attributes = {
        "trial_name": trial.trial_name,
        "reward": str(trial.reward) if trial.reward is not None else "unknown",
        "cost_usd": str(trial.cost_usd or 0),
        "n_uncached_input_tokens": str(trial.uncached_input_tokens or 0),
        "n_cached_input_tokens": str(trial.n_cache_tokens or 0),
        "n_output_tokens": str(trial.n_output_tokens or 0),
        "agent_execution_sec": (
            str(trial.agent_execution_sec) if trial.agent_execution_sec else "unknown"
        ),
    }
    if trial.exception_type:
        attributes["exception_type"] = trial.exception_type

    root_span = client.start_trace(
        name=trial.task_name,
        span_type="AGENT",
        experiment_id=experiment_id,
        inputs={"task_name": trial.task_name},
        attributes=attributes,
        tags=trace_tags,
        start_time_ns=start_ts,
    )
    # Attach the exception (type/message/stacktrace) so MLflow renders the
    # error block on this minimal trace just like the trajectory-based one.
    if trial.has_exception:
        event = _exception_span_event(trial, end_ts)
        if event is not None:
            root_span.add_event(event)
    trace_status = "ERROR" if trial.has_exception else "OK"
    client.end_trace(
        trace_id=root_span.trace_id,
        outputs={"reward": trial.reward},
        status=trace_status,
        end_time_ns=end_ts,
    )
    logger.info(f"  Uploaded trace (minimal): {trial.trial_name} (reward={trial.reward})")
    return root_span.trace_id


def _datetime_to_ns(dt: Optional[datetime]) -> Optional[int]:
    """Convert a datetime to nanoseconds since epoch."""
    return parse_datetime_ns(dt)


def _iso_to_ns(ts: Optional[str]) -> Optional[int]:
    """Convert an ISO timestamp string to nanoseconds since epoch."""
    return parse_datetime_ns(ts)


def _upload_trajectory_as_trace(
    trial: TrialData,
    experiment_id: str,
    run_id: str,
    attempt: int = 1,
) -> Optional[str]:
    """Reconstruct an MLflow trace from an ATIF trajectory file. Returns trace_id or None."""
    trajectory = load_json(trial.trajectory_path)
    if not trajectory or "steps" not in trajectory:
        return None

    steps = trajectory["steps"]
    if not steps:
        return None

    client = mlflow.MlflowClient()

    first_ts = _iso_to_ns(steps[0].get("timestamp"))
    last_ts = _iso_to_ns(steps[-1].get("timestamp"))
    if not first_ts or not last_ts:
        return None

    # Trial-level execution boundary — survives agent stalls (no more steps emitted).
    # When the agent times out, last_ts reflects the last *recorded* step, not the
    # actual end of execution. Use agent_execution.finished_at to close the trace
    # at the real boundary.
    agent_exec_end_ns = (
        _datetime_to_ns(trial.result.agent_execution.finished_at)
        if trial.result.agent_execution
        else None
    ) or last_ts
    trace_end_ns = max(agent_exec_end_ns, last_ts)

    # Identify tool calls that were issued but never resolved (no observation).
    # On a timeout, the last in-flight tool call ran until agent_exec_end_ns.
    unfinished_tool_call_ids: set[str] = set()
    last_unfinished: Optional[dict] = None
    for step in steps:
        if step.get("source") != "agent":
            continue
        for tc in step.get("tool_calls") or []:
            tc_id = tc.get("tool_call_id", "")
            if not tc_id:
                continue
            unfinished_tool_call_ids.add(tc_id)
    for step in steps:
        observation = step.get("observation") or {}
        for result in observation.get("results") or []:
            cid = result.get("source_call_id")
            if cid:
                unfinished_tool_call_ids.discard(cid)
    # Find the last tool_call step containing one of the unfinished ids
    for step in reversed(steps):
        if step.get("source") != "agent":
            continue
        for tc in step.get("tool_calls") or []:
            if tc.get("tool_call_id") in unfinished_tool_call_ids:
                last_unfinished = tc
                break
        if last_unfinished is not None:
            break

    reward = trial.reward
    agent_info = trajectory.get("agent", {})
    trace_tags = {
        "awsbench.task_name": trial.task_name,
        "awsbench.trial_name": trial.trial_name,
        "awsbench.attempt": str(attempt),
        "awsbench.run_id": run_id,
    }
    if reward is not None:
        trace_tags["awsbench.reward"] = str(reward)
    if trial.cost_usd:
        trace_tags["awsbench.cost_usd"] = str(trial.cost_usd)
    if trial.exception_type:
        trace_tags["awsbench.exception_type"] = trial.exception_type
    if last_unfinished is not None:
        trace_tags["awsbench.last_unfinished_tool"] = str(
            last_unfinished.get("function_name", "unknown_tool")
        )

    root_attributes = {
        "agent_name": agent_info.get("name", "unknown"),
        "agent_version": agent_info.get("version", "unknown"),
        "model_name": agent_info.get("model_name", "unknown"),
        "session_id": trajectory.get("session_id", ""),
        "trial_name": trial.trial_name,
        "reward": str(reward) if reward is not None else "unknown",
        "cost_usd": str(trial.cost_usd or 0),
        "n_uncached_input_tokens": str(trial.uncached_input_tokens or 0),
        "n_cached_input_tokens": str(trial.n_cache_tokens or 0),
        "n_output_tokens": str(trial.n_output_tokens or 0),
        "exception_type": trial.exception_type or "",
    }
    if last_unfinished is not None:
        root_attributes["last_unfinished_tool"] = str(
            last_unfinished.get("function_name", "unknown_tool")
        )
        last_args = last_unfinished.get("arguments") or {}
        try:
            root_attributes["last_unfinished_tool_args"] = json.dumps(last_args)[:2000]
        except (TypeError, ValueError):
            root_attributes["last_unfinished_tool_args"] = str(last_args)[:2000]

    root_span = client.start_trace(
        name=trial.task_name,
        span_type="AGENT",
        experiment_id=experiment_id,
        inputs={"task": steps[0].get("message", "") if steps[0].get("source") == "user" else ""},
        attributes=root_attributes,
        tags=trace_tags,
        start_time_ns=first_ts,
    )
    trace_id = root_span.trace_id
    root_span_id = root_span.span_id

    i = 0
    while i < len(steps):
        step = steps[i]
        source = step.get("source")
        step_ts = _iso_to_ns(step.get("timestamp")) or first_ts

        if source == "user":
            span = client.start_span(
                name="user_input",
                trace_id=trace_id,
                parent_id=root_span_id,
                span_type="UNKNOWN",
                inputs={"message": step.get("message", "")},
                start_time_ns=step_ts,
            )
            client.end_span(
                trace_id=trace_id,
                span_id=span.span_id,
                end_time_ns=step_ts,
            )
            i += 1
            continue

        if source == "agent" and step.get("tool_calls"):
            for tc in step["tool_calls"]:
                tool_name = tc.get("function_name", "unknown_tool")
                tool_args = tc.get("arguments", {})
                tool_call_id = tc.get("tool_call_id", "")

                end_ts = step_ts
                if i + 1 < len(steps):
                    end_ts = _iso_to_ns(steps[i + 1].get("timestamp")) or step_ts

                observation = step.get("observation", {})
                tool_output = ""
                if observation and observation.get("results"):
                    for result in observation["results"]:
                        if result.get("source_call_id") == tool_call_id:
                            content = result.get("content", "")
                            tool_output = content[:2000] if len(content) > 2000 else content
                            break
                    if not tool_output and observation["results"]:
                        content = observation["results"][0].get("content", "")
                        tool_output = content[:2000] if len(content) > 2000 else content

                # Tool call never resolved — extend it to the actual execution end
                # so the trace shows where time was spent.
                is_unfinished = tool_call_id in unfinished_tool_call_ids
                if is_unfinished:
                    end_ts = trace_end_ns

                span_attrs = {"tool_call_id": tool_call_id}
                if is_unfinished:
                    span_attrs["unfinished"] = "true"

                span = client.start_span(
                    name=tool_name,
                    trace_id=trace_id,
                    parent_id=root_span_id,
                    span_type="TOOL",
                    inputs=tool_args,
                    attributes=span_attrs,
                    start_time_ns=step_ts,
                )
                if is_unfinished:
                    event = _exception_span_event(trial, end_ts)
                    if event is not None:
                        span.add_event(event)
                client.end_span(
                    trace_id=trace_id,
                    span_id=span.span_id,
                    outputs={
                        "result": tool_output
                        or (
                            "[unfinished — agent execution ended before tool returned]"
                            if is_unfinished
                            else ""
                        )
                    },
                    status="ERROR" if is_unfinished else "OK",
                    end_time_ns=end_ts,
                )
            i += 1
            continue

        if source == "agent" and step.get("metrics"):
            end_ts = step_ts
            if i + 1 < len(steps):
                end_ts = _iso_to_ns(steps[i + 1].get("timestamp")) or step_ts

            step_metrics = step.get("metrics", {})
            attributes = {}
            if step_metrics.get("prompt_tokens"):
                attributes["prompt_tokens"] = str(step_metrics["prompt_tokens"])
            if step_metrics.get("completion_tokens"):
                attributes["completion_tokens"] = str(step_metrics["completion_tokens"])
            if step_metrics.get("cached_tokens"):
                attributes["cached_tokens"] = str(step_metrics["cached_tokens"])

            reasoning = step.get("reasoning_content", "")
            message = step.get("message", "")

            span = client.start_span(
                name=f"llm_step_{step['step_id']}",
                trace_id=trace_id,
                parent_id=root_span_id,
                span_type="LLM",
                inputs={"reasoning": reasoning[:1000] if reasoning else ""},
                attributes=attributes,
                start_time_ns=step_ts,
            )
            client.end_span(
                trace_id=trace_id,
                span_id=span.span_id,
                outputs={"message": message[:1000] if message else ""},
                end_time_ns=end_ts,
            )
            i += 1
            continue

        if source == "agent" and step.get("message"):
            end_ts = step_ts
            if i + 1 < len(steps):
                end_ts = _iso_to_ns(steps[i + 1].get("timestamp")) or step_ts

            span = client.start_span(
                name=f"agent_step_{step['step_id']}",
                trace_id=trace_id,
                parent_id=root_span_id,
                span_type="CHAIN",
                inputs={},
                start_time_ns=step_ts,
            )
            client.end_span(
                trace_id=trace_id,
                span_id=span.span_id,
                outputs={"message": step["message"][:1000]},
                end_time_ns=end_ts,
            )
            i += 1
            continue

        i += 1

    final_message = trial.last_agent_message[:2000]

    # Attach an OTel-style 'exception' event to the root span when the trial
    # errored. MLflow's UI renders this as a styled error block on the span
    # with type/message/stacktrace, so timeouts and crashes are visible at a
    # glance instead of being a red status pill with no detail.
    if trial.has_exception:
        event = _exception_span_event(trial, trace_end_ns)
        if event is not None:
            root_span.add_event(event)

    trace_status = "ERROR" if trial.has_exception else "OK"
    final_metrics = trajectory.get("final_metrics", {})
    client.end_trace(
        trace_id=trace_id,
        outputs={"final_answer": final_message},
        attributes={
            "total_prompt_tokens": str(final_metrics.get("total_prompt_tokens", 0)),
            "total_completion_tokens": str(final_metrics.get("total_completion_tokens", 0)),
            "total_cost_usd": str(final_metrics.get("total_cost_usd", 0)),
            "total_steps": str(final_metrics.get("total_steps", 0)),
        },
        status=trace_status,
        # Close the root span at the trial's actual execution boundary, not
        # the last logged step — when the agent stalls (e.g. blocks on a tool
        # call), trajectory steps stop being emitted but the harness keeps
        # running until the timeout fires. Using last_ts would hide the gap.
        end_time_ns=trace_end_ns,
    )
    logger.debug(f"    Created trace {trace_id} with {len(steps)} steps")
    return trace_id


def _get_or_create_experiment(experiment_name: str) -> str:
    """Get or create an MLflow experiment, returning its ID."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created MLflow experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment: {experiment_name}")
    mlflow.set_experiment(experiment_name)
    mlflow.set_experiment_tag("awsbench.benchmark", "awsbench")
    return experiment_id


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Upload aws-bench run results to MLflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Path to a single run directory (contains result.json)",
    )
    parser.add_argument(
        "--upload-all",
        type=Path,
        metavar="LOGS_DIR",
        help="Upload all completed runs in a logs directory",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment name (default: derived from dataset source)",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI)",
    )
    parser.add_argument(
        "--ui-uri",
        default=os.getenv("MLFLOW_UI_URI"),
        help="MLflow UI base URL for run links (default: $MLFLOW_UI_URI, "
        "falls back to tracking URI)",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="awsbench",
        help="Prefix for experiment names (default: 'awsbench'). "
        "Final name: <prefix>/<dataset_source> or <prefix>/<experiment-name>",
    )
    parser.add_argument(
        "--extra-params-file",
        type=Path,
        metavar="JSON_FILE",
        help="Path to a JSON file whose key-value pairs are logged as extra MLflow params",
    )
    parser.add_argument(
        "--extra-params-prefix",
        default="",
        metavar="PREFIX",
        help="Prefix prepended to each extra param/metric key (e.g. 'runner' → 'runner.model_id'). "
        "Applied to both --extra-params-file and --extra-metrics-file (despite the name, "
        "the prefix is shared with metrics for symmetry).",
    )
    parser.add_argument(
        "--extra-metrics-file",
        type=Path,
        metavar="JSON_FILE",
        help="Path to a JSON file mapping metric name → numeric value, logged as MLflow metrics. "
        "Keys are prefixed with --extra-params-prefix (shared with --extra-params-file).",
    )
    parser.add_argument(
        "--monitoring-only",
        action="store_true",
        help="Create a lightweight monitoring run (params + artifacts) without benchmark results. "
        "Use for failed runs or runs that didn't produce structured output.",
    )
    parser.add_argument(
        "--artifacts",
        nargs="*",
        type=Path,
        metavar="FILE",
        help="Files to upload as artifacts (used with --monitoring-only)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="MLflow run name (used with --monitoring-only, default: from extra params run_id)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate run data without uploading to MLflow",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload runs even if they already exist in MLflow",
    )

    args = parser.parse_args()

    if not args.run_dir and not args.upload_all and not args.monitoring_only:
        parser.error(
            "Provide either a run directory, --upload-all <logs-dir>, or --monitoring-only"
        )

    if not args.tracking_uri and not args.dry_run:
        parser.error("MLflow tracking URI required. Set MLFLOW_TRACKING_URI or use --tracking-uri")

    # --- Monitoring-only mode: create a lightweight run with params + artifacts ---
    if args.monitoring_only:
        extra_params: Optional[dict[str, str]] = None
        if args.extra_params_file:
            if not args.extra_params_file.exists():
                logger.error(f"Extra params file not found: {args.extra_params_file}")
                sys.exit(1)
            raw = json.loads(args.extra_params_file.read_text())
            prefix = f"{args.extra_params_prefix}." if args.extra_params_prefix else ""
            extra_params = {f"{prefix}{k}": str(v) for k, v in raw.items()}

        extra_metrics: Optional[dict[str, float]] = None
        if args.extra_metrics_file:
            if not args.extra_metrics_file.exists():
                logger.error(f"Extra metrics file not found: {args.extra_metrics_file}")
                sys.exit(1)
            raw_m = json.loads(args.extra_metrics_file.read_text())
            prefix = f"{args.extra_params_prefix}." if args.extra_params_prefix else ""
            extra_metrics = _flatten_metrics(raw_m, prefix)

        experiment_name = args.experiment_name or "scheduled-runs"
        if "/" not in experiment_name:
            experiment_name = f"{args.experiment_prefix.strip('/')}/{experiment_name}"

        mlflow.set_tracking_uri(args.tracking_uri)
        experiment_id = _get_or_create_experiment(experiment_name)

        run_name = args.run_name
        if not run_name and extra_params:
            prefix = f"{args.extra_params_prefix}." if args.extra_params_prefix else ""
            run_name = extra_params.get(f"{prefix}run_id")

        existing_run_id = os.environ.get("MLFLOW_RUN_ID")
        run = mlflow.start_run(
            run_id=existing_run_id,
            experiment_id=experiment_id,
            run_name=run_name,
        )

        if extra_params:
            mlflow.log_params(extra_params)
        if extra_metrics:
            mlflow.log_metrics(extra_metrics)

        status_param = (extra_params or {}).get(
            f"{args.extra_params_prefix + '.' if args.extra_params_prefix else ''}status",
            "",
        )
        mlflow.set_tag("awsbench.status", status_param or "unknown")

        if args.artifacts:
            for artifact_path in args.artifacts:
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path), "monitoring")
                else:
                    logger.warning(f"Artifact not found, skipping: {artifact_path}")

        mlflow_status = "FINISHED" if status_param == "completed" else "FAILED"
        mlflow.end_run(mlflow_status)

        ui_base = args.ui_uri or args.tracking_uri
        run_url = f"{ui_base}/#/experiments/{experiment_id}/runs/{run.info.run_id}"
        logger.info(f"Monitoring run uploaded: {run_url}")
        print(run.info.run_id)
        return

    # Collect run directories
    if args.upload_all:
        logs_dir = args.upload_all
        if not logs_dir.is_dir():
            logger.error(f"Not a directory: {logs_dir}")
            sys.exit(1)
        run_dirs = find_run_dirs(logs_dir)
        if not run_dirs:
            logger.error(f"No completed runs found in {logs_dir}")
            sys.exit(1)
        logger.info(f"Found {len(run_dirs)} run(s) to upload")
    else:
        run_dir = args.run_dir
        if not run_dir.is_dir():
            logger.error(f"Not a directory: {run_dir}")
            sys.exit(1)
        if not (run_dir / "result.json").exists():
            logger.error(f"No result.json found in {run_dir}")
            sys.exit(1)
        run_dirs = [run_dir]

    if args.dry_run:
        for rd in run_dirs:
            run = RunData.load(rd)
            if not run:
                continue
            pak = compute_pass_at_k_by_evals(list(run.iter_trial_results()))
            logger.info(
                f"[dry-run] {rd.name}: "
                f"{run.job_result.stats.n_completed_trials} completed, "
                f"{run.job_result.stats.n_errored_trials} errored, "
                f"{len(run.trials)} trial dirs"
            )
            if pak:
                for evals_key, pak_dict in pak.items():
                    pak_str = ", ".join(f"pass@{k}={v:.3f}" for k, v in pak_dict.items())
                    logger.info(f"  pass@k ({evals_key}): {pak_str}")
            if run.config and run.config.agents:
                agent = run.config.agents[0]
                logger.info(f"  Agent: {agent.name} / {agent.model_name}")
            has_analysis = (rd / "analysis.md").exists() or (rd / "analysis.json").exists()
            if has_analysis:
                logger.info("  Analysis artifacts: present")
        return

    # Load extra params file if provided
    extra_params: Optional[dict[str, str]] = None
    if args.extra_params_file:
        if not args.extra_params_file.exists():
            logger.error(f"Extra params file not found: {args.extra_params_file}")
            sys.exit(1)
        raw = json.loads(args.extra_params_file.read_text())
        prefix = f"{args.extra_params_prefix}." if args.extra_params_prefix else ""
        extra_params = {f"{prefix}{k}": str(v) for k, v in raw.items()}

    extra_metrics: Optional[dict[str, float]] = None
    if args.extra_metrics_file:
        if not args.extra_metrics_file.exists():
            logger.error(f"Extra metrics file not found: {args.extra_metrics_file}")
            sys.exit(1)
        raw_m = json.loads(args.extra_metrics_file.read_text())
        prefix = f"{args.extra_params_prefix}." if args.extra_params_prefix else ""
        extra_metrics = _flatten_metrics(raw_m, prefix)

    # Upload
    n_success = 0
    n_fail = 0
    prefix = args.experiment_prefix.strip("/")
    for rd in run_dirs:
        experiment = args.experiment_name
        if not experiment:
            cfg = load_model(JobConfig, rd / "config.json")
            if cfg and cfg.datasets:
                ds = cfg.datasets[0]
                dataset_name = ds.path.name if ds.path else ds.name
                if dataset_name:
                    experiment = f"{prefix}/{dataset_name}"
            if not experiment:
                experiment = f"{prefix}/default"
        elif "/" not in experiment:
            experiment = f"{prefix}/{experiment}"

        try:
            ok = upload_single_run(
                rd,
                experiment,
                args.tracking_uri,
                force=args.force,
                extra_params=extra_params,
                extra_metrics=extra_metrics,
                ui_uri=args.ui_uri,
            )
            if ok:
                n_success += 1
            else:
                n_fail += 1
        except Exception as e:
            logger.error(f"Failed to upload {rd.name}: {e}")
            n_fail += 1

    logger.info(f"Done. Uploaded: {n_success}, Failed: {n_fail}")


if __name__ == "__main__":
    main()
