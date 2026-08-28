"""In-container bridge from an aws-bench trial to a CloudGeni agent session.

This file is uploaded into the isolated Harbor environment by ``CloudGeniAgent``.
It deliberately uses only the Python standard library so the adapter adds no
runtime dependency to task images.
"""

from __future__ import annotations

import configparser
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

_PROMPT_PATH = Path("/installed-agent/cloudgeni-prompt.txt")
_LOGS_DIR = Path("/logs/agent")
_OUTPUT_TEXT_PATH = _LOGS_DIR / "agent-output.txt"
_OUTPUT_JSON_PATH = _LOGS_DIR / "agent-output.json"
_STATE_PATH = _LOGS_DIR / "cloudgeni-state.json"
_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}
_REQUIRED_ENV = (
    "CLOUDGENI_API_URL",
    "CLOUDGENI_API_KEY",
    "CLOUDGENI_ORGANIZATION_ID",
    "CLOUDGENI_WORKSPACE_ID",
)


class BridgeError(RuntimeError):
    """A user-safe bridge failure."""


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise BridgeError(f"Missing required environment variable: {name}")
    return value


def _api_root(value: str) -> str:
    root = value.rstrip("/")
    return root if root.endswith("/api/v1") else f"{root}/api/v1"


def _request(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    accepted_statuses: set[int] | None = None,
) -> tuple[int, dict[str, Any]]:
    api_url = _api_root(_required_env("CLOUDGENI_API_URL"))
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"{api_url}{path}",
        data=body,
        method=method,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-CLOUDGENI-API-KEY": _required_env("CLOUDGENI_API_KEY"),
        },
    )
    accepted = accepted_statuses or {200, 201}
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read()
    except urllib.error.URLError as error:
        raise BridgeError(f"CloudGeni API request failed: {error.reason}") from error

    try:
        decoded = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        decoded = {}
    if status not in accepted:
        message = decoded.get("message") or decoded.get("error") or f"HTTP {status}"
        if isinstance(message, dict):
            message = message.get("message") or message.get("code") or f"HTTP {status}"
        raise BridgeError(f"CloudGeni API request failed ({status}): {message}")
    return status, decoded


def _credentials() -> dict[str, str]:
    profile = os.environ.get("AWS_PROFILE", "default")
    credentials_path = Path(
        os.path.expandvars(os.environ.get("AWS_SHARED_CREDENTIALS_FILE", "$HOME/.aws/credentials"))
    ).expanduser()
    parser = configparser.RawConfigParser()
    if not parser.read(credentials_path):
        raise BridgeError("AWS-Bench credential file is unavailable")
    if not parser.has_section(profile):
        raise BridgeError(f"AWS-Bench credential profile is unavailable: {profile}")

    values = {
        "accessKeyId": parser.get(profile, "aws_access_key_id", fallback="").strip(),
        "secretAccessKey": parser.get(profile, "aws_secret_access_key", fallback="").strip(),
        "sessionToken": parser.get(profile, "aws_session_token", fallback="").strip(),
    }
    if not all(values.values()):
        raise BridgeError("AWS-Bench supplied an incomplete temporary credential profile")
    if not values["accessKeyId"].startswith("ASIA"):
        raise BridgeError("CloudGeni benchmark trials require temporary ASIA credentials")
    return values


def _caller_account_id() -> str:
    profile = os.environ.get("AWS_PROFILE", "default")
    region = os.environ.get("AWS_REGION", "us-east-1")
    try:
        result = subprocess.run(
            [
                "aws",
                "sts",
                "get-caller-identity",
                "--profile",
                profile,
                "--region",
                region,
                "--output",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        account_id = json.loads(result.stdout).get("Account", "")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise BridgeError("Unable to verify the AWS-Bench trial identity") from error
    if not re.fullmatch(r"\d{12}", account_id):
        raise BridgeError("AWS STS returned an invalid account identity")
    return account_id


def _workspace_path(suffix: str) -> str:
    organization_id = _required_env("CLOUDGENI_ORGANIZATION_ID")
    workspace_id = _required_env("CLOUDGENI_WORKSPACE_ID")
    return f"/organizations/{organization_id}/workspaces/{workspace_id}{suffix}"


def _write_state(state: dict[str, Any]) -> None:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = _STATE_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(state, indent=2, sort_keys=True))
    temporary_path.replace(_STATE_PATH)


def _create_lease(run_id: str, account_id: str, credentials: dict[str, str]) -> str:
    region = os.environ.get("AWS_REGION", "us-east-1")
    lease_seconds = int(os.environ.get("CLOUDGENI_AWS_BENCH_LEASE_SECONDS", "3500"))
    if not 120 <= lease_seconds <= 3600:
        raise BridgeError("CLOUDGENI_AWS_BENCH_LEASE_SECONDS must be between 120 and 3600")
    expires_at = datetime.now(UTC) + timedelta(seconds=lease_seconds)
    _, response = _request(
        "POST",
        _workspace_path("/integrations?method=aws-bench-credential-lease"),
        payload={
            "name": f"AWS-Bench {run_id}",
            "description": "Ephemeral AWS-Bench trial credential; purge after this run.",
            "type": "CLOUD",
            "provider": "AWS",
            "benchmarkRunId": run_id,
            "providerConfig": {"accountId": account_id, "defaultRegion": region},
            "credential": {
                "name": f"AWS-Bench {run_id}",
                **credentials,
                "expiresAt": expires_at.isoformat().replace("+00:00", "Z"),
            },
        },
    )
    integration_id = response.get("data", {}).get("id")
    if not isinstance(integration_id, str) or not integration_id:
        raise BridgeError("CloudGeni did not return a credential lease ID")
    return integration_id


def _create_session(run_id: str, integration_id: str, instruction: str) -> tuple[str, str]:
    bridge_instruction = (
        f"{instruction.rstrip()}\n\n"
        "CloudGeni bridge note: /logs/agent is owned by AWS-Bench and is not mounted in "
        "your sandbox. Return the exact intended file contents as your final response; "
        "the bridge will write that response to the requested AWS-Bench output path."
    )
    _, response = _request(
        "POST",
        _workspace_path("/agent-sessions"),
        payload={
            "idempotencyKey": f"aws-bench:{run_id}",
            "agentSlug": "aws-bench",
            "prompt": bridge_instruction,
            "cloudIntegrationIds": [integration_id],
        },
    )
    data = response.get("data", {})
    session_id = data.get("sessionId")
    cloudgeni_run_id = data.get("runId")
    if not isinstance(session_id, str) or not isinstance(cloudgeni_run_id, str):
        raise BridgeError("CloudGeni did not return an agent session identity")
    return session_id, cloudgeni_run_id


def _wait_for_session(session_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        _, response = _request("GET", _workspace_path(f"/agent-sessions/{session_id}"))
        state = response.get("data", {})
        status = state.get("latestRun", {}).get("status")
        if status in _TERMINAL_STATUSES:
            return state
        time.sleep(2)
    raise BridgeError(f"CloudGeni agent did not finish within {timeout_seconds} seconds")


def _text_from_part(part: Any) -> str:
    if not isinstance(part, dict):
        return ""
    text = part.get("text")
    if isinstance(text, str):
        return text
    output = part.get("output")
    if isinstance(output, str):
        return output
    return ""


def _result_output(session: dict[str, Any]) -> str:
    result = session.get("latestRun", {}).get("result")
    if not isinstance(result, dict):
        return ""
    output = result.get("output")
    return output.strip() if isinstance(output, str) else ""


def _final_assistant_text(session_id: str, session: dict[str, Any] | None = None) -> str:
    output = _result_output(session or {})
    if output:
        return output
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        _, response = _request("GET", _workspace_path(f"/agent-sessions/{session_id}/messages"))
        messages = response.get("data", {}).get("messages", [])
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            text = "".join(_text_from_part(part) for part in message.get("parts", [])).strip()
            if text:
                return text
        time.sleep(2)
    raise BridgeError("CloudGeni completed without a projected assistant response")


def _extract_json(text: str) -> Any:
    candidates = [text.strip()]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    )
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise BridgeError("CloudGeni did not return valid JSON for agent-output.json")


def _write_output(instruction: str, answer: str) -> None:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _OUTPUT_TEXT_PATH.write_text(answer.rstrip() + "\n")
    if "/logs/agent/agent-output.json" in instruction:
        _OUTPUT_JSON_PATH.write_text(
            json.dumps(_extract_json(answer), separators=(",", ":")) + "\n"
        )


def _cleanup(session_id: str | None, integration_id: str | None) -> dict[str, bool]:
    result = {"sessionDeleted": session_id is None, "leasePurged": integration_id is None}
    if session_id:
        try:
            _request(
                "DELETE",
                _workspace_path(f"/agent-sessions/{session_id}"),
                accepted_statuses={200, 404},
            )
            result["sessionDeleted"] = True
        except BridgeError:
            pass
    if integration_id:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                _request(
                    "DELETE",
                    _workspace_path(f"/integrations/{integration_id}/aws-bench-credential-lease"),
                    accepted_statuses={200, 404},
                )
                result["leasePurged"] = True
                break
            except BridgeError:
                time.sleep(2)
    return result


def main() -> int:
    """Run one CloudGeni-backed AWS-Bench trial."""
    for name in _REQUIRED_ENV:
        _required_env(name)
    instruction = _PROMPT_PATH.read_text()
    run_id = f"trial-{uuid4().hex}"
    state: dict[str, Any] = {
        "schemaVersion": 1,
        "benchmarkRunId": run_id,
        "sessionId": None,
        "cloudgeniRunId": None,
        "integrationId": None,
        "status": "STARTING",
        "cleanup": {"sessionDeleted": False, "leasePurged": False},
    }
    _write_state(state)

    def interrupt(_signum: int, _frame: Any) -> None:
        raise InterruptedError("AWS-Bench interrupted the CloudGeni bridge")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    try:
        credentials = _credentials()
        account_id = _caller_account_id()
        integration_id = _create_lease(run_id, account_id, credentials)
        state["integrationId"] = integration_id
        state["status"] = "LEASE_CREATED"
        _write_state(state)

        session_id, cloudgeni_run_id = _create_session(run_id, integration_id, instruction)
        state.update(
            {
                "sessionId": session_id,
                "cloudgeniRunId": cloudgeni_run_id,
                "status": "RUNNING",
            }
        )
        _write_state(state)

        timeout_seconds = int(os.environ.get("CLOUDGENI_AWS_BENCH_TIMEOUT_SECONDS", "3300"))
        if not 30 <= timeout_seconds <= 3480:
            raise BridgeError("CLOUDGENI_AWS_BENCH_TIMEOUT_SECONDS must be between 30 and 3480")
        session = _wait_for_session(session_id, timeout_seconds)
        status = session.get("latestRun", {}).get("status")
        state["status"] = status
        state["usage"] = {
            "inputTokens": session.get("totalInputTokens"),
            "outputTokens": session.get("totalOutputTokens"),
            "modelTokens": session.get("modelTokenUsage"),
        }
        _write_state(state)
        if status != "COMPLETED":
            raise BridgeError(f"CloudGeni agent ended with status {status}")
        _write_output(instruction, _final_assistant_text(session_id, session))
        return 0
    except (BridgeError, InterruptedError, ValueError) as error:
        state["status"] = "BRIDGE_FAILED"
        state["error"] = str(error)
        _OUTPUT_TEXT_PATH.write_text(f"CloudGeni bridge failed: {error}\n")
        return 1
    finally:
        state["cleanup"] = _cleanup(state.get("sessionId"), state.get("integrationId"))
        _write_state(state)


if __name__ == "__main__":
    sys.exit(main())
