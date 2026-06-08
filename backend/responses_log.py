"""Build the persistent Responses Log from session/job events."""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

VISIBLE_BATCH_SIZE = 15

JOB_TO_PLATFORM = {
    "hf_jobs": "hf-jobs",
    "gcp_vertex_jobs": "gcp-vertex",
    "aws_sagemaker_jobs": "aws-sagemaker",
}

TERMINAL_STATES = {
    "completed",
    "complete",
    "error",
    "succeeded",
    "success",
    "failed",
    "failure",
    "cancelled",
    "canceled",
    "interrupted",
    "expired",
    "billing_required",
}

SECRET_PATTERNS = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-+/=]+", re.IGNORECASE),
    re.compile(r"hf_[A-Za-z0-9_]{8,}"),
    re.compile(r"(AKIA)[A-Z0-9]{12,}"),
    re.compile(
        r"(?i)(token|secret|password|api[_-]?key|access[_-]?key)(\s*[=:]\s*)"
        r"([^\s,;]+)"
    ),
]

LIGA_MARKER_RE = re.compile(r"^(LIGA_[A-Z0-9_]+)=(.*)$", re.MULTILINE)


def redact_response_value(value: Any) -> Any:
    """Return a copy of value with common token/secret patterns removed."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): redact_response_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_response_value(item) for item in value]
    text = str(value)
    for pattern in SECRET_PATTERNS:
        if pattern.groups >= 3:
            text = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
        elif pattern.groups == 1:
            text = pattern.sub(lambda m: f"{m.group(1)}[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(redact_response_value(value)).strip()
    return text or None


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            return None
    return None


def _iso(value: Any) -> str | None:
    parsed = _parse_dt(value)
    return parsed.isoformat() if parsed else _as_str(value)


def _markers_from_payload(*values: Any) -> dict[str, str]:
    markers: dict[str, str] = {}
    for value in values:
        if value is None:
            continue
        text = str(value)
        for key, raw in LIGA_MARKER_RE.findall(text):
            markers[key] = _as_str(raw) or ""
    return markers


def _event_created_at(event: dict[str, Any], session: dict[str, Any]) -> str | None:
    return _iso(
        event.get("created_at")
        or event.get("timestamp")
        or (event.get("data") or {}).get("created_at")
        or session.get("created_at")
    )


def _job_id(platform: str, data: dict[str, Any]) -> str | None:
    if platform == "hf-jobs":
        return _as_str(
            data.get("job_id")
            or data.get("jobId")
            or data.get("id")
            or data.get("jobUrl")
            or data.get("job_url")
        )
    if platform == "gcp-vertex":
        return _as_str(
            data.get("jobName") or data.get("job_name") or data.get("job_id")
        )
    if platform == "aws-sagemaker":
        return _as_str(
            data.get("jobName")
            or data.get("TrainingJobName")
            or data.get("job_name")
            or data.get("job_id")
        )
    return _as_str(data.get("job_id") or data.get("jobName"))


def _final_artifact(platform: str, data: dict[str, Any]) -> str:
    markers = _markers_from_payload(
        data.get("logs"),
        data.get("log"),
        data.get("formatted"),
        data.get("output"),
        data.get("failureReason"),
    )
    if markers.get("LIGA_FINAL_MODEL_URL"):
        return redact_response_value(markers["LIGA_FINAL_MODEL_URL"])
    if markers.get("LIGA_HUB_MODEL_ID"):
        return redact_response_value(
            f"https://huggingface.co/{markers['LIGA_HUB_MODEL_ID']}"
        )
    if platform == "hf-jobs":
        return (
            _as_str(
                data.get("finalModelUrl")
                or data.get("final_model_url")
                or data.get("hubModelId")
                or data.get("hub_model_id")
                or data.get("jobUrl")
                or data.get("failureReason")
                or data.get("failure_reason")
                or data.get("error")
            )
            or _as_str(data.get("state"))
            or "unknown"
        )
    if platform == "gcp-vertex":
        return (
            _as_str(data.get("outputDir") or data.get("output_dir"))
            or _as_str(data.get("failureReason"))
            or _as_str(data.get("jobUrl"))
            or _as_str(data.get("state"))
            or "unknown"
        )
    if platform == "aws-sagemaker":
        return (
            _as_str(
                data.get("s3ModelArtifact")
                or data.get("S3ModelArtifacts")
                or data.get("s3OutputUri")
                or data.get("S3OutputPath")
                or data.get("cloudWatchLogsUrl")
            )
            or _as_str(data.get("state"))
            or "unknown"
        )
    return _as_str(data.get("state")) or "unknown"


def _completed_at(
    state: str, event: dict[str, Any], session: dict[str, Any]
) -> str | None:
    if state.lower() in TERMINAL_STATES:
        return _event_created_at(event, session)
    return None


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def build_responses_log(
    sessions: list[dict[str, Any]],
    *,
    load_events: Callable[
        [str], list[dict[str, Any]] | Awaitable[list[dict[str, Any]]]
    ],
) -> dict[str, list[dict[str, Any]]]:
    """Build visible response rows from persisted/live session events."""
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    discovery_order = 0

    for session in sessions:
        session_id = str(session.get("session_id") or "")
        if not session_id:
            continue
        events = await _maybe_await(load_events(session_id))
        for event in events or []:
            if not isinstance(event, dict):
                continue
            if event.get("event_type") != "tool_state_change":
                continue
            data = event.get("data") or {}
            if not isinstance(data, dict):
                continue
            tool = str(data.get("tool") or "")
            platform = JOB_TO_PLATFORM.get(tool)
            if not platform:
                continue
            state = _as_str(data.get("state")) or "unknown"
            real_job_id = _job_id(platform, data)
            key_id = real_job_id or _as_str(data.get("tool_call_id")) or ""
            if not key_id:
                continue
            if real_job_id is None and state.lower() not in TERMINAL_STATES:
                continue
            discovery_order += 1
            created_at = _event_created_at(event, session)
            key = (session_id, platform, key_id)
            prior = rows_by_key.get(key)
            row = {
                "display_session_number": 0,
                "actual_sequence_number": 0,
                "batch_number": 0,
                "session_id": session_id,
                "short_session_id": session_id[:8],
                "session_title": _as_str(session.get("title")),
                "model_name": _as_str(session.get("model")) or "unknown",
                "platform": platform,
                "run_type": _as_str(session.get("training_goal")) or "agent-decide",
                "result_storage": _as_str(session.get("output_policy")) or "unknown",
                "progress": state,
                "job_id": real_job_id or "",
                "final_artifact_or_result": _final_artifact(platform, data),
                "created_at": prior.get("created_at") if prior else created_at,
                "completed_at": _completed_at(state, event, session),
                "_order": prior.get("_order") if prior else discovery_order,
            }
            rows_by_key[key] = row

    all_rows = sorted(
        rows_by_key.values(),
        key=lambda row: (
            _parse_dt(row.get("created_at")) or datetime.min.replace(tzinfo=UTC),
            row.get("_order") or 0,
        ),
    )
    for index, row in enumerate(all_rows, start=1):
        batch = ((index - 1) // VISIBLE_BATCH_SIZE) + 1
        row["actual_sequence_number"] = index
        row["display_session_number"] = ((index - 1) % VISIBLE_BATCH_SIZE) + 1
        row["batch_number"] = batch
        row.pop("_order", None)

    if not all_rows:
        return {"rows": []}

    current_batch = all_rows[-1]["batch_number"]
    return {"rows": [row for row in all_rows if row["batch_number"] == current_batch]}


def build_responses_summary(
    rows: list[dict[str, Any]],
    *,
    total_responses: int | None = None,
) -> dict[str, Any]:
    total = len(rows) if total_responses is None else total_responses
    batch = ((max(total, 1) - 1) // VISIBLE_BATCH_SIZE) + 1
    return {
        "total_responses": total,
        "visible_count": len(rows),
        "batch_number": batch,
        "has_rows": bool(rows),
        "button_enabled": True,
    }
