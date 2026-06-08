"""Build the persistent Responses Log from session/job events."""

from __future__ import annotations

import inspect
import json
import math
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
    "blocked",
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
HF_JOB_URL_RE = re.compile(r"https://huggingface\.co/jobs/[^\s`\"')\]]+")
HF_JOB_ID_RE = re.compile(r"\*\*Job ID:\*\*\s*([^\s`]+)", re.IGNORECASE)
FINAL_STATUS_RE = re.compile(r"\*\*Final Status:\*\*\s*([A-Za-z_]+)", re.IGNORECASE)
GCP_JOB_RE = re.compile(r"\*\*Job:\*\*\s*`?([^`\n]+)`?", re.IGNORECASE)
GCP_STATE_RE = re.compile(r"\*\*State:\*\*\s*([A-Z_]+)", re.IGNORECASE)
GCP_OUTPUT_DIR_RE = re.compile(r"\*\*Output dir:\*\*\s*([^\s`]+)", re.IGNORECASE)
JSON_ID_RE = re.compile(r'"id"\s*:\s*"([^"]+)"')
JSON_STAGE_RE = re.compile(r'"stage"\s*:\s*"([^"]+)"')
JSON_MESSAGE_RE = re.compile(r'"message"\s*:\s*("([^"]*)"|null)')


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


def _normalize_progress(state: Any) -> str:
    text = (_as_str(state) or "unknown").lower()
    if text.startswith("job_state_"):
        text = text.removeprefix("job_state_")
    if text in {"succeeded", "success", "complete", "done", "finished"}:
        return "completed"
    if text in {"failure"}:
        return "failed"
    if text in {"errored"}:
        return "error"
    if text in {"canceled"}:
        return "cancelled"
    if text in {"billing_required", "missing_token", "approval_required"}:
        return "blocked"
    if text in {"scheduling", "scheduled", "pending"}:
        return "queued"
    if text in {
        "queued",
        "running",
        "completed",
        "failed",
        "error",
        "cancelled",
        "interrupted",
        "blocked",
    }:
        return text
    return text


def _is_fake_job_id(value: Any) -> bool:
    text = (_as_str(value) or "").lower()
    return bool(
        text.startswith("functions.")
        or text.startswith("tool_call_")
        or text.startswith("call_")
    )


def _job_id(platform: str, data: dict[str, Any]) -> str | None:
    if platform == "hf-jobs":
        candidate = _as_str(
            data.get("jobUrl")
            or data.get("job_url")
            or data.get("job_id")
            or data.get("jobId")
            or data.get("id")
        )
        return None if _is_fake_job_id(candidate) else candidate
    if platform == "gcp-vertex":
        candidate = _as_str(
            data.get("job_id") or data.get("jobName") or data.get("job_name")
        )
        return None if _is_fake_job_id(candidate) else candidate
    if platform == "aws-sagemaker":
        candidate = _as_str(
            data.get("jobName")
            or data.get("TrainingJobName")
            or data.get("job_name")
            or data.get("job_id")
        )
        return None if _is_fake_job_id(candidate) else candidate
    candidate = _as_str(data.get("job_id") or data.get("jobName"))
    return None if _is_fake_job_id(candidate) else candidate


def _row_id(session_id: str, platform: str, key_id: str) -> str:
    return f"{session_id}:{platform}:{key_id}"


def _provider_metadata(data: dict[str, Any]) -> dict[str, Any]:
    noisy_keys = {"logs", "log", "formatted", "output"}
    return {
        str(key): redact_response_value(value)
        for key, value in data.items()
        if key not in noisy_keys
    }


def _failure_reason(data: dict[str, Any]) -> str | None:
    return _as_str(
        data.get("failureReason")
        or data.get("failure_reason")
        or data.get("error")
        or data.get("message")
    )


def _extract_json_fence(text: str) -> Any:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None


def _first_json_job(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return item
    if isinstance(value, dict):
        return value
    return {}


def _extract_hf_tool_output_data(data: dict[str, Any]) -> dict[str, Any] | None:
    output = _as_str(data.get("output") or data.get("formatted") or data.get("result"))
    if not output:
        return None

    parsed = _first_json_job(_extract_json_fence(output))
    status = parsed.get("status") if isinstance(parsed.get("status"), dict) else {}
    job_url = next(iter(HF_JOB_URL_RE.findall(output)), None)
    job_id = (
        _as_str((HF_JOB_ID_RE.search(output) or [None, None])[1])
        or _as_str(parsed.get("id"))
        or _as_str(data.get("job_id") or data.get("jobId") or data.get("id"))
    )
    stage = (
        _as_str(status.get("stage"))
        or _as_str((FINAL_STATUS_RE.search(output) or [None, None])[1])
        or _as_str((JSON_STAGE_RE.search(output) or [None, None])[1])
    )
    message = _as_str(status.get("message")) or _as_str(
        (JSON_MESSAGE_RE.search(output) or [None, None, None])[2]
    )
    if not stage and data.get("success") is True and job_id:
        stage = "success"
    if not stage and data.get("success") is False and job_id:
        stage = "failed"
    if not stage and not job_url and not job_id:
        return None

    extracted = {
        "tool_call_id": data.get("tool_call_id"),
        "tool": "hf_jobs",
        "state": stage or "unknown",
        "output": output,
    }
    if job_url:
        extracted["jobUrl"] = job_url
    if job_id and not _is_fake_job_id(job_id):
        extracted["job_id"] = job_id
    if message:
        extracted["message"] = message
        extracted["failureReason"] = message
    if data.get("success") is not None:
        extracted["success"] = data.get("success")
    return extracted


def _extract_gcp_tool_output_data(data: dict[str, Any]) -> dict[str, Any] | None:
    output = _as_str(data.get("output") or data.get("formatted") or data.get("result"))
    if not output:
        return None

    job_match = GCP_JOB_RE.search(output)
    state_match = GCP_STATE_RE.search(output)
    output_dir_match = GCP_OUTPUT_DIR_RE.search(output)
    job_name = _as_str(job_match.group(1)) if job_match else None
    state = _as_str(state_match.group(1)) if state_match else None
    output_dir = _as_str(output_dir_match.group(1)) if output_dir_match else None

    if not state and data.get("success") is False and job_name:
        state = "failed"
    if not state and not job_name:
        return None

    extracted = {
        "tool_call_id": data.get("tool_call_id"),
        "tool": "gcp_vertex_jobs",
        "state": state or "unknown",
        "output": output,
    }
    if job_name and not _is_fake_job_id(job_name):
        extracted["jobName"] = job_name
    if output_dir:
        extracted["outputDir"] = output_dir
    if data.get("success") is not None:
        extracted["success"] = data.get("success")
    return extracted


def _response_event_data(event: dict[str, Any]) -> dict[str, Any] | None:
    data = event.get("data") or {}
    if not isinstance(data, dict):
        return None
    event_type = event.get("event_type")
    if event_type == "tool_state_change":
        return data
    if event_type == "tool_output" and data.get("tool") == "hf_jobs":
        return _extract_hf_tool_output_data(data)
    if event_type == "tool_output" and data.get("tool") == "gcp_vertex_jobs":
        return _extract_gcp_tool_output_data(data)
    return None


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
    if (
        state.lower() in TERMINAL_STATES
        or _normalize_progress(state) in TERMINAL_STATES
    ):
        return _event_created_at(event, session)
    return None


def _hf_job_slug(value: Any) -> str | None:
    text = _as_str(value)
    if not text:
        return None
    return text.rstrip("/").split("/")[-1]


def _same_hf_job(left: Any, right: Any) -> bool:
    left_text = _as_str(left)
    right_text = _as_str(right)
    if not left_text or not right_text:
        return False
    return left_text == right_text or _hf_job_slug(left_text) == _hf_job_slug(
        right_text
    )


def _matching_response_key(
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]],
    *,
    session_id: str,
    platform: str,
    key_id: str,
) -> tuple[str, str, str] | None:
    exact = (session_id, platform, key_id)
    if exact in rows_by_key:
        return exact
    if platform != "hf-jobs":
        return None
    for key, row in rows_by_key.items():
        if key[0] != session_id or key[1] != platform:
            continue
        if _same_hf_job(row.get("job_id") or key[2], key_id):
            return key
    return None


def _prefer_job_id(prior: Any, candidate: Any) -> str:
    prior_text = _as_str(prior) or ""
    candidate_text = _as_str(candidate) or ""
    if candidate_text.startswith("https://"):
        return candidate_text
    return prior_text or candidate_text


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
    """Build response rows from persisted/live session events."""
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
            data = _response_event_data(event)
            if not data:
                continue
            tool = str(data.get("tool") or "")
            platform = JOB_TO_PLATFORM.get(tool)
            if not platform:
                continue
            raw_state = _as_str(data.get("state")) or "unknown"
            progress = _normalize_progress(raw_state)
            real_job_id = _job_id(platform, data)
            key_id = real_job_id or _as_str(data.get("tool_call_id")) or ""
            if not key_id:
                continue
            if event.get("event_type") == "tool_output" and real_job_id is None:
                continue
            if real_job_id is None and progress not in TERMINAL_STATES:
                continue
            discovery_order += 1
            created_at = _event_created_at(event, session)
            key = _matching_response_key(
                rows_by_key,
                session_id=session_id,
                platform=platform,
                key_id=key_id,
            ) or (session_id, platform, key_id)
            prior = rows_by_key.get(key)
            display_job_id = _prefer_job_id(
                prior.get("job_id") if prior else None,
                real_job_id,
            )
            failure_reason = _failure_reason(data)
            final_artifact = _final_artifact(platform, data)
            if (
                prior
                and final_artifact in {"unknown", raw_state, progress}
                and prior.get("final_artifact_or_result")
            ):
                final_artifact = prior["final_artifact_or_result"]
            row = {
                "id": _row_id(session_id, platform, key[2]),
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
                "progress": progress,
                "job_id": display_job_id,
                "final_artifact_or_result": final_artifact,
                "created_at": prior.get("created_at") if prior else created_at,
                "completed_at": _completed_at(progress, event, session),
                "provider_metadata": _provider_metadata(data),
                "error": failure_reason if progress in TERMINAL_STATES else None,
                "user_id": _as_str(session.get("user_id")) or "dev",
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

    return {"rows": all_rows}


def filter_response_rows(
    rows: list[dict[str, Any]],
    *,
    platform: str | None = None,
    progress: str | None = None,
    model: str | None = None,
    session_id: str | None = None,
    job_id: str | None = None,
    q: str | None = None,
    page: int | None = None,
    page_size: int | None = None,
) -> list[dict[str, Any]]:
    """Apply API filter/search semantics to response rows."""

    del page, page_size

    def contains(row: dict[str, Any], field: str, needle: str | None) -> bool:
        if not needle:
            return True
        return needle.lower() in str(row.get(field) or "").lower()

    filtered = list(rows)
    if platform:
        filtered = [row for row in filtered if contains(row, "platform", platform)]
    if progress:
        filtered = [row for row in filtered if contains(row, "progress", progress)]
    if model:
        filtered = [row for row in filtered if contains(row, "model_name", model)]
    if session_id:
        filtered = [row for row in filtered if contains(row, "session_id", session_id)]
    if job_id:
        filtered = [row for row in filtered if contains(row, "job_id", job_id)]
    if q:
        needle = q.lower()
        search_fields = (
            "id",
            "session_id",
            "short_session_id",
            "session_title",
            "model_name",
            "platform",
            "run_type",
            "result_storage",
            "progress",
            "job_id",
            "final_artifact_or_result",
            "error",
        )
        filtered = [
            row
            for row in filtered
            if any(
                needle in str(row.get(field) or "").lower() for field in search_fields
            )
        ]
    return filtered


def paginate_response_rows(
    rows: list[dict[str, Any]],
    *,
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    """Return newest-first paginated API response."""

    safe_page = max(1, int(page or 1))
    safe_page_size = min(200, max(1, int(page_size or 50)))
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row.get("actual_sequence_number") or 0),
            _parse_dt(row.get("created_at")) or datetime.min.replace(tzinfo=UTC),
        ),
        reverse=True,
    )
    total_rows = len(ordered)
    total_pages = math.ceil(total_rows / safe_page_size) if total_rows else 0
    start = (safe_page - 1) * safe_page_size
    page_rows = ordered[start : start + safe_page_size]
    return {
        "rows": page_rows,
        "page": safe_page,
        "page_size": safe_page_size,
        "total_rows": total_rows,
        "total_pages": total_pages,
        "has_next": bool(total_pages and safe_page < total_pages),
        "has_previous": safe_page > 1 and total_pages > 0,
    }


def build_responses_summary(
    rows: list[dict[str, Any]],
    *,
    total_responses: int | None = None,
    durable: bool = False,
    store_type: str = "memory",
) -> dict[str, Any]:
    total = len(rows) if total_responses is None else total_responses
    batch = ((max(total, 1) - 1) // VISIBLE_BATCH_SIZE) + 1
    return {
        "total_responses": total,
        "visible_count": len(rows),
        "batch_number": batch,
        "has_rows": bool(rows),
        "button_enabled": True,
        "durable": durable,
        "store_type": store_type,
    }
