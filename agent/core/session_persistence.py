"""Optional durable session persistence for the hosted backend.

The public CLI must keep working without MongoDB.  This module therefore
exposes one small async store interface and returns a no-op implementation
unless ``MONGODB_URI`` is configured and reachable.
"""

from __future__ import annotations

import logging
import math
import os
import re
import uuid
from datetime import UTC, datetime
from typing import Any

from agent.core.background_runs import (
    RUN_TERMINAL_STATUSES,
    provider_metadata_from_event,
    run_status_from_event,
    safe_event_summary,
)
from agent.core.audit import (
    audit_timeline_enabled,
    build_audit_event,
    event_from_run_event,
    summarize_audit_events,
    training_preflight_audit_events,
)
from agent.core.dataset_discovery import (
    build_dataset_discovery_result,
    extract_hf_dataset_candidates_from_text,
)
from agent.core.post_training_evaluation import (
    build_post_training_evaluation,
    evaluation_context_from_liga_output,
    summarize_evaluations,
)
from agent.core.usage import (
    summarize_usage,
    usage_from_training_recommendation,
    usage_from_training_preflight,
    usage_from_approval_tool,
    usage_from_run_terminal,
    usage_from_tool_state,
)
from agent.core.redact import sanitize_for_persistence
from bson import BSON
from pymongo import AsyncMongoClient, DeleteMany, ReturnDocument, UpdateOne
from pymongo.errors import DuplicateKeyError, InvalidDocument, PyMongoError

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_BSON_BYTES = 15 * 1024 * 1024
TERMINAL_RESPONSE_PROGRESS = {
    "completed",
    "failed",
    "error",
    "cancelled",
    "interrupted",
    "blocked",
}
NO_DURABLE_STORE_WARNING = (
    "MONGODB_URI is not configured; sessions will not survive restarts"
)


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed
    return None


def _doc_id(session_id: str, idx: int) -> str:
    return f"{session_id}:{idx}"


def _hf_job_slug(value: Any) -> str:
    return str(value or "").strip().rstrip("/").split("/")[-1]


def _same_hf_job_id(left: Any, right: Any) -> bool:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text or not right_text:
        return False
    return left_text == right_text or _hf_job_slug(left_text) == _hf_job_slug(
        right_text
    )


def _hf_job_slug_regex(value: Any) -> dict[str, str] | None:
    slug = _hf_job_slug(value)
    if not slug:
        return None
    return {"$regex": rf"(^|/){re.escape(slug)}/?$"}


def _response_is_terminal(row: dict[str, Any]) -> bool:
    return str(row.get("progress") or "").lower() in TERMINAL_RESPONSE_PROGRESS


def _hf_model_artifact(value: Any) -> bool:
    text = str(value or "").strip()
    return text.startswith("https://huggingface.co/") and not text.startswith(
        (
            "https://huggingface.co/jobs/",
            "https://huggingface.co/datasets/",
            "https://huggingface.co/spaces/",
        )
    )


def _response_row_score(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        int(_response_is_terminal(row)),
        int(bool(row.get("completed_at"))),
        int(_hf_model_artifact(row.get("final_artifact_or_result"))),
        int(str(row.get("result_storage") or "") == "hf-hub"),
        int(str(row.get("run_type") or "") == "smoke-test"),
    )


def _find_existing_response_row(
    candidates: list[dict[str, Any]], row: dict[str, Any], row_user_id: str
) -> dict[str, Any]:
    platform = str(row.get("platform") or "")
    job_id = str(row.get("job_id") or "")
    session_id = str(row.get("session_id") or "")

    def platform_matches(doc: dict[str, Any]) -> bool:
        return str(doc.get("platform") or "") == platform

    def exact_job_matches(doc: dict[str, Any]) -> bool:
        return str(doc.get("job_id") or "") == job_id

    def hf_job_matches(doc: dict[str, Any]) -> bool:
        return platform == "hf-jobs" and _same_hf_job_id(doc.get("job_id"), job_id)

    matchers = (
        lambda doc: (
            str(doc.get("user_id") or "") == row_user_id
            and platform_matches(doc)
            and exact_job_matches(doc)
        ),
        lambda doc: (
            str(doc.get("user_id") or "") == row_user_id
            and platform_matches(doc)
            and hf_job_matches(doc)
        ),
        lambda doc: (
            session_id
            and str(doc.get("session_id") or "") == session_id
            and platform_matches(doc)
            and exact_job_matches(doc)
        ),
        lambda doc: (
            session_id
            and str(doc.get("session_id") or "") == session_id
            and platform_matches(doc)
            and hf_job_matches(doc)
        ),
    )
    matches = [doc for matcher in matchers for doc in candidates if matcher(doc)]
    if not matches:
        return {}
    return max(matches, key=_response_row_score)


def _merge_response_row(
    existing: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    if not existing:
        return row
    merged = dict(row)
    existing_terminal = _response_is_terminal(existing)
    incoming_terminal = _response_is_terminal(row)
    if existing_terminal and not incoming_terminal:
        for key in ("progress", "provider_metadata", "error"):
            if existing.get(key) is not None:
                merged[key] = existing[key]
    if existing.get("completed_at") and not row.get("completed_at"):
        merged["completed_at"] = existing["completed_at"]
    if (
        str(existing.get("result_storage") or "") == "hf-hub"
        and str(row.get("result_storage") or "") != "hf-hub"
    ):
        merged["result_storage"] = "hf-hub"
    if (
        str(existing.get("run_type") or "") == "smoke-test"
        and str(row.get("run_type") or "") == "agent-decide"
    ):
        merged["run_type"] = "smoke-test"
    if _hf_model_artifact(existing.get("final_artifact_or_result")) and not (
        _hf_model_artifact(row.get("final_artifact_or_result"))
    ):
        merged["final_artifact_or_result"] = existing["final_artifact_or_result"]
    if _hf_model_artifact(merged.get("final_artifact_or_result")):
        merged["result_storage"] = "hf-hub"
        if "smoke" in str(merged.get("final_artifact_or_result") or "").lower():
            merged["run_type"] = "smoke-test"
    return merged


def _safe_message_doc(message: dict[str, Any]) -> dict[str, Any]:
    """Return a Mongo-safe message document payload.

    Mongo's hard document limit is 16 MB.  We stay below that and store an
    explicit marker rather than failing the whole snapshot for one huge tool log.
    """
    safe_message = sanitize_for_persistence(message)
    try:
        if len(BSON.encode({"message": safe_message})) <= MAX_BSON_BYTES:
            return safe_message
    except (InvalidDocument, OverflowError):
        pass
    return {
        "role": "tool",
        "content": (
            "[SYSTEM: A single persisted message exceeded MongoDB's document "
            "size/encoding limit and was replaced by this marker.]"
        ),
        "ml_intern_persistence_error": "message_too_large_or_invalid",
    }


def _dataset_discovery_from_event(
    event_type: str, payload: dict[str, Any]
) -> dict[str, Any] | None:
    if event_type != "tool_output":
        return None
    tool = payload.get("tool")
    if tool not in {"dataset_discovery", "research"}:
        return None
    structured = payload.get("structured")
    if isinstance(structured, dict):
        if tool == "research" and structured.get("kind") != "dataset_discovery":
            return None
        return sanitize_for_persistence(structured)
    if tool == "research" and payload.get("success") is not False:
        text = str(payload.get("output") or payload.get("formatted") or "")
        candidates = extract_hf_dataset_candidates_from_text(text)
        if candidates:
            result = build_dataset_discovery_result(
                query="Research-derived dataset candidates",
                candidates=candidates,
            )
            return sanitize_for_persistence(result.to_dict())
    return None


def _training_recommendation_from_event(
    event_type: str, payload: dict[str, Any]
) -> dict[str, Any] | None:
    if event_type != "tool_output" or payload.get("tool") != "training_planner":
        return None
    structured = payload.get("structured")
    if isinstance(structured, dict):
        return sanitize_for_persistence(structured)
    return None


async def _resolve_active_run_id(
    store: Any,
    *,
    session_id: str,
    run_id: str | None,
) -> str | None:
    if run_id:
        return run_id
    list_runs = getattr(store, "list_runs", None)
    if not callable(list_runs):
        return None
    runs = await list_runs(session_id)
    if not isinstance(runs, list):
        return None
    for run in runs:
        if not isinstance(run, dict):
            continue
        status = str(run.get("status") or "")
        if status not in RUN_TERMINAL_STATUSES:
            return str(run.get("run_id") or run.get("_id") or "") or None
    if runs and isinstance(runs[0], dict):
        return str(runs[0].get("run_id") or runs[0].get("_id") or "") or None
    return None


class NoopSessionStore:
    """Async no-op store used when Mongo is not configured."""

    enabled = False

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}
        self._run_events: dict[str, list[dict[str, Any]]] = {}
        self._usage_entries: dict[str, dict[str, Any]] = {}
        self._audit_events: dict[str, dict[str, Any]] = {}
        self._evaluations: dict[str, dict[str, Any]] = {}
        self._session_preflights: dict[str, dict[str, Any]] = {}
        self._run_preflights: dict[str, dict[str, Any]] = {}

    async def init(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upsert_session(self, **_: Any) -> None:
        return None

    async def save_snapshot(self, **_: Any) -> None:
        return None

    async def load_session(self, *_: Any, **__: Any) -> dict[str, Any] | None:
        return None

    async def list_sessions(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []

    async def soft_delete_session(self, *_: Any, **__: Any) -> None:
        return None

    async def update_session_fields(self, *_: Any, **__: Any) -> None:
        return None

    async def append_event(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any] | None,
        run_id: str | None = None,
    ) -> int | None:
        if run_id:
            return await self.append_run_event(
                run_id=run_id,
                session_id=session_id,
                event_type=event_type,
                payload=sanitize_for_persistence(data or {}),
            )
        return None

    async def load_events_after(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []

    async def append_trace_message(self, *_: Any, **__: Any) -> int | None:
        return None

    async def load_trace_messages(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []

    async def get_quota(self, *_: Any, **__: Any) -> int | None:
        return None

    async def try_increment_quota(self, *_: Any, **__: Any) -> int | None:
        return None

    async def refund_quota(self, *_: Any, **__: Any) -> None:
        return None

    async def mark_pro_seen(self, *_: Any, **__: Any) -> dict[str, Any] | None:
        return None

    async def create_run(
        self,
        *,
        session_id: str,
        provider: str = "none",
        request_id: str | None = None,
        status: str = "queued",
    ) -> dict[str, Any]:
        now = _now()
        run_id = str(uuid.uuid4())
        run = {
            "_id": run_id,
            "run_id": run_id,
            "session_id": session_id,
            "status": status,
            "provider": provider or "none",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
            "last_event_seq": 0,
            "active_tool": None,
            "active_provider_job_id": None,
            "approval_id": None,
            "error_summary": None,
            "result_summary": None,
            "request_id": request_id,
            "provider_metadata": {},
        }
        self._runs[run_id] = run
        self._run_events[run_id] = []
        await self.append_run_event(
            run_id=run_id,
            session_id=session_id,
            event_type="run_created",
            payload={"request_id": request_id, "provider": provider},
        )
        return dict(run)

    async def update_run(self, run_id: str, **fields: Any) -> dict[str, Any] | None:
        run = self._runs.get(run_id)
        if not run:
            return None
        fields = sanitize_for_persistence(
            {k: v for k, v in fields.items() if v is not None}
        )
        if fields:
            run.update(fields)
        run["updated_at"] = _now()
        return dict(run)

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        run = self._runs.get(run_id)
        return dict(run) if run else None

    async def record_training_preflight(
        self,
        *,
        session_id: str,
        preflight: dict[str, Any],
        run_id: str | None = None,
        include_started_audit: bool = True,
    ) -> dict[str, Any]:
        clean = sanitize_for_persistence(preflight)
        if not isinstance(clean, dict):
            clean = {}
        clean["session_id"] = session_id
        if run_id:
            clean["run_id"] = run_id
        self._session_preflights[session_id] = dict(clean)
        if run_id:
            self._run_preflights[run_id] = dict(clean)
            run = self._runs.get(run_id)
            if run:
                provider_metadata = dict(run.get("provider_metadata") or {})
                provider_metadata["training_preflight"] = dict(clean)
                run["training_preflight"] = dict(clean)
                run["provider_metadata"] = provider_metadata
                run["updated_at"] = _now()
        for event in training_preflight_audit_events(
            clean, include_started=include_started_audit
        ):
            await self.record_audit_event(event)
        if run_id:
            usage_update = usage_from_training_preflight(
                session_id=session_id,
                run_id=run_id,
                preflight=clean,
            )
            if usage_update:
                usage_id, entry = usage_update
                await self.upsert_usage_entry(usage_id, entry)
        return dict(clean)

    async def get_latest_training_preflight(
        self, session_id: str
    ) -> dict[str, Any] | None:
        preflight = self._session_preflights.get(session_id)
        return dict(preflight) if preflight else None

    async def get_run_training_preflight(
        self, session_id: str, run_id: str
    ) -> dict[str, Any] | None:
        preflight = self._run_preflights.get(run_id)
        if preflight and preflight.get("session_id") == session_id:
            return dict(preflight)
        run = self._runs.get(run_id)
        if run and run.get("session_id") == session_id:
            value = run.get("training_preflight")
            if isinstance(value, dict):
                return dict(value)
        return None

    async def upsert_evaluation(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        now = _now()
        clean = sanitize_for_persistence(
            {key: value for key, value in evaluation.items() if value is not None}
        )
        evaluation_id = str(clean.get("evaluation_id") or clean.get("_id") or "")
        if not evaluation_id:
            evaluation_id = str(uuid.uuid4())
            clean["evaluation_id"] = evaluation_id
        clean["_id"] = evaluation_id
        clean.setdefault("created_at", now)
        clean["updated_at"] = now
        current = dict(self._evaluations.get(evaluation_id) or {})
        current.update(clean)
        self._evaluations[evaluation_id] = current
        run_id = str(current.get("run_id") or "")
        if run_id in self._runs:
            scores = (
                current.get("scores") if isinstance(current.get("scores"), dict) else {}
            )
            self._runs[run_id].update(
                {
                    "evaluation_status": current.get("status"),
                    "evaluation_score": scores.get("overall_score"),
                    "evaluation_id": evaluation_id,
                    "updated_at": now,
                }
            )
        await self._record_evaluation_audit_events(current)
        return dict(current)

    async def list_evaluations(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        provider: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        evaluations = list(self._evaluations.values())
        if session_id:
            evaluations = [
                evaluation
                for evaluation in evaluations
                if evaluation.get("session_id") == session_id
            ]
        if run_id:
            evaluations = [
                evaluation
                for evaluation in evaluations
                if evaluation.get("run_id") == run_id
            ]
        if provider:
            evaluations = [
                evaluation
                for evaluation in evaluations
                if evaluation.get("provider") == provider
            ]
        if status:
            evaluations = [
                evaluation
                for evaluation in evaluations
                if evaluation.get("status") == status
            ]
        evaluations = sorted(
            (dict(evaluation) for evaluation in evaluations),
            key=lambda item: (
                _parse_dt(item.get("updated_at"))
                or _parse_dt(item.get("completed_at"))
                or _now()
            ),
            reverse=True,
        )
        return evaluations[: max(1, min(int(limit or 100), 500))]

    async def get_evaluation_for_run(
        self, session_id: str, run_id: str
    ) -> dict[str, Any] | None:
        evaluations = await self.list_evaluations(
            session_id=session_id, run_id=run_id, limit=1
        )
        return evaluations[0] if evaluations else None

    async def evaluation_summary(self, **filters: Any) -> dict[str, Any]:
        evaluations = await self.list_evaluations(**filters)
        return {**summarize_evaluations(evaluations), "evaluations": evaluations}

    async def _record_evaluation_audit_events(self, evaluation: dict[str, Any]) -> None:
        session_id = str(evaluation.get("session_id") or "")
        run_id = str(evaluation.get("run_id") or "")
        if not session_id or not run_id:
            return
        status = str(evaluation.get("status") or "unknown")
        provider = str(evaluation.get("provider") or "unknown")
        evaluation_id = str(evaluation.get("evaluation_id") or "")
        lifecycle = [
            ("evaluation_planned", "planned", "Evaluation planned"),
            ("evaluation_started", "running", "Evaluation started"),
        ]
        if status == "succeeded":
            lifecycle.append(
                ("evaluation_completed", "succeeded", "Evaluation completed")
            )
        elif status == "skipped":
            lifecycle.append(("evaluation_skipped", "skipped", "Evaluation skipped"))
        elif status == "failed":
            lifecycle.append(("evaluation_failed", "failed", "Evaluation failed"))
        else:
            lifecycle.append(
                ("evaluation_unavailable", status, "Evaluation unavailable")
            )
        for event_type, event_status, title in lifecycle:
            await self.record_audit_event(
                build_audit_event(
                    session_id=session_id,
                    run_id=run_id,
                    event_type=event_type,
                    category="result",
                    severity="error" if event_status == "failed" else "info",
                    status=event_status,
                    actor="system",
                    title=title,
                    message=str(evaluation.get("recommendation") or title),
                    provider=provider,
                    entity_type="evaluation",
                    entity_id=evaluation_id,
                    artifact_url=evaluation.get("artifact_ref"),
                    model_name=evaluation.get("model_ref"),
                    safe_metadata={
                        "evaluation_id": evaluation_id,
                        "scores": evaluation.get("scores"),
                    },
                )
            )

    async def upsert_usage_entry(
        self, usage_id: str, fields: dict[str, Any]
    ) -> dict[str, Any]:
        now = _now()
        current = dict(self._usage_entries.get(usage_id) or {})
        if not current:
            current = {
                "_id": usage_id,
                "usage_id": usage_id,
                "created_at": fields.get("created_at") or now,
            }
        cleaned = sanitize_for_persistence(
            {key: value for key, value in fields.items() if value is not None}
        )
        current.update(cleaned)
        current["updated_at"] = now
        self._usage_entries[usage_id] = current
        return dict(current)

    async def list_usage_entries(
        self,
        *,
        provider: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        entries = list(self._usage_entries.values())
        if provider:
            entries = [entry for entry in entries if entry.get("provider") == provider]
        if session_id:
            entries = [
                entry for entry in entries if entry.get("session_id") == session_id
            ]
        if run_id:
            entries = [entry for entry in entries if entry.get("run_id") == run_id]
        if status:
            entries = [entry for entry in entries if entry.get("status") == status]
        entries = sorted(
            (dict(entry) for entry in entries),
            key=lambda item: item.get("updated_at") or item.get("created_at") or _now(),
            reverse=True,
        )
        return entries[: max(1, min(int(limit or 100), 500))]

    async def usage_summary(self, **filters: Any) -> dict[str, Any]:
        entries = await self.list_usage_entries(**filters)
        return {**summarize_usage(entries), "entries": entries}

    async def record_audit_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        if not audit_timeline_enabled():
            return None
        if not hasattr(self, "_audit_events"):
            self._audit_events = {}
        audit_id = str(event.get("audit_id") or event.get("_id") or "")
        if not audit_id:
            return None
        current = self._audit_events.get(audit_id)
        if current:
            return dict(current)
        event = sanitize_for_persistence(
            {key: value for key, value in event.items() if value is not None}
        )
        event.setdefault("_id", audit_id)
        event.setdefault("audit_id", audit_id)
        self._audit_events[audit_id] = dict(event)
        logger.info(
            "audit_event audit_id=%s session_id=%s run_id=%s category=%s "
            "event_type=%s provider=%s severity=%s status=%s",
            event.get("audit_id"),
            event.get("session_id"),
            event.get("run_id"),
            event.get("category"),
            event.get("event_type"),
            event.get("provider"),
            event.get("severity"),
            event.get("status"),
        )
        return dict(event)

    async def list_audit_events(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        provider: str | None = None,
        category: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not hasattr(self, "_audit_events"):
            self._audit_events = {}
        events = list(self._audit_events.values())
        if session_id:
            events = [
                event for event in events if event.get("session_id") == session_id
            ]
        if run_id:
            events = [event for event in events if event.get("run_id") == run_id]
        if provider:
            events = [event for event in events if event.get("provider") == provider]
        if category:
            events = [event for event in events if event.get("category") == category]
        if severity:
            events = [event for event in events if event.get("severity") == severity]
        if status:
            events = [event for event in events if event.get("status") == status]
        since_dt = _parse_dt(since)
        until_dt = _parse_dt(until)
        if since_dt:
            events = [
                event
                for event in events
                if (_parse_dt(event.get("timestamp")) or _now()) >= since_dt
            ]
        if until_dt:
            events = [
                event
                for event in events
                if (_parse_dt(event.get("timestamp")) or _now()) <= until_dt
            ]
        events = sorted(
            (dict(event) for event in events),
            key=lambda item: _parse_dt(item.get("timestamp")) or _now(),
        )
        return events[: max(1, min(int(limit or 100), 500))]

    async def audit_summary(self, **filters: Any) -> dict[str, Any]:
        events = await self.list_audit_events(**filters)
        return {**summarize_audit_events(events), "events": events}

    async def list_runs(self, session_id: str) -> list[dict[str, Any]]:
        runs = [
            dict(run)
            for run in self._runs.values()
            if run.get("session_id") == session_id
        ]
        return sorted(
            runs, key=lambda item: item.get("created_at") or _now(), reverse=True
        )

    async def append_run_event(
        self,
        *,
        run_id: str,
        session_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> int | None:
        if run_id not in self._runs:
            return None
        events = self._run_events.setdefault(run_id, [])
        seq = len(events) + 1
        safe_payload = sanitize_for_persistence(payload or {})
        doc = {
            "_id": _doc_id(run_id, seq),
            "run_id": run_id,
            "session_id": session_id,
            "seq": seq,
            "timestamp": _now(),
            "event_type": event_type,
            "payload": safe_payload,
            "safe_summary": safe_event_summary(event_type, safe_payload),
        }
        events.append(doc)
        await self._apply_run_event_update(run_id, event_type, safe_payload, seq)
        await self._record_audit_from_run_event(
            session_id=session_id,
            run_id=run_id,
            event_type=event_type,
            payload=safe_payload,
        )
        return seq

    async def load_run_events_after(
        self, run_id: str, after_seq: int = 0
    ) -> list[dict[str, Any]]:
        return [
            dict(event)
            for event in self._run_events.get(run_id, [])
            if int(event.get("seq") or 0) > int(after_seq or 0)
        ]

    async def _apply_run_event_update(
        self, run_id: str, event_type: str, payload: dict[str, Any], seq: int
    ) -> None:
        run = self._runs.get(run_id)
        if not run:
            return
        now = _now()
        update: dict[str, Any] = {
            "last_event_seq": seq,
            "updated_at": now,
        }
        if status := run_status_from_event(event_type, payload):
            update["status"] = status
            if status == "running" and run.get("started_at") is None:
                update["started_at"] = now
            if status in {"succeeded", "failed", "cancelled", "interrupted"}:
                update["completed_at"] = now
        provider_fields = provider_metadata_from_event(event_type, payload)
        provider_metadata = dict(run.get("provider_metadata") or {})
        if discovery := _dataset_discovery_from_event(event_type, payload):
            update["dataset_discovery"] = discovery
            provider_metadata["dataset_discovery"] = discovery
            update["provider_metadata"] = provider_metadata
        if recommendation := _training_recommendation_from_event(event_type, payload):
            update["training_recommendation"] = recommendation
            provider_metadata["training_recommendation"] = recommendation
            update["provider_metadata"] = provider_metadata
        if provider_fields:
            if provider := provider_fields.pop("provider", None):
                update["provider"] = provider
                provider_metadata["provider"] = provider
            for key, value in provider_fields.items():
                update[key] = value
                provider_metadata[key] = value
            provider_metadata["last_checked_at"] = now.isoformat()
            update["provider_metadata"] = provider_metadata
        if event_type == "approval_required":
            tools = payload.get("tools") if isinstance(payload, dict) else None
            first = tools[0] if isinstance(tools, list) and tools else {}
            if isinstance(first, dict):
                update["approval_id"] = first.get("approval_id") or first.get(
                    "tool_call_id"
                )
                update["active_tool"] = first.get("tool")
        if event_type in {"error", "stream_error"}:
            update["error_summary"] = str(payload.get("error") or "")[:500]
        if event_type in {"assistant_message", "turn_complete"}:
            summary = safe_event_summary(event_type, payload)
            if summary:
                update["result_summary"] = summary
        run.update(update)
        await self._apply_usage_event_update(run_id, event_type, payload, update)
        await self._apply_evaluation_event_update(run_id, event_type, payload)

    async def _apply_evaluation_event_update(
        self, run_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        if event_type not in {"tool_output", "assistant_message", "turn_complete"}:
            return
        run = self._runs.get(run_id)
        if not run:
            return
        text = ""
        for key in ("output", "content", "final_response", "formatted"):
            value = payload.get(key)
            if isinstance(value, str):
                text = value
                break
        context = evaluation_context_from_liga_output(
            session_id=str(run.get("session_id") or ""),
            run_id=run_id,
            output=text,
            fallback_provider=str(run.get("provider") or ""),
        )
        if not context:
            return
        evaluation = build_post_training_evaluation(context)
        await self.upsert_evaluation(evaluation)

    async def _apply_usage_event_update(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
        run_update: dict[str, Any],
    ) -> None:
        run = self._runs.get(run_id)
        session_id = str(
            (run or {}).get("session_id") or payload.get("session_id") or ""
        )
        if not session_id:
            return
        if event_type == "approval_required":
            tools = payload.get("tools") if isinstance(payload, dict) else None
            if isinstance(tools, list):
                for tool_payload in tools:
                    if isinstance(tool_payload, dict):
                        usage_id, entry = usage_from_approval_tool(
                            session_id=session_id,
                            run_id=run_id,
                            tool_payload=tool_payload,
                            event_payload=payload,
                        )
                        await self.upsert_usage_entry(usage_id, entry)
            return
        if recommendation := _training_recommendation_from_event(event_type, payload):
            usage_update = usage_from_training_recommendation(
                session_id=session_id,
                run_id=run_id,
                recommendation=recommendation,
            )
            if usage_update:
                usage_id, entry = usage_update
                await self.upsert_usage_entry(usage_id, entry)
            return
        existing = await self.list_usage_entries(session_id=session_id, run_id=run_id)
        if event_type == "tool_state_change":
            usage_update = usage_from_tool_state(
                session_id=session_id,
                run_id=run_id,
                payload=payload,
                existing=existing,
            )
            if usage_update:
                usage_id, fields = usage_update
                await self.upsert_usage_entry(usage_id, fields)
            return
        status = run_update.get("status")
        if status in {"succeeded", "failed", "cancelled", "interrupted"}:
            fields = usage_from_run_terminal(
                run_id=run_id,
                status=str(status),
                error_summary=run_update.get("error_summary"),
            )
            for entry in existing:
                await self.upsert_usage_entry(str(entry["usage_id"]), fields)

    async def _record_audit_from_run_event(
        self,
        *,
        session_id: str,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        for event in event_from_run_event(
            session_id=session_id,
            run_id=run_id,
            event_type=event_type,
            payload=payload,
        ):
            await self.record_audit_event(event)

    async def upsert_response_rows(self, *_: Any, **__: Any) -> None:
        return None

    async def list_response_rows(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {
            "rows": [],
            "page": 1,
            "page_size": 50,
            "total_rows": 0,
            "total_pages": 0,
            "has_next": False,
            "has_previous": False,
        }

    async def get_response_summary(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {
            "total_responses": 0,
            "visible_count": 0,
            "batch_number": 1,
            "has_rows": False,
            "button_enabled": True,
            "durable": False,
            "store_type": "memory",
        }


class MongoSessionStore(NoopSessionStore):
    """MongoDB-backed session store."""

    enabled = True

    def __init__(self, uri: str, db_name: str) -> None:
        super().__init__()
        self.uri = uri
        self.db_name = db_name
        self.enabled = False
        self.client: AsyncMongoClient | None = None
        self.db = None

    async def init(self) -> None:
        try:
            self.client = AsyncMongoClient(self.uri, serverSelectionTimeoutMS=3000)
            self.db = self.client[self.db_name]
            await self.client.admin.command("ping")
            await self._create_indexes()
            self.enabled = True
            logger.info("Mongo session persistence enabled (db=%s)", self.db_name)
        except Exception as e:
            logger.warning("Mongo session persistence disabled: %s", e)
            self.enabled = False
            if self.client is not None:
                await self.client.close()
            self.client = None
            self.db = None

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
        self.client = None
        self.db = None

    async def _create_indexes(self) -> None:
        if self.db is None:
            return
        await self.db.sessions.create_index(
            [("user_id", 1), ("visibility", 1), ("updated_at", -1)]
        )
        await self.db.sessions.create_index(
            [("visibility", 1), ("status", 1), ("last_active_at", -1)]
        )
        await self.db.session_messages.create_index(
            [("session_id", 1), ("idx", 1)], unique=True
        )
        await self.db.session_events.create_index(
            [("session_id", 1), ("seq", 1)], unique=True
        )
        await self.db.session_trace_messages.create_index(
            [("session_id", 1), ("seq", 1)], unique=True
        )
        await self.db.session_trace_messages.create_index([("created_at", -1)])
        await self.db.runs.create_index([("session_id", 1), ("updated_at", -1)])
        await self.db.runs.create_index([("status", 1), ("updated_at", -1)])
        await self.db.run_events.create_index([("run_id", 1), ("seq", 1)], unique=True)
        await self.db.usage_entries.create_index([("usage_id", 1)], unique=True)
        await self.db.usage_entries.create_index(
            [("session_id", 1), ("updated_at", -1)]
        )
        await self.db.usage_entries.create_index([("run_id", 1), ("updated_at", -1)])
        await self.db.usage_entries.create_index([("provider", 1), ("updated_at", -1)])
        await self.db.audit_events.create_index([("audit_id", 1)], unique=True)
        await self.db.audit_events.create_index([("session_id", 1), ("timestamp", -1)])
        await self.db.audit_events.create_index([("run_id", 1), ("timestamp", -1)])
        await self.db.audit_events.create_index([("provider", 1), ("timestamp", -1)])
        await self.db.audit_events.create_index([("category", 1), ("timestamp", -1)])
        await self.db.audit_events.create_index([("severity", 1), ("timestamp", -1)])
        await self.db.evaluations.create_index([("evaluation_id", 1)], unique=True)
        await self.db.evaluations.create_index([("session_id", 1), ("updated_at", -1)])
        await self.db.evaluations.create_index([("run_id", 1), ("updated_at", -1)])
        await self.db.evaluations.create_index([("provider", 1), ("updated_at", -1)])
        await self.db.evaluations.create_index([("status", 1), ("updated_at", -1)])
        await self.db.pro_users.create_index([("first_seen_pro_at", -1)])
        await self.db.response_rows.create_index(
            [("user_id", 1), ("actual_sequence_number", -1)]
        )
        await self.db.response_rows.create_index([("platform", 1), ("progress", 1)])
        await self.db.response_rows.create_index([("session_id", 1)])
        await self.db.response_rows.create_index([("job_id", 1)])

    def _ready(self) -> bool:
        return bool(self.enabled and self.db is not None)

    async def upsert_session(
        self,
        *,
        session_id: str,
        user_id: str,
        model: str,
        title: str | None = None,
        surface: str = "frontend",
        created_at: datetime | None = None,
        runtime_state: str = "idle",
        status: str = "active",
        message_count: int = 0,
        turn_count: int = 0,
        pending_approval: list[dict[str, Any]] | None = None,
        claude_counted: bool = False,
        notification_destinations: list[str] | None = None,
        auto_approval_enabled: bool = False,
        auto_approval_cost_cap_usd: float | None = None,
        auto_approval_estimated_spend_usd: float = 0.0,
        cloud_provider: str = "hf-jobs",
        training_goal: str = "agent-decide",
        output_policy: str = "cloud-and-hf-hub",
        uploaded_datasets: list[dict[str, Any]] | None = None,
        latest_dataset_discovery: dict[str, Any] | None = None,
        latest_training_recommendation: dict[str, Any] | None = None,
        latest_training_preflight: dict[str, Any] | None = None,
        current_run_id: str | None = None,
    ) -> None:
        if not self._ready():
            return
        now = _now()
        await self.db.sessions.update_one(
            {"_id": session_id},
            {
                "$setOnInsert": {
                    "_id": session_id,
                    "session_id": session_id,
                    "user_id": user_id,
                    "surface": surface,
                    "created_at": created_at or now,
                    "schema_version": SCHEMA_VERSION,
                    "visibility": "live",
                },
                "$set": {
                    "title": title,
                    "model": model,
                    "status": status,
                    "runtime_state": runtime_state,
                    "updated_at": now,
                    "last_active_at": now,
                    "message_count": message_count,
                    "turn_count": turn_count,
                    "pending_approval": sanitize_for_persistence(
                        pending_approval or []
                    ),
                    "claude_counted": claude_counted,
                    "notification_destinations": notification_destinations or [],
                    "auto_approval_enabled": auto_approval_enabled,
                    "auto_approval_cost_cap_usd": auto_approval_cost_cap_usd,
                    "auto_approval_estimated_spend_usd": auto_approval_estimated_spend_usd,
                    "cloud_provider": cloud_provider,
                    "training_goal": training_goal,
                    "output_policy": output_policy,
                    "uploaded_datasets": sanitize_for_persistence(
                        uploaded_datasets or []
                    ),
                    "latest_dataset_discovery": sanitize_for_persistence(
                        latest_dataset_discovery or {}
                    ),
                    "latest_training_recommendation": sanitize_for_persistence(
                        latest_training_recommendation or {}
                    ),
                    "latest_training_preflight": sanitize_for_persistence(
                        latest_training_preflight or {}
                    ),
                    "current_run_id": current_run_id,
                },
            },
            upsert=True,
        )

    async def save_snapshot(
        self,
        *,
        session_id: str,
        user_id: str,
        model: str,
        messages: list[dict[str, Any]],
        title: str | None = None,
        runtime_state: str = "idle",
        status: str = "active",
        turn_count: int = 0,
        pending_approval: list[dict[str, Any]] | None = None,
        claude_counted: bool = False,
        created_at: datetime | None = None,
        notification_destinations: list[str] | None = None,
        auto_approval_enabled: bool = False,
        auto_approval_cost_cap_usd: float | None = None,
        auto_approval_estimated_spend_usd: float = 0.0,
        cloud_provider: str = "hf-jobs",
        training_goal: str = "agent-decide",
        output_policy: str = "cloud-and-hf-hub",
        uploaded_datasets: list[dict[str, Any]] | None = None,
        latest_dataset_discovery: dict[str, Any] | None = None,
        latest_training_recommendation: dict[str, Any] | None = None,
        latest_training_preflight: dict[str, Any] | None = None,
        current_run_id: str | None = None,
    ) -> None:
        if not self._ready():
            return
        now = _now()
        await self.upsert_session(
            session_id=session_id,
            user_id=user_id,
            model=model,
            title=title,
            created_at=created_at,
            runtime_state=runtime_state,
            status=status,
            message_count=len(messages),
            turn_count=turn_count,
            pending_approval=pending_approval,
            claude_counted=claude_counted,
            notification_destinations=notification_destinations,
            auto_approval_enabled=auto_approval_enabled,
            auto_approval_cost_cap_usd=auto_approval_cost_cap_usd,
            auto_approval_estimated_spend_usd=auto_approval_estimated_spend_usd,
            cloud_provider=cloud_provider,
            training_goal=training_goal,
            output_policy=output_policy,
            uploaded_datasets=uploaded_datasets,
            latest_dataset_discovery=latest_dataset_discovery,
            latest_training_recommendation=latest_training_recommendation,
            latest_training_preflight=latest_training_preflight,
            current_run_id=current_run_id,
        )
        ops: list[Any] = []
        for idx, raw in enumerate(messages):
            ops.append(
                UpdateOne(
                    {"_id": _doc_id(session_id, idx)},
                    {
                        "$set": {
                            "session_id": session_id,
                            "idx": idx,
                            "message": _safe_message_doc(raw),
                            "updated_at": now,
                        },
                        "$setOnInsert": {"created_at": now},
                    },
                    upsert=True,
                )
            )
        ops.append(
            DeleteMany({"session_id": session_id, "idx": {"$gte": len(messages)}})
        )
        try:
            if ops:
                await self.db.session_messages.bulk_write(ops, ordered=False)
        except PyMongoError as e:
            logger.warning("Failed to persist session %s snapshot: %s", session_id, e)

    async def load_session(
        self, session_id: str, *, include_deleted: bool = False
    ) -> dict[str, Any] | None:
        if not self._ready():
            return None
        meta = await self.db.sessions.find_one({"_id": session_id})
        if not meta:
            return None
        if meta.get("visibility") == "deleted" and not include_deleted:
            return None
        cursor = self.db.session_messages.find({"session_id": session_id}).sort(
            "idx", 1
        )
        messages = [row.get("message") async for row in cursor]
        return {"metadata": meta, "messages": messages}

    async def list_sessions(
        self, user_id: str, *, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        if not self._ready():
            return []
        query: dict[str, Any] = {"user_id": user_id}
        if user_id == "dev":
            query = {}
        if not include_deleted:
            query["visibility"] = {"$ne": "deleted"}
        cursor = self.db.sessions.find(query).sort("updated_at", -1)
        return [row async for row in cursor]

    async def soft_delete_session(self, session_id: str) -> None:
        if not self._ready():
            return
        await self.db.sessions.update_one(
            {"_id": session_id},
            {
                "$set": {
                    "visibility": "deleted",
                    "runtime_state": "idle",
                    "updated_at": _now(),
                }
            },
        )

    async def update_session_fields(self, session_id: str, **fields: Any) -> None:
        if not self._ready() or not fields:
            return
        fields = sanitize_for_persistence(fields)
        fields["updated_at"] = _now()
        await self.db.sessions.update_one({"_id": session_id}, {"$set": fields})

    async def _next_seq(self, counter_id: str) -> int:
        doc = await self.db.counters.find_one_and_update(
            {"_id": counter_id},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return int(doc["seq"])

    async def append_event(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any] | None,
        run_id: str | None = None,
    ) -> int | None:
        if not self._ready():
            return None
        try:
            resolved_run_id = await _resolve_active_run_id(
                self,
                session_id=session_id,
                run_id=run_id,
            )
            seq = await self._next_seq(f"event:{session_id}")
            safe_data = sanitize_for_persistence(data or {})
            await self.db.session_events.insert_one(
                {
                    "_id": _doc_id(session_id, seq),
                    "session_id": session_id,
                    "seq": seq,
                    "event_type": event_type,
                    "data": safe_data,
                    "created_at": _now(),
                    "run_id": resolved_run_id,
                }
            )
            if resolved_run_id:
                return await self.append_run_event(
                    run_id=resolved_run_id,
                    session_id=session_id,
                    event_type=event_type,
                    payload=safe_data,
                )
            return seq
        except PyMongoError as e:
            logger.debug("Failed to append event for %s: %s", session_id, e)
            return None

    async def load_events_after(
        self, session_id: str, after_seq: int = 0
    ) -> list[dict[str, Any]]:
        if not self._ready():
            return []
        cursor = self.db.session_events.find(
            {"session_id": session_id, "seq": {"$gt": int(after_seq or 0)}}
        ).sort("seq", 1)
        return [row async for row in cursor]

    async def create_run(
        self,
        *,
        session_id: str,
        provider: str = "none",
        request_id: str | None = None,
        status: str = "queued",
    ) -> dict[str, Any]:
        if not self._ready():
            return await super().create_run(
                session_id=session_id,
                provider=provider,
                request_id=request_id,
                status=status,
            )
        now = _now()
        run_id = str(uuid.uuid4())
        run = {
            "_id": run_id,
            "run_id": run_id,
            "session_id": session_id,
            "status": status,
            "provider": provider or "none",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
            "last_event_seq": 0,
            "active_tool": None,
            "active_provider_job_id": None,
            "approval_id": None,
            "error_summary": None,
            "result_summary": None,
            "request_id": request_id,
            "provider_metadata": {},
            "schema_version": SCHEMA_VERSION,
        }
        await self.db.runs.insert_one(run)
        await self.append_run_event(
            run_id=run_id,
            session_id=session_id,
            event_type="run_created",
            payload={"request_id": request_id, "provider": provider},
        )
        return run

    async def update_run(self, run_id: str, **fields: Any) -> dict[str, Any] | None:
        if not self._ready():
            return await super().update_run(run_id, **fields)
        fields = sanitize_for_persistence(
            {k: v for k, v in fields.items() if v is not None}
        )
        fields["updated_at"] = _now()
        doc = await self.db.runs.find_one_and_update(
            {"_id": run_id},
            {"$set": fields},
            return_document=ReturnDocument.AFTER,
        )
        return doc

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        if not self._ready():
            return await super().get_run(run_id)
        return await self.db.runs.find_one({"_id": run_id})

    async def record_training_preflight(
        self,
        *,
        session_id: str,
        preflight: dict[str, Any],
        run_id: str | None = None,
        include_started_audit: bool = True,
    ) -> dict[str, Any]:
        if not self._ready():
            return await super().record_training_preflight(
                session_id=session_id,
                preflight=preflight,
                run_id=run_id,
                include_started_audit=include_started_audit,
            )
        clean = sanitize_for_persistence(preflight)
        if not isinstance(clean, dict):
            clean = {}
        clean["session_id"] = session_id
        if run_id:
            clean["run_id"] = run_id
        now = _now()
        await self.db.sessions.update_one(
            {"_id": session_id},
            {"$set": {"latest_training_preflight": clean, "updated_at": now}},
            upsert=False,
        )
        if run_id:
            existing_run = await self.get_run(run_id)
            provider_metadata = dict(
                (existing_run or {}).get("provider_metadata") or {}
            )
            provider_metadata["training_preflight"] = clean
            await self.update_run(
                run_id,
                training_preflight=clean,
                provider_metadata=provider_metadata,
            )
        for event in training_preflight_audit_events(
            clean, include_started=include_started_audit
        ):
            await self.record_audit_event(event)
        if run_id:
            usage_update = usage_from_training_preflight(
                session_id=session_id,
                run_id=run_id,
                preflight=clean,
            )
            if usage_update:
                usage_id, entry = usage_update
                await self.upsert_usage_entry(usage_id, entry)
        return dict(clean)

    async def get_latest_training_preflight(
        self, session_id: str
    ) -> dict[str, Any] | None:
        if not self._ready():
            return await super().get_latest_training_preflight(session_id)
        session = await self.db.sessions.find_one({"_id": session_id})
        preflight = (
            session.get("latest_training_preflight")
            if isinstance(session, dict)
            else None
        )
        return dict(preflight) if isinstance(preflight, dict) and preflight else None

    async def get_run_training_preflight(
        self, session_id: str, run_id: str
    ) -> dict[str, Any] | None:
        if not self._ready():
            return await super().get_run_training_preflight(session_id, run_id)
        run = await self.get_run(run_id)
        if not run or str(run.get("session_id")) != session_id:
            return None
        preflight = run.get("training_preflight")
        if isinstance(preflight, dict) and preflight:
            return dict(preflight)
        provider_metadata = run.get("provider_metadata")
        if isinstance(provider_metadata, dict) and isinstance(
            provider_metadata.get("training_preflight"), dict
        ):
            return dict(provider_metadata["training_preflight"])
        return None

    async def list_runs(self, session_id: str) -> list[dict[str, Any]]:
        if not self._ready():
            return await super().list_runs(session_id)
        cursor = self.db.runs.find({"session_id": session_id}).sort("updated_at", -1)
        return [row async for row in cursor]

    async def upsert_evaluation(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        if not self._ready():
            return await super().upsert_evaluation(evaluation)
        now = _now()
        clean = sanitize_for_persistence(
            {key: value for key, value in evaluation.items() if value is not None}
        )
        evaluation_id = str(
            clean.get("evaluation_id") or clean.get("_id") or uuid.uuid4()
        )
        clean["_id"] = evaluation_id
        clean["evaluation_id"] = evaluation_id
        clean["updated_at"] = now
        doc = await self.db.evaluations.find_one_and_update(
            {"evaluation_id": evaluation_id},
            {
                "$setOnInsert": {
                    "_id": evaluation_id,
                    "evaluation_id": evaluation_id,
                    "created_at": clean.get("created_at") or now,
                    "schema_version": SCHEMA_VERSION,
                },
                "$set": clean,
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        current = doc or clean
        run_id = str(current.get("run_id") or "")
        if run_id:
            scores = (
                current.get("scores") if isinstance(current.get("scores"), dict) else {}
            )
            await self.update_run(
                run_id,
                evaluation_status=current.get("status"),
                evaluation_score=scores.get("overall_score"),
                evaluation_id=evaluation_id,
            )
        await self._record_evaluation_audit_events(current)
        return current

    async def list_evaluations(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        provider: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not self._ready():
            return await super().list_evaluations(
                session_id=session_id,
                run_id=run_id,
                provider=provider,
                status=status,
                limit=limit,
            )
        query: dict[str, Any] = {}
        if session_id:
            query["session_id"] = session_id
        if run_id:
            query["run_id"] = run_id
        if provider:
            query["provider"] = provider
        if status:
            query["status"] = status
        cursor = (
            self.db.evaluations.find(query)
            .sort("updated_at", -1)
            .limit(max(1, min(int(limit or 100), 500)))
        )
        return [row async for row in cursor]

    async def upsert_usage_entry(
        self, usage_id: str, fields: dict[str, Any]
    ) -> dict[str, Any]:
        if not self._ready():
            return await super().upsert_usage_entry(usage_id, fields)
        now = _now()
        cleaned = sanitize_for_persistence(
            {key: value for key, value in fields.items() if value is not None}
        )
        cleaned.pop("_id", None)
        cleaned["usage_id"] = usage_id
        cleaned["updated_at"] = now
        doc = await self.db.usage_entries.find_one_and_update(
            {"usage_id": usage_id},
            {
                "$setOnInsert": {
                    "_id": usage_id,
                    "usage_id": usage_id,
                    "created_at": cleaned.get("created_at") or now,
                    "schema_version": SCHEMA_VERSION,
                },
                "$set": cleaned,
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return doc or cleaned

    async def list_usage_entries(
        self,
        *,
        provider: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not self._ready():
            return await super().list_usage_entries(
                provider=provider,
                session_id=session_id,
                run_id=run_id,
                status=status,
                limit=limit,
            )
        query: dict[str, Any] = {}
        if provider:
            query["provider"] = provider
        if session_id:
            query["session_id"] = session_id
        if run_id:
            query["run_id"] = run_id
        if status:
            query["status"] = status
        cursor = (
            self.db.usage_entries.find(query)
            .sort("updated_at", -1)
            .limit(max(1, min(int(limit or 100), 500)))
        )
        return [row async for row in cursor]

    async def usage_summary(self, **filters: Any) -> dict[str, Any]:
        entries = await self.list_usage_entries(**filters)
        return {**summarize_usage(entries), "entries": entries}

    async def record_audit_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        if not self._ready():
            return await super().record_audit_event(event)
        if not audit_timeline_enabled():
            return None
        audit_id = str(event.get("audit_id") or event.get("_id") or "")
        if not audit_id:
            return None
        cleaned = sanitize_for_persistence(
            {key: value for key, value in event.items() if value is not None}
        )
        cleaned["_id"] = audit_id
        cleaned["audit_id"] = audit_id
        try:
            result = await self.db.audit_events.find_one_and_update(
                {"audit_id": audit_id},
                {
                    "$setOnInsert": {
                        **cleaned,
                        "created_at": _now(),
                        "schema_version": SCHEMA_VERSION,
                    }
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            logger.info(
                "audit_event audit_id=%s session_id=%s run_id=%s category=%s "
                "event_type=%s provider=%s severity=%s status=%s",
                cleaned.get("audit_id"),
                cleaned.get("session_id"),
                cleaned.get("run_id"),
                cleaned.get("category"),
                cleaned.get("event_type"),
                cleaned.get("provider"),
                cleaned.get("severity"),
                cleaned.get("status"),
            )
            return result or cleaned
        except DuplicateKeyError:
            return await self.db.audit_events.find_one({"audit_id": audit_id})
        except PyMongoError as e:
            logger.debug("Failed to record audit event %s: %s", audit_id, e)
            return None

    async def list_audit_events(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        provider: str | None = None,
        category: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not self._ready():
            return await super().list_audit_events(
                session_id=session_id,
                run_id=run_id,
                provider=provider,
                category=category,
                severity=severity,
                status=status,
                since=since,
                until=until,
                limit=limit,
            )
        query: dict[str, Any] = {}
        if session_id:
            query["session_id"] = session_id
        if run_id:
            query["run_id"] = run_id
        if provider:
            query["provider"] = provider
        if category:
            query["category"] = category
        if severity:
            query["severity"] = severity
        if status:
            query["status"] = status
        timestamp_query: dict[str, Any] = {}
        if since_dt := _parse_dt(since):
            timestamp_query["$gte"] = since_dt
        if until_dt := _parse_dt(until):
            timestamp_query["$lte"] = until_dt
        if timestamp_query:
            query["timestamp"] = timestamp_query
        cursor = (
            self.db.audit_events.find(query)
            .sort("timestamp", 1)
            .limit(max(1, min(int(limit or 100), 500)))
        )
        return [row async for row in cursor]

    async def audit_summary(self, **filters: Any) -> dict[str, Any]:
        events = await self.list_audit_events(**filters)
        return {**summarize_audit_events(events), "events": events}

    async def append_run_event(
        self,
        *,
        run_id: str,
        session_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> int | None:
        if not self._ready():
            return await super().append_run_event(
                run_id=run_id,
                session_id=session_id,
                event_type=event_type,
                payload=payload,
            )
        try:
            seq = await self._next_seq(f"run_event:{run_id}")
            safe_payload = sanitize_for_persistence(payload or {})
            doc = {
                "_id": _doc_id(run_id, seq),
                "run_id": run_id,
                "session_id": session_id,
                "seq": seq,
                "timestamp": _now(),
                "event_type": event_type,
                "payload": safe_payload,
                "safe_summary": safe_event_summary(event_type, safe_payload),
                "schema_version": SCHEMA_VERSION,
            }
            await self.db.run_events.insert_one(doc)
            await self._apply_persisted_run_event_update(
                run_id, event_type, safe_payload, seq
            )
            await self._record_audit_from_run_event(
                session_id=session_id,
                run_id=run_id,
                event_type=event_type,
                payload=safe_payload,
            )
            return seq
        except PyMongoError as e:
            logger.debug("Failed to append run event for %s: %s", run_id, e)
            return None

    async def _apply_persisted_run_event_update(
        self, run_id: str, event_type: str, payload: dict[str, Any], seq: int
    ) -> None:
        now = _now()
        update: dict[str, Any] = {"last_event_seq": seq, "updated_at": now}
        run = await self.get_run(run_id)
        if status := run_status_from_event(event_type, payload):
            update["status"] = status
            if status == "running" and not (run or {}).get("started_at"):
                update["started_at"] = now
            if status in {"succeeded", "failed", "cancelled", "interrupted"}:
                update["completed_at"] = now
        provider_fields = provider_metadata_from_event(event_type, payload)
        provider_metadata = dict((run or {}).get("provider_metadata") or {})
        if discovery := _dataset_discovery_from_event(event_type, payload):
            update["dataset_discovery"] = discovery
            provider_metadata["dataset_discovery"] = discovery
            update["provider_metadata"] = provider_metadata
        if recommendation := _training_recommendation_from_event(event_type, payload):
            update["training_recommendation"] = recommendation
            provider_metadata["training_recommendation"] = recommendation
            update["provider_metadata"] = provider_metadata
        if provider_fields:
            if provider := provider_fields.pop("provider", None):
                update["provider"] = provider
                provider_metadata["provider"] = provider
            for key, value in provider_fields.items():
                update[key] = value
                provider_metadata[key] = value
            provider_metadata["last_checked_at"] = now.isoformat()
            update["provider_metadata"] = provider_metadata
        if event_type == "approval_required":
            tools = payload.get("tools") if isinstance(payload, dict) else None
            first = tools[0] if isinstance(tools, list) and tools else {}
            if isinstance(first, dict):
                update["approval_id"] = first.get("approval_id") or first.get(
                    "tool_call_id"
                )
                update["active_tool"] = first.get("tool")
        if event_type in {"error", "stream_error"}:
            update["error_summary"] = str(payload.get("error") or "")[:500]
        if event_type in {"assistant_message", "turn_complete"}:
            summary = safe_event_summary(event_type, payload)
            if summary:
                update["result_summary"] = summary
        await self.update_run(run_id, **update)
        await self._apply_persisted_usage_event_update(
            run_id, event_type, payload, update
        )
        await self._apply_persisted_evaluation_event_update(run_id, event_type, payload)

    async def _apply_persisted_evaluation_event_update(
        self, run_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        if event_type not in {"tool_output", "assistant_message", "turn_complete"}:
            return
        run = await self.get_run(run_id)
        if not run:
            return
        text = ""
        for key in ("output", "content", "final_response", "formatted"):
            value = payload.get(key)
            if isinstance(value, str):
                text = value
                break
        context = evaluation_context_from_liga_output(
            session_id=str(run.get("session_id") or ""),
            run_id=run_id,
            output=text,
            fallback_provider=str(run.get("provider") or ""),
        )
        if not context:
            return
        await self.upsert_evaluation(build_post_training_evaluation(context))

    async def _apply_persisted_usage_event_update(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
        run_update: dict[str, Any],
    ) -> None:
        run = await self.get_run(run_id)
        session_id = str(
            (run or {}).get("session_id") or payload.get("session_id") or ""
        )
        if not session_id:
            return
        if event_type == "approval_required":
            tools = payload.get("tools") if isinstance(payload, dict) else None
            if isinstance(tools, list):
                for tool_payload in tools:
                    if isinstance(tool_payload, dict):
                        usage_id, entry = usage_from_approval_tool(
                            session_id=session_id,
                            run_id=run_id,
                            tool_payload=tool_payload,
                            event_payload=payload,
                        )
                        await self.upsert_usage_entry(usage_id, entry)
            return
        if recommendation := _training_recommendation_from_event(event_type, payload):
            usage_update = usage_from_training_recommendation(
                session_id=session_id,
                run_id=run_id,
                recommendation=recommendation,
            )
            if usage_update:
                usage_id, entry = usage_update
                await self.upsert_usage_entry(usage_id, entry)
            return
        existing = await self.list_usage_entries(session_id=session_id, run_id=run_id)
        if event_type == "tool_state_change":
            usage_update = usage_from_tool_state(
                session_id=session_id,
                run_id=run_id,
                payload=payload,
                existing=existing,
            )
            if usage_update:
                usage_id, fields = usage_update
                await self.upsert_usage_entry(usage_id, fields)
            return
        status = run_update.get("status")
        if status in {"succeeded", "failed", "cancelled", "interrupted"}:
            fields = usage_from_run_terminal(
                run_id=run_id,
                status=str(status),
                error_summary=run_update.get("error_summary"),
            )
            for entry in existing:
                await self.upsert_usage_entry(str(entry["usage_id"]), fields)

    async def load_run_events_after(
        self, run_id: str, after_seq: int = 0
    ) -> list[dict[str, Any]]:
        if not self._ready():
            return await super().load_run_events_after(run_id, after_seq)
        cursor = self.db.run_events.find(
            {"run_id": run_id, "seq": {"$gt": int(after_seq or 0)}}
        ).sort("seq", 1)
        return [row async for row in cursor]

    async def append_trace_message(
        self, session_id: str, message: dict[str, Any], source: str = "message"
    ) -> int | None:
        if not self._ready():
            return None
        try:
            seq = await self._next_seq(f"trace:{session_id}")
            await self.db.session_trace_messages.insert_one(
                {
                    "_id": _doc_id(session_id, seq),
                    "session_id": session_id,
                    "seq": seq,
                    "role": message.get("role"),
                    "message": _safe_message_doc(message),
                    "source": source,
                    "created_at": _now(),
                }
            )
            return seq
        except PyMongoError as e:
            logger.debug("Failed to append trace message for %s: %s", session_id, e)
            return None

    async def load_trace_messages(self, session_id: str) -> list[dict[str, Any]]:
        if not self._ready():
            return []
        cursor = self.db.session_trace_messages.find({"session_id": session_id}).sort(
            "seq", 1
        )
        return [row async for row in cursor]

    async def get_quota(self, user_id: str, day: str) -> int | None:
        if not self._ready():
            return None
        doc = await self.db.claude_quotas.find_one({"_id": f"{user_id}:{day}"})
        return int(doc.get("count", 0)) if doc else 0

    async def try_increment_quota(self, user_id: str, day: str, cap: int) -> int | None:
        if not self._ready():
            return None
        key = f"{user_id}:{day}"
        now = _now()
        try:
            await self.db.claude_quotas.insert_one(
                {
                    "_id": key,
                    "user_id": user_id,
                    "day": day,
                    "count": 1,
                    "updated_at": now,
                }
            )
            return 1
        except DuplicateKeyError:
            pass
        doc = await self.db.claude_quotas.find_one_and_update(
            {"_id": key, "count": {"$lt": cap}},
            {"$inc": {"count": 1}, "$set": {"updated_at": now}},
            return_document=ReturnDocument.AFTER,
        )
        return int(doc["count"]) if doc else None

    async def refund_quota(self, user_id: str, day: str) -> None:
        if not self._ready():
            return
        await self.db.claude_quotas.update_one(
            {"_id": f"{user_id}:{day}", "count": {"$gt": 0}},
            {"$inc": {"count": -1}, "$set": {"updated_at": _now()}},
        )

    async def mark_pro_seen(
        self, user_id: str, *, is_pro: bool
    ) -> dict[str, Any] | None:
        """Track per-user Pro state and detect free→Pro conversions.

        Returns ``{"converted": True, "first_seen_at": ..."}`` exactly once
        per user — the first time we see them as Pro after having recorded
        them as non-Pro at least once. Otherwise returns ``None``.

        Storing ``ever_non_pro`` lets us distinguish "user joined as Pro"
        (no conversion) from "user upgraded" (conversion). The atomic
        ``find_one_and_update`` on a guarded filter makes the conversion
        emit at-most-once even under concurrent requests.
        """
        if not self._ready() or not user_id:
            return None
        now = _now()
        set_fields: dict[str, Any] = {"last_seen_at": now, "is_pro": bool(is_pro)}
        if not is_pro:
            set_fields["ever_non_pro"] = True
        try:
            await self.db.pro_users.update_one(
                {"_id": user_id},
                {
                    "$setOnInsert": {"_id": user_id, "first_seen_at": now},
                    "$set": set_fields,
                },
                upsert=True,
            )
        except PyMongoError as e:
            logger.debug("mark_pro_seen upsert failed for %s: %s", user_id, e)
            return None

        if not is_pro:
            return None

        try:
            doc = await self.db.pro_users.find_one_and_update(
                {
                    "_id": user_id,
                    "ever_non_pro": True,
                    "first_seen_pro_at": {"$exists": False},
                },
                {"$set": {"first_seen_pro_at": now}},
                return_document=ReturnDocument.AFTER,
            )
        except PyMongoError as e:
            logger.debug("mark_pro_seen conversion check failed for %s: %s", user_id, e)
            return None

        if not doc:
            return None
        return {
            "converted": True,
            "first_seen_at": (doc.get("first_seen_at") or now).isoformat(),
        }

    def _response_query(
        self,
        *,
        user_id: str,
        platform: str | None = None,
        progress: str | None = None,
        model: str | None = None,
        session_id: str | None = None,
        job_id: str | None = None,
        q: str | None = None,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {}
        if user_id != "dev":
            query["user_id"] = user_id

        def contains(field: str, value: str | None) -> None:
            if value:
                query[field] = {"$regex": re.escape(value), "$options": "i"}

        contains("platform", platform)
        contains("progress", progress)
        contains("model_name", model)
        contains("session_id", session_id)
        contains("job_id", job_id)
        if q:
            regex = {"$regex": re.escape(q), "$options": "i"}
            query["$or"] = [
                {"id": regex},
                {"session_id": regex},
                {"short_session_id": regex},
                {"session_title": regex},
                {"model_name": regex},
                {"platform": regex},
                {"run_type": regex},
                {"result_storage": regex},
                {"progress": regex},
                {"job_id": regex},
                {"final_artifact_or_result": regex},
                {"error": regex},
            ]
        return query

    def _mongo_safe_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            safe: dict[str, Any] = {}
            for key, item in value.items():
                text_key = str(key).replace(".", "_")
                if text_key.startswith("$"):
                    text_key = f"_{text_key[1:]}"
                safe[text_key] = self._mongo_safe_value(item)
            return safe
        if isinstance(value, list):
            return [self._mongo_safe_value(item) for item in value]
        return value

    def _normalize_response_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        progress = str(normalized.get("progress") or "").lower()
        state = str(
            (normalized.get("provider_metadata") or {}).get("state") or ""
        ).lower()
        status = state if progress in {"", "unknown"} else progress
        if status.startswith("job_state_"):
            status = status.removeprefix("job_state_")
        if status in {"succeeded", "success", "complete"}:
            normalized["progress"] = "completed"
        elif status == "failure":
            normalized["progress"] = "failed"
        elif status == "canceled":
            normalized["progress"] = "cancelled"
        elif status in {"scheduling", "scheduled", "pending"}:
            normalized["progress"] = "queued"
        elif status in {"billing_required", "missing_token", "approval_required"}:
            normalized["progress"] = "blocked"
        return normalized

    async def upsert_response_rows(
        self, rows: list[dict[str, Any]], *, user_id: str = "dev"
    ) -> None:
        if not self._ready() or not rows:
            return
        now = _now()
        ops: list[Any] = []
        raw_rows = [dict(raw) for raw in rows]
        row_ids = [str(row.get("id") or "") for row in raw_rows if row.get("id")]
        identity_filters = []
        for row in raw_rows:
            row_user_id = str(row.get("user_id") or user_id)
            platform = str(row.get("platform") or "")
            job_id = str(row.get("job_id") or "")
            session_id = str(row.get("session_id") or "")
            if platform and job_id:
                identity_filters.append(
                    {"user_id": row_user_id, "platform": platform, "job_id": job_id}
                )
                if session_id:
                    identity_filters.append(
                        {
                            "session_id": session_id,
                            "platform": platform,
                            "job_id": job_id,
                        }
                    )
                if platform == "hf-jobs":
                    slug_regex = _hf_job_slug_regex(job_id)
                    if slug_regex:
                        identity_filters.append(
                            {
                                "user_id": row_user_id,
                                "platform": platform,
                                "job_id": slug_regex,
                            }
                        )
                        if session_id:
                            identity_filters.append(
                                {
                                    "session_id": session_id,
                                    "platform": platform,
                                    "job_id": slug_regex,
                                }
                            )
        existing_by_id: dict[str, dict[str, Any]] = {}
        existing_candidates: list[dict[str, Any]] = []
        if row_ids:
            cursor = self.db.response_rows.find(
                {"_id": {"$in": row_ids}},
                {
                    "_id": 1,
                    "id": 1,
                    "user_id": 1,
                    "session_id": 1,
                    "platform": 1,
                    "job_id": 1,
                    "actual_sequence_number": 1,
                    "display_session_number": 1,
                    "batch_number": 1,
                    "created_at": 1,
                    "run_type": 1,
                    "result_storage": 1,
                    "progress": 1,
                    "final_artifact_or_result": 1,
                    "completed_at": 1,
                    "provider_metadata": 1,
                    "error": 1,
                },
            )
            existing_by_id = {str(doc["_id"]): doc async for doc in cursor}
            existing_candidates.extend(existing_by_id.values())
        if identity_filters:
            cursor = self.db.response_rows.find(
                {"$or": identity_filters},
                {
                    "_id": 1,
                    "id": 1,
                    "user_id": 1,
                    "session_id": 1,
                    "platform": 1,
                    "job_id": 1,
                    "actual_sequence_number": 1,
                    "display_session_number": 1,
                    "batch_number": 1,
                    "created_at": 1,
                    "run_type": 1,
                    "result_storage": 1,
                    "progress": 1,
                    "final_artifact_or_result": 1,
                    "completed_at": 1,
                    "provider_metadata": 1,
                    "error": 1,
                },
            )
            async for doc in cursor:
                existing_candidates.append(doc)
        for raw in raw_rows:
            row = self._mongo_safe_value(dict(raw))
            row_id = str(row.get("id") or "")
            if not row_id:
                continue
            row_user_id = str(row.get("user_id") or user_id)
            existing = (
                _find_existing_response_row(existing_candidates, row, row_user_id)
                or existing_by_id.get(row_id)
                or {}
            )
            row.pop("_id", None)
            row.pop("inserted_at", None)
            row.pop("schema_version", None)
            for key in (
                "actual_sequence_number",
                "display_session_number",
                "batch_number",
                "created_at",
            ):
                if existing.get(key) is not None:
                    row[key] = existing[key]
            if existing.get("_id"):
                row_id = str(existing["_id"])
                row["id"] = str(existing.get("id") or existing["_id"])
            if (
                existing.get("job_id")
                and str(row.get("platform") or "") == "hf-jobs"
                and _same_hf_job_id(existing.get("job_id"), row.get("job_id"))
                and str(existing.get("job_id")).startswith("https://")
                and not str(row.get("job_id") or "").startswith("https://")
            ):
                row["job_id"] = existing["job_id"]
            row = _merge_response_row(existing, row)
            row["user_id"] = row_user_id
            row["updated_at"] = now
            if str(row.get("platform") or "") == "hf-jobs" and row.get("job_id"):
                slug_regex = _hf_job_slug_regex(row.get("job_id"))
                has_duplicate = any(
                    str(candidate.get("_id") or "") != row_id
                    and str(candidate.get("platform") or "") == "hf-jobs"
                    and _same_hf_job_id(candidate.get("job_id"), row.get("job_id"))
                    for candidate in existing_candidates
                )
                if slug_regex and has_duplicate:
                    ops.append(
                        DeleteMany(
                            {
                                "user_id": row_user_id,
                                "platform": "hf-jobs",
                                "job_id": slug_regex,
                                "_id": {"$ne": row_id},
                            }
                        )
                    )
            ops.append(
                UpdateOne(
                    {"_id": row_id},
                    {
                        "$set": row,
                        "$setOnInsert": {
                            "_id": row_id,
                            "inserted_at": now,
                            "schema_version": SCHEMA_VERSION,
                        },
                    },
                    upsert=True,
                )
            )
        if not ops:
            return
        try:
            await self.db.response_rows.bulk_write(ops, ordered=False)
        except PyMongoError as e:
            logger.debug("Failed to upsert response rows: %s", e)

    async def list_response_rows(
        self,
        *,
        user_id: str,
        page: int = 1,
        page_size: int = 50,
        platform: str | None = None,
        progress: str | None = None,
        model: str | None = None,
        session_id: str | None = None,
        job_id: str | None = None,
        q: str | None = None,
    ) -> dict[str, Any]:
        safe_page = max(1, int(page or 1))
        safe_page_size = min(200, max(1, int(page_size or 50)))
        if not self._ready():
            return {
                "rows": [],
                "page": safe_page,
                "page_size": safe_page_size,
                "total_rows": 0,
                "total_pages": 0,
                "has_next": False,
                "has_previous": False,
            }
        query = self._response_query(
            user_id=user_id,
            platform=platform,
            progress=progress,
            model=model,
            session_id=session_id,
            job_id=job_id,
            q=q,
        )
        total_rows = await self.db.response_rows.count_documents(query)
        total_pages = math.ceil(total_rows / safe_page_size) if total_rows else 0
        cursor = (
            self.db.response_rows.find(query, {"_id": 0})
            .sort([("actual_sequence_number", -1), ("created_at", -1)])
            .skip((safe_page - 1) * safe_page_size)
            .limit(safe_page_size)
        )
        rows = [self._normalize_response_row(row) async for row in cursor]
        return {
            "rows": rows,
            "page": safe_page,
            "page_size": safe_page_size,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "has_next": bool(total_pages and safe_page < total_pages),
            "has_previous": safe_page > 1 and total_pages > 0,
        }

    async def get_response_summary(self, *, user_id: str) -> dict[str, Any]:
        if not self._ready():
            return await super().get_response_summary(user_id=user_id)
        query: dict[str, Any] = {}
        if user_id != "dev":
            query["user_id"] = user_id
        total = await self.db.response_rows.count_documents(query)
        latest = await self.db.response_rows.find_one(
            query,
            sort=[("actual_sequence_number", -1), ("created_at", -1)],
        )
        batch = int((latest or {}).get("batch_number") or 1)
        return {
            "total_responses": total,
            "visible_count": min(total, 50),
            "batch_number": batch,
            "has_rows": bool(total),
            "button_enabled": True,
            "durable": True,
            "store_type": "mongodb",
        }


_store: NoopSessionStore | MongoSessionStore | None = None


def get_session_store() -> NoopSessionStore | MongoSessionStore:
    global _store
    if _store is None:
        uri = os.environ.get("MONGODB_URI")
        db_name = os.environ.get("MONGODB_DB", "liga_ml")
        _store = MongoSessionStore(uri, db_name) if uri else NoopSessionStore()
    return _store


def session_store_status(
    store: NoopSessionStore | MongoSessionStore | None = None,
) -> dict[str, Any]:
    """Return a non-secret health summary for hosted session durability."""
    active_store = store or get_session_store()
    if isinstance(active_store, MongoSessionStore) or getattr(
        active_store, "enabled", False
    ):
        durable = bool(active_store.enabled)
        return {
            "type": "mongodb",
            "durable": durable,
            "warning": None
            if durable
            else "MongoDB session persistence is not available; sessions will not survive restarts",
        }
    return {
        "type": "noop",
        "durable": False,
        "warning": NO_DURABLE_STORE_WARNING,
    }


def _reset_store_for_tests(
    store: NoopSessionStore | MongoSessionStore | None = None,
) -> None:
    global _store
    _store = store
