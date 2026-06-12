"""Internal audit timeline events derived from run and usage state.

The audit timeline is intentionally provider-agnostic and self-contained. It
does not export to external observability vendors; it records a sanitized,
human-readable operational history for users and admins.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from agent.core.redact import redact_text
from agent.core.usage import (
    budget_warning_for,
    normalize_provider,
    sanitize_metadata,
)

AuditCategory = Literal[
    "session",
    "dataset",
    "chat",
    "planner",
    "approval",
    "tool",
    "provider_job",
    "usage",
    "result",
    "error",
    "system",
    "security",
]
AuditSeverity = Literal["info", "warning", "error", "critical"]
AuditActor = Literal["user", "assistant", "system", "provider", "admin"]

AUDIT_CATEGORIES: set[str] = {
    "session",
    "dataset",
    "chat",
    "planner",
    "approval",
    "tool",
    "provider_job",
    "usage",
    "result",
    "error",
    "system",
    "security",
}
AUDIT_SEVERITIES: set[str] = {"info", "warning", "error", "critical"}
AUDIT_ACTORS: set[str] = {"user", "assistant", "system", "provider", "admin"}
PROVIDER_TOOL_NAMES = {
    "hf_jobs": "hf-jobs",
    "gcp_vertex_jobs": "gcp-vertex",
    "aws_sagemaker_jobs": "aws-sagemaker",
}
TERMINAL_SUCCESS_STATES = {"succeeded", "completed", "success"}
TERMINAL_FAILURE_STATES = {
    "failed",
    "error",
    "billing_required",
    "blocked",
    "launch_blocked",
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def audit_timeline_enabled() -> bool:
    raw = os.environ.get("AUDIT_TIMELINE_ENABLED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def audit_retention_days() -> int:
    raw = os.environ.get("AUDIT_EVENT_RETENTION_DAYS", "30")
    try:
        parsed = int(str(raw).strip())
    except (TypeError, ValueError):
        return 30
    return max(1, parsed)


def audit_store_status(store_status: dict[str, Any]) -> dict[str, Any]:
    durable = bool(store_status.get("durable"))
    enabled = audit_timeline_enabled()
    warning = None
    if not enabled:
        warning = (
            "AUDIT_TIMELINE_ENABLED=false; audit timeline APIs return empty timelines."
        )
    elif not durable:
        warning = "MONGODB_URI is not configured; audit events are in-memory only."
    return {
        "type": str(store_status.get("type") or "unknown"),
        "durable": durable,
        "enabled": enabled,
        "warning": warning,
    }


def sanitize_audit_metadata(value: Any, *, depth: int = 0) -> Any:
    sanitized = sanitize_metadata(value, depth=depth)
    return _redact_secret_values(sanitized)


def _redact_secret_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): redacted
            for key, item in value.items()
            if (redacted := _redact_secret_values(item)) is not None
        }
    if isinstance(value, list):
        return [
            redacted
            for item in value
            if (redacted := _redact_secret_values(item)) is not None
        ]
    if isinstance(value, str):
        return redact_text(value)[:2000]
    return value


def _safe_text(value: Any, limit: int = 500) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    return redact_text(text)[:limit]


def _audit_hash(parts: list[Any]) -> str:
    raw = "|".join(str(part or "") for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_audit_event(
    *,
    session_id: str,
    event_type: str,
    category: str,
    severity: str = "info",
    status: str = "unknown",
    title: str,
    message: str,
    actor: str = "system",
    run_id: str | None = None,
    usage_id: str | None = None,
    provider: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    tool_name: str | None = None,
    operation: str | None = None,
    approval_id: str | None = None,
    job_id: str | None = None,
    job_url: str | None = None,
    artifact_url: str | None = None,
    dataset_name: str | None = None,
    model_name: str | None = None,
    output_policy: str | None = None,
    estimated_cost_usd: float | None = None,
    known_cost_usd: float | None = None,
    error_code: str | None = None,
    error_summary: str | None = None,
    safe_metadata: dict[str, Any] | None = None,
    timestamp: datetime | None = None,
    audit_id: str | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    normalized_category = category if category in AUDIT_CATEGORIES else "system"
    normalized_severity = severity if severity in AUDIT_SEVERITIES else "info"
    normalized_actor = actor if actor in AUDIT_ACTORS else "system"
    resolved_provider = normalize_provider(provider)
    resolved_entity_id = entity_id or job_id or approval_id or usage_id or run_id
    key = idempotency_key or ":".join(
        str(part or "")
        for part in (
            session_id,
            run_id,
            event_type,
            resolved_entity_id,
            status,
            resolved_provider,
        )
    )
    resolved_audit_id = audit_id or f"audit_{_audit_hash([key])}"
    return {
        "_id": resolved_audit_id,
        "audit_id": resolved_audit_id,
        "idempotency_key": key,
        "session_id": session_id,
        "run_id": run_id,
        "usage_id": usage_id,
        "provider": resolved_provider,
        "event_type": event_type,
        "category": normalized_category,
        "severity": normalized_severity,
        "status": status or "unknown",
        "title": _safe_text(title, 160) or event_type.replace("_", " "),
        "message": _safe_text(message, 1000) or "",
        "timestamp": timestamp or utc_now(),
        "actor": normalized_actor,
        "entity_type": entity_type,
        "entity_id": resolved_entity_id,
        "tool_name": tool_name,
        "operation": operation,
        "approval_id": approval_id,
        "job_id": job_id,
        "job_url": _safe_text(job_url, 1000),
        "artifact_url": _safe_text(artifact_url, 1000),
        "dataset_name": _safe_text(dataset_name, 300),
        "model_name": _safe_text(model_name, 300),
        "output_policy": output_policy,
        "estimated_cost_usd": estimated_cost_usd,
        "known_cost_usd": known_cost_usd,
        "error_code": error_code,
        "error_summary": _safe_text(error_summary, 500),
        "safe_metadata": sanitize_audit_metadata(safe_metadata or {}),
    }


def training_preflight_audit_events(
    preflight: dict[str, Any],
    *,
    include_started: bool = False,
) -> list[dict[str, Any]]:
    """Build sanitized audit events for a training preflight result."""

    session_id = str(preflight.get("session_id") or "")
    if not session_id:
        return []
    run_id = str(preflight.get("run_id") or "") or None
    preflight_id = str(preflight.get("preflight_id") or "")
    provider = str(preflight.get("provider") or "unknown")
    status = str(preflight.get("status") or "unknown")
    launch_ready = preflight.get("launch_ready") is True
    manual_approval_allowed = preflight.get("manual_approval_allowed") is True
    blocking = preflight.get("blocking_reasons")
    unknowns = preflight.get("unknown_reasons")
    warnings = preflight.get("warning_reasons")
    safe_metadata = {
        "preflight_id": preflight_id,
        "model_id": preflight.get("model_id"),
        "hardware_id": preflight.get("hardware_id"),
        "blocking_reasons": blocking if isinstance(blocking, list) else [],
        "unknown_reasons": unknowns if isinstance(unknowns, list) else [],
        "warning_reasons": warnings if isinstance(warnings, list) else [],
        "manual_approval_allowed": manual_approval_allowed,
        "manual_approval_reason": preflight.get("manual_approval_reason"),
        "metadata": preflight.get("metadata")
        if isinstance(preflight.get("metadata"), dict)
        else {},
    }
    common = {
        "session_id": session_id,
        "run_id": run_id,
        "category": "planner",
        "actor": "system",
        "provider": provider,
        "entity_type": "training_preflight",
        "entity_id": preflight_id,
        "model_name": preflight.get("model_id"),
        "output_policy": preflight.get("output_policy"),
        "safe_metadata": safe_metadata,
    }
    events: list[dict[str, Any]] = []
    if include_started:
        events.append(
            build_audit_event(
                **common,
                event_type="training_preflight_started",
                status="started",
                severity="info",
                title="Training preflight started",
                message="Local training preflight checks started.",
            )
        )
    if status == "failed":
        events.append(
            build_audit_event(
                **common,
                event_type="training_preflight_failed",
                status="failed",
                severity="error",
                title="Training preflight failed",
                message=preflight.get("safe_summary")
                or "Training preflight found blocking failures.",
                error_summary=preflight.get("safe_summary"),
            )
        )
    elif status == "unknown":
        events.append(
            build_audit_event(
                **common,
                event_type="training_preflight_unknown",
                status="unknown",
                severity="warning",
                title="Training preflight unknown",
                message=preflight.get("safe_summary")
                or "Training preflight requires live checks in a later slice.",
            )
        )
    else:
        events.append(
            build_audit_event(
                **common,
                event_type="training_preflight_completed",
                status=status,
                severity="info" if launch_ready else "warning",
                title="Training preflight completed",
                message=preflight.get("safe_summary")
                or "Training preflight completed.",
            )
        )
    events.append(
        build_audit_event(
            **common,
            event_type="training_preflight_launch_ready"
            if launch_ready
            else "training_preflight_launch_blocked",
            status="ready" if launch_ready else "blocked",
            severity="info" if launch_ready else "warning",
            title="Training launch ready"
            if launch_ready
            else "Training launch blocked",
            message=(
                "Training preflight marked the plan launch-ready."
                if launch_ready
                else (
                    "Training preflight allows bounded smoke launch with explicit approval."
                    if manual_approval_allowed
                    else "Training preflight did not mark the plan launch-ready."
                )
            ),
        )
    )
    if manual_approval_allowed and not launch_ready:
        events.append(
            build_audit_event(
                **common,
                event_type="training_preflight_manual_approval_allowed",
                status="approval_required",
                severity="warning",
                title="Bounded smoke manual approval allowed",
                message=preflight.get("manual_approval_reason")
                or "Only non-critical unknown checks remain; bounded smoke requires explicit approval.",
            )
        )
    return events


def event_from_run_event(
    *,
    session_id: str,
    run_id: str,
    event_type: str,
    payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    payload = payload or {}
    if event_type in {"assistant_chunk", "heartbeat", "tool_log"}:
        return []
    if event_type == "run_created":
        return [
            build_audit_event(
                session_id=session_id,
                run_id=run_id,
                event_type="run_created",
                category="chat",
                status="created",
                actor="system",
                title="Run created",
                message="A durable background run record was created.",
                provider=payload.get("provider"),
                entity_type="run",
                entity_id=run_id,
                safe_metadata={"request_id": payload.get("request_id")},
            )
        ]
    if event_type == "run_started":
        return [
            build_audit_event(
                session_id=session_id,
                run_id=run_id,
                event_type="run_started",
                category="chat",
                status="started",
                actor="system",
                title="Run started",
                message="The agent started processing the submitted prompt.",
                provider=payload.get("provider"),
                entity_type="run",
                entity_id=run_id,
                safe_metadata={"submission_id": payload.get("submission_id")},
            )
        ]
    if event_type == "approval_required":
        return _approval_required_events(session_id, run_id, payload)
    if event_type == "approval_resolved":
        return _approval_resolved_events(session_id, run_id, payload)
    if event_type == "tool_call" and payload.get("tool") == "dataset_discovery":
        return [
            build_audit_event(
                session_id=session_id,
                run_id=run_id,
                event_type="dataset_discovery_started",
                category="dataset",
                status="started",
                actor="assistant",
                title="Dataset discovery started",
                message="The agent started no-upload dataset discovery.",
                tool_name="dataset_discovery",
                entity_type="dataset_discovery",
                entity_id=payload.get("tool_call_id"),
                safe_metadata=payload.get("arguments")
                if isinstance(payload.get("arguments"), dict)
                else {},
            )
        ]
    if event_type == "tool_state_change":
        return _tool_state_events(session_id, run_id, payload)
    if event_type == "tool_output" and payload.get("tool") == "dataset_discovery":
        return _dataset_discovery_events(session_id, run_id, payload)
    if event_type == "tool_output" and payload.get("tool") == "training_planner":
        return _training_recommendation_events(session_id, run_id, payload)
    if event_type == "turn_complete":
        return [
            build_audit_event(
                session_id=session_id,
                run_id=run_id,
                event_type="final_result_available",
                category="result",
                status="completed",
                actor="assistant",
                title="Final result available",
                message=_safe_text(payload.get("final_response"), 500)
                or "The agent completed the turn.",
                entity_type="run",
                entity_id=run_id,
                safe_metadata={"history_size": payload.get("history_size")},
            )
        ]
    if event_type == "interrupted":
        return [
            build_audit_event(
                session_id=session_id,
                run_id=run_id,
                event_type="run_interrupted",
                category="chat",
                severity="warning",
                status="interrupted",
                actor="user",
                title="Run interrupted",
                message="The run was interrupted before completion.",
                entity_type="run",
                entity_id=run_id,
                safe_metadata=payload,
            )
        ]
    if event_type in {"error", "stream_error"}:
        error_summary = payload.get("error") or event_type
        return [
            build_audit_event(
                session_id=session_id,
                run_id=run_id,
                event_type="stream_error"
                if event_type == "stream_error"
                else "provider_error",
                category="error",
                severity="error",
                status="failed",
                actor="system",
                title="Run error",
                message=_safe_text(error_summary, 500) or "The run failed.",
                provider=payload.get("provider"),
                entity_type="run",
                entity_id=run_id,
                error_code=payload.get("error_type"),
                error_summary=error_summary,
                safe_metadata=payload,
            )
        ]
    return []


def _dataset_discovery_events(
    session_id: str, run_id: str, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    structured = payload.get("structured")
    if not isinstance(structured, dict) or payload.get("success") is False:
        return [
            build_audit_event(
                session_id=session_id,
                run_id=run_id,
                event_type="dataset_discovery_failed",
                category="dataset",
                severity="warning",
                status="failed",
                actor="assistant",
                title="Dataset discovery failed",
                message="Dataset discovery did not return structured candidates.",
                tool_name="dataset_discovery",
                entity_type="dataset_discovery",
                entity_id=payload.get("tool_call_id"),
                safe_metadata=payload,
            )
        ]

    candidates = structured.get("candidates") if isinstance(structured, dict) else []
    candidates = candidates if isinstance(candidates, list) else []
    recommended = structured.get("recommended_candidate")
    recommended = recommended if isinstance(recommended, dict) else None
    excluded = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("excluded") is True
    ]
    events = [
        build_audit_event(
            session_id=session_id,
            run_id=run_id,
            event_type="dataset_discovery_completed",
            category="dataset",
            status="completed",
            actor="assistant",
            title="Dataset discovery completed",
            message=f"Dataset discovery found {len(candidates)} candidate(s).",
            tool_name="dataset_discovery",
            entity_type="dataset_discovery",
            entity_id=payload.get("tool_call_id"),
            safe_metadata={
                "tool_call_id": payload.get("tool_call_id"),
                "query": structured.get("query"),
                "warnings": structured.get("warnings"),
                "candidate_count": len(candidates),
                "excluded_count": len(excluded),
                "recommended_dataset_id": recommended.get("dataset_id")
                if recommended
                else None,
                "uploads_performed": False,
                "downloads_performed": False,
                "provider_jobs_launched": False,
                "resources_created": False,
            },
        )
    ]
    if recommended:
        events.append(
            build_audit_event(
                session_id=session_id,
                run_id=run_id,
                event_type="dataset_candidate_recommended",
                category="dataset",
                status="recommended",
                actor="assistant",
                title="Dataset candidate recommended",
                message=str(
                    recommended.get("title")
                    or recommended.get("dataset_id")
                    or "Dataset candidate recommended"
                ),
                tool_name="dataset_discovery",
                entity_type="dataset_candidate",
                entity_id=recommended.get("dataset_id"),
                dataset_name=recommended.get("title") or recommended.get("dataset_id"),
                safe_metadata=recommended,
            )
        )
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate.get("excluded"):
            continue
        events.append(
            build_audit_event(
                session_id=session_id,
                run_id=run_id,
                event_type="dataset_candidate_excluded",
                category="dataset",
                severity="warning",
                status="excluded",
                actor="assistant",
                title="Dataset candidate excluded",
                message=str(
                    candidate.get("exclusion_reason")
                    or candidate.get("title")
                    or candidate.get("dataset_id")
                    or "Dataset candidate excluded"
                ),
                tool_name="dataset_discovery",
                entity_type="dataset_candidate",
                entity_id=candidate.get("dataset_id"),
                dataset_name=candidate.get("title") or candidate.get("dataset_id"),
                safe_metadata=candidate,
            )
        )
    return events


def _training_recommendation_events(
    session_id: str, run_id: str, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    structured = payload.get("structured")
    if not isinstance(structured, dict) or payload.get("success") is False:
        return []
    recommendation = structured.get("recommendation")
    recommendation = recommendation if isinstance(recommendation, dict) else {}
    selected_model = recommendation.get("selected_model")
    selected_model = selected_model if isinstance(selected_model, dict) else {}
    selected_provider = recommendation.get("selected_provider")
    selected_provider = selected_provider if isinstance(selected_provider, dict) else {}
    selected_hardware = recommendation.get("selected_hardware")
    selected_hardware = selected_hardware if isinstance(selected_hardware, dict) else {}
    events = [
        build_audit_event(
            session_id=session_id,
            run_id=run_id,
            event_type="model_recommendation_created",
            category="planner",
            status="recommended",
            actor="assistant",
            title="Model recommendation created",
            message=str(
                selected_model.get("model_id")
                or structured.get("recommended_model")
                or "Model recommended"
            ),
            provider=structured.get("provider"),
            tool_name="training_planner",
            entity_type="model",
            entity_id=selected_model.get("model_id")
            or structured.get("recommended_model"),
            model_name=selected_model.get("model_id")
            or structured.get("recommended_model"),
            output_policy=structured.get("output_policy"),
            estimated_cost_usd=recommendation.get("estimated_cost_usd"),
            safe_metadata=recommendation,
        ),
        build_audit_event(
            session_id=session_id,
            run_id=run_id,
            event_type="provider_recommendation_created",
            category="planner",
            status="recommended",
            actor="assistant",
            title="Provider recommendation created",
            message=str(
                selected_provider.get("display_name")
                or structured.get("provider")
                or "Provider recommended"
            ),
            provider=structured.get("provider"),
            tool_name="training_planner",
            entity_type="provider",
            entity_id=selected_provider.get("provider_id")
            or structured.get("provider"),
            output_policy=structured.get("output_policy"),
            estimated_cost_usd=recommendation.get("estimated_cost_usd"),
        ),
        build_audit_event(
            session_id=session_id,
            run_id=run_id,
            event_type="hardware_recommendation_created",
            category="planner",
            status="recommended",
            actor="assistant",
            title="Hardware recommendation created",
            message=str(
                selected_hardware.get("display_name") or "Hardware recommended"
            ),
            provider=structured.get("provider"),
            tool_name="training_planner",
            entity_type="hardware",
            entity_id=selected_hardware.get("hardware_id"),
            estimated_cost_usd=recommendation.get("estimated_cost_usd"),
            safe_metadata=selected_hardware,
        ),
    ]
    for fallback in recommendation.get("fallbacks") or []:
        if not isinstance(fallback, dict):
            continue
        events.append(
            build_audit_event(
                session_id=session_id,
                run_id=run_id,
                event_type="fallback_recommended",
                category="planner",
                severity="warning",
                status="recommended",
                actor="assistant",
                title="Fallback recommended",
                message=str(fallback.get("reason") or "Fallback recommended"),
                provider=structured.get("provider"),
                tool_name="training_planner",
                entity_type="fallback",
                entity_id=fallback.get("blocked_option"),
                safe_metadata=fallback,
            )
        )
    for warning in recommendation.get("warnings") or []:
        if not isinstance(warning, dict):
            continue
        if str(warning.get("category") or "").lower() != "quota":
            continue
        events.append(
            build_audit_event(
                session_id=session_id,
                run_id=run_id,
                event_type="quota_warning_recorded",
                category="planner",
                severity="warning",
                status="warning",
                actor="assistant",
                title="Quota warning recorded",
                message=str(warning.get("message") or "Quota warning recorded"),
                provider=structured.get("provider"),
                tool_name="training_planner",
                entity_type="quota",
                safe_metadata=warning,
            )
        )
    return events


def _approval_required_events(
    session_id: str, run_id: str, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    tools = payload.get("tools") if isinstance(payload, dict) else None
    events: list[dict[str, Any]] = []
    for tool_payload in tools if isinstance(tools, list) else []:
        if not isinstance(tool_payload, dict):
            continue
        tool_name = str(tool_payload.get("tool") or "tool")
        args = (
            tool_payload.get("arguments")
            if isinstance(tool_payload.get("arguments"), dict)
            else {}
        )
        provider = normalize_provider(tool_payload.get("provider"), tool_name)
        approval_id = str(
            tool_payload.get("approval_id") or tool_payload.get("tool_call_id") or ""
        )
        estimated_cost = _float_or_none(
            tool_payload.get("estimated_cost_usd") or payload.get("estimated_cost_usd")
        )
        dataset_name = args.get("dataset_name") or args.get("dataset_id")
        model_name = (
            args.get("model_id") or args.get("base_model") or args.get("model_name")
        )
        events.append(
            build_audit_event(
                session_id=session_id,
                run_id=run_id,
                event_type="approval_required",
                category="approval",
                severity="warning",
                status="pending",
                actor="assistant",
                title="Approval required",
                message=f"Review {tool_name} before the agent continues.",
                provider=provider,
                entity_type="approval",
                entity_id=approval_id or tool_name,
                tool_name=tool_name,
                operation=str(args.get("operation") or "run"),
                approval_id=approval_id or None,
                dataset_name=dataset_name,
                model_name=model_name,
                output_policy=args.get("output_policy"),
                estimated_cost_usd=estimated_cost,
                safe_metadata={"tool": tool_payload, "arguments": args},
            )
        )
        if estimated_cost is not None:
            usage_id = f"{run_id}:approval:{approval_id or tool_name}"
            events.append(
                build_audit_event(
                    session_id=session_id,
                    run_id=run_id,
                    usage_id=usage_id,
                    event_type="usage_estimated",
                    category="usage",
                    severity="info",
                    status="pending",
                    actor="system",
                    title="Usage estimated",
                    message=f"Estimated {provider} cost is ${estimated_cost:.2f}.",
                    provider=provider,
                    entity_type="usage",
                    entity_id=usage_id,
                    tool_name=tool_name,
                    operation=str(args.get("operation") or "run"),
                    approval_id=approval_id or None,
                    estimated_cost_usd=estimated_cost,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    output_policy=args.get("output_policy"),
                )
            )
            if warning := budget_warning_for(provider, estimated_cost):
                events.append(
                    build_audit_event(
                        session_id=session_id,
                        run_id=run_id,
                        usage_id=usage_id,
                        event_type="budget_warning",
                        category="usage",
                        severity="warning",
                        status="pending",
                        actor="system",
                        title="Budget warning",
                        message=warning,
                        provider=provider,
                        entity_type="usage",
                        entity_id=usage_id,
                        estimated_cost_usd=estimated_cost,
                    )
                )
    return events


def _approval_resolved_events(
    session_id: str, run_id: str, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    approvals = payload.get("approvals") if isinstance(payload, dict) else None
    events: list[dict[str, Any]] = []
    for approval in approvals if isinstance(approvals, list) else []:
        if not isinstance(approval, dict):
            continue
        approved = bool(approval.get("approved"))
        approval_id = str(
            approval.get("approval_id") or approval.get("tool_call_id") or ""
        )
        status = "approved" if approved else "rejected"
        events.append(
            build_audit_event(
                session_id=session_id,
                run_id=run_id,
                event_type=f"approval_{status}",
                category="approval",
                severity="info" if approved else "warning",
                status=status,
                actor="user",
                title=f"Approval {status}",
                message=f"The user {status} a pending tool request.",
                entity_type="approval",
                entity_id=approval_id,
                approval_id=approval_id or None,
                safe_metadata={"feedback": approval.get("feedback")},
            )
        )
    return events


def _tool_state_events(
    session_id: str, run_id: str, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    tool_name = str(payload.get("tool") or "tool")
    state = str(payload.get("state") or "unknown").lower()
    provider = normalize_provider(payload.get("provider"), tool_name)
    job_id = str(
        payload.get("jobName") or payload.get("job_id") or payload.get("jobId") or ""
    )
    tool_call_id = str(payload.get("tool_call_id") or job_id or tool_name)
    artifact_url = payload.get("outputDir") or payload.get("s3ModelArtifact")
    job_url = payload.get("jobUrl") or payload.get("cloudWatchLogsUrl")
    if tool_name == "training_planner":
        if state in {"running", "started", "queued"}:
            return [
                build_audit_event(
                    session_id=session_id,
                    run_id=run_id,
                    event_type="planner_started",
                    category="planner",
                    status="started",
                    actor="assistant",
                    title="Planner started",
                    message="The training planner started.",
                    tool_name=tool_name,
                    entity_type="tool_call",
                    entity_id=tool_call_id,
                )
            ]
        if state in TERMINAL_SUCCESS_STATES:
            return [
                build_audit_event(
                    session_id=session_id,
                    run_id=run_id,
                    event_type="planner_completed",
                    category="planner",
                    status="completed",
                    actor="assistant",
                    title="Planner completed",
                    message="The training planner completed.",
                    tool_name=tool_name,
                    entity_type="tool_call",
                    entity_id=tool_call_id,
                )
            ]
    if provider != "unknown" and tool_name in PROVIDER_TOOL_NAMES:
        return _provider_job_events(
            session_id=session_id,
            run_id=run_id,
            payload=payload,
            provider=provider,
            tool_name=tool_name,
            state=state,
            job_id=job_id,
            tool_call_id=tool_call_id,
            job_url=job_url,
            artifact_url=artifact_url,
        )
    status = {
        "running": "started",
        "started": "started",
        "succeeded": "succeeded",
        "completed": "completed",
        "failed": "failed",
        "error": "failed",
    }.get(state, state or "unknown")
    return [
        build_audit_event(
            session_id=session_id,
            run_id=run_id,
            event_type=f"tool_call_{'failed' if status == 'failed' else status}",
            category="tool",
            severity="error" if status == "failed" else "info",
            status=status,
            actor="assistant",
            title=f"{tool_name} {status}",
            message=f"Tool {tool_name} changed state to {state}.",
            tool_name=tool_name,
            entity_type="tool_call",
            entity_id=tool_call_id,
            error_summary=payload.get("failureReason") or payload.get("reason"),
            safe_metadata=payload,
        )
    ]


def _provider_job_events(
    *,
    session_id: str,
    run_id: str,
    payload: dict[str, Any],
    provider: str,
    tool_name: str,
    state: str,
    job_id: str,
    tool_call_id: str,
    job_url: Any,
    artifact_url: Any,
) -> list[dict[str, Any]]:
    if state in {"approved"}:
        status = "approved"
        event_type = "approval_approved"
        category = "approval"
        title = "Approval approved"
        message = f"{provider} job approval was accepted."
        severity = "info"
    elif state in {"running", "queued", "starting", "started"}:
        status = "running"
        event_type = "provider_job_started"
        category = "provider_job"
        title = f"{provider} job running"
        message = f"{provider} job {job_id or tool_call_id} is running."
        severity = "info"
    elif state in TERMINAL_SUCCESS_STATES:
        status = "succeeded"
        event_type = "provider_job_succeeded"
        category = "provider_job"
        title = f"{provider} job succeeded"
        message = f"{provider} job {job_id or tool_call_id} succeeded."
        severity = "info"
    elif state in {"blocked", "launch_blocked"}:
        status = "failed"
        event_type = "provider_launch_blocked"
        category = "provider_job"
        title = f"{provider} launch blocked"
        message = _safe_text(
            payload.get("reason") or payload.get("failureReason"), 500
        ) or (f"{provider} job was not launched before provider submission.")
        severity = "error"
    elif state in TERMINAL_FAILURE_STATES:
        status = "failed"
        event_type = "provider_job_failed"
        category = "provider_job"
        title = f"{provider} job failed"
        message = _safe_text(
            payload.get("failureReason") or payload.get("reason"), 500
        ) or (f"{provider} job {job_id or tool_call_id} failed.")
        severity = "error"
    elif state in {"rejected", "cancelled"}:
        status = "rejected" if state == "rejected" else "cancelled"
        event_type = f"provider_job_{status}"
        category = "provider_job"
        title = f"{provider} job {status}"
        message = f"{provider} job {job_id or tool_call_id} was {status}."
        severity = "warning"
    else:
        status = state or "unknown"
        event_type = "provider_job_running"
        category = "provider_job"
        title = f"{provider} job update"
        message = f"{provider} job {job_id or tool_call_id} changed state to {state}."
        severity = "info"
    events = [
        build_audit_event(
            session_id=session_id,
            run_id=run_id,
            event_type=event_type,
            category=category,
            severity=severity,
            status=status,
            actor="provider",
            title=title,
            message=message,
            provider=provider,
            entity_type="provider_job",
            entity_id=job_id or tool_call_id,
            tool_name=tool_name,
            job_id=job_id or None,
            job_url=job_url,
            artifact_url=artifact_url,
            output_policy=payload.get("outputPolicy"),
            error_summary=payload.get("failureReason") or payload.get("reason"),
            safe_metadata=payload,
        )
    ]
    if status == "succeeded" and artifact_url:
        events.append(
            build_audit_event(
                session_id=session_id,
                run_id=run_id,
                event_type="artifact_available",
                category="result",
                status="completed",
                actor="provider",
                title="Artifact available",
                message=f"{provider} produced an artifact.",
                provider=provider,
                entity_type="artifact",
                entity_id=job_id or tool_call_id,
                tool_name=tool_name,
                job_id=job_id or None,
                artifact_url=artifact_url,
                output_policy=payload.get("outputPolicy"),
            )
        )
    return events


def summarize_audit_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    by_provider: dict[str, int] = {}
    for event in events:
        for target, key in (
            (by_category, str(event.get("category") or "system")),
            (by_severity, str(event.get("severity") or "info")),
            (by_provider, str(event.get("provider") or "unknown")),
        ):
            target[key] = target.get(key, 0) + 1
    warnings_errors = [
        event
        for event in events
        if event.get("severity") in {"warning", "error", "critical"}
    ][:10]
    return {
        "total_events": len(events),
        "counts_by_category": dict(sorted(by_category.items())),
        "counts_by_severity": dict(sorted(by_severity.items())),
        "counts_by_provider": dict(sorted(by_provider.items())),
        "latest_warnings_errors": warnings_errors,
        "provider_job_timeline": [
            event for event in events if event.get("category") == "provider_job"
        ],
        "approval_timeline": [
            event for event in events if event.get("category") == "approval"
        ],
        "dataset_timeline": [
            event for event in events if event.get("category") == "dataset"
        ],
        "usage_cost_timeline": [
            event for event in events if event.get("category") == "usage"
        ],
    }


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def new_ad_hoc_audit_id() -> str:
    return f"audit_{uuid.uuid4().hex}"
