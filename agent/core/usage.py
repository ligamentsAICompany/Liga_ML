"""Provider-agnostic usage ledger for estimates, quotas, and budgets.

The ledger intentionally does not call live billing APIs.  It records the
non-secret metadata already emitted by approvals and provider tools so the UI
can explain expected spend and readiness risk without launching new work.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable, Literal

from agent.core.redact import SECRET_KEY_RE, redact_text

UsageProvider = Literal["hf-jobs", "gcp-vertex", "aws-sagemaker", "llm", "unknown"]

PROVIDER_BY_TOOL = {
    "hf_jobs": "hf-jobs",
    "gcp_vertex_jobs": "gcp-vertex",
    "aws_sagemaker_jobs": "aws-sagemaker",
}
TOOL_BY_PROVIDER = {value: key for key, value in PROVIDER_BY_TOOL.items()}
PROVIDERS: tuple[UsageProvider, ...] = (
    "hf-jobs",
    "gcp-vertex",
    "aws-sagemaker",
    "llm",
    "unknown",
)
TERMINAL_STATES = {
    "succeeded",
    "completed",
    "success",
    "failed",
    "error",
    "cancelled",
    "rejected",
}


@dataclass(frozen=True)
class BudgetConfig:
    daily_budget_usd: float | None
    monthly_budget_usd: float | None
    provider_daily_budgets_usd: dict[str, float | None]


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_budget(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def load_budget_config() -> BudgetConfig:
    return BudgetConfig(
        daily_budget_usd=parse_budget(os.environ.get("DEFAULT_DAILY_BUDGET_USD")),
        monthly_budget_usd=parse_budget(os.environ.get("DEFAULT_MONTHLY_BUDGET_USD")),
        provider_daily_budgets_usd={
            "hf-jobs": parse_budget(os.environ.get("HF_DAILY_BUDGET_USD")),
            "gcp-vertex": parse_budget(os.environ.get("GCLOUD_DAILY_BUDGET_USD")),
            "aws-sagemaker": parse_budget(os.environ.get("AWS_DAILY_BUDGET_USD")),
        },
    )


def usage_dashboard_enabled() -> bool:
    raw = os.environ.get("USAGE_DASHBOARD_ENABLED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def normalize_provider(value: Any, tool_name: Any = None) -> UsageProvider:
    provider = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "hf": "hf-jobs",
        "hf-jobs": "hf-jobs",
        "huggingface": "hf-jobs",
        "hugging-face": "hf-jobs",
        "gcp": "gcp-vertex",
        "google": "gcp-vertex",
        "vertex": "gcp-vertex",
        "gcp-vertex": "gcp-vertex",
        "aws": "aws-sagemaker",
        "sagemaker": "aws-sagemaker",
        "aws-sagemaker": "aws-sagemaker",
        "llm": "llm",
    }
    mapped = aliases.get(provider)
    if mapped:
        return mapped  # type: ignore[return-value]
    if tool_name:
        return PROVIDER_BY_TOOL.get(str(tool_name), "unknown")  # type: ignore[return-value]
    return "unknown"


def tool_operation(tool_name: str, args: dict[str, Any]) -> str:
    operation = str(args.get("operation") or "").strip().lower()
    if operation:
        return operation
    if tool_name in {"hf_jobs", "gcp_vertex_jobs", "aws_sagemaker_jobs"}:
        return "run"
    return "unknown"


def sanitize_metadata(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return None
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if SECRET_KEY_RE.search(key_text):
                clean[key_text] = "[REDACTED]"
                continue
            sanitized = sanitize_metadata(item, depth=depth + 1)
            if sanitized is not None:
                clean[key_text] = sanitized
        return clean
    if isinstance(value, list):
        return [
            item
            for item in (
                sanitize_metadata(item, depth=depth + 1) for item in value[:50]
            )
            if item is not None
        ]
    if isinstance(value, str):
        redacted = redact_text(value)
        if len(redacted) > 2000:
            return redacted[:200]
        return redacted
    if isinstance(value, int | float | bool) or value is None:
        return value
    return redact_text(str(value))[:500]


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _runtime_cap_seconds(provider: str, args: dict[str, Any]) -> int | None:
    if provider == "aws-sagemaker":
        return _int_or_none(args.get("max_run_seconds"))
    hours = _float_or_none(
        args.get("max_run_hours")
        or args.get("timeout_hours")
        or args.get("expected_run_hours")
    )
    if hours is not None:
        return int(hours * 3600)
    timeout = args.get("timeout")
    if isinstance(timeout, int | float):
        return int(float(timeout))
    if isinstance(timeout, str):
        match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$", timeout, re.I)
        if match:
            amount = float(match.group(1))
            unit = (match.group(2) or "s").lower()
            factor = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
            return int(amount * factor)
    return None


def _instance_type(provider: str, args: dict[str, Any]) -> str | None:
    if provider == "hf-jobs":
        return str(args.get("hardware_flavor") or args.get("hardware") or "") or None
    if provider == "gcp-vertex":
        machine = str(args.get("machine_type") or "") or None
        accelerator = str(args.get("accelerator_type") or "") or None
        return " + ".join(part for part in (machine, accelerator) if part) or None
    if provider == "aws-sagemaker":
        return str(args.get("instance_type") or "") or None
    return None


def _instance_count(args: dict[str, Any]) -> int | None:
    return _int_or_none(args.get("instance_count") or args.get("replica_count"))


def _budget_cap(
    provider: str, payload: dict[str, Any], args: dict[str, Any]
) -> float | None:
    explicit = _float_or_none(
        payload.get("budget_cap_usd")
        or payload.get("remaining_cap_usd")
        or args.get("budget_cap_usd")
    )
    if explicit is not None:
        return explicit
    budgets = load_budget_config()
    return budgets.provider_daily_budgets_usd.get(provider) or budgets.daily_budget_usd


def budget_warning_for(provider: str, estimated_cost: float | None) -> str | None:
    budgets = load_budget_config()
    cap = budgets.provider_daily_budgets_usd.get(provider) or budgets.daily_budget_usd
    if cap is None:
        return None
    if estimated_cost is not None and estimated_cost > cap:
        return (
            f"Estimated cost ${estimated_cost:.2f} exceeds configured daily "
            f"budget ${cap:.2f}."
        )
    return None


def no_budget_message(provider: str) -> str | None:
    budgets = load_budget_config()
    if (
        budgets.provider_daily_budgets_usd.get(provider) is None
        and budgets.daily_budget_usd is None
    ):
        return "No budget configured"
    return None


def base_usage_entry(
    *,
    session_id: str,
    run_id: str | None,
    provider: str,
    tool_name: str | None,
    operation: str,
    usage_id: str | None = None,
) -> dict[str, Any]:
    now = utc_now()
    resolved_usage_id = usage_id or str(uuid.uuid4())
    return {
        "_id": resolved_usage_id,
        "usage_id": resolved_usage_id,
        "session_id": session_id,
        "run_id": run_id,
        "provider": normalize_provider(provider, tool_name),
        "tool_name": tool_name,
        "operation": operation,
        "job_id": None,
        "job_url": None,
        "artifact_url": None,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
        "currency": "USD",
        "estimated_cost_usd": None,
        "known_cost_usd": None,
        "cost_source": "unknown",
        "cost_confidence": "unknown",
        "instance_type": None,
        "instance_count": None,
        "max_runtime_seconds": None,
        "actual_runtime_seconds": None,
        "dataset_name": None,
        "model_name": None,
        "output_policy": None,
        "approval_id": None,
        "approved": False,
        "budget_cap_usd": None,
        "quota_status": "unknown",
        "warning": None,
        "error_summary": None,
        "metadata": {},
    }


def usage_from_approval_tool(
    *,
    session_id: str,
    run_id: str | None,
    tool_payload: dict[str, Any],
    event_payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    tool_name = str(tool_payload.get("tool") or "")
    args = (
        tool_payload.get("arguments")
        if isinstance(tool_payload.get("arguments"), dict)
        else {}
    )
    provider = normalize_provider(tool_payload.get("provider"), tool_name)
    operation = tool_operation(tool_name, args)
    approval_id = str(
        tool_payload.get("approval_id")
        or tool_payload.get("tool_call_id")
        or event_payload.get("approval_id")
        or ""
    )
    usage_id = f"{run_id or session_id}:approval:{approval_id or tool_name}"
    estimated_cost = _float_or_none(
        tool_payload.get("estimated_cost_usd")
        or event_payload.get("estimated_cost_usd")
    )
    entry = base_usage_entry(
        session_id=session_id,
        run_id=run_id,
        provider=provider,
        tool_name=tool_name,
        operation=operation,
        usage_id=usage_id,
    )
    entry.update(
        {
            "status": "approval_required",
            "estimated_cost_usd": estimated_cost,
            "cost_source": "approval_estimate"
            if estimated_cost is not None
            else "unknown",
            "cost_confidence": "estimated" if estimated_cost is not None else "unknown",
            "instance_type": _instance_type(provider, args),
            "instance_count": _instance_count(args),
            "max_runtime_seconds": _runtime_cap_seconds(provider, args),
            "dataset_name": args.get("dataset_id")
            or args.get("dataset_name")
            or args.get("train_data"),
            "model_name": args.get("model_id")
            or args.get("base_model")
            or args.get("model_name"),
            "output_policy": args.get("output_policy"),
            "approval_id": approval_id or None,
            "budget_cap_usd": _budget_cap(provider, event_payload, args),
            "warning": budget_warning_for(provider, estimated_cost)
            or no_budget_message(provider),
            "metadata": sanitize_metadata(
                {"approval": tool_payload, "arguments": args}
            ),
        }
    )
    return usage_id, entry


def usage_from_tool_state(
    *,
    session_id: str,
    run_id: str | None,
    payload: dict[str, Any],
    existing: Iterable[dict[str, Any]] = (),
) -> tuple[str, dict[str, Any]] | None:
    tool_name = str(payload.get("tool") or "")
    provider = normalize_provider(payload.get("provider"), tool_name)
    if provider == "unknown" and tool_name not in PROVIDER_BY_TOOL:
        return None
    state = str(payload.get("state") or "update").lower()
    tool_call_id = str(payload.get("tool_call_id") or "")
    job_id = str(
        payload.get("jobName") or payload.get("job_id") or payload.get("jobId") or ""
    )
    usage_id = None
    if tool_call_id:
        approval_id = f"{run_id or session_id}:approval:{tool_call_id}"
        if any(item.get("usage_id") == approval_id for item in existing):
            usage_id = approval_id
    if usage_id is None:
        usage_id = (
            f"{run_id or session_id}:{provider}:{job_id or tool_call_id or state}"
        )
    status = {
        "running": "running",
        "queued": "running",
        "starting": "running",
        "approved": "approved",
        "rejected": "rejected",
        "succeeded": "succeeded",
        "completed": "succeeded",
        "success": "succeeded",
        "failed": "failed",
        "error": "failed",
        "billing_required": "blocked",
        "cancelled": "cancelled",
    }.get(state, state or "updated")
    now = utc_now()
    updates = {
        "usage_id": usage_id,
        "session_id": session_id,
        "run_id": run_id,
        "provider": provider,
        "tool_name": tool_name,
        "operation": "run",
        "job_id": job_id or None,
        "job_url": payload.get("jobUrl"),
        "artifact_url": payload.get("outputDir") or payload.get("s3ModelArtifact"),
        "status": status,
        "updated_at": now,
        "started_at": now if status == "running" else None,
        "completed_at": now if status in TERMINAL_STATES else None,
        "known_cost_usd": None,
        "cost_source": "unknown",
        "cost_confidence": "estimated",
        "output_policy": payload.get("outputPolicy"),
        "approved": True if status in {"approved", "running", "succeeded"} else None,
        "quota_status": "blocked" if status == "blocked" else "unknown",
        "warning": (
            payload.get("reason")
            if status == "blocked"
            else no_budget_message(provider)
        ),
        "error_summary": payload.get("failureReason") or payload.get("reason"),
        "metadata": sanitize_metadata(payload),
    }
    return usage_id, updates


def usage_from_training_recommendation(
    *,
    session_id: str,
    run_id: str,
    recommendation: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    structured = recommendation if isinstance(recommendation, dict) else {}
    details = structured.get("recommendation")
    if not isinstance(details, dict):
        return None
    provider = normalize_provider(structured.get("provider"))
    usage_id = f"{run_id}:planner:training_recommendation"
    selected_hardware = details.get("selected_hardware")
    selected_hardware = selected_hardware if isinstance(selected_hardware, dict) else {}
    hardware_args = selected_hardware.get("hardware_args")
    hardware_args = hardware_args if isinstance(hardware_args, dict) else {}
    selected_model = details.get("selected_model")
    selected_model = selected_model if isinstance(selected_model, dict) else {}
    entry = base_usage_entry(
        session_id=session_id,
        run_id=run_id,
        provider=provider,
        tool_name="training_planner",
        operation="recommend",
        usage_id=usage_id,
    )
    entry.update(
        {
            "status": "estimated",
            "estimated_cost_usd": _float_or_none(details.get("estimated_cost_usd")),
            "cost_source": "static_estimate",
            "cost_confidence": "estimated",
            "instance_type": selected_hardware.get("display_name")
            or _instance_type(provider, hardware_args),
            "instance_count": _instance_count(hardware_args),
            "max_runtime_seconds": _runtime_cap_seconds(provider, hardware_args),
            "model_name": selected_model.get("model_id")
            or structured.get("recommended_model"),
            "output_policy": structured.get("output_policy"),
            "budget_cap_usd": _float_or_none(details.get("budget_cap_usd")),
            "quota_status": "warning"
            if details.get("quota_warning_recorded")
            else "unknown",
            "warning": next(
                (
                    str(warning.get("message"))
                    for warning in details.get("warnings") or []
                    if isinstance(warning, dict) and warning.get("message")
                ),
                None,
            ),
            "metadata": sanitize_metadata(structured),
        }
    )
    return usage_id, entry


def usage_from_run_terminal(
    *,
    run_id: str,
    status: str,
    error_summary: str | None = None,
) -> dict[str, Any]:
    now = utc_now()
    update: dict[str, Any] = {"updated_at": now}
    if status in {"succeeded", "failed", "cancelled", "interrupted"}:
        update["completed_at"] = now
        if status != "succeeded":
            update["status"] = status
            update["error_summary"] = error_summary
    return update


def summarize_usage(entries: list[dict[str, Any]]) -> dict[str, Any]:
    estimated = round(
        sum(float(item.get("estimated_cost_usd") or 0.0) for item in entries), 4
    )
    known = round(sum(float(item.get("known_cost_usd") or 0.0) for item in entries), 4)
    by_provider: dict[str, dict[str, Any]] = {}
    by_session: dict[str, dict[str, Any]] = {}
    by_run: dict[str, dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []
    for item in entries:
        for bucket, key in (
            (by_provider, str(item.get("provider") or "unknown")),
            (by_session, str(item.get("session_id") or "unknown")),
            (by_run, str(item.get("run_id") or "none")),
        ):
            current = bucket.setdefault(
                key,
                {
                    "estimated_cost_usd": 0.0,
                    "known_cost_usd": 0.0,
                    "count": 0,
                },
            )
            current["estimated_cost_usd"] = round(
                current["estimated_cost_usd"]
                + float(item.get("estimated_cost_usd") or 0.0),
                4,
            )
            current["known_cost_usd"] = round(
                current["known_cost_usd"] + float(item.get("known_cost_usd") or 0.0),
                4,
            )
            current["count"] += 1
        if item.get("warning"):
            warnings.append(
                {
                    "usage_id": item.get("usage_id"),
                    "provider": item.get("provider"),
                    "message": item.get("warning"),
                }
            )
        if item.get("quota_status") == "blocked" or item.get("error_summary"):
            warnings.append(
                {
                    "usage_id": item.get("usage_id"),
                    "provider": item.get("provider"),
                    "message": item.get("error_summary") or item.get("quota_status"),
                }
            )
    return {
        "total_estimated_cost_usd": estimated,
        "total_known_cost_usd": known,
        "cost_by_provider": by_provider,
        "cost_by_session": by_session,
        "cost_by_run": by_run,
        "budget_warnings": [
            w
            for w in warnings
            if "budget" in str(w.get("message", "")).lower()
            or "No budget" in str(w.get("message", ""))
        ],
        "quota_warnings": [
            w
            for w in warnings
            if "quota" in str(w.get("message", "")).lower()
            or "blocked" in str(w.get("message", "")).lower()
        ],
    }
