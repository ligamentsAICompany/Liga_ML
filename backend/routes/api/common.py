"""Agent API routes — REST + SSE endpoints.

All routes (except /health) require authentication via the get_current_user
dependency. In dev mode (no OAUTH_CLIENT_ID), auth is bypassed automatically.
"""

# ruff: noqa: F401

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from dependencies import (
    INTERNAL_HF_TOKEN_KEY,
    get_current_user,
)
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from huggingface_hub.errors import HfHubHTTPError
from litellm import Message, acompletion
from pydantic import ValidationError
from starlette.datastructures import FormData, UploadFile
from dataset_uploads import (
    MAX_DATASET_UPLOAD_BYTES,
    dataset_context_note,
    dataset_session_metadata,
    push_dataset_upload_to_hub,
)
from models import (
    AuditEvent,
    AuditStoreHealth,
    AuditSummary,
    AuditTimelineResponse,
    ApprovalRequest,
    DatasetUploadResponse,
    DatasetDiscoveryResponse,
    HealthResponse,
    LLMHealthResponse,
    EvaluationSummary,
    PostTrainingEvaluation,
    RunEventInfo,
    RunSummary,
    SecurityHealth,
    SessionInfo,
    SessionNotificationsRequest,
    SessionResponse,
    SessionYoloRequest,
    SubmitRequest,
    TrainingPreflightRequest,
    TrainingPreflightResultModel,
    TruncateRequest,
    UsageEntry,
    UsageStoreHealth,
    UsageSummary,
)
from responses_log import (
    build_responses_log,
    build_responses_summary,
    filter_response_rows,
    paginate_response_rows,
)
from session_manager import (
    MAX_SESSIONS,
    AgentSession,
    SessionCapacityError,
    session_manager,
)

import user_quotas

from agent.core.aws_readiness import build_aws_sagemaker_readiness_snapshot
from agent.core.audit import (
    audit_events_from_terminal_response_row,
    audit_store_status,
    audit_timeline_enabled,
    build_audit_event,
)
from agent.core.background_runs import (
    RUN_TERMINAL_STATUSES,
    background_run_status,
    background_runs_in_process,
)
from agent.core.gcp_readiness import build_gcp_vertex_readiness_snapshot
from agent.core.hf_access import get_jobs_access
from agent.core.hf_tokens import resolve_hf_request_token, resolve_hf_router_token
from agent.core.llm_params import _resolve_llm_params
from agent.core.model_provider_selection import (
    hardware_catalog,
    model_catalog,
    provider_catalog,
)
from agent.core.post_training_evaluation import build_post_training_evaluation
from agent.core.redact import sanitize_for_frontend
from agent.core.session import Event
from agent.core.session_persistence import session_store_status
from agent.core.training_preflight import (
    run_training_preflight as execute_training_preflight,
)
from agent.core.usage import (
    usage_dashboard_enabled,
    usage_updates_from_terminal_response_row,
)

logger = logging.getLogger(__name__)

# Router split across routes/api/* modules
_background_teardown_tasks: set[asyncio.Task] = set()
_response_sync_tasks: set[asyncio.Task] = set()

DEFAULT_CLAUDE_MODEL_ID = "bedrock/us.anthropic.claude-opus-4-6-v1"
DEFAULT_FREE_MODEL_ID = "moonshotai/Kimi-K2.6"
PREMIUM_MODEL_IDS = {
    DEFAULT_CLAUDE_MODEL_ID,
    "openai/gpt-5.5",
}
DATASET_UPLOAD_MULTIPART_SLACK_BYTES = 1024 * 1024


def _claude_picker_model_id() -> str:
    """Return the model ID used by the Claude option in the UI.

    The app default may be Kimi, so only mirror the resolved config when it is
    actually an Anthropic/Bedrock model. Otherwise keep the Claude picker wired
    to the premium Claude default instead of duplicating the Kimi option.
    """
    configured_model = session_manager.config.model_name
    if "anthropic" in configured_model:
        return configured_model
    return DEFAULT_CLAUDE_MODEL_ID


def _available_models() -> list[dict[str, Any]]:
    models = [
        {
            "id": "moonshotai/Kimi-K2.6",
            "label": "Kimi K2.6",
            "provider": "huggingface",
            "tier": "free",
            "recommended": True,
        },
        {
            "id": _claude_picker_model_id(),
            "label": "Claude Opus 4.6",
            "provider": "anthropic",
            "tier": "pro",
            "recommended": True,
        },
        {
            "id": "openai/gpt-5.5",
            "label": "GPT-5.5",
            "provider": "openai",
            "tier": "pro",
        },
        {
            "id": "MiniMaxAI/MiniMax-M2.7",
            "label": "MiniMax M2.7",
            "provider": "huggingface",
            "tier": "free",
        },
        {
            "id": "zai-org/GLM-5.1",
            "label": "GLM 5.1",
            "provider": "huggingface",
            "tier": "free",
        },
        {
            "id": "deepseek-ai/DeepSeek-V4-Pro:deepinfra",
            "label": "DeepSeek V4 Pro",
            "provider": "huggingface",
            "tier": "free",
        },
    ]
    return models


AVAILABLE_MODELS = _available_models()
VALID_CLOUD_PROVIDERS = {"hf-jobs", "gcp-vertex", "aws-sagemaker"}
VALID_TRAINING_GOALS = {"smoke-test", "production", "agent-decide"}
VALID_OUTPUT_POLICIES = {"cloud-private", "hf-hub", "cloud-and-hf-hub"}
HF_JOB_URL_RE = re.compile(r"https://huggingface\.co/jobs/([^/\s]+)/([^/\s`\"')\]]+)")
TERMINAL_RESPONSE_PROGRESS = {
    "completed",
    "failed",
    "error",
    "cancelled",
    "interrupted",
    "blocked",
}


def _is_premium_model(model_id: str) -> bool:
    return model_id in PREMIUM_MODEL_IDS


def _cloud_provider_or_default(value: Any) -> str:
    if value in VALID_CLOUD_PROVIDERS:
        return str(value)
    return "hf-jobs"


def _training_goal_or_default(value: Any) -> str:
    if value in VALID_TRAINING_GOALS:
        return str(value)
    return "agent-decide"


def _output_policy_or_default(value: Any) -> str:
    if value in VALID_OUTPUT_POLICIES:
        return str(value)
    return "cloud-and-hf-hub"


def _output_policy_for_provider(value: Any, cloud_provider: str) -> str:
    if value in VALID_OUTPUT_POLICIES:
        return str(value)
    if cloud_provider in {"gcp-vertex", "aws-sagemaker"}:
        return "cloud-private"
    return "cloud-and-hf-hub"


async def _model_override_for_new_session(
    request: Request,
    requested_model: str | None,
) -> str | None:
    """Return the model override to use when creating a new session.

    Explicit premium model requests are allowed and charged at message-submit
    time. Implicit default sessions are more forgiving: when the configured
    default is premium, start them on the first free model instead of spending
    premium quota accidentally.
    """
    resolved_model = requested_model or session_manager.config.model_name
    if not _is_premium_model(resolved_model):
        return requested_model
    if requested_model:
        return requested_model

    logger.info(
        "Default premium model %s would spend quota; "
        "creating session with free fallback %s",
        resolved_model,
        DEFAULT_FREE_MODEL_ID,
    )
    return DEFAULT_FREE_MODEL_ID


async def _enforce_premium_model_quota(
    user: dict[str, Any],
    agent_session: AgentSession,
) -> None:
    """Charge the user's daily premium-model quota on first use in a session.

    Runs at *message-submit* time, not session-create time — so spinning up a
    premium-model session to look around doesn't burn quota. The
    ``claude_counted`` flag on ``AgentSession`` guards against re-counting the
    same session; the stored field name is kept for persistence compatibility.

    No-ops when the session's current model isn't premium, or when this
    session has already been charged. Raises 429 when the user has hit
    their daily cap.
    """
    if agent_session.claude_counted:
        return
    model_name = agent_session.session.config.model_name
    if not _is_premium_model(model_name):
        return
    user_id = user["user_id"]
    plan = user.get("plan", "free")
    cap = user_quotas.daily_cap_for(plan)
    new_count = await user_quotas.try_increment_claude(user_id, cap)
    if new_count is None:
        if plan == "pro":
            message = (
                "Daily premium model limit reached. Use a free model and try "
                "premium models again tomorrow."
            )
        else:
            message = (
                "Daily premium model limit reached. Upgrade to HF Pro for "
                f"{user_quotas.CLAUDE_PRO_DAILY}/day or use a free model."
            )
        raise HTTPException(
            status_code=429,
            detail={
                "error": "premium_model_daily_cap",
                "plan": plan,
                "cap": cap,
                "message": message,
            },
        )
    agent_session.claude_counted = True
    await session_manager.persist_session_snapshot(agent_session)


def _user_hf_token(user: dict[str, Any] | None) -> str | None:
    if not isinstance(user, dict):
        return None
    return user.get(INTERNAL_HF_TOKEN_KEY)


def security_health() -> SecurityHealth:
    token_encryption_configured = bool(os.environ.get("SESSION_TOKEN_ENCRYPTION_KEY"))
    return SecurityHealth(
        redaction_enabled=True,
        sandbox_private_default=True,
        secret_persistence_allowed=False,
        token_encryption_configured=token_encryption_configured,
        encrypted_handoff_enabled=token_encryption_configured
        and background_runs_in_process(),
    )


def _reject_oversize_dataset_upload(request: Request) -> None:
    raw_content_length = request.headers.get("content-length")
    if raw_content_length is None:
        return
    try:
        content_length = int(raw_content_length)
    except (TypeError, ValueError):
        return
    if content_length > MAX_DATASET_UPLOAD_BYTES + DATASET_UPLOAD_MULTIPART_SLACK_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Dataset upload exceeds the 100 MB limit.",
        )


def _dataset_upload_file_from_form(form: FormData) -> UploadFile:
    uploaded_files = [
        (key, value)
        for key, value in form.multi_items()
        if isinstance(value, UploadFile)
    ]
    if len(uploaded_files) != 1:
        raise HTTPException(
            status_code=400,
            detail="Upload exactly one dataset file.",
        )
    field_name, upload = uploaded_files[0]
    if field_name != "file":
        raise HTTPException(
            status_code=400,
            detail="Missing 'file' upload field.",
        )
    return upload


def _dataset_upload_hub_http_exception(error: HfHubHTTPError) -> HTTPException:
    status_code = getattr(error.response, "status_code", None)
    if status_code == 401:
        detail = "Hugging Face rejected the token used for the dataset upload."
        return HTTPException(status_code=401, detail=detail)
    if status_code == 403:
        detail = (
            "Hugging Face denied permission to create or write to the dataset repo."
        )
        return HTTPException(status_code=403, detail=detail)
    if status_code == 404:
        detail = "Could not find the Hugging Face namespace or dataset repo."
        return HTTPException(status_code=404, detail=detail)
    if status_code == 429:
        detail = "Hugging Face Hub rate limit reached while uploading the dataset."
        return HTTPException(status_code=429, detail=detail)
    return HTTPException(
        status_code=502,
        detail="Hugging Face Hub upload failed. Please try again.",
    )


def _session_capacity_http_exception(error: SessionCapacityError) -> HTTPException:
    capacity = dict(error.capacity or {})
    return HTTPException(
        status_code=503,
        detail={
            "error": "session_capacity",
            "message": str(error),
            "active_sessions": capacity.get("active_sessions"),
            "max_sessions": capacity.get("max_sessions"),
            "error_type": capacity.get("error_type") or error.error_type,
            "cleanup": capacity.get("cleanup") or {"cleared": 0, "skipped": 0},
        },
    )


async def _check_session_access(
    session_id: str,
    user: dict[str, Any],
    request: Request | None = None,
    preload_sandbox: bool = True,
) -> AgentSession:
    """Verify and lazily load the user's session. Raises 403 or 404."""
    hf_token = (
        resolve_hf_request_token(request)
        if request is not None
        else _user_hf_token(user)
    )
    agent_session = await session_manager.ensure_session_loaded(
        session_id,
        user["user_id"],
        hf_token=hf_token,
        hf_username=user.get("username"),
        preload_sandbox=preload_sandbox,
    )
    if not agent_session:
        raise HTTPException(status_code=404, detail="Session not found")
    if user["user_id"] != "dev" and agent_session.user_id not in {
        user["user_id"],
        "dev",
    }:
        raise HTTPException(status_code=403, detail="Access denied to this session")
    return agent_session




def usage_store_status() -> UsageStoreHealth:
    """Return non-secret usage ledger durability for health/API payloads."""
    store_status = session_store_status(session_manager.persistence_store)
    warning = (
        None
        if store_status["durable"]
        else ("MONGODB_URI is not configured; usage entries are in-memory only.")
    )
    return UsageStoreHealth(
        enabled=usage_dashboard_enabled(),
        durable=bool(store_status["durable"]),
        store=str(store_status["type"]),
        warning=warning,
    )


def audit_store_health() -> AuditStoreHealth:
    """Return non-secret audit timeline durability for health/API payloads."""
    return AuditStoreHealth(
        **audit_store_status(session_store_status(session_manager.persistence_store))
    )


def _safe_limit(limit: int = 100) -> int:
    return max(1, min(int(limit or 100), 500))


def _serialize_usage_entry(entry: dict[str, Any]) -> dict[str, Any]:
    def iso(value: Any) -> str | None:
        return (
            value.isoformat()
            if hasattr(value, "isoformat")
            else (str(value) if value else None)
        )

    return sanitize_for_frontend(
        {
            key: (iso(value) if key.endswith("_at") else value)
            for key, value in entry.items()
            if key not in {"_id", "schema_version"}
        }
    )


def _serialize_audit_event(event: dict[str, Any]) -> dict[str, Any]:
    def iso(value: Any) -> str | None:
        return (
            value.isoformat()
            if hasattr(value, "isoformat")
            else (str(value) if value else None)
        )

    return sanitize_for_frontend(
        {
            key: (
                iso(value)
                if key in {"timestamp", "created_at", "updated_at"}
                else value
            )
            for key, value in event.items()
            if key
            not in {
                "_id",
                "schema_version",
                "idempotency_key",
                "created_at",
                "updated_at",
            }
        }
    )


def _audit_filters(
    *,
    session_id: str | None = None,
    run_id: str | None = None,
    provider: str | None = None,
    category: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "run_id": run_id,
        "provider": provider,
        "category": category,
        "severity": severity,
        "status": status,
        "since": since,
        "until": until,
        "limit": _safe_limit(limit),
    }


def _audit_timeline_response(events: list[dict[str, Any]]) -> AuditTimelineResponse:
    return AuditTimelineResponse(
        enabled=audit_timeline_enabled(),
        audit_store=audit_store_health(),
        events=[AuditEvent(**_serialize_audit_event(event)) for event in events],
    )


def _audit_summary_response(raw: dict[str, Any]) -> AuditSummary:
    events = [
        AuditEvent(**_serialize_audit_event(event)) for event in raw.get("events", [])
    ]
    by_session: dict[str, list[AuditEvent]] = {}
    by_run: dict[str, list[AuditEvent]] = {}
    for event in events:
        by_session.setdefault(event.session_id, []).append(event)
        if event.run_id:
            by_run.setdefault(event.run_id, []).append(event)
    return AuditSummary(
        enabled=audit_timeline_enabled(),
        total_events=raw.get("total_events", 0),
        counts_by_category=raw.get("counts_by_category", {}),
        counts_by_severity=raw.get("counts_by_severity", {}),
        counts_by_provider=raw.get("counts_by_provider", {}),
        latest_warnings_errors=[
            AuditEvent(**_serialize_audit_event(event))
            for event in raw.get("latest_warnings_errors", [])
        ],
        provider_job_timeline=[
            AuditEvent(**_serialize_audit_event(event))
            for event in raw.get("provider_job_timeline", [])
        ],
        approval_timeline=[
            AuditEvent(**_serialize_audit_event(event))
            for event in raw.get("approval_timeline", [])
        ],
        dataset_timeline=[
            AuditEvent(**_serialize_audit_event(event))
            for event in raw.get("dataset_timeline", [])
        ],
        usage_cost_timeline=[
            AuditEvent(**_serialize_audit_event(event))
            for event in raw.get("usage_cost_timeline", [])
        ],
        timeline_by_session=by_session,
        timeline_by_run=by_run,
        audit_store=audit_store_health(),
    )


def _serialize_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    def iso(value: Any) -> str | None:
        return (
            value.isoformat()
            if hasattr(value, "isoformat")
            else (str(value) if value else None)
        )

    return sanitize_for_frontend(
        {
            key: (iso(value) if key.endswith("_at") else value)
            for key, value in evaluation.items()
            if key not in {"_id", "schema_version"}
        }
    )


def _evaluation_summary_response(raw: dict[str, Any]) -> EvaluationSummary:
    evaluations = [
        PostTrainingEvaluation(**_serialize_evaluation(evaluation))
        for evaluation in raw.get("evaluations", [])
    ]
    latest = raw.get("latest_evaluation")
    return EvaluationSummary(
        total_evaluations=raw.get("total_evaluations", 0),
        counts_by_status=raw.get("counts_by_status", {}),
        average_overall_score=raw.get("average_overall_score"),
        latest_evaluation=PostTrainingEvaluation(**_serialize_evaluation(latest))
        if isinstance(latest, dict)
        else None,
        evaluations=evaluations,
    )


def _usage_summary_payload(
    raw: dict[str, Any], provider_readiness: dict[str, Any] | None = None
) -> UsageSummary:
    entries = [_serialize_usage_entry(item) for item in raw.get("entries", [])]
    return UsageSummary(
        total_estimated_cost_usd=raw.get("total_estimated_cost_usd", 0.0),
        total_known_cost_usd=raw.get("total_known_cost_usd", 0.0),
        cost_by_provider=raw.get("cost_by_provider", {}),
        cost_by_session=raw.get("cost_by_session", {}),
        cost_by_run=raw.get("cost_by_run", {}),
        recent_usage_entries=[UsageEntry(**entry) for entry in entries],
        quota_warnings=raw.get("quota_warnings", []),
        budget_warnings=raw.get("budget_warnings", []),
        provider_readiness=provider_readiness or {},
        usage_store=usage_store_status(),
    )










































def _record_value(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _nested_record(record: dict[str, Any], key: str) -> dict[str, Any]:
    value = record.get(key)
    return value if isinstance(value, dict) else {}


def _known_recommendation_value(value: Any) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, str) and value.strip().lower() in {"unknown", "null", "none"}:
        return False
    return True


def _catalog_provider_record(provider_id: Any) -> dict[str, Any]:
    if not _known_recommendation_value(provider_id):
        return {}
    return next(
        (
            provider.to_dict()
            for provider in provider_catalog()
            if provider.provider_id == str(provider_id)
        ),
        {"provider_id": str(provider_id)},
    )


def _catalog_model_record(model_id: Any) -> dict[str, Any]:
    if not _known_recommendation_value(model_id):
        return {}
    return next(
        (
            model.to_dict()
            for model in model_catalog()
            if model.model_id == str(model_id)
        ),
        {"model_id": str(model_id)},
    )


def _hardware_id_from_args(provider_id: Any, hardware_args: Any) -> str | None:
    if not _known_recommendation_value(provider_id) or not isinstance(
        hardware_args, dict
    ):
        return None
    for hardware in hardware_catalog():
        if hardware.provider_id != str(provider_id):
            continue
        if all(
            hardware.hardware_args.get(key) == value
            for key, value in hardware_args.items()
        ):
            return hardware.hardware_id
    return None


def _catalog_hardware_record(hardware_id: Any) -> dict[str, Any]:
    if not _known_recommendation_value(hardware_id):
        return {}
    return next(
        (
            hardware.to_dict()
            for hardware in hardware_catalog()
            if hardware.hardware_id == str(hardware_id)
        ),
        {"hardware_id": str(hardware_id)},
    )


def _recommendation_body(record: dict[str, Any]) -> dict[str, Any]:
    nested = _nested_record(record, "recommendation")
    return nested if nested else record


def _normalize_preflight_recommendation(recommendation: Any) -> dict[str, Any] | None:
    if not isinstance(recommendation, dict) or not recommendation:
        return None
    payload = dict(recommendation)
    body = dict(_recommendation_body(payload))
    provider = _record_value(
        payload,
        "provider",
        "provider_id",
        "cloud_provider",
    ) or _record_value(_nested_record(body, "selected_provider"), "provider_id")
    model_id = _record_value(
        payload,
        "recommended_model",
        "model_id",
    ) or _record_value(_nested_record(body, "selected_model"), "model_id")
    hardware_id = (
        _record_value(payload, "hardware_id")
        or _record_value(_nested_record(body, "selected_hardware"), "hardware_id")
        or _hardware_id_from_args(provider, payload.get("recommended_hardware"))
    )
    output_policy = _record_value(payload, "output_policy") or _record_value(
        body,
        "output_policy",
    )

    if _known_recommendation_value(provider):
        payload["provider"] = str(provider)
        body["selected_provider"] = {
            **_catalog_provider_record(provider),
            **_nested_record(body, "selected_provider"),
            "provider_id": str(provider),
        }
    if _known_recommendation_value(model_id):
        payload["recommended_model"] = str(model_id)
        body["selected_model"] = {
            **_catalog_model_record(model_id),
            **_nested_record(body, "selected_model"),
            "model_id": str(model_id),
        }
    if _known_recommendation_value(hardware_id):
        payload["hardware_id"] = str(hardware_id)
        body["selected_hardware"] = {
            **_catalog_hardware_record(hardware_id),
            **_nested_record(body, "selected_hardware"),
            "hardware_id": str(hardware_id),
        }
    if _known_recommendation_value(output_policy):
        payload["output_policy"] = str(output_policy)
        body["output_policy"] = str(output_policy)

    payload["recommendation"] = body
    return payload


def _has_preflight_recommendation_fields(recommendation: Any) -> bool:
    recommendation = _normalize_preflight_recommendation(recommendation)
    if not isinstance(recommendation, dict) or not recommendation:
        return False
    nested = _nested_record(recommendation, "recommendation")
    source = nested if nested else recommendation
    provider = _record_value(
        recommendation,
        "provider",
        "provider_id",
    ) or _record_value(_nested_record(source, "selected_provider"), "provider_id")
    model_id = _record_value(
        recommendation,
        "recommended_model",
        "model_id",
    ) or _record_value(_nested_record(source, "selected_model"), "model_id")
    hardware_id = _record_value(
        recommendation,
        "hardware_id",
    ) or _record_value(_nested_record(source, "selected_hardware"), "hardware_id")
    output_policy = _record_value(recommendation, "output_policy") or _record_value(
        source,
        "output_policy",
    )
    return all(
        _known_recommendation_value(value)
        for value in (provider, model_id, hardware_id, output_policy)
    )


def _merge_preflight_recommendation(
    primary: Any,
    fallback: Any,
) -> dict[str, Any] | None:
    normalized_primary = _normalize_preflight_recommendation(primary)
    normalized_fallback = _normalize_preflight_recommendation(fallback)
    if normalized_primary is None:
        return normalized_fallback
    if normalized_fallback is None:
        return normalized_primary

    merged = {**normalized_fallback, **normalized_primary}
    primary_body = _nested_record(normalized_primary, "recommendation")
    fallback_body = _nested_record(normalized_fallback, "recommendation")
    merged_body = {**fallback_body, **primary_body}

    for selected_key, id_key in (
        ("selected_provider", "provider_id"),
        ("selected_model", "model_id"),
        ("selected_hardware", "hardware_id"),
    ):
        primary_selected = _nested_record(primary_body, selected_key)
        fallback_selected = _nested_record(fallback_body, selected_key)
        primary_id = _record_value(primary_selected, id_key)
        if _known_recommendation_value(primary_id):
            merged_body[selected_key] = {**fallback_selected, **primary_selected}
        elif fallback_selected:
            merged_body[selected_key] = fallback_selected

    for key in ("provider", "recommended_model", "hardware_id", "output_policy"):
        if not _known_recommendation_value(merged.get(key)):
            fallback_value = normalized_fallback.get(key)
            if _known_recommendation_value(fallback_value):
                merged[key] = fallback_value
    for key in ("training_goal", "domain", "task_type"):
        if not _known_recommendation_value(merged.get(key)):
            fallback_value = normalized_fallback.get(key)
            if _known_recommendation_value(fallback_value):
                merged[key] = fallback_value
        if not _known_recommendation_value(merged_body.get(key)):
            fallback_value = fallback_body.get(key)
            if _known_recommendation_value(fallback_value):
                merged_body[key] = fallback_value
    if not _known_recommendation_value(merged_body.get("output_policy")):
        fallback_value = fallback_body.get("output_policy")
        if _known_recommendation_value(fallback_value):
            merged_body["output_policy"] = fallback_value

    merged["recommendation"] = merged_body
    return _normalize_preflight_recommendation(merged)


async def _resolve_preflight_recommendation(
    request: TrainingPreflightRequest,
    agent_session: AgentSession,
) -> dict[str, Any] | None:
    request_recommendation = _normalize_preflight_recommendation(request.recommendation)
    session_recommendation = getattr(
        getattr(agent_session, "session", None),
        "latest_training_recommendation",
        None,
    )
    session_recommendation = (
        sanitize_for_frontend(session_recommendation)
        if isinstance(session_recommendation, dict)
        else None
    )
    if _has_preflight_recommendation_fields(request_recommendation):
        if session_recommendation:
            return _merge_preflight_recommendation(
                request_recommendation,
                session_recommendation,
            )
        return request_recommendation

    resolved = _merge_preflight_recommendation(
        request_recommendation,
        session_recommendation,
    )
    if _has_preflight_recommendation_fields(resolved):
        return resolved

    if request.run_id:
        run_recommendation = await session_manager.get_run_training_recommendation(
            request.session_id,
            request.run_id,
        )
        resolved = _merge_preflight_recommendation(resolved, run_recommendation)
        if _has_preflight_recommendation_fields(resolved):
            return resolved

    stored_recommendation = await session_manager.get_latest_training_recommendation(
        request.session_id
    )
    resolved = _merge_preflight_recommendation(resolved, stored_recommendation)
    return resolved


















def _active_response_sessions(user_id: str) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    active_sessions = getattr(session_manager, "sessions", {})
    for agent_session in active_sessions.values():
        if user_id != "dev" and agent_session.user_id not in {user_id, "dev"}:
            continue
        sessions.append(
            {
                "session_id": agent_session.session_id,
                "title": agent_session.title,
                "model": agent_session.session.config.model_name,
                "user_id": agent_session.user_id,
                "cloud_provider": agent_session.cloud_provider,
                "training_goal": agent_session.training_goal,
                "output_policy": agent_session.output_policy,
                "created_at": agent_session.created_at,
            }
        )
    return sessions


async def _sync_response_rows(
    user_id: str, *, include_persisted_sessions: bool = False
) -> list[dict[str, Any]]:
    if include_persisted_sessions:
        sessions = await session_manager.list_sessions(user_id=user_id)
    else:
        sessions = _active_response_sessions(user_id)
    response_log = await build_responses_log(
        sessions,
        load_events=session_manager.load_response_events,
    )
    store = session_manager.persistence_store
    if getattr(store, "enabled", False) and hasattr(store, "upsert_response_rows"):
        await store.upsert_response_rows(response_log["rows"], user_id=user_id)
        await _sync_runs_from_terminal_response_rows(response_log["rows"])
    return response_log["rows"]


def _stale_response_session_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("session_id"))
        for row in rows
        if row.get("platform") in {"hf-jobs", "gcp-vertex", "aws-sagemaker"}
        and str(row.get("progress") or "").lower() not in TERMINAL_RESPONSE_PROGRESS
        and row.get("session_id")
        and row.get("job_id")
    }


def _hf_job_identity(row: dict[str, Any]) -> tuple[str | None, str | None]:
    raw_job = str(row.get("job_id") or row.get("final_artifact_or_result") or "")
    match = HF_JOB_URL_RE.search(raw_job)
    if match:
        return match.group(2).rstrip("/"), match.group(1)
    if raw_job and "/" not in raw_job and not raw_job.startswith("http"):
        return raw_job.rstrip("/"), None
    return None, None


def _normalize_hf_job_progress(stage: Any) -> str:
    text = str(stage or "").strip().lower()
    if text.startswith("job_state_"):
        text = text.removeprefix("job_state_")
    if text in {"completed", "complete", "succeeded", "success", "done", "finished"}:
        return "completed"
    if text in {"failed", "failure", "expired"}:
        return "failed"
    if text in {"error", "errored"}:
        return "error"
    if text in {"canceled", "cancelled", "cancelling"}:
        return "cancelled"
    if text in {"scheduling", "scheduled", "pending"}:
        return "queued"
    if text in {"running", "queued"}:
        return text
    return text or "unknown"


def _gcp_job_identity(row: dict[str, Any]) -> str | None:
    raw_job = str(row.get("job_id") or "").strip()
    if re.fullmatch(r"projects/[^/]+/locations/[^/]+/customJobs/[^/]+", raw_job):
        return raw_job
    return None


def _gcp_job_location(job_name: str) -> str | None:
    match = re.fullmatch(
        r"projects/[^/]+/locations/(?P<location>[^/]+)/customJobs/[^/]+", job_name
    )
    return match.group("location") if match else None


def _gcp_job_console_url(job_name: str) -> str | None:
    match = re.fullmatch(
        r"projects/(?P<project>[^/]+)/locations/(?P<location>[^/]+)/customJobs/(?P<job_id>[^/]+)",
        job_name,
    )
    if not match:
        return None
    return (
        "https://console.cloud.google.com/vertex-ai/training/custom-jobs/"
        f"locations/{match.group('location')}/customJobs/{match.group('job_id')}"
        f"?project={match.group('project')}"
    )


def _gcp_describe_value(job_info: Any, key: str) -> Any:
    if isinstance(job_info, dict):
        return job_info.get(key)
    return getattr(job_info, key, None)


def _gcp_state_text(state: Any) -> str:
    text = str(getattr(state, "name", None) or state or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _iso_from_describe_time(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value if value.tzinfo else value.replace(tzinfo=UTC)
        return parsed.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _gcp_described_response_row(
    row: dict[str, Any], job_info: Any, *, job_name: str
) -> dict[str, Any]:
    state = _gcp_state_text(_gcp_describe_value(job_info, "state"))
    progress = _normalize_hf_job_progress(state)
    output_dir = (
        _gcp_describe_value(job_info, "output_dir")
        or _gcp_describe_value(job_info, "gcs_output_dir")
        or _gcp_describe_value(job_info, "artifact_uri")
        or row.get("final_artifact_or_result")
    )
    job_url = _gcp_describe_value(job_info, "job_url") or _gcp_job_console_url(job_name)
    completed_at = (
        _iso_from_describe_time(_gcp_describe_value(job_info, "end_time"))
        or _iso_from_describe_time(_gcp_describe_value(job_info, "update_time"))
        or _iso_from_describe_time(_gcp_describe_value(job_info, "updateTime"))
    )
    updated = dict(row)
    updated["progress"] = progress
    updated["job_id"] = job_name
    if output_dir:
        updated["final_artifact_or_result"] = str(output_dir)
    if progress in TERMINAL_RESPONSE_PROGRESS:
        updated["completed_at"] = (
            row.get("completed_at") or completed_at or datetime.now(UTC).isoformat()
        )
    updated["provider_metadata"] = {
        **dict(row.get("provider_metadata") or {}),
        "tool": "gcp_vertex_jobs",
        "state": state or progress,
        "refreshed_from": "gcp_vertex_describe",
    }
    if job_url:
        updated["provider_metadata"]["jobUrl"] = job_url
    return updated


async def _describe_gcp_vertex_job(job_name: str) -> Any:
    location = _gcp_job_location(job_name)
    if not location:
        raise ValueError(
            "Vertex job name must include project, location, and custom job id."
        )
    from google.cloud import aiplatform_v1

    client = aiplatform_v1.JobServiceAsyncClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )
    job = await client.get_custom_job(name=job_name)
    output_dir = ""
    try:
        output_dir = job.job_spec.base_output_directory.output_uri_prefix
    except AttributeError:
        output_dir = ""
    return {
        "state": _gcp_state_text(job.state),
        "update_time": getattr(job, "update_time", None),
        "end_time": getattr(job, "end_time", None),
        "output_dir": output_dir,
        "job_url": _gcp_job_console_url(job_name),
    }


def _completed_at_from_hf_job(job_info: Any) -> str:
    for attr in ("updated_at", "last_modified", "created_at"):
        value = getattr(job_info, attr, None)
        if isinstance(value, datetime):
            parsed = value if value.tzinfo else value.replace(tzinfo=UTC)
            return parsed.isoformat()
    return datetime.now(UTC).isoformat()


def _hub_only_artifact(value: Any) -> bool:
    text = str(value or "")
    return text.startswith("https://huggingface.co/") and not text.startswith(
        (
            "https://huggingface.co/jobs/",
            "https://huggingface.co/datasets/",
            "https://huggingface.co/spaces/",
        )
    )


def _hf_inspected_response_row(row: dict[str, Any], job_info: Any) -> dict[str, Any]:
    status = getattr(job_info, "status", None)
    stage = getattr(status, "stage", None)
    progress = _normalize_hf_job_progress(stage)
    updated = dict(row)
    updated["progress"] = progress
    updated["provider_metadata"] = {
        **dict(row.get("provider_metadata") or {}),
        "tool": "hf_jobs",
        "state": str(stage or progress).upper(),
        "refreshed_from": "hf_jobs_inspect",
    }
    if progress in TERMINAL_RESPONSE_PROGRESS:
        updated["completed_at"] = row.get("completed_at") or _completed_at_from_hf_job(
            job_info
        )
    if _hub_only_artifact(row.get("final_artifact_or_result")):
        updated["result_storage"] = "hf-hub"
        if "smoke" in str(row.get("final_artifact_or_result") or "").lower():
            updated["run_type"] = "smoke-test"
    return updated


async def _inspect_hf_job_status(job_id: str, namespace: str | None) -> Any:
    from huggingface_hub import HfApi

    kwargs: dict[str, Any] = {"job_id": job_id}
    if namespace:
        kwargs["namespace"] = namespace
    return await asyncio.to_thread(HfApi().inspect_job, **kwargs)


async def _refresh_stale_hf_rows_from_hub(
    rows: list[dict[str, Any]],
    *,
    user_id: str,
    inspect_job: Any = None,
) -> bool:
    stale_hf_rows = [
        row
        for row in rows
        if row.get("platform") == "hf-jobs"
        and str(row.get("progress") or "").lower() not in TERMINAL_RESPONSE_PROGRESS
        and row.get("job_id")
    ]
    if not stale_hf_rows:
        return False
    inspector = inspect_job or _inspect_hf_job_status
    refreshed: list[dict[str, Any]] = []
    for row in stale_hf_rows:
        job_id, namespace = _hf_job_identity(row)
        if not job_id:
            continue
        try:
            job_info = await inspector(job_id, namespace)
        except Exception as e:
            logger.debug("HF Jobs stale row refresh failed for %s: %s", job_id, e)
            continue
        updated = _hf_inspected_response_row(row, job_info)
        if updated.get("progress") != row.get("progress") or updated.get(
            "completed_at"
        ) != row.get("completed_at"):
            refreshed.append(updated)
    if not refreshed:
        return False
    store = session_manager.persistence_store
    if getattr(store, "enabled", False) and hasattr(store, "upsert_response_rows"):
        await store.upsert_response_rows(refreshed, user_id=user_id)
        await _sync_runs_from_terminal_response_rows(refreshed)
        return True
    return False


def _provider_state_from_response_row(row: dict[str, Any]) -> str:
    row_metadata = dict(row.get("provider_metadata") or {})
    state = str(row_metadata.get("state") or row_metadata.get("provider_state") or "")
    if state:
        return state
    progress = str(row.get("progress") or "").lower()
    if progress == "completed":
        return "JOB_STATE_SUCCEEDED"
    if progress in {"failed", "error"}:
        return "JOB_STATE_FAILED"
    if progress in {"cancelled", "canceled"}:
        return "JOB_STATE_CANCELLED"
    if progress == "interrupted":
        return "JOB_STATE_CANCELLED"
    return ""


async def _resolve_run_id_for_terminal_row(
    store: Any,
    *,
    session_id: str,
    job_id: str,
) -> str | None:
    if not job_id or not hasattr(store, "list_runs"):
        return None
    runs = await store.list_runs(session_id)
    for run in runs:
        if not isinstance(run, dict):
            continue
        active_job = str(run.get("active_provider_job_id") or "")
        if active_job and (
            active_job == job_id
            or active_job.endswith(job_id)
            or job_id.endswith(active_job)
        ):
            return str(run.get("run_id") or run.get("_id") or "") or None
    return None


async def _sync_usage_and_audit_from_terminal_response_rows(
    rows: list[dict[str, Any]],
) -> bool:
    """Update provider usage entries and audit events for terminal response rows."""
    store = session_manager.persistence_store
    if not getattr(store, "enabled", False):
        return False
    updated_any = False
    for row in rows:
        progress = str(row.get("progress") or "").lower()
        if progress not in TERMINAL_RESPONSE_PROGRESS:
            continue
        session_id = str(row.get("session_id") or "")
        job_id = str(row.get("job_id") or "")
        if not session_id:
            continue
        run_id = await _resolve_run_id_for_terminal_row(
            store, session_id=session_id, job_id=job_id
        )
        if hasattr(store, "list_usage_entries"):
            existing = await store.list_usage_entries(
                session_id=session_id,
                run_id=run_id,
                limit=50,
            )
            for usage_id, fields in usage_updates_from_terminal_response_row(
                session_id=session_id,
                run_id=run_id,
                row=row,
                existing=existing,
            ):
                await store.upsert_usage_entry(usage_id, fields)
                updated_any = True
        if audit_timeline_enabled() and hasattr(store, "record_audit_event"):
            events = audit_events_from_terminal_response_row(
                session_id=session_id,
                run_id=run_id,
                row=row,
            )
            if events:
                existing_audit = await store.list_audit_events(
                    session_id=session_id, limit=500
                )
                existing_keys = {
                    str(item.get("idempotency_key") or item.get("audit_id") or "")
                    for item in existing_audit
                }
                for event in events:
                    key = str(
                        event.get("idempotency_key") or event.get("audit_id") or ""
                    )
                    if key in existing_keys:
                        continue
                    await store.record_audit_event(event)
                    existing_keys.add(key)
                    updated_any = True
    return updated_any


async def _sync_runs_from_terminal_response_rows(
    rows: list[dict[str, Any]],
) -> bool:
    """Align durable run records with terminal provider response rows."""

    progress_to_run_status = {
        "completed": "succeeded",
        "succeeded": "succeeded",
        "success": "succeeded",
        "failed": "failed",
        "error": "failed",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "interrupted": "interrupted",
        "blocked": "failed",
    }
    updated_any = False
    store = session_manager.persistence_store
    if not getattr(store, "enabled", False) or not hasattr(store, "update_run"):
        return False
    for row in rows:
        progress = str(row.get("progress") or "").lower()
        if progress not in TERMINAL_RESPONSE_PROGRESS:
            continue
        session_id = str(row.get("session_id") or "")
        job_id = str(row.get("job_id") or "")
        if not session_id:
            continue
        target_status = progress_to_run_status.get(progress)
        if not target_status:
            continue
        runs = await store.list_runs(session_id)
        for run in runs:
            if not isinstance(run, dict):
                continue
            run_id = str(run.get("run_id") or run.get("_id") or "")
            if not run_id:
                continue
            active_job = str(run.get("active_provider_job_id") or "")
            if job_id and active_job:
                if not (
                    active_job == job_id
                    or active_job.endswith(job_id)
                    or job_id.endswith(active_job)
                ):
                    continue
            elif (
                job_id and not active_job and run.get("status") in RUN_TERMINAL_STATUSES
            ):
                continue
            provider_metadata = dict(run.get("provider_metadata") or {})
            current_provider_status = str(
                provider_metadata.get("provider_status")
                or provider_metadata.get("status")
                or ""
            ).lower()
            if (
                str(run.get("status") or "") == target_status
                and run.get("completed_at")
                and current_provider_status == progress
            ):
                continue
            provider_state = _provider_state_from_response_row(row)
            row_metadata = dict(row.get("provider_metadata") or {})
            provider_metadata.update(
                {
                    "provider_status": progress,
                    "status": progress,
                    "provider_state": provider_state or row_metadata.get("state"),
                    "active_provider_job_id": job_id or active_job or None,
                    "last_checked_at": datetime.now(UTC).isoformat(),
                    "refreshed_from": "response_row_terminal_sync",
                }
            )
            if row_metadata.get("jobUrl"):
                provider_metadata["provider_console_url"] = row_metadata.get("jobUrl")
            if row.get("error"):
                provider_metadata["failure_reason"] = row.get("error")
            if row.get("final_artifact_or_result"):
                provider_metadata["provider_artifact_path"] = row.get(
                    "final_artifact_or_result"
                )
                provider_metadata["artifact_path"] = row.get("final_artifact_or_result")
            await store.update_run(
                run_id,
                status=target_status,
                completed_at=row.get("completed_at") or datetime.now(UTC).isoformat(),
                provider_metadata=provider_metadata,
                error_summary=str(row.get("error") or "")[:500] or None,
                active_provider_job_id=job_id or active_job or None,
                result_summary=f"provider_{progress}",
            )
            updated_any = True
            break
    usage_audit_updated = await _sync_usage_and_audit_from_terminal_response_rows(rows)
    return updated_any or usage_audit_updated


async def _refresh_stale_gcp_rows_from_vertex(
    rows: list[dict[str, Any]],
    *,
    user_id: str,
    describe_job: Any = None,
) -> bool:
    stale_gcp_rows = [
        row
        for row in rows
        if row.get("platform") == "gcp-vertex"
        and str(row.get("progress") or "").lower() not in TERMINAL_RESPONSE_PROGRESS
        and row.get("job_id")
    ]
    if not stale_gcp_rows:
        return False
    describer = describe_job or _describe_gcp_vertex_job
    refreshed: list[dict[str, Any]] = []
    for row in stale_gcp_rows:
        job_name = _gcp_job_identity(row)
        if not job_name:
            continue
        try:
            job_info = await describer(job_name)
        except Exception as e:
            logger.debug("Vertex stale row refresh failed for %s: %s", job_name, e)
            continue
        updated = _gcp_described_response_row(row, job_info, job_name=job_name)
        if updated.get("progress") != row.get("progress") or updated.get(
            "completed_at"
        ) != row.get("completed_at"):
            refreshed.append(updated)
    if not refreshed:
        return False
    store = session_manager.persistence_store
    if getattr(store, "enabled", False) and hasattr(store, "upsert_response_rows"):
        await store.upsert_response_rows(refreshed, user_id=user_id)
        await _sync_runs_from_terminal_response_rows(refreshed)
        return True
    return False


async def _refresh_stale_response_rows(
    rows: list[dict[str, Any]],
    *,
    user_id: str,
) -> bool:
    refreshed = False
    if await _refresh_stale_hf_rows_from_hub(rows, user_id=user_id):
        refreshed = True
    if await _refresh_stale_gcp_rows_from_vertex(rows, user_id=user_id):
        refreshed = True
    return refreshed


async def _refresh_response_rows_for_evaluations(user_id: str) -> None:
    store = session_manager.persistence_store
    if not (getattr(store, "enabled", False) and hasattr(store, "list_response_rows")):
        return
    response_page = await store.list_response_rows(
        user_id=user_id,
        page=1,
        page_size=200,
    )
    rows = response_page.get("rows", [])
    stale_session_ids = _stale_response_session_ids(rows)
    if stale_session_ids:
        await _sync_response_sessions(user_id, stale_session_ids)
        response_page = await store.list_response_rows(
            user_id=user_id,
            page=1,
            page_size=200,
        )
        rows = response_page.get("rows", [])
    if await _refresh_stale_response_rows(rows, user_id=user_id):
        response_page = await store.list_response_rows(
            user_id=user_id,
            page=1,
            page_size=200,
        )
        rows = response_page.get("rows", [])
    await _sync_runs_from_terminal_response_rows(rows)


async def _sync_response_sessions(user_id: str, session_ids: set[str]) -> None:
    if not session_ids:
        return
    sessions = await session_manager.list_sessions(user_id=user_id)
    selected = [
        session
        for session in sessions
        if str(session.get("session_id") or "") in session_ids
    ]
    if not selected:
        return
    response_log = await build_responses_log(
        selected,
        load_events=session_manager.load_response_events,
    )
    store = session_manager.persistence_store
    if getattr(store, "enabled", False) and hasattr(store, "upsert_response_rows"):
        await store.upsert_response_rows(response_log["rows"], user_id=user_id)
        await _sync_runs_from_terminal_response_rows(response_log["rows"])


def _schedule_response_sync(user_id: str) -> None:
    task = asyncio.create_task(_sync_response_rows(user_id))
    _response_sync_tasks.add(task)
    task.add_done_callback(_response_sync_tasks.discard)








_TITLE_STRIP_CHARS = str.maketrans("", "", "`*_~#[]()")
















































# ---------------------------------------------------------------------------
# Shared SSE helpers
# ---------------------------------------------------------------------------
_TERMINAL_EVENTS = {
    "turn_complete",
    "approval_required",
    "error",
    "stream_error",
    "interrupted",
    "shutdown",
}
_SSE_KEEPALIVE_SECONDS = 15


def _last_event_seq(request: Request) -> int:
    raw = (
        request.headers.get("last-event-id") or request.query_params.get("after") or "0"
    )
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _format_sse(msg: dict[str, Any]) -> str:
    seq = msg.get("seq")
    body = {"event_type": msg.get("event_type"), "data": msg.get("data") or {}}
    if seq is not None:
        body["seq"] = seq
        return f"id: {seq}\ndata: {json.dumps(body)}\n\n"
    return f"data: {json.dumps(body)}\n\n"


def _event_doc_to_msg(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": doc.get("event_type"),
        "data": doc.get("data") or doc.get("payload") or {},
        "seq": doc.get("seq"),
    }


def _sse_response(
    broadcaster,
    event_queue,
    sub_id,
    *,
    request: Request | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    stream_started_at: float | None = None,
    replay_events: list[dict[str, Any]] | None = None,
    after_seq: int = 0,
) -> StreamingResponse:
    """Build a StreamingResponse that drains *event_queue* as SSE,
    sending keepalive comments every 15 s to prevent proxy timeouts."""

    async def event_generator():
        try:
            for doc in replay_events or []:
                msg = _event_doc_to_msg(doc)
                seq = msg.get("seq")
                if isinstance(seq, int) and seq <= after_seq:
                    continue
                yield _format_sse(msg)
                if msg.get("event_type", "") in _TERMINAL_EVENTS:
                    return

            while True:
                try:
                    msg = await asyncio.wait_for(
                        event_queue.get(), timeout=_SSE_KEEPALIVE_SECONDS
                    )
                except asyncio.TimeoutError:
                    if request is not None and await request.is_disconnected():
                        logger.info(
                            "chat_stream_event request_id=%s session_id=%s "
                            "event_type=client_disconnected duration_ms=%s",
                            request_id,
                            session_id,
                            int(
                                (
                                    time.monotonic()
                                    - (stream_started_at or time.monotonic())
                                )
                                * 1000
                            ),
                        )
                        break
                    yield _format_sse(
                        {
                            "event_type": "heartbeat",
                            "data": {
                                "request_id": request_id,
                                "session_id": session_id,
                            },
                        }
                    )
                    continue
                event_type = msg.get("event_type", "")
                data = msg.get("data") or {}
                safe_data = {
                    "request_id": data.get("request_id") or request_id,
                    "session_id": data.get("session_id") or session_id,
                }
                if event_type in {"tool_call", "tool_output"}:
                    safe_data["tool"] = data.get("tool")
                logger.info(
                    "chat_stream_event request_id=%s session_id=%s event_type=%s "
                    "cloud_provider=%s selected_model=%s duration_ms=%s",
                    safe_data.get("request_id"),
                    safe_data.get("session_id"),
                    event_type,
                    data.get("cloud_provider"),
                    data.get("model"),
                    int(
                        (time.monotonic() - (stream_started_at or time.monotonic()))
                        * 1000
                    ),
                )
                yield _format_sse(msg)
                if event_type in _TERMINAL_EVENTS:
                    break
        except Exception as e:
            logger.exception(
                "chat_stream_event request_id=%s session_id=%s event_type=stream_error",
                request_id,
                session_id,
            )
            yield _format_sse(
                {
                    "event_type": "stream_error",
                    "data": {
                        "error": str(e),
                        "request_id": request_id,
                        "session_id": session_id,
                    },
                }
            )
        finally:
            broadcaster.unsubscribe(sub_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )




















