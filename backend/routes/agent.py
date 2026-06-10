"""Agent API routes — REST + SSE endpoints.

All routes (except /health) require authentication via the get_current_user
dependency. In dev mode (no OAUTH_CLIENT_ID), auth is bypassed automatically.
"""

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
    audit_store_status,
    audit_timeline_enabled,
    build_audit_event,
)
from agent.core.background_runs import background_run_status, background_runs_in_process
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
from agent.core.usage import usage_dashboard_enabled

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["agent"])
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


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    store = session_manager.persistence_store
    store_status = session_store_status(store)
    return HealthResponse(
        status="ok",
        active_sessions=session_manager.active_session_count,
        max_sessions=MAX_SESSIONS,
        session_store=store_status,
        background_runs=background_run_status(store_status),
        usage_store=usage_store_status(),
        audit_store=audit_store_health(),
        security=security_health(),
        cloud_run_revision=os.environ.get("K_REVISION"),
    )


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


@router.get("/health/llm", response_model=LLMHealthResponse)
async def llm_health_check() -> LLMHealthResponse:
    """Check if the LLM provider is reachable and the API key is valid.

    Makes a minimal 1-token completion call.  Catches common errors:
    - 401 → invalid API key
    - 402/insufficient_quota → out of credits
    - 429 → rate limited
    - timeout / network → provider unreachable
    """
    model = session_manager.config.model_name
    try:
        llm_params = _resolve_llm_params(model, reasoning_effort="high")
        await acompletion(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            timeout=10,
            **llm_params,
        )
        return LLMHealthResponse(status="ok", model=model)
    except Exception as e:
        err_str = str(e).lower()
        error_type = "unknown"

        if (
            "401" in err_str
            or "auth" in err_str
            or "invalid" in err_str
            or "api key" in err_str
        ):
            error_type = "auth"
        elif (
            "402" in err_str
            or "credit" in err_str
            or "quota" in err_str
            or "insufficient" in err_str
            or "billing" in err_str
            or "spending limit" in err_str
            or "monthly spending" in err_str
        ):
            error_type = "quota"
        elif "429" in err_str or "rate" in err_str:
            error_type = "rate_limit"
        elif "timeout" in err_str or "connect" in err_str or "network" in err_str:
            error_type = "network"

        logger.warning(f"LLM health check failed ({error_type}): {e}")
        return LLMHealthResponse(
            status="error",
            model=model,
            error=str(e)[:500],
            error_type=error_type,
        )


@router.get("/health/providers")
async def provider_health() -> dict[str, Any]:
    """Return non-secret readiness for training providers."""
    hf_token_configured = bool(
        os.environ.get("HF_TOKEN") or os.environ.get("HF_ADMIN_TOKEN")
    )
    return {
        "hf_jobs": {
            "configured": hf_token_configured,
            "hf_token_configured": hf_token_configured,
            "notes": []
            if hf_token_configured
            else ["HF_TOKEN or user OAuth token is required to run HF Jobs."],
        },
        "gcp_vertex": build_gcp_vertex_readiness_snapshot(),
        "aws_sagemaker": build_aws_sagemaker_readiness_snapshot(),
        "session_store": session_store_status(session_manager.persistence_store),
        "audit_store": audit_store_health().model_dump(),
        "security": security_health().model_dump(),
    }


@router.get("/model-catalog")
async def get_model_catalog(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Return the static planner model catalog."""
    _ = user
    return {
        "models": [model.to_dict() for model in model_catalog()],
        "live_access_probed": False,
    }


@router.get("/provider-catalog")
async def get_provider_catalog(
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Return the static planner provider catalog."""
    _ = user
    return {
        "providers": [provider.to_dict() for provider in provider_catalog()],
        "readiness": await provider_health(),
        "live_quota_api_used": False,
    }


@router.get("/hardware-catalog")
async def get_hardware_catalog(
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Return the static planner hardware catalog."""
    _ = user
    return {
        "hardware": [hardware.to_dict() for hardware in hardware_catalog()],
        "live_availability_probed": False,
    }


@router.get("/usage", response_model=list[UsageEntry])
async def list_usage(
    provider: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    status: str | None = None,
    limit: int = 100,
    user: dict = Depends(get_current_user),
) -> list[UsageEntry]:
    """List non-secret usage ledger entries."""
    if session_id:
        await _check_session_access(session_id, user, preload_sandbox=False)
    entries = await session_manager.list_usage_entries(
        provider=provider,
        session_id=session_id,
        run_id=run_id,
        status=status,
        limit=_safe_limit(limit),
    )
    return [UsageEntry(**_serialize_usage_entry(entry)) for entry in entries]


@router.get("/usage/summary", response_model=UsageSummary)
async def usage_summary(
    provider: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    status: str | None = None,
    limit: int = 100,
    user: dict = Depends(get_current_user),
) -> UsageSummary:
    """Summarize estimated/known usage without live billing API calls."""
    if session_id:
        await _check_session_access(session_id, user, preload_sandbox=False)
    raw = await session_manager.usage_summary(
        provider=provider,
        session_id=session_id,
        run_id=run_id,
        status=status,
        limit=_safe_limit(limit),
    )
    return _usage_summary_payload(raw, provider_readiness=await provider_health())


@router.get("/usage/providers")
async def usage_providers(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Provider readiness and usage-store durability for the dashboard."""
    _ = user
    summary = await session_manager.usage_summary(limit=500)
    return {
        "enabled": usage_dashboard_enabled(),
        "usage_store": usage_store_status().model_dump(),
        "provider_readiness": await provider_health(),
        "cost_by_provider": summary.get("cost_by_provider", {}),
        "no_live_billing_api_configured": True,
        "notes": [
            "Estimated cost, not final bill",
            "Actual provider billing may differ",
            "Quota status may be unknown unless provider reports it",
        ],
    }


@router.get("/audit", response_model=AuditTimelineResponse)
async def list_audit(
    session_id: str | None = None,
    run_id: str | None = None,
    provider: str | None = None,
    category: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    limit: int = 100,
    since: str | None = None,
    until: str | None = None,
    user: dict = Depends(get_current_user),
) -> AuditTimelineResponse:
    """List sanitized audit events for the internal timeline."""
    if session_id:
        await _check_session_access(session_id, user, preload_sandbox=False)
    if not audit_timeline_enabled():
        return _audit_timeline_response([])
    events = await session_manager.list_audit_events(
        **_audit_filters(
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
    )
    return _audit_timeline_response(events)


@router.get("/audit/summary", response_model=AuditSummary)
async def audit_summary(
    session_id: str | None = None,
    run_id: str | None = None,
    provider: str | None = None,
    category: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    limit: int = 100,
    since: str | None = None,
    until: str | None = None,
    user: dict = Depends(get_current_user),
) -> AuditSummary:
    """Summarize sanitized audit events by category, severity, and provider."""
    if session_id:
        await _check_session_access(session_id, user, preload_sandbox=False)
    if not audit_timeline_enabled():
        return _audit_summary_response({"events": [], "total_events": 0})
    raw = await session_manager.audit_summary(
        **_audit_filters(
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
    )
    return _audit_summary_response(raw)


@router.get("/audit/providers")
async def audit_providers(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Provider readiness and audit-store durability for the timeline UI."""
    _ = user
    raw = (
        await session_manager.audit_summary(limit=500)
        if audit_timeline_enabled()
        else {"counts_by_provider": {}}
    )
    return {
        "enabled": audit_timeline_enabled(),
        "audit_store": audit_store_health().model_dump(),
        "provider_readiness": await provider_health(),
        "counts_by_provider": raw.get("counts_by_provider", {}),
        "notes": [
            "Internal audit timeline only",
            "No external observability exporter configured",
            "Sensitive metadata is redacted before persistence",
        ],
    }


@router.get("/evaluations", response_model=list[PostTrainingEvaluation])
async def list_evaluations(
    session_id: str | None = None,
    run_id: str | None = None,
    provider: str | None = None,
    status: str | None = None,
    limit: int = 100,
    user: dict = Depends(get_current_user),
) -> list[PostTrainingEvaluation]:
    """List safe static post-training evaluations."""
    if session_id:
        await _check_session_access(session_id, user, preload_sandbox=False)
    evaluations = await session_manager.list_evaluations(
        session_id=session_id,
        run_id=run_id,
        provider=provider,
        status=status,
        limit=_safe_limit(limit),
    )
    return [
        PostTrainingEvaluation(**_serialize_evaluation(evaluation))
        for evaluation in evaluations
    ]


@router.get("/evaluations/summary", response_model=EvaluationSummary)
async def evaluations_summary(
    session_id: str | None = None,
    run_id: str | None = None,
    provider: str | None = None,
    status: str | None = None,
    limit: int = 100,
    user: dict = Depends(get_current_user),
) -> EvaluationSummary:
    if session_id:
        await _check_session_access(session_id, user, preload_sandbox=False)
    raw = await session_manager.evaluation_summary(
        session_id=session_id,
        run_id=run_id,
        provider=provider,
        status=status,
        limit=_safe_limit(limit),
    )
    return _evaluation_summary_response(raw)


@router.get(
    "/session/{session_id}/evaluations", response_model=list[PostTrainingEvaluation]
)
async def list_session_evaluations(
    session_id: str,
    user: dict = Depends(get_current_user),
) -> list[PostTrainingEvaluation]:
    await _check_session_access(session_id, user, preload_sandbox=False)
    evaluations = await session_manager.list_evaluations(
        session_id=session_id, limit=500
    )
    return [
        PostTrainingEvaluation(**_serialize_evaluation(evaluation))
        for evaluation in evaluations
    ]


@router.get(
    "/session/{session_id}/runs/{run_id}/evaluation",
    response_model=PostTrainingEvaluation,
)
async def get_run_evaluation(
    session_id: str,
    run_id: str,
    user: dict = Depends(get_current_user),
) -> PostTrainingEvaluation:
    await _check_session_access(session_id, user, preload_sandbox=False)
    evaluation = await session_manager.get_evaluation_for_run(session_id, run_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return PostTrainingEvaluation(**_serialize_evaluation(evaluation))


@router.get("/session/{session_id}/runs/{run_id}/evaluation/report")
async def get_run_evaluation_report(
    session_id: str,
    run_id: str,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    evaluation = await get_run_evaluation(session_id, run_id, user)
    return {
        "evaluation_id": evaluation.evaluation_id,
        "status": evaluation.status,
        "report_markdown": evaluation.report_markdown or "",
    }


@router.get(
    "/session/{session_id}/dataset-discovery",
    response_model=DatasetDiscoveryResponse,
)
async def get_session_dataset_discovery(
    session_id: str,
    user: dict = Depends(get_current_user),
) -> DatasetDiscoveryResponse:
    await _check_session_access(session_id, user, preload_sandbox=False)
    discovery = await session_manager.get_latest_dataset_discovery(session_id)
    if not discovery:
        raise HTTPException(status_code=404, detail="Dataset discovery not found")
    return DatasetDiscoveryResponse(**discovery)


@router.get("/session/{session_id}/recommendations")
async def get_session_recommendations(
    session_id: str,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    await _check_session_access(session_id, user, preload_sandbox=False)
    recommendation = await session_manager.get_latest_training_recommendation(
        session_id
    )
    if not recommendation:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return recommendation


@router.get(
    "/session/{session_id}/runs/{run_id}/dataset-discovery",
    response_model=DatasetDiscoveryResponse,
)
async def get_run_dataset_discovery(
    session_id: str,
    run_id: str,
    user: dict = Depends(get_current_user),
) -> DatasetDiscoveryResponse:
    await _check_session_access(session_id, user, preload_sandbox=False)
    discovery = await session_manager.get_run_dataset_discovery(session_id, run_id)
    if not discovery:
        raise HTTPException(status_code=404, detail="Dataset discovery not found")
    return DatasetDiscoveryResponse(**discovery)


@router.get("/session/{session_id}/runs/{run_id}/recommendations")
async def get_run_recommendations(
    session_id: str,
    run_id: str,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    await _check_session_access(session_id, user, preload_sandbox=False)
    recommendation = await session_manager.get_run_training_recommendation(
        session_id, run_id
    )
    if not recommendation:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return recommendation


@router.post("/training-preflight", response_model=TrainingPreflightResultModel)
async def run_training_preflight(
    request: TrainingPreflightRequest,
    http_request: Request = None,
    user: dict = Depends(get_current_user),
) -> TrainingPreflightResultModel:
    """Run training preflight without launching provider jobs."""

    agent_session = await _check_session_access(
        request.session_id,
        user,
        preload_sandbox=False,
    )
    recommendation = request.recommendation
    if recommendation is None:
        session_recommendation = getattr(
            getattr(agent_session, "session", None),
            "latest_training_recommendation",
            None,
        )
        recommendation = (
            sanitize_for_frontend(session_recommendation)
            if isinstance(session_recommendation, dict)
            else None
        )
    if recommendation is None:
        recommendation = await session_manager.get_latest_training_recommendation(
            request.session_id
        )
    dataset_discovery = await session_manager.get_latest_dataset_discovery(
        request.session_id
    )
    hf_token = (
        resolve_hf_request_token(http_request)
        if http_request is not None
        else (getattr(agent_session, "hf_token", None) or _user_hf_token(user))
    )
    gcp_project_id = (
        request.metadata.get("gcp_project_id")
        or request.metadata.get("project_id")
        or request.metadata.get("google_cloud_project")
    )
    gcp_region = (
        request.metadata.get("gcp_region")
        or request.metadata.get("region")
        or request.metadata.get("google_cloud_region")
        or request.metadata.get("location")
    )
    aws_region = (
        request.metadata.get("aws_region")
        or request.metadata.get("region")
        or request.metadata.get("aws_default_region")
    )
    aws_execution_role_arn = (
        request.metadata.get("aws_execution_role_arn")
        or request.metadata.get("execution_role_arn")
        or request.metadata.get("sagemaker_role_arn")
        or request.metadata.get("role_arn")
    )
    result = await execute_training_preflight(
        session_id=request.session_id,
        run_id=request.run_id,
        recommendation=recommendation,
        dataset_summary=request.dataset_summary,
        dataset_discovery=dataset_discovery,
        target_namespace=request.target_namespace,
        target_repo_id=request.target_repo_id,
        target_bucket=request.target_bucket,
        include_fallbacks=request.include_fallbacks,
        force_refresh=request.force_refresh,
        timeout_seconds=request.timeout_seconds,
        metadata={
            **request.metadata,
            "agent_session_active": bool(getattr(agent_session, "is_active", False)),
        },
        allow_unknown_override=request.allow_unknown_override,
        hf_token=hf_token,
        gcp_project_id=str(gcp_project_id) if gcp_project_id else None,
        gcp_region=str(gcp_region) if gcp_region else None,
        aws_region=str(aws_region) if aws_region else None,
        aws_execution_role_arn=str(aws_execution_role_arn)
        if aws_execution_role_arn
        else None,
    )
    saved = await session_manager.record_training_preflight(
        session_id=request.session_id,
        run_id=request.run_id,
        preflight=result.to_dict(),
    )
    return TrainingPreflightResultModel(**saved)


@router.get(
    "/session/{session_id}/preflight", response_model=TrainingPreflightResultModel
)
async def get_session_preflight(
    session_id: str,
    user: dict = Depends(get_current_user),
) -> TrainingPreflightResultModel:
    await _check_session_access(session_id, user, preload_sandbox=False)
    preflight = await session_manager.get_latest_training_preflight(session_id)
    if not preflight:
        raise HTTPException(status_code=404, detail="Training preflight not found")
    return TrainingPreflightResultModel(**preflight)


@router.get(
    "/session/{session_id}/runs/{run_id}/preflight",
    response_model=TrainingPreflightResultModel,
)
async def get_run_preflight(
    session_id: str,
    run_id: str,
    user: dict = Depends(get_current_user),
) -> TrainingPreflightResultModel:
    await _check_session_access(session_id, user, preload_sandbox=False)
    preflight = await session_manager.get_run_training_preflight(session_id, run_id)
    if not preflight:
        raise HTTPException(status_code=404, detail="Training preflight not found")
    return TrainingPreflightResultModel(**preflight)


@router.post(
    "/session/{session_id}/runs/{run_id}/evaluation",
    response_model=PostTrainingEvaluation,
)
async def trigger_run_evaluation(
    session_id: str,
    run_id: str,
    user: dict = Depends(get_current_user),
) -> PostTrainingEvaluation:
    """Idempotently create a static evaluation without paid inference."""
    await _check_session_access(session_id, user, preload_sandbox=False)
    existing = await session_manager.get_evaluation_for_run(session_id, run_id)
    if existing:
        return PostTrainingEvaluation(**_serialize_evaluation(existing))
    run = await session_manager.get_run(session_id, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    provider_metadata = (
        run.get("provider_metadata")
        if isinstance(run.get("provider_metadata"), dict)
        else {}
    )
    artifact_ref = (
        provider_metadata.get("artifact_path")
        or run.get("provider_artifact_path")
        or run.get("result_summary")
    )
    evaluation = build_post_training_evaluation(
        {
            "session_id": session_id,
            "run_id": run_id,
            "provider": run.get("provider"),
            "job_id": run.get("active_provider_job_id"),
            "model_ref": run.get("result_summary"),
            "artifact_ref": artifact_ref,
            "dataset_ref": provider_metadata.get("dataset_name"),
            "training_status": run.get("status"),
            "metadata": {
                "manual_trigger": True,
                "mode": "static",
                "provider_metadata": provider_metadata,
                "dataset_discovery": run.get("dataset_discovery"),
            },
        }
    )
    saved = await session_manager.upsert_evaluation(evaluation)
    return PostTrainingEvaluation(**_serialize_evaluation(saved))


@router.get("/session/{session_id}/audit", response_model=AuditTimelineResponse)
async def list_session_audit(
    session_id: str,
    provider: str | None = None,
    category: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    limit: int = 100,
    since: str | None = None,
    until: str | None = None,
    user: dict = Depends(get_current_user),
) -> AuditTimelineResponse:
    await _check_session_access(session_id, user, preload_sandbox=False)
    if not audit_timeline_enabled():
        return _audit_timeline_response([])
    events = await session_manager.list_audit_events(
        **_audit_filters(
            session_id=session_id,
            provider=provider,
            category=category,
            severity=severity,
            status=status,
            since=since,
            until=until,
            limit=limit,
        )
    )
    return _audit_timeline_response(events)


@router.get(
    "/session/{session_id}/runs/{run_id}/audit", response_model=AuditTimelineResponse
)
async def list_run_audit(
    session_id: str,
    run_id: str,
    provider: str | None = None,
    category: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    limit: int = 100,
    since: str | None = None,
    until: str | None = None,
    user: dict = Depends(get_current_user),
) -> AuditTimelineResponse:
    await _check_session_access(session_id, user, preload_sandbox=False)
    run = await session_manager.get_run(session_id, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if not audit_timeline_enabled():
        return _audit_timeline_response([])
    events = await session_manager.list_audit_events(
        **_audit_filters(
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
    )
    return _audit_timeline_response(events)


@router.get("/session/{session_id}/usage", response_model=list[UsageEntry])
async def list_session_usage(
    session_id: str,
    provider: str | None = None,
    status: str | None = None,
    limit: int = 100,
    user: dict = Depends(get_current_user),
) -> list[UsageEntry]:
    await _check_session_access(session_id, user, preload_sandbox=False)
    entries = await session_manager.list_usage_entries(
        provider=provider,
        session_id=session_id,
        status=status,
        limit=_safe_limit(limit),
    )
    return [UsageEntry(**_serialize_usage_entry(entry)) for entry in entries]


@router.get(
    "/session/{session_id}/runs/{run_id}/usage", response_model=list[UsageEntry]
)
async def list_run_usage(
    session_id: str,
    run_id: str,
    provider: str | None = None,
    status: str | None = None,
    limit: int = 100,
    user: dict = Depends(get_current_user),
) -> list[UsageEntry]:
    await _check_session_access(session_id, user, preload_sandbox=False)
    run = await session_manager.get_run(session_id, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    entries = await session_manager.list_usage_entries(
        provider=provider,
        session_id=session_id,
        run_id=run_id,
        status=status,
        limit=_safe_limit(limit),
    )
    return [UsageEntry(**_serialize_usage_entry(entry)) for entry in entries]


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
        return True
    return False


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


def _schedule_response_sync(user_id: str) -> None:
    task = asyncio.create_task(_sync_response_rows(user_id))
    _response_sync_tasks.add(task)
    task.add_done_callback(_response_sync_tasks.discard)


@router.get("/responses")
async def get_responses(
    page: int = 1,
    page_size: int = 50,
    platform: str | None = None,
    progress: str | None = None,
    model: str | None = None,
    session_id: str | None = None,
    job_id: str | None = None,
    q: str | None = None,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Return paginated fine-tuning/job response rows."""
    store = session_manager.persistence_store
    if getattr(store, "enabled", False) and hasattr(store, "list_response_rows"):
        summary = await store.get_response_summary(user_id=user["user_id"])
        if summary.get("has_rows"):
            _schedule_response_sync(user["user_id"])
        else:
            await _sync_response_rows(
                user["user_id"],
                include_persisted_sessions=True,
            )
    else:
        rows = await _sync_response_rows(
            user["user_id"], include_persisted_sessions=True
        )
    filters = {
        "platform": platform,
        "progress": progress,
        "model": model,
        "session_id": session_id,
        "job_id": job_id,
        "q": q,
        "page": page,
        "page_size": page_size,
    }
    if getattr(store, "enabled", False) and hasattr(store, "list_response_rows"):
        response_page = await store.list_response_rows(
            user_id=user["user_id"], **filters
        )
        stale_session_ids = _stale_response_session_ids(response_page.get("rows", []))
        if stale_session_ids:
            if platform or progress or session_id or job_id or q:
                await _sync_response_sessions(user["user_id"], stale_session_ids)
                response_page = await store.list_response_rows(
                    user_id=user["user_id"], **filters
                )
                if await _refresh_stale_hf_rows_from_hub(
                    response_page.get("rows", []), user_id=user["user_id"]
                ):
                    response_page = await store.list_response_rows(
                        user_id=user["user_id"], **filters
                    )
            else:
                await _sync_response_sessions(user["user_id"], stale_session_ids)
                response_page = await store.list_response_rows(
                    user_id=user["user_id"], **filters
                )
                if await _refresh_stale_hf_rows_from_hub(
                    response_page.get("rows", []), user_id=user["user_id"]
                ):
                    response_page = await store.list_response_rows(
                        user_id=user["user_id"], **filters
                    )
        return response_page
    return paginate_response_rows(
        filter_response_rows(rows, **filters),
        page=page,
        page_size=page_size,
    )


@router.get("/responses/summary")
async def get_responses_summary(
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Return summary metadata for the Responses button."""
    store = session_manager.persistence_store
    if getattr(store, "enabled", False) and hasattr(store, "get_response_summary"):
        return await store.get_response_summary(user_id=user["user_id"])
    rows = await _sync_response_rows(user["user_id"])
    total_responses = 0
    for row in rows:
        total_responses = max(
            total_responses, int(row.get("actual_sequence_number") or 0)
        )
    return build_responses_summary(
        rows,
        total_responses=total_responses,
        durable=False,
        store_type="memory",
    )


@router.get("/config/model")
async def get_model() -> dict:
    """Get current model and available models. No auth required."""
    return {
        "current": session_manager.config.model_name,
        "available": AVAILABLE_MODELS,
    }


_TITLE_STRIP_CHARS = str.maketrans("", "", "`*_~#[]()")


@router.post("/title")
async def generate_title(
    request: SubmitRequest, user: dict = Depends(get_current_user)
) -> dict:
    """Generate a short title for a chat session based on the first user message.

    Always uses gpt-oss-120b via Cerebras on the HF router. The tab headline
    renders as plain text, so the model is told to avoid markdown and any
    stray formatting characters are stripped before returning. gpt-oss is a
    reasoning model — reasoning_effort=low keeps the reasoning budget small
    so the 60-token output budget isn't consumed before the title is written.
    """
    api_key = resolve_hf_router_token(_user_hf_token(user))
    try:
        response = await acompletion(
            # Double openai/ prefix: LiteLLM strips the first as its provider
            # prefix, leaving the HF model id on the wire for the router.
            model="openai/openai/gpt-oss-120b:cerebras",
            api_base="https://router.huggingface.co/v1",
            api_key=api_key,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a very short title (max 6 words) for a chat conversation "
                        "that starts with the following user message. "
                        "Reply with ONLY the title in plain text. "
                        "Do NOT use markdown, backticks, asterisks, quotes, brackets, or any "
                        "formatting characters. No punctuation at the end."
                    ),
                },
                {"role": "user", "content": request.text[:500]},
            ],
            max_tokens=60,
            temperature=0.3,
            timeout=10,
            reasoning_effort="low",
        )
        title = response.choices[0].message.content.strip().strip('"').strip("'")
        title = title.translate(_TITLE_STRIP_CHARS).strip()
        if len(title) > 50:
            title = title[:50].rstrip() + "…"
        try:
            await _check_session_access(request.session_id, user)
            await session_manager.update_session_title(request.session_id, title)
        except Exception:
            logger.debug(
                "Skipping title persistence for missing session %s", request.session_id
            )
        return {"title": title}
    except Exception as e:
        logger.warning(f"Title generation failed: {e}")
        fallback = request.text.strip()
        title = fallback[:40].rstrip() + "…" if len(fallback) > 40 else fallback
        try:
            await _check_session_access(request.session_id, user)
            await session_manager.update_session_title(request.session_id, title)
        except Exception:
            logger.debug(
                "Skipping fallback title persistence for missing session %s",
                request.session_id,
            )
        return {"title": title}


@router.post("/session", response_model=SessionResponse)
async def create_session(
    request: Request, user: dict = Depends(get_current_user)
) -> SessionResponse:
    """Create a new agent session bound to the authenticated user.

    The user's HF access token is extracted from the Authorization header
    and stored in the session so that tools (e.g. hf_jobs) can act on
    behalf of the user.

    Optional body ``{"model"?: <id>}`` selects the session's LLM; unknown
    ids are rejected (400). The premium-model quota runs at message-submit
    time, not here — spinning up a session to look around is free.

    Returns 503 if the server or user has reached the session limit.
    """
    # Extract the user's HF token (Bearer header, HttpOnly cookie, or env var)
    hf_token = resolve_hf_request_token(request)

    # Optional model override. Empty body falls back to the config default.
    model: str | None = None
    cloud_provider = "hf-jobs"
    training_goal = "agent-decide"
    output_policy = "cloud-and-hf-hub"
    try:
        body = await request.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        model = body.get("model")
        cloud_provider = _cloud_provider_or_default(body.get("cloud_provider"))
        training_goal = _training_goal_or_default(body.get("training_goal"))
        output_policy = _output_policy_for_provider(
            body.get("output_policy"), cloud_provider
        )

    valid_ids = {m["id"] for m in AVAILABLE_MODELS}
    if model and model not in valid_ids:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    # Explicit premium selections are allowed. If the implicit configured
    # default is premium, start the session on a free model instead.
    model = await _model_override_for_new_session(request, model)

    try:
        session_id = await session_manager.create_session(
            user_id=user["user_id"],
            hf_username=user.get("username"),
            hf_token=hf_token,
            model=model,
            is_pro=user.get("plan") == "pro",
            cloud_provider=cloud_provider,
            training_goal=training_goal,
            output_policy=output_policy,
        )
    except SessionCapacityError as e:
        raise _session_capacity_http_exception(e)

    return SessionResponse(
        session_id=session_id,
        ready=True,
        model=model or session_manager.config.model_name,
        cloud_provider=cloud_provider,
        training_goal=training_goal,
        output_policy=output_policy,
    )


@router.post("/session/restore-summary", response_model=SessionResponse)
async def restore_session_summary(
    request: Request, body: dict, user: dict = Depends(get_current_user)
) -> SessionResponse:
    """Create a new session seeded with a summary of the caller's prior
    conversation. The client sends its cached messages; we run the standard
    summarization prompt on them and drop the result into the new
    session's context as a user-role system note.

    Optional ``"model"`` in the body overrides the session's LLM. The
    premium-model quota runs at message-submit time, not here.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="Missing 'messages' array")

    hf_token = resolve_hf_request_token(request)

    model = body.get("model")
    cloud_provider = _cloud_provider_or_default(body.get("cloud_provider"))
    training_goal = _training_goal_or_default(body.get("training_goal"))
    output_policy = _output_policy_for_provider(
        body.get("output_policy"), cloud_provider
    )
    valid_ids = {m["id"] for m in AVAILABLE_MODELS}
    if model and model not in valid_ids:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    model = await _model_override_for_new_session(request, model)

    try:
        session_id = await session_manager.create_session(
            user_id=user["user_id"],
            hf_username=user.get("username"),
            hf_token=hf_token,
            model=model,
            is_pro=user.get("plan") == "pro",
            cloud_provider=cloud_provider,
            training_goal=training_goal,
            output_policy=output_policy,
        )
    except SessionCapacityError as e:
        raise _session_capacity_http_exception(e)

    try:
        summarized = await session_manager.seed_from_summary(session_id, messages)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("seed_from_summary failed")
        raise HTTPException(status_code=500, detail=f"Summary failed: {e}")

    logger.info(
        f"Seeded session {session_id} for {user.get('username', 'unknown')} "
        f"(summary of {summarized} messages)"
    )
    return SessionResponse(
        session_id=session_id,
        ready=True,
        model=model or session_manager.config.model_name,
        cloud_provider=cloud_provider,
        training_goal=training_goal,
        output_policy=output_policy,
    )


@router.post("/session/cleanup-stale")
async def cleanup_stale_sessions(user: dict = Depends(get_current_user)) -> dict:
    """Clear old idle runtime sessions for the current user."""
    return await session_manager.cleanup_stale_sessions(user_id=user["user_id"])


@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> SessionInfo:
    """Get session information. Only accessible by the session owner."""
    await _check_session_access(session_id, user, preload_sandbox=False)
    info = session_manager.get_session_info(session_id)
    info["runs"] = await session_manager.list_runs(session_id)
    return SessionInfo(**info)


@router.post("/session/{session_id}/model")
async def set_session_model(
    session_id: str,
    body: dict,
    request: Request,
    user: dict = Depends(get_current_user),
) -> dict:
    """Switch the active model for a single session (tab-scoped).

    Takes effect on the next LLM call in that session — other sessions
    (including other browser tabs) are unaffected. Model switches don't
    charge quota — the premium-model quota only fires at message-submit time.
    """
    agent_session = await _check_session_access(session_id, user, request)
    model_id = body.get("model")
    if not model_id:
        raise HTTPException(status_code=400, detail="Missing 'model' field")
    valid_ids = {m["id"] for m in AVAILABLE_MODELS}
    if model_id not in valid_ids:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")
    if not agent_session:
        raise HTTPException(status_code=404, detail="Session not found")
    await session_manager.update_session_model(session_id, model_id)
    logger.info(
        f"Session {session_id} model → {model_id} "
        f"(by {user.get('username', 'unknown')})"
    )
    return {"session_id": session_id, "model": model_id}


@router.post("/session/{session_id}/cloud-provider")
async def set_session_cloud_provider(
    session_id: str,
    body: dict,
    request: Request,
    user: dict = Depends(get_current_user),
) -> dict:
    """Switch the active training provider for a single session."""
    await _check_session_access(session_id, user, request)
    cloud_provider = body.get("cloud_provider")
    if cloud_provider not in VALID_CLOUD_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unknown cloud provider")
    training_goal = _training_goal_or_default(body.get("training_goal"))
    output_policy = _output_policy_for_provider(
        body.get("output_policy"), cloud_provider
    )
    success = await session_manager.update_session_cloud_provider(
        session_id, cloud_provider, training_goal, output_policy
    )
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    logger.info(
        f"Session {session_id} cloud provider → {cloud_provider} "
        f"(by {user.get('username', 'unknown')})"
    )
    return {
        "session_id": session_id,
        "cloud_provider": cloud_provider,
        "training_goal": training_goal,
        "output_policy": output_policy,
    }


@router.post("/session/{session_id}/notifications")
async def set_session_notifications(
    session_id: str,
    body: SessionNotificationsRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """Replace the session's auto-notification destinations."""
    agent_session = await _check_session_access(session_id, user)
    try:
        destinations = session_manager.set_notification_destinations(
            session_id, body.destinations
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    await session_manager.persist_session_snapshot(agent_session)
    return {
        "session_id": session_id,
        "notification_destinations": destinations,
    }


@router.post("/session/{session_id}/datasets", response_model=DatasetUploadResponse)
async def upload_session_dataset(
    session_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> DatasetUploadResponse:
    """Upload a CSV/JSON dataset file to a private Hub dataset for this session."""
    file: UploadFile | None = None
    filename_for_audit: str | None = None
    try:
        _reject_oversize_dataset_upload(request)
        agent_session = await _check_session_access(session_id, user, request)
        if not agent_session or not agent_session.is_active:
            raise HTTPException(status_code=404, detail="Session not found")
        if agent_session.is_processing:
            raise HTTPException(
                status_code=409,
                detail="Cannot upload a dataset while the agent is processing.",
            )
        if agent_session.session.pending_approval:
            raise HTTPException(
                status_code=409,
                detail="Approve or reject pending tools before uploading a dataset.",
            )

        hf_token = (
            resolve_hf_request_token(request, include_env_fallback=False)
            or _user_hf_token(user)
            or resolve_hf_request_token(request)
        )
        if not hf_token:
            raise HTTPException(
                status_code=401,
                detail="A Hugging Face token is required to upload datasets.",
            )

        form = await request.form(
            max_files=1,
            max_fields=1,
            max_part_size=MAX_DATASET_UPLOAD_BYTES,
        )
        file = _dataset_upload_file_from_form(form)
        filename_for_audit = file.filename
        await session_manager.record_audit_event(
            build_audit_event(
                session_id=session_id,
                event_type="dataset_upload_started",
                category="dataset",
                status="started",
                actor="user",
                title="Dataset upload started",
                message=f"Dataset upload started for {filename_for_audit}.",
                provider="hf-jobs",
                entity_type="dataset_upload",
                entity_id=filename_for_audit,
                dataset_name=filename_for_audit,
            )
        )
        hf_username = user.get("username") or agent_session.hf_username
        uploaded = await push_dataset_upload_to_hub(
            upload=file,
            session_id=session_id,
            hf_username=hf_username,
            hf_token=hf_token,
        )
        agent_session.session.context_manager.add_message(
            Message(role="user", content=dataset_context_note(uploaded))
        )
        if not hasattr(agent_session.session, "uploaded_datasets"):
            agent_session.session.uploaded_datasets = []
        agent_session.session.uploaded_datasets.append(
            dataset_session_metadata(uploaded)
        )
        await session_manager.persist_session_snapshot(agent_session)
        logger.info(
            "Uploaded dataset file %s to %s for session %s",
            uploaded.filename,
            uploaded.repo_id,
            session_id,
        )
        await session_manager.record_audit_event(
            build_audit_event(
                session_id=session_id,
                event_type="dataset_upload_succeeded",
                category="dataset",
                status="succeeded",
                actor="system",
                title="Dataset uploaded",
                message=f"Dataset {uploaded.filename} uploaded and normalized.",
                provider="hf-jobs",
                entity_type="dataset_upload",
                entity_id=uploaded.upload_id,
                dataset_name=uploaded.filename,
                artifact_url=uploaded.hub_url,
                safe_metadata={
                    "repo_id": uploaded.repo_id,
                    "normalized_row_count": uploaded.normalized_row_count,
                    "source_format": uploaded.source_format,
                    "size_bytes": uploaded.size_bytes,
                },
            )
        )
        return DatasetUploadResponse(**uploaded.response_payload())
    except HTTPException as e:
        await session_manager.record_audit_event(
            build_audit_event(
                session_id=session_id,
                event_type="dataset_upload_failed",
                category="dataset",
                severity="warning" if e.status_code < 500 else "error",
                status="failed",
                actor="system",
                title="Dataset upload failed",
                message=str(e.detail),
                provider="hf-jobs",
                entity_type="dataset_upload",
                entity_id=filename_for_audit,
                dataset_name=filename_for_audit,
                error_code=str(e.status_code),
                error_summary=str(e.detail)[:500],
            )
        )
        raise
    except HfHubHTTPError as e:
        logger.warning(
            "Hub rejected dataset upload for session %s: status=%s request_id=%s",
            session_id,
            getattr(e.response, "status_code", None),
            getattr(e, "request_id", None),
        )
        await session_manager.record_audit_event(
            build_audit_event(
                session_id=session_id,
                event_type="dataset_upload_failed",
                category="dataset",
                severity="error",
                status="failed",
                actor="provider",
                title="Dataset upload failed",
                message="Hugging Face Hub rejected the dataset upload.",
                provider="hf-jobs",
                entity_type="dataset_upload",
                entity_id=filename_for_audit,
                dataset_name=filename_for_audit,
                error_code=str(getattr(e.response, "status_code", "") or "hub_error"),
                error_summary=str(e)[:500],
            )
        )
        raise _dataset_upload_hub_http_exception(e)
    except Exception as e:
        logger.exception("Dataset upload failed for session %s", session_id)
        await session_manager.record_audit_event(
            build_audit_event(
                session_id=session_id,
                event_type="dataset_upload_failed",
                category="dataset",
                severity="error",
                status="failed",
                actor="system",
                title="Dataset upload failed",
                message="Dataset upload failed before it could be attached.",
                provider="hf-jobs",
                entity_type="dataset_upload",
                entity_id=filename_for_audit,
                dataset_name=filename_for_audit,
                error_code=type(e).__name__,
                error_summary=str(e)[:500],
            )
        )
        raise HTTPException(
            status_code=502,
            detail="Dataset upload failed. Please try again.",
        )
    finally:
        if file is not None:
            await file.close()


@router.patch("/session/{session_id}/yolo")
async def set_session_yolo(
    session_id: str,
    body: SessionYoloRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """Update the session-scoped auto-approval policy."""
    await _check_session_access(session_id, user)
    try:
        summary = await session_manager.update_session_auto_approval(
            session_id,
            enabled=body.enabled,
            cost_cap_usd=body.cost_cap_usd,
            cap_provided="cost_cap_usd" in body.model_fields_set,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"session_id": session_id, **summary}


@router.get("/user/quota")
async def get_user_quota(user: dict = Depends(get_current_user)) -> dict:
    """Return the user's plan tier and today's premium-model quota state."""
    plan = user.get("plan", "free")
    used = await user_quotas.get_claude_used_today(user["user_id"])
    cap = user_quotas.daily_cap_for(plan)
    remaining = max(0, cap - used)
    return {
        "plan": plan,
        "premium_used_today": used,
        "premium_daily_cap": cap,
        "premium_remaining": remaining,
    }


@router.get("/user/jobs-access")
async def get_jobs_access_info(
    request: Request, user: dict = Depends(get_current_user)
) -> dict:
    """Return the namespaces the current token can run HF Jobs under.

    Credits are enforced by the HF API at job-creation time, not here —
    the response only describes which wallets the caller is allowed to
    pick from. Pro is irrelevant.
    """
    token = resolve_hf_request_token(request)

    access = await get_jobs_access(token or "")
    return {
        "eligible_namespaces": access.eligible_namespaces if access else [],
        "default_namespace": access.default_namespace if access else None,
        "billing_url": "https://huggingface.co/settings/billing",
    }


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions(user: dict = Depends(get_current_user)) -> list[SessionInfo]:
    """List sessions belonging to the authenticated user."""
    sessions = await session_manager.list_sessions(user_id=user["user_id"])
    for session in sessions:
        session["runs"] = await session_manager.list_runs(session["session_id"])
    return [SessionInfo(**s) for s in sessions]


@router.post("/session/{session_id}/runs", response_model=RunSummary)
async def create_session_run(
    session_id: str,
    body: dict[str, Any] | None = None,
    user: dict = Depends(get_current_user),
) -> RunSummary:
    """Create a durable run record without launching provider work."""
    await _check_session_access(session_id, user)
    payload = body or {}
    run = await session_manager.create_run(
        session_id,
        provider=str(payload.get("provider") or "none"),
        request_id=str(payload.get("request_id") or uuid.uuid4()),
    )
    if not run:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return RunSummary(**run)


@router.get("/session/{session_id}/runs", response_model=list[RunSummary])
async def list_session_runs(
    session_id: str, user: dict = Depends(get_current_user)
) -> list[RunSummary]:
    await _check_session_access(session_id, user, preload_sandbox=False)
    return [RunSummary(**run) for run in await session_manager.list_runs(session_id)]


@router.get("/session/{session_id}/runs/{run_id}", response_model=RunSummary)
async def get_session_run(
    session_id: str, run_id: str, user: dict = Depends(get_current_user)
) -> RunSummary:
    await _check_session_access(session_id, user, preload_sandbox=False)
    run = await session_manager.get_run(session_id, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunSummary(**run)


@router.get(
    "/session/{session_id}/runs/{run_id}/events",
    response_model=list[RunEventInfo],
)
async def get_session_run_events(
    session_id: str,
    run_id: str,
    since: int = 0,
    user: dict = Depends(get_current_user),
) -> list[RunEventInfo]:
    await _check_session_access(session_id, user, preload_sandbox=False)
    events = await session_manager.load_run_events_after(session_id, run_id, since)
    if events is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return [RunEventInfo(**event) for event in events]


@router.post("/session/{session_id}/sandbox/teardown")
async def teardown_session_sandbox(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Best-effort sandbox teardown that preserves durable chat history."""
    await _check_session_access(session_id, user, preload_sandbox=False)
    task = asyncio.create_task(session_manager.teardown_sandbox(session_id))
    _background_teardown_tasks.add(task)
    task.add_done_callback(_background_teardown_tasks.discard)
    return {"status": "teardown_requested", "session_id": session_id}


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Delete a session. Only accessible by the session owner."""
    await _check_session_access(session_id, user, preload_sandbox=False)
    success = await session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.post("/submit")
async def submit_input(
    request: Request, user: dict = Depends(get_current_user)
) -> dict:
    """Submit user input to a session. Only accessible by the session owner."""
    # Parse the body manually so session ownership can be checked before the
    # text-length constraints fire — otherwise a non-owner sending an empty
    # or oversized text gets a 422 leaking the constraint instead of the 404
    # they'd get for any other access to a session they don't own.
    try:
        payload = await request.json()
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Body must be a JSON object")
    raw_session_id = payload.get("session_id")
    if not isinstance(raw_session_id, str) or not raw_session_id:
        raise RequestValidationError(
            [
                {
                    "type": "missing",
                    "loc": ("body", "session_id"),
                    "msg": "Field required",
                    "input": payload,
                }
            ]
        )
    agent_session = await _check_session_access(raw_session_id, user)
    try:
        body = SubmitRequest(**payload)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc
    await _enforce_premium_model_quota(user, agent_session)
    success = await session_manager.submit_user_input(
        body.session_id,
        body.text,
        body.cloud_provider,
        body.training_goal,
        body.output_policy,
        request_id=str(payload.get("request_id") or uuid.uuid4()),
    )
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "submitted", "session_id": body.session_id}


@router.post("/approve")
async def submit_approval(
    request: ApprovalRequest, user: dict = Depends(get_current_user)
) -> dict:
    """Submit tool approvals to a session. Only accessible by the session owner."""
    await _check_session_access(request.session_id, user)
    approvals = [
        {
            "tool_call_id": a.tool_call_id,
            "approved": a.approved,
            "approval_id": a.approval_id,
            "feedback": a.feedback,
            "edited_script": a.edited_script,
            "namespace": a.namespace,
        }
        for a in request.approvals
    ]
    success = await session_manager.submit_approval(request.session_id, approvals)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "submitted", "session_id": request.session_id}


@router.post("/chat/{session_id}")
async def chat_sse(
    session_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """SSE endpoint: submit input or approval, then stream events until turn ends."""
    agent_session = await _check_session_access(session_id, user, request)
    if not agent_session or not agent_session.is_active:
        raise HTTPException(status_code=404, detail="Session not found or inactive")

    # Parse body
    body = await request.json()
    request_id = str(body.get("request_id") or uuid.uuid4())
    stream_started_at = time.monotonic()

    # Subscribe BEFORE submitting so we never miss events — even if the
    # agent loop processes the submission before this coroutine continues.
    broadcaster = agent_session.broadcaster
    sub_id, event_queue = broadcaster.subscribe()

    # Submit the operation
    text = body.get("text")
    approvals = body.get("approvals")
    cloud_provider = (
        _cloud_provider_or_default(body.get("cloud_provider"))
        if "cloud_provider" in body
        else None
    )
    training_goal = (
        _training_goal_or_default(body.get("training_goal"))
        if "training_goal" in body
        else None
    )
    output_policy = (
        _output_policy_for_provider(
            body.get("output_policy"), cloud_provider or "hf-jobs"
        )
        if "output_policy" in body or cloud_provider == "gcp-vertex"
        else None
    )

    # Gate user-message sends against the daily premium-model quota. Approvals are
    # continuations of an in-progress turn — the session was already charged
    # on its first message, so we skip the gate there.
    if text is not None and not approvals:
        try:
            await _enforce_premium_model_quota(user, agent_session)
        except HTTPException:
            broadcaster.unsubscribe(sub_id)
            raise

    try:
        logger.info(
            "chat_stream_event request_id=%s session_id=%s cloud_provider=%s "
            "selected_model=%s event_type=stream_start",
            request_id,
            session_id,
            cloud_provider or agent_session.cloud_provider,
            agent_session.session.config.model_name,
        )
        if approvals:
            latest_run = await session_manager.latest_attachable_run(session_id)
            formatted = [
                {
                    "tool_call_id": a["tool_call_id"],
                    "approved": a["approved"],
                    "approval_id": a.get("approval_id"),
                    "feedback": a.get("feedback"),
                    "edited_script": a.get("edited_script"),
                    "namespace": a.get("namespace"),
                }
                for a in approvals
            ]
            success = await session_manager.submit_approval(
                session_id,
                formatted,
                run_id=latest_run["run_id"] if latest_run else None,
            )
        elif text is not None:
            success = await session_manager.submit_user_input(
                session_id,
                text,
                cloud_provider,
                training_goal,
                output_policy,
                request_id=request_id,
            )
        else:
            broadcaster.unsubscribe(sub_id)
            raise HTTPException(
                status_code=400, detail="Must provide 'text' or 'approvals'"
            )

        if not success:
            broadcaster.unsubscribe(sub_id)
            raise HTTPException(status_code=404, detail="Session not found or inactive")
    except HTTPException:
        broadcaster.unsubscribe(sub_id)
        raise
    except Exception as e:
        broadcaster.unsubscribe(sub_id)
        logger.exception(
            "chat_stream_event request_id=%s session_id=%s event_type=stream_error",
            request_id,
            session_id,
        )
        await agent_session.session.send_event(
            Event(
                event_type="stream_error",
                data={
                    "error": str(e),
                    "request_id": request_id,
                    "session_id": session_id,
                },
            )
        )
        raise

    return _sse_response(
        broadcaster,
        event_queue,
        sub_id,
        request=request,
        session_id=session_id,
        request_id=request_id,
        stream_started_at=stream_started_at,
    )


@router.post("/pro-click/{session_id}")
async def record_pro_click(
    session_id: str,
    body: dict,
    user: dict = Depends(get_current_user),
) -> dict:
    """Record a click on a Pro upgrade CTA shown from inside a session."""
    agent_session = await _check_session_access(session_id, user)

    from agent.core import telemetry

    await telemetry.record_pro_cta_click(
        agent_session.session,
        source=str(body.get("source") or "unknown"),
        target=str(body.get("target") or "pro_pricing"),
    )
    if agent_session.session.config.save_sessions:
        agent_session.session.save_and_upload_detached(
            agent_session.session.config.session_dataset_repo
        )
    return {"status": "ok"}


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


@router.get("/session/{session_id}/runs/{run_id}/stream")
async def stream_session_run(
    session_id: str,
    run_id: str,
    request: Request,
    since: int = 0,
    user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """Replay persisted run events, then attach to the live session broadcaster."""
    agent_session = await _check_session_access(
        session_id, user, request, preload_sandbox=False
    )
    replay_events = await session_manager.load_run_events_after(
        session_id, run_id, since
    )
    if replay_events is None:
        raise HTTPException(status_code=404, detail="Run not found")
    broadcaster = agent_session.broadcaster
    sub_id, event_queue = broadcaster.subscribe()
    return _sse_response(
        broadcaster,
        event_queue,
        sub_id,
        request=request,
        session_id=session_id,
        request_id=f"run-reconnect-{uuid.uuid4()}",
        stream_started_at=time.monotonic(),
        replay_events=replay_events,
        after_seq=since,
    )


@router.get("/events/{session_id}")
async def subscribe_events(
    session_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """Subscribe to events for a running session without submitting new input.

    Used by the frontend to re-attach after a connection drop (e.g. screen
    sleep).  Returns 404 if the session isn't active or isn't processing.
    """
    agent_session = await _check_session_access(session_id, user, request)
    if not agent_session or not agent_session.is_active:
        raise HTTPException(status_code=404, detail="Session not found or inactive")

    after_seq = _last_event_seq(request)
    replay_events = []
    if background_runs_in_process():
        replay_events = await session_manager._store().load_events_after(
            session_id, after_seq
        )
    broadcaster = agent_session.broadcaster
    sub_id, event_queue = broadcaster.subscribe()
    return _sse_response(
        broadcaster,
        event_queue,
        sub_id,
        request=request,
        session_id=session_id,
        request_id=f"reconnect-{uuid.uuid4()}",
        stream_started_at=time.monotonic(),
        replay_events=replay_events,
        after_seq=after_seq,
    )


@router.post("/interrupt/{session_id}")
async def interrupt_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Interrupt the current operation in a session."""
    await _check_session_access(session_id, user)
    success = await session_manager.interrupt(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "interrupted", "session_id": session_id}


@router.post("/session/{session_id}/runs/{run_id}/interrupt")
async def interrupt_session_run(
    session_id: str,
    run_id: str,
    user: dict = Depends(get_current_user),
) -> dict:
    """Interrupt a running durable run and mark its event log."""
    agent_session = await _check_session_access(session_id, user)
    if not await session_manager.get_run(session_id, run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    agent_session.session.current_run_id = run_id
    success = await session_manager.interrupt(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "interrupted", "session_id": session_id, "run_id": run_id}


@router.get("/session/{session_id}/messages")
async def get_session_messages(
    session_id: str, user: dict = Depends(get_current_user)
) -> list[dict]:
    """Return the session's message history from memory."""
    agent_session = await _check_session_access(session_id, user, preload_sandbox=False)
    if not agent_session or not agent_session.is_active:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return [
        msg.model_dump(mode="json")
        for msg in agent_session.session.context_manager.items
    ]


@router.post("/undo/{session_id}")
async def undo_session(session_id: str, user: dict = Depends(get_current_user)) -> dict:
    """Undo the last turn in a session."""
    await _check_session_access(session_id, user)
    success = await session_manager.undo(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "undo_requested", "session_id": session_id}


@router.post("/truncate/{session_id}")
async def truncate_session(
    session_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> dict:
    """Truncate conversation to before a specific user message."""
    # Check session ownership before parsing the request body so a 404 on a
    # non-existent / non-owned session_id beats the 422 schema-validation error
    # (otherwise the response leaks the required field name to non-owners).
    await _check_session_access(session_id, user)
    try:
        body = TruncateRequest(**(await request.json()))
    except ValidationError as exc:
        # Re-raise as RequestValidationError so FastAPI returns its standard
        # structured 422 schema (`{"detail": [{"type":..., "loc":..., ...}]}`)
        # instead of a string-stringified Pydantic dump.
        raise RequestValidationError(exc.errors()) from exc
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    success = await session_manager.truncate(session_id, body.user_message_index)
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Session not found, inactive, or message index out of range",
        )
    return {"status": "truncated", "session_id": session_id}


@router.post("/compact/{session_id}")
async def compact_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Compact the context in a session."""
    await _check_session_access(session_id, user)
    success = await session_manager.compact(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "compact_requested", "session_id": session_id}


@router.post("/shutdown/{session_id}")
async def shutdown_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Shutdown a session."""
    await _check_session_access(session_id, user)
    success = await session_manager.shutdown_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "shutdown_requested", "session_id": session_id}


@router.post("/feedback/{session_id}")
async def submit_feedback(
    session_id: str,
    body: dict,
    user: dict = Depends(get_current_user),
) -> dict:
    """Attach a user feedback signal to a session's event log.

    Body: {rating: "up"|"down"|"outcome_success"|"outcome_fail",
           turn_index?: int, comment?: str, message_id?: str}
    Appended as a `feedback` event and saved with the session trajectory.
    """
    agent_session = await _check_session_access(session_id, user)

    rating = body.get("rating")
    if rating not in {"up", "down", "outcome_success", "outcome_fail"}:
        raise HTTPException(status_code=400, detail="invalid rating")

    from agent.core import telemetry

    await telemetry.record_feedback(
        agent_session.session,
        rating=rating,
        turn_index=body.get("turn_index"),
        message_id=body.get("message_id"),
        comment=body.get("comment"),
    )
    # Fire-and-forget save so feedback reaches the dataset even if the user
    # closes the tab right after clicking.
    if agent_session.session.config.save_sessions:
        agent_session.session.save_and_upload_detached(
            agent_session.session.config.session_dataset_repo
        )
    return {"status": "ok"}
