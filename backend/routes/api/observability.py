"""Health, usage, audit, evaluation, and response log routes."""

# ruff: noqa: F403, F405, E402
from fastapi import APIRouter

import routes.api.common as _common
from routes.api.common import *

for _name, _value in vars(_common).items():
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = _value

router = APIRouter()


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
    await _refresh_response_rows_for_evaluations(user["user_id"])
    evaluations = await session_manager.list_evaluations(
        session_id=session_id,
        run_id=run_id,
        provider=provider,
        status=status,
        limit=_safe_limit(limit),
        user_id=user["user_id"],
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
    await _refresh_response_rows_for_evaluations(user["user_id"])
    raw = await session_manager.evaluation_summary(
        session_id=session_id,
        run_id=run_id,
        provider=provider,
        status=status,
        limit=_safe_limit(limit),
        user_id=user["user_id"],
    )
    return _evaluation_summary_response(raw)

@router.get("/session/{session_id}/runs/{run_id}/evaluation/report")
async def get_run_evaluation_report(
    session_id: str,
    run_id: str,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    await _check_session_access(session_id, user, preload_sandbox=False)
    await _refresh_response_rows_for_evaluations(user["user_id"])
    evaluation = await session_manager.get_evaluation_for_run(
        session_id, run_id, user_id=user["user_id"]
    )
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    payload = PostTrainingEvaluation(**_serialize_evaluation(evaluation))
    return {
        "evaluation_id": payload.evaluation_id,
        "status": payload.status,
        "report_markdown": payload.report_markdown or "",
    }

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
                if await _refresh_stale_response_rows(
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
                if await _refresh_stale_response_rows(
                    response_page.get("rows", []), user_id=user["user_id"]
                ):
                    response_page = await store.list_response_rows(
                        user_id=user["user_id"], **filters
                    )
        await _sync_runs_from_terminal_response_rows(response_page.get("rows", []))
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
