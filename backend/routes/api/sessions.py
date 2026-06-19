"""Session lifecycle, configuration, runs, and user quota routes."""

# ruff: noqa: F403, F405, E402
from fastapi import APIRouter

import routes.api.common as _common
from routes.api.common import *

for _name, _value in vars(_common).items():
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = _value

router = APIRouter()


@router.get(
    "/session/{session_id}/evaluations", response_model=list[PostTrainingEvaluation]
)
async def list_session_evaluations(
    session_id: str,
    user: dict = Depends(get_current_user),
) -> list[PostTrainingEvaluation]:
    await _check_session_access(session_id, user, preload_sandbox=False)
    evaluations = await session_manager.list_evaluations(
        session_id=session_id, limit=500, user_id=user["user_id"]
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
    await _refresh_response_rows_for_evaluations(user["user_id"])
    evaluation = await session_manager.get_evaluation_for_run(
        session_id, run_id, user_id=user["user_id"]
    )
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return PostTrainingEvaluation(**_serialize_evaluation(evaluation))

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
            preload_sandbox=False,
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
            preload_sandbox=False,
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
