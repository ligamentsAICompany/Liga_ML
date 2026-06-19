"""Chat submission, SSE streaming, and session control routes."""

# ruff: noqa: F403, F405, E402
from fastapi import APIRouter

import routes.api.common as _common
from routes.api.common import *

for _name, _value in vars(_common).items():
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = _value

router = APIRouter()


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

    return session_manager.build_sse_response(
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
    return session_manager.build_sse_response(
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
    return session_manager.build_sse_response(
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
