#!/usr/bin/env python3
"""Post-split patches for Phase 2 SSE decoupling."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAT = ROOT / "backend" / "routes" / "api" / "chat.py"
OBS = ROOT / "backend" / "routes" / "api" / "observability.py"
SESSIONS = ROOT / "backend" / "routes" / "api" / "sessions.py"

SSE_OLD = '''# ---------------------------------------------------------------------------
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


def _last_event_seq(request: Request) -> int:'''

SSE_NEW = '''# ---------------------------------------------------------------------------
# SSE reconnect helper (streaming lives in session_manager.build_sse_response)
# ---------------------------------------------------------------------------


def _last_event_seq(request: Request) -> int:'''


def patch_chat() -> None:
    text = CHAT.read_text(encoding="utf-8")
    text = text.replace(SSE_OLD, SSE_NEW)
    text = text.replace("return _sse_response(", "return session_manager.build_sse_response(")

    # Drop inlined _sse_response implementation if still present.
    marker = "def _format_sse(msg: dict[str, Any]) -> str:"
    end_marker = '@router.get("/session/{session_id}/runs/{run_id}/stream")'
    if marker in text and end_marker in text:
        start = text.index(marker)
        end = text.index(end_marker)
        text = text[:start] + text[end:]

    interrupt_route = '''@router.post("/session/{session_id}/runs/{run_id}/interrupt")
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



'''
    if interrupt_route.strip() not in text:
        text = text.replace(
            '@router.post("/interrupt/{session_id}")',
            interrupt_route + '@router.post("/interrupt/{session_id}")',
            1,
        )

    CHAT.write_text(text, encoding="utf-8")


def patch_observability() -> None:
    text = OBS.read_text(encoding="utf-8")
    old = '''    evaluation = await get_run_evaluation(session_id, run_id, user)
    return {
        "evaluation_id": evaluation.evaluation_id,
        "status": evaluation.status,
        "report_markdown": evaluation.report_markdown or "",
    }'''
    new = '''    await _check_session_access(session_id, user, preload_sandbox=False)
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
    }'''
    text = text.replace(old, new)
    OBS.write_text(text, encoding="utf-8")


def patch_sessions() -> None:
    text = SESSIONS.read_text(encoding="utf-8")
    run_interrupt = '''@router.post("/session/{session_id}/runs/{run_id}/interrupt")
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


'''
    if run_interrupt in text:
        text = text.replace(run_interrupt, "")
        SESSIONS.write_text(text, encoding="utf-8")


def main() -> None:
    patch_chat()
    patch_observability()
    patch_sessions()
    print("Applied Phase 2 post-split patches.")


if __name__ == "__main__":
    main()
