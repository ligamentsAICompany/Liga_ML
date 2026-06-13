"""Agent tool for live read-only training preflight before provider launch."""

from __future__ import annotations

import os
from typing import Any

from agent.core.training_preflight import run_training_preflight


def _session_recommendation(session: Any) -> dict[str, Any] | None:
    recommendation = getattr(session, "latest_training_recommendation", None)
    return recommendation if isinstance(recommendation, dict) else None


def _session_dataset_discovery(session: Any) -> dict[str, Any] | None:
    discovery = getattr(session, "latest_dataset_discovery", None)
    return discovery if isinstance(discovery, dict) and discovery else None


def _format_preflight_result(result_dict: dict[str, Any]) -> str:
    lines = [
        "## Training preflight",
        "",
        f"**Status:** {result_dict.get('status')}",
        f"**Launch ready:** {result_dict.get('launch_ready')}",
        f"**Manual approval allowed:** {result_dict.get('manual_approval_allowed')}",
    ]
    summary = str(result_dict.get("safe_summary") or "").strip()
    if summary:
        lines.extend(["", summary])
    blocking = result_dict.get("blocking_reasons")
    if isinstance(blocking, list) and blocking:
        lines.extend(["", "**Blocking reasons:**", *[f"- {item}" for item in blocking]])
    unknowns = result_dict.get("unknown_reasons")
    if isinstance(unknowns, list) and unknowns:
        lines.extend(["", "**Unknown checks:**", *[f"- {item}" for item in unknowns]])
    manual_reason = str(result_dict.get("manual_approval_reason") or "").strip()
    if manual_reason:
        lines.extend(["", f"**Manual approval note:** {manual_reason}"])
    if result_dict.get("manual_approval_allowed") is True:
        lines.extend(
            [
                "",
                "Bounded smoke may proceed to the approval-gated `gcp_vertex_jobs` "
                "run step when unknown quota/accelerator checks remain. Do not "
                "auto-launch; wait for explicit user approval on the provider card.",
            ]
        )
    elif blocking:
        lines.extend(
            [
                "",
                "Do not call `gcp_vertex_jobs` run until blocking preflight failures "
                "are resolved.",
            ]
        )
    return "\n".join(lines)


async def _persist_preflight(session: Any, preflight: dict[str, Any]) -> None:
    session_id = getattr(session, "session_id", None)
    run_id = getattr(session, "current_run_id", None)
    if not session_id:
        return
    setattr(session, "latest_training_preflight", dict(preflight))
    store = getattr(session, "persistence_store", None)
    if store is None or not hasattr(store, "record_training_preflight"):
        return
    await store.record_training_preflight(
        session_id=str(session_id),
        run_id=str(run_id) if run_id else None,
        preflight=preflight,
    )


async def execute_training_preflight_for_session(
    session: Any,
    *,
    recommendation: dict[str, Any] | None = None,
    force_refresh: bool = False,
) -> tuple[str, bool, dict[str, Any] | None]:
    """Run training preflight for the active session recommendation."""

    session_id = str(getattr(session, "session_id", "") or "")
    if not session_id:
        return "ERROR: Session is not initialized for training preflight.", False, None

    resolved_recommendation = recommendation or _session_recommendation(session)
    if not isinstance(resolved_recommendation, dict) or not resolved_recommendation:
        return (
            "ERROR: No training recommendation is available. Run training_planner first.",
            False,
            None,
        )

    discovery = _session_dataset_discovery(session)
    hf_token = getattr(session, "hf_token", None)
    gcp_project_id = (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT_ID")
        or os.environ.get("GCLOUD_PROJECT")
    )
    gcp_region = (
        os.environ.get("GOOGLE_CLOUD_REGION")
        or os.environ.get("VERTEX_AI_REGION")
        or os.environ.get("GCP_REGION")
        or "us-central1"
    )
    result = await run_training_preflight(
        session_id=session_id,
        run_id=getattr(session, "current_run_id", None),
        recommendation=resolved_recommendation,
        dataset_discovery=discovery,
        force_refresh=force_refresh,
        metadata={
            "training_goal": getattr(session, "training_goal", None),
            "bounded_vertex_smoke": bool(
                getattr(session, "bounded_vertex_smoke_for_turn", False)
            ),
        },
        hf_token=str(hf_token) if hf_token else None,
        gcp_project_id=str(gcp_project_id) if gcp_project_id else None,
        gcp_region=str(gcp_region) if gcp_region else None,
    )
    result_dict = result.to_dict()
    await _persist_preflight(session, result_dict)
    blocking = result_dict.get("blocking_reasons")
    has_blocking = isinstance(blocking, list) and bool(blocking)
    success = not has_blocking
    return _format_preflight_result(result_dict), success, result_dict


TRAINING_PREFLIGHT_TOOL_SPEC = {
    "name": "training_preflight",
    "description": (
        "Run live read-only training preflight checks for the current training "
        "recommendation. Does not launch provider jobs, create sandboxes, or "
        "mutate cloud resources. Required before bounded Google Vertex AI "
        "approval-gated launch when live preflight is requested."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["run"],
                "description": "Operation to execute. Only run is supported.",
            },
            "force_refresh": {
                "type": "boolean",
                "description": "When true, bypass cached preflight results.",
            },
        },
        "required": ["operation"],
    },
}


async def training_preflight_handler(
    arguments: dict[str, Any],
    session: Any = None,
    tool_call_id: str | None = None,
) -> tuple[str, bool]:
    if session is None:
        return "ERROR: Session is required for training preflight.", False
    operation = str(arguments.get("operation") or "run").strip().lower()
    if operation != "run":
        return (
            f'Unknown operation: "{operation}". Available operations: run.',
            False,
        )
    output, success, structured = await execute_training_preflight_for_session(
        session,
        force_refresh=bool(arguments.get("force_refresh")),
    )
    if structured is not None:
        outputs = getattr(session, "_structured_tool_outputs", None)
        if not isinstance(outputs, dict):
            outputs = {}
            setattr(session, "_structured_tool_outputs", outputs)
        if tool_call_id:
            outputs[tool_call_id] = structured
    return output, success
