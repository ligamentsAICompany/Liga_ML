"""Feature flags for durable background run support.

The Phase 1 implementation runs inside the API process.  ``external_worker`` is
reserved so deploys can expose intent without accidentally claiming support for
a separate worker that does not exist yet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

RunWorkerMode = Literal["disabled", "in_process", "external_worker"]

VALID_WORKER_MODES: set[str] = {"disabled", "in_process", "external_worker"}
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off", ""}


@dataclass(frozen=True)
class BackgroundRunSettings:
    enabled: bool
    worker_mode: RunWorkerMode
    token_encryption_configured: bool = False
    warning: str | None = None

    @property
    def in_process(self) -> bool:
        return self.enabled and self.worker_mode == "in_process"

    @property
    def implemented(self) -> bool:
        return self.in_process


def _parse_bool(raw: str | None, *, default: bool = False) -> tuple[bool, str | None]:
    if raw is None:
        return default, None
    normalized = raw.strip().lower()
    if normalized in TRUE_VALUES:
        return True, None
    if normalized in FALSE_VALUES:
        return False, None
    return default, (
        f"Invalid BACKGROUND_RUNS_ENABLED={raw!r}; expected true or false. "
        f"Using {str(default).lower()}."
    )


def _parse_worker_mode(raw: str | None) -> tuple[RunWorkerMode, str | None]:
    normalized = (raw or "disabled").strip().lower()
    if normalized in VALID_WORKER_MODES:
        return normalized, None  # type: ignore[return-value]
    return "disabled", (
        f"Invalid RUN_WORKER_MODE={raw!r}; expected disabled, in_process, "
        "or external_worker. Using disabled."
    )


def load_background_run_settings() -> BackgroundRunSettings:
    enabled, enabled_warning = _parse_bool(os.environ.get("BACKGROUND_RUNS_ENABLED"))
    worker_mode, mode_warning = _parse_worker_mode(os.environ.get("RUN_WORKER_MODE"))
    warning = enabled_warning or mode_warning
    return BackgroundRunSettings(
        enabled=enabled,
        worker_mode=worker_mode,
        token_encryption_configured=bool(
            os.environ.get("SESSION_TOKEN_ENCRYPTION_KEY")
        ),
        warning=warning,
    )


def background_runs_in_process() -> bool:
    return load_background_run_settings().in_process


def background_run_status(session_store: dict[str, Any]) -> dict[str, Any]:
    settings = load_background_run_settings()
    store_type = str(session_store.get("type") or "unknown")
    store_durable = bool(session_store.get("durable"))
    implemented = settings.implemented
    durable = bool(implemented and store_durable)

    warning = settings.warning
    if warning is None and settings.worker_mode == "external_worker":
        warning = "RUN_WORKER_MODE=external_worker is reserved and not implemented yet."
    elif warning is None and settings.enabled and settings.worker_mode == "disabled":
        warning = (
            "RUN_WORKER_MODE=disabled keeps the old chat flow; background runs are off."
        )
    elif warning is None and implemented and not store_durable:
        warning = (
            "BACKGROUND_RUNS_ENABLED=true with RUN_WORKER_MODE=in_process requires "
            "a durable MongoDB session store for replay and reconnect."
        )

    return {
        "enabled": settings.enabled,
        "worker_mode": settings.worker_mode,
        "implemented": implemented,
        "durable": durable,
        "store": store_type,
        "token_handoff_configured": settings.token_encryption_configured,
        "warning": warning,
    }


RUN_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "interrupted"}


def run_status_from_event(
    event_type: str, data: dict[str, Any] | None = None
) -> str | None:
    """Map existing stream events to durable run status updates."""
    data = data or {}
    if event_type == "run_started":
        return "running"
    if event_type == "approval_required":
        return "waiting_approval"
    if event_type == "tool_state_change":
        state = str(data.get("state") or "").lower()
        if state in {"running", "queued", "starting"}:
            return "waiting_provider"
        if state in {"succeeded", "completed", "success"}:
            return "running"
        if state in {"failed", "error", "billing_required"}:
            return "failed" if state != "billing_required" else "waiting_approval"
        if state in {"cancelled", "rejected", "abandoned"}:
            return "cancelled"
    if event_type == "turn_complete":
        if str(data.get("waiting_for_tool_approval") or "").lower() in {
            "1",
            "true",
            "yes",
        }:
            return "waiting_approval"
        return "succeeded"
    if event_type in {"error", "stream_error"}:
        return "failed"
    if event_type == "interrupted":
        return "interrupted"
    return None


def provider_metadata_from_event(
    event_type: str, data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Extract non-secret provider job fields emitted by existing tools."""
    if event_type != "tool_state_change" or not isinstance(data, dict):
        return {}
    tool = str(data.get("tool") or "")
    provider = {
        "hf_jobs": "hf-jobs",
        "gcp_vertex_jobs": "gcp-vertex",
        "aws_sagemaker_jobs": "aws-sagemaker",
    }.get(tool)
    metadata: dict[str, Any] = {}
    if provider:
        metadata["provider"] = provider
    field_map = {
        "jobName": "active_provider_job_id",
        "jobUrl": "provider_console_url",
        "outputDir": "provider_artifact_path",
        "cloudWatchLogsUrl": "provider_logs_url",
        "s3ModelArtifact": "provider_artifact_path",
        "outputPolicy": "provider_output_policy",
        "state": "provider_status",
        "tool": "active_tool",
    }
    for source, target in field_map.items():
        value = data.get(source)
        if value not in (None, ""):
            metadata[target] = value
    return metadata


def safe_event_summary(event_type: str, payload: dict[str, Any] | None = None) -> str:
    """Short, non-secret event description for diagnostics and run lists."""
    payload = payload or {}
    if event_type == "assistant_message":
        content = str(payload.get("content") or "")
        return content[:160]
    if event_type == "tool_call":
        return f"tool_call:{payload.get('tool') or 'unknown'}"
    if event_type == "tool_output":
        return f"tool_output:{payload.get('tool') or 'unknown'}"
    if event_type == "approval_required":
        tools = payload.get("tools")
        return f"approval_required:{len(tools) if isinstance(tools, list) else 0}"
    if event_type == "tool_state_change":
        return f"{payload.get('tool') or 'tool'}:{payload.get('state') or 'update'}"
    if event_type in {"error", "stream_error"}:
        return str(payload.get("error") or event_type)[:160]
    return event_type
