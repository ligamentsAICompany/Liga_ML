"""loop
Main agent implementation with integrated tool system and MCP support
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from litellm import (
    ChatCompletionMessageToolCall,
    Message,
    acompletion,
    stream_chunk_builder,
)
from litellm.exceptions import ContextWindowExceededError

from agent.config import Config
from agent.core.approval_policy import (
    is_scheduled_operation,
    normalize_tool_operation,
)
from agent.core.cost_estimation import CostEstimate, estimate_tool_cost
from agent.messaging.gateway import NotificationGateway
from agent.core import telemetry
from agent.core.doom_loop import check_for_doom_loop
from agent.core.llm_params import _resolve_llm_params
from agent.core.prompt_caching import with_prompt_caching
from agent.core.session import DEFAULT_SESSION_LOG_DIR, Event, OpType, Session
from agent.core.tools import ToolRouter
from agent.tools.jobs_tool import CPU_FLAVORS
from agent.tools.sandbox_tool import (
    DEFAULT_CPU_SANDBOX_HARDWARE,
    start_cpu_sandbox_preload,
    teardown_session_sandbox,
)

logger = logging.getLogger(__name__)

ToolCall = ChatCompletionMessageToolCall

_MALFORMED_TOOL_PREFIX = "ERROR: Tool call to '"
_MALFORMED_TOOL_SUFFIX = "' had malformed JSON arguments"
_NO_TOOL_INCOMPLETE_PLAN_RETRY_LIMIT = 2
_VERTEX_SMOKE_CONTINUATION_RETRY_LIMIT = 2
_ACTIVE_CLOUD_JOB_STATES = {
    "created",
    "creating",
    "pending",
    "queued",
    "starting",
    "started",
    "running",
    "inprogress",
    "in_progress",
    "stopping",
    "job_state_pending",
    "job_state_queued",
    "job_state_running",
}
_TERMINAL_CLOUD_JOB_STATES = {
    "completed",
    "succeeded",
    "success",
    "failed",
    "stopped",
    "cancelled",
    "canceled",
    "expired",
    "job_state_succeeded",
    "job_state_failed",
    "job_state_cancelled",
    "job_state_expired",
}
_AWS_MONITORING_ALLOWED_OPS = {"inspect", "logs", "ps", "cancel"}
_GCP_MONITORING_ALLOWED_OPS = {"inspect", "logs", "ps", "cancel"}
_AWS_PROVIDER_DRIFT_MESSAGE = (
    "An AWS SageMaker job is already active or terminal in this session. "
    "Use aws_sagemaker_jobs inspect/logs/ps. Do not use sandbox/bash/HF tools "
    "for AWS monitoring."
)
_GCP_PROVIDER_DRIFT_MESSAGE = (
    "Provider is Google Cloud Vertex AI and a Vertex AI job is active. "
    "Use gcp_vertex_jobs inspect/logs instead."
)
_AWS_SECOND_RUN_BLOCK_MESSAGE = (
    _AWS_PROVIDER_DRIFT_MESSAGE
    + " AWS job failed. No automatic retry was launched. Ask the user explicitly "
    "before preparing another paid AWS SageMaker run."
)
_IMAGE_TOKEN_RE = re.compile(r"<\|?image(?:[_-]\d+)?\|?>", re.IGNORECASE)


def _is_kimi_text_model(model_name: str | None) -> bool:
    normalized = str(model_name or "").lower()
    return "kimi" in normalized or "moonshot" in normalized


def _strip_image_tokens(text: str) -> str:
    return _IMAGE_TOKEN_RE.sub("", text)


def _content_part_has_actual_image(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    part_type = str(part.get("type") or "").lower()
    if part_type not in {"image", "image_url", "input_image"}:
        return False
    image_url = part.get("image_url")
    if isinstance(image_url, dict) and image_url.get("url"):
        return True
    if isinstance(image_url, str) and image_url:
        return True
    return bool(part.get("url") or part.get("data") or part.get("source"))


def _message_has_actual_image(message: Any) -> bool:
    images = (
        message.get("images")
        if isinstance(message, dict)
        else getattr(message, "images", None)
    )
    if images:
        return True
    content = (
        message.get("content")
        if isinstance(message, dict)
        else getattr(message, "content", None)
    )
    return isinstance(content, list) and any(
        _content_part_has_actual_image(part) for part in content
    )


def _sanitize_text_only_message_for_kimi(message: Any) -> Any:
    if _message_has_actual_image(message):
        return message

    is_dict = isinstance(message, dict)
    content = message.get("content") if is_dict else getattr(message, "content", None)
    images = message.get("images") if is_dict else getattr(message, "images", None)
    updates: dict[str, Any] = {}

    if isinstance(content, str):
        cleaned_content = _strip_image_tokens(content)
        if cleaned_content != content:
            updates["content"] = cleaned_content
    elif isinstance(content, list):
        text_parts = [
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict)
            and str(part.get("type") or "").lower() == "text"
            and part.get("text")
        ]
        updates["content"] = _strip_image_tokens("\n".join(text_parts))

    if images is not None:
        updates["images"] = None

    if not updates:
        return message
    if is_dict:
        cleaned = dict(message)
        if "content" in updates:
            cleaned["content"] = updates["content"]
        if "images" in updates:
            cleaned.pop("images", None)
        return cleaned
    return message.model_copy(update=updates)


def _sanitize_messages_for_model(
    messages: list[Any], model_name: str | None
) -> list[Any]:
    """Keep text-only Kimi/Moonshot requests free of stale image markers."""
    if not _is_kimi_text_model(model_name):
        return messages
    return [_sanitize_text_only_message_for_kimi(message) for message in messages]


def _is_terminal_provider_tool_output(
    tool_name: str, output: str, success: bool
) -> bool:
    if tool_name != "gcp_vertex_jobs":
        return False
    text = str(output or "").lower()
    if any(
        state in text
        for state in (
            "job_state_failed",
            "job_state_cancelled",
            "job_state_expired",
            "job_state_succeeded",
        )
    ):
        return True
    return not success and "vertex" in text


def _unfinished_plan_items(session: Session) -> list[dict[str, str]]:
    plan = getattr(session, "current_plan", None) or []
    unfinished: list[dict[str, str]] = []
    for item in plan:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        if status in {"pending", "in_progress"}:
            unfinished.append(item)
    return unfinished


def _format_plan_items_for_guard(items: list[dict[str, str]], limit: int = 4) -> str:
    formatted = []
    for item in items[:limit]:
        item_id = item.get("id") or "?"
        content = item.get("content") or "(unnamed task)"
        status = item.get("status") or "unknown"
        formatted.append(f"{item_id}. {content} [{status}]")
    if len(items) > limit:
        formatted.append(f"... and {len(items) - limit} more")
    return "; ".join(formatted)


def _structured_tool_output(
    session: "Session", tool_call_id: str | None
) -> dict | None:
    outputs = getattr(session, "_structured_tool_outputs", None)
    if not isinstance(outputs, dict) or not tool_call_id:
        return None
    value = outputs.pop(tool_call_id, None)
    return value if isinstance(value, dict) else None


def _no_tool_incomplete_plan_prompt(items: list[dict[str, str]]) -> str:
    summary = _format_plan_items_for_guard(items)
    return (
        "[SYSTEM: CONTINUATION GUARD] Your previous response ended without any "
        "tool calls, but the task is not complete. The current plan still has "
        f"unfinished items: {summary}. Do not return control to the user yet. "
        "Continue from the next unfinished item and make at least one tool call "
        "now. If you genuinely cannot continue, first use tools to inspect the "
        "state or verify the blocker."
    )


def _normalized_cloud_job_state(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _latest_cloud_job_state(session: Session, tool_name: str) -> str | None:
    """Return the latest observed state for a provider job tool in this session."""

    for event in reversed(getattr(session, "logged_events", []) or []):
        if not isinstance(event, dict):
            continue
        if event.get("event_type") != "tool_state_change":
            continue
        data = event.get("data") or {}
        if not isinstance(data, dict) or data.get("tool") != tool_name:
            continue
        state = _normalized_cloud_job_state(data.get("state"))
        if state:
            return state
    context_manager = getattr(session, "context_manager", None)
    for message in reversed(getattr(context_manager, "items", []) or []):
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        name = (
            message.get("name")
            if isinstance(message, dict)
            else getattr(message, "name", None)
        )
        if role != "tool" or name != tool_name:
            continue
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", "")
        )
        text = str(content or "").lower()
        if any(
            marker in text
            for marker in (
                "trainingjobstatus:** completed",
                "trainingjobstatus: completed",
                "job_state_succeeded",
                "status:** completed",
            )
        ):
            return "succeeded"
        if any(
            marker in text
            for marker in (
                "trainingjobstatus:** failed",
                "trainingjobstatus: failed",
                "job_state_failed",
                "status:** failed",
            )
        ):
            return "failed"
        if any(
            marker in text
            for marker in (
                "trainingjobstatus:** stopped",
                "trainingjobstatus: stopped",
                "job_state_cancelled",
                "job_state_expired",
            )
        ):
            return "stopped"
        if (
            tool_name == "aws_sagemaker_jobs"
            and "aws sagemaker training job submitted" in text
        ):
            return "running"
        if tool_name == "gcp_vertex_jobs" and "vertex ai job submitted" in text:
            return "running"
    return None


def _has_active_provider_job(session: Session, tool_name: str) -> bool:
    state = _latest_cloud_job_state(session, tool_name)
    if not state or state in _TERMINAL_CLOUD_JOB_STATES:
        return False
    return state in _ACTIVE_CLOUD_JOB_STATES or bool(state)


def _has_provider_job(session: Session, tool_name: str) -> bool:
    return _latest_cloud_job_state(session, tool_name) is not None


def _has_terminal_provider_job(session: Session, tool_name: str) -> bool:
    state = _latest_cloud_job_state(session, tool_name)
    return bool(state and state in _TERMINAL_CLOUD_JOB_STATES)


def _has_explicit_aws_paid_retry_intent(session: Session) -> bool:
    if bool(getattr(session, "aws_sagemaker_retry_authorized", False)):
        return True
    retry_markers = (
        "retry",
        "launch another",
        "run another",
        "second paid",
        "approve second",
        "try again",
    )
    context_manager = getattr(session, "context_manager", None)
    for message in reversed(getattr(context_manager, "items", []) or []):
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        if role != "user":
            continue
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", "")
        )
        text = str(content or "").lower()
        return any(marker in text for marker in retry_markers)
    return False


def _provider_tool_policy_violation(
    session: Session, tool_name: str, tool_args: dict[str, Any]
) -> str | None:
    """Block compute-provider drift while an active provider job is being monitored."""

    if (
        getattr(session, "training_planner_only_for_turn", False)
        and tool_name == "dataset_discovery"
    ):
        return (
            "The user explicitly requested the training planner only for this "
            "turn. Do not call dataset_discovery; call training_planner with "
            "provider gcp-vertex/hf-jobs/aws-sagemaker as requested, then "
            "summarize that dataset selection or discovery would require a "
            "separate user request."
        )

    if getattr(session, "compute_tools_blocked_for_turn", False) and tool_name in {
        "sandbox_create",
        "bash",
        "read",
        "write",
        "edit",
        "hf_jobs",
        "gcp_vertex_jobs",
        "aws_sagemaker_jobs",
    }:
        if (
            tool_name == "gcp_vertex_jobs"
            and _operation(tool_args) == "run"
            and _should_continue_vertex_smoke_launch(session)
        ):
            return None
        return (
            "The user explicitly requested planning/discovery only and forbade "
            "sandbox, provider jobs, downloads, uploads, and resource creation. "
            "Use dataset_discovery, training_planner, or a plain assistant "
            "summary instead."
        )

    provider = str(getattr(session, "cloud_provider", "hf-jobs") or "hf-jobs").strip()
    operation = _operation(tool_args)

    if provider == "aws-sagemaker" and _has_provider_job(session, "aws_sagemaker_jobs"):
        if tool_name == "aws_sagemaker_jobs":
            if operation in _AWS_MONITORING_ALLOWED_OPS:
                return None
            if operation == "run" and _has_terminal_provider_job(
                session, "aws_sagemaker_jobs"
            ):
                if _has_explicit_aws_paid_retry_intent(session):
                    return None
                return _AWS_SECOND_RUN_BLOCK_MESSAGE
            return _AWS_PROVIDER_DRIFT_MESSAGE
        if tool_name in {
            "sandbox_create",
            "bash",
            "hf_jobs",
            "gcp_vertex_jobs",
            "hf_repo_files",
        }:
            return _AWS_PROVIDER_DRIFT_MESSAGE

    if provider == "gcp-vertex" and _has_active_provider_job(
        session, "gcp_vertex_jobs"
    ):
        if tool_name == "gcp_vertex_jobs":
            if operation in _GCP_MONITORING_ALLOWED_OPS:
                return None
            return _GCP_PROVIDER_DRIFT_MESSAGE
        if tool_name in {"sandbox_create", "bash", "hf_jobs", "aws_sagemaker_jobs"}:
            return _GCP_PROVIDER_DRIFT_MESSAGE

    return None


def _user_explicitly_requests_bounded_provider_launch(text: str) -> bool:
    normalized = str(text or "").lower()
    vertex_requested = any(
        marker in normalized
        for marker in (
            "google vertex ai",
            "google cloud vertex",
            "gcp vertex",
            "gcp-vertex",
            "vertex ai",
        )
    )
    if not vertex_requested:
        return False
    if any(
        marker in normalized
        for marker in (
            "do not run google vertex",
            "do not use google vertex",
            "do not run vertex",
        )
    ):
        return False
    bounded_smoke = any(
        marker in normalized
        for marker in (
            "smoke-test",
            "smoke test",
            "bounded vertex",
            "bounded google vertex",
            "quick smoke",
            "smallest safe runtime",
        )
    )
    launch_intent = any(
        marker in normalized
        for marker in (
            "approve it and continue",
            "pricing/provider approval",
            "before launch",
            "run one bounded",
            "fine-tuning workflow",
            "gcp_vertex_jobs",
        )
    )
    return bounded_smoke and launch_intent


def _user_requested_no_compute_tools(text: str) -> bool:
    if _user_explicitly_requests_bounded_provider_launch(text):
        return False
    normalized = str(text or "").lower()
    sandbox_blocked = any(
        marker in normalized
        for marker in (
            "do not use sandbox",
            "do not create sandbox",
            "no sandbox",
            "without sandbox",
        )
    )
    provider_jobs_blocked = any(
        marker in normalized
        for marker in (
            "do not launch training",
            "do not run hugging face jobs",
            "do not run hf jobs",
            "do not run google vertex ai",
            "do not run aws sagemaker",
            "only use the application's no-upload dataset discovery",
            "planning tools",
        )
    )
    upload_download_blocked = any(
        marker in normalized
        for marker in (
            "do not upload",
            "do not download",
            "no-upload",
            "no upload",
        )
    )
    return sandbox_blocked and (provider_jobs_blocked or upload_download_blocked)


def _user_requested_training_planner_only(text: str) -> bool:
    normalized = str(text or "").lower()
    planner_only = any(
        marker in normalized
        for marker in (
            "training planner only",
            "planner only",
            "use the training_planner only",
            "use training_planner only",
        )
    )
    if not planner_only:
        return False
    discovery_requested = any(
        marker in normalized
        for marker in (
            "dataset discovery",
            "dataset_discovery",
            "discover dataset",
            "find datasets",
            "search datasets",
        )
    )
    return not discovery_requested


def _requested_training_provider_from_text(text: str) -> str | None:
    normalized = str(text or "").lower()
    rejected = _rejected_training_providers_from_text(normalized)
    if (
        any(
            marker in normalized
            for marker in (
                "google vertex ai",
                "vertex ai",
                "gcp vertex",
                "gcloud",
                "google cloud",
            )
        )
        and "gcp-vertex" not in rejected
    ):
        return "gcp-vertex"
    if (
        any(marker in normalized for marker in ("aws sagemaker", "sagemaker"))
        and "aws-sagemaker" not in rejected
    ):
        return "aws-sagemaker"
    if (
        any(
            marker in normalized
            for marker in ("hugging face jobs", "hf jobs", "hf-jobs")
        )
        and "hf-jobs" not in rejected
    ):
        return "hf-jobs"
    return None


def _rejected_training_providers_from_text(text: str) -> set[str]:
    normalized = str(text or "").lower()
    rejected: set[str] = set()
    rejection_patterns = {
        "gcp-vertex": (
            r"\bdo\s+not\s+(?:use|run)\b[^.?!\n]*(?:google\s+vertex|vertex\s+ai|gcp\s+vertex|google\s+cloud)",
            r"\bno\s+(?:google\s+vertex|vertex\s+ai|gcp\s+vertex)\b",
        ),
        "aws-sagemaker": (
            r"\bdo\s+not\s+(?:use|run)\b[^.?!\n]*(?:aws\s+sagemaker|sagemaker)",
            r"\bno\s+(?:aws\s+sagemaker|sagemaker)\b",
        ),
        "hf-jobs": (
            r"\bdo\s+not\s+(?:use|run)\b[^.?!\n]*(?:hugging\s+face\s+jobs|hf\s+jobs|hf-jobs)",
            r"\bno\s+(?:hugging\s+face\s+jobs|hf\s+jobs|hf-jobs)\b",
        ),
    }
    for provider, patterns in rejection_patterns.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            rejected.add(provider)
    return rejected


def _resolve_cloud_provider_for_turn(
    selected_provider: str | None, text: str
) -> str | None:
    requested_provider = _requested_training_provider_from_text(text)
    if requested_provider:
        return requested_provider
    rejected = _rejected_training_providers_from_text(text)
    if selected_provider in rejected:
        for fallback_provider in ("gcp-vertex", "hf-jobs", "aws-sagemaker"):
            if fallback_provider not in rejected:
                return fallback_provider
        return None
    return selected_provider


def _uploaded_dataset_instruction(session: Session) -> str | None:
    uploads = [
        upload
        for upload in (getattr(session, "uploaded_datasets", []) or [])
        if isinstance(upload, dict)
    ]
    if not uploads:
        return None
    latest = uploads[-1]
    required_fields = ("config_name", "repo_id", "normalized_row_count")
    missing_fields = [
        field for field in required_fields if latest.get(field) in {None, ""}
    ]
    supports_training = latest.get("supports_training", True)
    named = ", ".join(
        str(upload.get("filename"))
        for upload in uploads
        if isinstance(upload.get("filename"), str)
    )
    incomplete_note = ""
    if missing_fields or supports_training is False or latest.get("status") == "failed":
        missing = (
            ", ".join(missing_fields) if missing_fields else "training-ready status"
        )
        incomplete_note = (
            " The uploaded dataset metadata is incomplete or not training-ready "
            f"({missing}); explain the missing dataset requirements and ask for "
            "the missing dataset details before launching training."
        )
    return (
        "The user has uploaded data for this session. For fine-tuning or "
        "training requests, first inspect and use the uploaded normalized "
        "dataset config. Prefer the latest upload unless the user mentions one "
        f"by name. Latest upload: filename={latest.get('filename')}, "
        f"source_format={latest.get('source_format')}, "
        f"dataset_config={latest.get('config_name')}, "
        f"normalized_rows={latest.get('normalized_row_count')}, "
        f"repo_id={latest.get('repo_id')}. "
        "Use the normalized dataset config for training. Do not ask for a "
        "local file path. Do not ask the user to upload again unless the "
        "dataset load fails."
        + incomplete_note
        + (f" Available uploads: {named}." if named else "")
    )


def _malformed_tool_name(message: Message) -> str | None:
    """Return the tool name for malformed-json tool-result messages."""
    if getattr(message, "role", None) != "tool":
        return None
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        return None
    if not content.startswith(_MALFORMED_TOOL_PREFIX):
        return None
    end = content.find(_MALFORMED_TOOL_SUFFIX, len(_MALFORMED_TOOL_PREFIX))
    if end == -1:
        return None
    return content[len(_MALFORMED_TOOL_PREFIX) : end]


def _detect_repeated_malformed(
    items: list[Message],
    threshold: int = 2,
) -> str | None:
    """Return the repeated malformed tool name if the tail contains a streak.

    Walk backward over the current conversation tail. A streak counts only
    consecutive malformed tool-result messages for the same tool; any other
    tool result breaks it.
    """
    if threshold <= 0:
        return None

    streak_tool: str | None = None
    streak = 0

    for item in reversed(items):
        if getattr(item, "role", None) != "tool":
            continue

        malformed_tool = _malformed_tool_name(item)
        if malformed_tool is None:
            break

        if streak_tool is None:
            streak_tool = malformed_tool
            streak = 1
        elif malformed_tool == streak_tool:
            streak += 1
        else:
            break

        if streak >= threshold:
            return streak_tool

    return None


def _validate_tool_args(tool_args: dict) -> tuple[bool, str | None]:
    """
    Validate tool arguments structure.

    Returns:
        (is_valid, error_message)
    """
    args = tool_args.get("args", {})
    # Sometimes LLM passes args as string instead of dict
    if isinstance(args, str):
        return (
            False,
            f"Tool call error: 'args' must be a JSON object, not a string. You passed: {repr(args)}",
        )
    if not isinstance(args, dict) and args is not None:
        return (
            False,
            f"Tool call error: 'args' must be a JSON object. You passed type: {type(args).__name__}",
        )
    return True, None


_IMMEDIATE_HF_JOB_RUNS = {"run", "uv"}
_IMMEDIATE_GCP_VERTEX_JOB_RUNS = {"run"}
_APPROVAL_REQUIRED_GCP_VERTEX_OPS = {"run", "cancel"}
_IMMEDIATE_AWS_SAGEMAKER_JOB_RUNS = {"run"}
_APPROVAL_REQUIRED_AWS_SAGEMAKER_OPS = {"run", "cancel"}
_APPROVAL_TTL_MINUTES = 30


@dataclass(frozen=True)
class ApprovalDecision:
    requires_approval: bool
    auto_approved: bool = False
    auto_approval_blocked: bool = False
    block_reason: str | None = None
    estimated_cost_usd: float | None = None
    remaining_cap_usd: float | None = None
    billable: bool = False


def _operation(tool_args: dict) -> str:
    return normalize_tool_operation(tool_args.get("operation"))


def _is_immediate_hf_job_run(tool_name: str, tool_args: dict) -> bool:
    return tool_name == "hf_jobs" and _operation(tool_args) in _IMMEDIATE_HF_JOB_RUNS


def _is_immediate_gcp_vertex_job_run(tool_name: str, tool_args: dict) -> bool:
    return (
        tool_name == "gcp_vertex_jobs"
        and _operation(tool_args) in _IMMEDIATE_GCP_VERTEX_JOB_RUNS
    )


def _is_immediate_aws_sagemaker_job_run(tool_name: str, tool_args: dict) -> bool:
    return (
        tool_name == "aws_sagemaker_jobs"
        and _operation(tool_args) in _IMMEDIATE_AWS_SAGEMAKER_JOB_RUNS
    )


def _is_gcp_vertex_cancel(tool_name: str, tool_args: dict) -> bool:
    return tool_name == "gcp_vertex_jobs" and _operation(tool_args) == "cancel"


def _is_aws_sagemaker_cancel(tool_name: str, tool_args: dict) -> bool:
    return tool_name == "aws_sagemaker_jobs" and _operation(tool_args) == "cancel"


def _is_immediate_cloud_job_run(tool_name: str, tool_args: dict) -> bool:
    return (
        _is_immediate_hf_job_run(tool_name, tool_args)
        or _is_immediate_gcp_vertex_job_run(tool_name, tool_args)
        or _is_immediate_aws_sagemaker_job_run(tool_name, tool_args)
    )


def _is_scheduled_hf_job_run(tool_name: str, tool_args: dict) -> bool:
    return tool_name == "hf_jobs" and is_scheduled_operation(_operation(tool_args))


def _is_budgeted_auto_approval_target(tool_name: str, tool_args: dict) -> bool:
    return tool_name == "sandbox_create" or _is_immediate_cloud_job_run(
        tool_name, tool_args
    )


def _base_needs_approval(
    tool_name: str, tool_args: dict, config: Config | None = None
) -> bool:
    """Check if a tool call requires approval before YOLO policy is applied."""

    # If args are malformed, skip approval (validation error will be shown later)
    args_valid, _ = _validate_tool_args(tool_args)
    if not args_valid:
        return False

    if tool_name == "sandbox_create":
        hardware = tool_args.get("hardware") or DEFAULT_CPU_SANDBOX_HARDWARE
        return hardware != DEFAULT_CPU_SANDBOX_HARDWARE

    if tool_name == "hf_jobs":
        operation = _operation(tool_args)
        if is_scheduled_operation(operation):
            return True
        if operation not in _IMMEDIATE_HF_JOB_RUNS:
            return False

        # Check if this is a CPU-only job
        # hardware_flavor is at top level of tool_args, not nested in args
        hardware_flavor = (
            tool_args.get("hardware_flavor")
            or tool_args.get("flavor")
            or tool_args.get("hardware")
            or "cpu-basic"
        )
        is_cpu_job = hardware_flavor in CPU_FLAVORS

        if is_cpu_job:
            if config and not config.confirm_cpu_jobs:
                return False
            return True

        return True

    if tool_name == "gcp_vertex_jobs":
        return _operation(tool_args) in _APPROVAL_REQUIRED_GCP_VERTEX_OPS

    if tool_name == "aws_sagemaker_jobs":
        return _operation(tool_args) in _APPROVAL_REQUIRED_AWS_SAGEMAKER_OPS

    # Check for file upload operations (hf_private_repos or other tools)
    if tool_name == "hf_private_repos":
        operation = tool_args.get("operation", "")
        if operation == "upload_file":
            if config and config.auto_file_upload:
                return False
            return True
        # Other operations (create_repo, etc.) always require approval
        if operation in ["create_repo"]:
            return True

    # hf_repo_files: upload (can overwrite) and delete require approval
    if tool_name == "hf_repo_files":
        operation = tool_args.get("operation", "")
        if operation in ["upload", "delete"]:
            return True

    # hf_repo_git: destructive operations require approval
    if tool_name == "hf_repo_git":
        operation = tool_args.get("operation", "")
        if operation in [
            "delete_branch",
            "delete_tag",
            "merge_pr",
            "create_repo",
            "update_repo",
        ]:
            return True

    return False


def _needs_approval(
    tool_name: str, tool_args: dict, config: Config | None = None
) -> bool:
    """Legacy sync approval predicate used by tests and CLI display helpers."""
    if _is_scheduled_hf_job_run(tool_name, tool_args):
        return True
    if _is_gcp_vertex_cancel(tool_name, tool_args):
        return True
    if _is_aws_sagemaker_cancel(tool_name, tool_args):
        return True
    if config and config.yolo_mode:
        return False
    return _base_needs_approval(tool_name, tool_args, config)


def _session_auto_approval_enabled(session: Session | None) -> bool:
    return bool(session and getattr(session, "auto_approval_enabled", False))


def _effective_yolo_enabled(session: Session | None, config: Config | None) -> bool:
    return bool(
        (config and config.yolo_mode) or _session_auto_approval_enabled(session)
    )


def _remaining_budget_after_reservations(
    session: Session | None, reserved_spend_usd: float
) -> float | None:
    if not session or getattr(session, "auto_approval_cost_cap_usd", None) is None:
        return None
    cap = float(getattr(session, "auto_approval_cost_cap_usd") or 0.0)
    spent = float(getattr(session, "auto_approval_estimated_spend_usd", 0.0) or 0.0)
    return round(max(0.0, cap - spent - reserved_spend_usd), 4)


def _budget_block_reason(
    estimate: CostEstimate,
    *,
    remaining_cap_usd: float | None,
) -> str | None:
    if estimate.estimated_cost_usd is None:
        return estimate.block_reason or "Could not estimate the cost safely."
    if (
        remaining_cap_usd is not None
        and estimate.estimated_cost_usd > remaining_cap_usd
    ):
        return (
            f"Estimated cost ${estimate.estimated_cost_usd:.2f} exceeds "
            f"remaining YOLO cap ${remaining_cap_usd:.2f}."
        )
    return None


async def _approval_decision(
    tool_name: str,
    tool_args: dict,
    session: Session,
    *,
    reserved_spend_usd: float = 0.0,
) -> ApprovalDecision:
    """Return the approval decision for one parsed tool call."""
    config = session.config
    base_requires_approval = _base_needs_approval(tool_name, tool_args, config)

    # Scheduled jobs are recurring/unbounded enough that YOLO never bypasses
    # the human confirmation, including legacy config.yolo_mode.
    if _is_scheduled_hf_job_run(tool_name, tool_args):
        return ApprovalDecision(
            requires_approval=True,
            auto_approval_blocked=_effective_yolo_enabled(session, config),
            block_reason="Scheduled HF jobs always require manual approval.",
        )

    if _is_gcp_vertex_cancel(tool_name, tool_args):
        return ApprovalDecision(
            requires_approval=True,
            auto_approval_blocked=_effective_yolo_enabled(session, config),
            block_reason="Vertex AI job cancellation always requires manual approval.",
        )

    if _is_aws_sagemaker_cancel(tool_name, tool_args):
        return ApprovalDecision(
            requires_approval=True,
            auto_approval_blocked=_effective_yolo_enabled(session, config),
            block_reason=(
                "SageMaker job cancellation always requires manual approval."
            ),
        )

    if _is_immediate_aws_sagemaker_job_run(
        tool_name, tool_args
    ) and _has_terminal_provider_job(session, "aws_sagemaker_jobs"):
        return ApprovalDecision(
            requires_approval=True,
            auto_approval_blocked=True,
            block_reason=(
                "A second paid AWS SageMaker run after a terminal job requires "
                "explicit manual approval."
            ),
        )

    yolo_enabled = _effective_yolo_enabled(session, config)
    session_yolo_enabled = _session_auto_approval_enabled(session)
    budgeted_target = _is_budgeted_auto_approval_target(tool_name, tool_args)
    if _is_immediate_gcp_vertex_job_run(
        tool_name, tool_args
    ) or _is_immediate_aws_sagemaker_job_run(tool_name, tool_args):
        estimate = await estimate_tool_cost(tool_name, tool_args, session=session)
        remaining = _remaining_budget_after_reservations(session, reserved_spend_usd)
        if not session_yolo_enabled:
            return ApprovalDecision(
                requires_approval=True,
                auto_approval_blocked=yolo_enabled,
                block_reason=(
                    "Cloud training run requires manual approval unless session "
                    "auto-approval with a cost cap is enabled."
                ),
                estimated_cost_usd=estimate.estimated_cost_usd,
                remaining_cap_usd=remaining,
                billable=estimate.billable,
            )
        reason = _budget_block_reason(estimate, remaining_cap_usd=remaining)
        if reason:
            return ApprovalDecision(
                requires_approval=True,
                auto_approval_blocked=True,
                block_reason=reason,
                estimated_cost_usd=estimate.estimated_cost_usd,
                remaining_cap_usd=remaining,
                billable=estimate.billable,
            )
        return ApprovalDecision(
            requires_approval=False,
            auto_approved=base_requires_approval,
            estimated_cost_usd=estimate.estimated_cost_usd,
            remaining_cap_usd=remaining,
            billable=estimate.billable,
        )

    if _is_immediate_hf_job_run(tool_name, tool_args):
        estimate = await estimate_tool_cost(tool_name, tool_args, session=session)
        remaining = _remaining_budget_after_reservations(session, reserved_spend_usd)
        if not session_yolo_enabled:
            return ApprovalDecision(
                requires_approval=True,
                auto_approval_blocked=yolo_enabled,
                block_reason=(
                    "HF Jobs run requires manual approval unless session "
                    "auto-approval with a cost cap is enabled."
                ),
                estimated_cost_usd=estimate.estimated_cost_usd,
                remaining_cap_usd=remaining,
                billable=estimate.billable,
            )
        reason = _budget_block_reason(estimate, remaining_cap_usd=remaining)
        if reason:
            return ApprovalDecision(
                requires_approval=True,
                auto_approval_blocked=True,
                block_reason=reason,
                estimated_cost_usd=estimate.estimated_cost_usd,
                remaining_cap_usd=remaining,
                billable=estimate.billable,
            )
        return ApprovalDecision(
            requires_approval=False,
            auto_approved=base_requires_approval,
            estimated_cost_usd=estimate.estimated_cost_usd,
            remaining_cap_usd=remaining,
            billable=estimate.billable,
        )

    # Cost caps are a session-scoped web policy. Legacy config.yolo_mode
    # remains uncapped for CLI/headless, except for scheduled jobs above.
    if yolo_enabled and budgeted_target and session_yolo_enabled:
        estimate = await estimate_tool_cost(tool_name, tool_args, session=session)
        remaining = _remaining_budget_after_reservations(session, reserved_spend_usd)
        reason = _budget_block_reason(estimate, remaining_cap_usd=remaining)
        if reason:
            return ApprovalDecision(
                requires_approval=True,
                auto_approval_blocked=True,
                block_reason=reason,
                estimated_cost_usd=estimate.estimated_cost_usd,
                remaining_cap_usd=remaining,
                billable=estimate.billable,
            )
        if base_requires_approval:
            return ApprovalDecision(
                requires_approval=False,
                auto_approved=True,
                estimated_cost_usd=estimate.estimated_cost_usd,
                remaining_cap_usd=remaining,
                billable=estimate.billable,
            )
        return ApprovalDecision(
            requires_approval=False,
            estimated_cost_usd=estimate.estimated_cost_usd,
            remaining_cap_usd=remaining,
            billable=estimate.billable,
        )

    if base_requires_approval and yolo_enabled:
        return ApprovalDecision(requires_approval=False, auto_approved=True)

    return ApprovalDecision(requires_approval=base_requires_approval)


def _record_estimated_spend(session: Session, decision: ApprovalDecision) -> None:
    if not decision.billable or decision.estimated_cost_usd is None:
        return
    if hasattr(session, "add_auto_approval_estimated_spend"):
        session.add_auto_approval_estimated_spend(decision.estimated_cost_usd)
    else:
        session.auto_approval_estimated_spend_usd = round(
            float(getattr(session, "auto_approval_estimated_spend_usd", 0.0) or 0.0)
            + float(decision.estimated_cost_usd),
            4,
        )


def _approval_metadata(
    session: Session, tool_name: str, tool_args: dict[str, Any]
) -> dict[str, Any] | None:
    provider_by_tool = {
        "hf_jobs": "hf-jobs",
        "gcp_vertex_jobs": "gcp-vertex",
        "aws_sagemaker_jobs": "aws-sagemaker",
    }
    provider = provider_by_tool.get(tool_name)
    if provider is None:
        return None
    metadata: dict[str, Any] = {
        "provider": provider,
        "training_goal": getattr(session, "training_goal", None),
        "output_policy": getattr(session, "output_policy", None),
    }
    for source_key, target_key in (
        ("model_name", "model"),
        ("model", "model"),
        ("hub_model_id", "hub_model_id"),
        ("dataset_name", "dataset"),
        ("dataset_repo", "dataset"),
        ("dataset_repo_id", "dataset"),
        ("dataset_config", "dataset_config"),
        ("dataset_split", "dataset_split"),
        ("hardware_flavor", "hardware"),
        ("hardware", "hardware"),
        ("instance_type", "instance_type"),
        ("instance_count", "instance_count"),
        ("max_run_seconds", "max_run_seconds"),
        ("output_policy", "output_policy"),
    ):
        value = tool_args.get(source_key)
        if value not in {None, ""} and target_key not in metadata:
            metadata[target_key] = value
    upload = _latest_uploaded_dataset(session)
    if upload:
        metadata.setdefault("dataset", upload.get("repo_id"))
        metadata.setdefault("dataset_config", upload.get("config_name"))
        metadata.setdefault("dataset_rows", upload.get("normalized_row_count"))
    return {key: value for key, value in metadata.items() if value not in {None, ""}}


def _approval_record(
    tc: ToolCall, tool_name: str, tool_args: dict[str, Any]
) -> dict[str, Any]:
    operation = normalize_tool_operation(tool_args.get("operation"))
    created_at = datetime.now(UTC)
    provider_by_tool = {
        "hf_jobs": "hf-jobs",
        "gcp_vertex_jobs": "gcp-vertex",
        "aws_sagemaker_jobs": "aws-sagemaker",
    }
    return {
        "approval_id": tc.id,
        "tool_call_id": tc.id,
        "tool": tool_name,
        "operation": operation,
        "provider": provider_by_tool.get(tool_name),
        "created_at": created_at.isoformat(),
        "expires_at": (
            created_at + timedelta(minutes=_APPROVAL_TTL_MINUTES)
        ).isoformat(),
        "status": "pending",
    }


def _pending_approval_records(session: Session) -> dict[str, dict[str, Any]]:
    pending = session.pending_approval or {}
    records = pending.get("approvals") or []
    return {
        str(record.get("tool_call_id")): record
        for record in records
        if isinstance(record, dict) and record.get("tool_call_id")
    }


def _is_expired_approval_record(record: dict[str, Any]) -> bool:
    try:
        expires_at = datetime.fromisoformat(str(record.get("expires_at")))
    except (TypeError, ValueError):
        return False
    return datetime.now(UTC) > expires_at


def _looks_like_typed_approval(text: str | None) -> bool:
    normalized = (text or "").strip().lower()
    return normalized in {"approve", "approved", "yes", "y", "ok", "go", "run it"}


async def _explain_typed_approval_not_launched(session: Session) -> str:
    records = list(_pending_approval_records(session).values())
    active_ids = [
        str(record.get("approval_id") or record.get("tool_call_id"))
        for record in records
        if not _is_expired_approval_record(record)
    ]
    suffix = f" Active approval ID: {active_ids[0]}." if len(active_ids) == 1 else ""
    message = (
        "I did not launch the pending job from a typed approval. Use the approval "
        "card so the exact tool call, provider, output policy, runtime, and cost "
        f"match the pending request.{suffix}"
    )
    await _emit_visible_assistant_message(session, message)
    await session.send_event(
        Event(
            event_type="error",
            data={
                "error": message,
                "error_type": "approval_recovery",
                "active": True,
            },
        )
    )
    return message


async def _record_manual_approved_spend_if_needed(
    session: Session,
    tool_name: str,
    tool_args: dict,
) -> None:
    if not _session_auto_approval_enabled(session):
        return
    if not _is_budgeted_auto_approval_target(tool_name, tool_args):
        return
    estimate = await estimate_tool_cost(tool_name, tool_args, session=session)
    _record_estimated_spend(
        session,
        ApprovalDecision(
            requires_approval=False,
            billable=estimate.billable,
            estimated_cost_usd=estimate.estimated_cost_usd,
        ),
    )


# -- LLM retry constants --------------------------------------------------
_MAX_LLM_RETRIES = 3
_LLM_RETRY_DELAYS = [5, 15, 30]  # seconds between retries
_LLM_RATE_LIMIT_RETRY_DELAYS = [30, 60]  # exceed Bedrock's ~60s TPM bucket window


def _is_rate_limit_error(error: Exception) -> bool:
    """Return True for rate-limit / quota-bucket style provider errors."""
    err_str = str(error).lower()
    rate_limit_patterns = [
        "429",
        "rate limit",
        "rate_limit",
        "too many requests",
        "too many tokens",
        "request limit",
        "throttl",
    ]
    return any(pattern in err_str for pattern in rate_limit_patterns)


def _is_context_overflow_error(error: Exception) -> bool:
    """Return True when the prompt exceeded the model's context window."""
    if isinstance(error, ContextWindowExceededError):
        return True

    err_str = str(error).lower()
    overflow_patterns = [
        "context window exceeded",
        "maximum context length",
        "max context length",
        "prompt is too long",
        "context length exceeded",
        "too many input tokens",
        "input is too long",
    ]
    return any(pattern in err_str for pattern in overflow_patterns)


def _retry_delay_for(error: Exception, attempt_index: int) -> int | None:
    """Return the delay for this retry attempt, or None if it should not retry."""
    if _is_rate_limit_error(error):
        schedule = _LLM_RATE_LIMIT_RETRY_DELAYS
    elif _is_transient_error(error):
        schedule = _LLM_RETRY_DELAYS
    else:
        return None

    if attempt_index >= len(schedule):
        return None
    return schedule[attempt_index]


def _is_transient_error(error: Exception) -> bool:
    """Return True for errors that are likely transient and worth retrying."""
    err_str = str(error).lower()
    transient_patterns = [
        "timeout",
        "timed out",
        "503",
        "service unavailable",
        "502",
        "bad gateway",
        "500",
        "internal server error",
        "overloaded",
        "capacity",
        "connection reset",
        "connection refused",
        "connection error",
        "eof",
        "broken pipe",
    ]
    return _is_rate_limit_error(error) or any(
        pattern in err_str for pattern in transient_patterns
    )


def _is_effort_config_error(error: Exception) -> bool:
    """Catch the two 400s the effort probe also handles — thinking
    unsupported for this model, or the specific effort level invalid.

    This is our safety net for the case where ``/effort`` was changed
    mid-conversation (which clears the probe cache) and the new level
    doesn't work for the current model. We heal the cache and retry once.
    """
    from agent.core.effort_probe import _is_invalid_effort, _is_thinking_unsupported

    return _is_thinking_unsupported(error) or _is_invalid_effort(error)


async def _heal_effort_and_rebuild_params(
    session: Session,
    error: Exception,
    llm_params: dict,
) -> dict:
    """Update the session's effort cache based on ``error`` and return new
    llm_params. Called only when ``_is_effort_config_error(error)`` is True.

    Two branches:
      • thinking-unsupported → cache ``None`` for this model, next call
        strips thinking entirely
      • invalid-effort → re-run the full cascade probe; the result lands
        in the cache
    """
    from agent.core.effort_probe import (
        ProbeInconclusive,
        _is_thinking_unsupported,
        probe_effort,
    )

    model = session.config.model_name
    if _is_thinking_unsupported(error):
        session.model_effective_effort[model] = None
        logger.info("healed: %s doesn't support thinking — stripped", model)
    else:
        try:
            outcome = await probe_effort(
                model,
                session.config.reasoning_effort,
                session.hf_token,
                session=session,
            )
            session.model_effective_effort[model] = outcome.effective_effort
            logger.info(
                "healed: %s effort cascade → %s",
                model,
                outcome.effective_effort,
            )
        except ProbeInconclusive:
            # Transient during healing — strip thinking for safety, next
            # call will either succeed or surface the real error.
            session.model_effective_effort[model] = None
            logger.info("healed: %s probe inconclusive — stripped", model)

    return _resolve_llm_params(
        model,
        session.hf_token,
        reasoning_effort=session.effective_effort_for(model),
    )


def _friendly_error_message(error: Exception) -> str | None:
    """Return a user-friendly message for known error types, or None to fall back to traceback."""
    err_str = str(error).lower()

    if (
        "authentication" in err_str
        or "unauthorized" in err_str
        or "invalid x-api-key" in err_str
    ):
        return (
            "Authentication failed — your API key is missing or invalid.\n\n"
            "To fix this, set the API key for your model provider:\n"
            "  • Anthropic:   export ANTHROPIC_API_KEY=sk-...\n"
            "  • OpenAI:      export OPENAI_API_KEY=sk-...\n"
            "  • HF Router:   export HF_TOKEN=hf_...\n\n"
            "You can also add it to a .env file in the project root.\n"
            "To switch models, use the /model command."
        )

    if "insufficient" in err_str and "credit" in err_str:
        return (
            "Insufficient API credits. Please check your account balance "
            "at your model provider's dashboard."
        )

    if "not supported by provider" in err_str or "no provider supports" in err_str:
        return (
            "The model isn't served by the provider you pinned.\n\n"
            "Drop the ':<provider>' suffix to let the HF router auto-pick a "
            "provider, or use '/model' (no arg) to see which providers host "
            "which models."
        )

    if "model_not_found" in err_str or (
        "model" in err_str and ("not found" in err_str or "does not exist" in err_str)
    ):
        return (
            "Model not found. Use '/model' to list suggestions, or paste an "
            "HF model id like 'MiniMaxAI/MiniMax-M2.7'. Availability is shown "
            "when you switch."
        )

    return None


def _llm_error_type(error: Exception) -> str:
    """Classify provider failures for structured UI handling."""

    err_str = str(error).lower()
    quota_billing_patterns = (
        "quota",
        "billing",
        "credit",
        "insufficient",
        "spending limit",
        "monthly spending",
        "exceeded your monthly",
    )
    if any(
        pattern in err_str
        for pattern in (
            "401",
            "403",
            "authentication",
            "unauthorized",
            "forbidden",
            "invalid x-api-key",
            "invalid api key",
            "api key",
            "permission",
        )
    ):
        if any(pattern in err_str for pattern in quota_billing_patterns):
            return "quota"
        return "auth"
    if any(pattern in err_str for pattern in quota_billing_patterns):
        return "quota"
    if _is_rate_limit_error(error):
        return "rate_limit"
    if any(
        pattern in err_str
        for pattern in (
            "timeout",
            "timed out",
            "network",
            "connection",
            "eof",
            "broken pipe",
            "service unavailable",
            "bad gateway",
        )
    ):
        return "network"
    return "unknown"


def _llm_failure_message(
    error: Exception,
    *,
    model_name: str,
    include_traceback: bool = False,
) -> tuple[str, str]:
    """Return (error_type, visible message) for failed LLM calls."""

    error_type = _llm_error_type(error)
    raw = str(error).strip()
    if error_type == "quota":
        message = (
            "The selected model provider rejected the request because of quota, "
            "billing, or credits. Switch to another model, or fix the provider "
            "quota/billing state, then retry.\n\n"
            f"Model: {model_name}\nProvider error: {raw}"
        )
    elif error_type == "auth":
        message = (
            "The selected model provider rejected the request because credentials "
            "or permissions are missing or invalid. Switch to a configured model "
            "or update the provider credentials, then retry.\n\n"
            f"Model: {model_name}\nProvider error: {raw}"
        )
    elif error_type == "rate_limit":
        message = (
            "The selected model provider rate-limited this request. Wait a bit, "
            "switch to another model, or retry with a shorter request.\n\n"
            f"Model: {model_name}\nProvider error: {raw}"
        )
    elif error_type == "network":
        message = (
            "The selected model provider could not be reached reliably. Retry, "
            "or switch models if the provider stays unavailable.\n\n"
            f"Model: {model_name}\nProvider error: {raw}"
        )
    else:
        friendly = _friendly_error_message(error)
        if friendly:
            message = friendly
        else:
            message = (
                "The selected model failed before returning a usable response. "
                "No tool was launched. Switch models or retry after checking the "
                "provider status.\n\n"
                f"Model: {model_name}\nError: {raw}"
            )
            if include_traceback:
                import traceback

                message += "\n\n" + traceback.format_exc()
    return error_type, message


def _empty_llm_response_message(*, model_name: str, finish_reason: str | None) -> str:
    reason = f" Finish reason: {finish_reason}." if finish_reason else ""
    return (
        "The selected model returned an empty response with no tool calls, so I "
        "stopped instead of showing a blank assistant message. Switch to another "
        "model or retry the request; if it repeats, the provider may be having "
        f"issues.\n\nModel: {model_name}.{reason}"
    )


_HF_TRAINING_INTENT_TERMS = (
    "train",
    "training",
    "fine-tune",
    "finetune",
    "fine tune",
    "sft",
    "model adaptation",
)


def _last_tool_name(session: Session) -> str | None:
    for item in reversed(session.context_manager.items):
        if getattr(item, "role", None) == "tool":
            name = getattr(item, "name", None)
            return str(name) if name else None
    return None


def _latest_uploaded_dataset(session: Session) -> dict[str, Any] | None:
    uploads = [
        upload
        for upload in (getattr(session, "uploaded_datasets", []) or [])
        if isinstance(upload, dict)
    ]
    return uploads[-1] if uploads else None


def _has_training_intent(text: str | None) -> bool:
    haystack = (text or "").lower()
    return any(term in haystack for term in _HF_TRAINING_INTENT_TERMS)


def _should_emit_hf_planner_fallback(session: Session, text: str | None) -> bool:
    if getattr(session, "cloud_provider", "hf-jobs") != "hf-jobs":
        return False
    if _last_tool_name(session) != "training_planner":
        return False
    if not _has_training_intent(text):
        return False
    return _latest_uploaded_dataset(session) is not None


def _latest_training_preflight(session: Session) -> dict[str, Any] | None:
    preflight = getattr(session, "latest_training_preflight", None)
    return preflight if isinstance(preflight, dict) else None


def _bounded_vertex_smoke_workflow_active(session: Session) -> bool:
    if getattr(session, "cloud_provider", "hf-jobs") != "gcp-vertex":
        return False
    if getattr(session, "training_goal", "agent-decide") != "smoke-test":
        return False
    if not getattr(session, "bounded_vertex_smoke_for_turn", False):
        return False
    if _has_active_provider_job(
        session, "gcp_vertex_jobs"
    ) or _has_terminal_provider_job(session, "gcp_vertex_jobs"):
        return False
    discovery = getattr(session, "latest_dataset_discovery", None)
    recommendation = getattr(session, "latest_training_recommendation", None)
    return (
        isinstance(discovery, dict)
        and bool(discovery)
        and isinstance(recommendation, dict)
        and bool(recommendation)
    )


def _should_continue_vertex_smoke_launch(session: Session) -> bool:
    if getattr(session, "cloud_provider", "hf-jobs") != "gcp-vertex":
        return False
    if getattr(session, "training_goal", "agent-decide") != "smoke-test":
        return False
    preflight = _latest_training_preflight(session)
    if not isinstance(preflight, dict):
        return False
    if preflight.get("manual_approval_allowed") is not True:
        return False
    if preflight.get("launch_ready") is True:
        return False
    blocking = preflight.get("blocking_reasons")
    if isinstance(blocking, list) and blocking:
        return False
    if _has_active_provider_job(
        session, "gcp_vertex_jobs"
    ) or _has_terminal_provider_job(session, "gcp_vertex_jobs"):
        return False
    return True


def _should_avoid_planning_only_stop(session: Session) -> bool:
    return _bounded_vertex_smoke_workflow_active(
        session
    ) or _should_continue_vertex_smoke_launch(session)


def _vertex_smoke_continuation_prompt(session: Session) -> str:
    preflight = _latest_training_preflight(session) or {}
    manual_reason = str(preflight.get("manual_approval_reason") or "").strip()
    reason_text = f"{manual_reason} " if manual_reason else ""
    return (
        "[SYSTEM: VERTEX SMOKE CONTINUATION] Live preflight returned "
        "manual_approval_allowed=true with launch_ready=false because "
        "quota/accelerator checks remain unknown. "
        f"{reason_text}"
        "Preflight has unknown quota/accelerator checks, so launch is not "
        "automatically ready. Because this is a bounded smoke test with "
        "cost/runtime guardrails, proceed now by calling gcp_vertex_jobs with "
        "operation=run using the verified training recommendation. Do not "
        "auto-launch; the approval card must appear for explicit user approval. "
        "Do not return a planning-only summary."
    )


def _vertex_smoke_after_planner_prompt(session: Session) -> str:
    if _should_continue_vertex_smoke_launch(session):
        return _vertex_smoke_continuation_prompt(session)
    return (
        "[SYSTEM: VERTEX SMOKE CONTINUATION] Dataset discovery and training "
        "planner are complete for the bounded Google Vertex AI smoke workflow. "
        "Continue toward the approval-gated launch: once live preflight shows "
        "manual_approval_allowed=true with bounded cost/runtime, call "
        "gcp_vertex_jobs with operation=run immediately so the user can approve "
        "the bounded launch. Do not treat unknown quota/accelerator as passed "
        "and do not set launch_ready=true."
    )


def _should_emit_vertex_smoke_fallback(session: Session, text: str | None) -> bool:
    if getattr(session, "cloud_provider", "hf-jobs") != "gcp-vertex":
        return False
    if not _has_training_intent(text):
        return False
    preflight = _latest_training_preflight(session)
    if not isinstance(preflight, dict):
        return False
    if preflight.get("manual_approval_allowed") is True:
        return True
    last_tool = _last_tool_name(session)
    return last_tool in {"training_planner", "training_preflight"}


def _hf_planner_fallback_message(session: Session) -> str:
    upload = _latest_uploaded_dataset(session) or {}
    dataset_parts = []
    if upload.get("repo_id"):
        dataset_parts.append(f"repo `{upload.get('repo_id')}`")
    if upload.get("config_name"):
        dataset_parts.append(f"config `{upload.get('config_name')}`")
    if upload.get("normalized_row_count") is not None:
        dataset_parts.append(f"{upload.get('normalized_row_count')} normalized rows")
    dataset = ", ".join(dataset_parts) or "the uploaded normalized dataset"
    return (
        "I prepared the Hugging Face training plan, but the model stopped before "
        "launching the Hugging Face Jobs preflight. I can continue with the planned "
        f"Hugging Face fine-tuning workflow using {dataset}.\n\n"
        f"Preflight context: provider `hf-jobs`, training goal "
        f"`{getattr(session, 'training_goal', 'agent-decide')}`, output policy "
        f"`{getattr(session, 'output_policy', 'cloud-and-hf-hub')}`. The next "
        "approval-gated step is `hf_jobs`; before any job launches, I will show "
        "the `hf_jobs` approval card and wait for explicit approval."
    )


def _vertex_smoke_fallback_message(session: Session) -> str:
    preflight = _latest_training_preflight(session) or {}
    manual_allowed = preflight.get("manual_approval_allowed") is True
    manual_reason = str(preflight.get("manual_approval_reason") or "").strip()
    preflight_note = (
        f"{manual_reason} "
        if manual_reason
        else "Live preflight kept quota/accelerator unknown. "
    )
    approval_note = (
        "Preflight has unknowns; bounded smoke can proceed only with explicit approval. "
        if manual_allowed
        else "Run live preflight first, then proceed only if bounded smoke approval is allowed. "
    )
    return (
        "I prepared the Google Vertex AI training plan"
        + (" and live preflight" if preflight else "")
        + ", but the model stopped before launching the approval-gated Vertex job. "
        f"{preflight_note}{approval_note}"
        f"Preflight context: provider `gcp-vertex`, training goal "
        f"`{getattr(session, 'training_goal', 'agent-decide')}`, output policy "
        f"`{getattr(session, 'output_policy', 'cloud-and-hf-hub')}`. "
        "The next approval-gated step is `gcp_vertex_jobs` run; I will show the "
        "Vertex pricing/provider approval card and wait for explicit approval before "
        "launching any job."
    )


async def _emit_visible_error(
    session: Session,
    message: str,
    *,
    error_type: str,
    model_name: str | None = None,
) -> None:
    """Emit an assistant-visible message and a structured error event."""

    assistant_msg = Message(role="assistant", content=message)
    session.context_manager.add_message(assistant_msg)
    await session.send_event(
        Event(event_type="assistant_message", data={"content": message})
    )
    await session.send_event(
        Event(
            event_type="error",
            data={
                "error": message,
                "error_type": error_type,
                "model": model_name or session.config.model_name,
                "session_id": session.session_id,
                "turn_id": session.turn_count,
                "timestamp": datetime.now(UTC).isoformat(),
                "transient": error_type
                in {"rate_limit", "network", "empty_response", "unknown"},
                "active": True,
            },
        )
    )


async def _emit_visible_assistant_message(session: Session, message: str) -> None:
    assistant_msg = Message(role="assistant", content=message)
    session.context_manager.add_message(assistant_msg)
    await session.send_event(
        Event(event_type="assistant_message", data={"content": message})
    )


def _planning_only_completion_message(session: Session) -> str | None:
    if _should_avoid_planning_only_stop(session):
        return None
    if not getattr(session, "compute_tools_blocked_for_turn", False):
        return None

    discovery = getattr(session, "latest_dataset_discovery", None)
    if not isinstance(discovery, dict) or not discovery:
        return None

    candidates = discovery.get("candidates")
    candidates = candidates if isinstance(candidates, list) else []
    excluded_sources = discovery.get("excluded_sources")
    excluded_sources = excluded_sources if isinstance(excluded_sources, list) else []
    warnings = discovery.get("warnings")
    warnings = warnings if isinstance(warnings, list) else []
    recommended = discovery.get("recommended_candidate")
    recommendation = getattr(session, "latest_training_recommendation", None)
    if not isinstance(recommendation, dict) or not recommendation:
        return None
    risks = recommendation.get("risks")
    risks = risks if isinstance(risks, list) and risks else []
    no_candidates_reason = discovery.get("no_candidates_reason")
    no_candidates_reason = (
        no_candidates_reason if isinstance(no_candidates_reason, str) else None
    )

    lines = [
        "Planning/discovery complete. No datasets were uploaded or downloaded, "
        "no sandbox was created, no provider jobs were launched, and no cloud "
        "resources were created.",
        "",
        f"Dataset candidates found: {len(candidates)}.",
    ]
    if candidates:
        lines.extend(
            f"- {candidate.get('title') or candidate.get('dataset_id') or 'Unnamed dataset'}"
            for candidate in candidates
            if isinstance(candidate, dict)
        )
    else:
        lines.append(
            f"- {no_candidates_reason or 'No candidate datasets were available from the current no-upload discovery inputs.'}"
        )

    if isinstance(recommended, dict):
        lines.extend(
            [
                "",
                "Recommended candidate: "
                f"{recommended.get('title') or recommended.get('dataset_id')}.",
            ]
        )

    lines.extend(
        [
            "",
            "Risks and warnings:",
            *[f"- {risk}" for risk in risks],
            *[f"- {warning}" for warning in warnings],
            "",
            "Excluded sources:",
            *[f"- {source}" for source in excluded_sources],
            "",
            "Before launch: the user must explicitly select a dataset, approve "
            "any upload/download or provider job, and approve any billable "
            "cloud or sandbox resource creation.",
        ]
    )
    return "\n".join(lines)


async def _compact_and_notify(session: Session) -> None:
    """Run compaction and send event if context was reduced.

    Catches ``CompactionFailedError`` and ends the session cleanly instead
    of letting the caller retry. Pre-2026-05-04 the caller looped on
    ContextWindowExceededError → compact → re-trigger, burning Bedrock
    budget at ~$3/Opus retry while the session never reached the upload
    path (so the cost was invisible in the dataset).
    """
    from agent.context_manager.manager import CompactionFailedError

    cm = session.context_manager
    old_usage = cm.running_context_usage
    logger.debug(
        "Compaction check: usage=%d, max=%d, threshold=%d, needs_compact=%s",
        old_usage,
        cm.model_max_tokens,
        cm.compaction_threshold,
        cm.needs_compaction,
    )
    try:
        await cm.compact(
            model_name=session.config.model_name,
            tool_specs=session.tool_router.get_tool_specs_for_llm(),
            hf_token=session.hf_token,
            session=session,
        )
    except CompactionFailedError as e:
        logger.error(
            "Compaction failed for session %s: %s — terminating session",
            session.session_id,
            e,
        )
        # Persist the failure event so the dataset has a record of WHY this
        # session ended (and the cost it incurred up to that point) even if
        # save_and_upload_detached has issues downstream.
        await session.send_event(
            Event(
                event_type="session_terminated",
                data={
                    "reason": "compaction_failed",
                    "context_usage": cm.running_context_usage,
                    "context_threshold": cm.compaction_threshold,
                    "error": str(e)[:300],
                    "user_message": (
                        "Your conversation has grown too large to continue. "
                        "The work you've done is saved — start a new session to keep going."
                    ),
                },
            )
        )
        # Stop the agent loop; the finally in _run_session will fire
        # cleanup_sandbox + save_trajectory so the dataset captures
        # everything that did happen.
        session.is_running = False
        return

    new_usage = cm.running_context_usage
    if new_usage != old_usage:
        logger.warning(
            "Context compacted: %d -> %d tokens (max=%d, %d messages)",
            old_usage,
            new_usage,
            cm.model_max_tokens,
            len(cm.items),
        )
        await session.send_event(
            Event(
                event_type="compacted",
                data={"old_tokens": old_usage, "new_tokens": new_usage},
            )
        )


async def _cleanup_on_cancel(session: Session) -> None:
    """Kill sandbox processes and cancel HF jobs when the user interrupts."""
    # Kill active sandbox processes
    sandbox = getattr(session, "sandbox", None)
    if sandbox:
        try:
            await asyncio.to_thread(sandbox.kill_all)
            logger.info("Killed sandbox processes on cancel")
        except Exception as e:
            logger.warning("Failed to kill sandbox processes: %s", e)

    # Cancel running HF jobs
    job_ids = list(session._running_job_ids)
    if job_ids:
        from huggingface_hub import HfApi

        api = HfApi(token=session.hf_token)
        for job_id in job_ids:
            try:
                await asyncio.to_thread(api.cancel_job, job_id=job_id)
                logger.info("Cancelled HF job %s on interrupt", job_id)
            except Exception as e:
                logger.warning("Failed to cancel HF job %s: %s", job_id, e)
        session._running_job_ids.clear()


@dataclass
class LLMResult:
    """Result from an LLM call (streaming or non-streaming)."""

    content: str | None
    tool_calls_acc: dict[int, dict]
    token_count: int
    finish_reason: str | None
    usage: dict = field(default_factory=dict)
    thinking_blocks: list[dict[str, Any]] | None = None
    reasoning_content: str | None = None


def _extract_thinking_state(
    message: Any,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Return provider reasoning fields that must be replayed after tool calls."""
    provider_fields = getattr(message, "provider_specific_fields", None)
    if not isinstance(provider_fields, dict):
        provider_fields = {}

    thinking_blocks = (
        getattr(message, "thinking_blocks", None)
        or provider_fields.get("thinking_blocks")
        or None
    )
    reasoning_content = (
        getattr(message, "reasoning_content", None)
        or provider_fields.get("reasoning_content")
        or None
    )
    return thinking_blocks, reasoning_content


def _should_replay_thinking_state(model_name: str | None) -> bool:
    """Only Anthropic's native adapter accepts replayed thinking metadata."""
    return bool(model_name and model_name.startswith("anthropic/"))


def _is_invalid_thinking_signature_error(exc: Exception) -> bool:
    """Return True when Anthropic rejected replayed extended-thinking state."""
    text = str(exc)
    return (
        "Invalid `signature` in `thinking` block" in text
        or "Invalid signature in thinking block" in text
    )


def _strip_thinking_state_from_messages(messages: list[Any]) -> int:
    """Remove replayed thinking metadata from assistant history messages."""
    stripped = 0

    for message in messages:
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        if role != "assistant":
            continue

        if isinstance(message, dict):
            if message.pop("thinking_blocks", None) is not None:
                stripped += 1
            if message.pop("reasoning_content", None) is not None:
                stripped += 1
            provider_fields = message.get("provider_specific_fields")
            content = message.get("content")
        else:
            if getattr(message, "thinking_blocks", None) is not None:
                message.thinking_blocks = None
                stripped += 1
            if getattr(message, "reasoning_content", None) is not None:
                message.reasoning_content = None
                stripped += 1
            provider_fields = getattr(message, "provider_specific_fields", None)
            content = getattr(message, "content", None)

        if isinstance(provider_fields, dict):
            cleaned_fields = dict(provider_fields)
            if cleaned_fields.pop("thinking_blocks", None) is not None:
                stripped += 1
            if cleaned_fields.pop("reasoning_content", None) is not None:
                stripped += 1
            if cleaned_fields != provider_fields:
                if isinstance(message, dict):
                    message["provider_specific_fields"] = cleaned_fields
                else:
                    message.provider_specific_fields = cleaned_fields

        if isinstance(content, list):
            cleaned_content = [
                block
                for block in content
                if not (
                    isinstance(block, dict)
                    and block.get("type") in {"thinking", "redacted_thinking"}
                )
            ]
            if len(cleaned_content) != len(content):
                stripped += len(content) - len(cleaned_content)
                if isinstance(message, dict):
                    message["content"] = cleaned_content
                else:
                    message.content = cleaned_content

    return stripped


async def _maybe_heal_invalid_thinking_signature(
    session: Session,
    messages: list[Any],
    exc: Exception,
    *,
    already_healed: bool,
) -> bool:
    if already_healed or not _is_invalid_thinking_signature_error(exc):
        return False

    stripped = _strip_thinking_state_from_messages(messages)
    if not stripped:
        return False

    await session.send_event(
        Event(
            event_type="tool_log",
            data={
                "tool": "system",
                "log": (
                    "Anthropic rejected stale thinking signatures; retrying "
                    "without replayed thinking metadata."
                ),
            },
        )
    )
    return True


def _assistant_message_from_result(
    llm_result: LLMResult,
    *,
    model_name: str | None,
    tool_calls: list[ToolCall] | None = None,
) -> Message:
    """Build an assistant history message without dropping reasoning state."""
    kwargs: dict[str, Any] = {
        "role": "assistant",
        "content": llm_result.content,
    }
    if tool_calls is not None:
        kwargs["tool_calls"] = tool_calls
    if _should_replay_thinking_state(model_name):
        if llm_result.thinking_blocks:
            kwargs["thinking_blocks"] = llm_result.thinking_blocks
        if llm_result.reasoning_content:
            kwargs["reasoning_content"] = llm_result.reasoning_content
    return Message(**kwargs)


async def _call_llm_streaming(
    session: Session, messages, tools, llm_params
) -> LLMResult:
    """Call the LLM with streaming, emitting assistant_chunk events."""
    response = None
    _healed_effort = False  # one-shot safety net per call
    _healed_thinking_signature = False
    messages, tools = with_prompt_caching(messages, tools, llm_params.get("model"))
    t_start = time.monotonic()
    for _llm_attempt in range(_MAX_LLM_RETRIES):
        try:
            response = await acompletion(
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=True,
                stream_options={"include_usage": True},
                timeout=600,
                **llm_params,
            )
            break
        except ContextWindowExceededError:
            raise
        except Exception as e:
            if _is_context_overflow_error(e):
                raise ContextWindowExceededError(str(e)) from e
            if not _healed_effort and _is_effort_config_error(e):
                _healed_effort = True
                llm_params = await _heal_effort_and_rebuild_params(
                    session, e, llm_params
                )
                await session.send_event(
                    Event(
                        event_type="tool_log",
                        data={
                            "tool": "system",
                            "log": "Reasoning effort not supported for this model — adjusting and retrying.",
                        },
                    )
                )
                continue
            if await _maybe_heal_invalid_thinking_signature(
                session,
                messages,
                e,
                already_healed=_healed_thinking_signature,
            ):
                _healed_thinking_signature = True
                continue
            _delay = _retry_delay_for(e, _llm_attempt)
            if _llm_attempt < _MAX_LLM_RETRIES - 1 and _delay is not None:
                logger.warning(
                    "Transient LLM error (attempt %d/%d): %s — retrying in %ds",
                    _llm_attempt + 1,
                    _MAX_LLM_RETRIES,
                    e,
                    _delay,
                )
                await session.send_event(
                    Event(
                        event_type="tool_log",
                        data={
                            "tool": "system",
                            "log": f"LLM connection error, retrying in {_delay}s...",
                        },
                    )
                )
                await asyncio.sleep(_delay)
                continue
            raise

    full_content = ""
    tool_calls_acc: dict[int, dict] = {}
    token_count = 0
    finish_reason = None
    final_usage_chunk = None
    chunks = []
    should_replay_thinking = _should_replay_thinking_state(llm_params.get("model"))

    async for chunk in response:
        chunks.append(chunk)
        if session.is_cancelled:
            tool_calls_acc.clear()
            break

        choice = chunk.choices[0] if chunk.choices else None
        if not choice:
            if hasattr(chunk, "usage") and chunk.usage:
                token_count = chunk.usage.total_tokens
                final_usage_chunk = chunk
            continue

        delta = choice.delta
        if choice.finish_reason:
            finish_reason = choice.finish_reason

        if delta.content:
            full_content += delta.content
            await session.send_event(
                Event(event_type="assistant_chunk", data={"content": delta.content})
            )

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc_delta.id:
                    tool_calls_acc[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tool_calls_acc[idx]["function"]["name"] += (
                            tc_delta.function.name
                        )
                    if tc_delta.function.arguments:
                        tool_calls_acc[idx]["function"]["arguments"] += (
                            tc_delta.function.arguments
                        )

        if hasattr(chunk, "usage") and chunk.usage:
            token_count = chunk.usage.total_tokens
            final_usage_chunk = chunk

    usage = await telemetry.record_llm_call(
        session,
        model=llm_params.get("model", session.config.model_name),
        response=final_usage_chunk,
        latency_ms=int((time.monotonic() - t_start) * 1000),
        finish_reason=finish_reason,
    )
    thinking_blocks = None
    reasoning_content = None
    if chunks and should_replay_thinking:
        try:
            rebuilt = stream_chunk_builder(chunks, messages=messages)
            if rebuilt and getattr(rebuilt, "choices", None):
                rebuilt_msg = rebuilt.choices[0].message
                thinking_blocks, reasoning_content = _extract_thinking_state(
                    rebuilt_msg
                )
        except Exception:
            logger.debug("Failed to rebuild streaming thinking state", exc_info=True)

    return LLMResult(
        content=full_content or None,
        tool_calls_acc=tool_calls_acc,
        token_count=token_count,
        finish_reason=finish_reason,
        usage=usage,
        thinking_blocks=thinking_blocks,
        reasoning_content=reasoning_content,
    )


async def _call_llm_non_streaming(
    session: Session, messages, tools, llm_params
) -> LLMResult:
    """Call the LLM without streaming, emit assistant_message at the end."""
    response = None
    _healed_effort = False
    _healed_thinking_signature = False
    messages, tools = with_prompt_caching(messages, tools, llm_params.get("model"))
    t_start = time.monotonic()
    for _llm_attempt in range(_MAX_LLM_RETRIES):
        try:
            response = await acompletion(
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=False,
                timeout=600,
                **llm_params,
            )
            break
        except ContextWindowExceededError:
            raise
        except Exception as e:
            if _is_context_overflow_error(e):
                raise ContextWindowExceededError(str(e)) from e
            if not _healed_effort and _is_effort_config_error(e):
                _healed_effort = True
                llm_params = await _heal_effort_and_rebuild_params(
                    session, e, llm_params
                )
                await session.send_event(
                    Event(
                        event_type="tool_log",
                        data={
                            "tool": "system",
                            "log": "Reasoning effort not supported for this model — adjusting and retrying.",
                        },
                    )
                )
                continue
            if await _maybe_heal_invalid_thinking_signature(
                session,
                messages,
                e,
                already_healed=_healed_thinking_signature,
            ):
                _healed_thinking_signature = True
                continue
            _delay = _retry_delay_for(e, _llm_attempt)
            if _llm_attempt < _MAX_LLM_RETRIES - 1 and _delay is not None:
                logger.warning(
                    "Transient LLM error (attempt %d/%d): %s — retrying in %ds",
                    _llm_attempt + 1,
                    _MAX_LLM_RETRIES,
                    e,
                    _delay,
                )
                await session.send_event(
                    Event(
                        event_type="tool_log",
                        data={
                            "tool": "system",
                            "log": f"LLM connection error, retrying in {_delay}s...",
                        },
                    )
                )
                await asyncio.sleep(_delay)
                continue
            raise

    choice = response.choices[0]
    message = choice.message
    content = message.content or None
    finish_reason = choice.finish_reason
    token_count = response.usage.total_tokens if response.usage else 0
    thinking_blocks, reasoning_content = _extract_thinking_state(message)

    # Build tool_calls_acc in the same format as streaming
    tool_calls_acc: dict[int, dict] = {}
    if message.tool_calls:
        for idx, tc in enumerate(message.tool_calls):
            tool_calls_acc[idx] = {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }

    # Emit the full message as a single event
    if content:
        await session.send_event(
            Event(event_type="assistant_message", data={"content": content})
        )

    usage = await telemetry.record_llm_call(
        session,
        model=llm_params.get("model", session.config.model_name),
        response=response,
        latency_ms=int((time.monotonic() - t_start) * 1000),
        finish_reason=finish_reason,
    )

    return LLMResult(
        content=content,
        tool_calls_acc=tool_calls_acc,
        token_count=token_count,
        finish_reason=finish_reason,
        usage=usage,
        thinking_blocks=thinking_blocks,
        reasoning_content=reasoning_content,
    )


class Handlers:
    """Handler functions for each operation type"""

    @staticmethod
    async def _abandon_pending_approval(session: Session) -> None:
        """Cancel pending approval tools when the user continues the conversation.

        Injects rejection tool-result messages into the LLM context (so the
        history stays valid) and notifies the frontend that those tools were
        abandoned.
        """
        tool_calls = session.pending_approval.get("tool_calls", [])
        for tc in tool_calls:
            tool_name = tc.function.name
            abandon_msg = (
                "Task abandoned — user continued the conversation without approving."
            )

            # Keep LLM context valid: every tool_call needs a tool result
            tool_msg = Message(
                role="tool",
                content=abandon_msg,
                tool_call_id=tc.id,
                name=tool_name,
            )
            session.context_manager.add_message(tool_msg)

            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": tool_name,
                        "state": "abandoned",
                    },
                )
            )

        session.pending_approval = None
        logger.info("Abandoned %d pending approval tool(s)", len(tool_calls))

    @staticmethod
    async def run_agent(
        session: Session,
        text: str,
    ) -> str | None:
        """
        Handle user input (like user_input_or_turn in codex.rs:1291)
        Returns the final assistant response content, if any.
        """
        # Clear any stale cancellation flag from a previous run
        session.reset_cancel()

        if text and session.pending_approval and _looks_like_typed_approval(text):
            return await _explain_typed_approval_not_launched(session)

        # If there's a pending approval and the user sent a new message,
        # abandon the pending tools so the LLM context stays valid.
        if text and session.pending_approval:
            await Handlers._abandon_pending_approval(session)

        # Add user message to history only if there's actual content
        if text:
            user_msg = Message(role="user", content=text)
            session.context_manager.add_message(user_msg)

        # Send event that we're processing
        await session.send_event(
            Event(event_type="processing", data={"message": "Processing user input"})
        )

        # Agentic loop - continue until model doesn't call tools or max iterations is reached
        iteration = 0
        final_response = None
        errored = False
        max_iterations = session.config.max_iterations
        no_tool_incomplete_plan_retries = 0
        vertex_smoke_continuation_retries = 0

        while max_iterations == -1 or iteration < max_iterations:
            # ── Cancellation check: before LLM call ──
            if session.is_cancelled:
                break

            # Compact before calling the LLM if context is near the limit.
            # When _compact_and_notify catches CompactionFailedError it sets
            # session.is_running = False; we MUST exit the loop here, otherwise
            # the LLM call below fires with an over-threshold context, hits
            # ContextWindowExceededError, and we end up looping again on the
            # except path — exactly the bug this PR is supposed to fix.
            await _compact_and_notify(session)
            if not session.is_running:
                break

            # Doom-loop detection: break out of repeated tool call patterns
            doom_prompt = check_for_doom_loop(session.context_manager.items)
            if doom_prompt:
                session.context_manager.add_message(
                    Message(role="user", content=doom_prompt)
                )

            malformed_tool = _detect_repeated_malformed(session.context_manager.items)
            if malformed_tool:
                recovery_prompt = (
                    "[SYSTEM: Repeated malformed tool arguments detected for "
                    f"'{malformed_tool}'. Stop retrying the same tool call shape. "
                    "Use a different strategy that produces smaller, valid JSON. "
                    "For large file writes, prefer bash with a heredoc or split the "
                    "edit into multiple smaller tool calls.]"
                )
                session.context_manager.add_message(
                    Message(role="user", content=recovery_prompt)
                )
                await session.send_event(
                    Event(
                        event_type="tool_log",
                        data={
                            "tool": "system",
                            "log": (
                                "Repeated malformed tool arguments detected — "
                                f"forcing a different strategy for {malformed_tool}"
                            ),
                        },
                    )
                )

            messages = session.context_manager.get_messages()
            tools = session.tool_router.get_tool_specs_for_llm()
            try:
                # ── Call the LLM (streaming or non-streaming) ──
                # Pull the per-model probed effort from the session cache when
                # available; fall back to the raw preference for models we
                # haven't probed yet (e.g. research sub-model).
                llm_params = _resolve_llm_params(
                    session.config.model_name,
                    session.hf_token,
                    reasoning_effort=session.effective_effort_for(
                        session.config.model_name
                    ),
                )
                messages = _sanitize_messages_for_model(
                    messages, llm_params.get("model") or session.config.model_name
                )
                if session.stream:
                    llm_result = await _call_llm_streaming(
                        session, messages, tools, llm_params
                    )
                else:
                    llm_result = await _call_llm_non_streaming(
                        session, messages, tools, llm_params
                    )

                content = llm_result.content
                tool_calls_acc = llm_result.tool_calls_acc
                token_count = llm_result.token_count
                finish_reason = llm_result.finish_reason

                # If output was truncated, all tool call args are garbage.
                # Inject a system hint so the LLM retries with smaller content.
                if finish_reason == "length" and tool_calls_acc:
                    dropped_names = [
                        tc["function"]["name"]
                        for tc in tool_calls_acc.values()
                        if tc["function"]["name"]
                    ]
                    logger.warning(
                        "Output truncated (finish_reason=length) — dropping tool calls: %s",
                        dropped_names,
                    )
                    tool_calls_acc.clear()

                    # Tell the agent what happened so it can retry differently
                    truncation_hint = (
                        "Your previous response was truncated because the output hit the "
                        "token limit. The following tool calls were lost: "
                        f"{dropped_names}. "
                        "IMPORTANT: Do NOT retry with the same large content. Instead:\n"
                        "  • For 'write': use bash with cat<<'HEREDOC' to write the file, "
                        "or split into several smaller edit calls.\n"
                        "  • For other tools: reduce the size of your arguments or use bash."
                    )
                    if content:
                        assistant_msg = _assistant_message_from_result(
                            llm_result,
                            model_name=llm_params.get("model"),
                        )
                        session.context_manager.add_message(assistant_msg, token_count)
                    session.context_manager.add_message(
                        Message(role="user", content=f"[SYSTEM: {truncation_hint}]")
                    )
                    if session.stream:
                        await session.send_event(
                            Event(event_type="assistant_stream_end", data={})
                        )
                    await session.send_event(
                        Event(
                            event_type="tool_log",
                            data={
                                "tool": "system",
                                "log": f"Output truncated — retrying with smaller content ({dropped_names})",
                            },
                        )
                    )
                    iteration += 1
                    continue  # retry this iteration

                # Build tool_calls list from accumulated deltas
                tool_calls: list[ToolCall] = []
                for idx in sorted(tool_calls_acc.keys()):
                    tc_data = tool_calls_acc[idx]
                    tool_calls.append(
                        ToolCall(
                            id=tc_data["id"],
                            type="function",
                            function={
                                "name": tc_data["function"]["name"],
                                "arguments": tc_data["function"]["arguments"],
                            },
                        )
                    )

                # Signal end of streaming to the frontend
                if session.stream:
                    await session.send_event(
                        Event(event_type="assistant_stream_end", data={})
                    )

                # If no tool calls, add assistant message and we're done
                if not tool_calls:
                    if not content:
                        if _should_emit_hf_planner_fallback(session, text):
                            fallback_msg = _hf_planner_fallback_message(session)
                            await _emit_visible_assistant_message(session, fallback_msg)
                            final_response = fallback_msg
                            break
                        if _should_emit_vertex_smoke_fallback(session, text):
                            fallback_msg = _vertex_smoke_fallback_message(session)
                            await _emit_visible_assistant_message(session, fallback_msg)
                            final_response = fallback_msg
                            break
                        error_msg = _empty_llm_response_message(
                            model_name=session.config.model_name,
                            finish_reason=finish_reason,
                        )
                        await _emit_visible_error(
                            session,
                            error_msg,
                            error_type="empty_response",
                            model_name=session.config.model_name,
                        )
                        errored = True
                        break

                    unfinished_plan = _unfinished_plan_items(session)
                    if (
                        unfinished_plan
                        and no_tool_incomplete_plan_retries
                        < _NO_TOOL_INCOMPLETE_PLAN_RETRY_LIMIT
                    ):
                        logger.info(
                            "No tool calls with unfinished plan; retrying agent turn "
                            "(attempt %d/%d)",
                            no_tool_incomplete_plan_retries + 1,
                            _NO_TOOL_INCOMPLETE_PLAN_RETRY_LIMIT,
                        )
                        if content:
                            assistant_msg = _assistant_message_from_result(
                                llm_result,
                                model_name=llm_params.get("model"),
                            )
                            session.context_manager.add_message(
                                assistant_msg, token_count
                            )
                        session.context_manager.add_message(
                            Message(
                                role="user",
                                content=_no_tool_incomplete_plan_prompt(
                                    unfinished_plan
                                ),
                            )
                        )
                        no_tool_incomplete_plan_retries += 1
                        await session.send_event(
                            Event(
                                event_type="tool_log",
                                data={
                                    "tool": "system",
                                    "log": (
                                        "Plan still has unfinished items after a "
                                        "text-only response — retrying instead of "
                                        "returning to the prompt."
                                    ),
                                },
                            )
                        )
                        iteration += 1
                        continue

                    if (
                        _should_continue_vertex_smoke_launch(session)
                        and vertex_smoke_continuation_retries
                        < _VERTEX_SMOKE_CONTINUATION_RETRY_LIMIT
                    ):
                        logger.info(
                            "No tool calls after bounded Vertex smoke preflight; "
                            "retrying with launch continuation prompt "
                            "(attempt %d/%d)",
                            vertex_smoke_continuation_retries + 1,
                            _VERTEX_SMOKE_CONTINUATION_RETRY_LIMIT,
                        )
                        if content:
                            assistant_msg = _assistant_message_from_result(
                                llm_result,
                                model_name=llm_params.get("model"),
                            )
                            session.context_manager.add_message(
                                assistant_msg, token_count
                            )
                        session.context_manager.add_message(
                            Message(
                                role="user",
                                content=_vertex_smoke_continuation_prompt(session),
                            )
                        )
                        vertex_smoke_continuation_retries += 1
                        await session.send_event(
                            Event(
                                event_type="tool_log",
                                data={
                                    "tool": "system",
                                    "log": (
                                        "Bounded Vertex smoke preflight allows "
                                        "manual approval — retrying to request "
                                        "gcp_vertex_jobs run."
                                    ),
                                },
                            )
                        )
                        iteration += 1
                        continue

                    logger.debug(
                        "Agent loop ending: no tool calls. "
                        "finish_reason=%s, token_count=%d, "
                        "usage=%d, model_max_tokens=%d, "
                        "iteration=%d/%d, "
                        "response_text=%s",
                        finish_reason,
                        token_count,
                        session.context_manager.running_context_usage,
                        session.context_manager.model_max_tokens,
                        iteration,
                        max_iterations,
                        (content or "")[:500],
                    )
                    if content:
                        assistant_msg = _assistant_message_from_result(
                            llm_result,
                            model_name=llm_params.get("model"),
                        )
                        session.context_manager.add_message(assistant_msg, token_count)
                        final_response = content
                    break

                no_tool_incomplete_plan_retries = 0
                vertex_smoke_continuation_retries = 0

                # Validate tool call args (one json.loads per call, once)
                # and split into good vs bad
                good_tools: list[tuple[ToolCall, str, dict]] = []
                bad_tools: list[ToolCall] = []
                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                        good_tools.append((tc, tc.function.name, args))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        logger.warning(
                            "Malformed arguments for tool_call %s (%s) — skipping",
                            tc.id,
                            tc.function.name,
                        )
                        tc.function.arguments = "{}"
                        bad_tools.append(tc)

                # Add assistant message with all tool calls to context
                assistant_msg = _assistant_message_from_result(
                    llm_result,
                    model_name=llm_params.get("model"),
                    tool_calls=tool_calls,
                )
                session.context_manager.add_message(assistant_msg, token_count)

                # Add error results for bad tool calls so the LLM
                # knows what happened and can retry differently
                for tc in bad_tools:
                    error_msg = (
                        f"ERROR: Tool call to '{tc.function.name}' had malformed JSON "
                        f"arguments and was NOT executed. Retry with smaller content — "
                        f"for 'write', split into multiple smaller writes using 'edit'."
                    )
                    session.context_manager.add_message(
                        Message(
                            role="tool",
                            content=error_msg,
                            tool_call_id=tc.id,
                            name=tc.function.name,
                        )
                    )
                    await session.send_event(
                        Event(
                            event_type="tool_call",
                            data={
                                "tool": tc.function.name,
                                "arguments": {},
                                "tool_call_id": tc.id,
                            },
                        )
                    )
                    await session.send_event(
                        Event(
                            event_type="tool_output",
                            data={
                                "tool": tc.function.name,
                                "tool_call_id": tc.id,
                                "output": error_msg,
                                "success": False,
                            },
                        )
                    )

                policy_allowed_tools: list[tuple[ToolCall, str, dict]] = []
                for tc, tool_name, tool_args in good_tools:
                    violation = _provider_tool_policy_violation(
                        session, tool_name, tool_args
                    )
                    if violation is None:
                        policy_allowed_tools.append((tc, tool_name, tool_args))
                        continue

                    error_msg = f"ERROR: {violation}"
                    session.context_manager.add_message(
                        Message(
                            role="tool",
                            content=error_msg,
                            tool_call_id=tc.id,
                            name=tool_name,
                        )
                    )
                    await session.send_event(
                        Event(
                            event_type="tool_call",
                            data={
                                "tool": tool_name,
                                "arguments": tool_args,
                                "tool_call_id": tc.id,
                            },
                        )
                    )
                    await session.send_event(
                        Event(
                            event_type="tool_output",
                            data={
                                "tool": tool_name,
                                "tool_call_id": tc.id,
                                "output": error_msg,
                                "success": False,
                            },
                        )
                    )
                    await session.send_event(
                        Event(
                            event_type="tool_state_change",
                            data={
                                "tool_call_id": tc.id,
                                "tool": tool_name,
                                "state": "blocked",
                                "reason": violation,
                            },
                        )
                    )
                good_tools = policy_allowed_tools

                # ── Cancellation check: before tool execution ──
                if session.is_cancelled:
                    break

                # Separate good tools into approval-required vs auto-execute.
                # Track reserved spend while classifying a batch so two
                # auto-approved jobs in one model response cannot jointly
                # exceed the remaining session cap.
                approval_required_tools: list[
                    tuple[ToolCall, str, dict, ApprovalDecision]
                ] = []
                non_approval_tools: list[
                    tuple[ToolCall, str, dict, ApprovalDecision]
                ] = []
                reserved_auto_spend_usd = 0.0
                for tc, tool_name, tool_args in good_tools:
                    decision = await _approval_decision(
                        tool_name,
                        tool_args,
                        session,
                        reserved_spend_usd=reserved_auto_spend_usd,
                    )
                    if decision.requires_approval:
                        approval_required_tools.append(
                            (tc, tool_name, tool_args, decision)
                        )
                    else:
                        non_approval_tools.append((tc, tool_name, tool_args, decision))
                        if (
                            decision.auto_approved
                            and decision.billable
                            and decision.estimated_cost_usd is not None
                        ):
                            reserved_auto_spend_usd += decision.estimated_cost_usd

                # Execute non-approval tools (in parallel when possible)
                if non_approval_tools:
                    # 1. Validate args upfront
                    parsed_tools: list[
                        tuple[ToolCall, str, dict, ApprovalDecision, bool, str]
                    ] = []
                    for tc, tool_name, tool_args, decision in non_approval_tools:
                        args_valid, error_msg = _validate_tool_args(tool_args)
                        parsed_tools.append(
                            (tc, tool_name, tool_args, decision, args_valid, error_msg)
                        )

                    # 2. Send all tool_call events upfront (so frontend shows them all)
                    for (
                        tc,
                        tool_name,
                        tool_args,
                        _decision,
                        args_valid,
                        _,
                    ) in parsed_tools:
                        if args_valid:
                            await session.send_event(
                                Event(
                                    event_type="tool_call",
                                    data={
                                        "tool": tool_name,
                                        "arguments": tool_args,
                                        "tool_call_id": tc.id,
                                    },
                                )
                            )

                    # 3. Execute all valid tools in parallel, cancellable
                    async def _exec_tool(
                        tc: ToolCall,
                        name: str,
                        args: dict,
                        decision: ApprovalDecision,
                        valid: bool,
                        err: str,
                    ) -> tuple[ToolCall, str, dict, str, bool]:
                        if not valid:
                            return (tc, name, args, err, False)
                        if decision.billable:
                            _record_estimated_spend(session, decision)
                        try:
                            out, ok = await session.tool_router.call_tool(
                                name, args, session=session, tool_call_id=tc.id
                            )
                        except Exception as exc:
                            logger.exception(
                                "Tool call %s failed without returning an error",
                                name,
                            )
                            out = (
                                f"Tool error: {type(exc).__name__}: {exc}. "
                                "Continue with the available successful tool outputs "
                                "or retry this tool only if it is required."
                            )
                            ok = False
                        return (tc, name, args, out, ok)

                    gather_task = asyncio.ensure_future(
                        asyncio.gather(
                            *[
                                _exec_tool(tc, name, args, decision, valid, err)
                                for tc, name, args, decision, valid, err in parsed_tools
                            ]
                        )
                    )
                    cancel_task = asyncio.ensure_future(session._cancelled.wait())

                    done, _ = await asyncio.wait(
                        [gather_task, cancel_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if cancel_task in done:
                        gather_task.cancel()
                        try:
                            await gather_task
                        except asyncio.CancelledError:
                            pass
                        # Notify frontend that in-flight tools were cancelled
                        for tc, name, _args, _decision, valid, _ in parsed_tools:
                            if valid:
                                await session.send_event(
                                    Event(
                                        event_type="tool_state_change",
                                        data={
                                            "tool_call_id": tc.id,
                                            "tool": name,
                                            "state": "cancelled",
                                        },
                                    )
                                )
                        await _cleanup_on_cancel(session)
                        break

                    cancel_task.cancel()
                    results = gather_task.result()

                    # 4. Record results and send outputs (order preserved)
                    terminal_provider_output: str | None = None
                    for tc, tool_name, tool_args, output, success in results:
                        tool_msg = Message(
                            role="tool",
                            content=output,
                            tool_call_id=tc.id,
                            name=tool_name,
                        )
                        session.context_manager.add_message(tool_msg)

                        await session.send_event(
                            Event(
                                event_type="tool_output",
                                data={
                                    "tool": tool_name,
                                    "tool_call_id": tc.id,
                                    "output": output,
                                    "success": success,
                                    "structured": _structured_tool_output(
                                        session, tc.id
                                    ),
                                },
                            )
                        )
                        if _is_terminal_provider_tool_output(
                            tool_name, output, success
                        ):
                            terminal_provider_output = output

                    if terminal_provider_output is not None:
                        final_response = terminal_provider_output
                        break

                    if _should_avoid_planning_only_stop(session):
                        last_tool = _last_tool_name(session)
                        if last_tool in {"training_planner", "training_preflight"}:
                            session.context_manager.add_message(
                                Message(
                                    role="user",
                                    content=_vertex_smoke_after_planner_prompt(session),
                                )
                            )
                            await session.send_event(
                                Event(
                                    event_type="tool_log",
                                    data={
                                        "tool": "system",
                                        "log": (
                                            "Bounded Vertex smoke workflow active "
                                            "— continuing toward gcp_vertex_jobs "
                                            "approval instead of planning-only stop."
                                        ),
                                    },
                                )
                            )
                            iteration += 1
                            continue

                    planning_only_summary = _planning_only_completion_message(session)
                    if planning_only_summary is not None:
                        await _emit_visible_assistant_message(
                            session, planning_only_summary
                        )
                        final_response = planning_only_summary
                        break

                # If there are tools requiring approval, ask for batch approval
                if approval_required_tools:
                    # Prepare batch approval data
                    tools_data = []
                    blocked_payloads = []
                    approval_records = []
                    for tc, tool_name, tool_args, decision in approval_required_tools:
                        # Resolve sandbox file paths for hf_jobs scripts so the
                        # frontend can display & edit the actual file content.
                        if tool_name == "hf_jobs" and isinstance(
                            tool_args.get("script"), str
                        ):
                            from agent.tools.sandbox_tool import resolve_sandbox_script

                            sandbox = getattr(session, "sandbox", None)
                            resolved, _ = await resolve_sandbox_script(
                                sandbox, tool_args["script"]
                            )
                            if resolved:
                                tool_args = {**tool_args, "script": resolved}

                        tool_payload = {
                            "tool": tool_name,
                            "arguments": tool_args,
                            "tool_call_id": tc.id,
                        }
                        approval_record = _approval_record(tc, tool_name, tool_args)
                        approval_record.update(
                            {
                                "estimated_cost_usd": decision.estimated_cost_usd,
                                "remaining_cap_usd": decision.remaining_cap_usd,
                                "billable": decision.billable,
                            }
                        )
                        approval_records.append(approval_record)
                        tool_payload.update(approval_record)
                        metadata = _approval_metadata(session, tool_name, tool_args)
                        if metadata:
                            tool_payload["metadata"] = metadata
                        if decision.estimated_cost_usd is not None:
                            tool_payload["estimated_cost_usd"] = (
                                decision.estimated_cost_usd
                            )
                        if decision.remaining_cap_usd is not None:
                            tool_payload["remaining_cap_usd"] = (
                                decision.remaining_cap_usd
                            )
                        tool_payload["billable"] = decision.billable
                        if decision.auto_approval_blocked:
                            tool_payload.update(
                                {
                                    "auto_approval_blocked": True,
                                    "block_reason": decision.block_reason,
                                    "estimated_cost_usd": decision.estimated_cost_usd,
                                    "remaining_cap_usd": decision.remaining_cap_usd,
                                }
                            )
                            blocked_payloads.append(tool_payload)
                        tools_data.append(tool_payload)

                    event_data = {"tools": tools_data, "count": len(tools_data)}
                    if blocked_payloads:
                        first = blocked_payloads[0]
                        event_data.update(
                            {
                                "auto_approval_blocked": True,
                                "block_reason": first.get("block_reason"),
                                "estimated_cost_usd": first.get("estimated_cost_usd"),
                                "remaining_cap_usd": first.get("remaining_cap_usd"),
                            }
                        )
                    await session.send_event(
                        Event(
                            event_type="approval_required",
                            data=event_data,
                        )
                    )

                    # Store all approval-requiring tools (ToolCall objects for execution)
                    session.pending_approval = {
                        "tool_calls": [tc for tc, _, _, _ in approval_required_tools],
                        "approvals": approval_records,
                    }

                    # Return early - wait for EXEC_APPROVAL operation
                    return None

                iteration += 1

            except ContextWindowExceededError:
                # Force compact and retry this iteration.
                cm = session.context_manager
                logger.warning(
                    "ContextWindowExceededError at iteration %d — forcing compaction "
                    "(usage=%d, model_max_tokens=%d, messages=%d)",
                    iteration,
                    cm.running_context_usage,
                    cm.model_max_tokens,
                    len(cm.items),
                )
                cm.running_context_usage = cm.model_max_tokens + 1
                await _compact_and_notify(session)
                # Same guard as the top of the loop: if compaction couldn't
                # bring us under threshold, _compact_and_notify has already
                # emitted session_terminated and set is_running=False. Continue
                # would just re-call the LLM with the same too-big context.
                if not session.is_running:
                    break
                continue

            except Exception as e:
                error_type, error_msg = _llm_failure_message(
                    e,
                    model_name=session.config.model_name,
                    include_traceback=True,
                )
                await _emit_visible_error(
                    session,
                    error_msg,
                    error_type=error_type,
                    model_name=session.config.model_name,
                )
                errored = True
                break

        if session.is_cancelled:
            await _cleanup_on_cancel(session)
            await session.send_event(Event(event_type="interrupted"))
        elif not errored:
            await session.send_event(
                Event(
                    event_type="turn_complete",
                    data={
                        "history_size": len(session.context_manager.items),
                        "final_response": final_response
                        if isinstance(final_response, str)
                        else None,
                    },
                )
            )

        # Increment turn counter and check for auto-save
        session.increment_turn()
        await session.auto_save_if_needed()

        return final_response

    @staticmethod
    async def undo(session: Session) -> None:
        """Remove the last complete turn and notify the frontend."""
        removed = session.context_manager.undo_last_turn()
        if not removed:
            logger.warning("Undo: no user message found to remove")
        await session.send_event(Event(event_type="undo_complete"))

    @staticmethod
    async def new_conversation(session: Session, *, clear_screen: bool = False) -> None:
        """Start a fresh conversation inside the active runtime."""
        try:
            result = session.start_new_conversation()
        except Exception as e:
            await session.send_event(
                Event(event_type="error", data={"error": f"New chat failed: {e}"})
            )
            return
        result["clear_screen"] = clear_screen
        await session.send_event(Event(event_type="new_complete", data=result))

    @staticmethod
    async def resume(session: Session, path: str) -> None:
        """Reload context from a saved session log into the active session."""
        from agent.core.session_resume import restore_session_from_log

        try:
            result = restore_session_from_log(session, Path(path))
        except Exception as e:
            await session.send_event(
                Event(event_type="error", data={"error": f"Resume failed: {e}"})
            )
            return
        await session.send_event(Event(event_type="resume_complete", data=result))

    @staticmethod
    async def exec_approval(session: Session, approvals: list[dict]) -> None:
        """Handle batch job execution approval"""
        if not session.pending_approval:
            await session.send_event(
                Event(
                    event_type="error",
                    data={"error": "No pending approval to process"},
                )
            )
            return

        tool_calls = session.pending_approval.get("tool_calls", [])
        if not tool_calls:
            await session.send_event(
                Event(
                    event_type="error",
                    data={"error": "No pending tool calls found"},
                )
            )
            return

        # Create a map of tool_call_id -> approval decision
        approval_map = {a["tool_call_id"]: a for a in approvals}
        approval_records = _pending_approval_records(session)
        for a in approvals:
            if a.get("edited_script"):
                logger.info(
                    f"Received edited script for tool_call {a['tool_call_id']} ({len(a['edited_script'])} chars)"
                )

        # Separate approved and rejected tool calls
        approved_tasks = []
        rejected_tasks = []

        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError) as e:
                # Malformed arguments — treat as failed, notify agent
                logger.warning(f"Malformed tool arguments for {tool_name}: {e}")
                tool_msg = Message(
                    role="tool",
                    content=f"Malformed arguments: {e}",
                    tool_call_id=tc.id,
                    name=tool_name,
                )
                session.context_manager.add_message(tool_msg)
                await session.send_event(
                    Event(
                        event_type="tool_output",
                        data={
                            "tool": tool_name,
                            "tool_call_id": tc.id,
                            "output": f"Malformed arguments: {e}",
                            "success": False,
                        },
                    )
                )
                continue

            approval_decision = approval_map.get(tc.id, {"approved": False})
            approval_record = approval_records.get(tc.id)
            if approval_record and _is_expired_approval_record(approval_record):
                rejected_tasks.append(
                    (
                        tc,
                        tool_name,
                        {
                            "approved": False,
                            "feedback": (
                                "Original approval expired. Regenerate the preflight "
                                "before launching this job."
                            ),
                        },
                    )
                )
                continue

            if approval_decision.get("approved", False):
                if approval_decision.get("approval_id") not in (None, tc.id):
                    rejected_tasks.append(
                        (
                            tc,
                            tool_name,
                            {
                                "approved": False,
                                "feedback": (
                                    "Approval ID did not match the active pending job."
                                ),
                            },
                        )
                    )
                    continue
                edited_script = approval_decision.get("edited_script")
                was_edited = False
                if edited_script and "script" in tool_args:
                    tool_args["script"] = edited_script
                    was_edited = True
                    logger.info(f"Using user-edited script for {tool_name} ({tc.id})")
                selected_namespace = approval_decision.get("namespace")
                if selected_namespace and tool_name == "hf_jobs":
                    tool_args["namespace"] = selected_namespace
                approved_tasks.append((tc, tool_name, tool_args, was_edited))
            else:
                rejected_tasks.append((tc, tool_name, approval_decision))

        # Clear pending approval immediately so a page refresh during
        # execution won't re-show the approval dialog.
        session.pending_approval = None

        # Notify frontend of approval decisions immediately (before execution)
        for tc, tool_name, tool_args, _was_edited in approved_tasks:
            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": tool_name,
                        "state": "approved",
                    },
                )
            )
        for tc, tool_name, approval_decision in rejected_tasks:
            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": tool_name,
                        "state": "rejected",
                    },
                )
            )

        # Execute all approved tools concurrently
        async def execute_tool(tc, tool_name, tool_args, was_edited):
            """Execute a single tool and return its result.

            The TraceLog already exists on the frontend (created by
            approval_required), so we send tool_state_change instead of
            tool_call to avoid creating a duplicate.
            """
            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": tool_name,
                        "state": "running",
                    },
                )
            )

            await _record_manual_approved_spend_if_needed(session, tool_name, tool_args)

            output, success = await session.tool_router.call_tool(
                tool_name, tool_args, session=session, tool_call_id=tc.id
            )

            return (tc, tool_name, output, success, was_edited)

        # Execute all approved tools concurrently (cancellable)
        if approved_tasks:
            gather_task = asyncio.ensure_future(
                asyncio.gather(
                    *[
                        execute_tool(tc, tool_name, tool_args, was_edited)
                        for tc, tool_name, tool_args, was_edited in approved_tasks
                    ],
                    return_exceptions=True,
                )
            )
            cancel_task = asyncio.ensure_future(session._cancelled.wait())

            done, _ = await asyncio.wait(
                [gather_task, cancel_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if cancel_task in done:
                gather_task.cancel()
                try:
                    await gather_task
                except asyncio.CancelledError:
                    pass
                # Notify frontend that approved tools were cancelled
                for tc, tool_name, _args, _was_edited in approved_tasks:
                    await session.send_event(
                        Event(
                            event_type="tool_state_change",
                            data={
                                "tool_call_id": tc.id,
                                "tool": tool_name,
                                "state": "cancelled",
                            },
                        )
                    )
                await _cleanup_on_cancel(session)
                await session.send_event(Event(event_type="interrupted"))
                session.increment_turn()
                await session.auto_save_if_needed()
                return

            cancel_task.cancel()
            results = gather_task.result()

            # Process results and add to context
            for result in results:
                if isinstance(result, Exception):
                    # Handle execution error
                    logger.error(f"Tool execution error: {result}")
                    continue

                tc, tool_name, output, success, was_edited = result

                if was_edited:
                    output = f"[Note: The user edited the script before execution. The output below reflects the user-modified version, not your original script.]\n\n{output}"

                # Add tool result to context
                tool_msg = Message(
                    role="tool",
                    content=output,
                    tool_call_id=tc.id,
                    name=tool_name,
                )
                session.context_manager.add_message(tool_msg)

                await session.send_event(
                    Event(
                        event_type="tool_output",
                        data={
                            "tool": tool_name,
                            "tool_call_id": tc.id,
                            "output": output,
                            "success": success,
                            "structured": _structured_tool_output(session, tc.id),
                        },
                    )
                )

        # Process rejected tools
        for tc, tool_name, approval_decision in rejected_tasks:
            rejection_msg = "Job execution cancelled by user"
            user_feedback = approval_decision.get("feedback")
            if user_feedback:
                # Ensure feedback is a string and sanitize any problematic characters
                feedback_str = str(user_feedback).strip()
                # Remove any control characters that might break JSON parsing
                feedback_str = "".join(
                    char for char in feedback_str if ord(char) >= 32 or char in "\n\t"
                )
                rejection_msg += f". User feedback: {feedback_str}"

            # Ensure rejection_msg is a clean string
            rejection_msg = str(rejection_msg).strip()

            tool_msg = Message(
                role="tool",
                content=rejection_msg,
                tool_call_id=tc.id,
                name=tool_name,
            )
            session.context_manager.add_message(tool_msg)

            await session.send_event(
                Event(
                    event_type="tool_output",
                    data={
                        "tool": tool_name,
                        "tool_call_id": tc.id,
                        "output": rejection_msg,
                        "success": False,
                    },
                )
            )

        # Continue agent loop with empty input to process the tool results
        await Handlers.run_agent(session, "")

    @staticmethod
    async def shutdown(session: Session) -> bool:
        """Handle shutdown (like shutdown in codex.rs:1329)"""
        # Save session trajectory if enabled (fire-and-forget, returns immediately)
        if session.config.save_sessions:
            logger.info("Saving session...")
            repo_id = session.config.session_dataset_repo
            _ = session.save_and_upload_detached(repo_id)

        session.is_running = False
        if not getattr(session, "local_mode", False):
            await teardown_session_sandbox(session)
        await session.send_event(Event(event_type="shutdown"))
        return True


async def process_submission(session: Session, submission) -> bool:
    """
    Process a single submission and return whether to continue running.

    Returns:
        bool: True to continue, False to shutdown
    """
    op = submission.operation
    logger.debug("Received operation: %s", op.op_type.value)

    if op.op_type == OpType.USER_INPUT:
        text = op.data.get("text", "") if op.data else ""
        session.compute_tools_blocked_for_turn = _user_requested_no_compute_tools(text)
        session.bounded_vertex_smoke_for_turn = (
            _user_explicitly_requests_bounded_provider_launch(text)
        )
        session.training_planner_only_for_turn = _user_requested_training_planner_only(
            text
        )
        selected_cloud_provider = op.data.get("cloud_provider") if op.data else None
        cloud_provider = _resolve_cloud_provider_for_turn(selected_cloud_provider, text)
        training_goal = op.data.get("training_goal") if op.data else None
        output_policy = op.data.get("output_policy") if op.data else None
        if cloud_provider in {"hf-jobs", "gcp-vertex", "aws-sagemaker"}:
            session.cloud_provider = cloud_provider
            if training_goal in {"smoke-test", "production", "agent-decide"}:
                session.training_goal = training_goal
            else:
                training_goal = getattr(session, "training_goal", "agent-decide")
            if output_policy in {
                "cloud-private",
                "hf-hub",
                "cloud-and-hf-hub",
            }:
                session.output_policy = output_policy
            else:
                output_policy = getattr(session, "output_policy", "cloud-and-hf-hub")
            if cloud_provider == "gcp-vertex":
                provider_instruction = (
                    "User selected Google Cloud Vertex AI training backend. "
                    "Use gcp_vertex_jobs for fine-tuning/training. Use uploaded "
                    "dataset context and normalized dataset config if present. "
                    "Before launch, respect "
                    f"training_goal={training_goal} and output_policy={output_policy}. "
                    "If output_policy=cloud-private, do not push final model to "
                    "Hugging Face Hub. If output_policy=hf-hub, push final model "
                    "to Hugging Face Hub. If output_policy=cloud-and-hf-hub, save "
                    "to GCS and push to Hugging Face Hub. For sensitive domains "
                    "(medical, finance, legal, insurance, government, or internal "
                    "company data), recommend cloud-private unless user explicitly "
                    "chooses otherwise. gcp_vertex_jobs run and cancel operations "
                    "are approval-gated and billable; do not launch them without "
                    "approval. When live training preflight returns "
                    "manual_approval_allowed=true for a bounded smoke-test, "
                    "launch_ready remains false and unknown quota/accelerator checks "
                    "stay unknown, but you may proceed to call gcp_vertex_jobs run "
                    "so the user can explicitly approve the bounded smoke launch. "
                    "Do not treat launch_ready=false as a hard stop in that case. "
                    "After a Vertex AI job is launched, monitor only "
                    "with gcp_vertex_jobs inspect/logs/ps, or cancel if the user "
                    "approves cancellation. Do not use sandbox_create, bash, "
                    "hf_jobs, or aws_sagemaker_jobs for Vertex job cooldown, "
                    "polling, or monitoring. For non-training requests, use the "
                    "normal best-fit tools."
                )
            elif cloud_provider == "aws-sagemaker":
                provider_instruction = (
                    "The frontend training provider selector for this session is "
                    "set to AWS SageMaker AI. For training, fine-tuning, SFT, "
                    "model adaptation, or cloud compute requests, use "
                    "aws_sagemaker_jobs and do not route to Hugging Face Jobs or "
                    "Google Cloud Vertex AI compute unless the provider changes. "
                    "Use normalized uploaded dataset configs from this session "
                    "when present. For AWS training/fine-tuning/SFT, use "
                    "aws_sagemaker_jobs; it stages normalized datasets to S3 "
                    "and can submit SageMaker training jobs when readiness and "
                    "training image config are present. Before billable jobs, "
                    "use training_planner, then show a concise preflight/request "
                    "approval. For the built-in aws_sagemaker_jobs SFT template, "
                    "skip broad literature/research crawls and do not stop after "
                    "planning unless explicit clarification is required. Respect "
                    f"training_goal={training_goal} and output_policy={output_policy}. "
                    "For output_policy=cloud-private, keep final artifacts in "
                    "private S3 and do not push to Hugging Face Hub. Operation "
                    "run and cancel are approval-gated; do not use Hugging Face "
                    "Jobs or Google Cloud Vertex AI compute unless the provider "
                    "changes. After a SageMaker job is launched, monitor only "
                    "with aws_sagemaker_jobs inspect/logs/ps, or cancel if the "
                    "user approves cancellation. Do not use sandbox_create, bash, "
                    "hf_jobs, or gcp_vertex_jobs for SageMaker cooldown, polling, "
                    "or monitoring."
                )
            else:
                provider_instruction = (
                    "The frontend training provider selector for this session is "
                    "set to Hugging Face Jobs. For training, fine-tuning, SFT, "
                    "model adaptation, cloud compute, or model training job "
                    "requests, prefer hf_jobs unless the user explicitly asks "
                    "for another backend. Use uploaded dataset context from this "
                    "session when one is available. Before launch, respect "
                    f"training_goal={training_goal} and output_policy={output_policy}. "
                    "After a training_planner recommendation, continue to a "
                    "Hugging Face Jobs preflight and the approval-gated hf_jobs "
                    "step; do not stop after planning unless a real clarification "
                    "is required. hf_jobs run operations are approval-gated and "
                    "billable; do not launch them without approval. Keep compute "
                    "on Hugging Face Jobs unless the user changes provider. Do "
                    "not use Kaggle."
                )
            session.context_manager.add_message(
                Message(
                    role="user",
                    content=f"[SYSTEM: {provider_instruction}]",
                )
            )
        if getattr(session, "compute_tools_blocked_for_turn", False):
            if getattr(session, "bounded_vertex_smoke_for_turn", False):
                session.context_manager.add_message(
                    Message(
                        role="user",
                        content=(
                            "[SYSTEM: This turn is a bounded Google Vertex AI "
                            "smoke workflow. Do not create sandbox or use "
                            "hf_jobs/aws_sagemaker_jobs. After dataset_discovery "
                            "and training_planner, continue to gcp_vertex_jobs run "
                            "when live preflight allows bounded manual approval. "
                            "Do not stop with a planning-only summary.]"
                        ),
                    )
                )
            else:
                planning_tool_instruction = (
                    "training_planner only. Do not call dataset_discovery for this "
                    "turn; mention that dataset discovery requires a separate user "
                    "request if needed."
                    if getattr(session, "training_planner_only_for_turn", False)
                    else "dataset_discovery/training_planner"
                )
                session.context_manager.add_message(
                    Message(
                        role="user",
                        content=(
                            "[SYSTEM: This turn is planning/discovery only. Do not "
                            "call sandbox_create, bash, read, write, edit, hf_jobs, "
                            "gcp_vertex_jobs, or aws_sagemaker_jobs. Use "
                            f"{planning_tool_instruction} and explain what "
                            "approval would be required before any launch.]"
                        ),
                    )
                )
        upload_instruction = _uploaded_dataset_instruction(session)
        if upload_instruction:
            session.context_manager.add_message(
                Message(role="user", content=f"[SYSTEM: {upload_instruction}]")
            )
        await Handlers.run_agent(session, text)
        return True

    if op.op_type == OpType.COMPACT:
        await _compact_and_notify(session)
        return True

    if op.op_type == OpType.UNDO:
        await Handlers.undo(session)
        return True

    if op.op_type == OpType.NEW:
        clear_screen = bool((op.data or {}).get("clear_screen"))
        await Handlers.new_conversation(session, clear_screen=clear_screen)
        return True

    if op.op_type == OpType.RESUME:
        path = op.data.get("path") if op.data else None
        if path:
            await Handlers.resume(session, path)
        else:
            await session.send_event(
                Event(event_type="error", data={"error": "Resume requires a path"})
            )
        return True

    if op.op_type == OpType.EXEC_APPROVAL:
        approvals = op.data.get("approvals", []) if op.data else []
        await Handlers.exec_approval(session, approvals)
        return True

    if op.op_type == OpType.SHUTDOWN:
        return not await Handlers.shutdown(session)

    logger.warning(f"Unknown operation: {op.op_type}")
    return True


async def submission_loop(
    submission_queue: asyncio.Queue,
    event_queue: asyncio.Queue,
    config: Config,
    tool_router: ToolRouter | None = None,
    session_holder: list | None = None,
    hf_token: str | None = None,
    user_id: str | None = None,
    local_mode: bool = False,
    stream: bool = True,
    notification_gateway: NotificationGateway | None = None,
    notification_destinations: list[str] | None = None,
    defer_turn_complete_notification: bool = False,
) -> None:
    """
    Main agent loop - processes submissions and dispatches to handlers.
    This is the core of the agent (like submission_loop in codex.rs:1259-1340)
    """

    # Create session with tool router
    session = Session(
        event_queue,
        config=config,
        tool_router=tool_router,
        hf_token=hf_token,
        user_id=user_id,
        local_mode=local_mode,
        stream=stream,
        notification_gateway=notification_gateway,
        notification_destinations=notification_destinations,
        defer_turn_complete_notification=defer_turn_complete_notification,
    )
    if session_holder is not None:
        session_holder[0] = session
    if not local_mode:
        start_cpu_sandbox_preload(session)
    logger.info("Agent loop started")

    # Retry any failed uploads from previous sessions (fire-and-forget).
    # Includes the personal trace repo when enabled so a session that failed
    # to publish to the user's HF dataset gets a fresh attempt on next run.
    if config and config.save_sessions:
        Session.retry_failed_uploads_detached(
            directory=str(DEFAULT_SESSION_LOG_DIR),
            repo_id=config.session_dataset_repo,
            personal_repo_id=session._personal_trace_repo_id(),
        )

    try:
        # Main processing loop
        async with tool_router:
            # Emit ready event after initialization
            await session.send_event(
                Event(
                    event_type="ready",
                    data={
                        "message": "Agent initialized",
                        "tool_count": len(tool_router.tools),
                    },
                )
            )

            while session.is_running:
                submission = await submission_queue.get()

                try:
                    should_continue = await process_submission(session, submission)
                    if not should_continue:
                        break
                except asyncio.CancelledError:
                    logger.warning("Agent loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in agent loop: {e}")
                    await session.send_event(
                        Event(event_type="error", data={"error": str(e)})
                    )

        logger.info("Agent loop exited")

    finally:
        # Emergency save if session saving is enabled and shutdown wasn't called properly
        if session.config.save_sessions and session.is_running:
            logger.info("Emergency save: preserving session before exit...")
            try:
                local_path = session.save_and_upload_detached(
                    session.config.session_dataset_repo
                )
                if local_path:
                    logger.info("Emergency save successful, upload in progress")
            except Exception as e:
                logger.error(f"Emergency save failed: {e}")
