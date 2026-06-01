"""Pydantic models for API requests and responses."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

CloudProviderId = Literal["hf-jobs", "gcp-vertex"]
TrainingGoal = Literal["smoke-test", "production", "agent-decide"]
OutputPolicy = Literal["cloud-private", "hf-hub", "cloud-and-hf-hub"]
DatasetSourceFormat = Literal["csv", "json", "jsonl", "pdf", "docx", "xlsx", "md"]


class OpType(str, Enum):
    """Operation types matching agent/core/agent_loop.py."""

    USER_INPUT = "user_input"
    EXEC_APPROVAL = "exec_approval"
    INTERRUPT = "interrupt"
    UNDO = "undo"
    COMPACT = "compact"
    SHUTDOWN = "shutdown"


class Operation(BaseModel):
    """Operation to be submitted to the agent."""

    op_type: OpType
    data: dict[str, Any] | None = None


class Submission(BaseModel):
    """Submission wrapper with ID and operation."""

    id: str
    operation: Operation


class ToolApproval(BaseModel):
    """Approval decision for a single tool call."""

    tool_call_id: str
    approved: bool
    approval_id: str | None = None
    feedback: str | None = None
    edited_script: str | None = None
    namespace: str | None = None


class ApprovalRequest(BaseModel):
    """Request to approve/reject tool calls."""

    session_id: str
    approvals: list[ToolApproval]


class SubmitRequest(BaseModel):
    """Request to submit user input."""

    session_id: str
    # Cap text size to prevent context-bloat / cost-amplification: a malicious
    # or runaway client could otherwise attach megabytes that then ride along
    # in every subsequent turn until /api/compact is called.
    text: str = Field(..., min_length=1, max_length=100_000)
    cloud_provider: CloudProviderId | None = None
    training_goal: TrainingGoal | None = None
    output_policy: OutputPolicy | None = None


class TruncateRequest(BaseModel):
    """Request to truncate conversation history to before a specific user message."""

    user_message_index: int


class SessionResponse(BaseModel):
    """Response when creating a new session."""

    session_id: str
    ready: bool = True
    model: str | None = None
    cloud_provider: CloudProviderId = "hf-jobs"
    training_goal: TrainingGoal = "agent-decide"
    output_policy: OutputPolicy = "cloud-and-hf-hub"


class PendingApprovalTool(BaseModel):
    """A tool waiting for user approval."""

    tool: str
    tool_call_id: str
    arguments: dict[str, Any] = {}
    approval_id: str | None = None
    operation: str | None = None
    provider: str | None = None
    created_at: str | None = None
    expires_at: str | None = None
    status: str | None = None


class SessionAutoApprovalInfo(BaseModel):
    """Per-session auto-approval budget state."""

    enabled: bool = False
    cost_cap_usd: float | None = None
    estimated_spend_usd: float = 0.0
    remaining_usd: float | None = None


class UploadedDatasetInfo(BaseModel):
    """Minimal uploaded dataset metadata for session UI and planning context."""

    upload_id: str
    filename: str
    format: DatasetSourceFormat
    source_format: DatasetSourceFormat
    source: str = "session-upload"
    uploaded_at: str | None = None
    normalized_row_count: int
    normalized_format: Literal["jsonl"] = "jsonl"
    status: Literal["ready", "failed"] = "ready"
    supports_training: bool = True
    size_bytes: int | None = None
    config_name: str
    repo_id: str
    repo_type: Literal["dataset"] = "dataset"
    normalized_path_in_repo: str
    raw_path_in_repo: str
    hub_url: str
    load_dataset_snippet: str


class SessionInfo(BaseModel):
    """Session metadata."""

    session_id: str
    created_at: str
    is_active: bool
    is_processing: bool = False
    message_count: int
    user_id: str = "dev"
    pending_approval: list[PendingApprovalTool] | None = None
    model: str | None = None
    cloud_provider: CloudProviderId = "hf-jobs"
    training_goal: TrainingGoal = "agent-decide"
    output_policy: OutputPolicy = "cloud-and-hf-hub"
    title: str | None = None
    notification_destinations: list[str] = Field(default_factory=list)
    auto_approval: SessionAutoApprovalInfo = Field(
        default_factory=SessionAutoApprovalInfo
    )
    uploaded_datasets: list[UploadedDatasetInfo] = Field(default_factory=list)


class SessionNotificationsRequest(BaseModel):
    """Replace the session's auto-notification destinations."""

    destinations: list[str]


class SessionYoloRequest(BaseModel):
    """Update a session's auto-approval policy."""

    enabled: bool
    cost_cap_usd: float | None = Field(default=None, ge=0)


class DatasetUploadResponse(BaseModel):
    """Response for a dataset file uploaded to the Hub."""

    session_id: str
    repo_id: str
    repo_type: Literal["dataset"] = "dataset"
    private: bool = True
    upload_id: str
    config_name: str
    filename: str
    path_in_repo: str
    raw_path_in_repo: str
    normalized_path_in_repo: str
    normalized_format: Literal["jsonl"]
    normalized_row_count: int
    source_format: DatasetSourceFormat
    source: str = "session-upload"
    uploaded_at: str
    supports_training: bool
    size_bytes: int
    format: DatasetSourceFormat
    hub_url: str
    load_dataset_snippet: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    active_sessions: int = 0
    max_sessions: int = 0


class LLMHealthResponse(BaseModel):
    """LLM provider health check response."""

    status: str  # "ok" | "error"
    model: str
    error: str | None = None
    error_type: str | None = (
        None  # "quota" | "billing" | "auth" | "rate_limit" | "network" | "empty_response" | "unknown"
    )
