"""Pydantic models for API requests and responses."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

CloudProviderId = Literal["hf-jobs", "gcp-vertex", "aws-sagemaker"]
UsageProviderId = Literal["hf-jobs", "gcp-vertex", "aws-sagemaker", "llm", "unknown"]
AuditCategory = Literal[
    "session",
    "dataset",
    "chat",
    "planner",
    "approval",
    "tool",
    "provider_job",
    "usage",
    "result",
    "error",
    "system",
    "security",
]
AuditSeverity = Literal["info", "warning", "error", "critical"]
CostSource = Literal[
    "static_estimate",
    "provider_estimate",
    "approval_estimate",
    "actual_provider_billing",
    "unknown",
]
CostConfidence = Literal["known", "estimated", "unknown"]
TrainingGoal = Literal["smoke-test", "production", "agent-decide"]
OutputPolicy = Literal["cloud-private", "hf-hub", "cloud-and-hf-hub"]
DatasetSourceFormat = Literal["csv", "json", "jsonl", "pdf", "docx", "xlsx", "md"]
TrainingPreflightStatus = Literal[
    "not_run",
    "checking",
    "passed",
    "warning",
    "failed",
    "unknown",
    "skipped",
]
TrainingPreflightSeverity = Literal["info", "warning", "error", "blocking"]
TrainingPreflightCheckCategory = Literal[
    "credentials",
    "identity",
    "model_access",
    "metadata",
    "namespace",
    "storage",
    "api",
    "hardware",
    "quota",
    "compatibility",
    "output_policy",
    "fallback",
    "safety",
]


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


class DatasetDiscoveryResponse(BaseModel):
    query: str | None = None
    intent: dict[str, Any] = Field(default_factory=dict)
    allowed_sources: list[str] = Field(default_factory=list)
    excluded_sources: list[str] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    recommended_candidate: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    selected_candidate: dict[str, Any] | None = None
    no_candidates_reason: str | None = None
    timestamp: str | None = None
    requires_user_selection: bool = True


class TrainingPreflightCheckModel(BaseModel):
    check_id: str
    provider: str
    category: TrainingPreflightCheckCategory | str
    label: str
    status: TrainingPreflightStatus | str
    severity: TrainingPreflightSeverity | str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    error_code: str | None = None
    docs_verification_required: bool = False


class TrainingPreflightProviderResultModel(BaseModel):
    provider: str
    status: TrainingPreflightStatus | str
    launch_ready: bool = False
    checks: list[TrainingPreflightCheckModel] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    warning_reasons: list[str] = Field(default_factory=list)
    unknown_reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingPreflightFallbackResultModel(BaseModel):
    fallback_id: str
    provider: str
    model_id: str | None = None
    hardware_id: str | None = None
    status: TrainingPreflightStatus | str = "not_run"
    launch_ready: bool = False
    checks: list[TrainingPreflightCheckModel] = Field(default_factory=list)
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingPreflightCacheInfoModel(BaseModel):
    cache_key: str | None = None
    hit: bool = False
    ttl_seconds: int | None = None
    created_at: str | None = None
    expires_at: str | None = None


class TrainingPreflightResultModel(BaseModel):
    preflight_id: str
    session_id: str
    run_id: str | None = None
    created_at: str
    updated_at: str
    status: TrainingPreflightStatus | str
    launch_ready: bool = False
    provider: str
    model_id: str
    hardware_id: str | None = None
    output_policy: OutputPolicy | str
    primary: TrainingPreflightProviderResultModel
    fallbacks: list[TrainingPreflightFallbackResultModel] = Field(default_factory=list)
    verified_fallback: TrainingPreflightFallbackResultModel | None = None
    verified_recommendation: dict[str, Any] | None = None
    blocking_reasons: list[str] = Field(default_factory=list)
    warning_reasons: list[str] = Field(default_factory=list)
    unknown_reasons: list[str] = Field(default_factory=list)
    safe_summary: str = ""
    cache: TrainingPreflightCacheInfoModel = Field(
        default_factory=TrainingPreflightCacheInfoModel
    )
    metadata: dict[str, Any] = Field(
        default_factory=lambda: {
            "provider_jobs_launched": False,
            "resources_created": False,
            "live_checks_optional": True,
        }
    )


class TrainingPreflightRequest(BaseModel):
    session_id: str
    run_id: str | None = None
    provider: CloudProviderId | str | None = None
    model_id: str | None = None
    hardware_id: str | None = None
    output_policy: OutputPolicy | None = None
    recommendation: dict[str, Any] | None = None
    dataset_summary: dict[str, Any] | None = None
    target_namespace: str | None = None
    target_repo_id: str | None = None
    target_bucket: str | None = None
    include_fallbacks: bool = False
    force_refresh: bool = False
    timeout_seconds: int | None = Field(default=None, ge=1, le=60)
    allow_unknown_override: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    latest_dataset_discovery: DatasetDiscoveryResponse | None = None
    latest_training_recommendation: dict[str, Any] | None = None
    runs: list["RunSummary"] = Field(default_factory=list)


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


class SessionStoreHealth(BaseModel):
    """Non-secret session persistence status."""

    type: str
    durable: bool
    warning: str | None = None


class BackgroundRunsHealth(BaseModel):
    """Non-secret background run feature-flag status."""

    enabled: bool
    worker_mode: Literal["disabled", "in_process", "external_worker"]
    implemented: bool
    durable: bool
    store: str
    token_handoff_configured: bool = False
    warning: str | None = None


class UsageStoreHealth(BaseModel):
    """Non-secret usage ledger persistence status."""

    enabled: bool = True
    durable: bool
    store: str
    warning: str | None = None


class AuditStoreHealth(BaseModel):
    """Non-secret audit timeline persistence status."""

    type: str
    durable: bool
    enabled: bool = True
    warning: str | None = None


class SecurityHealth(BaseModel):
    """Non-secret security hardening status."""

    redaction_enabled: bool = True
    sandbox_private_default: bool = True
    secret_persistence_allowed: bool = False
    token_encryption_configured: bool = False
    encrypted_handoff_enabled: bool = False


class UsageEntry(BaseModel):
    usage_id: str
    session_id: str
    run_id: str | None = None
    provider: UsageProviderId | str = "unknown"
    tool_name: str | None = None
    operation: str = "unknown"
    job_id: str | None = None
    job_url: str | None = None
    artifact_url: str | None = None
    status: str = "pending"
    created_at: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    currency: str = "USD"
    estimated_cost_usd: float | None = None
    known_cost_usd: float | None = None
    cost_source: CostSource | str = "unknown"
    cost_confidence: CostConfidence | str = "unknown"
    instance_type: str | None = None
    instance_count: int | None = None
    max_runtime_seconds: int | None = None
    actual_runtime_seconds: int | None = None
    dataset_name: str | None = None
    model_name: str | None = None
    output_policy: str | None = None
    approval_id: str | None = None
    approved: bool = False
    budget_cap_usd: float | None = None
    quota_status: str = "unknown"
    warning: str | None = None
    error_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UsageSummary(BaseModel):
    total_estimated_cost_usd: float = 0.0
    total_known_cost_usd: float = 0.0
    cost_by_provider: dict[str, dict[str, Any]] = Field(default_factory=dict)
    cost_by_session: dict[str, dict[str, Any]] = Field(default_factory=dict)
    cost_by_run: dict[str, dict[str, Any]] = Field(default_factory=dict)
    recent_usage_entries: list[UsageEntry] = Field(default_factory=list)
    quota_warnings: list[dict[str, Any]] = Field(default_factory=list)
    budget_warnings: list[dict[str, Any]] = Field(default_factory=list)
    provider_readiness: dict[str, Any] = Field(default_factory=dict)
    usage_store: UsageStoreHealth | None = None


class AuditEvent(BaseModel):
    audit_id: str
    session_id: str
    run_id: str | None = None
    usage_id: str | None = None
    provider: str = "unknown"
    event_type: str
    category: AuditCategory | str
    severity: AuditSeverity | str = "info"
    status: str = "unknown"
    title: str
    message: str = ""
    timestamp: str | None = None
    actor: str = "system"
    entity_type: str | None = None
    entity_id: str | None = None
    tool_name: str | None = None
    operation: str | None = None
    approval_id: str | None = None
    job_id: str | None = None
    job_url: str | None = None
    artifact_url: str | None = None
    dataset_name: str | None = None
    model_name: str | None = None
    output_policy: str | None = None
    estimated_cost_usd: float | None = None
    known_cost_usd: float | None = None
    error_code: str | None = None
    error_summary: str | None = None
    safe_metadata: dict[str, Any] = Field(default_factory=dict)


class AuditTimelineResponse(BaseModel):
    enabled: bool = True
    audit_store: AuditStoreHealth | None = None
    events: list[AuditEvent] = Field(default_factory=list)


class AuditSummary(BaseModel):
    enabled: bool = True
    total_events: int = 0
    counts_by_category: dict[str, int] = Field(default_factory=dict)
    counts_by_severity: dict[str, int] = Field(default_factory=dict)
    counts_by_provider: dict[str, int] = Field(default_factory=dict)
    latest_warnings_errors: list[AuditEvent] = Field(default_factory=list)
    provider_job_timeline: list[AuditEvent] = Field(default_factory=list)
    approval_timeline: list[AuditEvent] = Field(default_factory=list)
    dataset_timeline: list[AuditEvent] = Field(default_factory=list)
    usage_cost_timeline: list[AuditEvent] = Field(default_factory=list)
    timeline_by_session: dict[str, list[AuditEvent]] = Field(default_factory=dict)
    timeline_by_run: dict[str, list[AuditEvent]] = Field(default_factory=dict)
    audit_store: AuditStoreHealth | None = None


class EvaluationScores(BaseModel):
    overall_score: float | None = None
    task_relevance_score: float | None = None
    safety_score: float | None = None
    privacy_score: float | None = None
    metric_quality_score: float | None = None
    confidence: float | None = None


class PostTrainingEvaluation(BaseModel):
    evaluation_id: str
    session_id: str
    run_id: str
    provider: str = "unknown"
    job_id: str | None = None
    model_ref: str | None = None
    artifact_ref: str | None = None
    dataset_ref: str | None = None
    status: str = "not_started"
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str | None = None
    evaluation_type: str = "static_result_review"
    domain: str = "unknown"
    task_type: str = "sft"
    test_prompts: list[str] = Field(default_factory=list)
    results: dict[str, Any] = Field(default_factory=dict)
    scores: EvaluationScores = Field(default_factory=EvaluationScores)
    safety_findings: list[dict[str, Any]] = Field(default_factory=list)
    privacy_findings: list[dict[str, Any]] = Field(default_factory=list)
    quality_summary: str | None = None
    failure_summary: str | None = None
    recommendation: str | None = None
    report_markdown: str | None = None
    artifact_paths: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    total_evaluations: int = 0
    counts_by_status: dict[str, int] = Field(default_factory=dict)
    average_overall_score: float | None = None
    latest_evaluation: PostTrainingEvaluation | None = None
    evaluations: list[PostTrainingEvaluation] = Field(default_factory=list)


class RunProviderMetadata(BaseModel):
    provider: str = "none"
    status: str | None = None
    job_id: str | None = None
    console_url: str | None = None
    logs_url: str | None = None
    artifact_path: str | None = None
    output_policy: str | None = None
    last_checked_at: str | None = None
    dataset_discovery: DatasetDiscoveryResponse | None = None
    training_recommendation: dict[str, Any] | None = None


class RunSummary(BaseModel):
    run_id: str
    session_id: str
    status: str
    provider: str = "none"
    created_at: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    last_event_seq: int = 0
    active_tool: str | None = None
    active_provider_job_id: str | None = None
    approval_id: str | None = None
    error_summary: str | None = None
    result_summary: str | None = None
    provider_metadata: RunProviderMetadata = Field(default_factory=RunProviderMetadata)
    evaluation_id: str | None = None
    evaluation_status: str | None = None
    evaluation_score: float | None = None
    estimated_cost_usd: float | None = None
    known_cost_usd: float | None = None
    usage_status: str = "unknown"
    budget_warning: str | None = None
    quota_warning: str | None = None
    audit_event_count: int = 0
    audit_warning_count: int = 0
    audit_error_count: int = 0
    latest_audit_event: AuditEvent | None = None
    dataset_discovery: DatasetDiscoveryResponse | None = None
    training_recommendation: dict[str, Any] | None = None


class RunEventInfo(BaseModel):
    run_id: str
    session_id: str
    seq: int
    timestamp: str | None = None
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    safe_summary: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    active_sessions: int = 0
    max_sessions: int = 0
    session_store: SessionStoreHealth | None = None
    background_runs: BackgroundRunsHealth | None = None
    usage_store: UsageStoreHealth | None = None
    audit_store: AuditStoreHealth | None = None
    security: SecurityHealth = Field(default_factory=SecurityHealth)
    cloud_run_revision: str | None = None


class LLMHealthResponse(BaseModel):
    """LLM provider health check response."""

    status: str  # "ok" | "error"
    model: str
    error: str | None = None
    error_type: str | None = (
        None  # "quota" | "billing" | "auth" | "rate_limit" | "network" | "empty_response" | "unknown"
    )
