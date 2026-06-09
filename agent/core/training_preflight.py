"""Foundation models and deterministic readiness for training preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from agent.core.redact import sanitize_for_frontend, sanitize_for_persistence


class PreflightStatus(str, Enum):
    NOT_RUN = "not_run"
    CHECKING = "checking"
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    UNKNOWN = "unknown"
    SKIPPED = "skipped"


class PreflightSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKING = "blocking"


class PreflightCheckCategory(str, Enum):
    CREDENTIALS = "credentials"
    IDENTITY = "identity"
    MODEL_ACCESS = "model_access"
    METADATA = "metadata"
    NAMESPACE = "namespace"
    STORAGE = "storage"
    API = "api"
    HARDWARE = "hardware"
    QUOTA = "quota"
    COMPATIBILITY = "compatibility"
    OUTPUT_POLICY = "output_policy"
    FALLBACK = "fallback"
    SAFETY = "safety"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _value(value: Enum | str | None) -> str | None:
    if isinstance(value, Enum):
        return str(value.value)
    return value


def _safe_dict(value: dict[str, Any]) -> dict[str, Any]:
    sanitized = sanitize_for_persistence(value)
    return sanitized if isinstance(sanitized, dict) else {}


def _reason(check: "TrainingPreflightCheck") -> str:
    return check.message or check.label or check.check_id


def _is_required(check: "TrainingPreflightCheck") -> bool:
    return check.details.get("required") is not False


def _is_non_applicable_skip(check: "TrainingPreflightCheck") -> bool:
    return check.status == PreflightStatus.SKIPPED and (
        check.details.get("applicable") is False
        or check.details.get("required") is False
    )


@dataclass(frozen=True)
class TrainingPreflightCheck:
    check_id: str
    provider: str
    category: PreflightCheckCategory | str
    label: str
    status: PreflightStatus | str
    severity: PreflightSeverity | str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    error_code: str | None = None
    docs_verification_required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", PreflightCheckCategory(self.category))
        object.__setattr__(self, "status", PreflightStatus(self.status))
        object.__setattr__(self, "severity", PreflightSeverity(self.severity))
        object.__setattr__(self, "details", _safe_dict(dict(self.details)))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["category"] = _value(self.category)
        payload["status"] = _value(self.status)
        payload["severity"] = _value(self.severity)
        return sanitize_for_frontend(payload)


@dataclass(frozen=True)
class TrainingPreflightProviderResult:
    provider: str
    status: PreflightStatus | str
    launch_ready: bool
    checks: list[TrainingPreflightCheck] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    warning_reasons: list[str] = field(default_factory=list)
    unknown_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", PreflightStatus(self.status))
        object.__setattr__(self, "metadata", _safe_dict(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return sanitize_for_frontend(
            {
                "provider": self.provider,
                "status": self.status.value,
                "launch_ready": self.launch_ready,
                "checks": [check.to_dict() for check in self.checks],
                "blocking_reasons": list(self.blocking_reasons),
                "warning_reasons": list(self.warning_reasons),
                "unknown_reasons": list(self.unknown_reasons),
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class TrainingPreflightFallbackResult:
    fallback_id: str
    provider: str
    model_id: str | None = None
    hardware_id: str | None = None
    status: PreflightStatus | str = PreflightStatus.NOT_RUN
    launch_ready: bool = False
    checks: list[TrainingPreflightCheck] = field(default_factory=list)
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", PreflightStatus(self.status))
        object.__setattr__(self, "metadata", _safe_dict(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return sanitize_for_frontend(
            {
                "fallback_id": self.fallback_id,
                "provider": self.provider,
                "model_id": self.model_id,
                "hardware_id": self.hardware_id,
                "status": self.status.value,
                "launch_ready": self.launch_ready,
                "checks": [check.to_dict() for check in self.checks],
                "reason": self.reason,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class TrainingPreflightCacheInfo:
    cache_key: str | None = None
    hit: bool = False
    ttl_seconds: int | None = None
    created_at: str | None = None
    expires_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingPreflightResult:
    preflight_id: str
    session_id: str
    run_id: str | None
    created_at: str
    updated_at: str
    status: PreflightStatus | str
    launch_ready: bool
    provider: str
    model_id: str
    hardware_id: str | None
    output_policy: str
    primary: TrainingPreflightProviderResult
    fallbacks: list[TrainingPreflightFallbackResult] = field(default_factory=list)
    verified_recommendation: dict[str, Any] | None = None
    blocking_reasons: list[str] = field(default_factory=list)
    warning_reasons: list[str] = field(default_factory=list)
    unknown_reasons: list[str] = field(default_factory=list)
    safe_summary: str = ""
    cache: TrainingPreflightCacheInfo = field(
        default_factory=TrainingPreflightCacheInfo
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", PreflightStatus(self.status))
        object.__setattr__(self, "metadata", _safe_dict(dict(self.metadata)))
        if self.verified_recommendation is not None:
            object.__setattr__(
                self,
                "verified_recommendation",
                _safe_dict(dict(self.verified_recommendation)),
            )

    def to_dict(self) -> dict[str, Any]:
        return sanitize_for_frontend(
            {
                "preflight_id": self.preflight_id,
                "session_id": self.session_id,
                "run_id": self.run_id,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "status": self.status.value,
                "launch_ready": self.launch_ready,
                "provider": self.provider,
                "model_id": self.model_id,
                "hardware_id": self.hardware_id,
                "output_policy": self.output_policy,
                "primary": self.primary.to_dict(),
                "fallbacks": [fallback.to_dict() for fallback in self.fallbacks],
                "verified_recommendation": self.verified_recommendation,
                "blocking_reasons": list(self.blocking_reasons),
                "warning_reasons": list(self.warning_reasons),
                "unknown_reasons": list(self.unknown_reasons),
                "safe_summary": self.safe_summary,
                "cache": self.cache.to_dict(),
                "metadata": self.metadata,
            }
        )


def derive_launch_ready(
    checks: list[TrainingPreflightCheck],
    *,
    allow_unknown_override: bool = False,
) -> tuple[bool, list[str], list[str], list[str]]:
    blocking_reasons: list[str] = []
    warning_reasons: list[str] = []
    unknown_reasons: list[str] = []

    for check in checks:
        if _is_non_applicable_skip(check):
            continue
        if check.status == PreflightStatus.FAILED and check.severity in {
            PreflightSeverity.ERROR,
            PreflightSeverity.BLOCKING,
        }:
            blocking_reasons.append(_reason(check))
        elif check.status == PreflightStatus.UNKNOWN and _is_required(check):
            unknown_reasons.append(_reason(check))
        elif check.status == PreflightStatus.SKIPPED and _is_required(check):
            unknown_reasons.append(_reason(check))
        elif (
            check.status == PreflightStatus.WARNING
            or check.severity == PreflightSeverity.WARNING
        ):
            warning_reasons.append(_reason(check))

    launch_ready = not blocking_reasons and (
        allow_unknown_override or not unknown_reasons
    )
    return launch_ready, blocking_reasons, warning_reasons, unknown_reasons


def _derive_status(
    checks: list[TrainingPreflightCheck],
    *,
    blocking_reasons: list[str],
    warning_reasons: list[str],
    unknown_reasons: list[str],
) -> PreflightStatus:
    if not checks:
        return PreflightStatus.NOT_RUN
    if blocking_reasons:
        return PreflightStatus.FAILED
    if unknown_reasons:
        return PreflightStatus.UNKNOWN
    if warning_reasons:
        return PreflightStatus.WARNING
    return PreflightStatus.PASSED


def _safe_summary(
    *,
    status: PreflightStatus,
    launch_ready: bool,
    blocking_reasons: list[str],
    warning_reasons: list[str],
    unknown_reasons: list[str],
) -> str:
    if blocking_reasons:
        reason = blocking_reasons[0]
    elif unknown_reasons:
        reason = unknown_reasons[0]
    elif warning_reasons:
        reason = warning_reasons[0]
    elif status == PreflightStatus.NOT_RUN:
        reason = "No preflight checks have run."
    else:
        reason = "All required preflight checks passed or were non-applicable."
    summary = (
        f"Training preflight {status.value}; "
        f"launch_ready={str(launch_ready).lower()}. {reason}"
    )
    return str(sanitize_for_frontend(summary))


def build_training_preflight_result(
    *,
    session_id: str,
    run_id: str | None = None,
    provider: str,
    model_id: str,
    hardware_id: str | None,
    output_policy: str,
    checks: list[TrainingPreflightCheck] | None = None,
    fallbacks: list[TrainingPreflightFallbackResult] | None = None,
    verified_recommendation: dict[str, Any] | None = None,
    cache: TrainingPreflightCacheInfo | None = None,
    metadata: dict[str, Any] | None = None,
    preflight_id: str | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
    allow_unknown_override: bool = False,
) -> TrainingPreflightResult:
    all_checks = list(checks or [])
    launch_ready, blocking, warnings, unknowns = derive_launch_ready(
        all_checks,
        allow_unknown_override=allow_unknown_override,
    )
    status = _derive_status(
        all_checks,
        blocking_reasons=blocking,
        warning_reasons=warnings,
        unknown_reasons=unknowns,
    )
    timestamp = _utc_now()
    safe_metadata = {
        "provider_jobs_launched": False,
        "resources_created": False,
        "live_checks_optional": True,
        **(metadata or {}),
    }
    primary = TrainingPreflightProviderResult(
        provider=provider,
        status=status,
        launch_ready=launch_ready,
        checks=all_checks,
        blocking_reasons=blocking,
        warning_reasons=warnings,
        unknown_reasons=unknowns,
    )
    return TrainingPreflightResult(
        preflight_id=preflight_id or f"preflight_{uuid4().hex}",
        session_id=session_id,
        run_id=run_id,
        created_at=created_at or timestamp,
        updated_at=updated_at or timestamp,
        status=status,
        launch_ready=launch_ready,
        provider=provider,
        model_id=model_id,
        hardware_id=hardware_id,
        output_policy=output_policy,
        primary=primary,
        fallbacks=list(fallbacks or []),
        verified_recommendation=verified_recommendation,
        blocking_reasons=blocking,
        warning_reasons=warnings,
        unknown_reasons=unknowns,
        safe_summary=_safe_summary(
            status=status,
            launch_ready=launch_ready,
            blocking_reasons=blocking,
            warning_reasons=warnings,
            unknown_reasons=unknowns,
        ),
        cache=cache or TrainingPreflightCacheInfo(),
        metadata=safe_metadata,
    )
