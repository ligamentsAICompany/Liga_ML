"""Foundation models and deterministic readiness for training preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from agent.core.model_provider_selection import (
    catalog_hardware,
    catalog_model,
    catalog_provider,
    hardware_catalog,
)
from agent.core.output_policy import (
    VALID_OUTPUT_POLICIES,
    output_policy_requires_cloud_storage,
    output_policy_requires_hub,
)
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
    verified_fallback: TrainingPreflightFallbackResult | None = None
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
                "verified_fallback": self.verified_fallback.to_dict()
                if self.verified_fallback
                else None,
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


def _recommendation_body(recommendation: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(recommendation, dict):
        return {}
    nested = recommendation.get("recommendation")
    return nested if isinstance(nested, dict) else {}


def _nested_mapping(
    recommendation: dict[str, Any],
    nested_key: str,
) -> dict[str, Any]:
    nested = _recommendation_body(recommendation).get(nested_key)
    return nested if isinstance(nested, dict) else {}


def _extract_provider(recommendation: dict[str, Any] | None) -> str:
    if not isinstance(recommendation, dict):
        return "unknown"
    selected = _nested_mapping(recommendation, "selected_provider")
    return str(
        selected.get("provider_id")
        or recommendation.get("provider")
        or recommendation.get("cloud_provider")
        or "unknown"
    )


def _extract_model_id(recommendation: dict[str, Any] | None) -> str:
    if not isinstance(recommendation, dict):
        return "unknown"
    selected = _nested_mapping(recommendation, "selected_model")
    return str(
        selected.get("model_id")
        or recommendation.get("recommended_model")
        or recommendation.get("model_id")
        or ""
    )


def _extract_hardware_id(recommendation: dict[str, Any] | None) -> str | None:
    if not isinstance(recommendation, dict):
        return None
    selected = _nested_mapping(recommendation, "selected_hardware")
    value = selected.get("hardware_id") or recommendation.get("hardware_id")
    return str(value) if value else None


def _extract_output_policy(recommendation: dict[str, Any] | None) -> str:
    if not isinstance(recommendation, dict):
        return ""
    nested = _recommendation_body(recommendation)
    return str(nested.get("output_policy") or recommendation.get("output_policy") or "")


def _has_dataset_context(
    dataset_summary: dict[str, Any] | None,
    dataset_discovery: dict[str, Any] | None,
) -> bool:
    if isinstance(dataset_summary, dict) and dataset_summary:
        return True
    if not isinstance(dataset_discovery, dict) or not dataset_discovery:
        return False
    return bool(
        dataset_discovery.get("selected_candidate")
        or dataset_discovery.get("recommended_candidate")
        or dataset_discovery.get("candidates")
    )


def _fallback_entries(recommendation: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(recommendation, dict):
        return []
    body = _recommendation_body(recommendation)
    value = body.get("fallbacks") or recommendation.get("fallbacks")
    return (
        [dict(item) for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def _first_compatible_hardware_id(provider: str, model_id: str) -> str | None:
    model = catalog_model(model_id) if model_id else None
    for hardware in hardware_catalog():
        if hardware.provider_id != provider:
            continue
        limit = hardware.suitable_model_size_b
        if model is None or limit is None or model.parameter_count_b <= limit:
            return hardware.hardware_id
    return None


def _status_from_reasons(
    checks: list[TrainingPreflightCheck],
    blocking: list[str],
    warnings: list[str],
    unknowns: list[str],
) -> PreflightStatus:
    return _derive_status(
        checks,
        blocking_reasons=blocking,
        warning_reasons=warnings,
        unknown_reasons=unknowns,
    )


def _fallback_preflight_result(
    *,
    index: int,
    fallback: dict[str, Any],
    provider: str,
    model_id: str,
    hardware_id: str | None,
    output_policy: str,
) -> TrainingPreflightFallbackResult:
    fallback_option = str(fallback.get("fallback_option") or "").strip()
    blocked_option = str(fallback.get("blocked_option") or "").strip()
    reason = str(fallback.get("reason") or "Static planner fallback.")
    fallback_provider = provider
    fallback_model_id = model_id
    fallback_hardware_id = hardware_id
    recognized = False
    mapping_message = (
        "Fallback option could not be mapped to a known provider, model, or hardware."
    )

    if catalog_provider(fallback_option):
        fallback_provider = fallback_option
        fallback_hardware_id = _first_compatible_hardware_id(
            fallback_provider, model_id
        )
        recognized = True
        mapping_message = (
            f"Fallback provider {fallback_provider} is in the static catalog."
        )
    elif hardware := catalog_hardware(fallback_option):
        fallback_provider = hardware.provider_id
        fallback_hardware_id = hardware.hardware_id
        recognized = True
        mapping_message = (
            f"Fallback hardware {fallback_hardware_id} is in the static catalog."
        )
    elif catalog_model(fallback_option):
        fallback_model_id = fallback_option
        recognized = True
        mapping_message = (
            f"Fallback model {fallback_model_id} is in the static catalog."
        )

    checks: list[TrainingPreflightCheck] = [
        _check(
            check_id="fallback_static_mapping",
            provider=fallback_provider,
            category=PreflightCheckCategory.FALLBACK,
            label="Fallback mapping",
            status=PreflightStatus.PASSED if recognized else PreflightStatus.UNKNOWN,
            severity=PreflightSeverity.INFO
            if recognized
            else PreflightSeverity.WARNING,
            message=mapping_message,
            error_code=None if recognized else "fallback_mapping_unknown",
            details={
                "blocked_option": blocked_option,
                "fallback_option": fallback_option,
            },
            docs_verification_required=not recognized,
        )
    ]

    provider_known = catalog_provider(fallback_provider) is not None
    checks.append(
        _check(
            check_id="fallback_provider_catalog",
            provider=fallback_provider,
            category=PreflightCheckCategory.FALLBACK,
            label="Fallback provider",
            status=PreflightStatus.PASSED
            if provider_known
            else PreflightStatus.UNKNOWN,
            severity=PreflightSeverity.INFO
            if provider_known
            else PreflightSeverity.WARNING,
            message=(
                f"Fallback provider {fallback_provider} is known."
                if provider_known
                else "Fallback provider is not in the static catalog."
            ),
            error_code=None if provider_known else "unsupported_provider",
            docs_verification_required=not provider_known,
        )
    )

    model_known = catalog_model(fallback_model_id) is not None
    checks.append(
        _check(
            check_id="fallback_model_catalog",
            provider=fallback_provider,
            category=PreflightCheckCategory.FALLBACK,
            label="Fallback model",
            status=PreflightStatus.PASSED if model_known else PreflightStatus.UNKNOWN,
            severity=PreflightSeverity.INFO
            if model_known
            else PreflightSeverity.WARNING,
            message=(
                f"Fallback model {fallback_model_id} is known."
                if model_known
                else "Fallback model is outside the static catalog."
            ),
            error_code=None if model_known else "model_catalog_unknown",
            docs_verification_required=not model_known,
        )
    )

    hardware = catalog_hardware(fallback_hardware_id) if fallback_hardware_id else None
    hardware_known = hardware is not None and hardware.provider_id == fallback_provider
    checks.append(
        _check(
            check_id="fallback_hardware_catalog",
            provider=fallback_provider,
            category=PreflightCheckCategory.FALLBACK,
            label="Fallback hardware",
            status=PreflightStatus.PASSED
            if hardware_known
            else PreflightStatus.UNKNOWN,
            severity=PreflightSeverity.INFO
            if hardware_known
            else PreflightSeverity.WARNING,
            message=(
                f"Fallback hardware {fallback_hardware_id} is known for {fallback_provider}."
                if hardware_known
                else "Fallback hardware is not known for the fallback provider."
            ),
            error_code=None if hardware_known else "hardware_catalog_unknown",
            docs_verification_required=not hardware_known,
        )
    )

    model = catalog_model(fallback_model_id) if fallback_model_id else None
    if recognized and model is not None and hardware is not None:
        limit = hardware.suitable_model_size_b
        fits = limit is None or model.parameter_count_b <= limit
        checks.append(
            _check(
                check_id="fallback_model_hardware_fit",
                provider=fallback_provider,
                category=PreflightCheckCategory.FALLBACK,
                label="Fallback model/hardware fit",
                status=PreflightStatus.PASSED if fits else PreflightStatus.FAILED,
                severity=PreflightSeverity.INFO if fits else PreflightSeverity.BLOCKING,
                message=(
                    "Fallback model fits the fallback hardware static memory estimate."
                    if fits
                    else "Fallback model exceeds the fallback hardware static memory estimate."
                ),
                error_code=None if fits else "hardware_memory_incompatible",
            )
        )

    checks.append(
        _check(
            check_id="fallback_not_executed",
            provider=fallback_provider,
            category=PreflightCheckCategory.SAFETY,
            label="Fallback execution",
            status=PreflightStatus.SKIPPED,
            severity=PreflightSeverity.INFO,
            message="Fallback verification is advisory only; no fallback was automatically launched.",
            required=False,
            applicable=False,
            details={
                "advisory_only": True,
                "fallback_executed": False,
                "automatic_fallback_execution": False,
            },
        )
    )

    launch_ready, blocking, warnings, unknowns = derive_launch_ready(checks)
    status = _status_from_reasons(checks, blocking, warnings, unknowns)
    return TrainingPreflightFallbackResult(
        fallback_id=f"fallback_{index}",
        provider=fallback_provider,
        model_id=fallback_model_id or None,
        hardware_id=fallback_hardware_id,
        status=status,
        launch_ready=launch_ready,
        checks=checks,
        reason=reason,
        metadata={
            "advisory_only": True,
            "fallback_executed": False,
            "automatic_fallback_execution": False,
            "provider_jobs_launched": False,
            "resources_created": False,
        },
    )


def _fallback_preflight_results(
    *,
    recommendation: dict[str, Any] | None,
    provider: str,
    model_id: str,
    hardware_id: str | None,
    output_policy: str,
    include_fallbacks: bool,
) -> list[TrainingPreflightFallbackResult]:
    if not include_fallbacks:
        return []
    return [
        _fallback_preflight_result(
            index=index,
            fallback=fallback,
            provider=provider,
            model_id=model_id,
            hardware_id=hardware_id,
            output_policy=output_policy,
        )
        for index, fallback in enumerate(_fallback_entries(recommendation), start=1)
    ]


def _check(
    *,
    check_id: str,
    provider: str,
    category: PreflightCheckCategory,
    label: str,
    status: PreflightStatus,
    severity: PreflightSeverity,
    message: str,
    required: bool = True,
    applicable: bool = True,
    details: dict[str, Any] | None = None,
    error_code: str | None = None,
    docs_verification_required: bool = False,
) -> TrainingPreflightCheck:
    return TrainingPreflightCheck(
        check_id=check_id,
        provider=provider,
        category=category,
        label=label,
        status=status,
        severity=severity,
        message=message,
        details={
            "required": required,
            "applicable": applicable,
            **(details or {}),
        },
        started_at=_utc_now(),
        completed_at=_utc_now(),
        duration_ms=0,
        error_code=error_code,
        docs_verification_required=docs_verification_required,
    )


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
    verified_fallback: TrainingPreflightFallbackResult | None = None,
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
        verified_fallback=verified_fallback,
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


def run_local_training_preflight(
    *,
    session_id: str,
    run_id: str | None = None,
    recommendation: dict[str, Any] | None = None,
    dataset_summary: dict[str, Any] | None = None,
    dataset_discovery: dict[str, Any] | None = None,
    target_namespace: str | None = None,
    target_repo_id: str | None = None,
    target_bucket: str | None = None,
    include_fallbacks: bool = False,
    force_refresh: bool = False,
    timeout_seconds: int | None = None,
    metadata: dict[str, Any] | None = None,
    allow_unknown_override: bool = False,
) -> TrainingPreflightResult:
    """Run Slice 2 local-only checks without provider SDKs or network calls."""

    _ = force_refresh, timeout_seconds
    provider = _extract_provider(recommendation)
    model_id = _extract_model_id(recommendation)
    hardware_id = _extract_hardware_id(recommendation)
    output_policy = _extract_output_policy(recommendation)
    checks: list[TrainingPreflightCheck] = []

    if not isinstance(recommendation, dict) or not recommendation:
        checks.append(
            _check(
                check_id="static_recommendation",
                provider=provider,
                category=PreflightCheckCategory.METADATA,
                label="Static recommendation",
                status=PreflightStatus.FAILED,
                severity=PreflightSeverity.BLOCKING,
                message="Static training recommendation is required before preflight.",
                error_code="missing_recommendation",
            )
        )
    else:
        checks.append(
            _check(
                check_id="static_recommendation",
                provider=provider,
                category=PreflightCheckCategory.METADATA,
                label="Static recommendation",
                status=PreflightStatus.PASSED,
                severity=PreflightSeverity.INFO,
                message="Static training recommendation is present.",
            )
        )

    provider_candidate = catalog_provider(provider)
    checks.append(
        _check(
            check_id="provider_present",
            provider=provider,
            category=PreflightCheckCategory.API,
            label="Provider",
            status=PreflightStatus.PASSED
            if provider_candidate
            else PreflightStatus.FAILED,
            severity=PreflightSeverity.INFO
            if provider_candidate
            else PreflightSeverity.BLOCKING,
            message=(
                f"Provider {provider} is in the static catalog."
                if provider_candidate
                else "Provider is missing or not recognized by the static catalog."
            ),
            error_code=None if provider_candidate else "unsupported_provider",
        )
    )

    model_candidate = catalog_model(model_id) if model_id else None
    checks.append(
        _check(
            check_id="model_id_present",
            provider=provider,
            category=PreflightCheckCategory.MODEL_ACCESS,
            label="Model id",
            status=PreflightStatus.PASSED if model_id else PreflightStatus.FAILED,
            severity=PreflightSeverity.INFO if model_id else PreflightSeverity.BLOCKING,
            message=(
                f"Model id {model_id} is present."
                if model_id
                else "Model id is required before preflight."
            ),
            error_code=None if model_id else "missing_model_id",
        )
    )
    if model_id and not model_candidate:
        checks.append(
            _check(
                check_id="model_static_catalog",
                provider=provider,
                category=PreflightCheckCategory.METADATA,
                label="Model static catalog",
                status=PreflightStatus.WARNING,
                severity=PreflightSeverity.WARNING,
                message="Model is outside the static catalog; live metadata verification is still required.",
                required=False,
            )
        )

    hardware_candidate = catalog_hardware(hardware_id) if hardware_id else None
    checks.append(
        _check(
            check_id="hardware_id_present",
            provider=provider,
            category=PreflightCheckCategory.HARDWARE,
            label="Hardware id",
            status=PreflightStatus.PASSED
            if hardware_candidate
            else PreflightStatus.FAILED,
            severity=PreflightSeverity.INFO
            if hardware_candidate
            else PreflightSeverity.BLOCKING,
            message=(
                f"Hardware id {hardware_id} is in the static catalog."
                if hardware_candidate
                else "Hardware id is required and must be recognized before preflight."
            ),
            error_code=None if hardware_candidate else "missing_or_unknown_hardware",
        )
    )

    valid_policy = output_policy in VALID_OUTPUT_POLICIES
    checks.append(
        _check(
            check_id="output_policy_present",
            provider=provider,
            category=PreflightCheckCategory.OUTPUT_POLICY,
            label="Output policy",
            status=PreflightStatus.PASSED if valid_policy else PreflightStatus.FAILED,
            severity=PreflightSeverity.INFO
            if valid_policy
            else PreflightSeverity.BLOCKING,
            message=(
                f"Output policy {output_policy} is recognized."
                if valid_policy
                else "Recognized output policy is required before preflight."
            ),
            error_code=None if valid_policy else "missing_or_unknown_output_policy",
        )
    )

    has_dataset = _has_dataset_context(dataset_summary, dataset_discovery)
    checks.append(
        _check(
            check_id="dataset_context",
            provider=provider,
            category=PreflightCheckCategory.METADATA,
            label="Dataset context",
            status=PreflightStatus.PASSED if has_dataset else PreflightStatus.WARNING,
            severity=PreflightSeverity.INFO
            if has_dataset
            else PreflightSeverity.WARNING,
            message=(
                "Dataset summary or discovery context is present."
                if has_dataset
                else "Dataset summary or selected discovery context is not available yet."
            ),
            required=False,
        )
    )

    if provider_candidate and valid_policy:
        checks.append(
            _check(
                check_id="provider_output_policy_compatibility",
                provider=provider,
                category=PreflightCheckCategory.COMPATIBILITY,
                label="Provider/output policy compatibility",
                status=PreflightStatus.PASSED,
                severity=PreflightSeverity.INFO,
                message="Provider and output policy are locally plausible.",
                details={
                    "supports_private_output": provider_candidate.supports_private_output,
                    "supports_hub_output": provider_candidate.supports_hub_output,
                },
            )
        )

    if model_candidate and hardware_candidate:
        limit = hardware_candidate.suitable_model_size_b
        fits = limit is None or model_candidate.parameter_count_b <= limit
        checks.append(
            _check(
                check_id="model_hardware_memory_fit",
                provider=provider,
                category=PreflightCheckCategory.HARDWARE,
                label="Model/hardware memory fit",
                status=PreflightStatus.PASSED if fits else PreflightStatus.FAILED,
                severity=PreflightSeverity.INFO if fits else PreflightSeverity.BLOCKING,
                message=(
                    "Static model size fits the selected hardware memory estimate."
                    if fits
                    else "Static model size exceeds the selected hardware memory estimate."
                ),
                error_code=None if fits else "hardware_memory_incompatible",
                details={
                    "model_parameter_count_b": model_candidate.parameter_count_b,
                    "hardware_suitable_model_size_b": limit,
                    "gpu_memory_gb": hardware_candidate.gpu_memory_gb,
                },
            )
        )

    checks.append(
        _check(
            check_id="provider_credentials_live",
            provider=provider,
            category=PreflightCheckCategory.CREDENTIALS,
            label="Provider credentials",
            status=PreflightStatus.UNKNOWN,
            severity=PreflightSeverity.ERROR,
            message="Live provider credential checks are not implemented in this slice.",
            error_code="live_probe_not_implemented",
            docs_verification_required=True,
        )
    )
    if valid_policy and output_policy_requires_hub(output_policy):
        checks.append(
            _check(
                check_id="hub_write_live",
                provider=provider,
                category=PreflightCheckCategory.OUTPUT_POLICY,
                label="Hub write access",
                status=PreflightStatus.UNKNOWN,
                severity=PreflightSeverity.ERROR,
                message="Hugging Face Hub write access requires a later live preflight probe.",
                details={
                    "target_namespace": target_namespace,
                    "target_repo_id": target_repo_id,
                },
                error_code="live_probe_not_implemented",
                docs_verification_required=True,
            )
        )
    if valid_policy and output_policy_requires_cloud_storage(output_policy):
        checks.append(
            _check(
                check_id="cloud_storage_live",
                provider=provider,
                category=PreflightCheckCategory.STORAGE,
                label="Cloud storage writability",
                status=PreflightStatus.UNKNOWN,
                severity=PreflightSeverity.ERROR,
                message="Cloud storage writability requires a later live preflight probe.",
                details={"target_bucket": target_bucket},
                error_code="live_probe_not_implemented",
                docs_verification_required=True,
            )
        )
    checks.append(
        _check(
            check_id="provider_job_launch",
            provider=provider,
            category=PreflightCheckCategory.SAFETY,
            label="Provider job launch",
            status=PreflightStatus.SKIPPED,
            severity=PreflightSeverity.INFO,
            message="Provider job launch is intentionally skipped for local preflight.",
            required=False,
            applicable=False,
        )
    )

    fallback_results = _fallback_preflight_results(
        recommendation=recommendation,
        provider=provider,
        model_id=model_id,
        hardware_id=hardware_id,
        output_policy=output_policy,
        include_fallbacks=include_fallbacks,
    )
    verified_fallback = next(
        (
            fallback
            for fallback in fallback_results
            if fallback.launch_ready and fallback.status == PreflightStatus.PASSED
        ),
        None,
    )

    result_metadata = {
        "mode": "local_non_network",
        "live_provider_checks": False,
        "live_preflight_probe_status": "not_implemented",
        "fallbacks_checked": include_fallbacks,
        "fallback_count": len(fallback_results),
        "verified_fallback_id": verified_fallback.fallback_id
        if verified_fallback
        else None,
        "automatic_fallback_execution": False,
        **(metadata or {}),
    }
    result = build_training_preflight_result(
        session_id=session_id,
        run_id=run_id,
        provider=provider,
        model_id=model_id or "unknown",
        hardware_id=hardware_id,
        output_policy=output_policy or "unknown",
        checks=checks,
        fallbacks=fallback_results,
        verified_fallback=verified_fallback,
        verified_recommendation=sanitize_for_frontend(recommendation)
        if isinstance(recommendation, dict)
        else None,
        metadata=result_metadata,
        allow_unknown_override=allow_unknown_override,
    )
    if result.status == PreflightStatus.UNKNOWN:
        summary = (
            f"{result.safe_summary} Live provider probes are not implemented in this slice, "
            "so launch_ready remains false until a later live preflight passes."
        )
        object.__setattr__(result, "safe_summary", str(sanitize_for_frontend(summary)))
    return result


def _needs_hf_live_checks(provider: str, model_id: str, output_policy: str) -> bool:
    if provider in {"gcp-vertex", "aws-sagemaker"}:
        return output_policy_requires_hub(output_policy)
    return (
        provider == "hf-jobs"
        or output_policy_requires_hub(output_policy)
        or bool(model_id and "/" in model_id)
    )


def _needs_gcp_live_checks(provider: str, output_policy: str) -> bool:
    return provider == "gcp-vertex" and (
        output_policy_requires_cloud_storage(output_policy)
        or output_policy_requires_hub(output_policy)
        or output_policy in VALID_OUTPUT_POLICIES
    )


def _needs_aws_live_checks(provider: str, output_policy: str) -> bool:
    return provider == "aws-sagemaker" and (
        output_policy_requires_cloud_storage(output_policy)
        or output_policy_requires_hub(output_policy)
        or output_policy in VALID_OUTPUT_POLICIES
    )


def _local_check_superseded_by_hf(
    check: TrainingPreflightCheck,
    *,
    provider: str,
    output_policy: str,
) -> bool:
    if check.check_id == "provider_credentials_live":
        return True
    if check.check_id == "hub_write_live" and output_policy_requires_hub(output_policy):
        return True
    return check.check_id == "cloud_storage_live" and provider == "hf-jobs"


def _local_check_superseded_by_gcp(
    check: TrainingPreflightCheck,
    *,
    provider: str,
    output_policy: str,
) -> bool:
    if provider != "gcp-vertex":
        return False
    if check.check_id == "provider_credentials_live":
        return True
    if check.check_id == "cloud_storage_live" and output_policy_requires_cloud_storage(
        output_policy
    ):
        return True
    return False


def _local_check_superseded_by_aws(
    check: TrainingPreflightCheck,
    *,
    provider: str,
    output_policy: str,
) -> bool:
    if provider != "aws-sagemaker":
        return False
    if check.check_id == "provider_credentials_live":
        return True
    if check.check_id == "cloud_storage_live" and output_policy_requires_cloud_storage(
        output_policy
    ):
        return True
    return False


async def run_training_preflight(
    *,
    session_id: str,
    run_id: str | None = None,
    recommendation: dict[str, Any] | None = None,
    dataset_summary: dict[str, Any] | None = None,
    dataset_discovery: dict[str, Any] | None = None,
    target_namespace: str | None = None,
    target_repo_id: str | None = None,
    target_bucket: str | None = None,
    include_fallbacks: bool = False,
    force_refresh: bool = False,
    timeout_seconds: int | None = None,
    metadata: dict[str, Any] | None = None,
    allow_unknown_override: bool = False,
    hf_token: str | None = None,
    hf_client_factory: Any | None = None,
    jobs_namespace_resolver: Any | None = None,
    gcp_project_id: str | None = None,
    gcp_region: str | None = None,
    gcp_client_factory: Any | None = None,
    aws_region: str | None = None,
    aws_execution_role_arn: str | None = None,
    aws_client_factory: Any | None = None,
) -> TrainingPreflightResult:
    """Run training preflight, including safe read-only provider probes."""

    local = run_local_training_preflight(
        session_id=session_id,
        run_id=run_id,
        recommendation=recommendation,
        dataset_summary=dataset_summary,
        dataset_discovery=dataset_discovery,
        target_namespace=target_namespace,
        target_repo_id=target_repo_id,
        target_bucket=target_bucket,
        include_fallbacks=include_fallbacks,
        force_refresh=force_refresh,
        timeout_seconds=timeout_seconds,
        metadata=metadata,
        allow_unknown_override=allow_unknown_override,
    )
    needs_gcp = _needs_gcp_live_checks(local.provider, local.output_policy)
    needs_aws = _needs_aws_live_checks(local.provider, local.output_policy)
    needs_hf = _needs_hf_live_checks(
        local.provider, local.model_id, local.output_policy
    )
    if not needs_gcp and not needs_aws and not needs_hf:
        return local

    local_checks = [
        check
        for check in local.primary.checks
        if not _local_check_superseded_by_gcp(
            check,
            provider=local.provider,
            output_policy=local.output_policy,
        )
        and not _local_check_superseded_by_aws(
            check,
            provider=local.provider,
            output_policy=local.output_policy,
        )
        and not _local_check_superseded_by_hf(
            check,
            provider=local.provider,
            output_policy=local.output_policy,
        )
    ]
    provider_checks: list[TrainingPreflightCheck] = []
    live_modes: list[str] = []
    if needs_gcp:
        from agent.core.preflight_gcp_vertex import run_gcp_vertex_preflight_checks

        gcp_result = await run_gcp_vertex_preflight_checks(
            provider=local.provider,
            model_id=local.model_id,
            hardware_id=local.hardware_id,
            output_policy=local.output_policy,
            target_bucket=target_bucket,
            project_id=gcp_project_id,
            region=gcp_region,
            timeout_seconds=timeout_seconds,
            gcp_client_factory=gcp_client_factory,
        )
        provider_checks.extend(gcp_result.checks)
        live_modes.append("gcp_vertex_read_only")
    if needs_aws:
        from agent.core.preflight_aws_sagemaker import (
            run_aws_sagemaker_preflight_checks,
        )

        aws_result = await run_aws_sagemaker_preflight_checks(
            provider=local.provider,
            model_id=local.model_id,
            hardware_id=local.hardware_id,
            output_policy=local.output_policy,
            target_bucket=target_bucket,
            aws_region=aws_region,
            execution_role_arn=aws_execution_role_arn,
            timeout_seconds=timeout_seconds,
            aws_client_factory=aws_client_factory,
        )
        provider_checks.extend(aws_result.checks)
        live_modes.append("aws_sagemaker_read_only")
    if needs_hf:
        from agent.core.preflight_hf import run_hf_preflight_checks

        hf_result = await run_hf_preflight_checks(
            provider=local.provider,
            model_id=local.model_id,
            hardware_id=local.hardware_id,
            output_policy=local.output_policy,
            target_namespace=target_namespace,
            target_repo_id=target_repo_id,
            hf_token=hf_token,
            timeout_seconds=timeout_seconds,
            hf_client_factory=hf_client_factory,
            jobs_namespace_resolver=jobs_namespace_resolver,
        )
        provider_checks.extend(hf_result.checks)
        live_modes.append("hf_read_only")

    combined_metadata = {
        **local.metadata,
        "mode": "+".join(live_modes),
        "live_provider_checks": True,
        "live_preflight_probe_status": "+".join(live_modes),
        "provider_jobs_launched": False,
        "resources_created": False,
    }
    result = build_training_preflight_result(
        session_id=session_id,
        run_id=run_id,
        provider=local.provider,
        model_id=local.model_id,
        hardware_id=local.hardware_id,
        output_policy=local.output_policy,
        checks=[*local_checks, *provider_checks],
        fallbacks=local.fallbacks,
        verified_fallback=local.verified_fallback,
        verified_recommendation=local.verified_recommendation,
        cache=local.cache,
        metadata=combined_metadata,
        preflight_id=local.preflight_id,
        created_at=local.created_at,
        allow_unknown_override=allow_unknown_override,
    )
    return result
