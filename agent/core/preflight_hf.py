"""Read-only Hugging Face preflight probes.

This module deliberately uses only metadata APIs. It never creates repos,
uploads files, downloads weights, launches jobs, or mutates provider state.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from agent.core.hf_access import JobsAccess, resolve_jobs_namespace
from agent.core.output_policy import output_policy_requires_hub
from agent.core.preflight_errors import normalize_provider_error
from agent.core.redact import sanitize_for_frontend
from agent.core.training_preflight import (
    PreflightCheckCategory,
    PreflightSeverity,
    PreflightStatus,
    TrainingPreflightCheck,
    TrainingPreflightProviderResult,
    _utc_now,
    derive_launch_ready,
)


class HfReadOnlyClient(Protocol):
    def whoami(self) -> dict[str, Any]: ...

    def model_info(self, repo_id: str) -> Any: ...

    def repo_info(self, repo_id: str, repo_type: str = "model") -> Any: ...


HfClientFactory = Callable[[str | None], HfReadOnlyClient]
JobsNamespaceResolver = Callable[
    [str, str | None],
    Awaitable[tuple[str, JobsAccess | None]],
]


@dataclass(frozen=True)
class _HfApiReadOnlyClient:
    token: str | None

    def __post_init__(self) -> None:
        from huggingface_hub import HfApi

        object.__setattr__(self, "_api", HfApi(token=self.token))

    def whoami(self) -> dict[str, Any]:
        return self._api.whoami()

    def model_info(self, repo_id: str) -> Any:
        return self._api.model_info(repo_id=repo_id)

    def repo_info(self, repo_id: str, repo_type: str = "model") -> Any:
        return self._api.repo_info(repo_id=repo_id, repo_type=repo_type)


def default_hf_client_factory(token: str | None) -> HfReadOnlyClient:
    return _HfApiReadOnlyClient(token)


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


def _error_check(
    *,
    check_id: str,
    provider: str,
    category: PreflightCheckCategory,
    label: str,
    error: BaseException | str,
    severity: PreflightSeverity = PreflightSeverity.BLOCKING,
    status: PreflightStatus | None = None,
) -> TrainingPreflightCheck:
    normalized = normalize_provider_error(error, provider=provider)
    return _check(
        check_id=check_id,
        provider=provider,
        category=category,
        label=label,
        status=status or PreflightStatus.FAILED,
        severity=severity,
        message=str(normalized["message"]),
        error_code=str(normalized["error_code"]),
        details={"provider": provider},
        docs_verification_required=True,
    )


async def _call_read_only(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(method, *args, **kwargs)


def _sibling_names(model_info: Any) -> set[str]:
    siblings = getattr(model_info, "siblings", None)
    if siblings is None and isinstance(model_info, dict):
        siblings = model_info.get("siblings")
    names: set[str] = set()
    if not isinstance(siblings, list):
        return names
    for sibling in siblings:
        if isinstance(sibling, str):
            names.add(sibling)
            continue
        if isinstance(sibling, dict):
            value = sibling.get("rfilename") or sibling.get("filename")
        else:
            value = getattr(sibling, "rfilename", None) or getattr(
                sibling, "filename", None
            )
        if isinstance(value, str):
            names.add(value)
    return names


def _metadata_check(provider: str, model_info: Any) -> TrainingPreflightCheck:
    names = _sibling_names(model_info)
    has_config = "config.json" in names
    has_tokenizer = bool(
        {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "sentencepiece.bpe.model",
            "spiece.model",
        }
        & names
    )
    if has_config and has_tokenizer:
        return _check(
            check_id="hf.model.metadata",
            provider=provider,
            category=PreflightCheckCategory.METADATA,
            label="HF model metadata",
            status=PreflightStatus.PASSED,
            severity=PreflightSeverity.INFO,
            message="Model config and tokenizer metadata are visible without downloading weights.",
            details={
                "config_present": True,
                "tokenizer_metadata_present": True,
                "inspected_files": sorted(names)[:20],
            },
        )
    return _check(
        check_id="hf.model.metadata",
        provider=provider,
        category=PreflightCheckCategory.METADATA,
        label="HF model metadata",
        status=PreflightStatus.UNKNOWN,
        severity=PreflightSeverity.WARNING,
        message="Model exists, but config/tokenizer metadata could not be fully verified.",
        details={
            "config_present": has_config,
            "tokenizer_metadata_present": has_tokenizer,
            "inspected_files": sorted(names)[:20],
        },
        error_code="metadata_incomplete",
        docs_verification_required=True,
    )


def _target_repo_required(output_policy: str) -> bool:
    return output_policy_requires_hub(output_policy)


def _token_required(provider: str, output_policy: str) -> bool:
    return provider == "hf-jobs" or output_policy_requires_hub(output_policy)


async def run_hf_preflight_checks(
    *,
    provider: str,
    model_id: str,
    hardware_id: str | None,
    output_policy: str,
    target_namespace: str | None = None,
    target_repo_id: str | None = None,
    hf_token: str | None = None,
    timeout_seconds: int | None = None,
    hf_client_factory: HfClientFactory | None = None,
    jobs_namespace_resolver: JobsNamespaceResolver | None = None,
) -> TrainingPreflightProviderResult:
    """Run Hugging Face read-only metadata probes and return provider result."""

    _ = timeout_seconds
    checks: list[TrainingPreflightCheck] = []
    token_required = _token_required(provider, output_policy)

    if token_required and not hf_token:
        checks.append(
            _check(
                check_id="hf.token.present",
                provider=provider,
                category=PreflightCheckCategory.CREDENTIALS,
                label="HF token present",
                status=PreflightStatus.FAILED,
                severity=PreflightSeverity.BLOCKING,
                message="Hugging Face token is required for this provider or output policy.",
                error_code="missing_credentials",
            )
        )
        launch_ready, blocking, warnings, unknowns = derive_launch_ready(checks)
        return TrainingPreflightProviderResult(
            provider=provider,
            status=PreflightStatus.FAILED,
            launch_ready=launch_ready,
            checks=checks,
            blocking_reasons=blocking,
            warning_reasons=warnings,
            unknown_reasons=unknowns,
            metadata={
                "provider_jobs_launched": False,
                "resources_created": False,
                "mode": "hf_read_only_live",
            },
        )

    checks.append(
        _check(
            check_id="hf.token.present",
            provider=provider,
            category=PreflightCheckCategory.CREDENTIALS,
            label="HF token present",
            status=PreflightStatus.PASSED if hf_token else PreflightStatus.SKIPPED,
            severity=PreflightSeverity.INFO,
            message="Hugging Face token is available."
            if hf_token
            else "Hugging Face token is not required for this read-only check.",
            required=token_required,
            applicable=token_required,
        )
    )

    client = (hf_client_factory or default_hf_client_factory)(hf_token)

    if hf_token:
        try:
            whoami = await _call_read_only(client.whoami)
            username = whoami.get("name") if isinstance(whoami, dict) else None
            checks.append(
                _check(
                    check_id="hf.identity.whoami",
                    provider=provider,
                    category=PreflightCheckCategory.IDENTITY,
                    label="HF identity",
                    status=PreflightStatus.PASSED,
                    severity=PreflightSeverity.INFO,
                    message="Hugging Face authenticated identity resolved.",
                    details={"username": username},
                )
            )
        except Exception as error:
            checks.append(
                _error_check(
                    check_id="hf.identity.whoami",
                    provider=provider,
                    category=PreflightCheckCategory.IDENTITY,
                    label="HF identity",
                    error=error,
                )
            )
    else:
        checks.append(
            _check(
                check_id="hf.identity.whoami",
                provider=provider,
                category=PreflightCheckCategory.IDENTITY,
                label="HF identity",
                status=PreflightStatus.SKIPPED,
                severity=PreflightSeverity.INFO,
                message="HF identity check skipped because no token is required for this read-only model metadata check.",
                required=False,
                applicable=False,
            )
        )

    resolved_namespace: str | None = None
    if hf_token:
        try:
            resolver = jobs_namespace_resolver or resolve_jobs_namespace
            resolved_namespace, access = await resolver(
                hf_token or "", target_namespace
            )
            checks.append(
                _check(
                    check_id="hf.namespace.usable",
                    provider=provider,
                    category=PreflightCheckCategory.NAMESPACE,
                    label="HF namespace",
                    status=PreflightStatus.PASSED,
                    severity=PreflightSeverity.INFO,
                    message="Hugging Face namespace is available for read-only preflight context.",
                    details={
                        "requested_namespace": target_namespace,
                        "resolved_namespace": resolved_namespace,
                        "eligible_namespaces": getattr(
                            access, "eligible_namespaces", None
                        ),
                    },
                )
            )
        except Exception as error:
            checks.append(
                _error_check(
                    check_id="hf.namespace.usable",
                    provider=provider,
                    category=PreflightCheckCategory.NAMESPACE,
                    label="HF namespace",
                    error=error,
                )
            )
    else:
        checks.append(
            _check(
                check_id="hf.namespace.usable",
                provider=provider,
                category=PreflightCheckCategory.NAMESPACE,
                label="HF namespace",
                status=PreflightStatus.SKIPPED,
                severity=PreflightSeverity.INFO,
                message="HF namespace check skipped because no token or Hub/Jobs namespace is required.",
                required=False,
                applicable=False,
            )
        )

    model_info: Any | None = None
    try:
        model_info = await _call_read_only(client.model_info, model_id)
        checks.append(
            _check(
                check_id="hf.model.access",
                provider=provider,
                category=PreflightCheckCategory.MODEL_ACCESS,
                label="HF model access",
                status=PreflightStatus.PASSED,
                severity=PreflightSeverity.INFO,
                message="Selected base model is readable via Hugging Face metadata API.",
                details={
                    "model_id": model_id,
                    "gated": getattr(model_info, "gated", None),
                    "private": getattr(model_info, "private", None),
                },
            )
        )
        checks.append(_metadata_check(provider, model_info))
    except Exception as error:
        checks.append(
            _error_check(
                check_id="hf.model.access",
                provider=provider,
                category=PreflightCheckCategory.MODEL_ACCESS,
                label="HF model access",
                error=error,
            )
        )
        checks.append(
            _check(
                check_id="hf.model.metadata",
                provider=provider,
                category=PreflightCheckCategory.METADATA,
                label="HF model metadata",
                status=PreflightStatus.SKIPPED,
                severity=PreflightSeverity.INFO,
                message="Model metadata was skipped because model access failed.",
                required=False,
                applicable=False,
            )
        )

    if _target_repo_required(output_policy):
        if not target_repo_id:
            checks.append(
                _check(
                    check_id="hf.repo.target",
                    provider=provider,
                    category=PreflightCheckCategory.STORAGE,
                    label="HF target repo",
                    status=PreflightStatus.UNKNOWN,
                    severity=PreflightSeverity.WARNING,
                    message="No target Hub repo was provided; read-only preflight will not create one.",
                    error_code="target_repo_missing",
                    details={
                        "target_namespace": target_namespace or resolved_namespace
                    },
                    docs_verification_required=True,
                )
            )
        else:
            try:
                await _call_read_only(
                    client.repo_info, target_repo_id, repo_type="model"
                )
                checks.append(
                    _check(
                        check_id="hf.repo.target",
                        provider=provider,
                        category=PreflightCheckCategory.STORAGE,
                        label="HF target repo",
                        status=PreflightStatus.PASSED,
                        severity=PreflightSeverity.INFO,
                        message="Target Hub repo exists and is readable; no repo was created.",
                        details={"target_repo_id": target_repo_id},
                    )
                )
            except Exception as error:
                normalized = normalize_provider_error(error, provider=provider)
                status = (
                    PreflightStatus.UNKNOWN
                    if normalized["error_code"] == "not_found"
                    else PreflightStatus.FAILED
                )
                severity = (
                    PreflightSeverity.WARNING
                    if status == PreflightStatus.UNKNOWN
                    else PreflightSeverity.BLOCKING
                )
                checks.append(
                    _check(
                        check_id="hf.repo.target",
                        provider=provider,
                        category=PreflightCheckCategory.STORAGE,
                        label="HF target repo",
                        status=status,
                        severity=severity,
                        message=(
                            "Target Hub repo was not found; read-only preflight did not create it."
                            if status == PreflightStatus.UNKNOWN
                            else str(normalized["message"])
                        ),
                        error_code=str(normalized["error_code"]),
                        details={"target_repo_id": target_repo_id},
                        docs_verification_required=True,
                    )
                )
    else:
        checks.append(
            _check(
                check_id="hf.repo.target",
                provider=provider,
                category=PreflightCheckCategory.STORAGE,
                label="HF target repo",
                status=PreflightStatus.SKIPPED,
                severity=PreflightSeverity.INFO,
                message="Output policy does not require Hugging Face Hub output.",
                required=False,
                applicable=False,
            )
        )

    if provider == "hf-jobs":
        namespace_check = next(
            (check for check in checks if check.check_id == "hf.namespace.usable"),
            None,
        )
        if namespace_check and namespace_check.status == PreflightStatus.PASSED:
            checks.append(
                _check(
                    check_id="hf.jobs.namespace",
                    provider=provider,
                    category=PreflightCheckCategory.API,
                    label="HF Jobs namespace",
                    status=PreflightStatus.PASSED,
                    severity=PreflightSeverity.INFO,
                    message="HF Jobs namespace resolved without launching a job.",
                    details={"resolved_namespace": resolved_namespace},
                )
            )
        else:
            checks.append(
                _check(
                    check_id="hf.jobs.namespace",
                    provider=provider,
                    category=PreflightCheckCategory.API,
                    label="HF Jobs namespace",
                    status=PreflightStatus.UNKNOWN,
                    severity=PreflightSeverity.WARNING,
                    message="HF Jobs namespace could not be proven because namespace resolution failed.",
                    error_code="namespace_unavailable",
                    docs_verification_required=True,
                )
            )
        checks.append(
            _check(
                check_id="hf.jobs.hardware_availability",
                provider=provider,
                category=PreflightCheckCategory.HARDWARE,
                label="HF Jobs hardware availability",
                status=PreflightStatus.UNKNOWN,
                severity=PreflightSeverity.WARNING,
                message="HF Jobs hardware availability, billing, and credits were not checked because no safe read-only API is verified.",
                details={"hardware_id": hardware_id},
                error_code="quota_unavailable",
                docs_verification_required=True,
            )
        )
    else:
        checks.append(
            _check(
                check_id="hf.jobs.namespace",
                provider=provider,
                category=PreflightCheckCategory.API,
                label="HF Jobs namespace",
                status=PreflightStatus.SKIPPED,
                severity=PreflightSeverity.INFO,
                message="HF Jobs namespace check is not applicable for this provider.",
                required=False,
                applicable=False,
            )
        )
        checks.append(
            _check(
                check_id="hf.jobs.hardware_availability",
                provider=provider,
                category=PreflightCheckCategory.HARDWARE,
                label="HF Jobs hardware availability",
                status=PreflightStatus.SKIPPED,
                severity=PreflightSeverity.INFO,
                message="HF Jobs hardware availability is not applicable for this provider.",
                required=False,
                applicable=False,
            )
        )

    launch_ready, blocking, warnings, unknowns = derive_launch_ready(checks)
    if blocking:
        status = PreflightStatus.FAILED
    elif unknowns:
        status = PreflightStatus.UNKNOWN
    elif warnings:
        status = PreflightStatus.WARNING
    else:
        status = PreflightStatus.PASSED
    return TrainingPreflightProviderResult(
        provider=provider,
        status=status,
        launch_ready=launch_ready,
        checks=checks,
        blocking_reasons=[str(sanitize_for_frontend(reason)) for reason in blocking],
        warning_reasons=[str(sanitize_for_frontend(reason)) for reason in warnings],
        unknown_reasons=[str(sanitize_for_frontend(reason)) for reason in unknowns],
        metadata={
            "provider_jobs_launched": False,
            "resources_created": False,
            "mode": "hf_read_only_live",
        },
    )
