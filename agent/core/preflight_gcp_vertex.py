"""Read-only Google Vertex AI preflight probes.

These checks intentionally inspect credentials/configuration and safe metadata
only. They never create Vertex jobs, models, endpoints, buckets, or objects.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from agent.core.model_provider_selection import catalog_hardware
from agent.core.output_policy import output_policy_requires_cloud_storage
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

_REGION_RE = re.compile(r"^[a-z]+-[a-z0-9]+[0-9]$")
_OBJECT_CREATE_PERMISSION = "storage.objects.create"


class GcpReadOnlyClient(Protocol):
    def discover_credentials(self) -> tuple[Any, str | None]: ...

    def refresh_credentials(self, credentials: Any) -> None: ...

    def check_vertex_api(self, *, project_id: str, region: str) -> bool | None: ...

    def get_bucket(self, bucket_name: str) -> Any: ...


GcpClientFactory = Callable[[], GcpReadOnlyClient]


def _probe_vertex_custom_jobs(client: Any, *, parent: str) -> bool | None:
    """Verify Vertex metadata access without requiring newer paging kwargs."""
    try:
        iterator = client.list_custom_jobs(parent=parent, page_size=1)
    except TypeError as error:
        if "page_size" not in str(error):
            return None
        try:
            iterator = client.list_custom_jobs(parent=parent)
        except TypeError:
            return None
    next(iter(iterator), None)
    return True


@dataclass(frozen=True)
class _GoogleReadOnlyClient:
    def discover_credentials(self) -> tuple[Any, str | None]:
        import google.auth

        cloud_platform_scope = ("https://www.googleapis.com/auth/cloud-platform",)
        credentials, project_id = google.auth.default()
        if hasattr(credentials, "with_scopes_if_required"):
            credentials = credentials.with_scopes_if_required(cloud_platform_scope)
        elif getattr(credentials, "scopes", None) is None and hasattr(
            credentials, "with_scopes"
        ):
            credentials = credentials.with_scopes(cloud_platform_scope)
        return credentials, project_id

    def refresh_credentials(self, credentials: Any) -> None:
        from google.auth.transport.requests import Request

        credentials.refresh(Request())

    def check_vertex_api(self, *, project_id: str, region: str) -> bool | None:
        try:
            from google.api_core.client_options import ClientOptions
            from google.cloud import aiplatform_v1

            endpoint = f"{region}-aiplatform.googleapis.com"
            client = aiplatform_v1.JobServiceClient(
                client_options=ClientOptions(api_endpoint=endpoint)
            )
            # Read-only, cheap request. An empty iterator still proves the API
            # endpoint and credentials can be used for Vertex metadata access.
            return _probe_vertex_custom_jobs(
                client,
                parent=f"projects/{project_id}/locations/{region}",
            )
        except ImportError:
            return None

    def get_bucket(self, bucket_name: str) -> Any:
        from google.cloud import storage

        return storage.Client().bucket(bucket_name)


def default_gcp_client_factory() -> GcpReadOnlyClient:
    return _GoogleReadOnlyClient()


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
    status: PreflightStatus = PreflightStatus.FAILED,
    severity: PreflightSeverity = PreflightSeverity.BLOCKING,
) -> TrainingPreflightCheck:
    normalized = normalize_provider_error(error, provider=provider)
    return _check(
        check_id=check_id,
        provider=provider,
        category=category,
        label=label,
        status=status,
        severity=severity,
        message=str(normalized["message"]),
        error_code=str(normalized["error_code"]),
        docs_verification_required=True,
    )


async def _call_read_only(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(method, *args, **kwargs)


def _bucket_name(value: str | None) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    if raw.startswith("gs://"):
        raw = raw.removeprefix("gs://").strip("/")
    name, _, _prefix = raw.partition("/")
    return name or None


def _env_project_id() -> str | None:
    return (
        os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        or os.environ.get("GCP_PROJECT", "").strip()
        or os.environ.get("GCLOUD_PROJECT", "").strip()
        or None
    )


def _env_region() -> str | None:
    return (
        os.environ.get("GOOGLE_CLOUD_REGION", "").strip()
        or os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
        or os.environ.get("GCP_REGION", "").strip()
        or None
    )


def _env_bucket() -> str | None:
    return (
        os.environ.get("GCS_BUCKET", "").strip()
        or os.environ.get("VERTEX_AI_STAGING_BUCKET", "").strip()
        or None
    )


def _gcp_required(provider: str, output_policy: str) -> bool:
    return provider == "gcp-vertex" or (
        provider == "gcp-vertex" and output_policy_requires_cloud_storage(output_policy)
    )


def _cloud_storage_required(provider: str, output_policy: str) -> bool:
    return provider == "gcp-vertex" and output_policy_requires_cloud_storage(
        output_policy
    )


def _status_from_reasons(
    blocking: list[str], warnings: list[str], unknowns: list[str]
) -> PreflightStatus:
    if blocking:
        return PreflightStatus.FAILED
    if unknowns:
        return PreflightStatus.UNKNOWN
    if warnings:
        return PreflightStatus.WARNING
    return PreflightStatus.PASSED


async def run_gcp_vertex_preflight_checks(
    *,
    provider: str,
    model_id: str,
    hardware_id: str | None,
    output_policy: str,
    target_bucket: str | None = None,
    project_id: str | None = None,
    region: str | None = None,
    timeout_seconds: int | None = None,
    gcp_client_factory: GcpClientFactory | None = None,
) -> TrainingPreflightProviderResult:
    """Run safe read-only GCP/Vertex checks and return provider result."""

    _ = model_id, timeout_seconds
    checks: list[TrainingPreflightCheck] = []
    required = _gcp_required(provider, output_policy)
    storage_required = _cloud_storage_required(provider, output_policy)
    client = (gcp_client_factory or default_gcp_client_factory)()

    credentials: Any | None = None
    discovered_project: str | None = None
    try:
        credentials, discovered_project = await _call_read_only(
            client.discover_credentials
        )
        checks.append(
            _check(
                check_id="gcp.credentials.present",
                provider=provider,
                category=PreflightCheckCategory.CREDENTIALS,
                label="GCP credentials",
                status=PreflightStatus.PASSED,
                severity=PreflightSeverity.INFO,
                message="Google ADC credentials are discoverable.",
                details={"credentials_detected": True},
            )
        )
    except Exception as error:
        checks.append(
            _error_check(
                check_id="gcp.credentials.present",
                provider=provider,
                category=PreflightCheckCategory.CREDENTIALS,
                label="GCP credentials",
                error=error,
                status=PreflightStatus.FAILED if required else PreflightStatus.UNKNOWN,
                severity=PreflightSeverity.BLOCKING
                if required
                else PreflightSeverity.WARNING,
            )
        )

    if credentials is not None:
        try:
            await _call_read_only(client.refresh_credentials, credentials)
            checks.append(
                _check(
                    check_id="gcp.credentials.refresh",
                    provider=provider,
                    category=PreflightCheckCategory.IDENTITY,
                    label="GCP credential refresh",
                    status=PreflightStatus.PASSED,
                    severity=PreflightSeverity.INFO,
                    message="Google credentials refreshed successfully.",
                )
            )
        except Exception as error:
            checks.append(
                _error_check(
                    check_id="gcp.credentials.refresh",
                    provider=provider,
                    category=PreflightCheckCategory.IDENTITY,
                    label="GCP credential refresh",
                    error=error,
                )
            )
    else:
        checks.append(
            _check(
                check_id="gcp.credentials.refresh",
                provider=provider,
                category=PreflightCheckCategory.IDENTITY,
                label="GCP credential refresh",
                status=PreflightStatus.SKIPPED,
                severity=PreflightSeverity.INFO,
                message="Credential refresh skipped because credentials were not available.",
                required=False,
                applicable=False,
            )
        )

    resolved_project = project_id or discovered_project or _env_project_id()
    if resolved_project:
        checks.append(
            _check(
                check_id="gcp.project.configured",
                provider=provider,
                category=PreflightCheckCategory.API,
                label="GCP project",
                status=PreflightStatus.PASSED,
                severity=PreflightSeverity.INFO,
                message="Google Cloud project is configured.",
                details={"project_id": resolved_project},
            )
        )
    else:
        checks.append(
            _check(
                check_id="gcp.project.configured",
                provider=provider,
                category=PreflightCheckCategory.API,
                label="GCP project",
                status=PreflightStatus.FAILED if required else PreflightStatus.UNKNOWN,
                severity=PreflightSeverity.BLOCKING
                if required
                else PreflightSeverity.WARNING,
                message="Google Cloud project is required for Vertex preflight.",
                error_code="missing_project_id",
            )
        )

    resolved_region = region or _env_region()
    if resolved_region and _REGION_RE.match(resolved_region):
        checks.append(
            _check(
                check_id="gcp.region.configured",
                provider=provider,
                category=PreflightCheckCategory.API,
                label="GCP region",
                status=PreflightStatus.PASSED,
                severity=PreflightSeverity.INFO,
                message="Google Cloud region is configured and locally plausible.",
                details={"region": resolved_region},
            )
        )
    else:
        checks.append(
            _check(
                check_id="gcp.region.configured",
                provider=provider,
                category=PreflightCheckCategory.API,
                label="GCP region",
                status=PreflightStatus.FAILED if required else PreflightStatus.UNKNOWN,
                severity=PreflightSeverity.BLOCKING
                if required
                else PreflightSeverity.WARNING,
                message="Google Cloud region/location is required for Vertex preflight.",
                error_code="missing_or_invalid_region",
            )
        )

    if resolved_project and resolved_region:
        try:
            enabled = await _call_read_only(
                client.check_vertex_api,
                project_id=resolved_project,
                region=resolved_region,
            )
            if enabled is True:
                checks.append(
                    _check(
                        check_id="gcp.vertex.api",
                        provider=provider,
                        category=PreflightCheckCategory.API,
                        label="Vertex API",
                        status=PreflightStatus.PASSED,
                        severity=PreflightSeverity.INFO,
                        message="Vertex AI API appears reachable via a read-only metadata request.",
                    )
                )
            elif enabled is False:
                checks.append(
                    _check(
                        check_id="gcp.vertex.api",
                        provider=provider,
                        category=PreflightCheckCategory.API,
                        label="Vertex API",
                        status=PreflightStatus.FAILED,
                        severity=PreflightSeverity.BLOCKING,
                        message="Vertex AI API appears disabled or unreachable.",
                        error_code="vertex_api_unavailable",
                        docs_verification_required=True,
                    )
                )
            else:
                checks.append(
                    _check(
                        check_id="gcp.vertex.api",
                        provider=provider,
                        category=PreflightCheckCategory.API,
                        label="Vertex API",
                        status=PreflightStatus.UNKNOWN,
                        severity=PreflightSeverity.INFO,
                        message=(
                            "Vertex AI API could not be verified without the SDK — "
                            "this is expected and does NOT block job submission. "
                            "Proceed to the approval gate."
                        ),
                        error_code="sdk_missing",
                        docs_verification_required=True,
                    )
                )
        except Exception as error:
            checks.append(
                _error_check(
                    check_id="gcp.vertex.api",
                    provider=provider,
                    category=PreflightCheckCategory.API,
                    label="Vertex API",
                    error=error,
                )
            )
    else:
        checks.append(
            _check(
                check_id="gcp.vertex.api",
                provider=provider,
                category=PreflightCheckCategory.API,
                label="Vertex API",
                status=PreflightStatus.SKIPPED,
                severity=PreflightSeverity.INFO,
                message="Vertex API check skipped until project and region are configured.",
                required=False,
                applicable=False,
            )
        )

    bucket_name = _bucket_name(target_bucket or _env_bucket())
    bucket: Any | None = None
    if storage_required:
        if not bucket_name:
            checks.append(
                _check(
                    check_id="gcp.gcs.bucket_read",
                    provider=provider,
                    category=PreflightCheckCategory.STORAGE,
                    label="GCS bucket read",
                    status=PreflightStatus.FAILED,
                    severity=PreflightSeverity.BLOCKING,
                    message="A GCS bucket is required for this Vertex output policy.",
                    error_code="missing_bucket",
                )
            )
        else:
            try:
                bucket = await _call_read_only(client.get_bucket, bucket_name)
                exists = await _call_read_only(bucket.exists)
                checks.append(
                    _check(
                        check_id="gcp.gcs.bucket_read",
                        provider=provider,
                        category=PreflightCheckCategory.STORAGE,
                        label="GCS bucket read",
                        status=PreflightStatus.PASSED
                        if exists
                        else PreflightStatus.FAILED,
                        severity=PreflightSeverity.INFO
                        if exists
                        else PreflightSeverity.BLOCKING,
                        message=(
                            "GCS bucket exists and is readable via metadata."
                            if exists
                            else "GCS bucket was not found."
                        ),
                        error_code=None if exists else "not_found",
                        details={"bucket": bucket_name},
                    )
                )
            except Exception as error:
                checks.append(
                    _error_check(
                        check_id="gcp.gcs.bucket_read",
                        provider=provider,
                        category=PreflightCheckCategory.STORAGE,
                        label="GCS bucket read",
                        error=error,
                    )
                )
    else:
        checks.append(
            _check(
                check_id="gcp.gcs.bucket_read",
                provider=provider,
                category=PreflightCheckCategory.STORAGE,
                label="GCS bucket read",
                status=PreflightStatus.SKIPPED,
                severity=PreflightSeverity.INFO,
                message="GCS bucket check is not required for this output policy.",
                required=False,
                applicable=False,
            )
        )

    if storage_required and bucket is not None:
        try:
            permissions = await _call_read_only(
                bucket.test_iam_permissions,
                [_OBJECT_CREATE_PERMISSION],
            )
            can_write = _OBJECT_CREATE_PERMISSION in set(permissions or [])
            checks.append(
                _check(
                    check_id="gcp.gcs.write_permission",
                    provider=provider,
                    category=PreflightCheckCategory.STORAGE,
                    label="GCS write permission",
                    status=PreflightStatus.PASSED
                    if can_write
                    else PreflightStatus.UNKNOWN,
                    severity=PreflightSeverity.INFO
                    if can_write
                    else PreflightSeverity.WARNING,
                    message=(
                        "GCS object create permission is confirmed by read-only IAM test."
                        if can_write
                        else "GCS object create permission could not be proven without writing an object."
                    ),
                    error_code=None if can_write else "permission_denied",
                    docs_verification_required=not can_write,
                )
            )
        except Exception as error:
            normalized = normalize_provider_error(error, provider=provider)
            checks.append(
                _check(
                    check_id="gcp.gcs.write_permission",
                    provider=provider,
                    category=PreflightCheckCategory.STORAGE,
                    label="GCS write permission",
                    status=PreflightStatus.UNKNOWN,
                    severity=PreflightSeverity.WARNING,
                    message=str(normalized["message"]),
                    error_code=str(normalized["error_code"]),
                    docs_verification_required=True,
                )
            )
    elif storage_required:
        checks.append(
            _check(
                check_id="gcp.gcs.write_permission",
                provider=provider,
                category=PreflightCheckCategory.STORAGE,
                label="GCS write permission",
                status=PreflightStatus.SKIPPED,
                severity=PreflightSeverity.INFO,
                message="GCS write permission check skipped because bucket metadata was unavailable.",
                required=False,
                applicable=False,
            )
        )
    else:
        checks.append(
            _check(
                check_id="gcp.gcs.write_permission",
                provider=provider,
                category=PreflightCheckCategory.STORAGE,
                label="GCS write permission",
                status=PreflightStatus.SKIPPED,
                severity=PreflightSeverity.INFO,
                message="GCS write permission is not required for this output policy.",
                required=False,
                applicable=False,
            )
        )

    hardware = catalog_hardware(hardware_id) if hardware_id else None
    if hardware and hardware.provider_id == "gcp-vertex":
        checks.append(
            _check(
                check_id="gcp.vertex.hardware_catalog",
                provider=provider,
                category=PreflightCheckCategory.HARDWARE,
                label="Vertex hardware catalog",
                status=PreflightStatus.PASSED,
                severity=PreflightSeverity.INFO,
                message="Selected hardware is recognized as Vertex-compatible in the static catalog.",
                details={"hardware_id": hardware_id, **hardware.hardware_args},
            )
        )
    else:
        checks.append(
            _check(
                check_id="gcp.vertex.hardware_catalog",
                provider=provider,
                category=PreflightCheckCategory.HARDWARE,
                label="Vertex hardware catalog",
                status=PreflightStatus.FAILED,
                severity=PreflightSeverity.BLOCKING,
                message="Selected hardware is missing or not Vertex-compatible.",
                error_code="unsupported",
                details={"hardware_id": hardware_id},
            )
        )

    checks.append(
        _check(
            check_id="gcp.vertex.quota_availability",
            provider=provider,
            category=PreflightCheckCategory.QUOTA,
            label="Vertex quota and accelerator availability",
            status=PreflightStatus.UNKNOWN,
            severity=PreflightSeverity.INFO,
            message="Quota check skipped — no safe read-only quota API is available. This is expected behavior and does NOT block job submission. Proceed to the approval gate.",
            details={"hardware_id": hardware_id},
            error_code="quota_unavailable",
            docs_verification_required=True,
        )
    )

    launch_ready, blocking, warnings, unknowns = derive_launch_ready(checks)
    return TrainingPreflightProviderResult(
        provider=provider,
        status=_status_from_reasons(blocking, warnings, unknowns),
        launch_ready=launch_ready,
        checks=checks,
        blocking_reasons=[str(sanitize_for_frontend(reason)) for reason in blocking],
        warning_reasons=[str(sanitize_for_frontend(reason)) for reason in warnings],
        unknown_reasons=[str(sanitize_for_frontend(reason)) for reason in unknowns],
        metadata={
            "provider_jobs_launched": False,
            "resources_created": False,
            "mode": "gcp_vertex_read_only_live",
        },
    )
