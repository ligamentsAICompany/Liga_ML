"""Read-only AWS SageMaker preflight probes.

These checks inspect credentials, identity, SageMaker/S3 metadata, and local
catalog compatibility only. They never create jobs, models, endpoints, buckets,
objects, policies, roles, or other AWS resources.
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

_REGION_RE = re.compile(r"^[a-z]{2}-[a-z]+-[0-9]$")
_ROLE_ARN_RE = re.compile(
    r"^arn:aws(-[a-z]+)?:iam::\d{12}:role\/[A-Za-z0-9+=,.@_\/-]+$"
)


class AwsReadOnlyClient(Protocol):
    def discover_credentials(self) -> Any | None: ...

    def get_caller_identity(self) -> dict[str, Any]: ...

    def list_training_jobs(self, *, max_results: int) -> dict[str, Any]: ...

    def head_bucket(self, *, bucket: str) -> dict[str, Any]: ...

    def get_bucket_location(self, *, bucket: str) -> dict[str, Any]: ...

    def simulate_s3_write_permission(
        self, *, bucket: str, role_arn: str | None
    ) -> bool | None: ...

    def get_role(self, *, role_name: str) -> dict[str, Any]: ...

    def get_service_quota(
        self, *, instance_type: str | None
    ) -> dict[str, Any] | None: ...


AwsClientFactory = Callable[[str | None], AwsReadOnlyClient]


@dataclass(frozen=True)
class _Boto3ReadOnlyClient:
    region: str | None

    def __post_init__(self) -> None:
        import boto3

        object.__setattr__(self, "_session", boto3.Session(region_name=self.region))

    def discover_credentials(self) -> Any | None:
        return self._session.get_credentials()

    def get_caller_identity(self) -> dict[str, Any]:
        return self._session.client("sts").get_caller_identity()

    def list_training_jobs(self, *, max_results: int) -> dict[str, Any]:
        return self._session.client("sagemaker").list_training_jobs(
            MaxResults=max_results
        )

    def head_bucket(self, *, bucket: str) -> dict[str, Any]:
        return self._session.client("s3").head_bucket(Bucket=bucket)

    def get_bucket_location(self, *, bucket: str) -> dict[str, Any]:
        return self._session.client("s3").get_bucket_location(Bucket=bucket)

    def simulate_s3_write_permission(
        self, *, bucket: str, role_arn: str | None
    ) -> bool | None:
        if not role_arn:
            return None
        response = self._session.client("iam").simulate_principal_policy(
            PolicySourceArn=role_arn,
            ActionNames=["s3:PutObject"],
            ResourceArns=[f"arn:aws:s3:::{bucket}/*"],
        )
        results = response.get("EvaluationResults") or []
        if not results:
            return None
        decision = str(results[0].get("EvalDecision") or "").lower()
        if decision == "allowed":
            return True
        if decision in {"explicitdeny", "implicitdeny"}:
            return False
        return None

    def get_role(self, *, role_name: str) -> dict[str, Any]:
        return self._session.client("iam").get_role(RoleName=role_name)

    def get_service_quota(self, *, instance_type: str | None) -> dict[str, Any] | None:
        _ = instance_type
        return None


def default_aws_client_factory(region: str | None) -> AwsReadOnlyClient:
    return _Boto3ReadOnlyClient(region)


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


def _env_region() -> str | None:
    return (
        os.environ.get("AWS_REGION", "").strip()
        or os.environ.get("AWS_DEFAULT_REGION", "").strip()
        or None
    )


def _env_bucket() -> str | None:
    return os.environ.get("AWS_S3_BUCKET", "").strip() or None


def _env_role() -> str | None:
    return os.environ.get("AWS_SAGEMAKER_ROLE_ARN", "").strip() or None


def _bucket_name(value: str | None) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    if raw.startswith("s3://"):
        raw = raw.removeprefix("s3://")
    name, _, _prefix = raw.strip("/").partition("/")
    return name or None


def _role_name(role_arn: str) -> str | None:
    if not role_arn or not _ROLE_ARN_RE.match(role_arn):
        return None
    _prefix, _, name = role_arn.partition(":role/")
    return name or None


def _instance_type(hardware_id: str | None) -> str | None:
    hardware = catalog_hardware(hardware_id) if hardware_id else None
    if not hardware:
        return None
    value = hardware.hardware_args.get("instance_type")
    return str(value) if value else None


def _aws_required(provider: str, output_policy: str) -> bool:
    return provider == "aws-sagemaker" and (
        output_policy_requires_cloud_storage(output_policy) or bool(output_policy)
    )


def _cloud_storage_required(provider: str, output_policy: str) -> bool:
    return provider == "aws-sagemaker" and output_policy_requires_cloud_storage(
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


async def run_aws_sagemaker_preflight_checks(
    *,
    provider: str,
    model_id: str,
    hardware_id: str | None,
    output_policy: str,
    target_bucket: str | None = None,
    aws_region: str | None = None,
    execution_role_arn: str | None = None,
    timeout_seconds: int | None = None,
    aws_client_factory: AwsClientFactory | None = None,
) -> TrainingPreflightProviderResult:
    """Run safe read-only AWS/SageMaker checks and return provider result."""

    _ = model_id, timeout_seconds
    checks: list[TrainingPreflightCheck] = []
    required = _aws_required(provider, output_policy)
    storage_required = _cloud_storage_required(provider, output_policy)
    resolved_region = aws_region or _env_region()
    client = (aws_client_factory or default_aws_client_factory)(resolved_region)

    credentials: Any | None = None
    try:
        credentials = await _call_read_only(client.discover_credentials)
        if credentials is None:
            raise RuntimeError("AWS credentials were not detected.")
        checks.append(
            _check(
                check_id="aws.credentials.present",
                provider=provider,
                category=PreflightCheckCategory.CREDENTIALS,
                label="AWS credentials",
                status=PreflightStatus.PASSED,
                severity=PreflightSeverity.INFO,
                message="AWS credentials are discoverable through the default provider chain.",
                details={"credentials_detected": True},
            )
        )
    except Exception as error:
        checks.append(
            _error_check(
                check_id="aws.credentials.present",
                provider=provider,
                category=PreflightCheckCategory.CREDENTIALS,
                label="AWS credentials",
                error=error,
                status=PreflightStatus.FAILED if required else PreflightStatus.UNKNOWN,
                severity=PreflightSeverity.BLOCKING
                if required
                else PreflightSeverity.WARNING,
            )
        )

    if credentials is not None:
        try:
            identity = await _call_read_only(client.get_caller_identity)
            arn = str(identity.get("Arn") or "")
            checks.append(
                _check(
                    check_id="aws.identity.sts",
                    provider=provider,
                    category=PreflightCheckCategory.IDENTITY,
                    label="AWS STS identity",
                    status=PreflightStatus.PASSED,
                    severity=PreflightSeverity.INFO,
                    message="AWS identity resolved through STS get_caller_identity.",
                    details={
                        "identity_resolved": True,
                        "account_present": bool(identity.get("Account")),
                        "arn_type": "role"
                        if ":assumed-role/" in arn or ":role/" in arn
                        else "principal"
                        if arn
                        else None,
                    },
                )
            )
        except Exception as error:
            checks.append(
                _error_check(
                    check_id="aws.identity.sts",
                    provider=provider,
                    category=PreflightCheckCategory.IDENTITY,
                    label="AWS STS identity",
                    error=error,
                )
            )
    else:
        checks.append(
            _check(
                check_id="aws.identity.sts",
                provider=provider,
                category=PreflightCheckCategory.IDENTITY,
                label="AWS STS identity",
                status=PreflightStatus.SKIPPED,
                severity=PreflightSeverity.INFO,
                message="STS identity check skipped because AWS credentials were not available.",
                required=False,
                applicable=False,
            )
        )

    if resolved_region and _REGION_RE.match(resolved_region):
        checks.append(
            _check(
                check_id="aws.region.configured",
                provider=provider,
                category=PreflightCheckCategory.API,
                label="AWS region",
                status=PreflightStatus.PASSED,
                severity=PreflightSeverity.INFO,
                message="AWS region is configured and locally plausible.",
                details={"region": resolved_region},
            )
        )
    else:
        checks.append(
            _check(
                check_id="aws.region.configured",
                provider=provider,
                category=PreflightCheckCategory.API,
                label="AWS region",
                status=PreflightStatus.FAILED if required else PreflightStatus.UNKNOWN,
                severity=PreflightSeverity.BLOCKING
                if required
                else PreflightSeverity.WARNING,
                message="AWS region is required for SageMaker preflight.",
                error_code="missing_or_invalid_region",
            )
        )

    if credentials is not None and resolved_region:
        try:
            await _call_read_only(client.list_training_jobs, max_results=1)
            checks.append(
                _check(
                    check_id="aws.sagemaker.api",
                    provider=provider,
                    category=PreflightCheckCategory.API,
                    label="SageMaker API",
                    status=PreflightStatus.PASSED,
                    severity=PreflightSeverity.INFO,
                    message="SageMaker API appears reachable via a read-only list request.",
                )
            )
        except Exception as error:
            checks.append(
                _error_check(
                    check_id="aws.sagemaker.api",
                    provider=provider,
                    category=PreflightCheckCategory.API,
                    label="SageMaker API",
                    error=error,
                )
            )
    else:
        checks.append(
            _check(
                check_id="aws.sagemaker.api",
                provider=provider,
                category=PreflightCheckCategory.API,
                label="SageMaker API",
                status=PreflightStatus.SKIPPED,
                severity=PreflightSeverity.INFO,
                message="SageMaker API check skipped until credentials and region are available.",
                required=False,
                applicable=False,
            )
        )

    bucket_name = _bucket_name(target_bucket or _env_bucket())
    bucket_readable = False
    if storage_required:
        if not bucket_name:
            checks.append(
                _check(
                    check_id="aws.s3.bucket_read",
                    provider=provider,
                    category=PreflightCheckCategory.STORAGE,
                    label="S3 bucket read",
                    status=PreflightStatus.FAILED,
                    severity=PreflightSeverity.BLOCKING,
                    message="An S3 bucket is required for this SageMaker output policy.",
                    error_code="missing_bucket",
                )
            )
        elif credentials is None:
            checks.append(
                _check(
                    check_id="aws.s3.bucket_read",
                    provider=provider,
                    category=PreflightCheckCategory.STORAGE,
                    label="S3 bucket read",
                    status=PreflightStatus.SKIPPED,
                    severity=PreflightSeverity.INFO,
                    message="S3 bucket check skipped because AWS credentials were not available.",
                    required=False,
                    applicable=False,
                )
            )
        else:
            try:
                await _call_read_only(client.head_bucket, bucket=bucket_name)
                await _call_read_only(client.get_bucket_location, bucket=bucket_name)
                bucket_readable = True
                checks.append(
                    _check(
                        check_id="aws.s3.bucket_read",
                        provider=provider,
                        category=PreflightCheckCategory.STORAGE,
                        label="S3 bucket read",
                        status=PreflightStatus.PASSED,
                        severity=PreflightSeverity.INFO,
                        message="S3 bucket exists and is readable via metadata.",
                        details={"bucket": bucket_name},
                    )
                )
            except Exception as error:
                checks.append(
                    _error_check(
                        check_id="aws.s3.bucket_read",
                        provider=provider,
                        category=PreflightCheckCategory.STORAGE,
                        label="S3 bucket read",
                        error=error,
                    )
                )
    else:
        checks.append(
            _check(
                check_id="aws.s3.bucket_read",
                provider=provider,
                category=PreflightCheckCategory.STORAGE,
                label="S3 bucket read",
                status=PreflightStatus.SKIPPED,
                severity=PreflightSeverity.INFO,
                message="S3 bucket check is not required for this output policy.",
                required=False,
                applicable=False,
            )
        )

    role_arn = execution_role_arn or _env_role()
    role_name = _role_name(role_arn or "")
    if provider == "aws-sagemaker":
        if not role_arn or not role_name:
            checks.append(
                _check(
                    check_id="aws.sagemaker.execution_role",
                    provider=provider,
                    category=PreflightCheckCategory.IDENTITY,
                    label="SageMaker execution role",
                    status=PreflightStatus.FAILED,
                    severity=PreflightSeverity.BLOCKING,
                    message="A valid SageMaker execution role ARN is required.",
                    error_code="missing_or_invalid_execution_role",
                )
            )
        elif credentials is None:
            checks.append(
                _check(
                    check_id="aws.sagemaker.execution_role",
                    provider=provider,
                    category=PreflightCheckCategory.IDENTITY,
                    label="SageMaker execution role",
                    status=PreflightStatus.UNKNOWN,
                    severity=PreflightSeverity.WARNING,
                    message="Execution role is configured, but IAM read access was not checked because credentials were unavailable.",
                    docs_verification_required=True,
                )
            )
        else:
            try:
                await _call_read_only(client.get_role, role_name=role_name)
                checks.append(
                    _check(
                        check_id="aws.sagemaker.execution_role",
                        provider=provider,
                        category=PreflightCheckCategory.IDENTITY,
                        label="SageMaker execution role",
                        status=PreflightStatus.PASSED,
                        severity=PreflightSeverity.INFO,
                        message="SageMaker execution role exists and is readable via IAM get_role.",
                        details={"role_configured": True, "role_name": role_name},
                    )
                )
            except Exception as error:
                normalized = normalize_provider_error(error, provider=provider)
                checks.append(
                    _check(
                        check_id="aws.sagemaker.execution_role",
                        provider=provider,
                        category=PreflightCheckCategory.IDENTITY,
                        label="SageMaker execution role",
                        status=PreflightStatus.UNKNOWN,
                        severity=PreflightSeverity.WARNING,
                        message=str(normalized["message"]),
                        error_code=str(normalized["error_code"]),
                        details={"role_configured": True},
                        docs_verification_required=True,
                    )
                )
    else:
        checks.append(
            _check(
                check_id="aws.sagemaker.execution_role",
                provider=provider,
                category=PreflightCheckCategory.IDENTITY,
                label="SageMaker execution role",
                status=PreflightStatus.SKIPPED,
                severity=PreflightSeverity.INFO,
                message="SageMaker execution role is not required for this provider.",
                required=False,
                applicable=False,
            )
        )

    if storage_required and bucket_readable:
        try:
            can_write = await _call_read_only(
                client.simulate_s3_write_permission,
                bucket=bucket_name or "",
                role_arn=role_arn,
            )
            checks.append(
                _check(
                    check_id="aws.s3.write_permission",
                    provider=provider,
                    category=PreflightCheckCategory.STORAGE,
                    label="S3 write permission",
                    status=PreflightStatus.PASSED
                    if can_write is True
                    else PreflightStatus.UNKNOWN,
                    severity=PreflightSeverity.INFO
                    if can_write is True
                    else PreflightSeverity.WARNING,
                    message=(
                        "S3 object write permission is confirmed by read-only IAM simulation."
                        if can_write is True
                        else "S3 object write permission could not be proven without creating an object."
                    ),
                    error_code=None if can_write is True else "permission_unverified",
                    docs_verification_required=can_write is not True,
                )
            )
        except Exception as error:
            normalized = normalize_provider_error(error, provider=provider)
            checks.append(
                _check(
                    check_id="aws.s3.write_permission",
                    provider=provider,
                    category=PreflightCheckCategory.STORAGE,
                    label="S3 write permission",
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
                check_id="aws.s3.write_permission",
                provider=provider,
                category=PreflightCheckCategory.STORAGE,
                label="S3 write permission",
                status=PreflightStatus.SKIPPED,
                severity=PreflightSeverity.INFO,
                message="S3 write permission check skipped because bucket metadata was unavailable.",
                required=False,
                applicable=False,
            )
        )
    else:
        checks.append(
            _check(
                check_id="aws.s3.write_permission",
                provider=provider,
                category=PreflightCheckCategory.STORAGE,
                label="S3 write permission",
                status=PreflightStatus.SKIPPED,
                severity=PreflightSeverity.INFO,
                message="S3 write permission is not required for this output policy.",
                required=False,
                applicable=False,
            )
        )

    hardware = catalog_hardware(hardware_id) if hardware_id else None
    if hardware and hardware.provider_id == "aws-sagemaker":
        checks.append(
            _check(
                check_id="aws.sagemaker.hardware_catalog",
                provider=provider,
                category=PreflightCheckCategory.HARDWARE,
                label="SageMaker hardware catalog",
                status=PreflightStatus.PASSED,
                severity=PreflightSeverity.INFO,
                message="Selected hardware is recognized as SageMaker-compatible in the static catalog.",
                details={"hardware_id": hardware_id, **hardware.hardware_args},
            )
        )
    else:
        checks.append(
            _check(
                check_id="aws.sagemaker.hardware_catalog",
                provider=provider,
                category=PreflightCheckCategory.HARDWARE,
                label="SageMaker hardware catalog",
                status=PreflightStatus.FAILED,
                severity=PreflightSeverity.BLOCKING,
                message="Selected hardware is missing or not SageMaker-compatible.",
                error_code="unsupported",
                details={"hardware_id": hardware_id},
            )
        )

    instance_type = _instance_type(hardware_id)
    try:
        quota = await _call_read_only(
            client.get_service_quota, instance_type=instance_type
        )
    except Exception as error:
        normalized = normalize_provider_error(error, provider=provider)
        checks.append(
            _check(
                check_id="aws.sagemaker.quota_availability",
                provider=provider,
                category=PreflightCheckCategory.QUOTA,
                label="SageMaker quota and instance availability",
                status=PreflightStatus.UNKNOWN,
                severity=PreflightSeverity.WARNING,
                message=str(normalized["message"]),
                error_code=str(normalized["error_code"]),
                docs_verification_required=True,
            )
        )
    else:
        if isinstance(quota, dict) and quota.get("available") is True:
            checks.append(
                _check(
                    check_id="aws.sagemaker.quota_availability",
                    provider=provider,
                    category=PreflightCheckCategory.QUOTA,
                    label="SageMaker quota and instance availability",
                    status=PreflightStatus.PASSED,
                    severity=PreflightSeverity.INFO,
                    message="SageMaker quota was confirmed by a read-only Service Quotas check.",
                    details={"instance_type": instance_type},
                )
            )
        else:
            checks.append(
                _check(
                    check_id="aws.sagemaker.quota_availability",
                    provider=provider,
                    category=PreflightCheckCategory.QUOTA,
                    label="SageMaker quota and instance availability",
                    status=PreflightStatus.UNKNOWN,
                    severity=PreflightSeverity.WARNING,
                    message="SageMaker instance quota was not checked because no safe quota mapping is verified.",
                    details={"instance_type": instance_type},
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
            "mode": "aws_sagemaker_read_only_live",
        },
    )
