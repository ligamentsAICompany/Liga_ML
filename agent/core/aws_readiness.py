"""AWS SageMaker readiness helpers.

Checks are intentionally local and fast. They validate environment defaults and
credential discovery without STS, SageMaker, S3, or any billable AWS API call.
"""

from __future__ import annotations

import os
from typing import Any

REQUIRED_AWS_SAGEMAKER_ENV = [
    "AWS_REGION",
    "AWS_S3_BUCKET",
    "AWS_SAGEMAKER_ROLE_ARN",
]

DEFAULT_AWS_REGION = "us-east-1"
DEFAULT_AWS_S3_PREFIX = "liga-ml"
DEFAULT_AWS_INSTANCE_TYPE = "ml.g5.xlarge"
DEFAULT_AWS_INSTANCE_COUNT = 1
DEFAULT_AWS_MAX_RUN_SECONDS = 3600
DEFAULT_AWS_OUTPUT_POLICY = "aws-private"
VALID_AWS_OUTPUT_POLICIES = {
    "aws-private",
    "cloud-private",
    "hf-hub",
    "cloud-and-hf-hub",
}


def _detect_aws_credentials() -> tuple[bool, list[str]]:
    try:
        import boto3
    except Exception as exc:
        return False, [f"boto3 is not available for AWS credential detection: {exc}"]

    try:
        credentials = boto3.Session().get_credentials()
    except Exception as exc:
        return False, [f"AWS credentials were not detected by boto3: {exc}"]
    return credentials is not None, [] if credentials is not None else [
        "AWS credentials were not detected by boto3's default provider chain."
    ]


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, "")
    return value.strip() or default


def _positive_int_env(name: str, default: int) -> tuple[int, list[str]]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default, []
    try:
        parsed = int(raw)
    except ValueError:
        return default, [f"{name} must be a positive integer."]
    if parsed <= 0:
        return default, [f"{name} must be a positive integer."]
    return parsed, []


def build_aws_sagemaker_readiness_snapshot() -> dict[str, Any]:
    region = _env_str("AWS_REGION", DEFAULT_AWS_REGION)
    s3_bucket = _env_str("AWS_S3_BUCKET")
    s3_prefix = _env_str("AWS_S3_PREFIX", DEFAULT_AWS_S3_PREFIX)
    role_arn = _env_str("AWS_SAGEMAKER_ROLE_ARN")
    instance_type = _env_str("AWS_DEFAULT_INSTANCE_TYPE", DEFAULT_AWS_INSTANCE_TYPE)
    output_policy = _env_str("AWS_OUTPUT_POLICY", DEFAULT_AWS_OUTPUT_POLICY)
    training_image_uri = _env_str("AWS_SAGEMAKER_TRAINING_IMAGE_URI")

    instance_count, count_errors = _positive_int_env(
        "AWS_DEFAULT_INSTANCE_COUNT", DEFAULT_AWS_INSTANCE_COUNT
    )
    max_run_seconds, runtime_errors = _positive_int_env(
        "AWS_DEFAULT_MAX_RUN_SECONDS", DEFAULT_AWS_MAX_RUN_SECONDS
    )

    required_values = {
        "AWS_REGION": region if os.environ.get("AWS_REGION", "").strip() else None,
        "AWS_S3_BUCKET": s3_bucket,
        "AWS_SAGEMAKER_ROLE_ARN": role_arn,
    }
    missing_env = [name for name, value in required_values.items() if not value]

    credentials_detected, credential_warnings = _detect_aws_credentials()
    warnings = list(credential_warnings)
    errors: list[str] = []
    if missing_env:
        errors.append("Missing required AWS environment variables.")
    errors.extend(count_errors)
    errors.extend(runtime_errors)
    if output_policy not in VALID_AWS_OUTPUT_POLICIES:
        errors.append(
            "AWS_OUTPUT_POLICY must be one of: "
            + ", ".join(sorted(VALID_AWS_OUTPUT_POLICIES))
            + "."
        )
        output_policy = DEFAULT_AWS_OUTPUT_POLICY

    configured = not missing_env and credentials_detected and not errors
    return {
        "configured": configured,
        "missing_env": missing_env,
        "region": region or DEFAULT_AWS_REGION,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix or DEFAULT_AWS_S3_PREFIX,
        "sagemaker_role_arn": role_arn,
        "default_instance_type": instance_type or DEFAULT_AWS_INSTANCE_TYPE,
        "default_instance_count": instance_count,
        "default_max_run_seconds": max_run_seconds,
        "output_policy": output_policy or DEFAULT_AWS_OUTPUT_POLICY,
        "training_image_uri": training_image_uri,
        "credentials_detected": credentials_detected,
        "warnings": warnings,
        "errors": errors,
    }
