"""Google Cloud / Vertex AI readiness helpers.

These checks are intentionally local and fast. They validate configuration and
credential discovery without calling billable or slow Google Cloud APIs.
"""

from __future__ import annotations

import os
from typing import Any


REQUIRED_GCP_VERTEX_ENV = [
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_REGION",
    "GCS_BUCKET",
]


def _gs_path(path_or_bucket: str | None, suffix: str | None = None) -> str | None:
    if not path_or_bucket:
        return None
    base = (
        path_or_bucket.strip()
        if path_or_bucket.strip().startswith("gs://")
        else f"gs://{path_or_bucket.strip()}"
    ).rstrip("/")
    if suffix:
        return f"{base}/{suffix.strip('/')}"
    return base


def _credential_service_account(credentials: Any) -> str | None:
    for attr in ("service_account_email", "signer_email"):
        value = getattr(credentials, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def _detect_adc() -> tuple[bool, str | None, list[str]]:
    try:
        import google.auth

        credentials, _project = google.auth.default()
        return True, _credential_service_account(credentials), []
    except Exception as exc:
        return False, None, [f"Google ADC credentials were not detected: {exc}"]


def build_gcp_vertex_readiness_snapshot() -> dict[str, Any]:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip() or None
    region = os.environ.get("GOOGLE_CLOUD_REGION", "").strip() or None
    bucket = os.environ.get("GCS_BUCKET", "").strip() or None
    missing_env = [
        name
        for name, value in {
            "GOOGLE_CLOUD_PROJECT": project,
            "GOOGLE_CLOUD_REGION": region,
            "GCS_BUCKET": bucket,
        }.items()
        if not value
    ]

    staging_bucket = os.environ.get("VERTEX_AI_STAGING_BUCKET", "").strip() or _gs_path(
        bucket, "vertex-staging"
    )
    output_dir = os.environ.get("VERTEX_AI_OUTPUT_DIR", "").strip() or _gs_path(
        bucket, "vertex-outputs"
    )

    credentials_detected, detected_service_account, credential_warnings = _detect_adc()
    service_account = (
        os.environ.get("VERTEX_AI_SERVICE_ACCOUNT", "").strip()
        or detected_service_account
        or None
    )
    warnings = list(credential_warnings)
    errors: list[str] = []
    if missing_env:
        errors.append("Missing required Google Cloud environment variables.")

    return {
        "configured": not missing_env and credentials_detected,
        "missing_env": missing_env,
        "project": project,
        "region": region,
        "bucket": bucket,
        "staging_bucket": staging_bucket,
        "output_dir": output_dir,
        "service_account": service_account,
        "credentials_detected": credentials_detected,
        "warnings": warnings,
        "errors": errors,
    }
