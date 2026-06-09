"""Safe provider error normalization for training preflight checks."""

from __future__ import annotations

import re
from typing import Any

from agent.core.redact import redact_text, sanitize_for_frontend

PREFLIGHT_ERROR_KINDS = {
    "missing_credentials",
    "auth_failed",
    "not_found",
    "permission_denied",
    "quota_unavailable",
    "quota_exceeded",
    "unsupported",
    "timeout",
    "network",
    "sdk_missing",
    "provider_error",
    "unknown",
}

_WINDOWS_CREDENTIAL_PATH_RE = re.compile(
    r"(?i)\b[A-Z]:\\[^\s\"']*(?:credentials?|service[-_]?account)[^\s\"']*\.json"
)
_POSIX_CREDENTIAL_PATH_RE = re.compile(
    r"(?i)\b/[^\s\"']*(?:credentials?|service[-_]?account)[^\s\"']*\.json"
)


def _without_credential_paths(message: str) -> str:
    out = _WINDOWS_CREDENTIAL_PATH_RE.sub("[REDACTED]", message)
    return _POSIX_CREDENTIAL_PATH_RE.sub("[REDACTED]", out)


def _safe_message(error: BaseException | str) -> str:
    raw = str(error) if error else "Unknown provider error."
    redacted = redact_text(raw)
    redacted = _without_credential_paths(redacted)
    return str(sanitize_for_frontend(redacted)) or "Unknown provider error."


def _error_code(message: str, *, provider: str | None) -> str:
    lower = message.lower()
    if any(token in lower for token in ("timed out", "timeout", "deadline exceeded")):
        return "timeout"
    if any(
        token in lower
        for token in ("missing credential", "no credential", "credentials not found")
    ):
        return "missing_credentials"
    if any(
        token in lower
        for token in ("unauthorized", "unauthenticated", "invalid token", "401")
    ):
        return "auth_failed"
    if any(
        token in lower
        for token in ("permission denied", "forbidden", "access denied", "403")
    ):
        return "permission_denied"
    if any(
        token in lower for token in ("not found", "404", "does not exist", "not exist")
    ):
        return "not_found"
    if any(
        token in lower
        for token in ("quota exceeded", "limit exceeded", "resource exhausted")
    ):
        return "quota_exceeded"
    if any(
        token in lower
        for token in ("quota unavailable", "quota not available", "quota unknown")
    ):
        return "quota_unavailable"
    if any(
        token in lower for token in ("unsupported", "not supported", "incompatible")
    ):
        return "unsupported"
    if any(
        token in lower
        for token in (
            "connection refused",
            "connection reset",
            "dns",
            "network",
            "temporary failure",
            "name resolution",
            "socket",
        )
    ):
        return "network"
    if any(
        token in lower
        for token in ("no module named", "module not found", "sdk missing")
    ):
        return "sdk_missing"
    return "provider_error" if provider else "unknown"


def normalize_provider_error(
    error: BaseException | str,
    *,
    provider: str | None = None,
) -> dict[str, Any]:
    """Return a stable, redacted provider error payload.

    The result is safe for logs, persistence, and frontend rendering. It does not
    include stack traces or raw credential paths.
    """

    safe_message = _safe_message(error)
    payload: dict[str, Any] = {
        "error_code": _error_code(safe_message, provider=provider),
        "message": safe_message,
    }
    if provider:
        payload["provider"] = str(provider)
    return payload
