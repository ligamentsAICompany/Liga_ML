"""Shared secret redaction policy for logs, persistence, and frontend payloads."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

REDACTED = "[REDACTED]"

SECRET_KEY_RE = re.compile(
    r"(?i)(token|secret|password|credential|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|session[_-]?token[_-]?encryption[_-]?key|"
    r"authorization|google_application_credentials|mongodb_uri|"
    r"aws_access_key_id|aws_secret_access_key|aws_session_token|"
    r"hf_token|huggingface_hub_token|openai_api_key|anthropic_api_key|"
    r"github_token)"
)
AUTHORIZATION_KEY_RE = re.compile(r"(?i)^authorization$")

_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"sk-(?!ant-|proj-)[A-Za-z0-9_\-]{32,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
)

_BEARER_RE = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9_\-\.=:/+]{12,}")
_ENV_ASSIGNMENT_RE = re.compile(
    r"(?im)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|CREDENTIALS?|API_KEY|"
    r"PRIVATE_KEY|ACCESS_KEY|SESSION_TOKEN|MONGODB_URI)[A-Z0-9_]*)\s*([=:])\s*"
    r"([^ \t\r\n\"']+)"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)
_MONGO_URI_RE = re.compile(
    r"\b(mongodb(?:\+srv)?://)([^/@\s:]+):([^/@\s]+)@([^\s/?#]+)([^\s]*)",
    re.IGNORECASE,
)
_SIGNED_URL_KEYS = re.compile(
    r"(?i)(X-Amz-Signature|X-Amz-Credential|X-Amz-Security-Token|"
    r"GoogleAccessId|Signature|X-Goog-Signature|X-Goog-Credential)"
)
_CREDENTIAL_PATH_RE = re.compile(
    r"(?i)(?:^|[\s=:])(/[^\s]*?(?:credentials?|service[-_]?account)[^\s]*?\.json)\b"
)


def _redact_url_if_needed(text: str) -> str:
    try:
        parsed = urlsplit(text)
    except ValueError:
        return text
    if not parsed.scheme or not parsed.netloc:
        return text
    changed = False
    netloc = parsed.netloc
    if "@" in netloc:
        host = netloc.rsplit("@", 1)[1]
        netloc = f"{REDACTED}@{host}"
        changed = True
    query = parsed.query
    if query and _SIGNED_URL_KEYS.search(query):
        query = REDACTED
        changed = True
    if not changed:
        return text
    return urlunsplit(
        SplitResult(parsed.scheme, netloc, parsed.path, query, parsed.fragment)
    )


def redact_text(text: str) -> str:
    """Redact known provider tokens, credentials, private keys, and auth URLs."""
    if not isinstance(text, str) or not text:
        return text
    out = _PRIVATE_KEY_RE.sub(REDACTED, text)
    out = _CREDENTIAL_PATH_RE.sub(
        lambda match: match.group(0).replace(match.group(1), REDACTED), out
    )
    out = _MONGO_URI_RE.sub(r"\1[REDACTED]@\4\5", out)
    out = _BEARER_RE.sub(lambda match: f"{match.group(1)} {REDACTED}", out)
    out = _ENV_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", out
    )
    for pattern in _TOKEN_PATTERNS:
        out = pattern.sub(REDACTED, out)
    if ("://" in out and "@" in out) or ("?" in out and _SIGNED_URL_KEYS.search(out)):
        parts = [
            (_redact_url_if_needed(part) if "://" in part else part)
            for part in out.split()
        ]
        if len(parts) > 1:
            out = " ".join(parts)
        else:
            out = _redact_url_if_needed(out)
    return out


def redact_json_like(value: Any) -> Any:
    """Recursively redact strings and secret-like mapping keys."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return redact_mapping(value)
    if isinstance(value, list):
        return [redact_json_like(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_json_like(item) for item in value)
    return value


def redact_mapping(mapping: dict) -> dict:
    """Return a redacted copy of a mapping without mutating the input."""
    clean: dict[str, Any] = {}
    for key, item in mapping.items():
        key_text = str(key)
        if AUTHORIZATION_KEY_RE.search(key_text) and isinstance(item, str):
            clean[key_text] = redact_text(item)
        elif SECRET_KEY_RE.search(key_text):
            clean[key_text] = REDACTED
        else:
            clean[key_text] = redact_json_like(item)
    return clean


def contains_secret_like_value(value: Any) -> bool:
    """Return True when a nested value appears to contain a secret."""
    return redact_json_like(value) != value


def sanitize_for_persistence(value: Any) -> Any:
    """Redact values before writing to durable or local storage."""
    return redact_json_like(value)


def sanitize_for_frontend(value: Any) -> Any:
    """Redact values before streaming or returning them to browsers."""
    return redact_json_like(value)


def scrub_string(s: str) -> str:
    """Backward-compatible alias for trajectory redaction."""
    return redact_text(s)


def scrub(obj: Any) -> Any:
    """Backward-compatible alias for recursive trajectory redaction."""
    return redact_json_like(obj)
