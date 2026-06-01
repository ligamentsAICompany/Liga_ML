#!/usr/bin/env python3
"""Print the local Google Cloud / Vertex AI readiness snapshot as JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.core.gcp_readiness import build_gcp_vertex_readiness_snapshot  # noqa: E402

_SENSITIVE_KEY_PARTS = ("token", "secret", "password", "private_key", "credential")


def _safe_snapshot(value: Any) -> Any:
    """Drop accidental secret-bearing fields before printing diagnostics."""
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
                if key == "credentials_detected":
                    safe[key] = bool(item)
                continue
            safe[key] = _safe_snapshot(item)
        return safe
    if isinstance(value, list):
        return [_safe_snapshot(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    _ = argv
    snapshot = _safe_snapshot(build_gcp_vertex_readiness_snapshot())
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0 if snapshot.get("configured") is True else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
