#!/usr/bin/env python3
"""Print the local AWS SageMaker readiness snapshot as safe JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from agent.core.aws_readiness import build_aws_sagemaker_readiness_snapshot  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env", override=False)

_SENSITIVE_KEY_PARTS = (
    "token",
    "secret",
    "password",
    "private_key",
    "credential",
    "access_key",
)


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
    snapshot = _safe_snapshot(build_aws_sagemaker_readiness_snapshot())
    training_image_uri = snapshot.get("training_image_uri")
    payload = {
        **snapshot,
        "training_image_configured": bool(training_image_uri),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("configured") is True else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
