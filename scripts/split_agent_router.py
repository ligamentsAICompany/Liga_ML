#!/usr/bin/env python3
"""One-off splitter: backend/routes/agent.py -> backend/routes/api/*."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "backend" / "routes" / "agent.py"
OUT = ROOT / "backend" / "routes" / "api"

MODULE_HEADER = '''"""{doc}"""

# ruff: noqa: F403, F405, E402
from fastapi import APIRouter

import routes.api.common as _common
from routes.api.common import *

for _name, _value in vars(_common).items():
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = _value

router = APIRouter()
'''

INIT_PY = '''"""Domain-specific API routers.

Re-exports handlers and helpers for unit tests that previously imported
``routes.agent``.
"""

# ruff: noqa: F403, F405
import routes.api.common as common
from routes.api.chat import *  # noqa: F403
from routes.api.observability import *  # noqa: F403
from routes.api.sessions import *  # noqa: F403
from routes.api.training import *  # noqa: F403

for _name, _value in vars(common).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
'''


def classify_route(block: str) -> str:
    if any(
        p in block
        for p in (
            "training-preflight",
            "dataset-discovery",
            "recommendations",
            "/preflight",
        )
    ):
        return "training"
    if "@router.post" in block and "evaluation" in block and "report" not in block:
        return "training"
    if any(
        p in block
        for p in (
            "/chat/",
            "/events/",
            '"/approve"',
            "/undo/",
            "/interrupt/",
            '"/submit"',
            "/truncate/",
            "/compact/",
            "/shutdown/",
            "/feedback/",
            "/messages",
            "pro-click",
            "/stream",
            "/runs/{run_id}/interrupt",
        )
    ):
        return "chat"
    if any(
        p in block
        for p in (
            "/health/",
            '"/health"',
            "/usage",
            "/audit",
            "/evaluations",
            "/responses",
            "catalog",
            "/evaluation/report",
        )
    ) and "/session/" not in block.split("async def", 1)[0]:
        return "observability"
    if any(
        p in block
        for p in (
            '"/responses"',
            '"/config/model"',
            '"/title"',
            "/evaluation/report",
        )
    ):
        return "observability"
    return "sessions"


def split_route_block(block_lines: list[str]) -> tuple[list[str], list[str]]:
    """Return (handler-only lines, helper lines trailing before next route)."""
    if not block_lines:
        return [], []
    block_text = "\n".join(block_lines) + "\n"
    tree = ast.parse(block_text)
    func = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        None,
    )
    if func is None:
        return block_lines, []

    start = func.lineno - 1
    if func.decorator_list:
        start = min(decorator.lineno for decorator in func.decorator_list) - 1
    end = func.end_lineno
    handler = block_lines[start:end]
    helpers = block_lines[end:]
    return handler, helpers


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    lines = text.splitlines()

    route_starts = [i for i, line in enumerate(lines) if line.startswith("@router.")]
    blocks: dict[str, list[str]] = {
        "sessions": [],
        "chat": [],
        "training": [],
        "observability": [],
    }
    extra_common: list[str] = []

    for idx, start in enumerate(route_starts):
        end = route_starts[idx + 1] if idx + 1 < len(route_starts) else len(lines)
        block_lines = lines[start:end]
        handler_lines, helper_lines = split_route_block(block_lines)
        extra_common.extend(helper_lines)
        block_text = "\n".join(handler_lines)
        mod = classify_route(block_text)
        blocks[mod].append(block_text)

    first_route = route_starts[0]
    pre_route = lines[:first_route]
    common_body = "\n".join(pre_route + extra_common)
    if "# ruff: noqa: F401" not in common_body:
        common_body = common_body.replace(
            'dependency. In dev mode (no OAUTH_CLIENT_ID), auth is bypassed automatically.\n"""',
            'dependency. In dev mode (no OAUTH_CLIENT_ID), auth is bypassed automatically.\n"""\n\n# ruff: noqa: F401',
            1,
        )
    common_body = common_body.replace(
        'router = APIRouter(prefix="/api", tags=["agent"])',
        "# Router split across routes/api/* modules",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "__init__.py").write_text(INIT_PY, encoding="utf-8")
    (OUT / "common.py").write_text(common_body + "\n", encoding="utf-8")

    docs = {
        "sessions": "Session lifecycle, configuration, runs, and user quota routes.",
        "chat": "Chat submission, SSE streaming, and session control routes.",
        "training": "Training preflight, discovery, and recommendation routes.",
        "observability": "Health, usage, audit, evaluation, and response log routes.",
    }
    for mod, doc in docs.items():
        body = MODULE_HEADER.format(doc=doc) + "\n\n" + "\n\n".join(blocks[mod]) + "\n"
        (OUT / f"{mod}.py").write_text(body, encoding="utf-8")

    print(
        "Wrote common.py and",
        ", ".join(f"{m}.py ({len(blocks[m])} routes)" for m in docs),
    )


if __name__ == "__main__":
    main()
