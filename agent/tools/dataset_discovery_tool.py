"""Read-only no-upload dataset discovery planner tool."""

from __future__ import annotations

from typing import Any

from agent.core.dataset_discovery import (
    DEFAULT_ALLOWED_SOURCES,
    DEFAULT_EXCLUDED_SOURCES,
    build_dataset_discovery_plan,
    format_dataset_discovery_plan,
)
from agent.tools.types import ToolResult


class DatasetDiscoveryTool:
    """Plan dataset discovery when no uploaded dataset is available."""

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        operation = str(params.get("operation", "")).strip().lower()
        if operation != "plan":
            return {
                "formatted": (
                    f'Unknown operation: "{operation}". Available operations: plan.'
                ),
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        if params.get("uploaded_dataset_available") is True:
            return {
                "formatted": (
                    "## Dataset Discovery Plan\n\n"
                    "Uploaded dataset available: use the uploaded normalized dataset first. "
                    "Do not start no-upload dataset discovery unless the uploaded dataset "
                    "is unusable or the user explicitly asks for alternatives.\n\n"
                    "Planning only: this tool never launches jobs, makes cloud calls, "
                    "uploads data, downloads datasets, or spends money."
                ),
                "totalResults": 1,
                "resultsShared": 1,
                "isError": False,
            }

        candidates = params.get("candidates")
        plan = build_dataset_discovery_plan(
            domain=str(params.get("domain") or "general"),
            task_type=str(params.get("task_type") or "general"),
            provider=str(params.get("provider") or "hf-jobs"),
            user_goal=params.get("user_goal")
            if isinstance(params.get("user_goal"), str)
            else None,
            allowed_sources=params.get("allowed_sources")
            if isinstance(params.get("allowed_sources"), list)
            else None,
            excluded_sources=params.get("excluded_sources")
            if isinstance(params.get("excluded_sources"), list)
            else None,
            candidates=candidates if isinstance(candidates, list) else None,
        )
        return {
            "formatted": format_dataset_discovery_plan(plan),
            "totalResults": len(plan.candidates) or 1,
            "resultsShared": len(plan.candidates) or 1,
            "isError": False,
        }


DATASET_DISCOVERY_TOOL_SPEC = {
    "name": "dataset_discovery",
    "description": (
        "Plan and rank no-upload dataset discovery candidates for fine-tuning. "
        "Use when the user wants training/fine-tuning but no uploaded dataset is "
        "available. Read-only and non-billable; never launches jobs, makes cloud "
        "calls, crawls the web, downloads datasets, uploads data, or spends money. "
        "Kaggle is excluded in this version."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["plan"],
                "description": "Operation to execute. Only plan is supported.",
            },
            "domain": {
                "type": "string",
                "description": "Domain such as finance, medical, manufacturing, customer_support, call_center, legal, general, or a custom domain.",
            },
            "task_type": {
                "type": "string",
                "enum": ["sft", "classification", "rag", "dpo", "general"],
                "description": "Target task type for dataset discovery.",
            },
            "provider": {
                "type": "string",
                "enum": ["hf-jobs", "gcp-vertex", "aws-sagemaker"],
                "description": "Provider the eventual training plan may target.",
            },
            "user_goal": {
                "type": "string",
                "description": "Short description of the user's training goal.",
            },
            "uploaded_dataset_available": {
                "type": "boolean",
                "description": "Whether an uploaded normalized dataset already exists in session context.",
            },
            "allowed_sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Allowed discovery sources. Defaults to "
                    f"{', '.join(DEFAULT_ALLOWED_SOURCES)}."
                ),
            },
            "excluded_sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Excluded discovery sources. Defaults to "
                    f"{', '.join(DEFAULT_EXCLUDED_SOURCES)}."
                ),
            },
            "candidates": {
                "type": "array",
                "description": "Optional candidate dataset metadata from search/research tools to normalize, rank, and format.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "source": {"type": "string"},
                        "url": {"type": "string"},
                        "domain": {"type": "string"},
                        "task_type": {"type": "string"},
                        "license": {"type": "string"},
                        "size": {"type": "string"},
                        "schema_hint": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "quality_notes": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "risks": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "score": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
        "required": ["operation"],
    },
}


async def dataset_discovery_handler(arguments: dict[str, Any]) -> tuple[str, bool]:
    tool = DatasetDiscoveryTool()
    result = await tool.execute(arguments)
    return result["formatted"], not result.get("isError", False)
