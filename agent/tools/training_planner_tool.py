"""Read-only training planner tool."""

from __future__ import annotations

import json
from typing import Any

from agent.core.training_planner import recommend_training_plan
from agent.tools.types import ToolResult


def _json_block(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def format_training_plan(plan: Any) -> str:
    result = plan.to_dict()
    hardware = _json_block(plan.recommended_hardware)
    training_args = _json_block(plan.training_args)
    privacy_warnings = plan.privacy_warnings or ["None."]
    risks = plan.risks or ["None."]
    discovery = plan.dataset_discovery or {}
    discovery_lines: list[str] = []
    if isinstance(discovery, dict) and discovery:
        recommended = discovery.get("recommended_candidate")
        if isinstance(recommended, dict):
            discovery_lines.append(
                f"- Recommended candidate: {recommended.get('title') or recommended.get('dataset_id')}"
            )
        discovery_lines.extend(
            f"- Warning: {warning}" for warning in discovery.get("warnings") or []
        )
        discovery_lines.append("- User selection required before training.")

    lines = [
        "## Training Planner Recommendation",
        "",
        f"**Provider:** {plan.provider}",
        f"**Training goal:** {plan.training_goal}",
        f"**Task type:** {plan.task_type}",
        f"**Domain:** {plan.domain}",
        f"**Recommended model:** {plan.recommended_model}",
        f"**Smoke-test model:** {plan.smoke_test_model}",
        f"**Production model:** {plan.production_model}",
        f"**Output policy:** {plan.output_policy}",
        "",
        "### Recommended Hardware",
        f"**Recommended hardware:** {plan.recommended_hardware}",
        "",
        "```json",
        hardware,
        "```",
        "",
        "### Training Arguments",
        "",
        "```json",
        training_args,
        "```",
        "",
        "### Privacy Warnings",
        "",
        *[f"- {warning}" for warning in privacy_warnings],
        "",
        "### Risks",
        "",
        *[f"- {risk}" for risk in risks],
        "",
        "### Reasoning",
        "",
        *[f"- {reason}" for reason in plan.reasoning],
        "",
        "### Dataset Discovery",
        "",
        *(discovery_lines or ["- No enhanced discovery result supplied."]),
        "",
        "### Structured Result",
        "",
        "```json",
        _json_block(result),
        "```",
        "",
        "Planning only: this tool never launches jobs, makes cloud calls, or spends money.",
    ]
    return "\n".join(lines)


class TrainingPlannerTool:
    """Recommend safe fine-tuning settings without launching work."""

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        operation = str(params.get("operation", "")).strip().lower()
        if operation != "recommend":
            return {
                "formatted": (
                    f'Unknown operation: "{operation}". Available operations: recommend.'
                ),
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        plan = recommend_training_plan(
            provider=str(params.get("provider") or "hf-jobs"),
            domain=str(params.get("domain") or "general"),
            training_goal=str(params.get("training_goal") or "agent-decide"),
            dataset_summary=params.get("dataset_summary")
            if isinstance(params.get("dataset_summary"), dict)
            else None,
            uploaded_dataset_available=params.get("uploaded_dataset_available")
            if isinstance(params.get("uploaded_dataset_available"), bool)
            else None,
            task_type=str(params.get("task_type") or "sft"),
            privacy_level=str(params.get("privacy_level") or "unknown"),
            budget_preference=str(params.get("budget_preference") or "balanced"),
            user_model_preference=params.get("user_model_preference"),
            intent_hint=params.get("intent_hint"),
            full_dataset_approved=params.get("full_dataset_approved") is True,
            dataset_discovery=params.get("dataset_discovery")
            if isinstance(params.get("dataset_discovery"), dict)
            else None,
        )
        return {
            "formatted": format_training_plan(plan),
            "totalResults": 1,
            "resultsShared": 1,
            "isError": False,
        }


TRAINING_PLANNER_TOOL_SPEC = {
    "name": "training_planner",
    "description": (
        "Recommend a safe fine-tuning plan including model, hardware, training goal, "
        "output policy, and risk notes. Planning only; never launches jobs, makes "
        "cloud calls, uploads data, or spends money."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["recommend"],
                "description": "Operation to execute. Only recommend is supported.",
            },
            "provider": {
                "type": "string",
                "enum": ["hf-jobs", "gcp-vertex", "aws-sagemaker"],
                "description": "Training provider to plan for.",
            },
            "domain": {
                "type": "string",
                "description": "Domain such as finance, medical, manufacturing, customer_support, call_center, legal, general, or a custom domain.",
            },
            "training_goal": {
                "type": "string",
                "enum": ["smoke-test", "production", "agent-decide"],
                "description": "Requested planning goal.",
            },
            "dataset_summary": {
                "type": "object",
                "description": "Dataset summary, e.g. {'rows': 100, 'columns': ['question', 'answer'], 'source_format': 'csv'}.",
                "properties": {
                    "rows": {"type": "integer"},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "source_format": {"type": "string"},
                },
            },
            "uploaded_dataset_available": {
                "type": "boolean",
                "description": "Whether an uploaded normalized dataset is already available in session context. If false and no dataset summary exists, dataset discovery is required before final planning.",
            },
            "task_type": {
                "type": "string",
                "enum": ["sft"],
                "description": "Task type. Initial planner defaults are for SFT.",
            },
            "privacy_level": {
                "type": "string",
                "enum": ["sensitive", "general", "unknown"],
                "description": "Privacy sensitivity of the data.",
            },
            "budget_preference": {
                "type": "string",
                "enum": ["low", "balanced", "performance"],
                "description": "Budget/performance tradeoff for recommendations.",
            },
            "user_model_preference": {
                "type": "string",
                "description": "Optional model id requested by the user; the planner respects it and adds risk notes when needed.",
            },
            "intent_hint": {
                "type": "string",
                "description": "Optional short user-intent text to help agent-decide choose smoke-test or production.",
            },
            "full_dataset_approved": {
                "type": "boolean",
                "description": "Set true only after explicit approval to plan full-dataset cloud training. Otherwise GCloud production plans use capped pilot samples.",
            },
            "dataset_discovery": {
                "type": "object",
                "description": "Optional enhanced dataset discovery result. Planner treats it as recommendation context and still requires user confirmation before launch.",
            },
        },
        "required": ["operation"],
    },
}


async def training_planner_handler(arguments: dict[str, Any]) -> tuple[str, bool]:
    tool = TrainingPlannerTool()
    result = await tool.execute(arguments)
    return result["formatted"], not result.get("isError", False)
