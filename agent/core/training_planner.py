"""Shared fine-tuning planning helpers.

The planner is intentionally static and read-only: it does not inspect remote
catalogs, launch jobs, or verify cloud availability. Recommendations are safe
defaults for preflight discussion and approval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agent.core.output_policy import (
    cloud_storage_label,
    default_output_policy_for_domain,
    is_sensitive_domain,
    output_policy_label,
    privacy_warning_for_policy,
)


SUPPORTED_PROVIDERS = {"hf-jobs", "gcp-vertex", "aws-sagemaker"}
SUPPORTED_TASK_TYPES = {"sft"}
VALID_TRAINING_GOALS = {"smoke-test", "production", "agent-decide"}
VALID_PRIVACY_LEVELS = {"sensitive", "general", "unknown"}
VALID_BUDGET_PREFERENCES = {"low", "balanced", "performance"}

SMOKE_TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PRODUCTION_MODELS = {
    "low": "Qwen/Qwen2.5-3B-Instruct",
    "balanced": "meta-llama/Llama-3.2-3B-Instruct",
    "performance": "mistralai/Mistral-7B-Instruct-v0.3",
}

KNOWN_DOMAIN_FAMILIES = {
    "finance",
    "medical",
    "manufacturing",
    "customer_support",
    "call_center",
    "legal",
    "general",
}

HARDWARE_CATALOG: dict[str, dict[str, dict[str, Any]]] = {
    "gcp-vertex": {
        "smoke-test": {
            "machine_type": "n1-standard-8",
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": 1,
        },
        "production": {
            "machine_type": "n1-standard-16",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1,
        },
    },
    "aws-sagemaker": {
        "smoke-test": {
            "instance_type": "ml.g5.xlarge",
            "instance_count": 1,
            "max_run_seconds": 3600,
        },
        "production": {
            "instance_type": "ml.g5.2xlarge",
            "instance_count": 1,
            "max_run_seconds": 7200,
        },
    },
    "hf-jobs": {
        "smoke-test": {"hardware_flavor": "t4-small"},
        "production": {"hardware_flavor": "a10g-largex2"},
    },
}


@dataclass(frozen=True)
class TrainingPlan:
    provider: str
    domain: str
    task_type: str
    training_goal: str
    recommended_model: str
    smoke_test_model: str
    production_model: str
    recommended_hardware: dict[str, Any]
    training_args: dict[str, Any]
    output_policy: str
    privacy_warnings: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_provider(provider: str | None) -> str:
    value = (provider or "hf-jobs").strip().lower()
    return value or "hf-jobs"


def normalize_domain(domain: str | None) -> str:
    value = (domain or "general").strip().lower().replace("-", "_").replace(" ", "_")
    return value or "general"


def normalize_privacy_level(privacy_level: str | None) -> str:
    value = (privacy_level or "unknown").strip().lower()
    return value if value in VALID_PRIVACY_LEVELS else "unknown"


def normalize_budget_preference(budget_preference: str | None) -> str:
    value = (budget_preference or "balanced").strip().lower()
    return value if value in VALID_BUDGET_PREFERENCES else "balanced"


def detect_privacy_level(domain: str | None, privacy_level: str | None = None) -> str:
    normalized_privacy = normalize_privacy_level(privacy_level)
    if normalized_privacy == "sensitive" or is_sensitive_domain(domain):
        return "sensitive"
    if normalized_privacy == "general":
        return "general"
    return "unknown"


def _dataset_rows(dataset_summary: dict[str, Any] | None) -> int | None:
    if not isinstance(dataset_summary, dict):
        return None
    rows = dataset_summary.get("rows")
    if isinstance(rows, bool):
        return None
    if isinstance(rows, (int, float)) and rows >= 0:
        return int(rows)
    return None


def _choose_training_goal(
    training_goal: str | None,
    *,
    dataset_rows: int | None,
    intent_hint: str | None,
) -> tuple[str, list[str]]:
    requested = (training_goal or "agent-decide").strip().lower()
    if requested in {"smoke_test", "smoke"}:
        requested = "smoke-test"
    if requested not in VALID_TRAINING_GOALS:
        requested = "agent-decide"

    reasoning: list[str] = []
    hint = (intent_hint or "").lower()
    if requested != "agent-decide":
        reasoning.append(f"Using requested training goal: {requested}.")
        return requested, reasoning

    if any(word in hint for word in ("test", "try", "demo", "smoke")):
        reasoning.append("Agent-decide selected smoke-test from user intent signal.")
        return "smoke-test", reasoning
    if any(word in hint for word in ("production", "deploy", "real", "final")):
        reasoning.append("Agent-decide selected production from user intent signal.")
        return "production", reasoning
    if dataset_rows is not None and dataset_rows <= 20:
        reasoning.append(
            "Agent-decide selected smoke-test because the dataset is tiny."
        )
        return "smoke-test", reasoning

    reasoning.append("Agent-decide selected production-ready balanced defaults.")
    return "production", reasoning


def _production_training_args(
    dataset_rows: int | None,
    budget: str,
    provider: str,
    *,
    full_dataset_approved: bool = False,
) -> dict[str, Any]:
    if provider == "gcp-vertex" and not full_dataset_approved:
        train_cap, eval_cap = (100, 20) if budget == "low" else (500, 50)
        return {
            "max_train_samples": train_cap,
            "max_eval_samples": eval_cap,
            "num_train_epochs": 1,
            "max_length": 2048 if budget == "performance" else 1024,
            "max_run_hours": 3 if budget == "performance" else 2,
        }

    if dataset_rows is None:
        max_train_samples: int | None = None
        max_eval_samples: int | None = None
        epochs = 2
    elif dataset_rows < 1_000:
        max_train_samples = None
        max_eval_samples = None
        epochs = 3
    elif dataset_rows < 10_000:
        max_train_samples = None
        max_eval_samples = None
        epochs = 2
    elif dataset_rows <= 50_000:
        max_train_samples = None
        max_eval_samples = None
        epochs = 1
    else:
        max_train_samples = 50_000 if budget != "performance" else 100_000
        max_eval_samples = None
        epochs = 1

    return {
        "max_train_samples": max_train_samples,
        "max_eval_samples": max_eval_samples,
        "num_train_epochs": epochs,
        "max_length": 2048 if budget == "performance" else 1024,
        "max_run_hours": 4 if budget == "performance" else 2,
    }


def _training_args(
    goal: str,
    dataset_rows: int | None,
    budget: str,
    provider: str,
    *,
    full_dataset_approved: bool = False,
) -> dict[str, Any]:
    if goal == "smoke-test":
        return {
            "max_train_samples": 5,
            "max_eval_samples": 2,
            "num_train_epochs": 1,
            "max_length": 512,
            "max_run_hours": 1,
        }
    return _production_training_args(
        dataset_rows,
        budget,
        provider,
        full_dataset_approved=full_dataset_approved,
    )


def _recommended_hardware(provider: str, goal: str) -> dict[str, Any]:
    provider_catalog = HARDWARE_CATALOG.get(provider) or HARDWARE_CATALOG["hf-jobs"]
    goal_key = "smoke-test" if goal == "smoke-test" else "production"
    return dict(provider_catalog[goal_key])


def _privacy_notes(provider: str, domain: str, privacy: str) -> tuple[str, list[str]]:
    output_policy = default_output_policy_for_domain(domain, provider)
    if privacy != "sensitive":
        return output_policy, []

    warnings = [
        "Sensitive or regulated data detected; prefer private cloud storage and avoid pushing to external registries unless explicitly approved."
    ]
    warnings.append(
        f"For {provider}, cloud-private means {output_policy_label(provider, output_policy)} "
        f"({cloud_storage_label(provider)})."
    )
    if hf_warning := privacy_warning_for_policy(provider, output_policy):
        warnings.append(hf_warning)
    return output_policy, warnings


def _model_risks(model_id: str, goal: str) -> list[str]:
    risks = [f"Using user-provided model preference `{model_id}`."]
    lowered = model_id.lower()
    known_prefixes = ("qwen/", "meta-llama/", "mistralai/", "google/", "microsoft/")
    if not lowered.startswith(known_prefixes):
        risks.append(
            "User-provided model is outside the local example catalog; verify architecture, license, tokenizer, and availability before training."
        )
    if any(token in lowered for token in ("70b", "65b", "34b", "30b", "22b", "large")):
        risks.append(
            "User-provided model may be large for the recommended smoke-test hardware; preflight memory before launch."
        )
    if goal == "smoke-test" and not any(
        token in lowered for token in ("0.5b", "1b", "1.5b", "3b")
    ):
        risks.append(
            "Smoke-test runs are safer with a small model; this preference may need stronger hardware."
        )
    return risks


def recommend_training_plan(
    *,
    provider: str = "hf-jobs",
    domain: str = "general",
    training_goal: str = "agent-decide",
    dataset_summary: dict[str, Any] | None = None,
    uploaded_dataset_available: bool | None = None,
    task_type: str = "sft",
    privacy_level: str = "unknown",
    budget_preference: str = "balanced",
    user_model_preference: str | None = None,
    intent_hint: str | None = None,
    full_dataset_approved: bool = False,
) -> TrainingPlan:
    normalized_provider = normalize_provider(provider)
    normalized_domain = normalize_domain(domain)
    normalized_task = (task_type or "sft").strip().lower()
    budget = normalize_budget_preference(budget_preference)
    rows = _dataset_rows(dataset_summary)
    goal, reasoning = _choose_training_goal(
        training_goal,
        dataset_rows=rows,
        intent_hint=intent_hint,
    )

    privacy = detect_privacy_level(normalized_domain, privacy_level)
    output_policy, privacy_warnings = _privacy_notes(
        normalized_provider,
        normalized_domain,
        privacy,
    )

    production_model = PRODUCTION_MODELS[budget]
    recommended_model = SMOKE_TEST_MODEL if goal == "smoke-test" else production_model
    risks: list[str] = []
    if user_model_preference:
        recommended_model = user_model_preference.strip()
        risks.extend(_model_risks(recommended_model, goal))

    if uploaded_dataset_available is False or rows is None:
        risks.append(
            "No training dataset summary is available; dataset discovery is required before final training plan approval."
        )
        reasoning.append(
            "Run dataset_discovery first, then search allowed public sources, inspect schema/license/privacy, and do not launch a cloud job until the user selects a dataset."
        )

    if normalized_provider not in SUPPORTED_PROVIDERS:
        risks.append(
            f"Unknown provider `{normalized_provider}`; using HF Jobs-style recommendation shape as a fallback."
        )
    if normalized_task not in SUPPORTED_TASK_TYPES:
        risks.append(
            f"Task type `{normalized_task}` is not in the initial planner catalog; defaults are tuned for SFT."
        )
    if normalized_domain not in KNOWN_DOMAIN_FAMILIES:
        reasoning.append(
            f"Domain `{normalized_domain}` is treated as an extensible custom domain."
        )

    if rows is not None:
        reasoning.append(f"Dataset summary reports {rows} rows.")
        if (
            normalized_provider == "gcp-vertex"
            and rows > 10_000
            and not full_dataset_approved
        ):
            risks.append(
                f"Dataset has about {rows:,} rows; use a capped production pilot first. Full dataset training requires separate approval."
            )
    reasoning.append(
        "Recommendations are static planning defaults and do not guarantee provider hardware availability."
    )

    return TrainingPlan(
        provider=normalized_provider,
        domain=normalized_domain,
        task_type=normalized_task,
        training_goal=goal,
        recommended_model=recommended_model,
        smoke_test_model=SMOKE_TEST_MODEL,
        production_model=production_model,
        recommended_hardware=_recommended_hardware(normalized_provider, goal),
        training_args=_training_args(
            goal,
            rows,
            budget,
            normalized_provider,
            full_dataset_approved=full_dataset_approved,
        ),
        output_policy=output_policy,
        privacy_warnings=privacy_warnings,
        risks=risks,
        reasoning=reasoning,
    )
