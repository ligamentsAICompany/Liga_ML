"""Shared fine-tuning planning helpers.

The planner is intentionally static and read-only: it does not inspect remote
catalogs, launch jobs, or verify cloud availability. Recommendations are safe
defaults for preflight discussion and approval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agent.core.model_provider_selection import (
    HardwareCandidate,
    ModelCandidate,
    SelectionFallback,
    SelectionReason,
    SelectionRisk,
    TrainingPlanRecommendation,
    catalog_model,
    catalog_provider,
    estimate_plan_cost,
    first_hardware,
    first_model,
    hardware_catalog,
)
from agent.core.output_policy import (
    OUTPUT_POLICY_CLOUD_AND_HF_HUB,
    OUTPUT_POLICY_CLOUD_PRIVATE,
    OUTPUT_POLICY_HF_HUB,
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
    dataset_discovery: dict[str, Any] | None = None
    recommendation: TrainingPlanRecommendation | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.recommendation is not None:
            payload["recommendation"] = self.recommendation.to_dict()
        return payload


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


def _readiness_for(
    provider: str, provider_readiness: dict[str, Any] | None
) -> dict[str, Any]:
    if not isinstance(provider_readiness, dict):
        return {}
    direct = provider_readiness.get(provider)
    if isinstance(direct, dict):
        return direct
    underscored = provider_readiness.get(provider.replace("-", "_"))
    return underscored if isinstance(underscored, dict) else {}


def _provider_ready(
    provider: str, provider_readiness: dict[str, Any] | None
) -> bool | None:
    readiness = _readiness_for(provider, provider_readiness)
    if not readiness:
        return None
    for key in ("configured", "ready", "available"):
        if key in readiness:
            return bool(readiness[key])
    return None


def _select_provider(
    provider: str,
    *,
    explicit_provider: bool,
    provider_readiness: dict[str, Any] | None,
) -> tuple[str, list[SelectionRisk], list[SelectionFallback], list[SelectionReason]]:
    warnings: list[SelectionRisk] = []
    fallbacks: list[SelectionFallback] = []
    reasons: list[SelectionReason] = []
    ready = _provider_ready(provider, provider_readiness)
    if provider == "gcp-vertex" and ready is False and not explicit_provider:
        warnings.append(
            SelectionRisk(
                "readiness",
                "warning",
                "GCloud Vertex readiness is false; using HF Jobs for the planner recommendation.",
            )
        )
        fallbacks.append(
            SelectionFallback(
                "gcp-vertex",
                "hf-jobs",
                "GCloud readiness is false and the provider was not explicitly locked.",
            )
        )
        return "hf-jobs", warnings, fallbacks, reasons
    if ready is False:
        warnings.append(
            SelectionRisk(
                "readiness",
                "warning",
                f"{provider} readiness is false; do not launch until configuration is fixed.",
            )
        )
    elif ready is None and provider in {"gcp-vertex", "aws-sagemaker"}:
        warnings.append(
            SelectionRisk(
                "readiness",
                "warning",
                f"{provider} readiness is unknown; verify configuration and quota before approval.",
            )
        )
    reasons.append(SelectionReason("provider", f"Using provider {provider}."))
    return provider, warnings, fallbacks, reasons


def _model_for_plan(
    *,
    goal: str,
    rows: int | None,
    budget: str,
    user_model_preference: str | None,
) -> tuple[
    ModelCandidate,
    str,
    list[SelectionRisk],
    list[SelectionFallback],
    list[SelectionReason],
]:
    warnings: list[SelectionRisk] = []
    fallbacks: list[SelectionFallback] = []
    reasons: list[SelectionReason] = []
    demo_model = first_model(lambda model: model.default_for_demo)

    if rows is not None and rows < 10_000:
        target_id = (
            "Qwen/Qwen2.5-1.5B-Instruct"
            if budget == "low"
            else "Qwen/Qwen2.5-3B-Instruct"
        )
    else:
        target_id = (
            "Qwen/Qwen2.5-3B-Instruct"
            if budget != "performance"
            else "mistralai/Mistral-7B-Instruct-v0.3"
        )

    if user_model_preference:
        requested = user_model_preference.strip()
        requested_model = catalog_model(requested)
        if requested_model and requested_model.gated:
            warnings.append(
                SelectionRisk(
                    "model_access",
                    "warning",
                    f"{requested} is gated; access is not assumed, so Qwen is recommended as the safe fallback.",
                )
            )
            fallbacks.append(
                SelectionFallback(
                    requested,
                    target_id,
                    "Llama/gated model access must be confirmed before it can be the default.",
                )
            )
        elif requested_model:
            reasons.append(
                SelectionReason(
                    "model", f"Using explicit model preference {requested}."
                )
            )
            return requested_model, requested, warnings, fallbacks, reasons
        else:
            model = ModelCandidate(
                model_id=requested,
                family=requested.split("/", 1)[0],
                parameter_count_b=0.0,
                license="unknown",
                access="unknown",
                notes=["User-provided model outside the static catalog."],
            )
            warnings.append(
                SelectionRisk(
                    "model",
                    "warning",
                    "User-provided model is outside the local catalog; verify architecture, license, tokenizer, and availability.",
                )
            )
            reasons.append(
                SelectionReason(
                    "model", f"Respecting user model preference {requested}."
                )
            )
            return model, requested, warnings, fallbacks, reasons

    if goal == "smoke-test" or (rows is not None and rows < 500):
        reasons.append(
            SelectionReason(
                "model",
                "Small dataset or smoke-test selected the smallest safe Qwen model.",
            )
        )
        return demo_model, demo_model.model_id, warnings, fallbacks, reasons

    model = catalog_model(target_id) or demo_model
    reasons.append(
        SelectionReason(
            "model", f"Selected {model.model_id} for dataset scale and budget."
        )
    )
    return model, model.model_id, warnings, fallbacks, reasons


def _quota_value(
    provider: str,
    hardware: HardwareCandidate,
    provider_readiness: dict[str, Any] | None,
) -> int | float | None:
    readiness = _readiness_for(provider, provider_readiness)
    quota = readiness.get("quota")
    if not isinstance(quota, dict):
        quota = readiness.get("quotas")
    if not isinstance(quota, dict):
        return None
    instance = hardware.hardware_args.get("instance_type") or hardware.display_name
    value = quota.get(str(instance))
    return (
        value
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _hardware_for_plan(
    *,
    provider: str,
    goal: str,
    model: ModelCandidate,
    provider_readiness: dict[str, Any] | None,
) -> tuple[
    HardwareCandidate,
    list[SelectionRisk],
    list[SelectionFallback],
    list[SelectionReason],
]:
    warnings: list[SelectionRisk] = []
    fallbacks: list[SelectionFallback] = []
    reasons: list[SelectionReason] = []
    if provider == "hf-jobs":
        if goal == "smoke-test" or model.parameter_count_b <= 3:
            hardware = first_hardware(
                provider, lambda item: item.hardware_id == "hf-jobs:t4-small"
            )
        else:
            hardware = first_hardware(
                provider, lambda item: item.hardware_id == "hf-jobs:a10g-largex2"
            )
    elif provider == "gcp-vertex":
        if goal == "smoke-test":
            hardware = first_hardware(
                provider, lambda item: item.hardware_id == "gcp-vertex:n1-standard-8-t4"
            )
        else:
            hardware = first_hardware(
                provider,
                lambda item: item.hardware_id == "gcp-vertex:n1-standard-16-l4",
            )
    elif provider == "aws-sagemaker":
        if goal == "smoke-test":
            hardware = first_hardware(
                provider,
                lambda item: item.hardware_id == "aws-sagemaker:ml.g4dn.xlarge",
            )
        else:
            preferred = first_hardware(
                provider, lambda item: item.hardware_id == "aws-sagemaker:ml.g5.xlarge"
            )
            quota = _quota_value(provider, preferred, provider_readiness)
            if quota == 0:
                hardware = first_hardware(
                    provider,
                    lambda item: item.hardware_id == "aws-sagemaker:ml.g4dn.xlarge",
                )
                warnings.append(
                    SelectionRisk(
                        "quota",
                        "warning",
                        "AWS quota for ml.g5.xlarge is 0; using ml.g4dn.xlarge fallback.",
                    )
                )
                fallbacks.append(
                    SelectionFallback(
                        "aws-sagemaker:ml.g5.xlarge",
                        "aws-sagemaker:ml.g4dn.xlarge",
                        "SageMaker quota for ml.g5.xlarge is unavailable.",
                    )
                )
            else:
                hardware = preferred
                if quota is None:
                    warnings.append(
                        SelectionRisk(
                            "quota",
                            "warning",
                            "AWS SageMaker quota is unknown; verify instance quota before approval.",
                        )
                    )
    else:
        hardware = first_hardware(
            "hf-jobs", lambda item: item.hardware_id == "hf-jobs:t4-small"
        )
    reasons.append(
        SelectionReason("hardware", f"Selected {hardware.display_name} for {provider}.")
    )
    return hardware, warnings, fallbacks, reasons


def _output_policy_for_plan(
    *,
    provider: str,
    domain: str,
    privacy: str,
    requested_output_policy: str | None,
) -> tuple[str, list[SelectionRisk], list[SelectionReason]]:
    warnings: list[SelectionRisk] = []
    reasons: list[SelectionReason] = []
    requested = (requested_output_policy or "").strip().lower()
    if privacy == "sensitive":
        if requested in {OUTPUT_POLICY_HF_HUB, OUTPUT_POLICY_CLOUD_AND_HF_HUB}:
            warnings.append(
                SelectionRisk(
                    "privacy",
                    "warning",
                    "Sensitive domain detected; overriding Hub-oriented output to cloud-private unless explicitly approved later.",
                )
            )
        reasons.append(
            SelectionReason(
                "output_policy", "Sensitive/private domain defaults to cloud-private."
            )
        )
        return OUTPUT_POLICY_CLOUD_PRIVATE, warnings, reasons
    if requested in {
        OUTPUT_POLICY_CLOUD_PRIVATE,
        OUTPUT_POLICY_HF_HUB,
        OUTPUT_POLICY_CLOUD_AND_HF_HUB,
    }:
        reasons.append(
            SelectionReason(
                "output_policy", f"Using requested output policy {requested}."
            )
        )
        return requested, warnings, reasons
    if provider == "hf-jobs":
        policy = OUTPUT_POLICY_CLOUD_AND_HF_HUB
    elif provider in {"gcp-vertex", "aws-sagemaker"}:
        policy = OUTPUT_POLICY_CLOUD_PRIVATE
    else:
        policy = default_output_policy_for_domain(domain, provider)
    reasons.append(
        SelectionReason("output_policy", f"Selected {policy} for provider and domain.")
    )
    return policy, warnings, reasons


def _with_runtime_args(
    hardware: HardwareCandidate, training_args: dict[str, Any]
) -> dict[str, Any]:
    if "max_run_seconds" in hardware.hardware_args and "max_run_hours" in training_args:
        args = dict(training_args)
        args["max_run_seconds"] = int(float(training_args["max_run_hours"]) * 3600)
        return args
    return training_args


def _production_alternative(
    provider: str, model: ModelCandidate
) -> dict[str, Any] | None:
    hardware = next(
        (
            item
            for item in hardware_catalog()
            if item.provider_id == provider and item.production_suitable
        ),
        None,
    )
    model_id = (
        "Qwen/Qwen2.5-3B-Instruct"
        if model.parameter_count_b < 3
        else "mistralai/Mistral-7B-Instruct-v0.3"
    )
    return {
        "model_id": model_id,
        "hardware_id": hardware.hardware_id if hardware else None,
        "reason": "Higher-quality production alternative after the pilot is validated.",
    }


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
    requested_output_policy: str | None = None,
    intent_hint: str | None = None,
    full_dataset_approved: bool = False,
    dataset_discovery: dict[str, Any] | None = None,
    budget_cap_usd: float | None = None,
    provider_readiness: dict[str, Any] | None = None,
    explicit_provider: bool = True,
) -> TrainingPlan:
    requested_provider = normalize_provider(provider)
    normalized_domain = normalize_domain(domain)
    normalized_task = (task_type or "sft").strip().lower()
    budget = normalize_budget_preference(budget_preference)
    rows = _dataset_rows(dataset_summary)
    goal, reasoning = _choose_training_goal(
        training_goal,
        dataset_rows=rows,
        intent_hint=intent_hint,
    )

    provider_warnings, provider_fallbacks, selection_reasons = [], [], []
    normalized_provider, provider_warnings, provider_fallbacks, selection_reasons = (
        _select_provider(
            requested_provider,
            explicit_provider=explicit_provider,
            provider_readiness=provider_readiness,
        )
    )

    privacy = detect_privacy_level(normalized_domain, privacy_level)
    output_policy, output_warnings, output_reasons = _output_policy_for_plan(
        provider=normalized_provider,
        domain=normalized_domain,
        privacy=privacy,
        requested_output_policy=requested_output_policy,
    )
    _, privacy_warnings = _privacy_notes(
        normalized_provider,
        normalized_domain,
        privacy,
    )

    (
        selected_model,
        recommended_model,
        model_warnings,
        model_fallbacks,
        model_reasons,
    ) = _model_for_plan(
        goal=goal,
        rows=rows,
        budget=budget,
        user_model_preference=user_model_preference,
    )
    production_model = (
        "Qwen/Qwen2.5-3B-Instruct"
        if budget in {"low", "balanced"}
        else "mistralai/Mistral-7B-Instruct-v0.3"
    )
    selected_hardware, hardware_warnings, hardware_fallbacks, hardware_reasons = (
        _hardware_for_plan(
            provider=normalized_provider,
            goal=goal,
            model=selected_model,
            provider_readiness=provider_readiness,
        )
    )
    risks: list[str] = []
    if user_model_preference and recommended_model == user_model_preference.strip():
        recommended_model = user_model_preference.strip()
        risks.extend(_model_risks(recommended_model, goal))

    if uploaded_dataset_available is False or rows is None:
        risks.append(
            "No training dataset summary is available; dataset discovery is required before final training plan approval."
        )
        reasoning.append(
            "Run dataset_discovery first, then search allowed public sources, inspect schema/license/privacy, and do not launch a cloud job until the user selects a dataset."
        )
        if isinstance(dataset_discovery, dict):
            recommended = dataset_discovery.get("recommended_candidate")
            if isinstance(recommended, dict):
                candidate_name = (
                    recommended.get("title")
                    or recommended.get("dataset_id")
                    or "the recommended dataset candidate"
                )
                reasoning.append(
                    f"Dataset discovery recommends {candidate_name}; user confirmation is still required before launch."
                )
            warnings = dataset_discovery.get("warnings")
            if isinstance(warnings, list):
                risks.extend(str(warning) for warning in warnings if warning)
            risks.append(
                "Discovered datasets are recommendations only; do not treat them as already selected or available."
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
        if rows > 50_000 and not full_dataset_approved:
            risks.append(
                f"Dataset has about {rows:,} rows; use capped pilot samples before full-dataset training."
            )
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

    training_args = _training_args(
        goal,
        rows,
        budget,
        normalized_provider,
        full_dataset_approved=full_dataset_approved,
    )
    training_args = _with_runtime_args(selected_hardware, training_args)
    estimated_cost = estimate_plan_cost(selected_hardware, training_args)
    recommendation_warnings = [
        *provider_warnings,
        *model_warnings,
        *hardware_warnings,
        *output_warnings,
    ]
    if (
        budget_cap_usd is not None
        and estimated_cost is not None
        and estimated_cost > budget_cap_usd
    ):
        recommendation_warnings.append(
            SelectionRisk(
                "budget",
                "warning",
                f"Estimated cost ${estimated_cost:.2f} exceeds budget cap ${budget_cap_usd:.2f}.",
            )
        )
    elif rows is not None and rows > 50_000:
        recommendation_warnings.append(
            SelectionRisk(
                "budget",
                "warning",
                "Large dataset may increase time and cost; planner caps samples for the pilot.",
            )
        )
    if privacy == "sensitive":
        recommendation_warnings.append(
            SelectionRisk(
                "safety",
                "warning",
                "Sensitive domain requires post-training safety eval and human review.",
            )
        )
    selected_provider = catalog_provider(normalized_provider) or catalog_provider(
        "hf-jobs"
    )
    if selected_provider is None:
        raise RuntimeError("Static provider catalog is missing hf-jobs.")
    recommendation = TrainingPlanRecommendation(
        selected_model=selected_model,
        selected_provider=selected_provider,
        selected_hardware=selected_hardware,
        output_policy=output_policy,
        training_args=training_args,
        estimated_cost_usd=estimated_cost,
        budget_cap_usd=budget_cap_usd,
        confidence=0.82 if not recommendation_warnings else 0.72,
        reasons=[
            *selection_reasons,
            *model_reasons,
            *hardware_reasons,
            *output_reasons,
            SelectionReason(
                "dataset", f"Dataset rows: {rows if rows is not None else 'unknown'}."
            ),
        ],
        warnings=recommendation_warnings,
        fallbacks=[*provider_fallbacks, *model_fallbacks, *hardware_fallbacks],
        alternatives=[
            {"model_id": item.model_id, "family": item.family}
            for item in (
                catalog_model("Qwen/Qwen2.5-1.5B-Instruct"),
                catalog_model("Qwen/Qwen2.5-3B-Instruct"),
            )
            if item and item.model_id != recommended_model
        ],
        production_alternative=_production_alternative(
            normalized_provider, selected_model
        ),
        quota_warning_recorded=any(
            w.category == "quota" for w in recommendation_warnings
        ),
        access_warning_recorded=any(
            w.category == "model_access" for w in recommendation_warnings
        ),
        recommended_evaluation_profile=(
            "safety_privacy_review"
            if privacy == "sensitive"
            else "standard_static_review"
        ),
    )

    return TrainingPlan(
        provider=normalized_provider,
        domain=normalized_domain,
        task_type=normalized_task,
        training_goal=goal,
        recommended_model=recommended_model,
        smoke_test_model=SMOKE_TEST_MODEL,
        production_model=production_model,
        recommended_hardware=dict(selected_hardware.hardware_args),
        training_args=training_args,
        output_policy=output_policy,
        privacy_warnings=privacy_warnings,
        risks=risks + [warning.message for warning in recommendation_warnings],
        reasoning=reasoning + [reason.message for reason in recommendation.reasons],
        dataset_discovery=dataset_discovery
        if isinstance(dataset_discovery, dict)
        else None,
        recommendation=recommendation,
    )
