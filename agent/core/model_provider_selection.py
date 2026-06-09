"""Static model, provider, hardware, and fallback selection helpers.

The catalog is intentionally curated and offline. It does not probe live model
access, provider quota, hardware availability, billing APIs, or benchmarks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SelectionReason:
    category: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionRisk:
    category: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionFallback:
    blocked_option: str
    fallback_option: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelCandidate:
    model_id: str
    family: str
    parameter_count_b: float
    license: str
    access: str = "open"
    gated: bool = False
    default_for_demo: bool = False
    production_suitable: bool = False
    sensitive_domain_notes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderCandidate:
    provider_id: str
    display_name: str
    readiness_required: bool = False
    readiness_status: str = "unknown"
    quota_status: str = "unknown"
    default_output_policy: str = "cloud-and-hf-hub"
    supports_private_output: bool = True
    supports_hub_output: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HardwareCandidate:
    hardware_id: str
    provider_id: str
    display_name: str
    hardware_args: dict[str, Any]
    estimated_hourly_cost_usd: float | None
    gpu_memory_gb: int | None = None
    suitable_model_size_b: float | None = None
    demo_suitable: bool = False
    production_suitable: bool = False
    quota_risk: str = "unknown"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingPlanRecommendation:
    selected_model: ModelCandidate
    selected_provider: ProviderCandidate
    selected_hardware: HardwareCandidate
    output_policy: str
    training_args: dict[str, Any]
    estimated_cost_usd: float | None
    budget_cap_usd: float | None = None
    confidence: float = 0.7
    reasons: list[SelectionReason] = field(default_factory=list)
    warnings: list[SelectionRisk] = field(default_factory=list)
    fallbacks: list[SelectionFallback] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    production_alternative: dict[str, Any] | None = None
    approval_required: bool = True
    quota_warning_recorded: bool = False
    access_warning_recorded: bool = False
    recommended_evaluation_profile: str = "standard_static_review"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_model"] = self.selected_model.to_dict()
        payload["selected_provider"] = self.selected_provider.to_dict()
        payload["selected_hardware"] = self.selected_hardware.to_dict()
        payload["reasons"] = [reason.to_dict() for reason in self.reasons]
        payload["warnings"] = [warning.to_dict() for warning in self.warnings]
        payload["fallbacks"] = [fallback.to_dict() for fallback in self.fallbacks]
        return payload


def model_catalog() -> list[ModelCandidate]:
    return [
        ModelCandidate(
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen",
            0.5,
            "apache-2.0",
            default_for_demo=True,
            notes=["Default demo and smoke-test model."],
        ),
        ModelCandidate(
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen",
            1.5,
            "apache-2.0",
            production_suitable=True,
            notes=["Medium low-cost production pilot option."],
        ),
        ModelCandidate(
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen",
            3.0,
            "apache-2.0",
            production_suitable=True,
            notes=["Balanced production pilot option when budget allows."],
        ),
        ModelCandidate(
            "meta-llama/Llama-3.2-1B-Instruct",
            "Llama",
            1.0,
            "llama-3.2-community",
            access="gated",
            gated=True,
            notes=[
                "Use only when the user's token has access and license fit is confirmed."
            ],
        ),
        ModelCandidate(
            "meta-llama/Llama-3.2-3B-Instruct",
            "Llama",
            3.0,
            "llama-3.2-community",
            access="gated",
            gated=True,
            production_suitable=True,
            notes=["Not a default because access may be gated."],
        ),
        ModelCandidate(
            "mistralai/Mistral-7B-Instruct-v0.3",
            "Mistral",
            7.0,
            "apache-2.0",
            production_suitable=True,
            notes=["Larger production option; verify memory and cost before launch."],
        ),
        ModelCandidate(
            "google/gemma-2-2b-it",
            "Gemma",
            2.0,
            "gemma",
            access="license-reviewed",
            production_suitable=True,
            notes=["Verify Gemma license terms for the intended use before training."],
        ),
    ]


def provider_catalog() -> list[ProviderCandidate]:
    return [
        ProviderCandidate(
            "hf-jobs",
            "Hugging Face Jobs",
            default_output_policy="cloud-and-hf-hub",
            notes=["Good default for demos and Hub artifact workflows."],
        ),
        ProviderCandidate(
            "gcp-vertex",
            "Google Cloud Vertex AI",
            readiness_required=True,
            default_output_policy="cloud-private",
            notes=[
                "Recommend only when GCloud readiness is configured or explicitly chosen."
            ],
        ),
        ProviderCandidate(
            "aws-sagemaker",
            "AWS SageMaker AI",
            readiness_required=True,
            default_output_policy="cloud-private",
            notes=[
                "Quota may be unknown; prefer g4dn fallback when g5 quota is unavailable."
            ],
        ),
    ]


def hardware_catalog() -> list[HardwareCandidate]:
    return [
        HardwareCandidate(
            "hf-jobs:t4-small",
            "hf-jobs",
            "t4-small",
            {"hardware_flavor": "t4-small"},
            0.60,
            gpu_memory_gb=16,
            suitable_model_size_b=3,
            demo_suitable=True,
            quota_risk="unknown",
        ),
        HardwareCandidate(
            "hf-jobs:a10g-small",
            "hf-jobs",
            "a10g-small",
            {"hardware_flavor": "a10g-small"},
            1.00,
            gpu_memory_gb=24,
            suitable_model_size_b=7,
            demo_suitable=True,
            production_suitable=True,
            quota_risk="unknown",
        ),
        HardwareCandidate(
            "hf-jobs:a10g-largex2",
            "hf-jobs",
            "a10g-largex2",
            {"hardware_flavor": "a10g-largex2"},
            4.00,
            gpu_memory_gb=48,
            suitable_model_size_b=13,
            production_suitable=True,
            quota_risk="unknown",
        ),
        HardwareCandidate(
            "gcp-vertex:n1-standard-8-t4",
            "gcp-vertex",
            "n1-standard-8 + T4",
            {
                "machine_type": "n1-standard-8",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            1.10,
            gpu_memory_gb=16,
            suitable_model_size_b=3,
            demo_suitable=True,
            quota_risk="unknown",
        ),
        HardwareCandidate(
            "gcp-vertex:n1-standard-16-l4",
            "gcp-vertex",
            "n1-standard-16 + L4",
            {
                "machine_type": "n1-standard-16",
                "accelerator_type": "NVIDIA_L4",
                "accelerator_count": 1,
            },
            1.90,
            gpu_memory_gb=24,
            suitable_model_size_b=7,
            production_suitable=True,
            quota_risk="unknown",
            notes=["Compatibility default for existing Vertex production plans."],
        ),
        HardwareCandidate(
            "gcp-vertex:g2-standard-16-l4",
            "gcp-vertex",
            "g2-standard-16 + L4",
            {
                "machine_type": "g2-standard-16",
                "accelerator_type": "NVIDIA_L4",
                "accelerator_count": 1,
            },
            1.90,
            gpu_memory_gb=24,
            suitable_model_size_b=7,
            production_suitable=True,
            quota_risk="unknown",
        ),
        HardwareCandidate(
            "gcp-vertex:a2-highgpu-1g-a100",
            "gcp-vertex",
            "a2-highgpu-1g + A100",
            {
                "machine_type": "a2-highgpu-1g",
                "accelerator_type": "NVIDIA_TESLA_A100",
                "accelerator_count": 1,
            },
            8.20,
            gpu_memory_gb=40,
            suitable_model_size_b=13,
            production_suitable=True,
            quota_risk="high-cost",
        ),
        HardwareCandidate(
            "aws-sagemaker:ml.g4dn.xlarge",
            "aws-sagemaker",
            "ml.g4dn.xlarge",
            {
                "instance_type": "ml.g4dn.xlarge",
                "instance_count": 1,
                "max_run_seconds": 3600,
            },
            0.90,
            gpu_memory_gb=16,
            suitable_model_size_b=3,
            demo_suitable=True,
            production_suitable=True,
            quota_risk="lower",
        ),
        HardwareCandidate(
            "aws-sagemaker:ml.g5.xlarge",
            "aws-sagemaker",
            "ml.g5.xlarge",
            {
                "instance_type": "ml.g5.xlarge",
                "instance_count": 1,
                "max_run_seconds": 3600,
            },
            1.50,
            gpu_memory_gb=24,
            suitable_model_size_b=7,
            demo_suitable=True,
            production_suitable=True,
            quota_risk="may-be-zero",
        ),
        HardwareCandidate(
            "aws-sagemaker:ml.g5.2xlarge",
            "aws-sagemaker",
            "ml.g5.2xlarge",
            {
                "instance_type": "ml.g5.2xlarge",
                "instance_count": 1,
                "max_run_seconds": 7200,
            },
            2.00,
            gpu_memory_gb=24,
            suitable_model_size_b=7,
            production_suitable=True,
            quota_risk="unknown",
        ),
        HardwareCandidate(
            "aws-sagemaker:ml.m5.xlarge",
            "aws-sagemaker",
            "ml.m5.xlarge",
            {
                "instance_type": "ml.m5.xlarge",
                "instance_count": 1,
                "max_run_seconds": 1800,
            },
            0.40,
            suitable_model_size_b=0.5,
            demo_suitable=True,
            quota_risk="low",
            notes=["CPU fallback for smoke tests only; not recommended for real SFT."],
        ),
    ]


def catalog_model(model_id: str) -> ModelCandidate | None:
    return next(
        (model for model in model_catalog() if model.model_id == model_id), None
    )


def catalog_provider(provider_id: str) -> ProviderCandidate | None:
    return next(
        (
            provider
            for provider in provider_catalog()
            if provider.provider_id == provider_id
        ),
        None,
    )


def catalog_hardware(hardware_id: str) -> HardwareCandidate | None:
    return next(
        (
            hardware
            for hardware in hardware_catalog()
            if hardware.hardware_id == hardware_id
        ),
        None,
    )


def first_model(predicate) -> ModelCandidate:
    return next(model for model in model_catalog() if predicate(model))


def first_hardware(provider_id: str, predicate) -> HardwareCandidate:
    return next(
        hardware
        for hardware in hardware_catalog()
        if hardware.provider_id == provider_id and predicate(hardware)
    )


def estimate_plan_cost(
    hardware: HardwareCandidate,
    training_args: dict[str, Any],
) -> float | None:
    if hardware.estimated_hourly_cost_usd is None:
        return None
    hours = training_args.get("max_run_hours")
    if hours is None and training_args.get("max_run_seconds"):
        hours = float(training_args["max_run_seconds"]) / 3600
    if hours is None:
        hours = float(hardware.hardware_args.get("max_run_seconds") or 3600) / 3600
    try:
        parsed_hours = float(hours)
    except (TypeError, ValueError):
        return None
    return round(hardware.estimated_hourly_cost_usd * max(parsed_hours, 0.0), 4)
