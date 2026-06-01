"""Shared output policy semantics for training planners and provider UI.

These helpers are intentionally declarative. They describe where artifacts
should land, but never launch jobs, call cloud APIs, or verify provider state.
"""

from __future__ import annotations

OUTPUT_POLICY_CLOUD_PRIVATE = "cloud-private"
OUTPUT_POLICY_HF_HUB = "hf-hub"
OUTPUT_POLICY_CLOUD_AND_HF_HUB = "cloud-and-hf-hub"

VALID_OUTPUT_POLICIES = {
    OUTPUT_POLICY_CLOUD_PRIVATE,
    OUTPUT_POLICY_HF_HUB,
    OUTPUT_POLICY_CLOUD_AND_HF_HUB,
}

SENSITIVE_DOMAIN_KEYWORDS = {
    "medical",
    "healthcare",
    "health",
    "finance",
    "financial",
    "banking",
    "bank",
    "insurance",
    "legal",
    "government",
    "gov",
    "internal",
    "company",
}


def validate_output_policy(
    value: str | None,
    default: str = OUTPUT_POLICY_CLOUD_AND_HF_HUB,
) -> str:
    """Return a known output policy, falling back to a known default."""

    normalized = (value or "").strip().lower()
    if normalized in VALID_OUTPUT_POLICIES:
        return normalized
    if default in VALID_OUTPUT_POLICIES:
        return default
    return OUTPUT_POLICY_CLOUD_AND_HF_HUB


def cloud_storage_label(provider: str) -> str:
    normalized = (provider or "").strip().lower()
    if normalized == "gcp-vertex":
        return "Google Cloud Storage"
    if normalized == "aws-sagemaker":
        return "Amazon S3"
    if normalized == "hf-jobs":
        return "Hugging Face Hub / job artifacts"
    return "provider-native private storage"


def output_policy_label(provider: str, policy: str) -> str:
    normalized_provider = (provider or "").strip().lower()
    normalized_policy = validate_output_policy(policy)

    if normalized_provider == "hf-jobs":
        if normalized_policy == OUTPUT_POLICY_CLOUD_PRIVATE:
            return "Private Hugging Face job/model artifacts"
        if normalized_policy == OUTPUT_POLICY_HF_HUB:
            return "Hugging Face Hub"
        return "Hugging Face Hub and job artifacts"

    storage_label = cloud_storage_label(normalized_provider)
    if normalized_policy == OUTPUT_POLICY_CLOUD_PRIVATE:
        return f"{storage_label} only"
    if normalized_policy == OUTPUT_POLICY_HF_HUB:
        return "Hugging Face Hub only"
    return f"Both {storage_label} and Hugging Face Hub"


def output_policy_requires_hub(policy: str) -> bool:
    return validate_output_policy(policy) in {
        OUTPUT_POLICY_HF_HUB,
        OUTPUT_POLICY_CLOUD_AND_HF_HUB,
    }


def output_policy_requires_cloud_storage(policy: str) -> bool:
    return validate_output_policy(policy) in {
        OUTPUT_POLICY_CLOUD_PRIVATE,
        OUTPUT_POLICY_CLOUD_AND_HF_HUB,
    }


def _domain_tokens(domain: str | None) -> set[str]:
    normalized = (domain or "general").strip().lower().replace("-", "_")
    spaced = normalized.replace("_", " ")
    return {normalized, spaced, *spaced.split()}


def is_sensitive_domain(domain: str | None) -> bool:
    return bool(_domain_tokens(domain) & SENSITIVE_DOMAIN_KEYWORDS)


def default_output_policy_for_domain(domain: str | None, provider: str) -> str:
    normalized_provider = (provider or "").strip().lower()
    if normalized_provider == "gcp-vertex":
        return OUTPUT_POLICY_CLOUD_PRIVATE
    if is_sensitive_domain(domain):
        return OUTPUT_POLICY_CLOUD_PRIVATE
    return OUTPUT_POLICY_CLOUD_AND_HF_HUB


def privacy_warning_for_policy(provider: str, policy: str) -> str | None:
    normalized_provider = (provider or "").strip().lower()
    normalized_policy = validate_output_policy(policy)
    if (
        normalized_provider == "hf-jobs"
        and normalized_policy == OUTPUT_POLICY_CLOUD_PRIVATE
    ):
        return (
            "For HF Jobs, cloud-private depends on private Hub repository and "
            "job artifact settings; verify privacy before any launch."
        )
    return None
