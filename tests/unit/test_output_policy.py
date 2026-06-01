from agent.core.output_policy import (
    OUTPUT_POLICY_CLOUD_AND_HF_HUB,
    OUTPUT_POLICY_CLOUD_PRIVATE,
    OUTPUT_POLICY_HF_HUB,
    cloud_storage_label,
    default_output_policy_for_domain,
    output_policy_label,
    output_policy_requires_cloud_storage,
    output_policy_requires_hub,
    privacy_warning_for_policy,
    validate_output_policy,
)


def test_valid_output_policies_are_accepted():
    assert validate_output_policy("cloud-private") == OUTPUT_POLICY_CLOUD_PRIVATE
    assert validate_output_policy("hf-hub") == OUTPUT_POLICY_HF_HUB
    assert validate_output_policy("cloud-and-hf-hub") == OUTPUT_POLICY_CLOUD_AND_HF_HUB


def test_invalid_output_policy_defaults_to_requested_default():
    assert validate_output_policy(None) == OUTPUT_POLICY_CLOUD_AND_HF_HUB
    assert validate_output_policy("invalid") == OUTPUT_POLICY_CLOUD_AND_HF_HUB
    assert (
        validate_output_policy("invalid", default="cloud-private")
        == OUTPUT_POLICY_CLOUD_PRIVATE
    )


def test_output_policy_destination_requirements():
    assert output_policy_requires_hub("cloud-private") is False
    assert output_policy_requires_hub("hf-hub") is True
    assert output_policy_requires_hub("cloud-and-hf-hub") is True

    assert output_policy_requires_cloud_storage("cloud-private") is True
    assert output_policy_requires_cloud_storage("hf-hub") is False
    assert output_policy_requires_cloud_storage("cloud-and-hf-hub") is True


def test_provider_cloud_storage_labels_are_specific():
    assert cloud_storage_label("gcp-vertex") == "Google Cloud Storage"
    assert cloud_storage_label("aws-sagemaker") == "Amazon S3"
    assert cloud_storage_label("hf-jobs") == "Hugging Face Hub / job artifacts"


def test_provider_output_policy_labels_are_specific():
    assert (
        output_policy_label("gcp-vertex", "cloud-private")
        == "Google Cloud Storage only"
    )
    assert output_policy_label("aws-sagemaker", "cloud-private") == "Amazon S3 only"
    assert (
        output_policy_label("hf-jobs", "cloud-private")
        == "Private Hugging Face job/model artifacts"
    )
    assert (
        output_policy_label("aws-sagemaker", "cloud-and-hf-hub")
        == "Both Amazon S3 and Hugging Face Hub"
    )


def test_sensitive_domains_default_to_cloud_private():
    for domain in [
        "medical",
        "healthcare",
        "finance",
        "banking",
        "insurance",
        "legal",
        "government",
        "internal company",
    ]:
        assert default_output_policy_for_domain(domain, "gcp-vertex") == "cloud-private"


def test_gcp_vertex_defaults_to_cloud_private_even_for_general_domains():
    assert (
        default_output_policy_for_domain("customer_support", "gcp-vertex")
        == "cloud-private"
    )


def test_general_hf_domains_default_to_cloud_and_hf_hub():
    assert default_output_policy_for_domain("general", "hf-jobs") == "cloud-and-hf-hub"


def test_hf_cloud_private_warning_mentions_private_hub_and_artifacts():
    warning = privacy_warning_for_policy("hf-jobs", "cloud-private")

    assert warning
    assert "private Hub" in warning
    assert "job artifact" in warning
