from agent.training_templates.validation import validate_sft_template_request
from agent.training_templates.verification import (
    classify_gcs_artifacts,
    classify_hf_model_files,
)


def test_validate_sft_template_request_requires_core_fields():
    errors = validate_sft_template_request({})

    assert "dataset_name is required" in errors
    assert "model_name is required" in errors
    assert "hub_model_id is required" in errors


def test_validate_sft_template_request_allows_cloud_private_without_hub_target():
    errors = validate_sft_template_request(
        {
            "dataset_name": "trl-lib/Capybara",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_policy": "cloud-private",
        }
    )

    assert "hub_model_id is required" not in errors


def test_validate_sft_template_request_requires_hub_target_for_hub_outputs():
    hf_only = validate_sft_template_request(
        {
            "dataset_name": "trl-lib/Capybara",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_policy": "hf-hub",
        }
    )
    both = validate_sft_template_request(
        {
            "dataset_name": "trl-lib/Capybara",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_policy": "cloud-and-hf-hub",
        }
    )

    assert "hub_model_id is required" in hf_only
    assert "hub_model_id is required" in both


def test_validate_sft_template_request_rejects_invalid_goal_and_output_policy():
    errors = validate_sft_template_request(
        {
            "dataset_name": "trl-lib/Capybara",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "hub_model_id": "ligaments-dev/test-model",
            "training_goal": "long-benchmark",
            "output_policy": "dropbox",
        }
    )

    assert (
        "training_goal must be one of: smoke-test, production, agent-decide" in errors
    )
    assert (
        "output_policy must be one of: cloud-private, hf-hub, cloud-and-hf-hub"
        in errors
    )


def test_validate_sft_template_request_rejects_risky_defaults():
    errors = validate_sft_template_request(
        {
            "dataset_name": "trl-lib/Capybara",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "hub_model_id": "ligaments-dev/test-model",
            "packing": True,
            "attn_implementation": "kernels-community/flash-attn2",
        }
    )

    assert "packing=True is not allowed for the stable Vertex SFT template" in errors
    assert (
        "attn_implementation is not allowed for the stable Vertex SFT template"
        in errors
    )


def test_validate_sft_template_request_rejects_invalid_numeric_fields():
    errors = validate_sft_template_request(
        {
            "dataset_name": "trl-lib/Capybara",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "hub_model_id": "ligaments-dev/test-model",
            "num_train_epochs": 0,
            "max_length": -1,
            "learning_rate": 0,
            "per_device_train_batch_size": 0,
            "gradient_accumulation_steps": 0,
            "max_train_samples": 0,
            "max_eval_samples": -5,
            "validation_split_ratio": 1,
        }
    )

    assert "num_train_epochs must be positive" in errors
    assert "max_length must be positive" in errors
    assert "learning_rate must be positive" in errors
    assert "per_device_train_batch_size must be positive" in errors
    assert "gradient_accumulation_steps must be positive" in errors
    assert "max_train_samples must be positive when provided" in errors
    assert "max_eval_samples must be positive when provided" in errors
    assert "validation_split_ratio must be greater than 0 and less than 1" in errors


def test_validate_sft_template_request_validates_column_mapping_shape():
    not_dict_errors = validate_sft_template_request(
        {
            "dataset_name": "trl-lib/Capybara",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "hub_model_id": "ligaments-dev/test-model",
            "column_mapping": ["not", "a", "dict"],
        }
    )
    bad_assistant_errors = validate_sft_template_request(
        {
            "dataset_name": "trl-lib/Capybara",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "hub_model_id": "ligaments-dev/test-model",
            "column_mapping": {"assistant": ["answer", ""]},
        }
    )

    assert "column_mapping must be an object" in not_dict_errors
    assert (
        "column_mapping.assistant must be a string or a list of non-empty strings"
        in bad_assistant_errors
    )


def test_validate_sft_template_request_rejects_dummy_vertex_runs():
    errors = validate_sft_template_request(
        {
            "display_name": "dummy-test-template",
            "dataset_name": "FreedomIntelligence/medical-o1-reasoning-SFT",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "hub_model_id": "ligaments-dev/dummy-test-model",
        }
    )

    assert "dummy placeholder values are not allowed for Vertex training jobs" in errors


def test_validate_sft_template_request_rejects_obvious_domain_mismatch():
    errors = validate_sft_template_request(
        {
            "display_name": "finance-sft",
            "dataset_name": "FreedomIntelligence/medical-o1-reasoning-SFT",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "hub_model_id": "ligaments-dev/finance-qwen-sft",
            "trackio_project": "finance-sft",
        }
    )

    assert "finance training jobs cannot use an obviously medical dataset" in errors


def test_classify_hf_model_files_requires_weights_and_tokenizer():
    usable = classify_hf_model_files(
        ["config.json", "model.safetensors", "tokenizer.json", "README.md"]
    )
    readme_only = classify_hf_model_files(["README.md", ".gitattributes"])

    assert usable.is_usable is True
    assert readme_only.is_usable is False
    assert "model weights" in readme_only.reason


def test_classify_gcs_artifacts_requires_model_files():
    usable = classify_gcs_artifacts(
        [
            "vertex-outputs/job/final/config.json",
            "vertex-outputs/job/final/model.safetensors",
            "vertex-outputs/job/final/tokenizer.json",
        ]
    )
    empty = classify_gcs_artifacts([])

    assert usable.is_usable is True
    assert empty.is_usable is False
    assert "No GCS artifacts" in empty.reason
