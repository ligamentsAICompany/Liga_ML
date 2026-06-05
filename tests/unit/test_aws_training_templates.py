import ast

import pytest

from agent.training_templates.aws_sft import (
    AwsSftTemplateConfig,
    DEFAULT_AWS_PACKAGES,
    build_aws_sft_training_script,
)
from agent.training_templates.aws_validation import validate_aws_sft_template_request


def _script(output_policy: str = "aws-private") -> str:
    return build_aws_sft_training_script(
        AwsSftTemplateConfig(
            dataset_split="train",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            output_model_id="owner/aws-output",
            output_policy=output_policy,
            hub_model_id="owner/aws-output" if output_policy != "aws-private" else None,
            column_mapping={"user": "question", "assistant": ["reasoning", "answer"]},
        )
    )


def test_aws_sft_template_generates_parseable_sagemaker_script_contract():
    script = _script()

    ast.parse(script)
    assert (
        'TRAIN_CHANNEL_DIR = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))'
        in script
    )
    assert 'MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))' in script
    assert (
        'OUTPUT_DATA_DIR = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))'
        in script
    )
    assert 'TRAIN_FILE = TRAIN_CHANNEL_DIR / "train.jsonl"' in script
    assert 'RESULT_FILE_NAME = "liga_training_result.json"' in script
    assert "result_path = MODEL_DIR / RESULT_FILE_NAME" in script
    assert "output_result_path = OUTPUT_DATA_DIR / RESULT_FILE_NAME" in script
    assert 'metrics_path = OUTPUT_DATA_DIR / "metrics.json"' in script
    assert 'provider": "aws-sagemaker"' in script
    assert '"eval_result": eval_metrics' in script
    assert '"training_args":' in script
    assert '"dataset_name": DATASET_NAME' in script
    assert '"model_name": MODEL_NAME' in script


def test_aws_sft_template_has_final_markers_and_dependency_skip_flag():
    script = _script("cloud-and-hf-hub")

    assert 'os.environ.get("LIGA_ML_SKIP_DEP_INSTALL") == "1"' in script
    for marker in [
        "LIGA_TRAINING_STATUS=succeeded",
        "LIGA_PROVIDER=aws-sagemaker",
        "LIGA_AWS_TRAINING_JOB_NAME=",
        "LIGA_AWS_REGION=",
        "LIGA_S3_MODEL_ARTIFACT=",
        "LIGA_S3_OUTPUT_DIR=",
        "LIGA_CLOUDWATCH_LOGS_URL=",
        "LIGA_FINAL_MODEL_URL=",
        "LIGA_HUB_MODEL_ID=",
        "LIGA_EVAL_RESULT_JSON=",
        "LIGA_RESULT_FILE=",
        "LIGA_TRAIN_ROWS=",
        "LIGA_EVAL_ROWS=",
        "LIGA_DATASET_SOURCE=",
    ]:
        assert marker in script


def test_aws_sft_template_disables_torchvision_before_transformers_imports():
    script = _script()

    disable_index = script.index('TRANSFORMERS_NO_TORCHVISION", "1"')
    transformers_index = script.index("from transformers import AutoModelForCausalLM")
    trl_index = script.index("from trl import SFTConfig, SFTTrainer")

    assert disable_index < transformers_index
    assert disable_index < trl_index
    assert 'HF_HUB_DISABLE_TELEMETRY", "1"' in script


def test_aws_sft_template_does_not_import_or_install_torchvision_directly():
    script = _script()
    tree = ast.parse(script)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "torchvision" not in imported_modules
    assert "torchvision" not in imported_from_modules
    assert '"torch==' not in script
    assert "torchvision" not in "\n".join(DEFAULT_AWS_PACKAGES)


def test_aws_sft_template_prints_dependency_sanity_info():
    script = _script()

    assert "Python version:" in script
    assert "torch version:" in script
    assert "torch cuda available:" in script
    assert "transformers version:" in script
    assert "datasets version:" in script
    assert "trl version:" in script
    assert "peft version:" in script


def test_aws_sft_template_treats_torchvision_incompatibility_as_text_only_non_fatal():
    script = _script()

    assert (
        "torchvision unavailable or incompatible; continuing because this is text-only SFT."
        in script
    )
    assert "except Exception as exc" in script
    assert "check_optional_torchvision()" in script


def test_aws_sft_template_output_policy_token_behavior_is_runtime_checked():
    private_script = _script("aws-private")
    hub_script = _script("hf-hub")
    cloud_hub_script = _script("cloud-and-hf-hub")

    assert 'OUTPUT_POLICY = "aws-private"' in private_script
    assert "requires HF_TOKEN or HUGGINGFACE_HUB_TOKEN" in hub_script
    assert "requires HF_TOKEN or HUGGINGFACE_HUB_TOKEN" in cloud_hub_script
    assert (
        'if OUTPUT_POLICY in {"hf-hub", "cloud-and-hf-hub"} and not HF_TOKEN:'
        in hub_script
    )
    assert "trainer.push_to_hub()" in hub_script
    assert "trainer.push_to_hub()" in cloud_hub_script
    assert "push_to_hub=OUTPUT_POLICY in PUBLISH_TO_HUB_POLICIES" in private_script


def test_aws_sft_template_cloud_private_is_s3_only():
    script = _script("cloud-private")

    assert "requires HF_TOKEN or HUGGINGFACE_HUB_TOKEN" in script
    assert "PUBLISH_TO_HUB_POLICIES" in script
    assert "push_to_hub=OUTPUT_POLICY in PUBLISH_TO_HUB_POLICIES" in script
    assert "if OUTPUT_POLICY in PUBLISH_TO_HUB_POLICIES:" in script
    assert (
        'hub_model_id_marker = HUB_MODEL_ID if OUTPUT_POLICY in PUBLISH_TO_HUB_POLICIES else ""'
        in script
    )


def test_aws_sft_template_formats_phase3_normalized_rows():
    script = _script()

    assert 'if "messages" in example:' in script
    assert 'if "text" in example:' in script
    assert '("prompt", "completion")' in script
    assert '("instruction", "output")' in script
    assert '("instruction", "response")' in script
    assert '("input", "output")' in script
    assert '("input", "response")' in script
    assert '("question", "answer")' in script
    assert "fallback_text_from_example" in script
    assert 'data = example.get("data")' in script
    assert "Mapped {kind} column is missing" in script


def test_aws_sft_template_validates_structured_messages_before_trainer():
    script = _script()

    assert "def validate_formatted_example" in script
    assert "valid_records" in script
    assert "Skipped" in script
    assert "No valid SFT records found" in script
    assert "user_text" in script
    assert "assistant_text" in script


def test_aws_sft_template_message_validation_does_not_call_string_value_with_scalar():
    script = _script()

    assert "_string_value(message.get(" not in script
    assert 'content = message.get("content")' in script


def test_aws_sft_template_uses_current_trl_processing_class_style():
    script = _script()

    assert "processing_class=tokenizer" in script
    assert "tokenizer=" not in script
    assert "eval_strategy" in script
    assert "evaluation_strategy" not in script
    assert "max_length=1024" in script


def test_aws_sft_template_eval_split_behavior_is_deterministic_and_small_safe():
    script = _script()

    assert "len(train_dataset) >= 20" in script
    assert "train_test_split" in script
    assert "seed=42" in script
    assert "No evaluation dataset was available; skipping evaluation." in script


@pytest.mark.parametrize(
    "params, expected",
    [
        ({}, "model_name is required"),
        ({"model_name": "model", "output_model_id": ""}, "output_model_id is required"),
        (
            {
                "model_name": "model",
                "output_model_id": "out",
                "output_policy": "public",
            },
            "output_policy must be one of",
        ),
        (
            {"model_name": "model", "output_model_id": "out", "num_train_epochs": 0},
            "num_train_epochs must be positive",
        ),
        (
            {
                "model_name": "model",
                "output_model_id": "out",
                "validation_split_ratio": 1,
            },
            "validation_split_ratio must be greater than 0 and less than 1",
        ),
        (
            {
                "model_name": "model",
                "output_model_id": "out",
                "column_mapping": {"assistant": []},
            },
            "column_mapping.assistant must be a string or a list of non-empty strings",
        ),
        (
            {"model_name": "dummy-model", "output_model_id": "owner/out"},
            "dummy placeholder values are not allowed",
        ),
    ],
)
def test_aws_sft_validation_rejects_invalid_requests(params, expected):
    errors = validate_aws_sft_template_request(params)

    assert any(expected in error for error in errors)


def test_aws_sft_validation_accepts_valid_minimal_config():
    errors = validate_aws_sft_template_request(
        {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_model_id": "owner/aws-output",
            "output_policy": "aws-private",
            "column_mapping": {"assistant": "answer"},
        }
    )

    assert errors == []
