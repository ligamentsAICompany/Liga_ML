import ast

import pytest

from agent.training_templates.aws_sft import (
    AwsSftTemplateConfig,
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
    assert 'metrics_path = OUTPUT_DATA_DIR / "metrics.json"' in script
    assert 'provider": "aws-sagemaker"' in script


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
    ]:
        assert marker in script


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
    assert 'OUTPUT_POLICY != "aws-private"' in private_script


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
