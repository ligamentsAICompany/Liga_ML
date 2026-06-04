import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "aws_sagemaker_dry_run.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("aws_sagemaker_dry_run", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample_args(*extra: str) -> list[str]:
    return [
        "--dataset-name",
        "example/dataset",
        "--model-name",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--output-model-id",
        "aws-smoke-model",
        "--max-run-seconds",
        "3600",
        *extra,
    ]


def test_dry_run_allow_missing_aws_and_image_outputs_json_without_submit(
    capsys,
) -> None:
    module = _load_script_module()
    submit_called = False

    module.build_aws_sagemaker_readiness_snapshot = lambda: {
        "configured": False,
        "missing_env": ["AWS_S3_BUCKET", "AWS_SAGEMAKER_ROLE_ARN"],
        "region": "us-east-1",
        "s3_bucket": None,
        "s3_prefix": "liga-ml",
        "sagemaker_role_arn": None,
        "default_instance_type": "ml.g5.xlarge",
        "default_instance_count": 1,
        "default_max_run_seconds": 3600,
        "output_policy": "aws-private",
        "training_image_uri": None,
        "credentials_detected": False,
        "warnings": ["credentials missing"],
        "errors": ["Missing required AWS environment variables."],
    }

    class ExplodingTool:
        async def execute(self, _params):
            nonlocal submit_called
            submit_called = True
            raise AssertionError("dry run must not submit SageMaker jobs")

    module.AwsSageMakerJobsTool = ExplodingTool

    assert (
        module.main(_sample_args("--allow-missing-aws", "--allow-missing-image")) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["submitted_sagemaker_job"] is False
    assert payload["aws_readiness"]["configured"] is False
    assert payload["aws_readiness"]["warning_only"] is True
    assert payload["image"]["configured"] is False
    assert payload["image"]["warning_only"] is True
    assert payload["template_validation"]["ok"] is True
    assert payload["script_checks"]["generated"] is True
    assert payload["script_checks"]["required_markers_present"] is True
    assert payload["script_checks"]["required_paths_present"] is True
    assert payload["cost_estimate"]["estimated_cost_usd"] is not None
    assert submit_called is False


def test_dry_run_missing_aws_fails_without_allow_flag(capsys) -> None:
    module = _load_script_module()
    module.build_aws_sagemaker_readiness_snapshot = lambda: {
        "configured": False,
        "missing_env": ["AWS_S3_BUCKET"],
        "credentials_detected": False,
        "warnings": [],
        "errors": ["Missing required AWS environment variables."],
    }

    assert module.main(_sample_args("--allow-missing-image")) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["aws_readiness"]["configured"] is False
    assert payload["submitted_sagemaker_job"] is False


def test_dry_run_missing_image_fails_without_allow_flag(capsys) -> None:
    module = _load_script_module()
    module.build_aws_sagemaker_readiness_snapshot = lambda: {
        "configured": True,
        "missing_env": [],
        "region": "us-east-1",
        "s3_bucket": "your-s3-bucket",
        "s3_prefix": "liga-ml",
        "sagemaker_role_arn": "arn:aws:iam::123456789012:role/LigaMLSageMakerExecutionRole",
        "default_instance_type": "ml.g5.xlarge",
        "default_instance_count": 1,
        "default_max_run_seconds": 3600,
        "output_policy": "aws-private",
        "training_image_uri": None,
        "credentials_detected": True,
        "warnings": [],
        "errors": [],
    }

    assert module.main(_sample_args("--allow-missing-aws")) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["image"]["configured"] is False
    assert payload["submitted_sagemaker_job"] is False
