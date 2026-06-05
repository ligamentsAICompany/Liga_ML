import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "check_aws_readiness.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("check_aws_readiness", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_aws_readiness_script_exits_zero_when_configured(capsys) -> None:
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
        "training_image_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/liga-train:latest",
        "credentials_detected": True,
        "warnings": [],
        "errors": [],
    }

    assert module.main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["configured"] is True
    assert payload["region"] == "us-east-1"
    assert payload["training_image_configured"] is True
    assert payload["credentials_detected"] is True


def test_aws_readiness_script_exits_one_when_missing_config_and_hides_secrets(
    capsys,
) -> None:
    module = _load_script_module()
    module.build_aws_sagemaker_readiness_snapshot = lambda: {
        "configured": False,
        "missing_env": ["AWS_S3_BUCKET", "AWS_SAGEMAKER_ROLE_ARN"],
        "region": "us-east-1",
        "s3_bucket": None,
        "s3_prefix": "liga-ml",
        "sagemaker_role_arn": None,
        "credentials_detected": False,
        "aws_access_key_id": "FAKE_ACCESS_KEY_ID",
        "aws_secret_access_key": "super-secret",
        "aws_session_token": "session-token",
        "credential_path": "/tmp/credentials",
        "warnings": ["credentials unavailable"],
        "errors": ["Missing required AWS environment variables."],
    }

    assert module.main([]) == 1
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["configured"] is False
    assert payload["missing_env"] == ["AWS_S3_BUCKET", "AWS_SAGEMAKER_ROLE_ARN"]
    assert "FAKE_ACCESS_KEY_ID" not in output
    assert "super-secret" not in output
    assert "session-token" not in output
    assert "credential_path" not in output
