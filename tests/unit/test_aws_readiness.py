import sys

from agent.core import aws_readiness


def _clear_aws_env(monkeypatch):
    for name in [
        "AWS_REGION",
        "AWS_S3_BUCKET",
        "AWS_S3_PREFIX",
        "AWS_SAGEMAKER_ROLE_ARN",
        "AWS_DEFAULT_INSTANCE_TYPE",
        "AWS_DEFAULT_INSTANCE_COUNT",
        "AWS_DEFAULT_MAX_RUN_SECONDS",
        "AWS_OUTPUT_POLICY",
        "AWS_SAGEMAKER_TRAINING_IMAGE_URI",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_aws_readiness_missing_env_is_safe(monkeypatch):
    _clear_aws_env(monkeypatch)
    monkeypatch.setattr(aws_readiness, "_detect_aws_credentials", lambda: (False, []))

    snapshot = aws_readiness.build_aws_sagemaker_readiness_snapshot()

    assert snapshot["configured"] is False
    assert snapshot["missing_env"] == [
        "AWS_REGION",
        "AWS_S3_BUCKET",
        "AWS_SAGEMAKER_ROLE_ARN",
    ]
    assert snapshot["region"] == "us-east-1"
    assert snapshot["s3_bucket"] is None
    assert snapshot["s3_prefix"] == "liga-ml"
    assert snapshot["sagemaker_role_arn"] is None
    assert snapshot["default_instance_type"] == "ml.g5.xlarge"
    assert snapshot["default_instance_count"] == 1
    assert snapshot["default_max_run_seconds"] == 3600
    assert snapshot["output_policy"] == "aws-private"
    assert snapshot["training_image_uri"] is None
    assert snapshot["credentials_detected"] is False
    assert "access_key" not in str(snapshot).lower()
    assert "secret" not in str(snapshot).lower()


def test_aws_readiness_required_env_and_mocked_credentials_true(monkeypatch):
    _clear_aws_env(monkeypatch)
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("AWS_S3_BUCKET", "training-bucket")
    monkeypatch.setenv("AWS_S3_PREFIX", "team-prefix")
    monkeypatch.setenv(
        "AWS_SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/TestRole"
    )
    monkeypatch.setenv("AWS_DEFAULT_INSTANCE_TYPE", "ml.g4dn.xlarge")
    monkeypatch.setenv("AWS_DEFAULT_INSTANCE_COUNT", "2")
    monkeypatch.setenv("AWS_DEFAULT_MAX_RUN_SECONDS", "7200")
    monkeypatch.setenv("AWS_OUTPUT_POLICY", "hf-hub")
    monkeypatch.setenv("AWS_SAGEMAKER_TRAINING_IMAGE_URI", "example-image")
    monkeypatch.setattr(aws_readiness, "_detect_aws_credentials", lambda: (True, []))

    snapshot = aws_readiness.build_aws_sagemaker_readiness_snapshot()

    assert snapshot["configured"] is True
    assert snapshot["missing_env"] == []
    assert snapshot["region"] == "us-west-2"
    assert snapshot["s3_bucket"] == "training-bucket"
    assert snapshot["s3_prefix"] == "team-prefix"
    assert snapshot["sagemaker_role_arn"] == "arn:aws:iam::123456789012:role/TestRole"
    assert snapshot["default_instance_type"] == "ml.g4dn.xlarge"
    assert snapshot["default_instance_count"] == 2
    assert snapshot["default_max_run_seconds"] == 7200
    assert snapshot["output_policy"] == "hf-hub"
    assert snapshot["training_image_uri"] == "example-image"
    assert snapshot["credentials_detected"] is True
    assert snapshot["warnings"] == []
    assert snapshot["errors"] == []


def test_aws_readiness_invalid_numeric_env_blocks_configured(monkeypatch):
    _clear_aws_env(monkeypatch)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_S3_BUCKET", "training-bucket")
    monkeypatch.setenv(
        "AWS_SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/TestRole"
    )
    monkeypatch.setenv("AWS_DEFAULT_INSTANCE_COUNT", "zero")
    monkeypatch.setenv("AWS_DEFAULT_MAX_RUN_SECONDS", "-1")
    monkeypatch.setattr(aws_readiness, "_detect_aws_credentials", lambda: (True, []))

    snapshot = aws_readiness.build_aws_sagemaker_readiness_snapshot()

    assert snapshot["configured"] is False
    assert snapshot["default_instance_count"] == 1
    assert snapshot["default_max_run_seconds"] == 3600
    assert any("AWS_DEFAULT_INSTANCE_COUNT" in error for error in snapshot["errors"])
    assert any("AWS_DEFAULT_MAX_RUN_SECONDS" in error for error in snapshot["errors"])


def test_aws_readiness_handles_missing_boto3_gracefully(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)

    detected, warnings = aws_readiness._detect_aws_credentials()

    assert detected is False
    assert any("boto3" in warning for warning in warnings)
