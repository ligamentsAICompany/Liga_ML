from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "docs" / "aws-sagemaker-deployment.md"


def _doc_text() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def test_aws_deployment_doc_covers_services_and_required_env() -> None:
    text = _doc_text()

    for phrase in [
        "SageMaker",
        "S3",
        "CloudWatch Logs",
        "IAM",
        "ECR",
        "Secrets Manager",
        "AWS_REGION",
        "AWS_S3_BUCKET",
        "AWS_S3_PREFIX",
        "AWS_SAGEMAKER_ROLE_ARN",
        "AWS_DEFAULT_INSTANCE_TYPE",
        "AWS_DEFAULT_INSTANCE_COUNT",
        "AWS_DEFAULT_MAX_RUN_SECONDS",
        "AWS_OUTPUT_POLICY",
        "AWS_SAGEMAKER_TRAINING_IMAGE_URI",
    ]:
        assert phrase in text


def test_aws_deployment_doc_covers_iam_s3_image_and_output_policies() -> None:
    text = _doc_text()

    for phrase in [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:ListTrainingJobs",
        "sagemaker:StopTrainingJob",
        "iam:PassRole",
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
        "logs:DescribeLogStreams",
        "logs:GetLogEvents",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "ecr:GetAuthorizationToken",
        "ecr:BatchGetImage",
        "input/train.jsonl",
        "code/train.py",
        "output/",
        "checkpoints/",
        "aws-private",
        "hf-hub",
        "cloud-and-hf-hub",
    ]:
        assert phrase in text


def test_aws_deployment_doc_covers_validation_smoke_and_troubleshooting() -> None:
    text = _doc_text()

    for phrase in [
        "uv run python scripts/check_aws_readiness.py",
        "uv run python scripts/aws_sagemaker_dry_run.py",
        "curl",
        "/api/health/providers",
        "Run a tiny AWS SageMaker SFT smoke test",
        "max_train_samples=5",
        "max_eval_samples=2",
        "num_train_epochs=1",
        "max_run_seconds=3600",
        "missing AWS_S3_BUCKET",
        "credentials",
        "iam:PassRole denied",
        "S3 access denied",
        "image pull",
        "CloudWatch logs not visible",
        "HF_TOKEN missing",
        "cost unknown",
    ]:
        assert phrase in text
