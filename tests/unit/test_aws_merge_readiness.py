from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "docs" / "aws-merge-readiness.md"


def _doc_text() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def test_aws_merge_readiness_doc_covers_phases_and_matrix() -> None:
    text = _doc_text()

    for phrase in [
        "Phase 1",
        "Phase 2",
        "Phase 3",
        "Phase 4",
        "Phase 5",
        "Phase 6",
        "Production Readiness Matrix",
        "Hugging Face",
        "GCloud Vertex",
        "AWS SageMaker",
        "provider selection",
        "dataset upload/normalization",
        "cloud storage",
        "job submission",
        "approval/cost",
        "logs/monitoring",
        "final parsing",
        "private output mode",
        "deployment docs",
        "CI/dry-run",
    ]:
        assert phrase in text


def test_aws_merge_readiness_doc_covers_validation_and_manual_acceptance() -> None:
    text = _doc_text()

    for phrase in [
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run pytest -q",
        "cd frontend && npm run lint",
        "cd frontend && npm run build",
        "cd frontend && npm run test:training-result",
        "cd frontend && npm run test:cloud-providers",
        "cd frontend && npm run test:aws-sagemaker-panel",
        "uv run python scripts/check_aws_readiness.py",
        "uv run python scripts/aws_sagemaker_dry_run.py",
        "--allow-missing-aws",
        "--allow-missing-image",
        "Manual Browser Acceptance Checklist",
        "Select AWS SageMaker",
        "approval",
        "CloudWatch",
        "/api/health/providers",
    ]:
        assert phrase in text


def test_aws_merge_readiness_doc_covers_limitations_and_merge_recommendation() -> None:
    text = _doc_text()

    for phrase in [
        "live AWS requires credentials/IAM/S3/ECR image",
        "only SFT productionized",
        "Kaggle future",
        "scanned PDFs need OCR",
        "Bedrock future",
        "open PR AWS -> gcloud or AWS -> main",
        "do not merge until CI passes",
        "controlled AWS smoke approved",
    ]:
        assert phrase in text
