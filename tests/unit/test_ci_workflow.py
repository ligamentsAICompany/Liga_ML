from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci.yml"


def test_ci_workflow_includes_backend_frontend_and_aws_panel_checks() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    for phrase in [
        "uv sync --all-extras",
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run pytest -q",
        "npm ci",
        "npm run lint",
        "npm run build",
        "npm run test:training-result",
        "npm run test:cloud-providers",
        "npm run test:aws-sagemaker-panel",
    ]:
        assert phrase in workflow


def test_ci_workflow_has_no_aws_secrets_or_deploy_commands() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    lowered = workflow.lower()

    forbidden = [
        "secrets.aws",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "create-training-job",
        "sagemaker create-training-job",
        "aws s3 cp",
        "gcloud run deploy",
        "gcloud builds submit",
    ]
    for phrase in forbidden:
        assert phrase not in lowered
