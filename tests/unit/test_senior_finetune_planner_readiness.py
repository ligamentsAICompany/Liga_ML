from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "docs" / "senior-finetune-planner-readiness.md"


def _doc_text() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def test_senior_finetune_planner_readiness_doc_covers_required_topics() -> None:
    text = _doc_text()

    for phrase in [
        "Uploaded Data",
        "Markdown",
        "training_planner",
        "dataset_discovery",
        "cloud-private",
        "hf-hub",
        "cloud-and-hf-hub",
        "Kaggle is future work",
        "Manual Smoke Checklist",
        "Validation Commands",
    ]:
        assert phrase in text


def test_senior_finetune_planner_readiness_doc_lists_validation_commands() -> None:
    text = _doc_text()

    for command in [
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run pytest -q",
        "cd frontend && npm run lint",
        "cd frontend && npm run build",
        "cd frontend && npm run test:training-result",
        "cd frontend && npm run test:cloud-providers",
        "cd frontend && npm run test:dataset-upload-ui",
        "cd frontend && npm run test:output-policy",
        "cd frontend && npm run test:training-planner-panel",
        "cd frontend && npm run test:dataset-discovery-panel",
    ]:
        assert command in text


def test_ci_runs_order3_frontend_readiness_tests() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    for script in [
        "npm run test:dataset-upload-ui",
        "npm run test:output-policy",
        "npm run test:training-planner-panel",
        "npm run test:dataset-discovery-panel",
    ]:
        assert script in workflow
