from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_gcloud_merge_readiness_doc_covers_phase6_acceptance() -> None:
    text = (ROOT / "docs" / "gcloud-merge-readiness.md").read_text(encoding="utf-8")

    for phrase in [
        "Phase 1",
        "Phase 2",
        "Phase 3",
        "Phase 4",
        "Phase 5",
        "Phase 6",
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run pytest -q",
        "cd frontend && npm run lint",
        "cd frontend && npm run build",
        "uv run python scripts/check_gcp_readiness.py",
        "scripts/gcloud_vertex_dry_run.py",
        "--allow-missing-gcp",
        "open PR gcloud->main",
        "only Vertex SFT productionized",
        "scanned PDFs",
        "real GCP project/bucket/IAM/secrets",
    ]:
        assert phrase in text


def test_ci_workflow_has_backend_frontend_without_secrets_or_deploy() -> None:
    workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "branches: [main, gcloud]" in workflow
    assert "uv sync --all-extras" in workflow
    assert "uv run ruff check ." in workflow
    assert "uv run ruff format --check ." in workflow
    assert "uv run pytest -q" in workflow
    assert "npm ci" in workflow
    assert "npm run lint" in workflow
    assert "npm run build" in workflow
    assert "npm run test:training-result" in workflow
    lowered = workflow.lower()
    assert "secrets." not in lowered
    assert "gcloud run deploy" not in lowered
    assert "gcloud builds submit" not in lowered


def test_readme_links_phase6_gcloud_docs_and_scripts() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    for phrase in [
        "docs/google-cloud-deployment.md",
        "docs/gcloud-merge-readiness.md",
        "scripts/check_gcp_readiness.py",
        "scripts/gcloud_vertex_dry_run.py",
    ]:
        assert phrase in text
