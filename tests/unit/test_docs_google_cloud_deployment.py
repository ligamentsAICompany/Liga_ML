from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "docs" / "google-cloud-deployment.md"


def _doc_text() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def test_google_cloud_deployment_doc_covers_required_apis_roles_and_secrets() -> None:
    text = _doc_text()

    for phrase in [
        "Vertex AI API",
        "Cloud Storage API",
        "Cloud Logging API",
        "Artifact Registry API",
        "Cloud Build API",
        "Secret Manager API",
        "Cloud Run Admin API",
        "IAM Credentials API",
        "HF_TOKEN",
        "GITHUB_TOKEN",
        "OPENAI_API_KEY",
        "roles/aiplatform.user",
        "roles/storage.objectAdmin",
        "roles/logging.viewer",
        "roles/artifactregistry.reader",
        "roles/secretmanager.secretAccessor",
        "roles/iam.serviceAccountUser",
        "roles/run.admin",
    ]:
        assert phrase in text


def test_google_cloud_deployment_doc_covers_deploy_verify_and_smoke_test() -> None:
    text = _doc_text()

    for phrase in [
        "gsutil mb",
        "gcloud builds submit",
        "/api/health",
        "/api/health/providers",
        "gcp_vertex",
        "configured",
        "max_train_samples=5",
        "max_eval_samples=2",
        "num_train_epochs=1",
        "max_run_hours=1",
        "push",
        "Hugging Face Hub",
    ]:
        assert phrase in text


def test_google_cloud_deployment_doc_covers_troubleshooting() -> None:
    text = _doc_text()

    for phrase in [
        "missing GOOGLE_CLOUD_PROJECT",
        "missing GCS_BUCKET",
        "ADC",
        "roles/iam.serviceAccountUser missing",
        "Vertex job cannot access GCS",
        "HF token missing in Vertex job",
        "Cloud Logging permission missing",
        "Cloud Run timeout too low",
        "backend cannot import Google libs",
    ]:
        assert phrase in text
