from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_dockerfile_builds_frontend_and_runs_backend_on_8080() -> None:
    text = _read("Dockerfile")

    assert "FROM node:20" in text
    assert "npm ci" in text
    assert "npm run build" in text
    assert "FROM python:" in text
    assert "uv sync --no-dev --frozen" in text
    assert "COPY --from=frontend-builder /app/frontend/dist ./static/" in text
    assert "EXPOSE 8080" in text
    assert "${PORT:-8080}" in text
    assert "uvicorn main:app --host 0.0.0.0" in text
    assert "USER user" in text
    assert "HEALTHCHECK" in text
    assert "COPY .env" not in text


def test_dockerignore_excludes_secrets_caches_and_local_datasets() -> None:
    text = _read(".dockerignore")

    required_patterns = [
        ".git",
        ".venv",
        "node_modules",
        "frontend/node_modules",
        "frontend/dist",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".env",
        ".env.*",
        "*.pem",
        "*.key",
        "*.p12",
        "*.log",
        ".DS_Store",
        "session_logs",
        "medical_patient_support_training_data.md",
        "hardware_support_real_world_dataset.md",
        "call_center_real_world_pilot.csv",
    ]
    for pattern in required_patterns:
        assert pattern in text

    assert "!.env.example" in text


def test_compose_runs_production_image_without_inline_secrets() -> None:
    text = _read("compose.yaml")

    assert "build:" in text
    assert "8080:8080" in text
    assert "env_file:" in text
    assert ".env" in text
    assert "HF_TOKEN=" not in text
    assert "OPENAI_API_KEY=" not in text
    assert "GOOGLE_APPLICATION_CREDENTIALS=" not in text


def test_docker_docs_are_linked_from_readme() -> None:
    docs = _read("docs/docker-deployment.md")
    readme = _read("README.md")

    for expected in [
        "docker build",
        "docker compose",
        "/api/health",
        "/api/health/providers",
        "Secret Manager",
        "Cloud Run",
        "HF_TOKEN",
        "MONGODB_URI",
    ]:
        assert expected in docs

    assert "docs/docker-deployment.md" in readme
