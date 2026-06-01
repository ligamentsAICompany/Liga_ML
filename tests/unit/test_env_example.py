from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _env_example() -> str:
    return (ROOT / ".env.example").read_text(encoding="utf-8")


def test_env_example_includes_cloud_run_and_vertex_placeholders() -> None:
    text = _env_example()

    expected_lines = [
        "HF_TOKEN=",
        "GITHUB_TOKEN=",
        "OPENAI_API_KEY=",
        "ML_INTERN_DEFAULT_MODEL_ID=moonshotai/Kimi-K2.6",
        "ML_INTERN_KPIS_DISABLED=1",
        "GOOGLE_CLOUD_PROJECT=",
        "GOOGLE_CLOUD_REGION=us-central1",
        "GCS_BUCKET=",
        "VERTEX_AI_STAGING_BUCKET=",
        "VERTEX_AI_OUTPUT_DIR=",
        "VERTEX_AI_SERVICE_ACCOUNT=",
        "CORS_ALLOW_ORIGINS=",
        "ALLOWED_HOSTS=",
    ]
    for line in expected_lines:
        assert line in text


def test_env_example_does_not_contain_real_looking_secrets() -> None:
    text = _env_example()

    forbidden = [
        "hf_",
        "github_pat_",
        "ghp_",
        "sk-",
        "-----BEGIN PRIVATE KEY-----",
    ]
    assert all(marker not in text for marker in forbidden)
