from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _load_cloudbuild() -> dict:
    return yaml.safe_load((ROOT / "cloudbuild.yaml").read_text(encoding="utf-8"))


def _step_text(step: dict) -> str:
    return "\n".join(str(arg) for arg in step.get("args", []))


def _deploy_step(config: dict) -> dict:
    matches = [
        step for step in config["steps"] if step.get("id") == "Deploy to Cloud Run"
    ]
    assert len(matches) == 1
    return matches[0]


def test_cloudbuild_deploys_cloud_run_on_port_8080_with_required_env() -> None:
    config = _load_cloudbuild()

    substitutions = config["substitutions"]
    assert substitutions["_REGION"] == "us-central1"
    assert substitutions["_SERVICE_NAME"] == "liga-ml-intern"
    assert substitutions["_ARTIFACT_REPO"] == "liga-ml-containers"
    assert substitutions["_IMAGE_NAME"] == "liga-ml-intern"
    assert substitutions["_GCS_BUCKET"] == "liga-ml-training"

    deploy_args = _step_text(_deploy_step(config))
    assert "gcloud run deploy" in deploy_args
    assert '--region="${_REGION}"' in deploy_args
    assert "--platform=managed" in deploy_args
    assert "--port=8080" in deploy_args
    assert "--memory=2Gi" in deploy_args
    assert "--cpu=2" in deploy_args
    assert "--timeout=3600" in deploy_args
    assert "--concurrency=20" in deploy_args

    env_vars = deploy_args
    for name in [
        "ML_INTERN_DEFAULT_MODEL_ID",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_REGION",
        "GCS_BUCKET",
        "VERTEX_AI_STAGING_BUCKET",
        "VERTEX_AI_OUTPUT_DIR",
        "ML_INTERN_KPIS_DISABLED",
    ]:
        assert f"{name}=" in env_vars


def test_cloudbuild_uses_secret_manager_without_raw_secret_values() -> None:
    config = _load_cloudbuild()
    rendered = (ROOT / "cloudbuild.yaml").read_text(encoding="utf-8")

    substitutions = config["substitutions"]
    assert substitutions["_HF_TOKEN_SECRET"] == "hf-token"
    assert substitutions["_GITHUB_TOKEN_SECRET"] == "github-token"
    assert substitutions["_OPENAI_API_KEY_SECRET"] == "openai-api-key"

    secrets_arg = _step_text(_deploy_step(config))
    assert "HF_TOKEN=${_HF_TOKEN_SECRET}:latest" in secrets_arg
    assert "GITHUB_TOKEN=${_GITHUB_TOKEN_SECRET}:latest" in secrets_arg
    assert "OPENAI_API_KEY=${_OPENAI_API_KEY_SECRET}:latest" in secrets_arg

    forbidden = ["hf_", "github_pat_", "ghp_", "sk-"]
    assert all(marker not in rendered for marker in forbidden)


def test_cloudbuild_builds_pushes_and_outputs_images() -> None:
    config = _load_cloudbuild()
    args_by_step = [" ".join(step.get("args", [])) for step in config["steps"]]
    rendered_steps = "\n".join(args_by_step)

    assert 'artifacts repositories describe "${_ARTIFACT_REPO}"' in rendered_steps
    assert 'artifacts repositories create "${_ARTIFACT_REPO}"' in rendered_steps
    assert any(step.get("id") == "Build Docker image" for step in config["steps"])
    assert "$COMMIT_SHA" in rendered_steps
    assert ":latest" in rendered_steps
    assert any("Pushed images:" in step for step in args_by_step)
    assert any(image.endswith(":$COMMIT_SHA") for image in config["images"])
    assert any(image.endswith(":latest") for image in config["images"])
