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
    assert "--memory=4Gi" in deploy_args
    assert "--cpu=2" in deploy_args
    assert "--timeout=3600" in deploy_args
    assert "--concurrency=5" in deploy_args
    assert "--min-instances=1" in deploy_args

    env_vars = deploy_args
    for name in [
        "ML_INTERN_DEFAULT_MODEL_ID",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_REGION",
        "GCS_BUCKET",
        "VERTEX_STAGING_BUCKET",
        "VERTEX_OUTPUT_DIR",
        "VERTEX_AI_STAGING_BUCKET",
        "VERTEX_AI_OUTPUT_DIR",
        "AWS_REGION",
        "AWS_S3_BUCKET",
        "AWS_S3_PREFIX",
        "AWS_SAGEMAKER_ROLE_ARN",
        "AWS_SAGEMAKER_TRAINING_IMAGE_URI",
        "AWS_DEFAULT_INSTANCE_TYPE",
        "AWS_DEFAULT_INSTANCE_COUNT",
        "AWS_DEFAULT_MAX_RUN_SECONDS",
        "AWS_OUTPUT_POLICY",
        "ML_INTERN_KPIS_DISABLED",
        "BACKGROUND_RUNS_ENABLED",
        "RUN_WORKER_MODE",
        "USAGE_DASHBOARD_ENABLED",
        "DEFAULT_DAILY_BUDGET_USD",
        "DEFAULT_MONTHLY_BUDGET_USD",
        "HF_DAILY_BUDGET_USD",
        "GCLOUD_DAILY_BUDGET_USD",
        "AWS_DAILY_BUDGET_USD",
    ]:
        assert f"{name}=" in env_vars

    assert "BACKGROUND_RUNS_ENABLED=true" in env_vars
    assert "RUN_WORKER_MODE=in_process" in env_vars
    assert "USAGE_DASHBOARD_ENABLED=true" in env_vars
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in deploy_args


def test_cloudbuild_uses_secret_manager_without_raw_secret_values() -> None:
    config = _load_cloudbuild()
    rendered = (ROOT / "cloudbuild.yaml").read_text(encoding="utf-8")

    substitutions = config["substitutions"]
    assert substitutions["_HF_TOKEN_SECRET"] == "hf-token"
    assert substitutions["_HUGGINGFACE_HUB_TOKEN_SECRET"] == "huggingface-hub-token"
    assert substitutions["_GITHUB_TOKEN_SECRET"] == "github-token"
    assert substitutions["_OPENAI_API_KEY_SECRET"] == "openai-api-key"
    assert substitutions["_AWS_ACCESS_KEY_ID_SECRET"] == "aws-access-key-id"
    assert substitutions["_AWS_SECRET_ACCESS_KEY_SECRET"] == "aws-secret-access-key"
    assert substitutions["_AWS_SESSION_TOKEN_SECRET"] == ""
    assert "_MONGODB_URI_SECRET" in substitutions
    assert "_SESSION_TOKEN_ENCRYPTION_KEY_SECRET" in substitutions
    assert "_DEFAULT_DAILY_BUDGET_USD" in substitutions
    assert "_DEFAULT_MONTHLY_BUDGET_USD" in substitutions
    assert "_HF_DAILY_BUDGET_USD" in substitutions
    assert "_GCLOUD_DAILY_BUDGET_USD" in substitutions
    assert "_AWS_DAILY_BUDGET_USD" in substitutions

    secrets_arg = _step_text(_deploy_step(config))
    assert "HF_TOKEN=${_HF_TOKEN_SECRET}:latest" in secrets_arg
    assert (
        "HUGGINGFACE_HUB_TOKEN=${_HUGGINGFACE_HUB_TOKEN_SECRET}:latest" in secrets_arg
    )
    assert "GITHUB_TOKEN=${_GITHUB_TOKEN_SECRET}:latest" in secrets_arg
    assert "OPENAI_API_KEY=${_OPENAI_API_KEY_SECRET}:latest" in secrets_arg
    assert "AWS_ACCESS_KEY_ID=${_AWS_ACCESS_KEY_ID_SECRET}:latest" in secrets_arg
    assert (
        "AWS_SECRET_ACCESS_KEY=${_AWS_SECRET_ACCESS_KEY_SECRET}:latest" in secrets_arg
    )
    assert "AWS_SESSION_TOKEN=${_AWS_SESSION_TOKEN_SECRET}:latest" in secrets_arg
    assert 'if [ -n "${_MONGODB_URI_SECRET}" ]; then' in secrets_arg
    assert "MONGODB_URI=${_MONGODB_URI_SECRET}:latest" in secrets_arg
    assert 'if [ -n "${_SESSION_TOKEN_ENCRYPTION_KEY_SECRET}" ]; then' in secrets_arg
    assert (
        "SESSION_TOKEN_ENCRYPTION_KEY=${_SESSION_TOKEN_ENCRYPTION_KEY_SECRET}:latest"
        in secrets_arg
    )

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
