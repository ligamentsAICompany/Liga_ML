from agent.core.gcp_readiness import build_gcp_vertex_readiness_snapshot


def _clear_gcp_env(monkeypatch):
    for name in [
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_REGION",
        "GCS_BUCKET",
        "VERTEX_AI_STAGING_BUCKET",
        "VERTEX_AI_OUTPUT_DIR",
        "VERTEX_AI_SERVICE_ACCOUNT",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_gcp_readiness_missing_env_is_safe(monkeypatch):
    _clear_gcp_env(monkeypatch)

    snapshot = build_gcp_vertex_readiness_snapshot()

    assert snapshot["configured"] is False
    assert snapshot["missing_env"] == [
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_REGION",
        "GCS_BUCKET",
    ]
    assert snapshot["project"] is None
    assert snapshot["bucket"] is None
    assert "credentials" not in snapshot


def test_gcp_readiness_derives_gcs_paths_without_exposing_secrets(monkeypatch):
    _clear_gcp_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "liga-prod")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    monkeypatch.setenv(
        "VERTEX_AI_SERVICE_ACCOUNT", "vertex-runner@liga-prod.iam.gserviceaccount.com"
    )
    monkeypatch.setattr(
        "agent.core.gcp_readiness._detect_adc", lambda: (True, None, [])
    )

    snapshot = build_gcp_vertex_readiness_snapshot()

    assert snapshot["configured"] is True
    assert snapshot["missing_env"] == []
    assert snapshot["project"] == "liga-prod"
    assert snapshot["region"] == "us-central1"
    assert snapshot["bucket"] == "liga-training"
    assert snapshot["staging_bucket"] == "gs://liga-training/vertex-staging"
    assert snapshot["output_dir"] == "gs://liga-training/vertex-outputs"
    assert snapshot["credentials_detected"] is True
    assert (
        snapshot["service_account"] == "vertex-runner@liga-prod.iam.gserviceaccount.com"
    )
    assert all("secret" not in key.lower() for key in snapshot)
