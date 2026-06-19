import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import patch_api_helper

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from routes import api as agent  # noqa: E402


@pytest.mark.asyncio
async def test_health_reports_non_durable_session_store_warning(monkeypatch):
    monkeypatch.setattr(
        agent.session_manager,
        "persistence_store",
        SimpleNamespace(enabled=False),
    )

    response = await agent.health_check()

    payload = response.model_dump()
    assert payload["session_store"] == {
        "type": "noop",
        "durable": False,
        "warning": "MONGODB_URI is not configured; sessions will not survive restarts",
    }


@pytest.mark.asyncio
async def test_health_reports_mongodb_session_store_as_durable(monkeypatch):
    monkeypatch.setattr(
        agent.session_manager,
        "persistence_store",
        SimpleNamespace(enabled=True),
    )

    response = await agent.health_check()

    assert response.model_dump()["session_store"] == {
        "type": "mongodb",
        "durable": True,
        "warning": None,
    }


@pytest.mark.asyncio
async def test_provider_health_returns_hf_gcp_and_aws(monkeypatch):
    patch_api_helper(monkeypatch, "build_gcp_vertex_readiness_snapshot",
        lambda: {
            "configured": False,
            "missing_env": ["GOOGLE_CLOUD_PROJECT"],
            "project": None,
            "region": "us-central1",
            "bucket": None,
            "staging_bucket": None,
            "output_dir": None,
            "service_account": None,
            "credentials_detected": False,
            "warnings": [],
            "errors": [],
        },
    )
    patch_api_helper(monkeypatch, "build_aws_sagemaker_readiness_snapshot",
        lambda: {
            "configured": False,
            "missing_env": ["AWS_S3_BUCKET"],
            "region": "us-east-1",
            "s3_bucket": None,
            "s3_prefix": "liga-ml",
            "sagemaker_role_arn": None,
            "default_instance_type": "ml.g5.xlarge",
            "default_instance_count": 1,
            "default_max_run_seconds": 3600,
            "output_policy": "aws-private",
            "training_image_uri": None,
            "credentials_detected": False,
            "warnings": [],
            "errors": ["Missing required AWS environment variables."],
        },
    )
    monkeypatch.setenv("HF_TOKEN", "hf-secret")

    response = await agent.provider_health()

    assert set(response) == {
        "hf_jobs",
        "gcp_vertex",
        "aws_sagemaker",
        "session_store",
        "audit_store",
        "security",
    }
    assert response["hf_jobs"]["configured"] is True
    assert response["hf_jobs"]["hf_token_configured"] is True
    assert response["gcp_vertex"]["configured"] is False
    assert response["gcp_vertex"]["missing_env"] == ["GOOGLE_CLOUD_PROJECT"]
    assert response["aws_sagemaker"]["configured"] is False
    assert response["aws_sagemaker"]["missing_env"] == ["AWS_S3_BUCKET"]
    assert response["aws_sagemaker"]["region"] == "us-east-1"
    assert response["session_store"]["type"] in {"mongodb", "noop"}
    assert isinstance(response["session_store"]["durable"], bool)
    assert response["audit_store"]["enabled"] is True
    assert response["security"]["redaction_enabled"] is True
    assert response["security"]["sandbox_private_default"] is True
    assert response["security"]["secret_persistence_allowed"] is False
    assert "hf-secret" not in str(response)
