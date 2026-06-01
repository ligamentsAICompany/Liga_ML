import sys
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from routes import agent  # noqa: E402


@pytest.mark.asyncio
async def test_provider_health_returns_hf_and_gcp(monkeypatch):
    monkeypatch.setattr(
        agent,
        "build_gcp_vertex_readiness_snapshot",
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
    monkeypatch.setenv("HF_TOKEN", "hf-secret")

    response = await agent.provider_health()

    assert set(response) == {"hf_jobs", "gcp_vertex"}
    assert response["hf_jobs"]["configured"] is True
    assert response["hf_jobs"]["hf_token_configured"] is True
    assert response["gcp_vertex"]["configured"] is False
    assert response["gcp_vertex"]["missing_env"] == ["GOOGLE_CLOUD_PROJECT"]
    assert "hf-secret" not in str(response)
