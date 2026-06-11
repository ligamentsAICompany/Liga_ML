"""Tests for Phase 7b local training preflight API routes."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from agent.core.session_persistence import NoopSessionStore  # noqa: E402
from models import DatasetDiscoveryResponse, TrainingPreflightRequest  # noqa: E402
from routes import agent  # noqa: E402


def _recommendation() -> dict:
    return {
        "provider": "hf-jobs",
        "recommended_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "hardware_id": "hf-jobs:t4-small",
        "output_policy": "cloud-and-hf-hub",
        "recommendation": {
            "selected_provider": {"provider_id": "hf-jobs"},
            "selected_model": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
            "selected_hardware": {"hardware_id": "hf-jobs:t4-small"},
            "output_policy": "cloud-and-hf-hub",
        },
    }


def _vertex_recommendation() -> dict:
    return {
        "provider": "gcp-vertex",
        "recommended_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "hardware_id": "gcp-vertex:n1-standard-8-t4",
        "output_policy": "cloud-private",
        "recommendation": {
            "selected_provider": {"provider_id": "gcp-vertex"},
            "selected_model": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
            "selected_hardware": {"hardware_id": "gcp-vertex:n1-standard-8-t4"},
            "output_policy": "cloud-private",
        },
    }


def test_dataset_discovery_response_preserves_no_candidate_reason():
    response = DatasetDiscoveryResponse(
        candidates=[],
        no_candidates_reason="No safe public GST datasets were found.",
    )

    assert response.model_dump()["no_candidates_reason"] == (
        "No safe public GST datasets were found."
    )


@pytest.fixture()
def preflight_store(monkeypatch):
    store = NoopSessionStore()
    monkeypatch.setattr(agent.session_manager, "persistence_store", store)
    monkeypatch.setattr(agent.session_manager, "_store", lambda: store)
    return store


@pytest.fixture()
def allow_access(monkeypatch):
    calls: list[bool] = []

    async def _allow_access(session_id, user, request=None, preload_sandbox=True):
        calls.append(preload_sandbox)
        return SimpleNamespace(
            session_id=session_id,
            user_id=user["user_id"],
            session=SimpleNamespace(
                latest_training_recommendation=_recommendation(),
                latest_dataset_discovery={"selected_candidate": {"dataset_id": "d1"}},
            ),
            is_active=True,
        )

    monkeypatch.setattr(agent, "_check_session_access", _allow_access)
    return calls


@pytest.mark.asyncio
async def test_post_training_preflight_returns_and_persists_hf_token_failure(
    preflight_store,
    allow_access,
):
    run = await preflight_store.create_run(session_id="s1", provider="hf-jobs")

    response = await agent.run_training_preflight(
        TrainingPreflightRequest(
            session_id="s1",
            run_id=run["run_id"],
            recommendation=_recommendation(),
            dataset_summary={"rows": 10},
        ),
        user={"user_id": "dev"},
    )

    payload = response.model_dump()

    assert payload["status"] == "failed"
    assert payload["launch_ready"] is False
    assert payload["metadata"]["provider_jobs_launched"] is False
    assert payload["metadata"]["resources_created"] is False
    assert any(
        check["check_id"] == "hf.token.present" and check["status"] == "failed"
        for check in payload["primary"]["checks"]
    )
    assert allow_access == [False]
    assert await preflight_store.get_run_training_preflight("s1", run["run_id"])


@pytest.mark.asyncio
async def test_post_training_preflight_uses_latest_session_recommendation(
    preflight_store,
    allow_access,
):
    response = await agent.run_training_preflight(
        TrainingPreflightRequest(session_id="s1", dataset_summary={"rows": 10}),
        user={"user_id": "dev"},
    )

    assert response.provider == "hf-jobs"
    assert response.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert response.hardware_id == "hf-jobs:t4-small"
    assert response.output_policy == "cloud-and-hf-hub"
    assert response.launch_ready is False


@pytest.mark.asyncio
async def test_post_training_preflight_uses_latest_when_request_recommendation_incomplete(
    preflight_store,
    allow_access,
):
    response = await agent.run_training_preflight(
        TrainingPreflightRequest(
            session_id="s1",
            recommendation={"selected_provider": {"provider_id": "unknown"}},
            dataset_summary={"rows": 10},
        ),
        user={"user_id": "dev"},
    )

    assert response.provider == "hf-jobs"
    assert response.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert response.hardware_id == "hf-jobs:t4-small"
    assert response.launch_ready is False


@pytest.mark.asyncio
async def test_post_training_preflight_merges_partial_request_from_latest_vertex(
    preflight_store,
    monkeypatch,
):
    async def _allow_access(session_id, user, request=None, preload_sandbox=True):
        return SimpleNamespace(
            session_id=session_id,
            user_id=user["user_id"],
            session=SimpleNamespace(
                latest_training_recommendation=_vertex_recommendation(),
                latest_dataset_discovery=None,
            ),
            is_active=True,
        )

    monkeypatch.setattr(agent, "_check_session_access", _allow_access)

    response = await agent.run_training_preflight(
        TrainingPreflightRequest(
            session_id="s1",
            recommendation={
                "selected_provider": {"provider_id": "unknown"},
                "selected_model": {"model_id": "unknown"},
                "selected_hardware": {"hardware_id": None},
            },
            dataset_summary={"rows": 10},
        ),
        user={"user_id": "dev"},
    )
    saved = await preflight_store.get_latest_training_preflight("s1")

    assert response.provider == "gcp-vertex"
    assert response.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert response.hardware_id == "gcp-vertex:n1-standard-8-t4"
    assert response.output_policy == "cloud-private"
    assert response.metadata["provider_jobs_launched"] is False
    assert response.metadata["resources_created"] is False
    assert saved["provider"] == "gcp-vertex"
    assert saved["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert saved["hardware_id"] == "gcp-vertex:n1-standard-8-t4"
    assert any(
        check.provider == "gcp-vertex" and check.check_id.startswith("gcp.")
        for check in response.primary.checks
    )


@pytest.mark.asyncio
async def test_post_training_preflight_without_recommendation_returns_failed_result(
    preflight_store,
    monkeypatch,
):
    async def _allow_access(session_id, user, request=None, preload_sandbox=True):
        return SimpleNamespace(
            session_id=session_id,
            user_id=user["user_id"],
            session=SimpleNamespace(latest_training_recommendation=None),
        )

    async def _missing_recommendation(_session_id):
        return None

    monkeypatch.setattr(agent, "_check_session_access", _allow_access)
    monkeypatch.setattr(
        agent.session_manager,
        "get_latest_training_recommendation",
        _missing_recommendation,
    )

    response = await agent.run_training_preflight(
        TrainingPreflightRequest(session_id="s1"),
        user={"user_id": "dev"},
    )

    assert response.status == "failed"
    assert response.launch_ready is False
    assert any(
        "recommendation" in reason.lower() for reason in response.blocking_reasons
    )


@pytest.mark.asyncio
async def test_get_preflight_routes_return_latest_and_404_when_absent(
    preflight_store,
    allow_access,
):
    run = await preflight_store.create_run(session_id="s1", provider="hf-jobs")
    created = await agent.run_training_preflight(
        TrainingPreflightRequest(
            session_id="s1",
            run_id=run["run_id"],
            recommendation=_recommendation(),
            dataset_summary={"rows": 10},
        ),
        user={"user_id": "dev"},
    )

    session_response = await agent.get_session_preflight("s1", user={"user_id": "dev"})
    run_response = await agent.get_run_preflight(
        "s1", run["run_id"], user={"user_id": "dev"}
    )

    assert session_response.preflight_id == created.preflight_id
    assert run_response.preflight_id == created.preflight_id

    with pytest.raises(HTTPException) as exc_info:
        await agent.get_session_preflight("missing", user={"user_id": "dev"})
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_preflight_routes_enforce_session_access(monkeypatch):
    async def _deny(*_args, **_kwargs):
        raise HTTPException(status_code=403, detail="Access denied")

    monkeypatch.setattr(agent, "_check_session_access", _deny)

    with pytest.raises(HTTPException) as exc_info:
        await agent.run_training_preflight(
            TrainingPreflightRequest(session_id="s1", recommendation=_recommendation()),
            user={"user_id": "blocked"},
        )

    assert exc_info.value.status_code == 403
