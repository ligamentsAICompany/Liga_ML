"""Tests for premium model handling in backend/routes/agent.py."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from routes import agent  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_quota_store():
    agent.user_quotas._reset_for_tests()
    yield
    agent.user_quotas._reset_for_tests()


def test_premium_model_predicate_includes_bedrock_claude_and_gpt55_only():
    assert agent._is_premium_model("bedrock/us.anthropic.claude-opus-4-6-v1")
    assert agent._is_premium_model("openai/gpt-5.5")
    assert not agent._is_premium_model("anthropic/claude-opus-4-6")
    assert not agent._is_premium_model("moonshotai/Kimi-K2.6")


@pytest.mark.parametrize("cloud_provider", ["hf-jobs", "gcp-vertex", "aws-sagemaker"])
def test_submit_request_accepts_known_training_providers(cloud_provider):
    request = agent.SubmitRequest(
        session_id="s1",
        text="fine tune this model",
        cloud_provider=cloud_provider,
    )

    assert request.cloud_provider == cloud_provider


def test_submit_request_rejects_unknown_training_provider():
    with pytest.raises(agent.ValidationError):
        agent.SubmitRequest(
            session_id="s1",
            text="fine tune this model",
            cloud_provider="unknown-provider",
        )


@pytest.mark.asyncio
async def test_default_premium_session_falls_back_to_free_model(monkeypatch):
    monkeypatch.setattr(
        agent.session_manager.config,
        "model_name",
        agent.DEFAULT_CLAUDE_MODEL_ID,
    )

    model = await agent._model_override_for_new_session(None, None)

    assert model == agent.DEFAULT_FREE_MODEL_ID


@pytest.mark.asyncio
async def test_default_free_session_keeps_config_default(monkeypatch):
    monkeypatch.setattr(
        agent.session_manager.config,
        "model_name",
        agent.DEFAULT_FREE_MODEL_ID,
    )

    model = await agent._model_override_for_new_session(None, None)

    assert model is None


@pytest.mark.asyncio
async def test_explicit_premium_session_allowed_for_authenticated_user():
    model = await agent._model_override_for_new_session(
        None,
        agent.DEFAULT_CLAUDE_MODEL_ID,
    )

    assert model == agent.DEFAULT_CLAUDE_MODEL_ID


@pytest.mark.asyncio
async def test_switching_to_premium_model_is_allowed_for_authenticated_user(
    monkeypatch,
):
    updated = []

    async def fake_check_session_access(session_id, user, request=None):
        assert session_id == "s1"
        assert user["user_id"] == "u1"
        return SimpleNamespace(user_id="u1")

    async def fake_update_session_model(session_id, model_id):
        updated.append((session_id, model_id))

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)
    monkeypatch.setattr(
        agent.session_manager,
        "update_session_model",
        fake_update_session_model,
    )

    response = await agent.set_session_model(
        "s1",
        {"model": "openai/gpt-5.5"},
        request=None,
        user={"user_id": "u1", "plan": "free"},
    )

    assert response == {"session_id": "s1", "model": "openai/gpt-5.5"}
    assert updated == [("s1", "openai/gpt-5.5")]


@pytest.mark.asyncio
async def test_switching_cloud_provider_persists_for_session(monkeypatch):
    updated = []

    async def fake_check_session_access(session_id, user, request=None):
        assert session_id == "s1"
        assert user["user_id"] == "u1"
        return SimpleNamespace(user_id="u1")

    async def fake_update_session_cloud_provider(
        session_id,
        cloud_provider,
        training_goal=None,
        output_policy=None,
    ):
        updated.append((session_id, cloud_provider))
        return True

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)
    monkeypatch.setattr(
        agent.session_manager,
        "update_session_cloud_provider",
        fake_update_session_cloud_provider,
    )

    response = await agent.set_session_cloud_provider(
        "s1",
        {"cloud_provider": "gcp-vertex"},
        request=None,
        user={"user_id": "u1", "plan": "free"},
    )

    assert response == {
        "session_id": "s1",
        "cloud_provider": "gcp-vertex",
        "training_goal": "agent-decide",
        "output_policy": "cloud-private",
    }
    assert updated == [("s1", "gcp-vertex")]


@pytest.mark.asyncio
async def test_switching_cloud_provider_persists_gcloud_preflight_options(monkeypatch):
    updated = []

    async def fake_check_session_access(session_id, user, request=None):
        assert session_id == "s1"
        assert user["user_id"] == "u1"
        return SimpleNamespace(user_id="u1")

    async def fake_update_session_cloud_provider(
        session_id,
        cloud_provider,
        training_goal=None,
        output_policy=None,
    ):
        updated.append((session_id, cloud_provider, training_goal, output_policy))
        return True

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)
    monkeypatch.setattr(
        agent.session_manager,
        "update_session_cloud_provider",
        fake_update_session_cloud_provider,
    )

    response = await agent.set_session_cloud_provider(
        "s1",
        {
            "cloud_provider": "gcp-vertex",
            "training_goal": "smoke-test",
            "output_policy": "cloud-private",
        },
        request=None,
        user={"user_id": "u1", "plan": "free"},
    )

    assert response == {
        "session_id": "s1",
        "cloud_provider": "gcp-vertex",
        "training_goal": "smoke-test",
        "output_policy": "cloud-private",
    }
    assert updated == [("s1", "gcp-vertex", "smoke-test", "cloud-private")]


@pytest.mark.asyncio
async def test_switching_cloud_provider_accepts_aws_sagemaker(monkeypatch):
    updated = []

    async def fake_check_session_access(session_id, user, request=None):
        assert session_id == "s1"
        assert user["user_id"] == "u1"
        return SimpleNamespace(user_id="u1")

    async def fake_update_session_cloud_provider(
        session_id,
        cloud_provider,
        training_goal=None,
        output_policy=None,
    ):
        updated.append((session_id, cloud_provider, training_goal, output_policy))
        return True

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)
    monkeypatch.setattr(
        agent.session_manager,
        "update_session_cloud_provider",
        fake_update_session_cloud_provider,
    )

    response = await agent.set_session_cloud_provider(
        "s1",
        {"cloud_provider": "aws-sagemaker"},
        request=None,
        user={"user_id": "u1", "plan": "free"},
    )

    assert response == {
        "session_id": "s1",
        "cloud_provider": "aws-sagemaker",
        "training_goal": "agent-decide",
        "output_policy": "cloud-private",
    }
    assert updated == [("s1", "aws-sagemaker", "agent-decide", "cloud-private")]


@pytest.mark.asyncio
async def test_switching_cloud_provider_rejects_unknown_provider(monkeypatch):
    async def fake_check_session_access(session_id, user, request=None):
        return SimpleNamespace(user_id=user["user_id"])

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)

    with pytest.raises(HTTPException) as exc_info:
        await agent.set_session_cloud_provider(
            "s1",
            {"cloud_provider": "not-a-provider"},
            request=None,
            user={"user_id": "u1", "plan": "free"},
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_premium_quota_charges_gpt55(monkeypatch):
    persisted = []

    async def fake_persist_session_snapshot(agent_session):
        persisted.append(agent_session)

    monkeypatch.setattr(
        agent.session_manager,
        "persist_session_snapshot",
        fake_persist_session_snapshot,
    )

    agent_session = SimpleNamespace(
        claude_counted=False,
        session=SimpleNamespace(
            config=SimpleNamespace(model_name="openai/gpt-5.5"),
        ),
    )

    await agent._enforce_premium_model_quota(
        {"user_id": "u1", "plan": "free"},
        agent_session,
    )

    assert agent_session.claude_counted is True
    assert persisted == [agent_session]
    assert await agent.user_quotas.get_claude_used_today("u1") == 1


@pytest.mark.asyncio
async def test_free_user_premium_quota_rejects_second_session(monkeypatch):
    async def fake_persist_session_snapshot(_agent_session):
        return None

    monkeypatch.setattr(
        agent.session_manager,
        "persist_session_snapshot",
        fake_persist_session_snapshot,
    )

    first_session = SimpleNamespace(
        claude_counted=False,
        session=SimpleNamespace(
            config=SimpleNamespace(model_name="openai/gpt-5.5"),
        ),
    )
    second_session = SimpleNamespace(
        claude_counted=False,
        session=SimpleNamespace(
            config=SimpleNamespace(model_name="openai/gpt-5.5"),
        ),
    )

    await agent._enforce_premium_model_quota(
        {"user_id": "free-user", "plan": "free"},
        first_session,
    )
    with pytest.raises(HTTPException) as exc_info:
        await agent._enforce_premium_model_quota(
            {"user_id": "free-user", "plan": "free"},
            second_session,
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["error"] == "premium_model_daily_cap"
    assert exc_info.value.detail["plan"] == "free"


@pytest.mark.asyncio
async def test_pro_user_uses_pro_premium_quota(monkeypatch):
    async def fake_persist_session_snapshot(_agent_session):
        return None

    monkeypatch.setattr(
        agent.session_manager,
        "persist_session_snapshot",
        fake_persist_session_snapshot,
    )

    for index in range(2):
        agent_session = SimpleNamespace(
            claude_counted=False,
            session=SimpleNamespace(
                config=SimpleNamespace(model_name="openai/gpt-5.5"),
            ),
        )
        await agent._enforce_premium_model_quota(
            {"user_id": "pro-user", "plan": "pro"},
            agent_session,
        )
        assert agent_session.claude_counted is True
        assert await agent.user_quotas.get_claude_used_today("pro-user") == index + 1


@pytest.mark.asyncio
async def test_org_plan_uses_free_premium_quota(monkeypatch):
    async def fake_persist_session_snapshot(_agent_session):
        return None

    monkeypatch.setattr(
        agent.session_manager,
        "persist_session_snapshot",
        fake_persist_session_snapshot,
    )

    first_session = SimpleNamespace(
        claude_counted=False,
        session=SimpleNamespace(
            config=SimpleNamespace(model_name="openai/gpt-5.5"),
        ),
    )
    second_session = SimpleNamespace(
        claude_counted=False,
        session=SimpleNamespace(
            config=SimpleNamespace(model_name="openai/gpt-5.5"),
        ),
    )

    await agent._enforce_premium_model_quota(
        {"user_id": "org-user", "plan": "org"},
        first_session,
    )
    with pytest.raises(HTTPException) as exc_info:
        await agent._enforce_premium_model_quota(
            {"user_id": "org-user", "plan": "org"},
            second_session,
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["plan"] == "org"
    assert "Upgrade to HF Pro" in exc_info.value.detail["message"]


@pytest.mark.asyncio
async def test_premium_quota_skips_direct_anthropic(monkeypatch):
    async def fail_if_persisted(_agent_session):
        raise AssertionError("direct Anthropic should not consume premium quota")

    monkeypatch.setattr(
        agent.session_manager,
        "persist_session_snapshot",
        fail_if_persisted,
    )

    agent_session = SimpleNamespace(
        claude_counted=False,
        session=SimpleNamespace(
            config=SimpleNamespace(model_name="anthropic/claude-opus-4-6"),
        ),
    )

    await agent._enforce_premium_model_quota(
        {"user_id": "u1", "plan": "free"},
        agent_session,
    )

    assert agent_session.claude_counted is False
    assert await agent.user_quotas.get_claude_used_today("u1") == 0


@pytest.mark.asyncio
async def test_user_quota_response_uses_premium_fields_only(monkeypatch):
    async def fake_get_used_today(user_id):
        assert user_id == "u1"
        return 2

    monkeypatch.setattr(agent.user_quotas, "get_claude_used_today", fake_get_used_today)
    monkeypatch.setattr(agent.user_quotas, "daily_cap_for", lambda plan: 5)

    response = await agent.get_user_quota({"user_id": "u1", "plan": "pro"})

    assert response == {
        "plan": "pro",
        "premium_used_today": 2,
        "premium_daily_cap": 5,
        "premium_remaining": 3,
    }


@pytest.mark.asyncio
async def test_set_session_yolo_calls_manager_with_cap_presence(monkeypatch):
    async def fake_check_session_access(session_id, user, request=None):
        assert session_id == "s1"
        assert user["user_id"] == "u1"
        return object()

    calls = []

    async def fake_update_session_auto_approval(session_id, **kwargs):
        calls.append((session_id, kwargs))
        return {
            "enabled": kwargs["enabled"],
            "cost_cap_usd": 7.5,
            "estimated_spend_usd": 0.0,
            "remaining_usd": 7.5,
        }

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)
    monkeypatch.setattr(
        agent.session_manager,
        "update_session_auto_approval",
        fake_update_session_auto_approval,
    )

    response = await agent.set_session_yolo(
        "s1",
        agent.SessionYoloRequest(enabled=True, cost_cap_usd=7.5),
        {"user_id": "u1"},
    )

    assert response["enabled"] is True
    assert response["remaining_usd"] == 7.5
    assert calls == [
        (
            "s1",
            {
                "enabled": True,
                "cost_cap_usd": 7.5,
                "cap_provided": True,
            },
        )
    ]


@pytest.mark.asyncio
async def test_delete_session_access_check_skips_sandbox_preload(monkeypatch):
    ensure_calls = []
    delete_calls = []

    async def fake_ensure_session_loaded(session_id, user_id, **kwargs):
        ensure_calls.append((session_id, user_id, kwargs))
        return SimpleNamespace(user_id=user_id)

    async def fake_delete_session(session_id):
        delete_calls.append(session_id)
        return True

    monkeypatch.setattr(
        agent.session_manager,
        "ensure_session_loaded",
        fake_ensure_session_loaded,
    )
    monkeypatch.setattr(agent.session_manager, "delete_session", fake_delete_session)

    response = await agent.delete_session("s1", {"user_id": "u1"})

    assert response == {"status": "deleted", "session_id": "s1"}
    assert delete_calls == ["s1"]
    assert ensure_calls[0][2]["preload_sandbox"] is False


@pytest.mark.asyncio
async def test_get_session_access_check_skips_sandbox_preload(monkeypatch):
    ensure_calls = []

    async def fake_ensure_session_loaded(session_id, user_id, **kwargs):
        ensure_calls.append((session_id, user_id, kwargs))
        return SimpleNamespace(user_id=user_id)

    monkeypatch.setattr(
        agent.session_manager,
        "ensure_session_loaded",
        fake_ensure_session_loaded,
    )
    monkeypatch.setattr(
        agent.session_manager,
        "get_session_info",
        lambda session_id: {
            "session_id": session_id,
            "created_at": "2026-06-10T00:00:00Z",
            "is_active": True,
            "is_processing": False,
            "message_count": 0,
            "user_id": "u1",
        },
    )

    response = await agent.get_session("s1", {"user_id": "u1"})

    assert response.session_id == "s1"
    assert ensure_calls[0][2]["preload_sandbox"] is False


@pytest.mark.asyncio
async def test_teardown_session_access_check_skips_sandbox_preload(monkeypatch):
    ensure_calls = []
    teardown_calls = []

    async def fake_ensure_session_loaded(session_id, user_id, **kwargs):
        ensure_calls.append((session_id, user_id, kwargs))
        return SimpleNamespace(user_id=user_id)

    async def fake_teardown_sandbox(session_id):
        teardown_calls.append(session_id)
        return True

    monkeypatch.setattr(
        agent.session_manager,
        "ensure_session_loaded",
        fake_ensure_session_loaded,
    )
    monkeypatch.setattr(
        agent.session_manager, "teardown_sandbox", fake_teardown_sandbox
    )

    response = await agent.teardown_session_sandbox("s1", {"user_id": "u1"})
    await asyncio.sleep(0)

    assert response == {"status": "teardown_requested", "session_id": "s1"}
    assert teardown_calls == ["s1"]
    assert ensure_calls[0][2]["preload_sandbox"] is False


@pytest.mark.asyncio
async def test_create_session_capacity_response_includes_actionable_metadata(
    monkeypatch,
):
    error = agent.SessionCapacityError(
        "You have too many active sessions. Clear stale inactive sessions or "
        "delete old sessions to continue.",
        error_type="per_user",
    )
    error.capacity = {
        "active_sessions": 10,
        "max_sessions": 10,
        "error_type": "per_user",
        "cleanup": {"cleared": 0, "skipped": 10},
    }

    async def fake_create_session(**_kwargs):
        raise error

    async def request_json():
        return {}

    monkeypatch.setattr(agent.session_manager, "create_session", fake_create_session)
    monkeypatch.setattr(agent, "resolve_hf_request_token", lambda _request: None)

    with pytest.raises(HTTPException) as exc_info:
        await agent.create_session(
            SimpleNamespace(json=request_json),
            {"user_id": "u1", "username": "alice", "plan": "free"},
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {
        "error": "session_capacity",
        "message": (
            "You have too many active sessions. Clear stale inactive sessions or "
            "delete old sessions to continue."
        ),
        "active_sessions": 10,
        "max_sessions": 10,
        "error_type": "per_user",
        "cleanup": {"cleared": 0, "skipped": 10},
    }


@pytest.mark.asyncio
async def test_create_session_route_disables_automatic_sandbox_preload(monkeypatch):
    captured: dict = {}

    async def fake_create_session(**kwargs):
        captured.update(kwargs)
        return "session-1"

    async def request_json():
        return {}

    monkeypatch.setattr(agent.session_manager, "create_session", fake_create_session)
    monkeypatch.setattr(agent, "resolve_hf_request_token", lambda _request: None)

    response = await agent.create_session(
        SimpleNamespace(json=request_json),
        {"user_id": "u1", "username": "alice", "plan": "free"},
    )

    assert response.session_id == "session-1"
    assert captured["preload_sandbox"] is False
