import asyncio
from types import SimpleNamespace

import pytest
from litellm import Message

from agent.config import Config
from agent.core import agent_loop
from agent.core.agent_loop import Handlers, LLMResult, process_submission
from agent.core.session import OpType, Session


class EmptyToolRouter:
    def get_tool_specs_for_llm(self):
        return []

    async def call_tool(self, name, arguments, session=None, tool_call_id=None):
        raise AssertionError(f"unexpected tool call: {name}")


def _session() -> Session:
    return Session(
        asyncio.Queue(),
        Config.model_validate({"model_name": "openai/test", "save_sessions": False}),
        tool_router=EmptyToolRouter(),
        stream=False,
    )


async def _drain_events(session: Session):
    events = []
    while not session.event_queue.empty():
        events.append(await session.event_queue.get())
    return events


@pytest.mark.asyncio
async def test_empty_llm_response_emits_visible_error(monkeypatch):
    session = _session()

    async def fake_call_llm_non_streaming(session, messages, tools, llm_params):
        return LLMResult(
            content=None,
            tool_calls_acc={},
            token_count=0,
            finish_reason="stop",
        )

    monkeypatch.setattr(
        agent_loop, "_resolve_llm_params", lambda *_, **__: {"model": "openai/test"}
    )
    monkeypatch.setattr(
        agent_loop, "_call_llm_non_streaming", fake_call_llm_non_streaming
    )

    final = await Handlers.run_agent(session, "train a tiny model")

    events = await _drain_events(session)
    assert final is None
    assert any(
        event.event_type == "assistant_message"
        and "empty response" in (event.data or {}).get("content", "").lower()
        for event in events
    )
    assert any(
        event.event_type == "error"
        and "empty response" in (event.data or {}).get("error", "").lower()
        for event in events
    )
    assert not any(event.event_type == "turn_complete" for event in events)


@pytest.mark.asyncio
async def test_empty_llm_response_after_hf_planner_emits_visible_preflight_fallback(
    monkeypatch,
):
    session = _session()
    session.cloud_provider = "hf-jobs"
    session.training_goal = "production"
    session.output_policy = "hf-hub"
    session.uploaded_datasets = [
        {
            "repo_id": "owner/call-center-upload",
            "config_name": "normalized",
            "normalized_row_count": 42,
        }
    ]
    session.context_manager.add_message(
        Message(
            role="tool", name="training_planner", tool_call_id="call_plan", content="ok"
        )
    )

    async def fake_call_llm_non_streaming(session, messages, tools, llm_params):
        return LLMResult(
            content=None,
            tool_calls_acc={},
            token_count=0,
            finish_reason="stop",
        )

    monkeypatch.setattr(
        agent_loop, "_resolve_llm_params", lambda *_, **__: {"model": "openai/test"}
    )
    monkeypatch.setattr(
        agent_loop, "_call_llm_non_streaming", fake_call_llm_non_streaming
    )

    final = await Handlers.run_agent(session, "fine-tune this with Hugging Face Jobs")

    events = await _drain_events(session)
    assert final is not None
    message = next(
        (event.data or {}).get("content", "")
        for event in events
        if event.event_type == "assistant_message"
    )
    assert "hf_jobs" in message
    assert "approval card" in message
    assert "owner/call-center-upload" in message
    assert "gcp_vertex_jobs" not in message
    assert "aws_sagemaker_jobs" not in message
    assert not any(event.event_type == "error" for event in events)


@pytest.mark.asyncio
async def test_provider_quota_failure_emits_visible_retryable_error(monkeypatch):
    session = _session()

    async def fake_call_llm_non_streaming(session, messages, tools, llm_params):
        raise RuntimeError("403 quota/billing limit exceeded for provider")

    monkeypatch.setattr(
        agent_loop, "_resolve_llm_params", lambda *_, **__: {"model": "openai/test"}
    )
    monkeypatch.setattr(
        agent_loop, "_call_llm_non_streaming", fake_call_llm_non_streaming
    )

    final = await Handlers.run_agent(session, "use my uploaded dataset")

    events = await _drain_events(session)
    assert final is None
    message = next(
        (event.data or {}).get("content", "")
        for event in events
        if event.event_type == "assistant_message"
    )
    assert "quota" in message.lower() or "billing" in message.lower()
    assert "switch" in message.lower()
    assert any(
        event.event_type == "error" and (event.data or {}).get("error_type") == "quota"
        for event in events
    )
    assert not any(event.event_type == "turn_complete" for event in events)


@pytest.mark.asyncio
async def test_provider_spending_limit_failure_is_quota(monkeypatch):
    session = _session()

    async def fake_call_llm_non_streaming(session, messages, tools, llm_params):
        raise RuntimeError(
            "litellm.APIError: Error code: 403 - "
            "{'error': 'You have exceeded your monthly spending limit for Inference Providers.'}"
        )

    monkeypatch.setattr(
        agent_loop, "_resolve_llm_params", lambda *_, **__: {"model": "openai/test"}
    )
    monkeypatch.setattr(
        agent_loop, "_call_llm_non_streaming", fake_call_llm_non_streaming
    )

    final = await Handlers.run_agent(session, "use my uploaded dataset")

    events = await _drain_events(session)
    assert final is None
    message = next(
        (event.data or {}).get("content", "")
        for event in events
        if event.event_type == "assistant_message"
    )
    assert "quota" in message.lower() or "billing" in message.lower()
    assert "spending limit" in message.lower()
    assert any(
        event.event_type == "error" and (event.data or {}).get("error_type") == "quota"
        for event in events
    )


@pytest.mark.asyncio
async def test_gcp_provider_note_strongly_routes_training_to_vertex(monkeypatch):
    session = _session()
    seen_messages = []

    async def fake_run_agent(session, text):
        seen_messages.extend(session.context_manager.items)
        return "ok"

    monkeypatch.setattr(Handlers, "run_agent", fake_run_agent)
    submission = SimpleNamespace(
        operation=SimpleNamespace(
            op_type=OpType.USER_INPUT,
            data={"text": "fine tune this model", "cloud_provider": "gcp-vertex"},
        )
    )

    await process_submission(session, submission)

    note = "\n".join(str(getattr(message, "content", "")) for message in seen_messages)
    assert "gcp_vertex_jobs" in note
    assert "hf_jobs" in note
    assert "uploaded dataset" in note.lower()
    assert "approval" in note.lower()


@pytest.mark.asyncio
async def test_hf_provider_note_preserves_goal_policy_and_routes_to_hf(monkeypatch):
    session = _session()
    seen_messages = []

    async def fake_run_agent(session, text):
        seen_messages.extend(session.context_manager.items)
        return "ok"

    monkeypatch.setattr(Handlers, "run_agent", fake_run_agent)
    submission = SimpleNamespace(
        operation=SimpleNamespace(
            op_type=OpType.USER_INPUT,
            data={
                "text": "fine tune this model",
                "cloud_provider": "hf-jobs",
                "training_goal": "production",
                "output_policy": "hf-hub",
            },
        )
    )

    await process_submission(session, submission)

    note = "\n".join(str(getattr(message, "content", "")) for message in seen_messages)
    assert session.training_goal == "production"
    assert session.output_policy == "hf-hub"
    assert "training_goal=production" in note
    assert "output_policy=hf-hub" in note
    assert "hf_jobs" in note
    assert "gcp_vertex_jobs" in note
    assert "unless the user changes provider" in note
