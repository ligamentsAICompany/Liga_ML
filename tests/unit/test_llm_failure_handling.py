import asyncio
import json
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


class VertexTerminalToolRouter:
    def __init__(self):
        self.calls = []

    def get_tool_specs_for_llm(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "gcp_vertex_jobs",
                    "description": "Inspect Vertex jobs",
                    "parameters": {"type": "object"},
                },
            }
        ]

    async def call_tool(self, name, arguments, session=None, tool_call_id=None):
        self.calls.append((name, arguments, tool_call_id))
        await session.send_event(
            agent_loop.Event(
                event_type="tool_state_change",
                data={
                    "tool": "gcp_vertex_jobs",
                    "tool_call_id": tool_call_id,
                    "state": "failed",
                    "jobName": "projects/test/locations/us/customJobs/123",
                },
            )
        )
        return (
            "**Vertex AI job details:**\n\n"
            "**Job:** `projects/test/locations/us/customJobs/123`\n"
            "**State:** JOB_STATE_FAILED\n"
            "**Failure reason:** missing runtime token",
            True,
        )


def _session() -> Session:
    return Session(
        asyncio.Queue(),
        Config.model_validate({"model_name": "openai/test", "save_sessions": False}),
        tool_router=EmptyToolRouter(),
        stream=False,
    )


def _session_with_router(router, model_name: str = "openai/test") -> Session:
    return Session(
        asyncio.Queue(),
        Config.model_validate({"model_name": model_name, "save_sessions": False}),
        tool_router=router,
        stream=False,
    )


async def _drain_events(session: Session):
    events = []
    while not session.event_queue.empty():
        events.append(await session.event_queue.get())
    return events


def test_kimi_text_only_messages_drop_stale_image_placeholders():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "user",
            "content": "Run an HF Jobs smoke test <image>.",
            "images": [],
        },
        {"role": "assistant", "content": "Prior answer with <|image_1|> marker."},
    ]

    sanitized = agent_loop._sanitize_messages_for_model(
        messages, "openai/moonshotai/Kimi-K2.6"
    )

    dumped = [
        message.model_dump(mode="json") if hasattr(message, "model_dump") else message
        for message in sanitized
    ]
    payload_text = json.dumps(dumped)
    assert "<image>" not in payload_text
    assert "<|image_1|>" not in payload_text
    assert "images" not in dumped[1]
    assert dumped[1]["content"] == "Run an HF Jobs smoke test ."


def test_kimi_sanitizer_preserves_actual_image_parts():
    messages = [
        {
            "role": "user",
            "content": "Describe this image.",
            "images": [{"url": "data:image/png;base64,abc123"}],
        }
    ]

    sanitized = agent_loop._sanitize_messages_for_model(
        messages, "openai/moonshotai/Kimi-K2.6"
    )

    assert sanitized == messages


def test_kimi_sanitizer_omits_empty_image_content_parts():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Run HF Jobs <|image_1|> smoke test."},
                {"type": "image_url", "image_url": {}},
                {"type": "input_image", "image_url": ""},
            ],
        }
    ]

    sanitized = agent_loop._sanitize_messages_for_model(
        messages, "openai/moonshotai/Kimi-K2.6"
    )

    assert sanitized == [{"role": "user", "content": "Run HF Jobs  smoke test."}]
    assert "image" not in json.dumps(sanitized).lower()


def test_kimi_sanitizer_leaves_non_kimi_models_unchanged():
    messages = [
        {
            "role": "user",
            "content": "Keep provider-specific placeholders <image> unchanged.",
            "images": [],
        }
    ]

    sanitized = agent_loop._sanitize_messages_for_model(messages, "openai/gpt-5.5")

    assert sanitized is messages


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
async def test_vertex_terminal_failure_ends_turn_without_extra_llm_retry(monkeypatch):
    router = VertexTerminalToolRouter()
    session = _session_with_router(router)
    session.cloud_provider = "gcp-vertex"
    calls = 0

    async def fake_call_llm_non_streaming(session, messages, tools, llm_params):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("terminal provider failure should end the turn")
        return LLMResult(
            content=None,
            tool_calls_acc={
                0: {
                    "id": "call_vertex",
                    "function": {
                        "name": "gcp_vertex_jobs",
                        "arguments": json.dumps(
                            {
                                "operation": "inspect",
                                "job_name": "projects/test/locations/us/customJobs/123",
                            }
                        ),
                    },
                }
            },
            token_count=10,
            finish_reason="tool_calls",
        )

    monkeypatch.setattr(
        agent_loop, "_resolve_llm_params", lambda *_, **__: {"model": "openai/test"}
    )
    monkeypatch.setattr(
        agent_loop, "_call_llm_non_streaming", fake_call_llm_non_streaming
    )

    await Handlers.run_agent(session, "check the Vertex job")

    events = await _drain_events(session)
    assert calls == 1
    assert router.calls == [
        (
            "gcp_vertex_jobs",
            {
                "operation": "inspect",
                "job_name": "projects/test/locations/us/customJobs/123",
            },
            "call_vertex",
        )
    ]
    assert any(
        event.event_type == "tool_state_change"
        and (event.data or {}).get("state") == "failed"
        for event in events
    )
    assert any(event.event_type == "turn_complete" for event in events)


@pytest.mark.asyncio
async def test_bounded_vertex_run_without_preflight_marks_run_blocked(monkeypatch):
    session = _session()
    session.cloud_provider = "gcp-vertex"
    session.training_goal = "smoke-test"
    session.bounded_vertex_smoke_for_turn = True
    calls = 0

    async def fake_call_llm_non_streaming(session, messages, tools, llm_params):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("blocked provider launch should end the turn")
        return LLMResult(
            content=None,
            tool_calls_acc={
                0: {
                    "id": "call_vertex_run",
                    "function": {
                        "name": "gcp_vertex_jobs",
                        "arguments": json.dumps({"operation": "run"}),
                    },
                }
            },
            token_count=10,
            finish_reason="tool_calls",
        )

    monkeypatch.setattr(
        agent_loop, "_resolve_llm_params", lambda *_, **__: {"model": "openai/test"}
    )
    monkeypatch.setattr(
        agent_loop, "_call_llm_non_streaming", fake_call_llm_non_streaming
    )

    await Handlers.run_agent(session, "launch bounded vertex smoke")

    events = await _drain_events(session)
    assert calls == 1
    assert any(
        event.event_type == "tool_state_change"
        and (event.data or {}).get("state") == "blocked"
        for event in events
    )
    assert any(
        event.event_type == "turn_complete"
        and (event.data or {}).get("run_outcome") == "provider_launch_blocked"
        for event in events
    )
    assert session.provider_launch_blocked_for_turn is False


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


@pytest.mark.asyncio
async def test_aws_provider_note_routes_training_to_sagemaker_tool(monkeypatch):
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
                "cloud_provider": "aws-sagemaker",
            },
        )
    )

    await process_submission(session, submission)

    note = "\n".join(str(getattr(message, "content", "")) for message in seen_messages)
    assert "AWS SageMaker AI" in note
    assert "aws_sagemaker_jobs" in note
    assert "stages normalized datasets to S3" in note
    assert "can submit SageMaker training jobs" in note
    assert "training image config" in note
    assert "training_planner" in note
    assert "skip broad literature/research crawls" in note
    assert "do not stop after planning" in note
    assert "do not route to Hugging Face Jobs or Google Cloud Vertex AI" in note
    assert "uploaded dataset" in note.lower()
    assert "approval-gated" in note.lower()


@pytest.mark.asyncio
async def test_hf_provider_note_does_not_include_aws_routing(monkeypatch):
    session = _session()
    seen_messages = []

    async def fake_run_agent(session, text):
        seen_messages.extend(session.context_manager.items)
        return "ok"

    monkeypatch.setattr(Handlers, "run_agent", fake_run_agent)
    submission = SimpleNamespace(
        operation=SimpleNamespace(
            op_type=OpType.USER_INPUT,
            data={"text": "fine tune this model", "cloud_provider": "hf-jobs"},
        )
    )

    await process_submission(session, submission)

    provider_notes = [
        str(getattr(message, "content", "")) for message in seen_messages[1:]
    ]
    note = "\n".join(provider_notes)
    assert "Hugging Face Jobs" in note
    assert "AWS SageMaker AI" not in note
    assert "aws_sagemaker_jobs" not in note
