from types import SimpleNamespace

import pytest
from litellm.types.utils import ChatCompletionMessageToolCall

from agent.tools import research_tool


class FakeToolRouter:
    def get_tool_specs_for_llm(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    async def call_tool(self, tool_name, arguments, session=None, tool_call_id=None):
        return "still searching", True


class FakeSession:
    def __init__(self):
        self.config = SimpleNamespace(
            model_name="openai/gpt-5.5", reasoning_effort=None
        )
        self.tool_router = FakeToolRouter()
        self.hf_token = None
        self.events = []

    async def send_event(self, event):
        self.events.append(event)


def _tool_call_response():
    tool_call = ChatCompletionMessageToolCall(
        id="call_search",
        function={
            "name": "web_search",
            "arguments": '{"query":"manufacturing qa dataset"}',
        },
        type="function",
    )
    message = SimpleNamespace(content="", tool_calls=[tool_call])
    return SimpleNamespace(
        usage=SimpleNamespace(total_tokens=100),
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
    )


def _summary_response():
    message = SimpleNamespace(content="Partial manufacturing dataset findings.")
    return SimpleNamespace(
        usage=SimpleNamespace(total_tokens=200),
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
    )


@pytest.mark.asyncio
async def test_research_agent_wall_clock_deadline_forces_partial_summary(
    monkeypatch,
):
    calls = []

    async def fake_acompletion(*, tools=None, **kwargs):
        calls.append(tools)
        if tools is None:
            return _summary_response()
        return _tool_call_response()

    times = iter([0.0, 0.1, 2.0, 2.1, 2.2])

    monkeypatch.setattr(research_tool, "acompletion", fake_acompletion)
    monkeypatch.setattr(research_tool.time, "monotonic", lambda: next(times, 2.2))
    monkeypatch.setattr(
        research_tool, "_RESEARCH_WALL_CLOCK_MAX_SECONDS", 1.0, raising=False
    )
    monkeypatch.setattr(
        research_tool.telemetry, "record_llm_call", lambda *args, **kwargs: None
    )

    output, ok = await research_tool.research_handler(
        {"task": "Find manufacturing datasets."},
        session=FakeSession(),
        tool_call_id="call_research",
    )

    assert ok is True
    assert output == "Partial manufacturing dataset findings."
    assert calls[-1] is None
    assert len([call for call in calls if call is not None]) == 1
