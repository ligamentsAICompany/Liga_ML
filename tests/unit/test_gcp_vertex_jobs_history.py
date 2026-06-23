from litellm import Message

from agent.core.agent_loop import (
    MAX_GCP_VERTEX_JOBS_HISTORY,
    _add_tool_message_to_context,
    _cap_gcp_vertex_jobs_history,
)


class FakeContextManager:
    def __init__(self):
        self.items: list[Message] = [Message(role="system", content="system")]

    def add_message(self, message: Message) -> None:
        self.items.append(message)


class FakeSession:
    def __init__(self):
        self.context_manager = FakeContextManager()


def _assistant_with_vertex_call(call_id: str) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "gcp_vertex_jobs",
                    "arguments": '{"operation":"inspect"}',
                },
            }
        ],
    )


def _vertex_tool_result(call_id: str, body: str = "ok") -> Message:
    return Message(
        role="tool",
        content=body,
        tool_call_id=call_id,
        name="gcp_vertex_jobs",
    )


def test_cap_gcp_vertex_jobs_history_keeps_latest_pairs():
    session = FakeSession()
    cm = session.context_manager

    for index in range(8):
        call_id = f"call-{index}"
        cm.add_message(_assistant_with_vertex_call(call_id))
        _add_tool_message_to_context(session, _vertex_tool_result(call_id, f"out-{index}"))

    vertex_tool_messages = [
        msg
        for msg in cm.items
        if getattr(msg, "role", None) == "tool" and msg.name == "gcp_vertex_jobs"
    ]
    vertex_calls = [
        tc
        for msg in cm.items
        if getattr(msg, "role", None) == "assistant"
        for tc in getattr(msg, "tool_calls", None) or []
        if tc.function.name == "gcp_vertex_jobs"
    ]

    assert len(vertex_tool_messages) == MAX_GCP_VERTEX_JOBS_HISTORY
    assert len(vertex_calls) == MAX_GCP_VERTEX_JOBS_HISTORY
    assert vertex_tool_messages[0].content == f"out-{8 - MAX_GCP_VERTEX_JOBS_HISTORY}"
    assert vertex_tool_messages[-1].content == "out-7"


def test_cap_gcp_vertex_jobs_history_preserves_other_tools():
    session = FakeSession()
    cm = session.context_manager

    cm.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "bash-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        )
    )
    cm.add_message(
        Message(role="tool", content="bash output", tool_call_id="bash-1", name="bash")
    )

    for index in range(7):
        call_id = f"vertex-{index}"
        cm.add_message(_assistant_with_vertex_call(call_id))
        _add_tool_message_to_context(session, _vertex_tool_result(call_id))

    bash_results = [
        msg for msg in cm.items if getattr(msg, "role", None) == "tool" and msg.name == "bash"
    ]
    assert len(bash_results) == 1

    _cap_gcp_vertex_jobs_history(session)
    bash_results = [
        msg for msg in cm.items if getattr(msg, "role", None) == "tool" and msg.name == "bash"
    ]
    assert len(bash_results) == 1
