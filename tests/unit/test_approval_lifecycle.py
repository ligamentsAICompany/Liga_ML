import asyncio
import json
from types import SimpleNamespace

import pytest
from litellm import ChatCompletionMessageToolCall as ToolCall

from agent.core import agent_loop
from agent.core.agent_loop import Handlers, _restore_pending_approval_from_snapshot
from agent.core.background_runs import run_status_from_event
from agent.core.session import Session


class _FakeToolRouter:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def get_tool_specs_for_llm(self):
        return []

    async def call_tool(self, tool_name, tool_args, session=None, tool_call_id=None):
        self.calls.append((tool_name, tool_args))
        return ("launched", True)


def _vertex_tool_call() -> ToolCall:
    return ToolCall(
        id="functions.gcp_vertex_jobs:4",
        type="function",
        function={
            "name": "gcp_vertex_jobs",
            "arguments": json.dumps(
                {
                    "operation": "run",
                    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
                    "dataset_name": "transitionGap/gst-india-preference-dataset-prep-small",
                    "training_goal": "smoke-test",
                    "output_policy": "cloud-private",
                    "machine_type": "n1-standard-8",
                    "accelerator_type": "NVIDIA_TESLA_T4",
                    "accelerator_count": 1,
                    "max_run_hours": 1,
                }
            ),
        },
    )


@pytest.mark.asyncio
async def test_exec_approval_restores_pending_from_snapshot_and_launches():
    events: list[str] = []
    session = Session(
        event_queue=SimpleNamespace(put_nowait=lambda _event: None),
        config=SimpleNamespace(
            model_name="openai/gpt-5.5",
            max_iterations=1,
            save_sessions=False,
        ),
        tool_router=_FakeToolRouter(),
        local_mode=True,
    )

    async def _send_event(event):
        events.append(event.event_type)

    session.send_event = _send_event  # type: ignore[method-assign]
    session.pending_approval = None
    session.pending_approval_snapshot = {
        "tool_calls": [_vertex_tool_call().model_dump(mode="json")],
        "approvals": [
            {
                "approval_id": "functions.gcp_vertex_jobs:4",
                "tool_call_id": "functions.gcp_vertex_jobs:4",
                "tool": "gcp_vertex_jobs",
                "operation": "run",
                "provider": "gcp-vertex",
                "status": "pending",
            }
        ],
    }
    session.context_manager = SimpleNamespace(items=[], add_message=lambda _msg: None)
    session._cancelled = asyncio.Event()
    session.increment_turn = lambda: None  # type: ignore[method-assign]
    session.auto_save_if_needed = lambda: None  # type: ignore[method-assign]

    async def fake_run_agent(_session, _text):
        return None

    original_run_agent = Handlers.run_agent
    Handlers.run_agent = staticmethod(fake_run_agent)  # type: ignore[method-assign]
    try:
        await Handlers.exec_approval(
            session,
            [
                {
                    "tool_call_id": "functions.gcp_vertex_jobs:4",
                    "approved": True,
                    "approval_id": "functions.gcp_vertex_jobs:4",
                }
            ],
        )
    finally:
        Handlers.run_agent = staticmethod(original_run_agent)  # type: ignore[method-assign]

    assert session.tool_router.calls
    assert session.tool_router.calls[0][0] == "gcp_vertex_jobs"
    assert "error" not in events


@pytest.mark.asyncio
async def test_duplicate_exec_approval_is_idempotent():
    session = Session(
        event_queue=SimpleNamespace(put_nowait=lambda _event: None),
        config=SimpleNamespace(
            model_name="openai/gpt-5.5",
            max_iterations=1,
            save_sessions=False,
        ),
        tool_router=_FakeToolRouter(),
        local_mode=True,
    )
    errors: list[str] = []

    async def _send_event(event):
        if event.event_type == "error":
            errors.append(str(event.data.get("error")))

    session.send_event = _send_event  # type: ignore[method-assign]
    session.pending_approval = None
    session.pending_approval_snapshot = None
    session.consumed_approval_tool_call_ids = {"functions.gcp_vertex_jobs:4"}

    await Handlers.exec_approval(
        session,
        [
            {
                "tool_call_id": "functions.gcp_vertex_jobs:4",
                "approved": True,
            }
        ],
    )

    assert session.tool_router.calls == []
    assert errors == []


def test_restore_pending_approval_from_snapshot_round_trip():
    session = SimpleNamespace(pending_approval=None)
    session.pending_approval_snapshot = {
        "tool_calls": [_vertex_tool_call().model_dump(mode="json")],
        "approvals": [{"tool_call_id": "functions.gcp_vertex_jobs:4"}],
    }

    restored = _restore_pending_approval_from_snapshot(session)  # type: ignore[arg-type]

    assert restored is not None
    assert session.pending_approval is restored
    assert restored["tool_calls"][0].id == "functions.gcp_vertex_jobs:4"


def test_bounded_vertex_prompt_skips_sandbox_preload():
    session = SimpleNamespace(
        skip_sandbox_preload=True,
        bounded_vertex_smoke_for_turn=True,
        compute_tools_blocked_for_turn=False,
        latest_user_prompt="",
    )

    assert agent_loop._should_skip_sandbox_preload(session) is True


def test_no_sandbox_prompt_blocks_sandbox_create_tool():
    session = SimpleNamespace(
        skip_sandbox_preload=True,
        bounded_vertex_smoke_for_turn=True,
        compute_tools_blocked_for_turn=False,
        latest_user_prompt="Do not create sandbox",
        training_planner_only_for_turn=False,
        cloud_provider="gcp-vertex",
        logged_events=[],
        context_manager=SimpleNamespace(items=[]),
    )

    violation = agent_loop._provider_tool_policy_violation(
        session,  # type: ignore[arg-type]
        "sandbox_create",
        {"hardware": "cpu-basic"},
    )

    assert violation is not None
    assert "no-sandbox" in violation.lower()


def test_turn_complete_does_not_mark_waiting_approval_run_succeeded():
    assert run_status_from_event(
        "turn_complete", {"waiting_for_tool_approval": True}
    ) == ("waiting_approval")
    assert run_status_from_event("turn_complete", {}) == "succeeded"


def test_turn_complete_marks_provider_launch_blocked_run_failed():
    assert (
        run_status_from_event(
            "turn_complete", {"run_outcome": "provider_launch_blocked"}
        )
        == "failed"
    )


def test_tool_state_change_blocked_marks_run_failed():
    assert (
        run_status_from_event(
            "tool_state_change",
            {"tool": "gcp_vertex_jobs", "state": "blocked"},
        )
        == "failed"
    )
