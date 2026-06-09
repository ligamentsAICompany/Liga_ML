import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from agent.core.background_runs import (  # noqa: E402
    background_run_status,
    load_background_run_settings,
)
from agent.core.session import Event, Session  # noqa: E402
from routes import agent  # noqa: E402


class _FakeConfig:
    model_name = "test-model"
    save_sessions = False
    session_dataset_repo = "fake/repo"
    auto_save_interval = 1
    heartbeat_interval_s = 60
    max_iterations = 10
    yolo_mode = False
    confirm_cpu_jobs = False
    auto_file_upload = False
    reasoning_effort = None
    mcpServers: dict = {}


class _EventStore:
    enabled = True

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    async def append_event(
        self,
        session_id: str,
        event_type: str,
        data: dict | None,
        run_id: str | None = None,
    ) -> int:
        self.events.append((session_id, event_type, data or {}, run_id))
        return len(self.events)


def _session_with_store(store: _EventStore) -> Session:
    context_manager = SimpleNamespace(items=[], on_message_added=None)
    return Session(
        event_queue=agent.asyncio.Queue(),
        config=_FakeConfig(),
        tool_router=None,
        context_manager=context_manager,
        session_id="s1",
        persistence_store=store,
    )


def test_background_runs_default_to_old_flow(monkeypatch):
    monkeypatch.delenv("BACKGROUND_RUNS_ENABLED", raising=False)
    monkeypatch.delenv("RUN_WORKER_MODE", raising=False)

    settings = load_background_run_settings()

    assert settings.enabled is False
    assert settings.worker_mode == "disabled"
    assert settings.in_process is False


def test_background_runs_accept_in_process_mode(monkeypatch):
    monkeypatch.setenv("BACKGROUND_RUNS_ENABLED", "true")
    monkeypatch.setenv("RUN_WORKER_MODE", "in_process")

    settings = load_background_run_settings()

    assert settings.enabled is True
    assert settings.worker_mode == "in_process"
    assert settings.in_process is True


def test_invalid_worker_mode_falls_back_to_disabled(monkeypatch):
    monkeypatch.setenv("BACKGROUND_RUNS_ENABLED", "true")
    monkeypatch.setenv("RUN_WORKER_MODE", "sidecar")

    settings = load_background_run_settings()

    assert settings.worker_mode == "disabled"
    assert settings.in_process is False
    assert "Invalid RUN_WORKER_MODE" in (settings.warning or "")


def test_background_run_status_reports_durable_in_process(monkeypatch):
    monkeypatch.setenv("BACKGROUND_RUNS_ENABLED", "true")
    monkeypatch.setenv("RUN_WORKER_MODE", "in_process")

    status = background_run_status({"type": "mongodb", "durable": True})

    assert status == {
        "enabled": True,
        "worker_mode": "in_process",
        "implemented": True,
        "durable": True,
        "store": "mongodb",
        "token_handoff_configured": False,
        "warning": None,
    }


def test_background_run_status_marks_external_worker_reserved(monkeypatch):
    monkeypatch.setenv("BACKGROUND_RUNS_ENABLED", "true")
    monkeypatch.setenv("RUN_WORKER_MODE", "external_worker")

    status = background_run_status({"type": "mongodb", "durable": True})

    assert status["enabled"] is True
    assert status["worker_mode"] == "external_worker"
    assert status["implemented"] is False
    assert status["durable"] is False
    assert status["token_handoff_configured"] is False
    assert "reserved" in status["warning"]


@pytest.mark.asyncio
async def test_health_includes_background_runs(monkeypatch):
    monkeypatch.setenv("BACKGROUND_RUNS_ENABLED", "true")
    monkeypatch.setenv("RUN_WORKER_MODE", "in_process")
    monkeypatch.setattr(
        agent.session_manager,
        "persistence_store",
        SimpleNamespace(enabled=True),
    )

    response = await agent.health_check()

    assert response.model_dump()["background_runs"] == {
        "enabled": True,
        "worker_mode": "in_process",
        "implemented": True,
        "durable": True,
        "store": "mongodb",
        "token_handoff_configured": False,
        "warning": None,
    }


@pytest.mark.asyncio
async def test_send_event_does_not_persist_events_when_disabled(monkeypatch):
    monkeypatch.setenv("BACKGROUND_RUNS_ENABLED", "false")
    monkeypatch.setenv("RUN_WORKER_MODE", "disabled")
    store = _EventStore()
    session = _session_with_store(store)

    await session.send_event(Event(event_type="processing", data={"step": "old"}))

    assert store.events == []


@pytest.mark.asyncio
async def test_send_event_persists_events_in_process(monkeypatch):
    monkeypatch.setenv("BACKGROUND_RUNS_ENABLED", "true")
    monkeypatch.setenv("RUN_WORKER_MODE", "in_process")
    store = _EventStore()
    session = _session_with_store(store)

    event = Event(event_type="processing", data={"step": "durable"})
    await session.send_event(event)

    assert store.events == [("s1", "processing", {"step": "durable"}, None)]
    assert event.seq == 1


@pytest.mark.asyncio
async def test_in_memory_run_events_replay_from_sequence():
    from agent.core.session_persistence import NoopSessionStore

    store = NoopSessionStore()
    run = await store.create_run(session_id="s1", provider="hf-jobs")
    run_id = run["run_id"]
    await store.append_run_event(
        run_id=run_id,
        session_id="s1",
        event_type="assistant_message",
        payload={"content": "hello"},
    )
    await store.append_run_event(
        run_id=run_id,
        session_id="s1",
        event_type="turn_complete",
        payload={"history_size": 2},
    )

    replay = await store.load_run_events_after(run_id, 1)

    assert [event["event_type"] for event in replay] == [
        "assistant_message",
        "turn_complete",
    ]
    assert replay[0]["seq"] == 2
    saved = await store.get_run(run_id)
    assert saved["status"] == "succeeded"
    assert saved["last_event_seq"] == 3


@pytest.mark.asyncio
async def test_in_memory_run_events_sanitize_secret_payloads():
    from agent.core.session_persistence import NoopSessionStore

    store = NoopSessionStore()
    run = await store.create_run(session_id="s1", provider="hf-jobs")
    run_id = run["run_id"]

    await store.append_run_event(
        run_id=run_id,
        session_id="s1",
        event_type="tool_output",
        payload={
            "tool": "bash",
            "output": "HF_TOKEN=hf_" + "A" * 35,
            "headers": {"Authorization": "Bearer " + "b" * 32},
        },
    )

    replay = await store.load_run_events_after(run_id, 0)
    serialized = str(replay)

    assert "hf_" not in serialized
    assert "Bearer b" not in serialized
    assert replay[-1]["payload"]["output"] == "HF_TOKEN=[REDACTED]"
    assert replay[-1]["payload"]["headers"]["Authorization"] == "Bearer [REDACTED]"


@pytest.mark.asyncio
async def test_approval_pending_and_provider_metadata_survive_replay():
    from agent.core.session_persistence import NoopSessionStore

    store = NoopSessionStore()
    run = await store.create_run(session_id="s1", provider="aws-sagemaker")
    run_id = run["run_id"]
    await store.append_run_event(
        run_id=run_id,
        session_id="s1",
        event_type="approval_required",
        payload={
            "tools": [
                {
                    "tool": "aws_sagemaker_jobs",
                    "tool_call_id": "tc1",
                    "approval_id": "approve-1",
                    "arguments": {"operation": "run"},
                }
            ]
        },
    )
    await store.append_run_event(
        run_id=run_id,
        session_id="s1",
        event_type="tool_state_change",
        payload={
            "tool": "aws_sagemaker_jobs",
            "tool_call_id": "tc1",
            "state": "running",
            "jobName": "train-job",
            "jobUrl": "https://console.aws.amazon.com/sagemaker/",
            "cloudWatchLogsUrl": "https://console.aws.amazon.com/cloudwatch/",
            "s3ModelArtifact": "s3://bucket/model.tar.gz",
        },
    )

    saved = await store.get_run(run_id)
    replay = await store.load_run_events_after(run_id, 0)

    assert saved["status"] == "waiting_provider"
    assert saved["approval_id"] == "approve-1"
    assert saved["active_provider_job_id"] == "train-job"
    assert saved["provider_metadata"]["provider_artifact_path"] == (
        "s3://bucket/model.tar.gz"
    )
    assert [event["event_type"] for event in replay] == [
        "run_created",
        "approval_required",
        "tool_state_change",
    ]


@pytest.mark.asyncio
async def test_run_interrupt_marks_interrupted():
    from agent.core.session_persistence import NoopSessionStore

    store = NoopSessionStore()
    run = await store.create_run(session_id="s1", provider="gcp-vertex")
    await store.append_run_event(
        run_id=run["run_id"],
        session_id="s1",
        event_type="interrupted",
        payload={"reason": "user_interrupt"},
    )

    saved = await store.get_run(run["run_id"])

    assert saved["status"] == "interrupted"
    assert saved["completed_at"] is not None
