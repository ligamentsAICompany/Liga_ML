"""Unit tests for the optional durable session store abstraction."""

import asyncio
from types import SimpleNamespace

import pytest

from agent.core.session import Event, Session
from agent.core.session_persistence import (
    MongoSessionStore,
    NoopSessionStore,
    _safe_message_doc,
)


@pytest.mark.asyncio
async def test_noop_store_keeps_local_cli_and_tests_db_free():
    store = NoopSessionStore()

    await store.init()
    await store.upsert_session(session_id="s1", user_id="u1", model="m")
    await store.save_snapshot(
        session_id="s1",
        user_id="u1",
        model="m",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert await store.load_session("s1") is None
    assert await store.list_sessions("u1") == []
    assert await store.append_event("s1", "processing", {}) is None
    assert await store.try_increment_quota("u1", "2099-01-01", 1) is None


@pytest.mark.asyncio
async def test_noop_store_persists_training_recommendation_from_tool_output():
    store = NoopSessionStore()
    run = await store.create_run(session_id="s1", provider="hf-jobs", request_id="r1")

    await store.append_run_event(
        run_id=run["run_id"],
        session_id="s1",
        event_type="tool_output",
        payload={
            "tool": "training_planner",
            "success": True,
            "structured": {
                "recommended_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "provider": "hf-jobs",
                "recommended_hardware": {"hardware_flavor": "t4-small"},
                "recommendation": {
                    "selected_model": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
                    "estimated_cost_usd": 0.6,
                    "recommended_evaluation_profile": "standard_static_review",
                },
            },
        },
    )

    saved = await store.get_run(run["run_id"])
    assert (
        saved["training_recommendation"]["recommended_model"]
        == "Qwen/Qwen2.5-0.5B-Instruct"
    )
    assert (
        saved["provider_metadata"]["training_recommendation"]["recommendation"][
            "estimated_cost_usd"
        ]
        == 0.6
    )


@pytest.mark.asyncio
async def test_noop_store_records_sanitized_training_preflight_for_session_and_run():
    store = NoopSessionStore()
    run = await store.create_run(session_id="s1", provider="hf-jobs", request_id="r1")
    preflight = {
        "preflight_id": "pf1",
        "session_id": "s1",
        "run_id": run["run_id"],
        "provider": "hf-jobs",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "hardware_id": "hf-jobs:t4-small",
        "output_policy": "cloud-and-hf-hub",
        "status": "unknown",
        "launch_ready": False,
        "safe_summary": "live checks not implemented",
        "blocking_reasons": [],
        "warning_reasons": [],
        "unknown_reasons": ["HF_TOKEN=hf_" + "A" * 35],
        "metadata": {"provider_jobs_launched": False, "resources_created": False},
    }

    await store.record_training_preflight(
        session_id="s1",
        run_id=run["run_id"],
        preflight=preflight,
    )

    latest = await store.get_latest_training_preflight("s1")
    by_run = await store.get_run_training_preflight("s1", run["run_id"])
    updated = await store.get_run(run["run_id"])

    assert latest["preflight_id"] == "pf1"
    assert by_run["preflight_id"] == "pf1"
    assert updated["provider_metadata"]["training_preflight"]["preflight_id"] == "pf1"
    assert "hf_" not in str(latest)


@pytest.mark.asyncio
async def test_session_send_event_persists_run_scoped_dataset_discovery(monkeypatch):
    monkeypatch.setattr("agent.core.session.background_runs_in_process", lambda: False)
    store = NoopSessionStore()
    run = await store.create_run(session_id="s1", provider="hf-jobs", request_id="r1")
    session = object.__new__(Session)
    session.session_id = "s1"
    session.current_run_id = run["run_id"]
    session.persistence_store = store
    session.event_queue = asyncio.Queue()
    session.logged_events = []
    session.config = SimpleNamespace(save_sessions=False)

    async def _no_notifications(_event):
        return None

    session._enqueue_auto_notification_requests = _no_notifications
    structured = {
        "query": "hardware support",
        "allowed_sources": ["huggingface"],
        "excluded_sources": ["kaggle"],
        "candidates": [
            {
                "dataset_id": "public/hardware-support",
                "title": "Hardware Support QA",
                "overall_score": 0.91,
            }
        ],
        "recommended_candidate": {"dataset_id": "public/hardware-support"},
        "warnings": ["User selection required before training."],
        "timestamp": "2026-06-10T00:00:00+00:00",
        "requires_user_selection": True,
    }

    await Session.send_event(
        session,
        Event(
            event_type="tool_call",
            data={
                "tool": "dataset_discovery",
                "tool_call_id": "tc1",
                "arguments": {"operation": "plan", "query": "hardware support"},
            },
        ),
    )
    await Session.send_event(
        session,
        Event(
            event_type="tool_output",
            data={
                "tool": "dataset_discovery",
                "tool_call_id": "tc1",
                "success": True,
                "structured": structured,
            },
        ),
    )

    saved = await store.get_run(run["run_id"])
    audits = await store.list_audit_events(session_id="s1", run_id=run["run_id"])

    assert saved["dataset_discovery"]["recommended_candidate"]["dataset_id"] == (
        "public/hardware-support"
    )
    assert saved["provider_metadata"]["dataset_discovery"]["timestamp"]
    assert {event["event_type"] for event in audits} >= {
        "dataset_discovery_started",
        "dataset_discovery_completed",
    }
    assert "hf_" not in str(saved)


def test_unsafe_message_payload_is_replaced_with_marker():
    marker = _safe_message_doc({"role": "assistant", "content": object()})

    assert marker["role"] == "tool"
    assert marker["ml_intern_persistence_error"] == "message_too_large_or_invalid"


# ── mark_pro_seen ─────────────────────────────────────────────────────────


class _FakeProUsers:
    """In-memory stand-in for the ``pro_users`` collection.

    Supports just enough of the Motor API to exercise ``mark_pro_seen``:
    ``update_one`` with ``$setOnInsert`` + ``$set`` + ``upsert=True``, and
    ``find_one_and_update`` with the guarded filter the conversion check uses.
    """

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}

    async def update_one(self, filt, update, upsert=False):
        _id = filt["_id"]
        doc = self.docs.get(_id)
        if doc is None and upsert:
            doc = dict(update.get("$setOnInsert") or {})
            self.docs[_id] = doc
        if doc is None:
            return
        for k, v in (update.get("$set") or {}).items():
            doc[k] = v

    async def find_one_and_update(self, filt, update, return_document=None):
        _id = filt["_id"]
        doc = self.docs.get(_id)
        if doc is None:
            return None
        # Guard checks the conversion test uses: ever_non_pro=True AND
        # first_seen_pro_at missing.
        for k, v in filt.items():
            if k == "_id":
                continue
            if isinstance(v, dict) and "$exists" in v:
                if v["$exists"] and k not in doc:
                    return None
                if not v["$exists"] and k in doc:
                    return None
            elif doc.get(k) != v:
                return None
        for k, v in (update.get("$set") or {}).items():
            doc[k] = v
        return dict(doc)


class _FakeDB:
    def __init__(self) -> None:
        self.pro_users = _FakeProUsers()


def _store_with_fake_db() -> MongoSessionStore:
    s = MongoSessionStore.__new__(MongoSessionStore)
    s.enabled = True
    s.db = _FakeDB()
    return s


@pytest.mark.asyncio
async def test_mark_pro_seen_returns_none_when_unknown_user_starts_pro():
    """Joining as Pro shouldn't count as a conversion."""
    store = _store_with_fake_db()
    assert await store.mark_pro_seen("u-new-pro", is_pro=True) is None


@pytest.mark.asyncio
async def test_mark_pro_seen_emits_conversion_after_seeing_user_as_free():
    store = _store_with_fake_db()
    assert await store.mark_pro_seen("u1", is_pro=False) is None
    result = await store.mark_pro_seen("u1", is_pro=True)
    assert result is not None
    assert result["converted"] is True
    assert isinstance(result["first_seen_at"], str)


@pytest.mark.asyncio
async def test_mark_pro_seen_only_fires_conversion_once():
    """Re-checking a converted user must not re-emit the event."""
    store = _store_with_fake_db()
    await store.mark_pro_seen("u1", is_pro=False)
    first = await store.mark_pro_seen("u1", is_pro=True)
    assert first is not None and first["converted"] is True
    second = await store.mark_pro_seen("u1", is_pro=True)
    assert second is None


@pytest.mark.asyncio
async def test_noop_store_mark_pro_seen_returns_none():
    store = NoopSessionStore()
    assert await store.mark_pro_seen("u1", is_pro=True) is None
    assert await store.mark_pro_seen("u1", is_pro=False) is None
