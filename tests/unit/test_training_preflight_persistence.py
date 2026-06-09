"""Tests for Phase 7b preflight persistence and audit integration."""

import pytest

from agent.core.audit import training_preflight_audit_events
from agent.core.session_persistence import NoopSessionStore
from agent.core.training_preflight import run_local_training_preflight


def _result(session_id: str = "s1", run_id: str | None = "r1"):
    return run_local_training_preflight(
        session_id=session_id,
        run_id=run_id,
        recommendation={
            "provider": "hf-jobs",
            "recommended_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_policy": "cloud-and-hf-hub",
            "recommendation": {
                "selected_provider": {"provider_id": "hf-jobs"},
                "selected_model": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
                "selected_hardware": {"hardware_id": "hf-jobs:t4-small"},
                "output_policy": "cloud-and-hf-hub",
            },
        },
        dataset_summary={"rows": 10},
        metadata={"note": "HF_TOKEN=hf_" + "A" * 35},
    ).to_dict()


@pytest.mark.asyncio
async def test_noop_store_records_and_restores_session_and_run_preflight():
    store = NoopSessionStore()
    run = await store.create_run(session_id="s1", provider="hf-jobs")
    result = _result(run_id=run["run_id"])

    saved = await store.record_training_preflight(
        session_id="s1",
        preflight=result,
        run_id=run["run_id"],
    )

    latest = await store.get_latest_training_preflight("s1")
    by_run = await store.get_run_training_preflight("s1", run["run_id"])
    run_doc = await store.get_run(run["run_id"])

    assert saved["status"] == "unknown"
    assert latest["preflight_id"] == saved["preflight_id"]
    assert by_run["preflight_id"] == saved["preflight_id"]
    assert run_doc["training_preflight"]["preflight_id"] == saved["preflight_id"]
    assert run_doc["provider_metadata"]["training_preflight"]["status"] == "unknown"
    assert "hf_" not in str(saved)
    assert saved["metadata"]["provider_jobs_launched"] is False
    assert saved["metadata"]["resources_created"] is False


def test_training_preflight_audit_events_unknown_and_launch_blocked_are_sanitized():
    result = _result()

    events = training_preflight_audit_events(result, include_started=True)
    event_types = [event["event_type"] for event in events]

    assert "training_preflight_started" in event_types
    assert "training_preflight_unknown" in event_types
    assert "training_preflight_launch_blocked" in event_types
    assert "training_preflight_launch_ready" not in event_types
    assert "hf_" not in str(events)


def test_training_preflight_audit_events_failed_and_passed_paths_are_explicit():
    failed = dict(_result())
    failed.update(
        {
            "status": "failed",
            "launch_ready": False,
            "blocking_reasons": ["Required model id is missing."],
            "unknown_reasons": [],
        }
    )
    passed = dict(_result())
    passed.update(
        {
            "status": "passed",
            "launch_ready": True,
            "blocking_reasons": [],
            "unknown_reasons": [],
        }
    )

    failed_types = [
        event["event_type"] for event in training_preflight_audit_events(failed)
    ]
    passed_types = [
        event["event_type"] for event in training_preflight_audit_events(passed)
    ]

    assert "training_preflight_failed" in failed_types
    assert "training_preflight_launch_blocked" in failed_types
    assert "training_preflight_completed" in passed_types
    assert "training_preflight_launch_ready" in passed_types
