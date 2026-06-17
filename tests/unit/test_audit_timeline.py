import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from agent.core.audit import (  # noqa: E402
    audit_events_from_terminal_response_row,
    audit_timeline_enabled,
    build_audit_event,
    event_from_run_event,
    sanitize_audit_metadata,
    training_preflight_audit_events,
)
from agent.core.session_persistence import NoopSessionStore  # noqa: E402
from routes import agent  # noqa: E402


@pytest.mark.asyncio
async def test_audit_event_creation_sanitizes_secret_metadata(monkeypatch):
    monkeypatch.setenv("AUDIT_TIMELINE_ENABLED", "true")
    store = NoopSessionStore()

    event = await store.record_audit_event(
        build_audit_event(
            session_id="s1",
            run_id="r1",
            event_type="provider_job_started",
            category="provider_job",
            severity="info",
            status="running",
            actor="provider",
            title="AWS SageMaker job started",
            message="Job train-1 is running.",
            provider="aws-sagemaker",
            job_id="train-1",
            job_url="https://console.aws.amazon.com/sagemaker/train-1",
            safe_metadata={"HF_TOKEN": "hf_secret", "safe": "ok"},
        )
    )

    assert event["audit_id"]
    assert event["category"] == "provider_job"
    assert event["provider"] == "aws-sagemaker"
    assert event["safe_metadata"] == {"HF_TOKEN": "[REDACTED]", "safe": "ok"}
    assert "hf_secret" not in str(event)


@pytest.mark.asyncio
async def test_audit_events_are_idempotent_for_repeated_monitoring(monkeypatch):
    monkeypatch.setenv("AUDIT_TIMELINE_ENABLED", "true")
    store = NoopSessionStore()
    payload = build_audit_event(
        session_id="s1",
        run_id="r1",
        event_type="provider_job_running",
        category="provider_job",
        severity="info",
        status="running",
        actor="provider",
        title="AWS SageMaker job running",
        message="Job train-1 is still running.",
        provider="aws-sagemaker",
        job_id="train-1",
        entity_id="train-1",
    )

    await store.record_audit_event(payload)
    await store.record_audit_event(payload)

    events = await store.list_audit_events(session_id="s1")
    assert len(events) == 1
    assert events[0]["event_type"] == "provider_job_running"


@pytest.mark.asyncio
async def test_run_events_generate_approval_provider_usage_and_result_audit(
    monkeypatch,
):
    monkeypatch.setenv("AUDIT_TIMELINE_ENABLED", "true")
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
                    "approval_id": "approval-1",
                    "estimated_cost_usd": 1.75,
                    "arguments": {"operation": "run", "dataset_name": "safe-dataset"},
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
            "jobName": "train-1",
            "jobUrl": "https://console.aws.amazon.com/sagemaker/train-1",
        },
    )
    await store.append_run_event(
        run_id=run_id,
        session_id="s1",
        event_type="turn_complete",
        payload={"final_response": "Training plan ready."},
    )

    events = await store.list_audit_events(session_id="s1", run_id=run_id)
    assert [event["event_type"] for event in events] == [
        "run_created",
        "approval_required",
        "usage_estimated",
        "provider_job_started",
        "final_result_available",
    ]
    assert events[2]["estimated_cost_usd"] == 1.75
    assert events[3]["job_id"] == "train-1"


@pytest.mark.asyncio
async def test_audit_summary_counts_and_filters(monkeypatch):
    monkeypatch.setenv("AUDIT_TIMELINE_ENABLED", "true")
    store = NoopSessionStore()
    await store.record_audit_event(
        build_audit_event(
            session_id="s1",
            event_type="budget_warning",
            category="usage",
            severity="warning",
            status="pending",
            actor="system",
            title="Budget warning",
            message="Estimated cost exceeds budget.",
            provider="hf-jobs",
        )
    )
    await store.record_audit_event(
        build_audit_event(
            session_id="s1",
            event_type="provider_error",
            category="error",
            severity="error",
            status="failed",
            actor="provider",
            title="Provider error",
            message="Quota blocked.",
            provider="aws-sagemaker",
        )
    )

    summary = await store.audit_summary(session_id="s1")
    aws_events = await store.list_audit_events(provider="aws-sagemaker")

    assert summary["counts_by_category"] == {"error": 1, "usage": 1}
    assert summary["counts_by_severity"] == {"error": 1, "warning": 1}
    assert summary["counts_by_provider"]["aws-sagemaker"] == 1
    assert len(summary["latest_warnings_errors"]) == 2
    assert len(aws_events) == 1
    assert aws_events[0]["event_type"] == "provider_error"


@pytest.mark.asyncio
async def test_audit_apis_and_disabled_behavior(monkeypatch):
    monkeypatch.setenv("AUDIT_TIMELINE_ENABLED", "false")
    assert audit_timeline_enabled() is False

    async def _allow_access(*args, **kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(agent, "_check_session_access", _allow_access)
    response = await agent.list_audit(session_id="s1", user={"user_id": "dev"})
    summary = await agent.audit_summary(session_id="s1", user={"user_id": "dev"})
    response_payload = response.model_dump()
    summary_payload = summary.model_dump()

    assert response_payload["enabled"] is False
    assert response_payload["events"] == []
    assert summary_payload["enabled"] is False
    assert summary_payload["total_events"] == 0


@pytest.mark.asyncio
async def test_audit_api_response_omits_secrets(monkeypatch):
    monkeypatch.setenv("AUDIT_TIMELINE_ENABLED", "true")
    store = NoopSessionStore()
    await store.record_audit_event(
        build_audit_event(
            session_id="s1",
            event_type="dataset_upload_succeeded",
            category="dataset",
            severity="info",
            status="succeeded",
            actor="system",
            title="Dataset uploaded",
            message="Dataset data.csv uploaded.",
            dataset_name="data.csv",
            safe_metadata={"OPENAI_API_KEY": "sk-secret", "rows": 3},
        )
    )
    monkeypatch.setattr(
        agent.session_manager, "list_audit_events", store.list_audit_events
    )

    async def _allow_access(*args, **kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(agent, "_check_session_access", _allow_access)
    response = await agent.list_audit(session_id="s1", user={"user_id": "dev"})
    response_payload = response.model_dump()

    assert response_payload["events"][0]["safe_metadata"] == {
        "OPENAI_API_KEY": "[REDACTED]",
        "rows": 3,
    }
    assert "sk-secret" not in str(response_payload).lower()


def test_event_from_run_event_maps_failures_and_artifacts():
    failed = event_from_run_event(
        session_id="s1",
        run_id="r1",
        event_type="tool_state_change",
        payload={
            "tool": "hf_jobs",
            "state": "failed",
            "jobName": "job-1",
            "failureReason": "provider quota blocked",
        },
    )
    artifact = event_from_run_event(
        session_id="s1",
        run_id="r1",
        event_type="tool_state_change",
        payload={
            "tool": "hf_jobs",
            "state": "succeeded",
            "jobName": "job-1",
            "outputDir": "https://huggingface.co/models/demo/model",
        },
    )

    assert failed[0]["event_type"] == "provider_job_failed"
    assert failed[0]["severity"] == "error"
    assert failed[0]["error_summary"] == "provider quota blocked"
    assert artifact[0]["event_type"] == "provider_job_succeeded"
    assert artifact[1]["event_type"] == "artifact_available"


def test_event_from_run_event_maps_dataset_discovery_lifecycle():
    events = event_from_run_event(
        session_id="s1",
        run_id="r1",
        event_type="tool_output",
        payload={
            "tool": "dataset_discovery",
            "success": True,
            "structured": {
                "query": "Find hardware support data",
                "recommended_candidate": {
                    "dataset_id": "public/hardware-support",
                    "title": "Hardware Support QA",
                    "overall_score": 0.91,
                },
                "candidates": [
                    {
                        "dataset_id": "public/hardware-support",
                        "title": "Hardware Support QA",
                        "excluded": False,
                    },
                    {
                        "dataset_id": "kaggle/ipl",
                        "title": "IPL Kaggle",
                        "excluded": True,
                        "exclusion_reason": "Kaggle is future work only.",
                    },
                ],
                "warnings": ["User selection required before training."],
            },
        },
    )

    assert [event["event_type"] for event in events] == [
        "dataset_discovery_completed",
        "dataset_candidate_recommended",
        "dataset_candidate_excluded",
    ]
    assert events[0]["category"] == "dataset"
    assert events[0]["safe_metadata"]["candidate_count"] == 2
    assert events[1]["dataset_name"] == "Hardware Support QA"
    assert events[2]["severity"] == "warning"


def test_training_preflight_audit_events_record_unknown_and_blocked():
    events = training_preflight_audit_events(
        {
            "preflight_id": "pf1",
            "session_id": "s1",
            "run_id": "r1",
            "provider": "hf-jobs",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "hardware_id": "hf-jobs:t4-small",
            "output_policy": "cloud-and-hf-hub",
            "status": "unknown",
            "launch_ready": False,
            "safe_summary": "HF_TOKEN=hf_" + "A" * 35,
            "blocking_reasons": [],
            "unknown_reasons": ["Live checks are not implemented."],
            "warning_reasons": [],
            "metadata": {"provider_jobs_launched": False, "resources_created": False},
        },
        include_started=True,
    )

    assert [event["event_type"] for event in events] == [
        "training_preflight_started",
        "training_preflight_unknown",
        "training_preflight_launch_blocked",
    ]
    assert events[1]["severity"] == "warning"
    assert "hf_" not in str(events)


@pytest.mark.asyncio
async def test_run_event_persists_dataset_discovery_metadata():
    store = NoopSessionStore()
    run = await store.create_run(session_id="s1", provider="hf-jobs")
    run_id = run["run_id"]
    discovery = {
        "query": "Find hardware support data",
        "recommended_candidate": {"dataset_id": "public/hardware-support"},
        "candidates": [{"dataset_id": "public/hardware-support"}],
        "warnings": ["User selection required before training."],
    }

    await store.append_run_event(
        run_id=run_id,
        session_id="s1",
        event_type="tool_output",
        payload={
            "tool": "dataset_discovery",
            "success": True,
            "structured": discovery,
        },
    )

    updated = await store.get_run(run_id)

    assert updated["dataset_discovery"]["query"] == "Find hardware support data"
    assert updated["provider_metadata"]["dataset_discovery"]["query"] == (
        "Find hardware support data"
    )


def test_sanitize_audit_metadata_redacts_secret_like_values():
    sanitized = sanitize_audit_metadata(
        {
            "nested": {"token": "hf_secret"},
            "safe_url": "https://huggingface.co/jobs/1",
            "message": "AWS_SECRET_ACCESS_KEY=abc",
        }
    )

    assert sanitized == {
        "nested": {"token": "[REDACTED]"},
        "safe_url": "https://huggingface.co/jobs/1",
        "message": "AWS_SECRET_ACCESS_KEY=[REDACTED]",
    }


def test_terminal_response_row_emits_provider_job_completed_once():
    job_id = "projects/demo/locations/us-central1/customJobs/456"
    row = {
        "platform": "gcp-vertex",
        "job_id": job_id,
        "progress": "completed",
        "completed_at": "2026-06-17T15:34:34+00:00",
        "final_artifact_or_result": "gs://liga-ml/vertex-outputs/smoke",
        "provider_metadata": {
            "state": "JOB_STATE_SUCCEEDED",
            "jobUrl": "https://console.cloud.google.com/vertex-ai/jobs/456",
        },
    }
    events = audit_events_from_terminal_response_row(
        session_id="s1",
        run_id="r1",
        row=row,
    )
    assert len(events) == 1
    assert events[0]["event_type"] == "provider_job_completed"
    assert events[0]["job_id"] == job_id
    assert events[0]["artifact_url"] == "gs://liga-ml/vertex-outputs/smoke"
    assert "sk-" not in str(events[0]).lower()


def test_terminal_response_row_emits_provider_job_failed():
    events = audit_events_from_terminal_response_row(
        session_id="s1",
        run_id="r1",
        row={
            "platform": "gcp-vertex",
            "job_id": "projects/demo/locations/us-central1/customJobs/789",
            "progress": "failed",
            "error": "worker crashed",
            "provider_metadata": {"state": "JOB_STATE_FAILED"},
        },
    )
    assert events[0]["event_type"] == "provider_job_failed"
    assert events[0]["error_summary"] == "worker crashed"


@pytest.mark.asyncio
async def test_terminal_response_row_audit_sync_is_idempotent(monkeypatch):
    monkeypatch.setenv("AUDIT_TIMELINE_ENABLED", "true")
    store = NoopSessionStore()
    store.enabled = True
    run = await store.create_run(session_id="s-audit", provider="gcp-vertex")
    run_id = run["run_id"]
    job_id = "projects/demo/locations/us-central1/customJobs/456"
    await store.update_run(run_id, active_provider_job_id=job_id)
    monkeypatch.setattr(agent.session_manager, "persistence_store", store)
    row = {
        "session_id": "s-audit",
        "job_id": job_id,
        "platform": "gcp-vertex",
        "progress": "completed",
        "final_artifact_or_result": "gs://liga-ml/vertex-outputs/smoke",
        "provider_metadata": {"state": "JOB_STATE_SUCCEEDED"},
    }
    assert await agent._sync_usage_and_audit_from_terminal_response_rows([row]) is True
    assert await agent._sync_usage_and_audit_from_terminal_response_rows([row]) is False
    events = await store.list_audit_events(session_id="s-audit")
    completed = [
        event for event in events if event["event_type"] == "provider_job_completed"
    ]
    assert len(completed) == 1
