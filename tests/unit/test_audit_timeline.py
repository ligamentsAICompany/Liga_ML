import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from agent.core.audit import (  # noqa: E402
    audit_timeline_enabled,
    build_audit_event,
    event_from_run_event,
    sanitize_audit_metadata,
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
    assert event["safe_metadata"] == {"safe": "ok"}
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

    assert response_payload["events"][0]["safe_metadata"] == {"rows": 3}
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


def test_sanitize_audit_metadata_redacts_secret_like_values():
    sanitized = sanitize_audit_metadata(
        {
            "nested": {"token": "hf_secret"},
            "safe_url": "https://huggingface.co/jobs/1",
            "message": "AWS_SECRET_ACCESS_KEY=abc",
        }
    )

    assert sanitized == {
        "nested": {},
        "safe_url": "https://huggingface.co/jobs/1",
        "message": "[redacted]",
    }
