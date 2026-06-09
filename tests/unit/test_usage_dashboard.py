import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from agent.core.session_persistence import NoopSessionStore  # noqa: E402
from agent.core.usage import sanitize_metadata  # noqa: E402
from routes import agent  # noqa: E402


async def _store_with_usage(monkeypatch, estimate=1.25, budget="1"):
    if budget is None:
        monkeypatch.delenv("DEFAULT_DAILY_BUDGET_USD", raising=False)
    else:
        monkeypatch.setenv("DEFAULT_DAILY_BUDGET_USD", budget)
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
                    "approval_id": "tc1",
                    "estimated_cost_usd": estimate,
                    "remaining_cap_usd": 1,
                    "arguments": {
                        "operation": "run",
                        "instance_type": "ml.g5.xlarge",
                        "instance_count": 1,
                        "max_run_seconds": 3600,
                        "dataset_name": "safe-dataset",
                    },
                }
            ]
        },
    )
    return store, run_id


async def _store_with_vertex_smoke_usage(monkeypatch):
    monkeypatch.delenv("DEFAULT_DAILY_BUDGET_USD", raising=False)
    monkeypatch.delenv("GCLOUD_DAILY_BUDGET_USD", raising=False)
    store = NoopSessionStore()
    run = await store.create_run(session_id="s-vertex", provider="gcp-vertex")
    run_id = run["run_id"]
    await store.append_run_event(
        run_id=run_id,
        session_id="s-vertex",
        event_type="approval_required",
        payload={
            "tools": [
                {
                    "tool": "gcp_vertex_jobs",
                    "tool_call_id": "vertex-call-1",
                    "approval_id": "vertex-call-1",
                    "provider": "gcp-vertex",
                    "arguments": {
                        "operation": "run",
                        "training_goal": "smoke-test",
                        "machine_type": "n1-standard-8",
                        "accelerator_type": "NVIDIA_TESLA_T4",
                        "accelerator_count": 1,
                        "replica_count": 1,
                        "max_run_hours": 1,
                        "display_name": "liga-vertex-smoke",
                        "output_dir": "gs://liga-training/vertex-outputs/smoke",
                        "output_policy": "cloud-private",
                    },
                    "metadata": {"HF_TOKEN": "hf_secret", "run_type": "smoke-test"},
                }
            ]
        },
    )
    return store, run_id


@pytest.mark.asyncio
async def test_usage_entry_from_approval_pending(monkeypatch):
    store, run_id = await _store_with_usage(monkeypatch)

    entries = await store.list_usage_entries(run_id=run_id)

    assert len(entries) == 1
    assert entries[0]["provider"] == "aws-sagemaker"
    assert entries[0]["status"] == "approval_required"
    assert entries[0]["estimated_cost_usd"] == 1.25
    assert entries[0]["cost_source"] == "approval_estimate"
    assert entries[0]["cost_confidence"] == "estimated"
    assert entries[0]["budget_cap_usd"] == 1
    assert "exceeds configured daily budget" in entries[0]["warning"]


@pytest.mark.asyncio
async def test_vertex_smoke_approval_creates_non_zero_estimate(monkeypatch):
    store, run_id = await _store_with_vertex_smoke_usage(monkeypatch)

    entries = await store.list_usage_entries(run_id=run_id)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["provider"] == "gcp-vertex"
    assert entry["tool_name"] == "gcp_vertex_jobs"
    assert entry["status"] == "approval_required"
    assert entry["estimated_cost_usd"] is not None
    assert entry["estimated_cost_usd"] > 0
    assert entry.get("known_cost_usd") in (None, 0)
    assert entry["cost_source"] == "static_estimate"
    assert entry["cost_confidence"] == "estimated"
    assert entry["instance_type"] == "n1-standard-8 + NVIDIA_TESLA_T4"
    assert entry["instance_count"] == 1
    assert entry["max_runtime_seconds"] == 3600
    assert entry["output_policy"] == "cloud-private"
    assert entry["metadata"]["arguments"]["training_goal"] == "smoke-test"
    assert entry["metadata"]["arguments"]["accelerator_count"] == 1
    assert "hf_secret" not in str(entry).lower()


@pytest.mark.asyncio
async def test_vertex_job_launch_confirms_usage_without_duplicate(monkeypatch):
    store, run_id = await _store_with_vertex_smoke_usage(monkeypatch)

    launch_payload = {
        "tool": "gcp_vertex_jobs",
        "tool_call_id": "vertex-call-1",
        "state": "running",
        "jobName": "projects/489651394276/locations/us-central1/customJobs/1263998000355606528",
        "jobUrl": "https://console.cloud.google.com/vertex-ai/locations/us-central1/training/1263998000355606528",
        "outputDir": "gs://liga-training/vertex-outputs/smoke",
        "machine_type": "n1-standard-8",
        "accelerator_type": "NVIDIA_TESLA_T4",
        "accelerator_count": 1,
        "replica_count": 1,
        "max_run_hours": 1,
        "training_goal": "smoke-test",
        "outputPolicy": "cloud-private",
    }
    await store.append_run_event(
        run_id=run_id,
        session_id="s-vertex",
        event_type="tool_state_change",
        payload=launch_payload,
    )
    await store.append_run_event(
        run_id=run_id,
        session_id="s-vertex",
        event_type="tool_state_change",
        payload=launch_payload,
    )

    entries = await store.list_usage_entries(run_id=run_id)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["status"] == "running"
    assert entry["approved"] is True
    assert entry["job_id"].endswith("/customJobs/1263998000355606528")
    assert entry["artifact_url"] == "gs://liga-training/vertex-outputs/smoke"
    assert entry["estimated_cost_usd"] is not None
    assert entry["estimated_cost_usd"] > 0
    assert entry.get("known_cost_usd") in (None, 0)


@pytest.mark.asyncio
async def test_vertex_usage_summary_and_api_include_estimate(monkeypatch):
    store, run_id = await _store_with_vertex_smoke_usage(monkeypatch)
    monkeypatch.setattr(agent.session_manager, "usage_summary", store.usage_summary)
    monkeypatch.setattr(
        agent.session_manager, "list_usage_entries", store.list_usage_entries
    )

    async def _allow_access(*args, **kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(agent, "_check_session_access", _allow_access)

    summary = await store.usage_summary(session_id="s-vertex")
    api_summary = await agent.usage_summary(
        session_id="s-vertex", user={"user_id": "dev"}
    )
    api_entries = await agent.list_usage(session_id="s-vertex", user={"user_id": "dev"})

    assert summary["total_estimated_cost_usd"] > 0
    assert summary["total_known_cost_usd"] == 0
    assert summary["cost_by_provider"]["gcp-vertex"]["estimated_cost_usd"] > 0
    assert summary["cost_by_run"][run_id]["estimated_cost_usd"] > 0
    assert api_summary.total_estimated_cost_usd > 0
    assert api_summary.recent_usage_entries[0].provider == "gcp-vertex"
    assert api_entries[0].provider == "gcp-vertex"
    assert api_entries[0].estimated_cost_usd is not None
    assert api_entries[0].estimated_cost_usd > 0


@pytest.mark.asyncio
async def test_usage_updates_after_approval_and_provider_start(monkeypatch):
    store, run_id = await _store_with_usage(monkeypatch, estimate=0.5, budget=None)

    await store.append_run_event(
        run_id=run_id,
        session_id="s1",
        event_type="tool_state_change",
        payload={
            "tool": "aws_sagemaker_jobs",
            "tool_call_id": "tc1",
            "state": "approved",
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

    entries = await store.list_usage_entries(run_id=run_id)
    assert len(entries) == 1
    assert entries[0]["approved"] is True
    assert entries[0]["job_id"] == "train-1"
    assert entries[0]["job_url"].endswith("train-1")
    assert entries[0]["status"] == "running"


@pytest.mark.asyncio
async def test_usage_updates_after_provider_success_and_failure(monkeypatch):
    store, run_id = await _store_with_usage(monkeypatch, estimate=0.5, budget=None)

    await store.append_run_event(
        run_id=run_id,
        session_id="s1",
        event_type="tool_state_change",
        payload={
            "tool": "aws_sagemaker_jobs",
            "tool_call_id": "tc1",
            "state": "succeeded",
            "jobName": "train-1",
            "s3ModelArtifact": "s3://bucket/model.tar.gz",
        },
    )
    success = (await store.list_usage_entries(run_id=run_id))[0]
    assert success["status"] == "succeeded"
    assert success["artifact_url"] == "s3://bucket/model.tar.gz"

    await store.append_run_event(
        run_id=run_id,
        session_id="s1",
        event_type="tool_state_change",
        payload={
            "tool": "aws_sagemaker_jobs",
            "tool_call_id": "tc1",
            "state": "failed",
            "failureReason": "ml.g5.xlarge training quota is 0",
        },
    )
    failed = (await store.list_usage_entries(run_id=run_id))[0]
    assert failed["status"] == "failed"
    assert failed["error_summary"] == "ml.g5.xlarge training quota is 0"


@pytest.mark.asyncio
async def test_repeated_provider_updates_are_idempotent(monkeypatch):
    store, run_id = await _store_with_usage(monkeypatch, estimate=0.5, budget=None)
    payload = {
        "tool": "aws_sagemaker_jobs",
        "tool_call_id": "tc1",
        "state": "running",
        "jobName": "train-1",
    }

    await store.append_run_event(
        run_id=run_id, session_id="s1", event_type="tool_state_change", payload=payload
    )
    await store.append_run_event(
        run_id=run_id, session_id="s1", event_type="tool_state_change", payload=payload
    )

    assert len(await store.list_usage_entries(run_id=run_id)) == 1


@pytest.mark.asyncio
async def test_usage_summary_and_session_run_filters(monkeypatch):
    store, run_id = await _store_with_usage(monkeypatch, estimate=0.5)

    summary = await store.usage_summary(session_id="s1")
    session_entries = await store.list_usage_entries(session_id="s1")
    run_entries = await store.list_usage_entries(run_id=run_id)

    assert summary["total_estimated_cost_usd"] == 0.5
    assert summary["cost_by_provider"]["aws-sagemaker"]["count"] == 1
    assert len(session_entries) == 1
    assert len(run_entries) == 1


@pytest.mark.asyncio
async def test_no_budget_warning_when_no_budget_configured(monkeypatch):
    monkeypatch.delenv("DEFAULT_DAILY_BUDGET_USD", raising=False)
    monkeypatch.delenv("AWS_DAILY_BUDGET_USD", raising=False)
    store, run_id = await _store_with_usage(monkeypatch, estimate=0.5, budget=None)

    entry = (await store.list_usage_entries(run_id=run_id))[0]

    assert entry["warning"] == "No budget configured"


def test_usage_metadata_redacts_secrets():
    sanitized = sanitize_metadata(
        {
            "HF_TOKEN": "hf_secret",
            "nested": {"aws_secret_access_key": "secret"},
            "safe": "ok",
        }
    )

    assert sanitized == {
        "HF_TOKEN": "[REDACTED]",
        "nested": {"aws_secret_access_key": "[REDACTED]"},
        "safe": "ok",
    }
    assert "hf_secret" not in str(sanitized)
    assert "'secret'" not in str(sanitized)


@pytest.mark.asyncio
async def test_usage_api_response_omits_secrets(monkeypatch):
    store, _run_id = await _store_with_usage(monkeypatch, estimate=0.5)
    monkeypatch.setattr(
        agent.session_manager, "list_usage_entries", store.list_usage_entries
    )

    async def _allow_access(*args, **kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(agent, "_check_session_access", _allow_access)

    response = await agent.list_usage(session_id="s1", user={"user_id": "dev"})

    assert response
    assert "secret" not in str([item.model_dump() for item in response]).lower()
