import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from responses_log import (  # noqa: E402
    build_responses_log,
    build_responses_summary,
    redact_response_value,
)
from routes import agent  # noqa: E402


def _event(tool: str, state: str, **data):
    return {
        "event_type": "tool_state_change",
        "created_at": data.pop("created_at", datetime(2026, 1, 1, tzinfo=UTC)),
        "data": {
            "tool_call_id": data.pop("tool_call_id", f"{tool}-{state}"),
            "tool": tool,
            "state": state,
            **data,
        },
    }


def _session(
    session_id: str,
    *,
    provider: str = "hf-jobs",
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    goal: str = "smoke-test",
    output_policy: str = "cloud-and-hf-hub",
    events=None,
):
    return {
        "session_id": session_id,
        "title": f"Session {session_id}",
        "model": model,
        "cloud_provider": provider,
        "training_goal": goal,
        "output_policy": output_policy,
        "created_at": "2026-01-01T00:00:00+00:00",
        "is_processing": False,
        "pending_approval": None,
        "_events": list(events or []),
    }


@pytest.mark.asyncio
async def test_responses_log_empty_and_summary_enabled():
    result = await build_responses_log([], load_events=lambda _sid: [])

    assert result == {"rows": []}
    assert build_responses_summary(result["rows"]) == {
        "total_responses": 0,
        "visible_count": 0,
        "batch_number": 1,
        "has_rows": False,
        "button_enabled": True,
    }


@pytest.mark.asyncio
async def test_hf_terminal_row_extracts_job_and_artifact():
    events = [
        _event("hf_jobs", "running", jobUrl="https://huggingface.co/jobs/acme/123"),
        _event(
            "hf_jobs",
            "succeeded",
            jobUrl="https://huggingface.co/jobs/acme/123",
            logs="LIGA_FINAL_MODEL_URL=https://huggingface.co/acme/final-model\n",
        ),
    ]

    result = await build_responses_log(
        [_session("s1", events=events)], load_events=lambda _sid: events
    )

    row = result["rows"][0]
    assert row["display_session_number"] == 1
    assert row["actual_sequence_number"] == 1
    assert row["batch_number"] == 1
    assert row["session_id"] == "s1"
    assert row["model_name"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert row["platform"] == "hf-jobs"
    assert row["run_type"] == "smoke-test"
    assert row["result_storage"] == "cloud-and-hf-hub"
    assert row["progress"] == "succeeded"
    assert row["job_id"] == "https://huggingface.co/jobs/acme/123"
    assert row["final_artifact_or_result"] == "https://huggingface.co/acme/final-model"


@pytest.mark.asyncio
async def test_gcp_and_aws_terminal_rows_extract_provider_artifacts():
    gcp_events = [
        _event(
            "gcp_vertex_jobs",
            "succeeded",
            jobName="projects/p/locations/us/customJobs/456",
            outputDir="gs://liga-output/job-456",
            jobUrl="https://console.cloud.google.com/vertex-ai/jobs/456",
        )
    ]
    aws_events = [
        _event(
            "aws_sagemaker_jobs",
            "completed",
            jobName="liga-train-789",
            s3OutputUri="s3://liga-output/job-789",
            s3ModelArtifact="s3://liga-output/job-789/output/model.tar.gz",
            cloudWatchLogsUrl="https://console.aws.amazon.com/cloudwatch/logs",
        )
    ]

    result = await build_responses_log(
        [
            _session(
                "gcp",
                provider="gcp-vertex",
                output_policy="cloud-private",
                events=gcp_events,
            ),
            _session(
                "aws",
                provider="aws-sagemaker",
                output_policy="cloud-private",
                events=aws_events,
            ),
        ],
        load_events=lambda sid: gcp_events if sid == "gcp" else aws_events,
    )

    gcp, aws = result["rows"]
    assert gcp["platform"] == "gcp-vertex"
    assert gcp["job_id"] == "projects/p/locations/us/customJobs/456"
    assert gcp["final_artifact_or_result"] == "gs://liga-output/job-456"
    assert aws["platform"] == "aws-sagemaker"
    assert aws["job_id"] == "liga-train-789"
    assert (
        aws["final_artifact_or_result"]
        == "s3://liga-output/job-789/output/model.tar.gz"
    )


@pytest.mark.asyncio
async def test_failed_row_extracts_failure_reason_and_redacts_secrets():
    sample_token_value = "hf_fake_token_123456789"
    events = [
        _event(
            "gcp_vertex_jobs",
            "failed",
            jobName="projects/p/locations/us/customJobs/failed",
            failureReason=f"bad token {sample_token_value}",
        )
    ]

    result = await build_responses_log(
        [_session("s1", provider="gcp-vertex", events=events)],
        load_events=lambda _sid: events,
    )

    row = result["rows"][0]
    assert row["progress"] == "failed"
    assert row["final_artifact_or_result"] == "bad token [REDACTED]"
    assert sample_token_value not in str(row)
    assert (
        redact_response_value(f"Authorization: Bearer {sample_token_value}")
        == "Authorization: Bearer [REDACTED]"
    )


@pytest.mark.asyncio
async def test_hf_running_without_provider_job_id_is_not_fake_job_row():
    events = [
        _event("hf_jobs", "running", tool_call_id="functions.hf_jobs:10"),
    ]

    result = await build_responses_log(
        [_session("s1", events=events)], load_events=lambda _sid: events
    )

    assert result == {"rows": []}


@pytest.mark.asyncio
async def test_hf_error_without_provider_job_id_records_failure_not_fake_job_id():
    events = [
        _event(
            "hf_jobs",
            "error",
            tool_call_id="functions.hf_jobs:10",
            error="No HF token available to resolve a jobs namespace.",
        ),
    ]

    result = await build_responses_log(
        [_session("s1", events=events)], load_events=lambda _sid: events
    )

    row = result["rows"][0]
    assert row["progress"] == "error"
    assert row["job_id"] == ""
    assert row["final_artifact_or_result"] == (
        "No HF token available to resolve a jobs namespace."
    )
    assert row["completed_at"] is not None


@pytest.mark.asyncio
async def test_rolling_visible_batch_keeps_actual_sequence_continuing():
    sessions = []
    event_map = {}
    base = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(16):
        sid = f"s{index + 1}"
        event_map[sid] = [
            _event(
                "hf_jobs",
                "succeeded",
                created_at=base + timedelta(minutes=index),
                jobUrl=f"https://huggingface.co/jobs/acme/{index + 1}",
            )
        ]
        sessions.append(_session(sid, events=event_map[sid]))

    result = await build_responses_log(sessions, load_events=lambda sid: event_map[sid])

    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["actual_sequence_number"] == 16
    assert row["display_session_number"] == 1
    assert row["batch_number"] == 2
    assert (
        build_responses_summary(result["rows"], total_responses=16)["visible_count"]
        == 1
    )


class _StoreBackedManager:
    async def list_sessions(self, user_id="dev"):
        return [_session("persisted", events=[])]

    async def load_response_events(self, session_id):
        assert session_id == "persisted"
        return [
            _event(
                "hf_jobs",
                "succeeded",
                jobUrl="https://huggingface.co/jobs/acme/persisted",
            )
        ]


@pytest.mark.asyncio
async def test_responses_routes_use_session_manager_source_of_truth(monkeypatch):
    manager = _StoreBackedManager()
    monkeypatch.setattr(agent, "session_manager", manager)

    rows = await agent.get_responses(user={"user_id": "dev"})
    summary = await agent.get_responses_summary(user={"user_id": "dev"})

    assert rows["rows"][0]["session_id"] == "persisted"
    assert summary["button_enabled"] is True
    assert summary["total_responses"] == 1
