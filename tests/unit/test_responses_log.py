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
    filter_response_rows,
    paginate_response_rows,
    redact_response_value,
)
from agent.core.session_persistence import MongoSessionStore  # noqa: E402
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


def _tool_output(tool: str, output: str, **data):
    return {
        "event_type": "tool_output",
        "created_at": data.pop("created_at", datetime(2026, 1, 1, tzinfo=UTC)),
        "data": {
            "tool_call_id": data.pop("tool_call_id", f"{tool}-output"),
            "tool": tool,
            "output": output,
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
        "durable": False,
        "store_type": "memory",
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
    assert row["progress"] == "completed"
    assert row["job_id"] == "https://huggingface.co/jobs/acme/123"
    assert row["final_artifact_or_result"] == "https://huggingface.co/acme/final-model"


@pytest.mark.asyncio
async def test_hf_running_row_updates_from_completed_tool_output():
    events = [
        _event(
            "hf_jobs",
            "running",
            jobUrl="https://huggingface.co/jobs/acme/job-123",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        _tool_output(
            "hf_jobs",
            """
**Job Details** (1 job):

```json
[
  {
    "id": "job-123",
    "status": {"stage": "COMPLETED", "message": null}
  }
]
```
LIGA_FINAL_MODEL_URL=https://huggingface.co/acme/final-model
""",
            tool_call_id="functions.hf_jobs:99",
            created_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        ),
    ]

    result = await build_responses_log(
        [_session("s1", events=events)], load_events=lambda _sid: events
    )

    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["progress"] == "completed"
    assert row["job_id"] == "https://huggingface.co/jobs/acme/job-123"
    assert row["completed_at"] == "2026-01-01T00:01:00+00:00"
    assert row["final_artifact_or_result"] == "https://huggingface.co/acme/final-model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    ["COMPLETED", "completed", "succeeded", "success", "SUCCEEDED", "done", "finished"],
)
async def test_hf_success_variants_normalize_to_completed(state):
    events = [
        _event("hf_jobs", state, jobUrl="https://huggingface.co/jobs/acme/job-123"),
    ]

    result = await build_responses_log(
        [_session("s1", events=events)], load_events=lambda _sid: events
    )

    assert result["rows"][0]["progress"] == "completed"


@pytest.mark.asyncio
async def test_hf_tool_output_success_true_normalizes_to_completed():
    events = [
        _tool_output(
            "hf_jobs",
            """
Python job completed!

**Job ID:** job-123
**View at:** https://huggingface.co/jobs/acme/job-123
""",
            success=True,
        )
    ]

    result = await build_responses_log(
        [_session("s1", events=events)], load_events=lambda _sid: events
    )

    assert result["rows"][0]["progress"] == "completed"
    row = result["rows"][0]
    assert row["job_id"] == "https://huggingface.co/jobs/acme/job-123"


@pytest.mark.asyncio
async def test_hf_tool_output_only_terminal_update_works():
    events = [
        _tool_output(
            "hf_jobs",
            """
**Job Details** (1 job):

```json
[{"id": "job-123", "status": {"stage": "COMPLETED"}}]
```
""",
        )
    ]

    result = await build_responses_log(
        [_session("s1", events=events)], load_events=lambda _sid: events
    )

    assert result["rows"][0]["progress"] == "completed"
    assert result["rows"][0]["job_id"] == "job-123"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fake_id", ["functions.hf_jobs:10", "tool_call_abc", "call_abc"]
)
async def test_hf_fake_internal_ids_do_not_become_job_ids(fake_id):
    events = [
        _tool_output(
            "hf_jobs",
            f'**Job Details**: {{"id": "{fake_id}", "status": {{"stage": "COMPLETED"}}}}',
            success=True,
        )
    ]

    result = await build_responses_log(
        [_session("s1", events=events)], load_events=lambda _sid: events
    )

    assert result == {"rows": []}


@pytest.mark.asyncio
async def test_hf_failed_tool_output_updates_running_row_with_reason():
    events = [
        _event("hf_jobs", "running", jobUrl="https://huggingface.co/jobs/acme/job-123"),
        _tool_output(
            "hf_jobs",
            """
**Job Details** (1 job):

```json
[{"id": "job-123", "status": {"stage": "FAILED", "message": "trainer crashed"}}]
```
""",
            created_at=datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
        ),
    ]

    result = await build_responses_log(
        [_session("s1", events=events)], load_events=lambda _sid: events
    )

    row = result["rows"][0]
    assert row["progress"] == "failed"
    assert row["job_id"] == "https://huggingface.co/jobs/acme/job-123"
    assert row["completed_at"] == "2026-01-01T00:02:00+00:00"
    assert row["error"] == "trainer crashed"


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

    assert len(result["rows"]) == 16
    row = result["rows"][-1]
    assert row["actual_sequence_number"] == 16
    assert row["display_session_number"] == 1
    assert row["batch_number"] == 2
    assert (
        build_responses_summary(result["rows"], total_responses=16)["visible_count"]
        == 16
    )


@pytest.mark.asyncio
async def test_response_pagination_returns_latest_rows_newest_first():
    sessions = []
    event_map = {}
    base = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(55):
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
    page_1 = paginate_response_rows(result["rows"], page=1, page_size=50)
    page_2 = paginate_response_rows(result["rows"], page=2, page_size=50)

    assert page_1["page"] == 1
    assert page_1["page_size"] == 50
    assert page_1["total_rows"] == 55
    assert page_1["total_pages"] == 2
    assert page_1["has_next"] is True
    assert page_1["has_previous"] is False
    assert page_1["rows"][0]["actual_sequence_number"] == 55
    assert page_1["rows"][-1]["actual_sequence_number"] == 6
    assert page_2["has_next"] is False
    assert page_2["has_previous"] is True
    assert [row["actual_sequence_number"] for row in page_2["rows"]] == [5, 4, 3, 2, 1]


@pytest.mark.asyncio
async def test_response_search_and_filters_match_backend_fields():
    hf_events = [
        _event(
            "hf_jobs",
            "succeeded",
            jobUrl="https://huggingface.co/jobs/acme/housing",
            logs="LIGA_FINAL_MODEL_URL=https://huggingface.co/acme/housing-model\n",
        )
    ]
    aws_events = [
        _event(
            "aws_sagemaker_jobs",
            "failed",
            jobName="liga-aws-123",
            failureReason="training image failed",
        )
    ]
    result = await build_responses_log(
        [
            _session("hf-session", events=hf_events, model="moonshotai/Kimi-K2.6"),
            _session(
                "aws-session",
                provider="aws-sagemaker",
                events=aws_events,
                model="Claude",
            ),
        ],
        load_events=lambda sid: hf_events if sid == "hf-session" else aws_events,
    )

    assert [
        row["session_id"]
        for row in filter_response_rows(result["rows"], platform="hf-jobs")
    ] == ["hf-session"]
    assert [
        row["session_id"]
        for row in filter_response_rows(result["rows"], progress="failed")
    ] == ["aws-session"]
    assert [
        row["session_id"] for row in filter_response_rows(result["rows"], model="kimi")
    ] == ["hf-session"]
    assert [
        row["session_id"]
        for row in filter_response_rows(result["rows"], job_id="aws-123")
    ] == ["aws-session"]
    assert [
        row["session_id"] for row in filter_response_rows(result["rows"], q="housing")
    ] == ["hf-session"]


def test_mongo_response_rows_normalize_persisted_provider_states():
    store = MongoSessionStore("mongodb://example.invalid", "liga_ml")

    assert (
        store._normalize_response_row({"progress": "scheduling"})["progress"]
        == "queued"
    )
    assert (
        store._normalize_response_row(
            {"progress": "unknown", "provider_metadata": {"state": "succeeded"}}
        )["progress"]
        == "completed"
    )


class _DurableResponseStore:
    enabled = True
    response_rows = []

    async def upsert_response_rows(self, rows, **_kwargs):
        if not rows:
            return
        self.response_rows = [dict(row) for row in rows]

    async def list_response_rows(self, **kwargs):
        filter_kwargs = {
            key: value for key, value in kwargs.items() if key != "user_id"
        }
        rows = filter_response_rows(self.response_rows, **filter_kwargs)
        return paginate_response_rows(
            rows,
            page=kwargs.get("page", 1),
            page_size=kwargs.get("page_size", 50),
        )

    async def get_response_summary(self, **_kwargs):
        return build_responses_summary(
            self.response_rows,
            total_responses=len(self.response_rows),
            durable=True,
            store_type="mongodb",
        )


class _StoreBackedManager:
    def __init__(self):
        self.persistence_store = _DurableResponseStore()

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

    assert rows["page"] == 1
    assert rows["page_size"] == 50
    assert rows["total_rows"] == 1
    assert rows["total_pages"] == 1
    assert rows["has_next"] is False
    assert rows["has_previous"] is False
    assert rows["rows"][0]["session_id"] == "persisted"
    assert summary["button_enabled"] is True
    assert summary["total_responses"] == 1
    assert summary["durable"] is True
    assert summary["store_type"] == "mongodb"


@pytest.mark.asyncio
async def test_responses_routes_can_read_rows_after_manager_restart(monkeypatch):
    manager = _StoreBackedManager()
    await manager.persistence_store.upsert_response_rows(
        [
            {
                "id": "persisted:hf-jobs:https://huggingface.co/jobs/acme/persisted",
                "display_session_number": 1,
                "actual_sequence_number": 1,
                "batch_number": 1,
                "session_id": "persisted",
                "short_session_id": "persiste",
                "session_title": "Persisted run",
                "model_name": "moonshotai/Kimi-K2.6",
                "platform": "hf-jobs",
                "run_type": "smoke-test",
                "result_storage": "cloud-and-hf-hub",
                "progress": "completed",
                "job_id": "https://huggingface.co/jobs/acme/persisted",
                "final_artifact_or_result": "https://huggingface.co/acme/model",
                "created_at": "2026-01-01T00:00:00+00:00",
                "completed_at": "2026-01-01T00:01:00+00:00",
                "provider_metadata": {},
            }
        ]
    )

    async def no_sessions(user_id="dev"):
        return []

    manager.list_sessions = no_sessions
    monkeypatch.setattr(agent, "session_manager", manager)

    rows = await agent.get_responses(user={"user_id": "dev"})

    assert rows["rows"][0]["session_id"] == "persisted"
    assert rows["total_rows"] == 1


@pytest.mark.asyncio
async def test_responses_routes_refresh_stale_hf_rows_from_persisted_events(
    monkeypatch,
):
    manager = _StoreBackedManager()
    await manager.persistence_store.upsert_response_rows(
        [
            {
                "id": "persisted:hf-jobs:https://huggingface.co/jobs/acme/persisted",
                "display_session_number": 1,
                "actual_sequence_number": 1,
                "batch_number": 1,
                "session_id": "persisted",
                "short_session_id": "persiste",
                "session_title": "Persisted run",
                "model_name": "moonshotai/Kimi-K2.6",
                "platform": "hf-jobs",
                "run_type": "smoke-test",
                "result_storage": "cloud-and-hf-hub",
                "progress": "running",
                "job_id": "https://huggingface.co/jobs/acme/persisted",
                "final_artifact_or_result": "https://huggingface.co/jobs/acme/persisted",
                "created_at": "2026-01-01T00:00:00+00:00",
                "completed_at": None,
                "provider_metadata": {},
            }
        ]
    )
    manager.sessions = {}

    async def completed_events(session_id):
        assert session_id == "persisted"
        return [
            _event(
                "hf_jobs",
                "running",
                jobUrl="https://huggingface.co/jobs/acme/persisted",
            ),
            _tool_output(
                "hf_jobs",
                """
**Job Details** (1 job):

```json
[{"id": "persisted", "status": {"stage": "COMPLETED"}}]
```
""",
            ),
        ]

    manager.load_response_events = completed_events
    monkeypatch.setattr(agent, "session_manager", manager)

    rows = await agent.get_responses(
        job_id="persisted",
        user={"user_id": "dev"},
    )

    assert rows["rows"][0]["progress"] == "completed"
    assert rows["rows"][0]["completed_at"] is not None
