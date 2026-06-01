from types import SimpleNamespace

import pytest
from litellm import ChatCompletionMessageToolCall as ToolCall

from agent.config import Config
from agent.core import agent_loop
from agent.core.cost_estimation import CostEstimate


def _config(**overrides):
    data = {
        "model_name": "moonshotai/Kimi-K2.6",
        "confirm_cpu_jobs": True,
        "auto_file_upload": False,
        "yolo_mode": False,
        **overrides,
    }
    return Config.model_validate(data)


def _session(*, cap=5.0, spent=0.0, enabled=True):
    return SimpleNamespace(
        config=_config(),
        auto_approval_enabled=enabled,
        auto_approval_cost_cap_usd=cap,
        auto_approval_estimated_spend_usd=spent,
        sandbox=None,
    )


@pytest.mark.asyncio
async def test_session_yolo_auto_approves_non_costed_approval_tool():
    decision = await agent_loop._approval_decision(
        "hf_repo_files",
        {"operation": "upload", "path": "README.md"},
        _session(),
    )

    assert decision.requires_approval is False
    assert decision.auto_approved is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    ["scheduled run", "scheduled uv", "scheduled  run"],
)
async def test_scheduled_hf_jobs_always_require_manual_approval(operation):
    session = _session()
    session.config.yolo_mode = True

    decision = await agent_loop._approval_decision(
        "hf_jobs",
        {"operation": operation, "script": "print(1)"},
        session,
    )

    assert decision.requires_approval is True
    assert decision.auto_approval_blocked is True
    assert "Scheduled HF jobs" in decision.block_reason
    assert agent_loop._needs_approval(
        "hf_jobs", {"operation": operation}, session.config
    )


def test_gcp_vertex_run_and_cancel_require_approval():
    config = _config(confirm_cpu_jobs=False)

    assert agent_loop._needs_approval("gcp_vertex_jobs", {"operation": "run"}, config)
    assert agent_loop._needs_approval(
        "gcp_vertex_jobs",
        {"operation": "cancel", "job_name": "projects/p/locations/r/customJobs/1"},
        config,
    )


def test_gcp_vertex_read_only_operations_do_not_require_approval():
    config = _config(confirm_cpu_jobs=True)

    for operation in ["ps", "logs", "inspect"]:
        assert not agent_loop._needs_approval(
            "gcp_vertex_jobs",
            {"operation": operation, "job_name": "projects/p/locations/r/customJobs/1"},
            config,
        )


def test_existing_sandbox_approval_behavior_is_unchanged():
    config = _config()

    assert not agent_loop._needs_approval(
        "sandbox_create", {"hardware": "cpu-basic"}, config
    )
    assert agent_loop._needs_approval(
        "sandbox_create", {"hardware": "t4-small"}, config
    )


@pytest.mark.asyncio
async def test_immediate_hf_job_under_cap_auto_runs(monkeypatch):
    async def fake_estimate(*args, **kwargs):
        return CostEstimate(estimated_cost_usd=2.0, billable=True)

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)

    decision = await agent_loop._approval_decision(
        "hf_jobs",
        {"operation": "run", "hardware_flavor": "a10g-large", "timeout": "1h"},
        _session(cap=5.0, spent=1.0),
    )

    assert decision.requires_approval is False
    assert decision.auto_approved is True
    assert decision.estimated_cost_usd == 2.0


@pytest.mark.asyncio
async def test_immediate_hf_job_global_yolo_still_requires_manual_approval(monkeypatch):
    async def fake_estimate(*args, **kwargs):
        return CostEstimate(estimated_cost_usd=2.0, billable=True)

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)
    session = _session(enabled=False, cap=None, spent=0.0)
    session.config.yolo_mode = True

    decision = await agent_loop._approval_decision(
        "hf_jobs",
        {"operation": "run", "hardware_flavor": "a10g-large", "timeout": "1h"},
        session,
    )

    assert decision.requires_approval is True
    assert decision.auto_approval_blocked is True
    assert decision.auto_approved is False
    assert decision.estimated_cost_usd == 2.0
    assert "manual approval" in decision.block_reason


def test_hf_jobs_approval_metadata_includes_provider_model_and_dataset():
    session = SimpleNamespace(
        training_goal="production",
        output_policy="hf-hub",
        uploaded_datasets=[
            {
                "repo_id": "owner/uploaded-dataset",
                "config_name": "normalized",
                "normalized_row_count": 42,
            }
        ],
    )

    metadata = agent_loop._approval_metadata(
        session,
        "hf_jobs",
        {
            "operation": "run",
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "hardware_flavor": "t4-small",
        },
    )

    assert metadata == {
        "provider": "hf-jobs",
        "training_goal": "production",
        "output_policy": "hf-hub",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "hardware": "t4-small",
        "dataset": "owner/uploaded-dataset",
        "dataset_config": "normalized",
        "dataset_rows": 42,
    }


def test_approval_record_adds_recoverable_identity_and_expiry():
    tc = ToolCall(
        id="call-gcp-1",
        type="function",
        function={
            "name": "gcp_vertex_jobs",
            "arguments": '{"operation":"run"}',
        },
    )

    record = agent_loop._approval_record(
        tc,
        "gcp_vertex_jobs",
        {"operation": "run", "output_policy": "cloud-private"},
    )

    assert record["approval_id"] == "call-gcp-1"
    assert record["tool_call_id"] == "call-gcp-1"
    assert record["tool"] == "gcp_vertex_jobs"
    assert record["operation"] == "run"
    assert record["provider"] == "gcp-vertex"
    assert record["status"] == "pending"
    assert record["created_at"]
    assert record["expires_at"]


def test_typed_approval_words_are_detected_without_auto_launching():
    assert agent_loop._looks_like_typed_approval("approved")
    assert agent_loop._looks_like_typed_approval("Run it")
    assert not agent_loop._looks_like_typed_approval("please change the dataset")


@pytest.mark.asyncio
async def test_gcp_vertex_job_under_cap_auto_runs_when_cost_is_known(monkeypatch):
    async def fake_estimate(*args, **kwargs):
        return CostEstimate(estimated_cost_usd=1.5, billable=True)

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)

    decision = await agent_loop._approval_decision(
        "gcp_vertex_jobs",
        {"operation": "run", "machine_type": "n1-standard-8", "max_run_hours": 2},
        _session(cap=5.0, spent=1.0),
    )

    assert decision.requires_approval is False
    assert decision.auto_approved is True
    assert decision.estimated_cost_usd == 1.5


@pytest.mark.asyncio
async def test_gcp_vertex_global_yolo_still_requires_manual_approval(monkeypatch):
    async def fake_estimate(*args, **kwargs):
        return CostEstimate(estimated_cost_usd=1.5, billable=True)

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)
    session = _session(enabled=False, cap=None, spent=0.0)
    session.config.yolo_mode = True

    decision = await agent_loop._approval_decision(
        "gcp_vertex_jobs",
        {"operation": "run", "machine_type": "n1-standard-8", "max_run_hours": 1},
        session,
    )

    assert decision.requires_approval is True
    assert decision.auto_approval_blocked is True
    assert decision.auto_approved is False
    assert decision.estimated_cost_usd == 1.5
    assert "manual approval" in decision.block_reason


@pytest.mark.asyncio
async def test_gcp_vertex_unknown_cost_blocks_auto_approval(monkeypatch):
    async def fake_estimate(*args, **kwargs):
        return CostEstimate(
            estimated_cost_usd=None,
            billable=True,
            block_reason="Vertex AI cost requires max_run_hours.",
        )

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)

    decision = await agent_loop._approval_decision(
        "gcp_vertex_jobs",
        {"operation": "run", "machine_type": "n1-standard-8"},
        _session(cap=5.0, spent=0.0),
    )

    assert decision.requires_approval is True
    assert decision.auto_approval_blocked is True
    assert "max_run_hours" in decision.block_reason


@pytest.mark.asyncio
async def test_immediate_hf_job_over_cap_falls_back_to_approval(monkeypatch):
    async def fake_estimate(*args, **kwargs):
        return CostEstimate(estimated_cost_usd=2.0, billable=True)

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)

    decision = await agent_loop._approval_decision(
        "hf_jobs",
        {"operation": "run", "hardware_flavor": "a10g-large", "timeout": "1h"},
        _session(cap=5.0, spent=4.0),
    )

    assert decision.requires_approval is True
    assert decision.auto_approval_blocked is True
    assert "exceeds" in decision.block_reason
    assert decision.remaining_cap_usd == 1.0


@pytest.mark.asyncio
async def test_unknown_cost_falls_back_to_approval(monkeypatch):
    async def fake_estimate(*args, **kwargs):
        return CostEstimate(
            estimated_cost_usd=None,
            billable=True,
            block_reason="No price is available.",
        )

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)

    decision = await agent_loop._approval_decision(
        "sandbox_create",
        {"hardware": "mystery-gpu"},
        _session(),
    )

    assert decision.requires_approval is True
    assert decision.auto_approval_blocked is True
    assert decision.estimated_cost_usd is None


@pytest.mark.asyncio
async def test_batch_reservation_blocks_second_over_budget_job(monkeypatch):
    async def fake_estimate(*args, **kwargs):
        return CostEstimate(estimated_cost_usd=3.0, billable=True)

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)
    session = _session(cap=5.0, spent=0.0)

    first = await agent_loop._approval_decision(
        "hf_jobs",
        {"operation": "run", "hardware_flavor": "a10g-large"},
        session,
        reserved_spend_usd=0.0,
    )
    second = await agent_loop._approval_decision(
        "hf_jobs",
        {"operation": "run", "hardware_flavor": "a10g-large"},
        session,
        reserved_spend_usd=first.estimated_cost_usd or 0.0,
    )

    assert first.requires_approval is False
    assert second.requires_approval is True
    assert second.remaining_cap_usd == 2.0


@pytest.mark.asyncio
async def test_manual_approval_does_not_record_spend_when_session_yolo_disabled(
    monkeypatch,
):
    called = False

    async def fake_estimate(*args, **kwargs):
        nonlocal called
        called = True
        return CostEstimate(estimated_cost_usd=2.0, billable=True)

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)
    session = _session(enabled=False, cap=5.0, spent=0.0)

    await agent_loop._record_manual_approved_spend_if_needed(
        session,
        "sandbox_create",
        {"hardware": "a10g-large"},
    )

    assert called is False
    assert session.auto_approval_estimated_spend_usd == 0.0


@pytest.mark.asyncio
async def test_manual_approval_records_spend_when_session_yolo_enabled(monkeypatch):
    async def fake_estimate(*args, **kwargs):
        return CostEstimate(estimated_cost_usd=1.25, billable=True)

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)
    session = _session(enabled=True, cap=5.0, spent=0.5)

    await agent_loop._record_manual_approved_spend_if_needed(
        session,
        "sandbox_create",
        {"hardware": "a10g-large"},
    )

    assert session.auto_approval_estimated_spend_usd == 1.75
