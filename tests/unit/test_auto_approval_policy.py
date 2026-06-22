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
        **overrides,
    }
    return Config.model_validate(data)


def _session(*, cap=5.0, spent=0.0, enabled=True):
    return SimpleNamespace(
        config=_config(),
        sandbox=None,
        logged_events=[],
        context_manager=SimpleNamespace(items=[]),
    )


def _aws_state(state: str):
    return {
        "event_type": "tool_state_change",
        "data": {"tool": "aws_sagemaker_jobs", "state": state, "jobName": "job-1"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    ["scheduled run", "scheduled uv", "scheduled  run"],
)
async def test_scheduled_hf_jobs_always_require_manual_approval(operation):
    session = _session()

    decision = await agent_loop._approval_decision(
        "hf_jobs",
        {"operation": operation, "script": "print(1)"},
        session,
    )

    assert decision.requires_approval is True
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


def test_aws_sagemaker_run_and_cancel_require_approval():
    config = _config(confirm_cpu_jobs=False)

    assert agent_loop._needs_approval(
        "aws_sagemaker_jobs", {"operation": "run"}, config
    )
    assert agent_loop._needs_approval(
        "aws_sagemaker_jobs",
        {"operation": "cancel", "job_name": "training-job"},
        config,
    )


def test_aws_sagemaker_read_only_operations_do_not_require_approval():
    config = _config(confirm_cpu_jobs=True)

    for operation in ["ps", "logs", "inspect"]:
        assert not agent_loop._needs_approval(
            "aws_sagemaker_jobs",
            {"operation": operation, "job_name": "training-job"},
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
async def test_immediate_hf_job_requires_approval(monkeypatch):
    async def fake_estimate(*args, **kwargs):
        return CostEstimate(estimated_cost_usd=2.0, billable=True)

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)

    decision = await agent_loop._approval_decision(
        "hf_jobs",
        {"operation": "run", "hardware_flavor": "a10g-large", "timeout": "1h"},
        _session(cap=5.0, spent=1.0),
    )

    assert decision.requires_approval is True
    assert decision.estimated_cost_usd == 2.0


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


def test_aws_sagemaker_approval_metadata_includes_provider_dataset_and_runtime():
    session = SimpleNamespace(
        training_goal="production",
        output_policy="cloud-private",
        uploaded_datasets=[
            {
                "repo_id": "owner/uploaded-hardware-dataset",
                "config_name": "normalized",
                "normalized_row_count": 40,
            }
        ],
    )

    metadata = agent_loop._approval_metadata(
        session,
        "aws_sagemaker_jobs",
        {
            "operation": "run",
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "instance_type": "ml.g5.xlarge",
            "instance_count": 1,
            "max_run_seconds": 7200,
            "output_policy": "cloud-private",
        },
    )

    assert metadata == {
        "provider": "aws-sagemaker",
        "training_goal": "production",
        "output_policy": "cloud-private",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "instance_type": "ml.g5.xlarge",
        "instance_count": 1,
        "max_run_seconds": 7200,
        "dataset": "owner/uploaded-hardware-dataset",
        "dataset_config": "normalized",
        "dataset_rows": 40,
    }


def test_gcp_vertex_approval_metadata_includes_preflight_summary():
    session = SimpleNamespace(
        training_goal="smoke-test",
        output_policy="cloud-private",
        uploaded_datasets=[],
        latest_training_preflight={
            "preflight_id": "pf-1",
            "status": "unknown",
            "manual_approval_allowed": True,
            "launch_ready": False,
        },
    )

    metadata = agent_loop._approval_metadata(
        session,
        "gcp_vertex_jobs",
        {
            "operation": "run",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "column_mapping": {"question": "question", "response": "response"},
        },
    )

    assert metadata["provider"] == "gcp-vertex"
    assert metadata["training_preflight"] == {
        "preflight_id": "pf-1",
        "status": "unknown",
        "manual_approval_allowed": True,
        "launch_ready": False,
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


def test_aws_sagemaker_approval_record_adds_provider_identity():
    tc = ToolCall(
        id="call-aws-1",
        type="function",
        function={
            "name": "aws_sagemaker_jobs",
            "arguments": '{"operation":"run"}',
        },
    )

    record = agent_loop._approval_record(
        tc,
        "aws_sagemaker_jobs",
        {"operation": "run", "output_policy": "cloud-private"},
    )

    assert record["approval_id"] == "call-aws-1"
    assert record["tool_call_id"] == "call-aws-1"
    assert record["tool"] == "aws_sagemaker_jobs"
    assert record["operation"] == "run"
    assert record["provider"] == "aws-sagemaker"
    assert record["status"] == "pending"
    assert record["created_at"]
    assert record["expires_at"]


def test_typed_approval_words_are_detected_without_auto_launching():
    assert agent_loop._looks_like_typed_approval("approved")
    assert agent_loop._looks_like_typed_approval("Run it")
    assert not agent_loop._looks_like_typed_approval("please change the dataset")


@pytest.mark.asyncio
async def test_gcp_vertex_job_requires_approval(monkeypatch):
    async def fake_estimate(*args, **kwargs):
        return CostEstimate(estimated_cost_usd=1.5, billable=True)

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)

    decision = await agent_loop._approval_decision(
        "gcp_vertex_jobs",
        {"operation": "run", "machine_type": "n1-standard-8", "max_run_hours": 2},
        _session(cap=5.0, spent=1.0),
    )

    assert decision.requires_approval is True
    assert decision.estimated_cost_usd == 1.5


@pytest.mark.asyncio
async def test_second_aws_sagemaker_run_after_terminal_job_requires_manual_approval(
    monkeypatch,
):
    async def fake_estimate(*args, **kwargs):
        return CostEstimate(estimated_cost_usd=1.5, billable=True)

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)
    session = _session(cap=5.0, spent=1.0)
    session.logged_events = [_aws_state("failed")]
    session.aws_sagemaker_retry_authorized = True

    decision = await agent_loop._approval_decision(
        "aws_sagemaker_jobs",
        {
            "operation": "run",
            "instance_type": "ml.g5.xlarge",
            "max_run_seconds": 3600,
        },
        session,
    )

    assert decision.requires_approval is True
    assert "second paid AWS SageMaker run" in decision.block_reason


@pytest.mark.asyncio
async def test_aws_sagemaker_cancel_requires_approval():
    decision = await agent_loop._approval_decision(
        "aws_sagemaker_jobs",
        {"operation": "cancel", "job_name": "training-job"},
        _session(cap=5.0, spent=0.0),
    )

    assert decision.requires_approval is True
    assert "cancellation" in decision.block_reason
