import pytest

from agent.tools.training_planner_tool import (
    TRAINING_PLANNER_TOOL_SPEC,
    training_planner_handler,
)


@pytest.mark.asyncio
async def test_training_planner_recommend_operation_returns_markdown_plan():
    output, ok = await training_planner_handler(
        {
            "operation": "recommend",
            "provider": "gcp-vertex",
            "domain": "medical",
            "training_goal": "smoke-test",
            "dataset_summary": {
                "rows": 100,
                "columns": ["question", "answer"],
                "source_format": "csv",
            },
            "task_type": "sft",
            "privacy_level": "sensitive",
            "budget_preference": "balanced",
        }
    )

    assert ok is True
    assert "## Training Planner Recommendation" in output
    assert "**Recommended model:**" in output
    assert "**Recommended hardware:**" in output
    assert "**Output policy:** cloud-private" in output
    assert "Google Cloud Storage" in output
    assert "Risks" in output


@pytest.mark.asyncio
async def test_training_planner_no_dataset_available_guides_discovery_first():
    output, ok = await training_planner_handler(
        {
            "operation": "recommend",
            "provider": "aws-sagemaker",
            "domain": "finance",
            "training_goal": "production",
            "uploaded_dataset_available": False,
        }
    )

    assert ok is True
    assert "dataset discovery is required" in output.lower()
    assert "dataset_discovery" in output
    assert "do not launch" in output.lower()


@pytest.mark.asyncio
async def test_training_planner_tool_formats_aws_cloud_private_storage():
    output, ok = await training_planner_handler(
        {
            "operation": "recommend",
            "provider": "aws-sagemaker",
            "domain": "medical",
            "training_goal": "production",
            "dataset_summary": {"rows": 1_500},
        }
    )

    assert ok is True
    assert "**Output policy:** cloud-private" in output
    assert "Amazon S3" in output


@pytest.mark.asyncio
async def test_training_planner_tool_formats_hf_jobs_privacy_warning():
    output, ok = await training_planner_handler(
        {
            "operation": "recommend",
            "provider": "hf-jobs",
            "domain": "legal",
            "training_goal": "production",
            "dataset_summary": {"rows": 1_500},
        }
    )

    assert ok is True
    assert "**Output policy:** cloud-private" in output
    assert "private Hub" in output
    assert "job artifact" in output


@pytest.mark.asyncio
async def test_training_planner_unknown_operation_returns_clear_error():
    output, ok = await training_planner_handler({"operation": "launch"})

    assert ok is False
    assert "Unknown operation" in output
    assert "recommend" in output


def test_training_planner_schema_is_read_only_recommend_only():
    assert TRAINING_PLANNER_TOOL_SPEC["name"] == "training_planner"
    assert "never launches jobs" in TRAINING_PLANNER_TOOL_SPEC["description"]
    operation = TRAINING_PLANNER_TOOL_SPEC["parameters"]["properties"]["operation"]
    assert operation["enum"] == ["recommend"]
    assert TRAINING_PLANNER_TOOL_SPEC["parameters"]["required"] == ["operation"]
