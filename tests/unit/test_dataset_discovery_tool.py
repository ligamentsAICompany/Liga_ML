import pytest

from agent.tools.dataset_discovery_tool import (
    DATASET_DISCOVERY_TOOL_SPEC,
    dataset_discovery_handler,
)


@pytest.mark.asyncio
async def test_dataset_discovery_plan_returns_no_upload_guidance():
    output, ok = await dataset_discovery_handler(
        {
            "operation": "plan",
            "domain": "medical",
            "task_type": "sft",
            "provider": "gcp-vertex",
            "user_goal": "fine-tune a medical QA model on GCloud",
            "uploaded_dataset_available": False,
        }
    )

    assert ok is True
    assert "No uploaded dataset detected" in output
    assert "Hugging Face Datasets" in output
    assert "GitHub" in output
    assert "papers" in output
    assert "public web" in output
    assert "Kaggle" in output
    assert "Excluded Sources" in output
    assert "search Hugging Face Datasets" in output
    assert "ask the user to approve" in output
    assert "Planning only" in output


@pytest.mark.asyncio
async def test_dataset_discovery_unknown_operation_returns_error():
    output, ok = await dataset_discovery_handler({"operation": "crawl"})

    assert ok is False
    assert "Unknown operation" in output
    assert "plan" in output


@pytest.mark.asyncio
async def test_dataset_discovery_uploaded_dataset_available_warns_to_use_upload():
    output, ok = await dataset_discovery_handler(
        {
            "operation": "plan",
            "domain": "finance",
            "task_type": "sft",
            "provider": "aws-sagemaker",
            "uploaded_dataset_available": True,
        }
    )

    assert ok is True
    assert "Uploaded dataset available" in output
    assert "use the uploaded normalized dataset first" in output


def test_dataset_discovery_schema_is_read_only_plan_only():
    assert DATASET_DISCOVERY_TOOL_SPEC["name"] == "dataset_discovery"
    assert "never launches jobs" in DATASET_DISCOVERY_TOOL_SPEC["description"]
    operation = DATASET_DISCOVERY_TOOL_SPEC["parameters"]["properties"]["operation"]
    assert operation["enum"] == ["plan"]
    assert DATASET_DISCOVERY_TOOL_SPEC["parameters"]["required"] == ["operation"]
