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
async def test_dataset_discovery_plan_persists_explicit_no_candidate_reason():
    class FakeSession:
        latest_dataset_discovery = None

    session = FakeSession()

    output, ok = await dataset_discovery_handler(
        {
            "operation": "plan",
            "query": "Find GST tax support datasets",
            "provider": "gcp-vertex",
        },
        session=session,
        tool_call_id="tool-1",
    )

    assert ok is True
    assert "No candidate datasets supplied yet" in output
    assert session.latest_dataset_discovery["candidates"] == []
    assert session.latest_dataset_discovery["no_candidates_reason"]


@pytest.mark.asyncio
async def test_dataset_discovery_plan_includes_intent_scores_and_load_dataset_snippet():
    output, ok = await dataset_discovery_handler(
        {
            "operation": "plan",
            "query": "Find a safe hardware troubleshooting dataset for AWS SageMaker fine-tuning.",
            "provider": "aws-sagemaker",
            "candidates": [
                {
                    "dataset_id": "public/hardware-support",
                    "source": "huggingface",
                    "repo_id": "public/hardware-support",
                    "title": "Hardware Support QA",
                    "description": "Instruction response data for PC troubleshooting",
                    "license": "mit",
                    "columns": ["instruction", "output", "category"],
                    "row_count": 5_000,
                }
            ],
        }
    )

    assert ok is True
    assert "Extracted Intent" in output
    assert "hardware_support" in output
    assert "Recommended" in output
    assert "Overall score" in output
    assert "License: mit (clear)" in output
    assert "Privacy: low" in output
    assert "Schema: compatible" in output
    assert "from datasets import load_dataset" in output
    assert "User selection required before training" in output


@pytest.mark.asyncio
async def test_dataset_discovery_plan_persists_gst_candidate_results():
    class FakeSession:
        latest_dataset_discovery = None

    session = FakeSession()

    _output, ok = await dataset_discovery_handler(
        {
            "operation": "plan",
            "query": "Find GST tax support data for Vertex fine-tuning",
            "provider": "gcp-vertex",
            "candidates": [
                {
                    "dataset_id": "transitionGap/gst-india-preference-dataset-prep-small",
                    "source": "huggingface",
                    "repo_id": "transitionGap/gst-india-preference-dataset-prep-small",
                    "title": "GST India Preference Dataset Prep Small",
                    "license": "unknown",
                    "columns": ["prompt", "chosen", "rejected"],
                    "row_count": 1000,
                },
                {
                    "dataset_id": "Kahrhoff/openfinancial-chatbot-dataset",
                    "source": "huggingface",
                    "repo_id": "Kahrhoff/openfinancial-chatbot-dataset",
                    "title": "Open Financial Chatbot Dataset",
                    "license": "unknown",
                    "columns": ["question", "answer"],
                    "row_count": 1000,
                },
            ],
        },
        session=session,
        tool_call_id="tool-2",
    )

    assert ok is True
    assert [
        candidate["dataset_id"]
        for candidate in session.latest_dataset_discovery["candidates"]
    ] == [
        "transitionGap/gst-india-preference-dataset-prep-small",
        "Kahrhoff/openfinancial-chatbot-dataset",
    ]


@pytest.mark.asyncio
async def test_dataset_discovery_tool_preserves_extracted_intent_without_overrides():
    output, ok = await dataset_discovery_handler(
        {
            "operation": "plan",
            "query": "I need house price prediction data for a small model.",
            "provider": "hf-jobs",
        }
    )

    assert ok is True
    assert "Domain: real_estate" in output
    assert "Task type: regression" in output


@pytest.mark.asyncio
async def test_dataset_discovery_excludes_kaggle_as_future_work():
    output, ok = await dataset_discovery_handler(
        {
            "operation": "plan",
            "query": "Find IPL cricket data",
            "candidates": [
                {
                    "dataset_id": "kaggle/ipl-matches",
                    "source": "kaggle",
                    "title": "IPL Matches",
                    "license": "unknown",
                    "columns": ["team", "runs"],
                }
            ],
        }
    )

    assert ok is True
    assert "Kaggle (future work only; not connected)" in output
    assert "Excluded: Kaggle is future work only." in output


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
