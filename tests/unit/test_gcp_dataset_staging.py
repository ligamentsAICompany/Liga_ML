import json

import pytest

from agent.core.gcp_dataset_staging import (
    GcpDatasetStagingResult,
    detect_dataset_schema,
    normalize_row_to_sft,
    resolve_mapping_source_columns,
    stage_hf_dataset_to_gcs,
)


def test_detect_preference_schema():
    row = {
        "prompt": "What is GST?",
        "chosen": "GST is a consumption tax.",
        "rejected": "GST is optional everywhere.",
    }
    assert detect_dataset_schema(row) == "prompt_chosen_rejected"


def test_normalize_preference_row_uses_chosen_for_sft():
    row = {
        "prompt": "What is GST?",
        "chosen": "GST is a consumption tax.",
        "rejected": "GST is optional everywhere.",
    }
    normalized, schema = normalize_row_to_sft(row)
    assert schema == "prompt_chosen_rejected"
    assert normalized["messages"][0]["content"] == "What is GST?"
    assert normalized["messages"][1]["content"] == "GST is a consumption tax."


def test_normalize_messages_schema_passes_through():
    row = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    }
    normalized, schema = normalize_row_to_sft(row)
    assert schema == "messages"
    assert normalized["messages"] == row["messages"]


def test_normalize_question_answer_schema():
    row = {"question": "Rate?", "answer": "18%"}
    normalized, schema = normalize_row_to_sft(row)
    assert schema == "question_answer"
    assert normalized["messages"][0]["role"] == "user"


def test_detect_question_response_schema():
    row = {
        "question": "What is GST?",
        "response": "Goods and Services Tax.",
        "coherence": 4.2,
        "complexity": 2.1,
        "correctness": 4.8,
        "helpfulness": 4.5,
        "verbosity": 3.0,
    }
    assert detect_dataset_schema(row) == "question_response"


def test_normalize_question_response_with_metadata_columns():
    row = {
        "question": "What is GST?",
        "response": "Goods and Services Tax.",
        "coherence": 4.2,
        "complexity": 2.1,
        "correctness": 4.8,
        "helpfulness": 4.5,
        "verbosity": 3.0,
    }
    normalized, schema = normalize_row_to_sft(row)
    assert schema == "question_response"
    assert normalized["messages"][0]["content"] == "What is GST?"
    assert normalized["messages"][1]["content"] == "Goods and Services Tax."


def test_column_mapping_user_question_assistant_response():
    row = {
        "question": "Who pays GST?",
        "response": "Businesses collect it.",
        "coherence": 1,
    }
    mapping = {"user": "question", "assistant": "response"}
    assert resolve_mapping_source_columns(row, mapping) == ("question", ["response"])
    normalized, schema = normalize_row_to_sft(row, column_mapping=mapping)
    assert schema == "question_response"
    assert normalized["messages"][1]["content"] == "Businesses collect it."


def test_bad_column_mapping_does_not_override_auto_detect():
    row = {"question": "Rate?", "response": "18%"}
    mapping = {"user": "missing_user_col", "assistant": "missing_response_col"}
    assert resolve_mapping_source_columns(row, mapping) is None
    normalized, schema = normalize_row_to_sft(row, column_mapping=mapping)
    assert schema == "question_response"


def test_unsupported_schema_fails_before_launch():
    with pytest.raises(ValueError, match="Unsupported dataset schema"):
        normalize_row_to_sft({"foo": "bar", "baz": "qux"})


@pytest.mark.asyncio
async def test_stage_hf_dataset_uploads_question_response_jsonl(monkeypatch):
    uploads: list[tuple[str, str]] = []

    class FakeSplit:
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            return iter(self._rows)

    def fake_loader(**kwargs):
        return FakeSplit(
            [
                {
                    "question": "What is GST?",
                    "response": "Goods and Services Tax.",
                    "coherence": 4.2,
                    "helpfulness": 4.5,
                }
            ]
        )

    def fake_upload(local_path, gcs_uri):
        uploads.append((local_path, gcs_uri))
        if not str(gcs_uri).endswith("train.jsonl"):
            return
        with open(local_path, encoding="utf-8") as handle:
            payload = handle.read()
        lines = [json.loads(line) for line in payload.splitlines() if line.strip()]
        assert lines[0]["messages"][1]["content"] == "Goods and Services Tax."

    result = await stage_hf_dataset_to_gcs(
        dataset_name="transitionGap/gst-india-preference-dataset-prep-small",
        dataset_config=None,
        dataset_split="train",
        gcs_bucket="liga-training",
        display_name="gst-smoke",
        max_rows=1,
        loader=fake_loader,
        upload_file=fake_upload,
        column_mapping={"user": "question", "assistant": "response"},
    )

    assert result.detected_schema == "question_response"
    assert result.row_count == 1


@pytest.mark.asyncio
async def test_stage_hf_dataset_uploads_bounded_jsonl(monkeypatch):
    uploads: list[tuple[str, str]] = []

    class FakeSplit:
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            return iter(self._rows)

    def fake_loader(**kwargs):
        assert kwargs["path"] == "transitionGap/gst-india-preference-dataset-prep-small"
        return FakeSplit(
            [
                {
                    "prompt": "What is GST?",
                    "chosen": "Goods and Services Tax.",
                    "rejected": "No tax.",
                },
                {
                    "prompt": "Who pays GST?",
                    "chosen": "Businesses collect it.",
                    "rejected": "Nobody pays GST.",
                },
            ]
        )

    def fake_upload(local_path, gcs_uri):
        uploads.append((local_path, gcs_uri))
        if not str(gcs_uri).endswith("train.jsonl"):
            return
        with open(local_path, encoding="utf-8") as handle:
            payload = handle.read()
        lines = [json.loads(line) for line in payload.splitlines() if line.strip()]
        assert len(lines) == 2
        assert lines[0]["messages"][1]["content"] == "Goods and Services Tax."

    result = await stage_hf_dataset_to_gcs(
        dataset_name="transitionGap/gst-india-preference-dataset-prep-small",
        dataset_config=None,
        dataset_split="train",
        gcs_bucket="liga-training",
        display_name="gst-smoke",
        max_rows=2,
        loader=fake_loader,
        upload_file=fake_upload,
    )

    assert isinstance(result, GcpDatasetStagingResult)
    assert (
        result.train_gcs_uri == "gs://liga-training/vertex-inputs/gst-smoke/train.jsonl"
    )
    assert result.row_count == 2
    assert result.detected_schema == "prompt_chosen_rejected"
    assert any(uri.endswith("train.jsonl") for _, uri in uploads)


@pytest.mark.asyncio
async def test_stage_hf_dataset_reports_load_failure(monkeypatch):
    def failing_loader(**kwargs):
        raise RuntimeError("504 Gateway Time-out")

    with pytest.raises(
        RuntimeError, match="Dataset staging failed before provider launch"
    ):
        await stage_hf_dataset_to_gcs(
            dataset_name="transitionGap/gst-india-preference-dataset-prep-small",
            dataset_config=None,
            dataset_split="train",
            gcs_bucket="liga-training",
            display_name="gst-smoke",
            loader=failing_loader,
            upload_file=lambda *_args, **_kwargs: None,
        )
