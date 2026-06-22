"""Stage normalized Hugging Face datasets to GCS for Vertex custom jobs."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from datasets import load_dataset

DEFAULT_HF_LOAD_RETRIES = 3
DEFAULT_HF_LOAD_RETRY_SECONDS = 2.0
DEFAULT_HF_LOAD_TIMEOUT_SECONDS = 45.0
DEFAULT_BOUNDED_STAGING_ROWS = 100

METADATA_COLUMN_NAMES = frozenset(
    {
        "coherence",
        "complexity",
        "correctness",
        "helpfulness",
        "verbosity",
        "rating",
        "score",
    }
)

SUPPORTED_SCHEMA_LABELS = (
    "messages, text, prompt+completion, prompt+chosen(+rejected), "
    "instruction/output, instruction/output+context, "
    "question/answer, question/response, user/assistant"
)


@dataclass(frozen=True)
class GcpDatasetStagingResult:
    train_gcs_uri: str
    gcs_prefix_uri: str
    row_count: int
    bytes_uploaded: int
    dataset_name: str
    dataset_config: str | None
    dataset_split: str
    source_format: str
    detected_schema: str


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _reject_binary(value: Any, *, row_index: int) -> Any:
    if isinstance(value, bytes | bytearray | memoryview):
        raise ValueError(f"Dataset row {row_index} contains binary data.")
    if isinstance(value, dict):
        return {
            str(key): _reject_binary(child, row_index=row_index)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_reject_binary(child, row_index=row_index) for child in value]
    if isinstance(value, tuple):
        return [_reject_binary(child, row_index=row_index) for child in value]
    return value


def _messages_have_user_and_assistant(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    has_user = False
    has_assistant = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        content = _string_value(message.get("content"))
        if role == "user" and content:
            has_user = True
        if role == "assistant" and content:
            has_assistant = True
    return has_user and has_assistant


def _messages_from_pair(
    row: dict[str, Any],
    user_column: str,
    assistant_columns: list[str],
    context_column: str | None = None,
) -> dict[str, Any]:
    user_text = _string_value(row.get(user_column))
    if not user_text:
        raise ValueError(f"Missing user content in column {user_column!r}.")
    if context_column:
        context_text = _string_value(row.get(context_column))
        if context_text:
            user_text = f"{context_text}\n\n{user_text}"
    assistant_text = "\n\n".join(
        _string_value(row.get(column))
        for column in assistant_columns
        if _string_value(row.get(column))
    )
    if not assistant_text:
        raise ValueError(f"Missing assistant content in columns {assistant_columns!r}.")
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def _assistant_columns_from_mapping(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    return [str(value).strip()]


def resolve_mapping_source_columns(
    row: dict[str, Any], column_mapping: dict[str, Any] | None
) -> tuple[str, list[str]] | None:
    """Resolve role-style column_mapping to actual dataset columns when possible."""
    if not isinstance(column_mapping, dict) or not column_mapping:
        return None
    user_column = column_mapping.get("user") or column_mapping.get("question")
    assistant_columns = _assistant_columns_from_mapping(
        column_mapping.get("assistant") or column_mapping.get("response")
    )
    if not user_column or not assistant_columns:
        return None
    user_name = str(user_column).strip()
    if user_name not in row:
        return None
    resolved_assistant = [
        column
        for column in assistant_columns
        if column in row and _string_value(row.get(column))
    ]
    if not _string_value(row.get(user_name)) or not resolved_assistant:
        return None
    return user_name, resolved_assistant


def detect_dataset_schema(
    row: dict[str, Any],
    *,
    column_mapping: dict[str, Any] | None = None,
) -> str | None:
    """Return a supported schema label for a dataset row, or None if unsupported."""
    mapped = resolve_mapping_source_columns(row, column_mapping)
    if mapped is not None:
        user_column, assistant_columns = mapped
        assistant_col = assistant_columns[0]
        # If the mapped user column is "instruction" and the row has a non-empty
        # "input" context column, promote to the _with_context schema so that the
        # optional context field is prepended to the user message.
        if (
            user_column == "instruction"
            and assistant_col in ("output", "response")
            and "input" in row
            and _string_value(row.get("input"))
        ):
            return f"instruction_{assistant_col}_with_context"
        return f"{user_column}_{assistant_col}"

    if _messages_have_user_and_assistant(row.get("messages")):
        return "messages"
    if _string_value(row.get("text")):
        return "text"
    if _string_value(row.get("prompt")) and _string_value(row.get("completion")):
        return "prompt_completion"
    if _string_value(row.get("prompt")) and _string_value(row.get("chosen")):
        return "prompt_chosen_rejected"
    for user_column, assistant_column in (
        ("instruction", "output"),
        ("instruction", "response"),
        ("input", "output"),
        ("input", "response"),
        ("question", "response"),
        ("question", "answer"),
        ("user", "assistant"),
        ("prompt", "response"),
    ):
        if user_column in row and assistant_column in row:
            if _string_value(row.get(user_column)) and _string_value(
                row.get(assistant_column)
            ):
                return f"{user_column}_{assistant_column}"
    # Fallback: instruction + output/response with optional input context column.
    # Handles datasets like sahil2801/CodeAlpaca-20k (instruction, input, output).
    for assistant_column in ("output", "response"):
        if (
            "instruction" in row
            and assistant_column in row
            and _string_value(row.get("instruction"))
            and _string_value(row.get(assistant_column))
        ):
            if "input" in row:
                return f"instruction_{assistant_column}_with_context"
            return f"instruction_{assistant_column}"
    return None


def normalize_row_to_sft(
    row: dict[str, Any],
    *,
    column_mapping: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Normalize one dataset row to SFT JSONL shape and return schema label."""
    schema = detect_dataset_schema(row, column_mapping=column_mapping)
    if schema is None:
        content_keys = sorted(
            str(key)
            for key in row.keys()
            if str(key).lower() not in METADATA_COLUMN_NAMES
        )
        keys = ", ".join(content_keys)
        raise ValueError(
            "Unsupported dataset schema. Expected one of: "
            f"{SUPPORTED_SCHEMA_LABELS}. Found columns: {keys}"
        )
    if schema == "messages":
        return {"messages": row["messages"]}, schema
    if schema == "text":
        return {"text": _string_value(row.get("text"))}, schema
    if schema == "prompt_completion":
        return _messages_from_pair(row, "prompt", ["completion"]), schema
    if schema == "prompt_chosen_rejected":
        return _messages_from_pair(row, "prompt", ["chosen"]), schema
    if schema == "instruction_output":
        return _messages_from_pair(row, "instruction", ["output"]), schema
    if schema == "instruction_output_with_context":
        return _messages_from_pair(row, "instruction", ["output"], context_column="input"), schema
    if schema == "instruction_response":
        return _messages_from_pair(row, "instruction", ["response"]), schema
    if schema == "instruction_response_with_context":
        return _messages_from_pair(row, "instruction", ["response"], context_column="input"), schema
    if schema == "input_output":
        return _messages_from_pair(row, "input", ["output"]), schema
    if schema == "input_response":
        return _messages_from_pair(row, "input", ["response"]), schema
    if schema == "question_answer":
        return _messages_from_pair(row, "question", ["answer"]), schema
    if schema == "question_response":
        return _messages_from_pair(row, "question", ["response"]), schema
    if schema == "user_assistant":
        return _messages_from_pair(row, "user", ["assistant"]), schema
    if schema == "prompt_response":
        return _messages_from_pair(row, "prompt", ["response"]), schema
    mapped = resolve_mapping_source_columns(row, column_mapping)
    if mapped is not None:
        user_column, assistant_columns = mapped
        return _messages_from_pair(row, user_column, assistant_columns), schema
    raise ValueError(f"Unsupported dataset schema: {schema}")


def _bounded_rows(
    dataset: Any,
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    limit = max(1, int(max_rows))
    rows: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(dataset, start=1):
        if len(rows) >= limit:
            break
        row = dict(raw_row) if not isinstance(raw_row, dict) else raw_row
        if not isinstance(row, dict):
            raise ValueError(
                f"Dataset row {row_index} must serialize as a JSON object."
            )
        rows.append(_reject_binary(row, row_index=row_index))
    if not rows:
        raise ValueError("Loaded dataset split contains no rows.")
    return rows


def _rows_to_normalized_jsonl_bytes(
    rows: list[dict[str, Any]],
    *,
    column_mapping: dict[str, Any] | None = None,
) -> tuple[bytes, int, str, str]:
    normalized_rows: list[dict[str, Any]] = []
    detected_schema = ""
    source_format = ""
    for row_index, row in enumerate(rows, start=1):
        normalized, schema = normalize_row_to_sft(row, column_mapping=column_mapping)
        normalized_rows.append(normalized)
        if not detected_schema:
            detected_schema = schema
            source_format = schema
        elif schema != detected_schema:
            raise ValueError(
                f"Dataset row {row_index} uses schema {schema!r}, but earlier rows "
                f"use {detected_schema!r}. Mixed schemas are not supported."
            )
    lines = [
        json.dumps(item, ensure_ascii=False, sort_keys=True) for item in normalized_rows
    ]
    return (
        ("\n".join(lines) + "\n").encode("utf-8"),
        len(lines),
        source_format,
        detected_schema,
    )


def _gs_uri(bucket: str, key: str) -> str:
    return f"gs://{bucket}/{key}"


def _load_dataset_with_retries(
    *,
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    hf_token: str | None,
    retries: int = DEFAULT_HF_LOAD_RETRIES,
    retry_seconds: float = DEFAULT_HF_LOAD_RETRY_SECONDS,
    timeout_seconds: float = DEFAULT_HF_LOAD_TIMEOUT_SECONDS,
    loader: Callable[..., Any] | None = None,
) -> Any:
    load_fn = loader or load_dataset
    kwargs: dict[str, Any] = {
        "path": dataset_name,
        "split": dataset_split,
    }
    if dataset_config:
        kwargs["name"] = dataset_config
    if hf_token:
        kwargs["token"] = hf_token

    last_error: Exception | None = None
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            return load_fn(**kwargs)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            time.sleep(max(0.0, float(retry_seconds)))
    raise RuntimeError(
        "Dataset staging failed before provider launch. "
        f"Could not load Hugging Face dataset {dataset_name!r} after {attempts} "
        f"attempt(s) within about {timeout_seconds:.0f}s: {last_error}"
    ) from last_error


async def stage_hf_dataset_to_gcs(
    *,
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    gcs_bucket: str,
    display_name: str,
    hf_token: str | None = None,
    max_rows: int = DEFAULT_BOUNDED_STAGING_ROWS,
    upload_file: Callable[[str, str], None] | None = None,
    loader: Callable[..., Any] | None = None,
    column_mapping: dict[str, Any] | None = None,
) -> GcpDatasetStagingResult:
    """Load a bounded Hub split, normalize to SFT JSONL, and upload to GCS."""

    if not dataset_name.strip():
        raise ValueError("dataset_name is required for Hub dataset staging.")
    bounded_rows = max(1, int(max_rows))

    try:
        dataset = await asyncio.to_thread(
            _load_dataset_with_retries,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            dataset_split=dataset_split,
            hf_token=hf_token,
            loader=loader,
        )
    except Exception as exc:
        message = str(exc)
        if "Dataset staging failed before provider launch." not in message:
            raise RuntimeError(
                "Dataset staging failed before provider launch. "
                f"Could not load Hugging Face dataset: {exc}"
            ) from exc
        raise

    try:
        raw_rows = _bounded_rows(dataset, max_rows=bounded_rows)
        payload, row_count, source_format, detected_schema = (
            _rows_to_normalized_jsonl_bytes(raw_rows, column_mapping=column_mapping)
        )
    except ValueError as exc:
        raise RuntimeError(
            "Dataset staging failed before provider launch. "
            f"Schema normalization failed: {exc}"
        ) from exc

    prefix = f"vertex-inputs/{display_name}"
    train_key = f"{prefix}/train.jsonl"
    metadata_key = f"{prefix}/metadata.json"
    train_gcs_uri = _gs_uri(gcs_bucket, train_key)
    metadata_gcs_uri = _gs_uri(gcs_bucket, metadata_key)
    metadata = {
        "dataset_source": "staged_gcs_jsonl",
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_split": dataset_split,
        "source_format": source_format,
        "detected_schema": detected_schema,
        "normalized_row_count": row_count,
        "bounded_sample_rows": bounded_rows,
        "staged_train_uri": train_gcs_uri,
    }

    if upload_file is None:
        try:
            from google.cloud import storage
        except Exception as exc:
            raise RuntimeError(
                f"google-cloud-storage is required to upload staged datasets: {exc}"
            ) from exc

        def _default_upload(local_path: str, gcs_uri: str) -> None:
            bucket_name, blob_name = gcs_uri.replace("gs://", "", 1).split("/", 1)
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            bucket.blob(blob_name).upload_from_filename(local_path)

        upload_file = _default_upload

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".jsonl", delete=False
    ) as train_handle:
        train_handle.write(payload)
        train_path = train_handle.name
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as metadata_handle:
        json.dump(metadata, metadata_handle, sort_keys=True)
        metadata_path = metadata_handle.name

    try:
        await asyncio.to_thread(upload_file, train_path, train_gcs_uri)
        await asyncio.to_thread(upload_file, metadata_path, metadata_gcs_uri)
    except Exception as exc:
        raise RuntimeError(
            "Dataset staging failed before provider launch. "
            f"Could not upload staged dataset to GCS: {exc}"
        ) from exc
    finally:
        for path in (train_path, metadata_path):
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass

    return GcpDatasetStagingResult(
        train_gcs_uri=train_gcs_uri,
        gcs_prefix_uri=_gs_uri(gcs_bucket, f"{prefix}/"),
        row_count=row_count,
        bytes_uploaded=len(payload),
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        source_format=source_format,
        detected_schema=detected_schema,
    )
