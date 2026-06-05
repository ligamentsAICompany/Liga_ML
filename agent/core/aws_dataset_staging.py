"""Stage normalized Hugging Face datasets to S3 for future SageMaker jobs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass(frozen=True)
class AwsDatasetStagingResult:
    s3_train_uri: str
    s3_prefix_uri: str
    s3_output_uri: str
    s3_checkpoint_uri: str
    row_count: int
    bytes_uploaded: int
    dataset_name: str
    dataset_config: str | None
    dataset_split: str


def _normalize_s3_prefix(prefix: str | None) -> str:
    cleaned = "/".join(part for part in str(prefix or "").split("/") if part)
    return cleaned or "liga-ml"


def _s3_uri(bucket: str, key_or_prefix: str) -> str:
    return f"s3://{bucket}/{key_or_prefix}"


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


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def _sft_record_supported_fields(row: dict[str, Any]) -> set[str]:
    supported: set[str] = set()
    if _messages_have_user_and_assistant(row.get("messages")):
        supported.add("messages")
    if _string_value(row.get("prompt")) and _string_value(row.get("completion")):
        supported.add("prompt_completion")
    if _string_value(row.get("text")):
        supported.add("text")
    return supported


def _is_supported_sft_record(row: dict[str, Any]) -> bool:
    return bool(_sft_record_supported_fields(row))


def validate_sft_records_for_aws(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Validate staged SFT rows before a paid SageMaker job can be requested."""
    summary = {
        "total_records": len(rows),
        "valid_records": 0,
        "invalid_records": 0,
        "messages_records": 0,
        "prompt_completion_records": 0,
        "text_records": 0,
    }
    for row in rows:
        supported_fields = _sft_record_supported_fields(row)
        if not supported_fields:
            summary["invalid_records"] += 1
            continue
        summary["valid_records"] += 1
        if "messages" in supported_fields:
            summary["messages_records"] += 1
        if "prompt_completion" in supported_fields:
            summary["prompt_completion_records"] += 1
        if "text" in supported_fields:
            summary["text_records"] += 1
    if summary["valid_records"] <= 0:
        raise ValueError(
            "Loaded dataset split contains no valid SFT records. Expected at least one "
            "row with messages, prompt+completion, or text."
        )
    return summary


def _valid_sft_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_sft_records_for_aws(rows)
    return [row for row in rows if _is_supported_sft_record(row)]


def _rows_to_jsonl_bytes(dataset: Any) -> tuple[bytes, int]:
    rows: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(dataset, start=1):
        row = dict(raw_row) if not isinstance(raw_row, dict) else raw_row
        if not isinstance(row, dict):
            raise ValueError(
                f"Dataset row {row_index} must serialize as a JSON object."
            )
        safe_row = _reject_binary(row, row_index=row_index)
        rows.append(safe_row)
    if not rows:
        raise ValueError("Loaded dataset split contains no rows.")

    valid_rows = _valid_sft_rows(rows)
    lines: list[str] = []
    for row_index, safe_row in enumerate(valid_rows, start=1):
        try:
            lines.append(json.dumps(safe_row, ensure_ascii=False, sort_keys=True))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Dataset row {row_index} could not be serialized as JSON: {exc}"
            ) from exc
    return ("\n".join(lines) + "\n").encode("utf-8"), len(lines)


async def stage_hf_dataset_to_s3(
    *,
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    s3_bucket: str,
    s3_prefix: str,
    job_name: str,
    session_id: str | None = None,
    hf_token: str | None = None,
    s3_client: Any | None = None,
) -> AwsDatasetStagingResult:
    """Load a normalized Hub dataset split, reserialize it as JSONL, and upload to S3."""

    del session_id  # Reserved for future path partitioning without exposing it today.
    prefix = _normalize_s3_prefix(s3_prefix)
    job_prefix = f"{prefix}/jobs/{job_name}"
    train_key = f"{job_prefix}/input/train.jsonl"

    try:
        dataset = await asyncio.to_thread(
            load_dataset,
            dataset_name,
            dataset_config,
            split=dataset_split,
            token=hf_token,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load dataset. If this is a private uploaded dataset, an HF token is required."
        ) from exc

    payload, row_count = _rows_to_jsonl_bytes(dataset)

    if s3_client is None:
        try:
            import boto3
        except Exception as exc:
            raise RuntimeError(
                f"boto3 is required to upload staged datasets to S3: {exc}"
            ) from exc
        s3_client = boto3.client("s3")

    try:
        await asyncio.to_thread(
            s3_client.put_object,
            Bucket=s3_bucket,
            Key=train_key,
            Body=payload,
            ContentType="application/jsonl; charset=utf-8",
        )
    except Exception as exc:
        raise RuntimeError(f"Could not upload staged dataset to S3: {exc}") from exc

    s3_prefix_uri = _s3_uri(s3_bucket, f"{job_prefix}/")
    return AwsDatasetStagingResult(
        s3_train_uri=_s3_uri(s3_bucket, train_key),
        s3_prefix_uri=s3_prefix_uri,
        s3_output_uri=_s3_uri(s3_bucket, f"{job_prefix}/output/"),
        s3_checkpoint_uri=_s3_uri(s3_bucket, f"{job_prefix}/checkpoints/"),
        row_count=row_count,
        bytes_uploaded=len(payload),
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
    )
