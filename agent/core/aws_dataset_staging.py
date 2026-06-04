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


def _rows_to_jsonl_bytes(dataset: Any) -> tuple[bytes, int]:
    lines: list[str] = []
    for row_index, raw_row in enumerate(dataset, start=1):
        row = dict(raw_row) if not isinstance(raw_row, dict) else raw_row
        if not isinstance(row, dict):
            raise ValueError(
                f"Dataset row {row_index} must serialize as a JSON object."
            )
        safe_row = _reject_binary(row, row_index=row_index)
        try:
            lines.append(json.dumps(safe_row, ensure_ascii=False, sort_keys=True))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Dataset row {row_index} could not be serialized as JSON: {exc}"
            ) from exc
    if not lines:
        raise ValueError("Loaded dataset split contains no rows.")
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
