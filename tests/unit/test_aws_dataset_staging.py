import json

import pytest

from agent.core.aws_dataset_staging import stage_hf_dataset_to_s3


class FakeS3Client:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.puts = []

    def put_object(self, **kwargs):
        if self.fail:
            raise RuntimeError("s3 denied")
        self.puts.append(kwargs)
        return {"ETag": '"fake"'}


@pytest.mark.asyncio
async def test_stage_hf_dataset_serializes_rows_and_uploads_expected_s3_key(
    monkeypatch,
):
    calls = []

    def fake_load_dataset(dataset_name, dataset_config=None, *, split, token=None):
        calls.append(
            {
                "dataset_name": dataset_name,
                "dataset_config": dataset_config,
                "split": split,
                "token": token,
            }
        )
        return [
            {"text": "hello", "nested": {"answer": 1}},
            {"messages": [{"role": "user", "content": "hi"}]},
        ]

    monkeypatch.setattr(
        "agent.core.aws_dataset_staging.load_dataset", fake_load_dataset
    )
    s3 = FakeS3Client()

    result = await stage_hf_dataset_to_s3(
        dataset_name="owner/uploaded-dataset",
        dataset_config="upload_abc123",
        dataset_split="train",
        s3_bucket="training-bucket",
        s3_prefix="/team/prefix/",
        job_name="aws-job-1",
        session_id="session-1",
        hf_token="hf_private_token",
        s3_client=s3,
    )

    assert calls == [
        {
            "dataset_name": "owner/uploaded-dataset",
            "dataset_config": "upload_abc123",
            "split": "train",
            "token": "hf_private_token",
        }
    ]
    assert result.s3_train_uri == (
        "s3://training-bucket/team/prefix/jobs/aws-job-1/input/train.jsonl"
    )
    assert result.s3_prefix_uri == "s3://training-bucket/team/prefix/jobs/aws-job-1/"
    assert result.s3_output_uri == (
        "s3://training-bucket/team/prefix/jobs/aws-job-1/output/"
    )
    assert result.s3_checkpoint_uri == (
        "s3://training-bucket/team/prefix/jobs/aws-job-1/checkpoints/"
    )
    assert result.row_count == 2
    assert result.dataset_name == "owner/uploaded-dataset"
    assert result.dataset_config == "upload_abc123"
    assert result.dataset_split == "train"

    assert len(s3.puts) == 1
    put = s3.puts[0]
    assert put["Bucket"] == "training-bucket"
    assert put["Key"] == "team/prefix/jobs/aws-job-1/input/train.jsonl"
    assert put["ContentType"] == "application/jsonl; charset=utf-8"
    body = put["Body"]
    assert isinstance(body, bytes)
    assert result.bytes_uploaded == len(body)
    decoded_lines = body.decode("utf-8").splitlines()
    assert [json.loads(line) for line in decoded_lines] == [
        {"nested": {"answer": 1}, "text": "hello"},
        {"messages": [{"content": "hi", "role": "user"}]},
    ]

    public_text = str(result) + repr(s3.puts)
    assert "hf_private_token" not in public_text


@pytest.mark.asyncio
async def test_stage_hf_dataset_defaults_empty_prefix_to_liga_ml(monkeypatch):
    monkeypatch.setattr(
        "agent.core.aws_dataset_staging.load_dataset",
        lambda *_args, **_kwargs: [{"text": "one"}],
    )
    s3 = FakeS3Client()

    result = await stage_hf_dataset_to_s3(
        dataset_name="owner/dataset",
        dataset_config=None,
        dataset_split="validation",
        s3_bucket="bucket",
        s3_prefix="",
        job_name="job",
        s3_client=s3,
    )

    assert s3.puts[0]["Key"] == "liga-ml/jobs/job/input/train.jsonl"
    assert result.s3_train_uri == "s3://bucket/liga-ml/jobs/job/input/train.jsonl"
    assert result.dataset_config is None
    assert result.dataset_split == "validation"


@pytest.mark.asyncio
async def test_stage_hf_dataset_rejects_empty_dataset(monkeypatch):
    monkeypatch.setattr(
        "agent.core.aws_dataset_staging.load_dataset",
        lambda *_args, **_kwargs: [],
    )

    with pytest.raises(ValueError, match="Loaded dataset split contains no rows"):
        await stage_hf_dataset_to_s3(
            dataset_name="owner/dataset",
            dataset_config="default",
            dataset_split="train",
            s3_bucket="bucket",
            s3_prefix="prefix",
            job_name="job",
            s3_client=FakeS3Client(),
        )


@pytest.mark.asyncio
async def test_stage_hf_dataset_reports_s3_upload_failure(monkeypatch):
    monkeypatch.setattr(
        "agent.core.aws_dataset_staging.load_dataset",
        lambda *_args, **_kwargs: [{"text": "one"}],
    )

    with pytest.raises(RuntimeError, match="Could not upload staged dataset to S3"):
        await stage_hf_dataset_to_s3(
            dataset_name="owner/dataset",
            dataset_config="default",
            dataset_split="train",
            s3_bucket="bucket",
            s3_prefix="prefix",
            job_name="job",
            s3_client=FakeS3Client(fail=True),
        )


@pytest.mark.asyncio
async def test_stage_hf_dataset_reports_load_failure(monkeypatch):
    def fake_load_dataset(*_args, **_kwargs):
        raise RuntimeError("private repo")

    monkeypatch.setattr(
        "agent.core.aws_dataset_staging.load_dataset", fake_load_dataset
    )

    with pytest.raises(RuntimeError, match="Could not load dataset"):
        await stage_hf_dataset_to_s3(
            dataset_name="owner/private-dataset",
            dataset_config="default",
            dataset_split="train",
            s3_bucket="bucket",
            s3_prefix="prefix",
            job_name="job",
            s3_client=FakeS3Client(),
        )
