from types import SimpleNamespace

import pytest

from agent.tools.aws_sagemaker_jobs_tool import (
    AwsSageMakerJobsTool,
    aws_sagemaker_jobs_handler,
)


def _ready_snapshot(**overrides):
    return {
        "configured": True,
        "missing_env": [],
        "region": "us-east-1",
        "s3_bucket": "training-bucket",
        "s3_prefix": "liga-ml",
        "sagemaker_role_arn": "arn:aws:iam::123456789012:role/TestRole",
        "training_image_uri": None,
        "default_instance_type": "ml.g5.xlarge",
        "default_instance_count": 1,
        "default_max_run_seconds": 3600,
        "output_policy": "aws-private",
        "credentials_detected": True,
        "warnings": [],
        "errors": [],
        **overrides,
    }


def _staged(**overrides):
    return SimpleNamespace(
        s3_train_uri="s3://training-bucket/liga-ml/jobs/custom-job-name/input/train.jsonl",
        s3_prefix_uri="s3://training-bucket/liga-ml/jobs/custom-job-name/",
        s3_output_uri="s3://training-bucket/liga-ml/jobs/custom-job-name/output/",
        s3_checkpoint_uri="s3://training-bucket/liga-ml/jobs/custom-job-name/checkpoints/",
        row_count=3,
        bytes_uploaded=123,
        dataset_name="owner/dataset",
        dataset_config="default",
        dataset_split="train",
        **overrides,
    )


def _run_args(**overrides):
    return {
        "operation": "run",
        "template": "sft",
        "dataset_name": "owner/dataset",
        "dataset_config": "default",
        "dataset_split": "train",
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "output_model_id": "owner/aws-output",
        "instance_type": "ml.g5.xlarge",
        "instance_count": 1,
        "max_run_seconds": 3600,
        "job_name": "custom-job-name",
        **overrides,
    }


class FakeS3Client:
    def __init__(self):
        self.puts = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        return {"ETag": '"fake"'}


class FakeSageMakerClient:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []
        self.stopped_jobs = []

    def create_training_job(self, **kwargs):
        if self.fail:
            raise RuntimeError("sagemaker denied")
        self.calls.append(kwargs)
        return {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/custom-job-name"
        }

    def list_training_jobs(self, **kwargs):
        self.calls.append({"list_training_jobs": kwargs})
        return {
            "TrainingJobSummaries": [
                {
                    "TrainingJobName": "job-completed",
                    "TrainingJobStatus": "Completed",
                    "CreationTime": "2026-05-30T09:00:00+00:00",
                },
                {
                    "TrainingJobName": "job-running",
                    "TrainingJobStatus": "InProgress",
                    "CreationTime": "2026-05-30T09:10:00+00:00",
                },
            ]
        }

    def describe_training_job(self, **kwargs):
        self.calls.append({"describe_training_job": kwargs})
        return {
            "TrainingJobName": kwargs["TrainingJobName"],
            "TrainingJobStatus": "Completed",
            "SecondaryStatus": "Completed",
            "TrainingStartTime": "2026-05-30T09:00:00+00:00",
            "TrainingEndTime": "2026-05-30T09:30:00+00:00",
            "ResourceConfig": {
                "InstanceType": "ml.g5.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            },
            "OutputDataConfig": {
                "S3OutputPath": "s3://training-bucket/liga-ml/jobs/job-completed/output/"
            },
            "ModelArtifacts": {
                "S3ModelArtifacts": "s3://training-bucket/liga-ml/jobs/job-completed/output/model.tar.gz"
            },
        }

    def stop_training_job(self, **kwargs):
        self.stopped_jobs.append(kwargs)
        return {}


class FakeLogsClient:
    def __init__(self, *, streams=None, events=None):
        self.streams = (
            streams
            if streams is not None
            else [{"logStreamName": "job-running/algo-1-123"}]
        )
        self.events = (
            events
            if events is not None
            else [
                {"timestamp": 1, "message": "starting training"},
                {"timestamp": 2, "message": "LIGA_TRAINING_STATUS=succeeded"},
            ]
        )
        self.calls = []

    def describe_log_streams(self, **kwargs):
        self.calls.append({"describe_log_streams": kwargs})
        return {"logStreams": self.streams}

    def get_log_events(self, **kwargs):
        self.calls.append({"get_log_events": kwargs})
        return {"events": self.events}


class FakeSession:
    hf_token = "hf-session-token"
    session_id = "session-1"

    def __init__(self):
        self.events = []

    async def send_event(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_run_missing_aws_config_is_actionable(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(
            configured=False,
            missing_env=["AWS_S3_BUCKET", "AWS_SAGEMAKER_ROLE_ARN"],
            s3_bucket=None,
            sagemaker_role_arn=None,
            credentials_detected=False,
            errors=["Missing required AWS environment variables."],
        ),
    )

    result = await AwsSageMakerJobsTool().execute(
        {"operation": "run", "template": "sft"}
    )

    assert result["isError"] is True
    assert "AWS_S3_BUCKET" in result["formatted"]
    assert "AWS_SAGEMAKER_ROLE_ARN" in result["formatted"]
    assert "AWS_REGION" in result["formatted"]
    assert "/api/health/providers" in result["formatted"]
    assert "AWS credentials" in result["formatted"]
    assert "secret" not in result["formatted"].lower()


@pytest.mark.asyncio
async def test_run_requires_image_uri_before_staging_or_submission(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(training_image_uri=None),
    )
    staged_called = False

    async def fake_stage(**_kwargs):
        nonlocal staged_called
        staged_called = True
        return _staged()

    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.stage_hf_dataset_to_s3", fake_stage
    )
    sagemaker = FakeSageMakerClient()

    result = await AwsSageMakerJobsTool(sagemaker_client=sagemaker).execute(_run_args())

    assert result["isError"] is True
    assert "image_uri" in result["formatted"]
    assert "AWS_SAGEMAKER_TRAINING_IMAGE_URI" in result["formatted"]
    assert staged_called is False
    assert sagemaker.calls == []


@pytest.mark.asyncio
async def test_run_with_image_submits_training_job_and_uploads_script(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(
            training_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/liga-train:latest"
        ),
    )
    staged_calls = []

    async def fake_stage(**kwargs):
        staged_calls.append(kwargs)
        return _staged()

    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.stage_hf_dataset_to_s3", fake_stage
    )
    session = FakeSession()
    s3 = FakeS3Client()
    sagemaker = FakeSageMakerClient()
    tool = AwsSageMakerJobsTool(
        session=session,
        tool_call_id="call-1",
        s3_client=s3,
        sagemaker_client=sagemaker,
    )

    result = await tool.execute(_run_args(output_policy="cloud-and-hf-hub"))

    assert not result.get("isError")
    assert "AWS SageMaker training job submitted" in result["formatted"]
    assert "custom-job-name" in result["formatted"]
    assert (
        "s3://training-bucket/liga-ml/jobs/custom-job-name/input/train.jsonl"
        in result["formatted"]
    )
    assert (
        "s3://training-bucket/liga-ml/jobs/custom-job-name/output/"
        in result["formatted"]
    )
    assert "SageMaker console" in result["formatted"]
    assert "CloudWatch logs" in result["formatted"]
    assert "Conservative cost estimate" in result["formatted"]
    assert "arn:aws:iam::123456789012:role/TestRole" not in result["formatted"]

    assert staged_calls[0]["dataset_name"] == "owner/dataset"
    assert staged_calls[0]["s3_bucket"] == "training-bucket"
    assert staged_calls[0]["hf_token"] == "hf-session-token"

    script_puts = [put for put in s3.puts if put["Key"].endswith("/code/train.py")]
    assert len(script_puts) == 1
    assert script_puts[0]["Bucket"] == "training-bucket"
    assert "LIGA_PROVIDER=aws-sagemaker" in script_puts[0]["Body"].decode("utf-8")

    assert len(sagemaker.calls) == 1
    request = sagemaker.calls[0]
    for field in [
        "TrainingJobName",
        "RoleArn",
        "InputDataConfig",
        "OutputDataConfig",
        "ResourceConfig",
        "StoppingCondition",
        "AlgorithmSpecification",
        "Environment",
    ]:
        assert field in request
    assert request["TrainingJobName"] == "custom-job-name"
    assert request["RoleArn"] == "arn:aws:iam::123456789012:role/TestRole"
    assert request["AlgorithmSpecification"]["TrainingImage"].endswith(
        "liga-train:latest"
    )
    assert request["InputDataConfig"][0]["ChannelName"] == "train"
    assert request["InputDataConfig"][0]["DataSource"]["S3DataSource"][
        "S3Uri"
    ].endswith("/input/")
    assert request["OutputDataConfig"]["S3OutputPath"].endswith("/output/")
    assert request["ResourceConfig"] == {
        "InstanceType": "ml.g5.xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 30,
    }
    assert request["StoppingCondition"]["MaxRuntimeInSeconds"] == 3600
    assert request["Environment"]["LIGA_OUTPUT_POLICY"] == "cloud-and-hf-hub"
    assert request["Environment"]["LIGA_S3_MODEL_ARTIFACT"].endswith(
        "/output/model.tar.gz"
    )
    assert "HF_TOKEN" not in request["Environment"]
    assert request["HyperParameters"]["sagemaker_program"] == "train.py"

    event = session.events[-1]
    assert event.event_type == "tool_state_change"
    assert event.data["tool"] == "aws_sagemaker_jobs"
    assert event.data["state"] == "running"
    assert event.data["jobName"] == "custom-job-name"
    assert event.data["jobUrl"].startswith(
        "https://us-east-1.console.aws.amazon.com/sagemaker/"
    )
    assert event.data["s3TrainUri"].endswith("/input/train.jsonl")
    assert event.data["s3ModelArtifact"].endswith("/output/model.tar.gz")


@pytest.mark.asyncio
async def test_run_uses_explicit_image_uri_over_readiness_default(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(training_image_uri="readiness-image"),
    )

    async def fake_stage(**_kwargs):
        return _staged()

    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.stage_hf_dataset_to_s3", fake_stage
    )
    sagemaker = FakeSageMakerClient()

    result = await AwsSageMakerJobsTool(
        s3_client=FakeS3Client(),
        sagemaker_client=sagemaker,
    ).execute(_run_args(image_uri="explicit-image"))

    assert not result.get("isError")
    assert (
        sagemaker.calls[0]["AlgorithmSpecification"]["TrainingImage"]
        == "explicit-image"
    )


@pytest.mark.asyncio
async def test_run_dataset_staging_errors_stop_job_submission(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(training_image_uri="image"),
    )

    async def fake_stage(**_kwargs):
        raise RuntimeError("Could not upload staged dataset to S3: access denied")

    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.stage_hf_dataset_to_s3", fake_stage
    )
    sagemaker = FakeSageMakerClient()

    result = await AwsSageMakerJobsTool(sagemaker_client=sagemaker).execute(_run_args())

    assert result["isError"] is True
    assert "Could not upload staged dataset to S3" in result["formatted"]
    assert sagemaker.calls == []


@pytest.mark.asyncio
async def test_run_sagemaker_client_errors_are_actionable(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(training_image_uri="image"),
    )

    async def fake_stage(**_kwargs):
        return _staged()

    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.stage_hf_dataset_to_s3", fake_stage
    )

    result = await AwsSageMakerJobsTool(
        s3_client=FakeS3Client(),
        sagemaker_client=FakeSageMakerClient(fail=True),
    ).execute(_run_args())

    assert result["isError"] is True
    assert "Could not submit SageMaker training job" in result["formatted"]
    assert "sagemaker denied" in result["formatted"]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["inspect", "logs", "cancel"])
async def test_job_name_operations_require_job_name(operation, monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(),
    )

    result = await AwsSageMakerJobsTool().execute({"operation": operation})

    assert result["isError"] is True
    assert "job_name is required" in result["formatted"]


@pytest.mark.asyncio
async def test_ps_uses_mocked_list_training_jobs_and_formats_results(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(),
    )
    sagemaker = FakeSageMakerClient()

    result = await AwsSageMakerJobsTool(sagemaker_client=sagemaker).execute(
        {"operation": "ps"}
    )

    assert not result.get("isError")
    assert "job-completed" in result["formatted"]
    assert "Completed" in result["formatted"]
    assert "job-running" in result["formatted"]
    assert "InProgress" in result["formatted"]
    assert sagemaker.calls[0]["list_training_jobs"]["SortBy"] == "CreationTime"


@pytest.mark.asyncio
async def test_ps_missing_config_returns_safe_readiness_message(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(
            configured=False,
            missing_env=["AWS_S3_BUCKET"],
            credentials_detected=False,
            errors=["Missing required AWS environment variables."],
        ),
    )

    result = await AwsSageMakerJobsTool(sagemaker_client=FakeSageMakerClient()).execute(
        {"operation": "ps"}
    )

    assert result["isError"] is True
    assert "AWS_S3_BUCKET" in result["formatted"]
    assert "AWS credentials" in result["formatted"]


@pytest.mark.asyncio
async def test_inspect_uses_mocked_describe_training_job_and_formats_status(
    monkeypatch,
):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(),
    )
    session = FakeSession()
    sagemaker = FakeSageMakerClient()

    result = await AwsSageMakerJobsTool(
        session=session,
        tool_call_id="call-inspect",
        sagemaker_client=sagemaker,
    ).execute({"operation": "inspect", "job_name": "job-completed"})

    assert not result.get("isError")
    assert "TrainingJobName" in result["formatted"]
    assert "job-completed" in result["formatted"]
    assert "Completed" in result["formatted"]
    assert "S3OutputPath" in result["formatted"]
    assert "S3ModelArtifacts" in result["formatted"]
    assert "SageMaker console" in result["formatted"]
    assert "CloudWatch logs" in result["formatted"]
    assert (
        sagemaker.calls[0]["describe_training_job"]["TrainingJobName"]
        == "job-completed"
    )
    event = session.events[-1]
    assert event.data["state"] == "succeeded"
    assert event.data["s3OutputUri"].endswith("/output/")
    assert event.data["s3ModelArtifact"].endswith("/model.tar.gz")


@pytest.mark.asyncio
async def test_logs_uses_mocked_cloudwatch_client_and_formats_recent_logs(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(),
    )
    logs = FakeLogsClient()

    result = await AwsSageMakerJobsTool(logs_client=logs).execute(
        {"operation": "logs", "job_name": "job-running"}
    )

    assert not result.get("isError")
    assert "CloudWatch logs for `job-running`" in result["formatted"]
    assert "starting training" in result["formatted"]
    assert "LIGA_TRAINING_STATUS=succeeded" in result["formatted"]
    assert (
        logs.calls[0]["describe_log_streams"]["logGroupName"]
        == "/aws/sagemaker/TrainingJobs"
    )


@pytest.mark.asyncio
async def test_logs_handles_no_streams_or_events(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(),
    )

    no_streams = await AwsSageMakerJobsTool(
        logs_client=FakeLogsClient(streams=[])
    ).execute({"operation": "logs", "job_name": "job-running"})
    no_events = await AwsSageMakerJobsTool(
        logs_client=FakeLogsClient(events=[])
    ).execute({"operation": "logs", "job_name": "job-running"})

    assert "No CloudWatch log streams found yet" in no_streams["formatted"]
    assert "No CloudWatch log events found yet" in no_events["formatted"]


@pytest.mark.asyncio
async def test_cancel_calls_mocked_stop_training_job(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(),
    )
    sagemaker = FakeSageMakerClient()

    result = await AwsSageMakerJobsTool(sagemaker_client=sagemaker).execute(
        {"operation": "cancel", "job_name": "job-running"}
    )

    assert not result.get("isError")
    assert sagemaker.stopped_jobs == [{"TrainingJobName": "job-running"}]
    assert "Cancellation requested" in result["formatted"]
    assert "SageMaker console" in result["formatted"]


def test_sagemaker_status_mapping():
    from agent.tools.aws_sagemaker_jobs_tool import map_sagemaker_status

    assert map_sagemaker_status("Completed") == "succeeded"
    assert map_sagemaker_status("Failed") == "failed"
    assert map_sagemaker_status("Stopped") == "stopped"
    assert map_sagemaker_status("Stopping") == "stopping"
    assert map_sagemaker_status("InProgress") == "running"


def test_registered_tool_is_available():
    from agent.core.tools import create_builtin_tools

    tool_names = {tool.name for tool in create_builtin_tools(local_mode=True)}

    assert "aws_sagemaker_jobs" in tool_names


@pytest.mark.asyncio
async def test_handler_submits_job_with_env_image(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(training_image_uri="image"),
    )

    async def fake_stage(**_kwargs):
        return _staged()

    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.stage_hf_dataset_to_s3",
        fake_stage,
    )
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool._load_sagemaker_client",
        lambda _region=None: FakeSageMakerClient(),
    )
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool._load_s3_client",
        lambda _region=None: FakeS3Client(),
    )

    output, ok = await aws_sagemaker_jobs_handler(_run_args())

    assert ok is True
    assert "AWS SageMaker training job submitted" in output
    assert "custom-job-name" in output
