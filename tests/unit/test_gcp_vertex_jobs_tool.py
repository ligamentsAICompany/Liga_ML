import base64
import json
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from agent.tools.gcp_vertex_jobs_tool import GcpVertexJobsTool, gcp_vertex_jobs_handler


@pytest.fixture(autouse=True)
def _mock_hub_dataset_staging_for_vertex_tests(monkeypatch):
    from agent.core.gcp_dataset_staging import GcpDatasetStagingResult

    async def fake_stage_hf_dataset_to_gcs(**kwargs):
        display_name = str(kwargs.get("display_name") or "vertex-job")
        bucket = str(kwargs.get("gcs_bucket") or "liga-training")
        train_uri = f"gs://{bucket}/vertex-inputs/{display_name}/train.jsonl"
        return GcpDatasetStagingResult(
            train_gcs_uri=train_uri,
            gcs_prefix_uri=f"gs://{bucket}/vertex-inputs/{display_name}/",
            row_count=int(kwargs.get("max_rows") or 3),
            bytes_uploaded=256,
            dataset_name=str(kwargs.get("dataset_name") or "test/dataset"),
            dataset_config=kwargs.get("dataset_config"),
            dataset_split=str(kwargs.get("dataset_split") or "train"),
            source_format="messages",
            detected_schema="messages",
        )

    monkeypatch.setattr(
        "agent.tools.gcp_vertex_jobs_tool.stage_hf_dataset_to_gcs",
        fake_stage_hf_dataset_to_gcs,
    )


class FakeCustomJob:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.resource_name = (
            "projects/test-project/locations/us-central1/customJobs/123"
        )
        self.name = kwargs["display_name"]
        FakeCustomJob.instances.append(self)

    def run(self, **kwargs):
        self.run_kwargs = kwargs


class FakeSession:
    hf_token = "hf-session-token"
    sandbox = None

    def __init__(self):
        self.events = []
        self.uploaded_datasets = []

    async def send_event(self, event):
        self.events.append(event)


class FakeState:
    def __init__(self, name):
        self.name = name


class FakeJobServiceClient:
    get_calls = 0
    list_calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_custom_job(self, name):
        FakeJobServiceClient.get_calls += 1
        return SimpleNamespace(
            name=name,
            display_name="finance-sft",
            state=FakeState("JOB_STATE_RUNNING"),
            create_time="created",
            update_time="updated",
        )

    def list_custom_jobs(self, **kwargs):
        FakeJobServiceClient.list_calls.append(kwargs)
        return []


class FakeLoggingClient:
    list_calls = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def list_entries(self, **kwargs):
        FakeLoggingClient.list_calls += 1
        return []


class FakeFailedJobServiceClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_custom_job(self, name):
        return SimpleNamespace(
            name=name,
            display_name="medical-uploaded-sft",
            state=FakeState("JOB_STATE_FAILED"),
            create_time="created",
            update_time="updated",
            error=SimpleNamespace(
                message="WorkerPool replica exited with exit code 1."
            ),
        )


class FakeFailureLoggingClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def list_entries(self, **kwargs):
        return [
            SimpleNamespace(payload="INFO preparing training"),
            SimpleNamespace(
                payload=(
                    "datasets.exceptions.DatasetNotFoundError: Dataset "
                    "'ligaments-dev/ml-intern-567e31d0-datasets' doesn't exist "
                    "on the Hub or cannot be accessed."
                )
            ),
        ]


class FakeBucket:
    uploads = []

    def blob(self, name):
        return SimpleNamespace(
            upload_from_filename=lambda filename: FakeBucket.uploads.append(
                (name, filename)
            )
        )


class FakeStorageClient:
    def bucket(self, bucket_name):
        self.bucket_name = bucket_name
        return FakeBucket()


@pytest.mark.asyncio
async def test_run_command_submits_vertex_custom_job(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    monkeypatch.setenv("VERTEX_AI_OUTPUT_DIR", "gs://liga-training/outputs")
    FakeCustomJob.instances = []

    session = FakeSession()
    tool = GcpVertexJobsTool(
        session=session,
        tool_call_id="call-1",
        custom_job_cls=FakeCustomJob,
    )

    result = await tool.execute(
        {
            "operation": "run",
            "command": ["python", "train.py"],
            "image": "python:3.12",
            "display_name": "gst-train",
            "machine_type": "n1-standard-8",
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": 1,
            "max_run_hours": 1,
            "training_goal": "smoke-test",
            "env": {"DATASET_ID": "ligaments/gst"},
        }
    )

    assert not result.get("isError")
    assert "Vertex AI job submitted" in result["formatted"]
    job = FakeCustomJob.instances[0]
    worker_pool = job.kwargs["worker_pool_specs"][0]
    assert worker_pool["machine_spec"] == {
        "machine_type": "n1-standard-8",
        "accelerator_type": "NVIDIA_TESLA_T4",
        "accelerator_count": 1,
    }
    assert worker_pool["container_spec"]["image_uri"] == "python:3.12"
    assert worker_pool["container_spec"]["command"] == ["python", "train.py"]
    env = {item["name"]: item["value"] for item in worker_pool["container_spec"]["env"]}
    assert env["DATASET_ID"] == "ligaments/gst"
    assert env["AIP_MODEL_DIR"] == "gs://liga-training/outputs/gst-train"
    assert "HF_TOKEN" not in env
    assert job.run_kwargs["sync"] is False
    assert session.events[0].data["tool"] == "gcp_vertex_jobs"
    assert session.events[0].data["state"] == "queued"
    assert session.events[0].data["provider"] == "gcp-vertex"
    assert session.events[0].data["machine_type"] == "n1-standard-8"
    assert session.events[0].data["accelerator_type"] == "NVIDIA_TESLA_T4"
    assert session.events[0].data["accelerator_count"] == 1
    assert session.events[0].data["max_run_hours"] == 1
    assert session.events[0].data["training_goal"] == "smoke-test"
    assert session.events[0].data["outputDir"] == "gs://liga-training/outputs/gst-train"


@pytest.mark.asyncio
async def test_run_uses_modern_pytorch_vertex_image_by_default(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    monkeypatch.delenv("GCP_VERTEX_DEFAULT_IMAGE", raising=False)
    FakeCustomJob.instances = []

    tool = GcpVertexJobsTool(custom_job_cls=FakeCustomJob)

    result = await tool.execute(
        {
            "operation": "run",
            "command": ["python", "train.py"],
            "display_name": "medical-sft",
            "accelerator_type": "NVIDIA_TESLA_T4",
        }
    )

    assert not result.get("isError")
    worker_pool = FakeCustomJob.instances[0].kwargs["worker_pool_specs"][0]
    assert (
        worker_pool["container_spec"]["image_uri"]
        == "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310"
    )


@pytest.mark.asyncio
async def test_run_sft_template_generates_vertex_training_script(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    FakeCustomJob.instances = []

    session = FakeSession()
    tool = GcpVertexJobsTool(session=session, custom_job_cls=FakeCustomJob)

    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "display_name": "medical-sft",
            "dataset_name": "FreedomIntelligence/medical-o1-reasoning-SFT",
            "dataset_config": "en",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "hub_model_id": "ligaments-dev/medical-qwen2.5-0.5b-sft",
            "output_policy": "cloud-and-hf-hub",
            "column_mapping": {
                "user": "Question",
                "assistant": ["Complex_CoT", "Response"],
            },
            "trackio_project": "medical-sft",
            "trackio_space_id": "ligaments-dev/ml-intern-trackio",
        }
    )

    assert not result.get("isError")
    assert (
        "**HF model target:** https://huggingface.co/ligaments-dev/medical-qwen2.5-0.5b-sft"
        in result["formatted"]
    )
    worker_pool = FakeCustomJob.instances[0].kwargs["worker_pool_specs"][0]
    encoded_runner = worker_pool["container_spec"]["command"][-1]
    encoded_script = re.search(r"b64decode\('([^']+)'\)", encoded_runner).group(1)
    decoded_script = base64.b64decode(encoded_script).decode("utf-8")
    assert "FreedomIntelligence/medical-o1-reasoning-SFT" in decoded_script
    assert "packing=False" in decoded_script
    env = {item["name"]: item["value"] for item in worker_pool["container_spec"]["env"]}
    assert env["TRACKIO_MODE"] == "disabled"
    assert env["TRACKIO_PROJECT"] == "medical-sft"
    assert env["TRACKIO_SPACE_ID"] == "ligaments-dev/ml-intern-trackio"
    assert env["HF_TOKEN"] == "hf-session-token"
    assert env["HUGGINGFACE_HUB_TOKEN"] == "hf-session-token"
    assert "hf-session-token" not in result["formatted"]


@pytest.mark.asyncio
async def test_run_sft_template_cloud_private_does_not_require_or_inject_hf_token(
    monkeypatch,
):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    FakeCustomJob.instances = []

    session = FakeSession()
    session.hf_token = None
    tool = GcpVertexJobsTool(session=session, custom_job_cls=FakeCustomJob)

    result = await tool.execute(
        {
            "operation": "run",
            "command": ["python", "train.py"],
            "display_name": "private-run",
            "output_policy": "cloud-private",
        }
    )

    assert not result.get("isError")
    worker_pool = FakeCustomJob.instances[0].kwargs["worker_pool_specs"][0]
    env = {item["name"]: item["value"] for item in worker_pool["container_spec"]["env"]}
    assert "HF_TOKEN" not in env
    assert "HUGGINGFACE_HUB_TOKEN" not in env


@pytest.mark.asyncio
async def test_run_sft_template_preserves_secret_resource_for_cloud_private(
    monkeypatch,
):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    monkeypatch.setenv(
        "HF_TOKEN_SECRET_RESOURCE",
        "projects/test-project/secrets/hf-token/versions/latest",
    )
    FakeCustomJob.instances = []

    session = FakeSession()
    session.hf_token = None
    tool = GcpVertexJobsTool(session=session, custom_job_cls=FakeCustomJob)

    result = await tool.execute(
        {
            "operation": "run",
            "command": ["python", "train.py"],
            "display_name": "private-run-with-secret",
            "output_policy": "cloud-private",
        }
    )

    assert not result.get("isError")
    worker_pool = FakeCustomJob.instances[0].kwargs["worker_pool_specs"][0]
    env = {item["name"]: item["value"] for item in worker_pool["container_spec"]["env"]}
    assert env["HF_TOKEN_SECRET_RESOURCE"] == (
        "projects/test-project/secrets/hf-token/versions/latest"
    )
    assert "HF_TOKEN" not in env
    assert "HUGGINGFACE_HUB_TOKEN" not in env


@pytest.mark.asyncio
async def test_run_sft_template_hf_hub_policy_requires_hf_token_before_submit(
    monkeypatch,
):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN_SECRET_RESOURCE", raising=False)
    monkeypatch.setattr(
        "agent.tools.gcp_vertex_jobs_tool._resolve_vertex_hf_token",
        lambda session=None: None,
    )
    FakeCustomJob.instances = []

    session = FakeSession()
    session.hf_token = None
    tool = GcpVertexJobsTool(session=session, custom_job_cls=FakeCustomJob)

    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "dataset_name": "trl-lib/Capybara",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "hub_model_id": "ligaments-dev/needs-token",
            "output_policy": "cloud-and-hf-hub",
        }
    )

    assert result["isError"] is True
    assert "Hugging Face token" in result["formatted"]
    assert "hf_" not in result["formatted"]
    assert FakeCustomJob.instances == []


@pytest.mark.asyncio
async def test_run_sft_template_stages_session_uploaded_dataset_to_gcs(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    FakeCustomJob.instances = []
    FakeBucket.uploads = []

    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"text": "first training row", "data": {"text": "first"}}),
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "hello"},
                            {"role": "assistant", "content": "hi"},
                        ]
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_hf_hub_download(**kwargs):
        assert kwargs["repo_id"] == "ligaments-dev/ml-intern-567e31d0-datasets"
        assert kwargs["filename"] == "uploads/bb5048fac571/train.jsonl"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["token"] == "hf-session-token"
        return str(train_jsonl)

    monkeypatch.setattr(
        "agent.tools.gcp_vertex_jobs_tool.hf_hub_download",
        fake_hf_hub_download,
        raising=False,
    )
    monkeypatch.setattr(
        "agent.tools.gcp_vertex_jobs_tool._load_storage_client_cls",
        lambda: FakeStorageClient,
        raising=False,
    )

    session = FakeSession()
    session.uploaded_datasets = [
        {
            "repo_id": "ligaments-dev/ml-intern-567e31d0-datasets",
            "config_name": "upload_bb5048fac571",
            "normalized_path_in_repo": "uploads/bb5048fac571/train.jsonl",
            "normalized_row_count": 2,
            "source_format": "md",
            "source": "session-upload",
            "supports_training": True,
        }
    ]
    tool = GcpVertexJobsTool(session=session, custom_job_cls=FakeCustomJob)

    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "display_name": "medical-uploaded-sft",
            "dataset_name": "ligaments-dev/ml-intern-567e31d0-datasets",
            "dataset_config": "upload_bb5048fac571",
            "dataset_split": "train",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_policy": "cloud-private",
            "max_train_samples": 2,
        }
    )

    assert not result.get("isError")
    assert "uploaded-gcs" in result["formatted"]
    assert (
        "gs://liga-training/vertex-inputs/medical-uploaded-sft/train.jsonl"
        in result["formatted"]
    )
    assert ("vertex-inputs/medical-uploaded-sft/train.jsonl", str(train_jsonl)) in (
        FakeBucket.uploads
    )
    assert any(name.endswith("metadata.json") for name, _ in FakeBucket.uploads)
    worker_pool = FakeCustomJob.instances[0].kwargs["worker_pool_specs"][0]
    encoded_runner = worker_pool["container_spec"]["command"][-1]
    encoded_script = re.search(r"b64decode\('([^']+)'\)", encoded_runner).group(1)
    decoded_script = base64.b64decode(encoded_script).decode("utf-8")
    assert '"dataset_source": "gcs_jsonl"' in decoded_script
    assert (
        '"train_gcs_uri": "gs://liga-training/vertex-inputs/medical-uploaded-sft/train.jsonl"'
        in decoded_script
    )
    assert 'load_dataset("json"' in decoded_script
    assert "load_dataset(**dataset_kwargs)" in decoded_script


@pytest.mark.asyncio
async def test_run_sft_template_rejects_empty_uploaded_jsonl_before_submit(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    FakeCustomJob.instances = []

    empty_jsonl = tmp_path / "train.jsonl"
    empty_jsonl.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "agent.tools.gcp_vertex_jobs_tool.hf_hub_download",
        lambda **_: str(empty_jsonl),
        raising=False,
    )
    session = FakeSession()
    session.uploaded_datasets = [
        {
            "repo_id": "owner/session-datasets",
            "config_name": "upload_abc",
            "normalized_path_in_repo": "uploads/abc/train.jsonl",
            "normalized_row_count": 0,
            "source_format": "csv",
            "source": "session-upload",
            "supports_training": True,
        }
    ]
    tool = GcpVertexJobsTool(session=session, custom_job_cls=FakeCustomJob)

    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "dataset_name": "owner/session-datasets",
            "dataset_config": "upload_abc",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_policy": "cloud-private",
        }
    )

    assert result["isError"] is True
    assert "staged JSONL" in result["formatted"]
    assert FakeCustomJob.instances == []


@pytest.mark.asyncio
async def test_run_sft_template_propagates_phase4_parameters(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    FakeCustomJob.instances = []

    tool = GcpVertexJobsTool(custom_job_cls=FakeCustomJob)

    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "display_name": "phase4-sft",
            "dataset_name": "trl-lib/Capybara",
            "dataset_split": "train",
            "eval_dataset_split": "validation",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "hub_model_id": "ligaments-dev/phase4-sft",
            "max_train_samples": 32,
            "max_eval_samples": 8,
            "validation_split_ratio": 0.25,
            "num_train_epochs": 2,
            "max_length": 768,
            "learning_rate": 0.0001,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "run_name": "phase4-run",
        }
    )

    assert not result.get("isError")
    worker_pool = FakeCustomJob.instances[0].kwargs["worker_pool_specs"][0]
    encoded_runner = worker_pool["container_spec"]["command"][-1]
    encoded_script = re.search(r"b64decode\('([^']+)'\)", encoded_runner).group(1)
    decoded_script = base64.b64decode(encoded_script).decode("utf-8")
    assert '"eval_dataset_split": "validation"' in decoded_script
    assert '"max_eval_samples": 8' in decoded_script
    assert '"validation_split_ratio": 0.25' in decoded_script
    assert '"max_length": 768' in decoded_script
    assert '"learning_rate": 0.0001' in decoded_script
    assert '"per_device_train_batch_size": 2' in decoded_script
    assert '"gradient_accumulation_steps": 4' in decoded_script
    assert '"run_name": "phase4-run"' in decoded_script
    assert "max_length=768" in decoded_script
    assert 'learning_rate=float(CONFIG["learning_rate"])' in decoded_script
    assert (
        'per_device_train_batch_size=int(CONFIG["per_device_train_batch_size"])'
        in decoded_script
    )
    assert (
        'gradient_accumulation_steps=int(CONFIG["gradient_accumulation_steps"])'
        in decoded_script
    )


@pytest.mark.asyncio
async def test_run_sft_template_accepts_training_goal_and_cloud_private_policy(
    monkeypatch,
):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    FakeCustomJob.instances = []

    tool = GcpVertexJobsTool(custom_job_cls=FakeCustomJob)

    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "display_name": "sensitive-medical-smoke",
            "dataset_name": "ligaments-dev/private-medical-upload",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "training_goal": "smoke-test",
            "output_policy": "cloud-private",
            "max_train_samples": 8,
        }
    )

    assert not result.get("isError")
    assert "HF model target" not in result["formatted"]
    assert "**Output policy:** cloud-private" in result["formatted"]
    worker_pool = FakeCustomJob.instances[0].kwargs["worker_pool_specs"][0]
    encoded_runner = worker_pool["container_spec"]["command"][-1]
    encoded_script = re.search(r"b64decode\('([^']+)'\)", encoded_runner).group(1)
    decoded_script = base64.b64decode(encoded_script).decode("utf-8")
    assert '"training_goal": "smoke-test"' in decoded_script
    assert '"output_policy": "cloud-private"' in decoded_script
    assert "push_to_hub=PUSH_TO_HUB" in decoded_script
    assert 'print("LIGA_FINAL_MODEL_URL=", flush=True)' in decoded_script


@pytest.mark.asyncio
async def test_run_sft_template_defaults_to_cloud_private_and_caps_large_dataset(
    monkeypatch,
):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    FakeCustomJob.instances = []

    tool = GcpVertexJobsTool(custom_job_cls=FakeCustomJob)

    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "display_name": "hardware-llama-pilot",
            "dataset_name": "ligaments-dev/hardware-upload",
            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
            "dataset_rows": 199_867,
        }
    )

    assert not result.get("isError")
    assert "**Output policy:** cloud-private" in result["formatted"]
    assert "max_train_samples=500" in result["formatted"]
    worker_pool = FakeCustomJob.instances[0].kwargs["worker_pool_specs"][0]
    encoded_runner = worker_pool["container_spec"]["command"][-1]
    encoded_script = re.search(r"b64decode\('([^']+)'\)", encoded_runner).group(1)
    decoded_script = base64.b64decode(encoded_script).decode("utf-8")
    assert '"output_policy": "cloud-private"' in decoded_script
    assert '"max_train_samples": 500' in decoded_script
    assert '"max_eval_samples": 50' in decoded_script


@pytest.mark.asyncio
async def test_run_sft_template_uses_session_token_for_hf_hub_policy(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    FakeCustomJob.instances = []

    session = FakeSession()
    tool = GcpVertexJobsTool(session=session, custom_job_cls=FakeCustomJob)

    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "dataset_name": "ligaments-dev/private-dataset",
            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
            "hub_model_id": "ligaments-dev/private-dataset-model",
            "output_policy": "hf-hub",
        }
    )

    assert not result.get("isError")
    worker_pool = FakeCustomJob.instances[0].kwargs["worker_pool_specs"][0]
    env = {item["name"]: item["value"] for item in worker_pool["container_spec"]["env"]}
    assert env["HF_TOKEN"] == "hf-session-token"
    assert env["HUGGINGFACE_HUB_TOKEN"] == "hf-session-token"
    assert "hf-session-token" not in result["formatted"]


@pytest.mark.asyncio
async def test_ps_omits_filter_arg_when_not_provided(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    FakeJobServiceClient.list_calls = []

    tool = GcpVertexJobsTool(job_service_client_cls=FakeJobServiceClient)

    result = await tool.execute({"operation": "ps"})

    assert not result.get("isError")
    assert "No Vertex AI custom jobs found." in result["formatted"]
    assert FakeJobServiceClient.list_calls == [
        {"parent": "projects/test-project/locations/us-central1"}
    ]


@pytest.mark.asyncio
async def test_run_sft_template_requires_core_parameters(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")

    tool = GcpVertexJobsTool(custom_job_cls=FakeCustomJob)

    result = await tool.execute({"operation": "run", "template": "sft"})

    assert result["isError"] is True
    assert "dataset_name is required" in result["formatted"]


@pytest.mark.asyncio
async def test_run_rejects_template_script_mix_and_unsupported_template(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")

    tool = GcpVertexJobsTool(custom_job_cls=FakeCustomJob)

    mixed = await tool.execute(
        {"operation": "run", "template": "sft", "script": "print('nope')"}
    )
    unsupported = await tool.execute({"operation": "run", "template": "dpo"})

    assert mixed["isError"] is True
    assert (
        "'template' cannot be combined with 'script' or 'command'."
        in mixed["formatted"]
    )
    assert unsupported["isError"] is True
    assert "Unsupported template: dpo" in unsupported["formatted"]


@pytest.mark.asyncio
async def test_run_sft_template_rejects_risky_options_before_submit(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    FakeCustomJob.instances = []

    tool = GcpVertexJobsTool(custom_job_cls=FakeCustomJob)

    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "dataset_name": "trl-lib/Capybara",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "hub_model_id": "ligaments-dev/test-model",
            "packing": True,
        }
    )

    assert result["isError"] is True
    assert "packing=True is not allowed" in result["formatted"]
    assert FakeCustomJob.instances == []


@pytest.mark.asyncio
async def test_vertex_monitoring_is_throttled_per_session(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    FakeJobServiceClient.get_calls = 0
    FakeLoggingClient.list_calls = 0

    session = FakeSession()
    tool = GcpVertexJobsTool(
        session=session,
        job_service_client_cls=FakeJobServiceClient,
        logging_client_cls=FakeLoggingClient,
    )
    job_name = "projects/test-project/locations/us-central1/customJobs/123"

    first = await tool.execute({"operation": "inspect", "job_name": job_name})
    second = await tool.execute({"operation": "logs", "job_name": job_name})

    assert "JOB_STATE_RUNNING" in first["formatted"]
    assert "Vertex job monitoring is rate-limited" in second["formatted"]
    assert "wait" in second["formatted"].lower()
    assert not second.get("isError")
    assert FakeJobServiceClient.get_calls == 1
    assert FakeLoggingClient.list_calls == 0


@pytest.mark.asyncio
async def test_vertex_inspect_failed_state_bypasses_cooldown_and_surfaces_root_error(
    monkeypatch,
):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")

    session = FakeSession()
    job_name = "projects/test-project/locations/us-central1/customJobs/123"
    session._gcp_vertex_monitor_cache = {
        job_name: {"monotonic": 999999999.0, "state": "JOB_STATE_RUNNING"}
    }
    tool = GcpVertexJobsTool(
        session=session,
        tool_call_id="call-failed",
        job_service_client_cls=FakeFailedJobServiceClient,
        logging_client_cls=FakeFailureLoggingClient,
    )

    result = await tool.execute({"operation": "inspect", "job_name": job_name})

    assert not result.get("isError")
    assert "JOB_STATE_FAILED" in result["formatted"]
    assert "DatasetNotFoundError" in result["formatted"]
    assert "View in Vertex AI" in result["formatted"]
    assert session.events[-1].data["state"] == "failed"
    assert session.events[-1].data["failureReason"]
    assert "hf-session-token" not in result["formatted"]


@pytest.mark.asyncio
async def test_run_requires_project_region_and_bucket(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_REGION", raising=False)
    monkeypatch.delenv("GCS_BUCKET", raising=False)

    tool = GcpVertexJobsTool(custom_job_cls=FakeCustomJob)

    result = await tool.execute({"operation": "run", "command": ["python", "train.py"]})

    assert result["isError"] is True
    assert "GOOGLE_CLOUD_PROJECT" in result["formatted"]
    assert "GOOGLE_CLOUD_REGION" in result["formatted"]
    assert "GCS_BUCKET" in result["formatted"]
    assert "Set GOOGLE_CLOUD_PROJECT" in result["formatted"]
    assert "/api/health/providers" in result["formatted"]


@pytest.mark.asyncio
async def test_handler_reads_script_from_active_sandbox(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    FakeCustomJob.instances = []

    class Sandbox:
        pass

    session = FakeSession()
    session.sandbox = Sandbox()

    async def fake_resolve_sandbox_script(sandbox, script):
        assert script == "/app/train.py"
        return "print('train')", None

    monkeypatch.setattr(
        "agent.tools.sandbox_tool.resolve_sandbox_script",
        fake_resolve_sandbox_script,
    )
    monkeypatch.setattr(
        "agent.tools.gcp_vertex_jobs_tool._load_custom_job_cls",
        lambda: FakeCustomJob,
    )

    output, ok = await gcp_vertex_jobs_handler(
        {"operation": "run", "script": "/app/train.py"},
        session=session,
        tool_call_id="call-2",
    )

    assert ok is True
    assert "Vertex AI job submitted" in output
    command = FakeCustomJob.instances[0].kwargs["worker_pool_specs"][0][
        "container_spec"
    ]["command"]
    encoded = re.search(r"b64decode\('([^']+)'\)", command[-1]).group(1)
    decoded = base64.b64decode(encoded).decode("utf-8")
    assert "pip" in decoded and "install" in decoded
    assert "print('train')" in decoded


def test_registered_tool_is_available():
    from agent.core.tools import create_builtin_tools

    tool_names = {tool.name for tool in create_builtin_tools(local_mode=True)}

    assert "gcp_vertex_jobs" in tool_names


@pytest.mark.asyncio
async def test_run_sft_template_staging_failure_emits_blocked_state(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    FakeCustomJob.instances = []

    async def failing_stage(**kwargs):
        raise RuntimeError(
            "Dataset staging failed before provider launch. "
            "Schema normalization failed: Unsupported dataset schema."
        )

    monkeypatch.setattr(
        "agent.tools.gcp_vertex_jobs_tool.stage_hf_dataset_to_gcs",
        failing_stage,
    )

    session = FakeSession()
    tool = GcpVertexJobsTool(
        session=session,
        tool_call_id="call-staging-fail",
        custom_job_cls=FakeCustomJob,
    )

    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "display_name": "gst-smoke",
            "dataset_name": "transitionGap/gst-india-preference-dataset-prep-small",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_policy": "cloud-private",
            "training_goal": "smoke-test",
            "max_run_hours": 1,
        }
    )

    assert result.get("isError") is True
    assert "Dataset staging failed before provider launch" in result["formatted"]
    assert session.events
    assert session.events[0].data["state"] == "blocked"
    assert session.events[0].data["tool"] == "gcp_vertex_jobs"
    assert FakeCustomJob.instances == []


class FakePendingJobServiceClient:
    def __init__(self, *, create_time, state_name="JOB_STATE_PENDING"):
        self.create_time = create_time
        self.state_name = state_name
        self.kwargs = {}

    def get_custom_job(self, name):
        return SimpleNamespace(
            name=name,
            display_name="pending-job",
            state=FakeState(self.state_name),
            create_time=self.create_time,
            update_time="updated",
        )


@pytest.mark.asyncio
async def test_cooldown_response_includes_poll_blocked_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    monkeypatch.setenv("GCP_VERTEX_MONITOR_COOLDOWN_SECONDS", "120")

    session = FakeSession()
    job_name = "projects/test-project/locations/us-central1/customJobs/123"
    session._gcp_vertex_monitor_cache = {
        job_name: {"monotonic": 999999999.0, "state": "JOB_STATE_RUNNING"}
    }
    tool = GcpVertexJobsTool(
        session=session,
        job_service_client_cls=FakeJobServiceClient,
    )

    result = await tool.execute({"operation": "logs", "job_name": job_name})

    assert result["poll_blocked"] is True
    assert result["retry_after_seconds"] > 0


@pytest.mark.asyncio
async def test_inspect_pending_too_long_sets_flag(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")

    create_time = datetime.now(timezone.utc) - timedelta(minutes=20)
    tool = GcpVertexJobsTool(
        job_service_client_cls=lambda **kwargs: FakePendingJobServiceClient(
            create_time=create_time
        ),
    )
    job_name = "projects/test-project/locations/us-central1/customJobs/123"

    result = await tool.execute({"operation": "inspect", "job_name": job_name})

    assert result["pending_too_long"] is True
    assert result["suggested_fallback_machine"] == "n1-standard-8"
    assert result["suggested_fallback_accelerator"] == "NVIDIA_TESLA_T4"


@pytest.mark.asyncio
async def test_inspect_pending_recent_no_flag(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")

    create_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    tool = GcpVertexJobsTool(
        job_service_client_cls=lambda **kwargs: FakePendingJobServiceClient(
            create_time=create_time
        ),
    )
    job_name = "projects/test-project/locations/us-central1/customJobs/123"

    result = await tool.execute({"operation": "inspect", "job_name": job_name})

    assert "pending_too_long" not in result


@pytest.mark.asyncio
async def test_inspect_terminal_job_sets_sandbox_stop_flag(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")

    session = FakeSession()
    session.sandbox = object()
    tool = GcpVertexJobsTool(
        session=session,
        tool_call_id="call-terminal",
        job_service_client_cls=FakeFailedJobServiceClient,
        logging_client_cls=FakeFailureLoggingClient,
    )
    job_name = "projects/test-project/locations/us-central1/customJobs/123"

    result = await tool.execute({"operation": "inspect", "job_name": job_name})

    assert result["sandbox_still_running"] is True
    assert result["sandbox_stop_recommended"] is True


def test_trim_inspect_logs_result_truncates_large_payload():
    from agent.tools.gcp_vertex_jobs_tool import _trim_inspect_logs_result

    large_body = "line\n" * 5000
    result = _trim_inspect_logs_result(
        {
            "formatted": f"**Vertex AI logs:**\n\n```text\n{large_body}\n```",
            "totalResults": 5000,
            "resultsShared": 5000,
        }
    )

    assert result["truncated"] is True
    assert "[... " in result["formatted"]
    assert "lines omitted for brevity" in result["formatted"]
    assert len(json.dumps(result, default=str)) <= 4000 or len(result["formatted"]) <= 4000 + 200


def test_trim_inspect_logs_result_keeps_small_payload():
    from agent.tools.gcp_vertex_jobs_tool import _trim_inspect_logs_result

    result = _trim_inspect_logs_result(
        {
            "formatted": "small payload",
            "totalResults": 1,
            "resultsShared": 1,
        }
    )

    assert "truncated" not in result
    assert result["formatted"] == "small payload"
