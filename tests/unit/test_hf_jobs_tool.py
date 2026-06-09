import re
from types import SimpleNamespace

import pytest

from agent.tools.jobs_tool import HF_JOBS_TOOL_SPEC, HfJobsTool


class FakeHfJobsApi:
    def __init__(self):
        self.run_job_kwargs = None

    def run_job(self, **kwargs):
        self.run_job_kwargs = kwargs
        return SimpleNamespace(
            id="job-template-123",
            url="https://huggingface.co/jobs/acme/job-template-123",
            status=SimpleNamespace(stage="COMPLETED", message=None),
        )

    def fetch_job_logs(self, **_kwargs):
        return iter(
            [
                "training finished",
                "LIGA_FINAL_MODEL_URL=https://huggingface.co/acme/final-model",
            ]
        )

    def inspect_job(self, **_kwargs):
        return SimpleNamespace(status=SimpleNamespace(stage="COMPLETED"))


def _decode_inline_script(command):
    assert command[:2] == ["/bin/sh", "-lc"]
    match = re.search(r'echo "([^"]+)" \| base64 -d \| uv run', command[2])
    assert match, command
    import base64

    return base64.b64decode(match.group(1)).decode("utf-8")


@pytest.mark.asyncio
async def test_hf_jobs_sft_template_uses_stable_runtime_script():
    api = FakeHfJobsApi()
    tool = HfJobsTool(hf_token="hf-session-token", namespace="acme")
    tool.api = api

    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "dataset_name": "trl-lib/Capybara",
            "dataset_split": "train",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "hub_model_id": "acme/qwen25-capybara-smoke",
            "training_goal": "smoke-test",
            "output_policy": "cloud-and-hf-hub",
            "max_train_samples": 5,
            "max_eval_samples": 2,
            "timeout": "2h",
        }
    )

    assert not result.get("isError")
    assert api.run_job_kwargs is not None
    script = _decode_inline_script(api.run_job_kwargs["command"])
    assert "torch==2.4.0" in script
    assert "trl==1.5.1" in script
    assert "processing_class=tokenizer" in script
    assert "tokenizer.pad_token = tokenizer.eos_token" in script
    assert "tokenizer=" not in script
    assert "max_seq_length" not in script
    assert "HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required" in script
    assert api.run_job_kwargs["secrets"]["HF_TOKEN"] == "hf-session-token"
    assert api.run_job_kwargs["secrets"]["HUGGINGFACE_HUB_TOKEN"] == "hf-session-token"


@pytest.mark.asyncio
async def test_hf_jobs_sft_template_validates_before_submit():
    api = FakeHfJobsApi()
    tool = HfJobsTool(hf_token="hf-session-token", namespace="acme")
    tool.api = api

    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "dataset_name": "trl-lib/Capybara",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_policy": "hf-hub",
        }
    )

    assert result["isError"] is True
    assert "hub_model_id is required" in result["formatted"]
    assert api.run_job_kwargs is None


def test_hf_jobs_tool_spec_exposes_stable_sft_template():
    properties = HF_JOBS_TOOL_SPEC["parameters"]["properties"]

    assert properties["template"]["enum"] == ["sft"]
    assert "dataset_name" in properties
    assert "hub_model_id" in properties
    assert "stable SFT template" in properties["template"]["description"]
