from pathlib import Path


def test_prompt_mentions_training_planner_before_cloud_training():
    prompt = Path("agent/prompts/system_prompt_v3.yaml").read_text()

    assert "training_planner" in prompt
    assert "Before launching cloud training" in prompt
    assert "user approval still required" in prompt


def test_prompt_keeps_kaggle_as_future_work_not_download_source():
    prompt = Path("agent/prompts/system_prompt_v3.yaml").read_text()

    assert "Kaggle is future work" in prompt
    assert "Do not rely on Kaggle downloads" in prompt


def test_prompt_uses_dataset_discovery_for_no_upload_training():
    prompt = Path("agent/prompts/system_prompt_v3.yaml").read_text()

    assert "For fine-tuning requests, check uploaded session data first" in prompt
    assert "dataset_discovery" in prompt
    assert "no uploaded dataset exists" in prompt
    assert "ask the user to approve/select" in prompt


def test_prompt_includes_shared_output_policy_contract():
    prompt = Path("agent/prompts/system_prompt_v3.yaml").read_text()

    assert "shared output policy contract" in prompt
    assert "cloud-private" in prompt
    assert "provider-native private cloud/job storage" in prompt
    assert "hf-hub" in prompt
    assert "cloud-and-hf-hub" in prompt


def test_prompt_guides_sensitive_domains_to_cloud_private():
    prompt = Path("agent/prompts/system_prompt_v3.yaml").read_text()

    assert "Sensitive domains" in prompt
    assert "recommend `cloud-private`" in prompt
    assert "AWS `cloud-private` means S3" in prompt
    assert "GCloud `cloud-private` means GCS" in prompt
    assert "HF Jobs privacy depends on private Hub/job artifact settings" in prompt


def test_prompt_continues_hf_jobs_after_training_planner():
    prompt = Path("agent/prompts/system_prompt_v3.yaml").read_text()

    assert "cloud_provider=hf-jobs" in prompt
    assert "After the plan is prepared" in prompt
    assert "continue to a Hugging Face Jobs preflight" in prompt
    assert "Use the `hf_jobs` backend" in prompt
    assert "Do not route to `gcp_vertex_jobs` or `aws_sagemaker_jobs`" in prompt
