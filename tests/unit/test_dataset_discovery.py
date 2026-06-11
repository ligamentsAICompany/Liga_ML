from agent.core.dataset_discovery import (
    DatasetCandidate,
    build_dataset_discovery_result,
    build_dataset_discovery_plan,
    extract_dataset_intent,
    extract_hf_dataset_candidates_from_text,
    format_dataset_discovery_plan,
    rank_candidates,
)


def test_default_sources_exclude_kaggle_and_require_user_selection():
    plan = build_dataset_discovery_plan(
        domain="medical",
        task_type="sft",
        user_goal="fine-tune a medical QA assistant",
    )

    assert plan.allowed_sources == [
        "huggingface",
        "github",
        "papers",
        "public_web",
    ]
    assert plan.excluded_sources == ["kaggle"]
    assert plan.requires_user_selection is True


def test_candidate_ranking_sorts_by_score_descending():
    low = DatasetCandidate(
        name="Low",
        source="github",
        url=None,
        domain="finance",
        task_type="sft",
        license=None,
        size=None,
        schema_hint=[],
        quality_notes=[],
        risks=[],
        score=0.25,
        reason="Weak match.",
    )
    high = DatasetCandidate(
        name="High",
        source="huggingface",
        url="https://huggingface.co/datasets/example/high",
        domain="finance",
        task_type="sft",
        license="mit",
        size="10k rows",
        schema_hint=["messages"],
        quality_notes=["Instruction format."],
        risks=["Verify PII handling."],
        score=0.92,
        reason="Strong match.",
    )

    assert rank_candidates([low, high]) == [high, low]


def test_markdown_format_mentions_sources_candidates_and_approval():
    plan = build_dataset_discovery_plan(
        domain="finance",
        task_type="sft",
        user_goal="fine-tune a finance assistant",
        candidates=[
            {
                "name": "Finance QA",
                "source": "huggingface",
                "url": "https://huggingface.co/datasets/example/finance-qa",
                "domain": "finance",
                "task_type": "sft",
                "license": "cc-by-4.0",
                "size": "5k rows",
                "schema_hint": ["question", "answer"],
                "quality_notes": ["QA-style columns."],
                "risks": ["Confirm commercial license fit."],
                "score": 0.88,
                "reason": "Matches finance QA fine-tuning.",
            }
        ],
    )

    output = format_dataset_discovery_plan(plan)

    assert "No uploaded dataset detected" in output
    assert "Hugging Face Datasets" in output
    assert "GitHub" in output
    assert "papers" in output
    assert "public web" in output
    assert "Kaggle" in output
    assert "Excluded Sources" in output
    assert "Finance QA" in output
    assert "question, answer" in output
    assert "Please select" in output


def test_extract_dataset_intent_detects_domain_task_provider_and_sensitivity():
    manufacturing = extract_dataset_intent(
        "Fine-tune me an LLM on manufacturing dataset using GCloud."
    )
    ipl = extract_dataset_intent(
        "Fine-tune a small model on IPL cricket data using Hugging Face."
    )
    hardware = extract_dataset_intent(
        "Find a safe hardware troubleshooting dataset for AWS SageMaker fine-tuning."
    )
    medical = extract_dataset_intent("Train on medical patient support conversations.")
    housing = extract_dataset_intent("I need house price prediction data.")

    assert manufacturing.domain == "manufacturing"
    assert manufacturing.task_type == "sft"
    assert manufacturing.target_provider == "gcp-vertex"
    assert manufacturing.data_modality in {"text", "tabular", "mixed"}
    assert ipl.domain == "sports_cricket_ipl"
    assert ipl.target_provider == "hf-jobs"
    assert hardware.domain == "hardware_support"
    assert hardware.target_provider == "aws-sagemaker"
    assert medical.privacy_sensitivity == "high"
    assert housing.domain == "real_estate"
    assert housing.task_type == "regression"


def test_build_dataset_discovery_result_scores_risks_and_excludes_kaggle():
    result = build_dataset_discovery_result(
        query="Find medical troubleshooting data",
        candidates=[
            {
                "dataset_id": "med/private-patients",
                "source": "huggingface",
                "repo_id": "med/private-patients",
                "title": "Private Patient Conversations",
                "description": "Medical patient support data",
                "license": None,
                "columns": ["patient_name", "symptom", "answer"],
                "row_count": 100,
            },
            {
                "dataset_id": "kaggle/ipl",
                "source": "kaggle",
                "title": "IPL Matches",
                "license": "unknown",
                "columns": ["team", "score"],
                "row_count": 10_000,
            },
            {
                "dataset_id": "public/hardware-support",
                "source": "huggingface",
                "repo_id": "public/hardware-support",
                "title": "Hardware Support QA",
                "description": "Troubleshooting instruction response examples",
                "license": "mit",
                "columns": ["instruction", "output", "category"],
                "row_count": 5_000,
            },
        ],
    )

    by_id = {candidate.dataset_id: candidate for candidate in result.candidates}

    assert result.intent.domain == "medical"
    assert result.allowed_sources == ["huggingface", "github", "papers", "public_web"]
    assert "kaggle" in result.excluded_sources
    assert by_id["kaggle/ipl"].excluded is True
    assert by_id["kaggle/ipl"].exclusion_reason == "Kaggle is future work only."
    assert by_id["med/private-patients"].privacy_status == "high"
    assert by_id["med/private-patients"].license_status == "missing"
    assert by_id["med/private-patients"].schema_status == "needs_mapping"
    assert by_id["public/hardware-support"].schema_status == "compatible"
    assert by_id["public/hardware-support"].load_dataset_snippet.startswith(
        "from datasets import load_dataset"
    )
    assert result.recommended_candidate.dataset_id == "public/hardware-support"
    assert result.requires_user_selection is True


def test_discovery_result_redacts_secret_like_candidate_text():
    result = build_dataset_discovery_result(
        query="hardware support HF_TOKEN=hf_secret",
        candidates=[
            {
                "dataset_id": "safe/support",
                "source": "huggingface",
                "repo_id": "safe/support",
                "title": "Support token sk-test-secret",
                "license": "apache-2.0",
                "columns": ["prompt", "completion"],
                "row_count": 1_500,
            }
        ],
    )

    payload = result.to_dict()

    assert "hf_secret" not in str(payload)
    assert "sk-test-secret" not in str(payload)
    assert "[REDACTED]" in str(payload)


def test_no_candidate_result_includes_explicit_reason():
    result = build_dataset_discovery_result(query="Find GST tax support data")
    payload = result.to_dict()

    assert payload["candidates"] == []
    assert payload["no_candidates_reason"]
    assert "No candidate datasets supplied yet" in payload["no_candidates_reason"]


def test_extracts_hf_dataset_candidates_from_research_text_for_persistence():
    candidates = extract_hf_dataset_candidates_from_text(
        """
        Recommended datasets:
        1. transitionGap/gst-india-preference-dataset-prep-small is GST-specific.
        2. https://huggingface.co/datasets/Kahrhoff/openfinancial-chatbot-dataset
        """
    )

    assert [candidate["dataset_id"] for candidate in candidates] == [
        "transitionGap/gst-india-preference-dataset-prep-small",
        "Kahrhoff/openfinancial-chatbot-dataset",
    ]
