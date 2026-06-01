from agent.core.dataset_discovery import (
    DatasetCandidate,
    build_dataset_discovery_plan,
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
