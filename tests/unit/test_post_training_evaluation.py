import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from agent.core.post_training_evaluation import (  # noqa: E402
    build_post_training_evaluation,
    domain_from_context,
    evaluation_enabled,
    metric_summary,
    plan_post_training_evaluation,
)
from agent.core.session_persistence import NoopSessionStore  # noqa: E402
from routes import agent  # noqa: E402


def _context(**overrides):
    base = {
        "session_id": "s1",
        "run_id": "r1",
        "provider": "aws-sagemaker",
        "job_id": "train-job",
        "model_ref": "owner/model",
        "artifact_ref": "s3://bucket/model.tar.gz",
        "dataset_ref": "owner/hardware-dataset",
        "training_goal": "production",
        "task_type": "support",
        "metrics": {"eval_loss": 0.42, "eval_mean_token_accuracy": 0.88},
        "output_policy": "aws-private",
    }
    base.update(overrides)
    return base


def test_evaluation_flags_default_to_static(monkeypatch):
    monkeypatch.delenv("POST_TRAINING_EVAL_ENABLED", raising=False)
    monkeypatch.delenv("POST_TRAINING_EVAL_USE_PAID_JUDGE", raising=False)

    assert evaluation_enabled() is True


@pytest.mark.parametrize(
    ("text", "expected", "prompt_fragment", "safety_fragment"),
    [
        (
            "GPU overheating CPU throttling RAM compatibility SSD failure PSU repair",
            "hardware",
            "GPU overheating",
            "Do not instruct users to open or repair a power supply unit.",
        ),
        (
            "medical patient support symptoms medication diagnosis appointment",
            "medical",
            "symptoms",
            "Do not present model output as a medical diagnosis.",
        ),
        (
            "house price real estate appraisal rent bedroom mortgage estimate",
            "real_estate",
            "house price",
            "Do not claim an exact appraisal or guaranteed sale price.",
        ),
        (
            "customer support refund account troubleshooting",
            "generic_support",
            "support request",
            "Do not request passwords, tokens, or private credentials.",
        ),
        (
            "unclassified custom dataset",
            "unknown",
            "representative user request",
            "Treat generated answers as unverified until reviewed by a human.",
        ),
    ],
)
def test_planner_covers_domains(text, expected, prompt_fragment, safety_fragment):
    context = _context(dataset_ref=text)
    assert domain_from_context(context) == expected

    plan = plan_post_training_evaluation(context)

    assert plan["domain"] == expected
    assert any(prompt_fragment in prompt for prompt in plan["test_prompts"])
    assert any(
        safety_fragment in finding["message"] for finding in plan["safety_checks"]
    )
    assert plan["limitations"]


def test_metric_summary_extracts_known_training_metrics():
    summary = metric_summary(
        {
            "eval_loss": 0.25,
            "eval_mean_token_accuracy": 0.91,
            "train_runtime": 123.4,
            "secret_token": "hf_should_be_redacted",
        }
    )

    assert summary["available"] is True
    assert summary["metrics"]["eval_loss"] == 0.25
    assert summary["metrics"]["eval_mean_token_accuracy"] == 0.91
    assert "secret" not in str(summary).lower()


def test_build_evaluation_redacts_report_and_scores_static_result():
    evaluation = build_post_training_evaluation(
        _context(
            dataset_ref="medical patient support",
            metrics={"eval_loss": 0.3, "eval_mean_token_accuracy": 0.9},
            metadata={"note": "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456"},
        )
    )

    assert evaluation["status"] == "succeeded"
    assert evaluation["evaluation_type"] == "static_result_review"
    assert evaluation["scores"]["overall_score"] > 0
    assert evaluation["scores"]["safety_score"] < 1
    assert "not a certified benchmark" in evaluation["report_markdown"].lower()
    assert "Bearer abc" not in str(evaluation)


def test_failed_training_is_skipped():
    evaluation = build_post_training_evaluation(
        _context(training_status="failed", artifact_ref="s3://bucket/model.tar.gz")
    )

    assert evaluation["status"] == "skipped"
    assert "training did not succeed" in evaluation["failure_summary"].lower()


@pytest.mark.asyncio
async def test_store_upserts_evaluation_idempotently_and_updates_run_summary(
    monkeypatch,
):
    monkeypatch.setenv("AUDIT_TIMELINE_ENABLED", "true")
    store = NoopSessionStore()
    run = await store.create_run(session_id="s1", provider="aws-sagemaker")

    first = await store.upsert_evaluation(
        build_post_training_evaluation(_context(run_id=run["run_id"]))
    )
    second = await store.upsert_evaluation(
        build_post_training_evaluation(_context(run_id=run["run_id"]))
    )
    saved_run = await store.get_run(run["run_id"])
    audits = await store.list_audit_events(run_id=run["run_id"])

    assert first["evaluation_id"] == second["evaluation_id"]
    assert saved_run["evaluation_status"] == "succeeded"
    assert saved_run["evaluation_score"] == first["scores"]["overall_score"]
    assert {event["event_type"] for event in audits} >= {
        "evaluation_planned",
        "evaluation_started",
        "evaluation_completed",
    }


@pytest.mark.asyncio
async def test_evaluation_api_endpoints_and_manual_trigger_are_static_idempotent(
    monkeypatch,
):
    store = NoopSessionStore()
    run = await store.create_run(session_id="s1", provider="hf-jobs")
    await store.upsert_evaluation(
        build_post_training_evaluation(
            _context(
                run_id=run["run_id"],
                provider="hf-jobs",
                artifact_ref="https://huggingface.co/owner/model",
            )
        )
    )

    async def _allow_access(session_id, user, request=None, preload_sandbox=True):
        return SimpleNamespace(session_id=session_id, user_id=user["user_id"])

    monkeypatch.setattr(agent, "_check_session_access", _allow_access)
    monkeypatch.setattr(agent.session_manager, "persistence_store", store)
    monkeypatch.setattr(agent.session_manager, "_store", lambda: store)

    by_session = await agent.list_session_evaluations("s1", user={"user_id": "dev"})
    by_run = await agent.get_run_evaluation(
        "s1", run["run_id"], user={"user_id": "dev"}
    )
    report = await agent.get_run_evaluation_report(
        "s1", run["run_id"], user={"user_id": "dev"}
    )
    triggered = await agent.trigger_run_evaluation(
        "s1", run["run_id"], user={"user_id": "dev"}
    )

    assert len(by_session) == 1
    assert by_run.evaluation_id == triggered.evaluation_id
    assert report["report_markdown"] == by_run.report_markdown
    assert triggered.metadata["mode"] == "static"
