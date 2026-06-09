"""Phase 7b integrated readiness, fallback, and safety behavior tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent.core.session_persistence import NoopSessionStore
from agent.core.training_preflight import (
    PreflightCheckCategory,
    PreflightSeverity,
    PreflightStatus,
    TrainingPreflightCheck,
    build_training_preflight_result,
    run_local_training_preflight,
)

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from models import TrainingPreflightResultModel  # noqa: E402


def _check(
    status: PreflightStatus,
    *,
    severity: PreflightSeverity = PreflightSeverity.INFO,
    required: bool = True,
    message: str = "check completed",
) -> TrainingPreflightCheck:
    return TrainingPreflightCheck(
        check_id="integration_check",
        provider="hf-jobs",
        category=PreflightCheckCategory.CREDENTIALS,
        label="Integration check",
        status=status,
        severity=severity,
        message=message,
        details={"required": required, "applicable": required},
    )


def _recommendation(
    *,
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
    hardware_id: str = "hf-jobs:t4-small",
    fallback_option: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> dict:
    return {
        "provider": "hf-jobs",
        "recommended_model": model_id,
        "output_policy": "hf-hub",
        "recommendation": {
            "selected_provider": {"provider_id": "hf-jobs"},
            "selected_model": {"model_id": model_id},
            "selected_hardware": {"hardware_id": hardware_id},
            "output_policy": "hf-hub",
            "fallbacks": [
                {
                    "blocked_option": model_id,
                    "fallback_option": fallback_option,
                    "reason": "Use a smaller model for the selected hardware.",
                }
            ],
        },
    }


def test_readiness_semantics_cover_pass_fail_unknown_warning_and_skip():
    passed = build_training_preflight_result(
        session_id="s1",
        provider="hf-jobs",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="hf-jobs:t4-small",
        output_policy="hf-hub",
        checks=[_check(PreflightStatus.PASSED)],
    )
    failed = build_training_preflight_result(
        session_id="s1",
        provider="hf-jobs",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="hf-jobs:t4-small",
        output_policy="hf-hub",
        checks=[
            _check(
                PreflightStatus.FAILED,
                severity=PreflightSeverity.BLOCKING,
                message="Blocking failure.",
            )
        ],
    )
    unknown = build_training_preflight_result(
        session_id="s1",
        provider="hf-jobs",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="hf-jobs:t4-small",
        output_policy="hf-hub",
        checks=[_check(PreflightStatus.UNKNOWN, message="Required unknown.")],
    )
    warning = build_training_preflight_result(
        session_id="s1",
        provider="hf-jobs",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="hf-jobs:t4-small",
        output_policy="hf-hub",
        checks=[
            _check(
                PreflightStatus.WARNING,
                severity=PreflightSeverity.WARNING,
                required=False,
                message="Non-blocking warning.",
            )
        ],
    )
    skipped = build_training_preflight_result(
        session_id="s1",
        provider="hf-jobs",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="hf-jobs:t4-small",
        output_policy="hf-hub",
        checks=[
            _check(
                PreflightStatus.SKIPPED,
                required=False,
                message="Non-applicable skip.",
            )
        ],
    )

    assert passed.launch_ready is True
    assert failed.launch_ready is False
    assert unknown.launch_ready is False
    assert warning.launch_ready is True
    assert skipped.launch_ready is True
    assert unknown.status == PreflightStatus.UNKNOWN


def test_include_fallbacks_false_does_not_probe_or_attach_fallbacks():
    result = run_local_training_preflight(
        session_id="s1",
        recommendation=_recommendation(),
        dataset_summary={"rows": 10},
        include_fallbacks=False,
    )

    assert result.fallbacks == []
    assert result.verified_fallback is None
    assert result.metadata["fallbacks_checked"] is False


def test_primary_failed_with_static_fallback_passed_sets_advisory_verified_fallback():
    result = run_local_training_preflight(
        session_id="s1",
        recommendation=_recommendation(),
        dataset_summary={"rows": 10},
        include_fallbacks=True,
    )

    assert result.launch_ready is False
    assert result.fallbacks
    assert result.fallbacks[0].status == PreflightStatus.PASSED
    assert result.fallbacks[0].launch_ready is True
    assert result.fallbacks[0].metadata["advisory_only"] is True
    assert result.fallbacks[0].metadata["fallback_executed"] is False
    assert result.verified_fallback is not None
    assert result.verified_fallback.fallback_id == result.fallbacks[0].fallback_id
    assert result.metadata["automatic_fallback_execution"] is False
    assert result.metadata["provider_jobs_launched"] is False
    assert result.metadata["resources_created"] is False


def test_primary_failed_with_unknown_fallback_shows_unverified_fallback_only():
    result = run_local_training_preflight(
        session_id="s1",
        recommendation=_recommendation(fallback_option="unknown/custom-model"),
        dataset_summary={"rows": 10},
        include_fallbacks=True,
    )

    assert result.launch_ready is False
    assert result.fallbacks
    assert result.fallbacks[0].status == PreflightStatus.UNKNOWN
    assert result.fallbacks[0].launch_ready is False
    assert result.verified_fallback is None


@pytest.mark.asyncio
async def test_serialized_api_and_persistence_preserve_fallback_metadata_safely():
    result = run_local_training_preflight(
        session_id="s1",
        run_id="r1",
        recommendation=_recommendation(),
        dataset_summary={"rows": 10},
        include_fallbacks=True,
        metadata={"note": "HF_TOKEN=hf_" + "A" * 35},
    )
    payload = result.to_dict()

    model = TrainingPreflightResultModel(**payload)
    store = NoopSessionStore()
    run = await store.create_run(session_id="s1", provider="hf-jobs")
    saved = await store.record_training_preflight(
        session_id="s1",
        run_id=run["run_id"],
        preflight={**model.model_dump(), "run_id": run["run_id"]},
    )
    restored = await store.get_run_training_preflight("s1", run["run_id"])

    assert model.verified_fallback is not None
    assert saved["verified_fallback"]["metadata"]["advisory_only"] is True
    assert restored["fallbacks"][0]["metadata"]["fallback_executed"] is False
    assert "hf_" not in str(saved)
    assert saved["metadata"]["provider_jobs_launched"] is False
    assert saved["metadata"]["resources_created"] is False
