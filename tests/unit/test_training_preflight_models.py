"""Tests for Phase 7b training preflight foundation models."""

import pytest

from agent.core.preflight_errors import normalize_provider_error
from agent.core.training_preflight import (
    PreflightCheckCategory,
    PreflightSeverity,
    PreflightStatus,
    TrainingPreflightCheck,
    build_training_preflight_result,
    derive_launch_ready,
    derive_manual_approval_policy,
    run_local_training_preflight,
)


def _check(
    status: PreflightStatus,
    *,
    check_id: str = "credentials",
    category: PreflightCheckCategory = PreflightCheckCategory.CREDENTIALS,
    severity: PreflightSeverity = PreflightSeverity.INFO,
    required: bool = True,
    message: str = "Credential check completed.",
    details: dict | None = None,
) -> TrainingPreflightCheck:
    return TrainingPreflightCheck(
        check_id=check_id,
        provider="hf-jobs",
        category=category,
        label="Credentials",
        status=status,
        severity=severity,
        message=message,
        details=details or {"required": required},
        docs_verification_required=False,
    )


def test_status_enum_accepts_expected_values_only():
    assert [status.value for status in PreflightStatus] == [
        "not_run",
        "checking",
        "passed",
        "warning",
        "failed",
        "unknown",
        "skipped",
    ]

    with pytest.raises(ValueError):
        PreflightStatus("ready")


def test_blocking_failed_check_makes_launch_ready_false():
    checks = [
        _check(
            PreflightStatus.FAILED,
            severity=PreflightSeverity.BLOCKING,
            message="Credential validation failed.",
        )
    ]

    launch_ready, blocking, warnings, unknowns = derive_launch_ready(checks)

    assert launch_ready is False
    assert blocking == ["Credential validation failed."]
    assert warnings == []
    assert unknowns == []


def test_required_unknown_check_makes_launch_ready_false():
    checks = [
        _check(
            PreflightStatus.UNKNOWN,
            severity=PreflightSeverity.ERROR,
            message="Model access could not be verified.",
        )
    ]

    launch_ready, blocking, warnings, unknowns = derive_launch_ready(checks)

    assert launch_ready is False
    assert blocking == []
    assert warnings == []
    assert unknowns == ["Model access could not be verified."]


def test_required_unknown_check_with_warning_severity_still_blocks_launch():
    checks = [
        _check(
            PreflightStatus.UNKNOWN,
            category=PreflightCheckCategory.QUOTA,
            severity=PreflightSeverity.WARNING,
            message="Quota could not be verified.",
        )
    ]

    launch_ready, blocking, warnings, unknowns = derive_launch_ready(checks)

    assert launch_ready is False
    assert blocking == []
    assert warnings == []
    assert unknowns == ["Quota could not be verified."]


def test_warning_only_checks_do_not_block_launch():
    checks = [
        _check(
            PreflightStatus.WARNING,
            category=PreflightCheckCategory.QUOTA,
            severity=PreflightSeverity.WARNING,
            required=False,
            message="Quota API is unavailable; use conservative defaults.",
        )
    ]

    launch_ready, blocking, warnings, unknowns = derive_launch_ready(checks)

    assert launch_ready is True
    assert blocking == []
    assert warnings == ["Quota API is unavailable; use conservative defaults."]
    assert unknowns == []


def test_passed_checks_produce_launch_ready_true():
    checks = [_check(PreflightStatus.PASSED)]

    launch_ready, blocking, warnings, unknowns = derive_launch_ready(checks)

    assert launch_ready is True
    assert blocking == []
    assert warnings == []
    assert unknowns == []


def test_skipped_non_applicable_checks_do_not_block():
    checks = [
        _check(
            PreflightStatus.SKIPPED,
            category=PreflightCheckCategory.STORAGE,
            required=False,
            message="Cloud storage is not required for this output policy.",
            details={"required": False, "applicable": False},
        )
    ]

    launch_ready, blocking, warnings, unknowns = derive_launch_ready(checks)

    assert launch_ready is True
    assert blocking == []
    assert warnings == []
    assert unknowns == []


def test_top_level_result_includes_safe_metadata_flags_and_summary():
    result = build_training_preflight_result(
        session_id="session-1",
        run_id="run-1",
        provider="hf-jobs",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="hf-jobs:t4-small",
        output_policy="cloud-and-hf-hub",
        checks=[_check(PreflightStatus.PASSED)],
        metadata={"note": "HF_TOKEN=hf_" + "A" * 35},
    )

    payload = result.to_dict()

    assert payload["launch_ready"] is True
    assert payload["status"] == "passed"
    assert payload["metadata"]["provider_jobs_launched"] is False
    assert payload["metadata"]["resources_created"] is False
    assert payload["metadata"]["live_checks_optional"] is True
    assert "hf_" not in payload["safe_summary"]
    assert "[REDACTED]" in str(payload["metadata"])


def test_unknown_does_not_become_passed_in_top_level_status():
    result = build_training_preflight_result(
        session_id="session-1",
        run_id="run-1",
        provider="hf-jobs",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="hf-jobs:t4-small",
        output_policy="cloud-and-hf-hub",
        checks=[
            _check(
                PreflightStatus.UNKNOWN,
                message="Required model access check did not run.",
            )
        ],
    )

    assert result.status == PreflightStatus.UNKNOWN
    assert result.launch_ready is False


def test_normalize_provider_error_redacts_secret_material():
    private_key = "-----BEGIN PRIVATE KEY-----\nabc123\n-----END PRIVATE KEY-----"
    fake_aws_key = "AKIA" + "ABCDEFGHIJKLMNOP"
    message = " ".join(
        [
            "HF token hf_" + "A" * 35,
            "AWS key " + fake_aws_key,
            "mongo mongodb+srv://user:pass@example.mongodb.net/db",
            private_key,
        ]
    )

    normalized = normalize_provider_error(message, provider="hf-jobs")

    assert normalized["provider"] == "hf-jobs"
    assert "hf_" not in normalized["message"]
    assert fake_aws_key not in normalized["message"]
    assert "user:pass@" not in normalized["message"]
    assert "BEGIN PRIVATE KEY" not in normalized["message"]
    assert "[REDACTED]" in normalized["message"]


@pytest.mark.parametrize(
    ("message", "expected_code"),
    [
        ("Request timed out while checking model metadata", "timeout"),
        ("401 Unauthorized: invalid token", "auth_failed"),
        ("403 permission denied for project", "permission_denied"),
        ("404 model not found", "not_found"),
        ("Temporary DNS failure / connection refused", "network"),
    ],
)
def test_normalize_provider_error_maps_common_errors(message, expected_code):
    normalized = normalize_provider_error(message)

    assert normalized["error_code"] == expected_code
    assert "traceback" not in normalized
    assert "message" in normalized


def _recommendation(
    *,
    provider: str = "hf-jobs",
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    hardware_id: str = "hf-jobs:t4-small",
    output_policy: str = "cloud-and-hf-hub",
) -> dict:
    return {
        "provider": provider,
        "recommended_model": model_id,
        "output_policy": output_policy,
        "recommendation": {
            "selected_provider": {"provider_id": provider},
            "selected_model": {"model_id": model_id},
            "selected_hardware": {"hardware_id": hardware_id},
            "output_policy": output_policy,
        },
    }


def test_local_preflight_keeps_launch_blocked_without_live_probes():
    result = run_local_training_preflight(
        session_id="session-1",
        run_id="run-1",
        recommendation=_recommendation(),
        dataset_summary={"rows": 25},
    )

    payload = result.to_dict()

    assert payload["status"] == "unknown"
    assert payload["launch_ready"] is False
    assert payload["metadata"]["provider_jobs_launched"] is False
    assert payload["metadata"]["resources_created"] is False
    assert any("live" in reason.lower() for reason in payload["unknown_reasons"])
    assert "not implemented" in payload["safe_summary"].lower()


def test_local_preflight_missing_required_static_fields_fails():
    result = run_local_training_preflight(
        session_id="session-1",
        recommendation={"provider": "hf-jobs", "output_policy": "cloud-and-hf-hub"},
    )

    assert result.status == PreflightStatus.FAILED
    assert result.launch_ready is False
    assert any("model" in reason.lower() for reason in result.blocking_reasons)
    assert any("hardware" in reason.lower() for reason in result.blocking_reasons)


def test_local_preflight_unknown_output_or_storage_checks_block_by_default():
    result = run_local_training_preflight(
        session_id="session-1",
        recommendation=_recommendation(output_policy="hf-hub"),
        dataset_summary={"rows": 25},
    )

    assert result.status == PreflightStatus.UNKNOWN
    assert result.launch_ready is False
    assert any("hub" in reason.lower() for reason in result.unknown_reasons)


def test_local_preflight_model_hardware_incompatibility_blocks_launch():
    result = run_local_training_preflight(
        session_id="session-1",
        recommendation=_recommendation(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            hardware_id="hf-jobs:t4-small",
        ),
        dataset_summary={"rows": 25},
    )

    assert result.status == PreflightStatus.FAILED
    assert result.launch_ready is False
    assert any("memory" in reason.lower() for reason in result.blocking_reasons)


def _smoke_recommendation(*, training_goal: str = "smoke-test") -> dict:
    return {
        "provider": "gcp-vertex",
        "training_goal": training_goal,
        "output_policy": "cloud-private",
        "recommendation": {
            "training_goal": training_goal,
            "estimated_cost_usd": 1.1,
            "selected_provider": {"provider_id": "gcp-vertex"},
            "selected_model": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
            "selected_hardware": {"hardware_id": "gcp-vertex:n1-standard-8-t4"},
        },
    }


def test_manual_approval_allowed_for_vertex_smoke_with_only_quota_unknown():
    checks = [
        TrainingPreflightCheck(
            check_id="gcp.vertex.quota_availability",
            provider="gcp-vertex",
            category=PreflightCheckCategory.QUOTA,
            label="Vertex quota",
            status=PreflightStatus.UNKNOWN,
            severity=PreflightSeverity.WARNING,
            message="Quota unknown.",
            details={"required": True},
        )
    ]
    launch_ready, blocking, warnings, unknowns = derive_launch_ready(checks)
    allowed, approval_required, reason = derive_manual_approval_policy(
        checks,
        provider="gcp-vertex",
        blocking_reasons=blocking,
        unknown_reasons=unknowns,
        recommendation=_smoke_recommendation(),
    )

    assert launch_ready is False
    assert allowed is True
    assert approval_required is True
    assert reason
    assert "launch_ready remains false" in reason


def test_manual_approval_disallowed_for_production_run():
    checks = [
        TrainingPreflightCheck(
            check_id="gcp.vertex.quota_availability",
            provider="gcp-vertex",
            category=PreflightCheckCategory.QUOTA,
            label="Vertex quota",
            status=PreflightStatus.UNKNOWN,
            severity=PreflightSeverity.WARNING,
            message="Quota unknown.",
            details={"required": True},
        )
    ]
    _, blocking, _, unknowns = derive_launch_ready(checks)
    allowed, approval_required, reason = derive_manual_approval_policy(
        checks,
        provider="gcp-vertex",
        blocking_reasons=blocking,
        unknown_reasons=unknowns,
        recommendation=_smoke_recommendation(training_goal="production"),
    )

    assert allowed is False
    assert approval_required is False
    assert reason is None


def test_manual_approval_disallowed_when_bucket_check_failed():
    checks = [
        TrainingPreflightCheck(
            check_id="gcp.gcs.bucket_read",
            provider="gcp-vertex",
            category=PreflightCheckCategory.STORAGE,
            label="GCS bucket",
            status=PreflightStatus.FAILED,
            severity=PreflightSeverity.BLOCKING,
            message="Bucket missing.",
            details={"required": True},
        ),
        TrainingPreflightCheck(
            check_id="gcp.vertex.quota_availability",
            provider="gcp-vertex",
            category=PreflightCheckCategory.QUOTA,
            label="Vertex quota",
            status=PreflightStatus.UNKNOWN,
            severity=PreflightSeverity.WARNING,
            message="Quota unknown.",
            details={"required": True},
        ),
    ]
    _, blocking, _, unknowns = derive_launch_ready(checks)
    allowed, _, _ = derive_manual_approval_policy(
        checks,
        provider="gcp-vertex",
        blocking_reasons=blocking,
        unknown_reasons=unknowns,
        recommendation=_smoke_recommendation(),
    )

    assert allowed is False
