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
