"""Validation helpers for AWS SageMaker SFT templates."""

from __future__ import annotations

from typing import Any

VALID_AWS_OUTPUT_POLICIES = {
    "aws-private",
    "cloud-private",
    "hf-hub",
    "cloud-and-hf-hub",
}


def _is_positive_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _validate_positive_field(
    params: dict[str, Any], field: str, errors: list[str], *, optional: bool = False
) -> None:
    value = params.get(field)
    if optional and value in (None, ""):
        return
    if not _is_positive_number(value):
        suffix = " when provided" if optional else ""
        errors.append(f"{field} must be positive{suffix}")


def _has_dummy_value(params: dict[str, Any]) -> bool:
    submitted_text = " ".join(
        str(params.get(field) or "")
        for field in (
            "job_name",
            "model_name",
            "output_model_id",
            "hub_model_id",
            "trackio_project",
            "trackio_space_id",
        )
    ).lower()
    return "dummy" in submitted_text or "placeholder" in submitted_text


def validate_aws_sft_template_request(params: dict[str, Any]) -> list[str]:
    """Return human-readable validation errors for AWS SFT template requests."""

    errors: list[str] = []
    for field in ("model_name", "output_model_id"):
        if not str(params.get(field) or "").strip():
            errors.append(f"{field} is required")

    output_policy = str(params.get("output_policy") or "aws-private")
    if output_policy not in VALID_AWS_OUTPUT_POLICIES:
        errors.append(
            "output_policy must be one of: "
            + ", ".join(sorted(VALID_AWS_OUTPUT_POLICIES))
        )

    for field in (
        "num_train_epochs",
        "max_length",
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
    ):
        if field in params:
            _validate_positive_field(params, field, errors)

    for field in ("max_train_samples", "max_eval_samples"):
        _validate_positive_field(params, field, errors, optional=True)

    if "validation_split_ratio" in params:
        try:
            ratio = float(params.get("validation_split_ratio"))
        except (TypeError, ValueError):
            ratio = 0.0
        if ratio <= 0 or ratio >= 1:
            errors.append(
                "validation_split_ratio must be greater than 0 and less than 1"
            )

    column_mapping = params.get("column_mapping")
    if column_mapping not in (None, ""):
        if not isinstance(column_mapping, dict):
            errors.append("column_mapping must be an object")
        else:
            assistant = column_mapping.get("assistant")
            if assistant is not None:
                assistant_is_valid = False
                if isinstance(assistant, str):
                    assistant_is_valid = bool(assistant.strip())
                elif isinstance(assistant, list):
                    assistant_is_valid = bool(assistant) and all(
                        isinstance(item, str) and item.strip() for item in assistant
                    )
                if not assistant_is_valid:
                    errors.append(
                        "column_mapping.assistant must be a string or a list of non-empty strings"
                    )

    if _has_dummy_value(params):
        errors.append("dummy placeholder values are not allowed for AWS training jobs")

    return errors
