"""Validation helpers for stable cloud training templates."""

from __future__ import annotations

from typing import Any

VALID_TRAINING_GOALS = {"smoke-test", "production", "agent-decide"}
VALID_OUTPUT_POLICIES = {"cloud-private", "hf-hub", "cloud-and-hf-hub"}
VALID_TRACKIO_MODES = {"disabled", "reuse-existing", "create-if-allowed"}


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


def _optional_int(params: dict[str, Any], field: str) -> int | None:
    value = params.get(field)
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def validate_sft_template_request(params: dict[str, Any]) -> list[str]:
    """Return human-readable validation errors for an SFT template request."""

    errors: list[str] = []
    output_policy = str(params.get("output_policy") or "cloud-and-hf-hub").strip()
    training_goal = str(params.get("training_goal") or "agent-decide").strip()
    trackio_mode = str(params.get("trackio_mode") or "disabled").strip()

    if training_goal not in VALID_TRAINING_GOALS:
        errors.append(
            "training_goal must be one of: smoke-test, production, agent-decide"
        )
    if output_policy not in VALID_OUTPUT_POLICIES:
        errors.append(
            "output_policy must be one of: cloud-private, hf-hub, cloud-and-hf-hub"
        )
    if trackio_mode not in VALID_TRACKIO_MODES:
        errors.append(
            "trackio_mode must be one of: disabled, reuse-existing, create-if-allowed"
        )

    required_fields = ["dataset_name", "model_name"]
    if output_policy in {"hf-hub", "cloud-and-hf-hub"}:
        required_fields.append("hub_model_id")
    for field in required_fields:
        if not str(params.get(field) or "").strip():
            errors.append(f"{field} is required")

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
        value = params.get("validation_split_ratio")
        try:
            ratio = float(value)
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

    task_type = str(params.get("task_type") or "sft").strip().lower()
    if task_type != "sft":
        errors.append("Only task_type='sft' is supported by the stable SFT template")

    if params.get("packing") is True:
        errors.append("packing=True is not allowed for the stable Vertex SFT template")

    if params.get("attn_implementation"):
        errors.append(
            "attn_implementation is not allowed for the stable Vertex SFT template"
        )

    if params.get("flash_attention") or params.get("use_flash_attention"):
        errors.append(
            "flash attention is not allowed for the stable Vertex SFT template"
        )

    dataset_rows = _optional_int(params, "dataset_rows") or _optional_int(
        params, "dataset_num_rows"
    )
    max_train_samples = _optional_int(params, "max_train_samples")
    if (
        dataset_rows is not None
        and dataset_rows > 10_000
        and max_train_samples is None
        and params.get("full_dataset_approved") is True
        and not params.get("max_run_hours")
    ):
        errors.append(
            "full_dataset_approved=True for datasets over 10,000 rows requires max_run_hours for explicit cost-bounded approval"
        )

    submitted_text = " ".join(
        str(params.get(field) or "")
        for field in (
            "display_name",
            "dataset_name",
            "hub_model_id",
            "trackio_project",
            "trackio_space_id",
        )
    ).lower()
    if "dummy" in submitted_text or "placeholder" in submitted_text:
        errors.append(
            "dummy placeholder values are not allowed for Vertex training jobs"
        )

    target_text = " ".join(
        str(params.get(field) or "")
        for field in (
            "display_name",
            "hub_model_id",
            "trackio_project",
            "trackio_space_id",
        )
    ).lower()
    dataset_name = str(params.get("dataset_name") or "").lower()
    if "finance" in target_text and "medical" in dataset_name:
        errors.append("finance training jobs cannot use an obviously medical dataset")

    return errors
