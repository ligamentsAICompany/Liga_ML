#!/usr/bin/env python3
"""Non-billable local validation for the AWS SageMaker SFT path."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from agent.core.aws_readiness import build_aws_sagemaker_readiness_snapshot  # noqa: E402
from agent.core.cost_estimation import estimate_aws_sagemaker_job_cost  # noqa: E402
from agent.tools.aws_sagemaker_jobs_tool import AwsSageMakerJobsTool  # noqa: E402
from agent.training_templates.aws_sft import (  # noqa: E402
    AwsSftTemplateConfig,
    build_aws_sft_training_script,
)
from agent.training_templates.aws_validation import (  # noqa: E402
    validate_aws_sft_template_request,
)

load_dotenv(PROJECT_ROOT / ".env", override=False)

_SENSITIVE_KEY_PARTS = (
    "token",
    "secret",
    "password",
    "private_key",
    "credential",
    "access_key",
)
_REQUIRED_MARKERS = [
    "LIGA_TRAINING_STATUS=succeeded",
    "LIGA_PROVIDER=aws-sagemaker",
    "LIGA_AWS_TRAINING_JOB_NAME=",
    "LIGA_AWS_REGION=",
    "LIGA_S3_MODEL_ARTIFACT=",
    "LIGA_S3_OUTPUT_DIR=",
    "LIGA_CLOUDWATCH_LOGS_URL=",
    "LIGA_FINAL_MODEL_URL=",
    "LIGA_HUB_MODEL_ID=",
    "LIGA_EVAL_RESULT_JSON=",
    "LIGA_RESULT_FILE=",
]
_REQUIRED_PATHS = [
    "/opt/ml/input/data/train",
    "/opt/ml/model",
    "/opt/ml/output/data",
]


def _safe_snapshot(value: Any) -> Any:
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
                if key == "credentials_detected":
                    safe[key] = bool(item)
                continue
            safe[key] = _safe_snapshot(item)
        return safe
    if isinstance(value, list):
        return [_safe_snapshot(item) for item in value]
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a sample AWS SageMaker SFT request without live AWS calls."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-model-id", required=True)
    parser.add_argument("--max-run-seconds", type=int, required=True)
    parser.add_argument("--output-policy")
    parser.add_argument("--instance-type")
    parser.add_argument("--instance-count", type=int)
    parser.add_argument("--image-uri")
    parser.add_argument(
        "--allow-missing-aws",
        action="store_true",
        help="Treat missing local AWS readiness as a warning.",
    )
    parser.add_argument(
        "--allow-missing-image",
        action="store_true",
        help="Treat missing SageMaker image URI as a warning.",
    )
    return parser


def _request_from_args(
    args: argparse.Namespace, readiness: dict[str, Any]
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "operation": "run",
        "template": "sft",
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "model_name": args.model_name,
        "output_model_id": args.output_model_id,
        "max_run_seconds": args.max_run_seconds,
        "output_policy": args.output_policy or readiness.get("output_policy"),
        "instance_type": args.instance_type or readiness.get("default_instance_type"),
        "instance_count": args.instance_count
        if args.instance_count is not None
        else readiness.get("default_instance_count"),
        "image_uri": args.image_uri or readiness.get("training_image_uri"),
    }
    return {key: value for key, value in request.items() if value not in (None, "")}


def _template_config(request: dict[str, Any]) -> AwsSftTemplateConfig:
    return AwsSftTemplateConfig(
        dataset_split=str(request.get("dataset_split") or "train"),
        model_name=str(request["model_name"]),
        output_model_id=str(request["output_model_id"]),
        output_policy=str(request.get("output_policy") or "aws-private"),
    )


def _validate_run_request(request: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if str(request.get("operation") or "").strip().lower() != "run":
        errors.append("operation must be run")
    if str(request.get("template") or "").strip().lower() != "sft":
        errors.append("template must be sft")
    for key in ("dataset_name", "model_name", "output_model_id"):
        if not str(request.get(key) or "").strip():
            errors.append(f"{key} is required for SageMaker training")
    if int(request.get("max_run_seconds") or 0) <= 0:
        errors.append("max_run_seconds must be positive")
    return errors


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    readiness = _safe_snapshot(build_aws_sagemaker_readiness_snapshot())
    request = _request_from_args(args, readiness)

    validation_errors = _validate_run_request(request)
    validation_errors.extend(validate_aws_sft_template_request(request))
    validation_errors = list(dict.fromkeys(validation_errors))

    script = ""
    script_error = None
    if not validation_errors:
        try:
            script = build_aws_sft_training_script(_template_config(request))
        except Exception as exc:  # pragma: no cover - defensive summary path
            script_error = str(exc)

    missing_markers = [marker for marker in _REQUIRED_MARKERS if marker not in script]
    missing_paths = [path for path in _REQUIRED_PATHS if path not in script]
    cost_estimate = asyncio.run(estimate_aws_sagemaker_job_cost(request))
    aws_ready = readiness.get("configured") is True
    aws_warning_only = bool(args.allow_missing_aws and not aws_ready)
    image_uri = request.get("image_uri")
    image_configured = bool(image_uri)
    image_warning_only = bool(args.allow_missing_image and not image_configured)

    template_ok = not validation_errors and script_error is None
    script_ok = bool(script) and not missing_markers and not missing_paths
    cost_ok = (
        cost_estimate.estimated_cost_usd is not None
        or cost_estimate.block_reason is not None
    )
    ok = (
        (aws_ready or aws_warning_only)
        and (image_configured or image_warning_only)
        and template_ok
        and script_ok
        and cost_ok
    )

    payload = {
        "ok": ok,
        "submitted_sagemaker_job": False,
        "uploaded_s3_objects": False,
        "fetched_cloudwatch_logs": False,
        "submission_tool_available": getattr(
            AwsSageMakerJobsTool, "__name__", "AwsSageMakerJobsTool"
        ),
        "aws_readiness": {
            **readiness,
            "warning_only": aws_warning_only,
        },
        "template_request": {
            "operation": request["operation"],
            "template": request["template"],
            "dataset_name": request["dataset_name"],
            "dataset_config": request.get("dataset_config"),
            "dataset_split": request["dataset_split"],
            "model_name": request["model_name"],
            "output_model_id": request["output_model_id"],
            "output_policy": request.get("output_policy"),
            "instance_type": request.get("instance_type"),
            "instance_count": request.get("instance_count"),
            "max_run_seconds": request["max_run_seconds"],
        },
        "template_validation": {
            "ok": template_ok,
            "errors": validation_errors,
            "script_error": script_error,
        },
        "script_checks": {
            "generated": bool(script),
            "required_markers_present": not missing_markers,
            "missing_required_markers": missing_markers,
            "required_paths_present": not missing_paths,
            "missing_required_paths": missing_paths,
        },
        "image": {
            "configured": image_configured,
            "warning_only": image_warning_only,
            "image_uri": image_uri,
        },
        "cost_estimate": {
            "estimated_cost_usd": cost_estimate.estimated_cost_usd,
            "billable": cost_estimate.billable,
            "block_reason": cost_estimate.block_reason,
            "label": cost_estimate.label,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
