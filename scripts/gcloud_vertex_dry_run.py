#!/usr/bin/env python3
"""Non-billable local validation for the Vertex AI SFT path."""

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

from agent.core.cost_estimation import estimate_gcp_vertex_job_cost  # noqa: E402
from agent.core.gcp_readiness import build_gcp_vertex_readiness_snapshot  # noqa: E402
from agent.tools.gcp_vertex_jobs_tool import GcpVertexJobsTool  # noqa: E402
from agent.training_templates.sft import (  # noqa: E402
    SftTemplateConfig,
    build_sft_training_script,
)
from agent.training_templates.validation import validate_sft_template_request  # noqa: E402

_SENSITIVE_KEY_PARTS = ("token", "secret", "password", "private_key", "credential")
_FINAL_MARKERS = [
    "LIGA_TRAINING_STATUS=",
    "LIGA_PROVIDER=gcp-vertex",
    "LIGA_FINAL_MODEL_URL=",
    "LIGA_HUB_MODEL_ID=",
    "LIGA_GCS_OUTPUT_DIR=",
    "LIGA_EVAL_RESULT_JSON=",
    "LIGA_RESULT_FILE=",
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
        description="Validate a sample GCP Vertex SFT request without submitting a job."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--eval-dataset-split")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--hub-model-id", required=True)
    parser.add_argument("--max-run-hours", type=float, required=True)
    parser.add_argument("--machine-type", default="n1-standard-8")
    parser.add_argument("--accelerator-type")
    parser.add_argument("--accelerator-count", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--allow-missing-gcp",
        action="store_true",
        help="Treat missing local GCP readiness as a warning while still validating templates and cost.",
    )
    return parser


def _request_from_args(args: argparse.Namespace) -> dict[str, Any]:
    request: dict[str, Any] = {
        "operation": "run",
        "template": "sft",
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "eval_dataset_split": args.eval_dataset_split,
        "model_name": args.model_name,
        "hub_model_id": args.hub_model_id,
        "max_run_hours": args.max_run_hours,
        "machine_type": args.machine_type,
        "accelerator_type": args.accelerator_type,
        "accelerator_count": args.accelerator_count,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "num_train_epochs": args.num_train_epochs,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    return {key: value for key, value in request.items() if value is not None}


def _template_config(request: dict[str, Any]) -> SftTemplateConfig:
    return SftTemplateConfig(
        dataset_name=str(request["dataset_name"]),
        dataset_config=request.get("dataset_config"),
        dataset_split=str(request.get("dataset_split") or "train"),
        eval_dataset_split=request.get("eval_dataset_split"),
        model_name=str(request["model_name"]),
        hub_model_id=str(request["hub_model_id"]),
        max_train_samples=request.get("max_train_samples"),
        max_eval_samples=request.get("max_eval_samples"),
        num_train_epochs=int(request.get("num_train_epochs") or 1),
        max_length=int(request.get("max_length") or 1024),
        learning_rate=float(request.get("learning_rate") or 2e-4),
        per_device_train_batch_size=int(
            request.get("per_device_train_batch_size") or 1
        ),
        gradient_accumulation_steps=int(
            request.get("gradient_accumulation_steps") or 8
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = _request_from_args(args)
    readiness = _safe_snapshot(build_gcp_vertex_readiness_snapshot())
    validation_errors = validate_sft_template_request(request)
    script = ""
    script_error = None
    if not validation_errors:
        try:
            script = build_sft_training_script(_template_config(request))
        except Exception as exc:  # pragma: no cover - defensive summary path
            script_error = str(exc)

    missing_markers = [marker for marker in _FINAL_MARKERS if marker not in script]
    cost_estimate = asyncio.run(estimate_gcp_vertex_job_cost(request))
    gcp_ready = readiness.get("configured") is True
    gcp_warning_only = bool(args.allow_missing_gcp and not gcp_ready)
    cost_ok = cost_estimate.estimated_cost_usd is not None
    template_ok = not validation_errors and script_error is None
    script_ok = bool(script) and not missing_markers
    max_run_hours_ok = "max_run_hours" in request and request["max_run_hours"] > 0
    ok = (
        (gcp_ready or gcp_warning_only)
        and template_ok
        and script_ok
        and max_run_hours_ok
        and cost_ok
    )

    payload = {
        "ok": ok,
        "submitted_vertex_job": False,
        "submission_tool_available": GcpVertexJobsTool.__name__,
        "gcp_readiness": {
            **readiness,
            "warning_only": gcp_warning_only,
        },
        "template_request": {
            "operation": request["operation"],
            "template": request["template"],
            "dataset_name": request["dataset_name"],
            "dataset_config": request.get("dataset_config"),
            "dataset_split": request["dataset_split"],
            "model_name": request["model_name"],
            "hub_model_id": request["hub_model_id"],
            "machine_type": request["machine_type"],
            "accelerator_type": request.get("accelerator_type"),
            "accelerator_count": request.get("accelerator_count"),
            "max_run_hours": request["max_run_hours"],
        },
        "template_validation": {
            "ok": template_ok,
            "errors": validation_errors,
            "script_error": script_error,
        },
        "script_checks": {
            "generated": bool(script),
            "final_markers_present": not missing_markers,
            "missing_final_markers": missing_markers,
            "max_run_hours_present": max_run_hours_ok,
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
