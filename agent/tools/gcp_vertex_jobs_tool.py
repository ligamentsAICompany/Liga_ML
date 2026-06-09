"""Google Cloud Vertex AI Jobs tool.

This mirrors the shape of ``hf_jobs`` while using Vertex AI Custom Training as
the execution backend. Intermediate artifacts live in GCS; successful training
scripts should still push final models to Hugging Face Hub for a common output
registry across backends.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import hf_hub_download

from agent.core.hf_tokens import resolve_hf_token
from agent.core.redact import SECRET_KEY_RE, redact_text
from agent.core.session import Event
from agent.training_templates.sft import SftTemplateConfig, build_sft_training_script
from agent.training_templates.validation import validate_sft_template_request
from agent.tools.types import ToolResult


DEFAULT_VERTEX_IMAGE = os.environ.get(
    "GCP_VERTEX_DEFAULT_IMAGE",
    "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310",
)
DEFAULT_MACHINE_TYPE = "n1-standard-8"
DEFAULT_REPLICA_COUNT = 1
DEFAULT_MONITOR_COOLDOWN_SECONDS = int(
    os.environ.get("GCP_VERTEX_MONITOR_COOLDOWN_SECONDS", "120")
)
GCP_REQUIRED_ENV_HELP = (
    "Set GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_REGION, GCS_BUCKET, "
    "VERTEX_AI_STAGING_BUCKET, and VERTEX_AI_OUTPUT_DIR on Cloud Run or in .env. "
    "Use an attached Cloud Run service account with Vertex AI, GCS, logging, "
    "and Artifact Registry permissions. Check /api/health/providers for readiness."
)
TERMINAL_JOB_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}
VALID_TRAINING_GOALS = {"smoke-test", "production", "agent-decide"}
VALID_OUTPUT_POLICIES = {"cloud-private", "hf-hub", "cloud-and-hf-hub"}
DEFAULT_PILOT_MAX_TRAIN_SAMPLES = 500
DEFAULT_PILOT_MAX_EVAL_SAMPLES = 50
LOW_BUDGET_PILOT_MAX_TRAIN_SAMPLES = 100
LOW_BUDGET_PILOT_MAX_EVAL_SAMPLES = 20
LARGE_DATASET_ROW_THRESHOLD = 10_000
GCS_JSONL_DATASET_SOURCE = "gcs_jsonl"
UPLOADED_GCS_DATASET_SOURCE = "uploaded-gcs"
ROOT_ERROR_PATTERNS = (
    "DatasetNotFoundError",
    "PermissionDenied",
    "PERMISSION_DENIED",
    "FileNotFoundError",
    "ValueError",
    "RuntimeError",
    "Traceback",
    "exit code 1",
)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9-]+", "-", value.strip()).strip("-").lower()
    return slug[:64] or "liga-ml-vertex-job"


def _now_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _required_config() -> tuple[dict[str, str], list[str]]:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    region = os.environ.get("GOOGLE_CLOUD_REGION", "").strip()
    bucket = os.environ.get("GCS_BUCKET", "").strip()
    missing = [
        name
        for name, value in {
            "GOOGLE_CLOUD_PROJECT": project,
            "GOOGLE_CLOUD_REGION": region,
            "GCS_BUCKET": bucket,
        }.items()
        if not value
    ]
    return {"project": project, "region": region, "bucket": bucket}, missing


def _gs_path(path_or_bucket: str, suffix: str | None = None) -> str:
    base = (
        path_or_bucket
        if path_or_bucket.startswith("gs://")
        else f"gs://{path_or_bucket}"
    )
    base = base.rstrip("/")
    if suffix:
        return f"{base}/{suffix.strip('/')}"
    return base


def _default_staging_bucket(bucket: str) -> str:
    return os.environ.get("VERTEX_AI_STAGING_BUCKET", "").strip() or _gs_path(
        bucket, "vertex-staging"
    )


def _default_output_dir(bucket: str) -> str:
    return os.environ.get("VERTEX_AI_OUTPUT_DIR", "").strip() or _gs_path(
        bucket, "vertex-outputs"
    )


def _env_list(env: dict[str, Any]) -> list[dict[str, str]]:
    return [{"name": str(k), "value": str(v)} for k, v in sorted(env.items())]


def _requires_hf_runtime_token(output_policy: str) -> bool:
    return output_policy in {"hf-hub", "cloud-and-hf-hub"}


def _resolve_vertex_hf_token(session: Any = None) -> str | None:
    return resolve_hf_token(
        getattr(session, "hf_token", None) if session is not None else None,
        os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        os.environ.get("HF_TOKEN"),
    )


def _optional_int(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _pilot_caps(args: dict[str, Any]) -> tuple[int, int]:
    budget = str(args.get("budget_preference") or "").strip().lower()
    machine = str(args.get("machine_type") or "").strip().lower()
    if budget == "low" or "standard-8" in machine:
        return LOW_BUDGET_PILOT_MAX_TRAIN_SAMPLES, LOW_BUDGET_PILOT_MAX_EVAL_SAMPLES
    return DEFAULT_PILOT_MAX_TRAIN_SAMPLES, DEFAULT_PILOT_MAX_EVAL_SAMPLES


def _apply_dataset_scale_guardrails(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    dataset_rows = _optional_int(args.get("dataset_rows")) or _optional_int(
        args.get("dataset_num_rows")
    )
    if dataset_rows is None or dataset_rows <= LARGE_DATASET_ROW_THRESHOLD:
        return args, ""
    if args.get("full_dataset_approved") is True:
        return args, (
            f"Dataset has about {dataset_rows:,} rows; full-dataset training was "
            "explicitly approved for this cost-bounded Vertex run."
        )
    if _optional_int(args.get("max_train_samples")) is not None:
        return args, (
            f"Dataset has about {dataset_rows:,} rows; using caller-provided "
            f"sample cap max_train_samples={args.get('max_train_samples')}."
        )

    train_cap, eval_cap = _pilot_caps(args)
    capped = {**args, "max_train_samples": train_cap}
    if _optional_int(capped.get("max_eval_samples")) is None:
        capped["max_eval_samples"] = eval_cap
    return capped, (
        f"Dataset has about {dataset_rows:,} rows; for this production pilot the "
        f"Vertex SFT template is capped at max_train_samples={train_cap} and "
        f"max_eval_samples={capped['max_eval_samples']}. Full dataset training "
        "requires a separate explicit approval."
    )


def _script_command(script: str, script_args: list[str] | None = None) -> list[str]:
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    args_json = json.dumps(script_args or [])
    runner = (
        "import base64,json,pathlib,runpy,sys;"
        "p=pathlib.Path('/tmp/liga_vertex_train.py');"
        f"p.write_text(base64.b64decode('{encoded}').decode('utf-8'));"
        f"sys.argv=[str(p)]+json.loads({args_json!r});"
        "runpy.run_path(str(p), run_name='__main__')"
    )
    return ["python", "-c", runner]


def _load_custom_job_cls():
    from google.cloud import aiplatform

    return aiplatform.CustomJob


def _init_aiplatform(project: str, region: str, staging_bucket: str) -> None:
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region, staging_bucket=staging_bucket)


def _load_job_service_client_cls():
    from google.cloud import aiplatform_v1

    return aiplatform_v1.JobServiceClient


def _load_logging_client_cls():
    from google.cloud import logging_v2

    return logging_v2.Client


def _load_storage_client_cls():
    from google.cloud import storage

    return storage.Client


def _parse_gs_uri(gcs_uri: str) -> tuple[str, str]:
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    bucket_and_prefix = gcs_uri.removeprefix("gs://").strip("/")
    bucket_name, _, prefix = bucket_and_prefix.partition("/")
    if not bucket_name or not prefix:
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    return bucket_name, prefix


def _has_training_fields(row: dict[str, Any]) -> bool:
    return any(
        key in row
        for key in (
            "text",
            "messages",
            "data",
            "prompt",
            "completion",
            "instruction",
            "output",
            "input",
            "response",
            "question",
            "answer",
        )
    )


def _validate_normalized_jsonl(path: str | Path) -> tuple[int, list[str]]:
    row_count = 0
    sample_keys: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Uploaded staged JSONL is not parseable at line {line_number}: {exc.msg}."
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Uploaded staged JSONL row {line_number} must be a JSON object."
                )
            if not _has_training_fields(row):
                raise ValueError(
                    "Uploaded staged JSONL rows must include normalized text/data fields."
                )
            sample_keys.update(str(key) for key in row.keys())
            row_count += 1
    if row_count <= 0:
        raise ValueError("Uploaded staged JSONL must contain at least one row.")
    return row_count, sorted(sample_keys)


def _extract_root_error_lines(text: str, *, limit: int = 8) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(pattern in line for pattern in ROOT_ERROR_PATTERNS):
            if line not in lines:
                lines.append(line[:1200])
        if len(lines) >= limit:
            break
    return lines


def _sanitize_failure_text(text: str) -> str:
    return redact_text(text)


class GcpVertexJobsTool:
    """Manage Vertex AI Custom Training jobs for Liga ML."""

    def __init__(
        self,
        *,
        session: Any = None,
        tool_call_id: str | None = None,
        custom_job_cls: Any | None = None,
        job_service_client_cls: Any | None = None,
        logging_client_cls: Any | None = None,
        init_aiplatform: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self.session = session
        self.tool_call_id = tool_call_id
        self.custom_job_cls = custom_job_cls
        self.job_service_client_cls = job_service_client_cls
        self.logging_client_cls = logging_client_cls
        self.init_aiplatform = init_aiplatform

    def _find_session_uploaded_dataset(
        self, args: dict[str, Any]
    ) -> dict[str, Any] | None:
        if self.session is None:
            return None
        uploads = [
            upload
            for upload in (getattr(self.session, "uploaded_datasets", []) or [])
            if isinstance(upload, dict)
        ]
        if not uploads:
            return None
        dataset_name = str(args.get("dataset_name") or "").strip()
        dataset_config = str(args.get("dataset_config") or "").strip()
        for upload in reversed(uploads):
            if (
                upload.get("supports_training") is False
                or upload.get("status") == "failed"
            ):
                continue
            if dataset_name and dataset_name != str(upload.get("repo_id") or ""):
                continue
            if dataset_config and dataset_config != str(
                upload.get("config_name") or ""
            ):
                continue
            return upload
        return None

    def _download_uploaded_train_jsonl(self, upload: dict[str, Any]) -> str:
        token = (
            getattr(self.session, "hf_token", None)
            if self.session is not None
            else None
        )
        if not token:
            raise ValueError(
                "A session Hugging Face token is required to stage the uploaded dataset."
            )
        repo_id = str(upload.get("repo_id") or "").strip()
        path_in_repo = str(upload.get("normalized_path_in_repo") or "").strip()
        if not repo_id or not path_in_repo:
            raise ValueError(
                "Uploaded dataset metadata is missing repo_id or normalized path."
            )
        return hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            token=token,
        )

    def _upload_file_to_gcs(self, source_path: str | Path, gcs_uri: str) -> None:
        bucket_name, blob_name = _parse_gs_uri(gcs_uri)
        storage_client_cls = _load_storage_client_cls()
        client = storage_client_cls()
        bucket = client.bucket(bucket_name)
        bucket.blob(blob_name).upload_from_filename(str(source_path))

    def _stage_uploaded_dataset_for_vertex(
        self,
        *,
        upload: dict[str, Any],
        display_name: str,
        bucket: str,
    ) -> dict[str, Any]:
        local_train_path = self._download_uploaded_train_jsonl(upload)
        row_count, sample_keys = _validate_normalized_jsonl(local_train_path)
        input_prefix = _gs_path(bucket, f"vertex-inputs/{display_name}")
        train_gcs_uri = _gs_path(input_prefix, "train.jsonl")
        metadata_gcs_uri = _gs_path(input_prefix, "metadata.json")
        metadata = {
            "dataset_source": UPLOADED_GCS_DATASET_SOURCE,
            "repo_id": upload.get("repo_id"),
            "config_name": upload.get("config_name"),
            "normalized_path_in_repo": upload.get("normalized_path_in_repo"),
            "source_format": upload.get("source_format"),
            "normalized_row_count": row_count,
            "sample_keys": sample_keys,
            "staged_train_uri": train_gcs_uri,
            "staged_at": datetime.now(timezone.utc).isoformat(),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", delete=False
        ) as handle:
            json.dump(metadata, handle, sort_keys=True)
            metadata_path = handle.name
        try:
            self._upload_file_to_gcs(local_train_path, train_gcs_uri)
            self._upload_file_to_gcs(metadata_path, metadata_gcs_uri)
        finally:
            try:
                Path(metadata_path).unlink(missing_ok=True)
            except Exception:
                pass
        return {
            "dataset_source": GCS_JSONL_DATASET_SOURCE,
            "display_dataset_source": UPLOADED_GCS_DATASET_SOURCE,
            "train_gcs_uri": train_gcs_uri,
            "staged_train_uri": train_gcs_uri,
            "metadata_gcs_uri": metadata_gcs_uri,
            "train_rows": row_count,
            "source_format": str(upload.get("source_format") or ""),
        }

    def _preflight_hf_dataset(self, args: dict[str, Any]) -> None:
        # Unit tests inject fake Vertex classes and should not call live Hub APIs.
        if self.custom_job_cls is not None:
            return
        from datasets import load_dataset

        dataset_name = str(args.get("dataset_name") or "").strip()
        if not dataset_name:
            return
        kwargs: dict[str, Any] = {
            "path": dataset_name,
            "split": str(args.get("dataset_split") or "train"),
        }
        if args.get("dataset_config"):
            kwargs["name"] = str(args["dataset_config"])
        token = (
            getattr(self.session, "hf_token", None)
            if self.session is not None
            else None
        )
        if token:
            kwargs["token"] = token
        load_dataset(**kwargs)

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        operation = str(params.get("operation", "")).lower().strip()
        if not operation:
            return self._error("'operation' parameter is required.")

        try:
            if operation == "run":
                return await self._run_job(params)
            if operation == "ps":
                return await self._list_jobs(params)
            if operation == "inspect":
                return await self._inspect_job(params)
            if operation == "cancel":
                return await self._cancel_job(params)
            if operation == "logs":
                return await self._get_logs(params)
            return self._error(
                f'Unknown operation: "{operation}". Available operations: run, ps, logs, inspect, cancel.'
            )
        except Exception as e:
            return self._error(f"Error executing {operation}: {e}")

    async def _run_job(self, args: dict[str, Any]) -> ToolResult:
        config, missing = _required_config()
        if missing:
            return self._error(
                "Missing Google Cloud configuration: "
                + ", ".join(missing)
                + ". "
                + GCP_REQUIRED_ENV_HELP
            )

        script = args.get("script")
        command = args.get("command")
        template = str(args.get("template") or "").strip().lower()
        training_goal = str(args.get("training_goal") or "agent-decide").strip()
        output_policy = str(args.get("output_policy") or "cloud-private").strip()
        trackio_mode = str(args.get("trackio_mode") or "disabled").strip()
        if training_goal not in VALID_TRAINING_GOALS:
            return self._error(
                "training_goal must be one of: smoke-test, production, agent-decide"
            )
        if output_policy not in VALID_OUTPUT_POLICIES:
            return self._error(
                "output_policy must be one of: cloud-private, hf-hub, cloud-and-hf-hub"
            )
        display_name = _slug(args.get("display_name") or f"liga-ml-{_now_suffix()}")
        hf_model_target = ""
        scale_warning = ""
        staged_dataset: dict[str, Any] | None = None
        if template:
            if script or command:
                return self._error(
                    "'template' cannot be combined with 'script' or 'command'."
                )
            if template != "sft":
                return self._error(
                    f"Unsupported template: {template}. Available templates: sft."
                )
            args = {
                **args,
                "output_policy": output_policy,
                "trackio_mode": trackio_mode,
            }
            validation_errors = validate_sft_template_request(args)
            if validation_errors:
                return self._error("; ".join(validation_errors))
            try:
                args, scale_warning = _apply_dataset_scale_guardrails(args)
                upload = self._find_session_uploaded_dataset(args)
                if upload is not None:
                    staged_dataset = self._stage_uploaded_dataset_for_vertex(
                        upload=upload,
                        display_name=display_name,
                        bucket=config["bucket"],
                    )
                    args = {
                        **args,
                        "dataset_source": GCS_JSONL_DATASET_SOURCE,
                        "train_gcs_uri": staged_dataset["train_gcs_uri"],
                        "staged_train_uri": staged_dataset["staged_train_uri"],
                        "train_rows": staged_dataset["train_rows"],
                        "source_format": staged_dataset["source_format"],
                    }
                else:
                    self._preflight_hf_dataset(args)
                template_config = SftTemplateConfig(
                    dataset_name=str(args.get("dataset_name") or ""),
                    dataset_config=args.get("dataset_config"),
                    dataset_split=str(args.get("dataset_split") or "train"),
                    eval_dataset_split=args.get("eval_dataset_split"),
                    validation_split_ratio=float(
                        args.get("validation_split_ratio") or 0.1
                    ),
                    model_name=str(args.get("model_name") or ""),
                    hub_model_id=str(args.get("hub_model_id") or ""),
                    training_goal=training_goal,
                    output_policy=output_policy,
                    task_type=str(args.get("task_type") or "sft"),
                    column_mapping=dict(args.get("column_mapping") or {}),
                    max_train_samples=args.get("max_train_samples"),
                    max_eval_samples=args.get("max_eval_samples"),
                    num_train_epochs=int(args.get("num_train_epochs") or 1),
                    max_length=int(args.get("max_length") or 1024),
                    learning_rate=float(args.get("learning_rate") or 2e-4),
                    per_device_train_batch_size=int(
                        args.get("per_device_train_batch_size") or 1
                    ),
                    gradient_accumulation_steps=int(
                        args.get("gradient_accumulation_steps") or 8
                    ),
                    trackio_mode=trackio_mode,
                    trackio_project=args.get("trackio_project"),
                    trackio_space_id=args.get("trackio_space_id"),
                    run_name=args.get("run_name"),
                    dataset_source=str(args.get("dataset_source") or "hf"),
                    train_gcs_uri=args.get("train_gcs_uri"),
                    staged_train_uri=args.get("staged_train_uri"),
                    train_rows=args.get("train_rows"),
                    source_format=args.get("source_format"),
                )
                script = build_sft_training_script(template_config)
                if output_policy in {"hf-hub", "cloud-and-hf-hub"}:
                    hf_model_target = template_config.hub_model_id
            except Exception as e:
                return self._error(str(e))

        if script and command:
            return self._error("'script' and 'command' are mutually exclusive.")
        if not script and not command:
            return self._error("Either 'script' or 'command' is required.")

        staging_bucket = args.get("staging_bucket") or _default_staging_bucket(
            config["bucket"]
        )
        output_root = args.get("output_dir") or _default_output_dir(config["bucket"])
        output_dir = _gs_path(output_root, display_name)

        run_command = (
            _script_command(str(script), args.get("script_args"))
            if script
            else [str(part) for part in command]
        )
        image = args.get("image") or DEFAULT_VERTEX_IMAGE
        machine_spec: dict[str, Any] = {
            "machine_type": args.get("machine_type") or DEFAULT_MACHINE_TYPE,
        }
        if accelerator_type := args.get("accelerator_type"):
            machine_spec["accelerator_type"] = accelerator_type
            machine_spec["accelerator_count"] = int(args.get("accelerator_count") or 1)

        env = {
            str(k): str(v)
            for k, v in (args.get("env") or {}).items()
            if not SECRET_KEY_RE.search(str(k))
        }
        env.setdefault("TRACKIO_MODE", trackio_mode)
        if args.get("trackio_project"):
            env.setdefault("TRACKIO_PROJECT", str(args["trackio_project"]))
        if args.get("trackio_space_id"):
            env.setdefault("TRACKIO_SPACE_ID", str(args["trackio_space_id"]))
        env.setdefault("AIP_MODEL_DIR", output_dir)
        env.setdefault("LIGA_ML_OUTPUT_DIR", output_dir)
        env.setdefault("LIGA_ML_TRAINING_GOAL", training_goal)
        env.setdefault("LIGA_ML_OUTPUT_POLICY", output_policy)
        env.setdefault("GOOGLE_CLOUD_PROJECT", config["project"])
        env.setdefault("GOOGLE_CLOUD_REGION", config["region"])
        if secret_resource := (
            args.get("hf_token_secret_resource")
            or os.environ.get("HF_TOKEN_SECRET_RESOURCE")
        ):
            env.setdefault("HF_TOKEN_SECRET_RESOURCE", str(secret_resource))
        if _requires_hf_runtime_token(output_policy):
            hf_token = _resolve_vertex_hf_token(self.session)
            if not hf_token:
                return self._error(
                    "Hugging Face token is required before launching Vertex AI "
                    f"with output_policy={output_policy}. Configure HF_TOKEN or "
                    "HUGGINGFACE_HUB_TOKEN on the server, then retry."
                )
            env.setdefault("HF_TOKEN", hf_token)
            env.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

        worker_pool_specs = [
            {
                "machine_spec": machine_spec,
                "replica_count": int(
                    args.get("replica_count") or DEFAULT_REPLICA_COUNT
                ),
                "container_spec": {
                    "image_uri": image,
                    "command": run_command,
                    "env": _env_list(env),
                },
            }
        ]

        custom_job_cls = self.custom_job_cls or _load_custom_job_cls()
        init = self.init_aiplatform or _init_aiplatform
        if self.custom_job_cls is None:
            init(config["project"], config["region"], staging_bucket)

        job = custom_job_cls(
            display_name=display_name,
            worker_pool_specs=worker_pool_specs,
            staging_bucket=staging_bucket,
        )
        service_account = args.get("service_account") or os.environ.get(
            "VERTEX_AI_SERVICE_ACCOUNT"
        )

        submit_kwargs: dict[str, Any] = {}
        if service_account:
            submit_kwargs["service_account"] = service_account
        if hasattr(job, "submit"):
            await asyncio.to_thread(job.submit, **submit_kwargs)
        else:
            await asyncio.to_thread(job.run, sync=False, **submit_kwargs)

        resource_name = _safe_job_resource_name(job)
        console_url = _vertex_console_url(
            config["project"], config["region"], resource_name
        )
        if self.session and self.tool_call_id:
            await self.session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": self.tool_call_id,
                        "tool": "gcp_vertex_jobs",
                        "state": "running",
                        "jobName": resource_name,
                        "jobUrl": console_url,
                        "outputDir": output_dir,
                    },
                )
            )

        hf_target_line = (
            f"**HF model target:** https://huggingface.co/{hf_model_target}\n"
            if hf_model_target
            else ""
        )
        scale_warning_line = (
            f"**Dataset scale guardrail:** {scale_warning}\n" if scale_warning else ""
        )
        staged_dataset_line = ""
        if staged_dataset:
            staged_dataset_line = (
                f"**Dataset source:** {staged_dataset['display_dataset_source']}\n"
                f"**Staged train URI:** {staged_dataset['staged_train_uri']}\n"
                f"**Staged train rows:** {staged_dataset['train_rows']}\n"
            )
        return {
            "formatted": (
                "Vertex AI job submitted.\n\n"
                f"**Job:** {resource_name}\n"
                f"**Display name:** {display_name}\n"
                f"**Region:** {config['region']}\n"
                f"**Image:** {image}\n"
                f"**Output dir:** {output_dir}\n"
                f"**Training goal:** {training_goal}\n"
                f"**Output policy:** {output_policy}\n"
                f"**Trackio mode:** {trackio_mode}\n"
                f"{scale_warning_line}"
                f"{staged_dataset_line}"
                f"{hf_target_line}"
                f"**Console:** {console_url}\n\n"
                "Use `gcp_vertex_jobs` with `operation='inspect'` or `operation='logs'` "
                "to monitor it, but do not poll tightly. Active Vertex job monitoring is "
                "rate-limited per session; wait for the cooldown message before checking "
                "the same job again. Training scripts should follow the selected "
                "output policy for final model storage."
            ),
            "totalResults": 1,
            "resultsShared": 1,
        }

    async def _list_jobs(self, args: dict[str, Any]) -> ToolResult:
        config, missing = _required_config()
        if missing:
            return self._error(
                "Missing Google Cloud configuration: "
                + ", ".join(missing)
                + ". "
                + GCP_REQUIRED_ENV_HELP
            )
        client = self._job_service_client(config["region"])
        parent = f"projects/{config['project']}/locations/{config['region']}"
        list_kwargs = {"parent": parent}
        if args.get("filter"):
            list_kwargs["filter"] = args.get("filter")
        jobs = await asyncio.to_thread(
            lambda: list(client.list_custom_jobs(**list_kwargs))
        )
        lines = ["**Vertex AI jobs:**"]
        for job in jobs[: int(args.get("limit") or 20)]:
            lines.append(
                f"- `{job.name}` — {job.display_name} — {_state_name(job.state)}"
            )
        if len(lines) == 1:
            lines.append("No Vertex AI custom jobs found.")
        return {
            "formatted": "\n".join(lines),
            "totalResults": len(jobs),
            "resultsShared": min(len(jobs), int(args.get("limit") or 20)),
        }

    async def _inspect_job(self, args: dict[str, Any]) -> ToolResult:
        job_name = args.get("job_name") or args.get("job_id")
        if not job_name:
            return self._error("job_name is required for inspect.")
        config, missing = _required_config()
        if missing:
            return self._error(
                "Missing Google Cloud configuration: "
                + ", ".join(missing)
                + ". "
                + GCP_REQUIRED_ENV_HELP
            )
        client = self._job_service_client(config["region"])
        job = await asyncio.to_thread(client.get_custom_job, name=job_name)
        state = _state_name(job.state)
        if state not in TERMINAL_JOB_STATES:
            if cooldown := self._monitor_cooldown_response(job_name):
                return cooldown
        self._record_monitor_poll(job_name, state)
        console_url = _vertex_console_url(config["project"], config["region"], job.name)
        failure_reason = ""
        logs_unavailable = False
        if state in {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
            failure_reason, logs_unavailable = await self._failure_reason_for_job(
                config, job
            )
            await self._emit_vertex_state_change(
                job,
                state=state,
                console_url=console_url,
                failure_reason=failure_reason,
                logs_unavailable=logs_unavailable,
            )
        failure_section = ""
        if failure_reason:
            failure_section = (
                f"\n\n**Failure reason:**\n\n```text\n{failure_reason}\n```"
            )
        elif state == "JOB_STATE_FAILED":
            failure_section = (
                "\n\n**Failure reason:** Vertex reported failure, but logs are not "
                f"available yet. View in Vertex AI: {console_url}"
            )
        return {
            "formatted": (
                "**Vertex AI job details:**\n\n"
                f"**Job:** `{job.name}`\n"
                f"**Display name:** {job.display_name}\n"
                f"**State:** {state}\n"
                f"**Create time:** {getattr(job, 'create_time', '')}\n"
                f"**Update time:** {getattr(job, 'update_time', '')}\n"
                f"**View in Vertex AI:** {console_url}"
                f"{failure_section}"
            ),
            "totalResults": 1,
            "resultsShared": 1,
        }

    async def _cancel_job(self, args: dict[str, Any]) -> ToolResult:
        job_name = args.get("job_name") or args.get("job_id")
        if not job_name:
            return self._error("job_name is required for cancel.")
        config, missing = _required_config()
        if missing:
            return self._error(
                "Missing Google Cloud configuration: "
                + ", ".join(missing)
                + ". "
                + GCP_REQUIRED_ENV_HELP
            )
        client = self._job_service_client(config["region"])
        await asyncio.to_thread(client.cancel_custom_job, name=job_name)
        return {
            "formatted": f"Cancel requested for Vertex AI job `{job_name}`.",
            "totalResults": 1,
            "resultsShared": 1,
        }

    async def _get_logs(self, args: dict[str, Any]) -> ToolResult:
        job_name = args.get("job_name") or args.get("job_id")
        if not job_name:
            return self._error("job_name is required for logs.")
        if cooldown := self._monitor_cooldown_response(job_name):
            return cooldown
        config, missing = _required_config()
        if missing:
            return self._error(
                "Missing Google Cloud configuration: "
                + ", ".join(missing)
                + ". "
                + GCP_REQUIRED_ENV_HELP
            )
        client_cls = self.logging_client_cls or _load_logging_client_cls()
        client = client_cls(project=config["project"])
        custom_job_id = job_name.rstrip("/").split("/")[-1]
        log_filter = (
            'resource.type="ml_job" '
            f'AND labels."ml.googleapis.com/job_id"="{custom_job_id}"'
        )
        limit = int(args.get("limit") or 100)
        entries = await asyncio.to_thread(
            lambda: list(client.list_entries(filter_=log_filter, page_size=limit))
        )
        self._record_monitor_poll(job_name, "JOB_STATE_MONITORING")
        lines = [str(getattr(entry, "payload", entry)) for entry in entries[-limit:]]
        return {
            "formatted": "**Vertex AI logs:**\n\n```text\n"
            + ("\n".join(lines) if lines else "No logs found yet.")
            + "\n```",
            "totalResults": len(entries),
            "resultsShared": len(lines),
        }

    async def _failure_reason_for_job(
        self, config: dict[str, str], job: Any
    ) -> tuple[str, bool]:
        lines: list[str] = []
        logs_unavailable = False
        try:
            client_cls = self.logging_client_cls or _load_logging_client_cls()
            client = client_cls(project=config["project"])
            custom_job_id = str(job.name).rstrip("/").split("/")[-1]
            log_filter = (
                'resource.type="ml_job" '
                f'AND labels."ml.googleapis.com/job_id"="{custom_job_id}"'
            )
            entries = await asyncio.to_thread(
                lambda: list(client.list_entries(filter_=log_filter, page_size=100))
            )
            log_text = "\n".join(
                str(getattr(entry, "payload", entry)) for entry in entries
            )
            lines.extend(_extract_root_error_lines(log_text))
            if not entries:
                logs_unavailable = True
        except Exception:
            logs_unavailable = True

        vertex_error = getattr(job, "error", None)
        vertex_message = str(getattr(vertex_error, "message", "") or "").strip()
        if vertex_message:
            lines.extend(_extract_root_error_lines(vertex_message))
            if not lines:
                lines.append(vertex_message[:1200])

        if not lines:
            logs_unavailable = True
            lines.append(
                "Vertex reported a terminal failure, but detailed logs are not available yet."
            )
        return _sanitize_failure_text("\n".join(dict.fromkeys(lines))), logs_unavailable

    async def _emit_vertex_state_change(
        self,
        job: Any,
        *,
        state: str,
        console_url: str,
        failure_reason: str = "",
        logs_unavailable: bool = False,
    ) -> None:
        if not self.session or not self.tool_call_id:
            return
        state_label = {
            "JOB_STATE_SUCCEEDED": "succeeded",
            "JOB_STATE_FAILED": "failed",
            "JOB_STATE_CANCELLED": "cancelled",
            "JOB_STATE_EXPIRED": "expired",
        }.get(state, state.lower())
        await self.session.send_event(
            Event(
                event_type="tool_state_change",
                data={
                    "tool_call_id": self.tool_call_id,
                    "tool": "gcp_vertex_jobs",
                    "state": state_label,
                    "jobName": getattr(job, "name", ""),
                    "jobUrl": console_url,
                    "failureReason": failure_reason,
                    "logsUnavailable": logs_unavailable,
                },
            )
        )

    def _job_service_client(self, region: str):
        client_cls = self.job_service_client_cls or _load_job_service_client_cls()
        endpoint = f"{region}-aiplatform.googleapis.com"
        return client_cls(client_options={"api_endpoint": endpoint})

    def _monitor_cooldown_response(self, job_name: str) -> ToolResult | None:
        """Prevent tight inspect/log polling loops for the same active job."""

        if self.session is None:
            return None
        cache = getattr(self.session, "_gcp_vertex_monitor_cache", None)
        if not isinstance(cache, dict):
            return None
        record = cache.get(str(job_name))
        if not record:
            return None

        last_state = str(record.get("state") or "")
        if last_state in TERMINAL_JOB_STATES:
            return None

        cooldown_seconds = int(
            os.environ.get(
                "GCP_VERTEX_MONITOR_COOLDOWN_SECONDS",
                str(DEFAULT_MONITOR_COOLDOWN_SECONDS),
            )
        )
        elapsed = time.monotonic() - float(record.get("monotonic", 0.0))
        remaining = int(max(0, cooldown_seconds - elapsed))
        if remaining <= 0:
            return None

        return {
            "formatted": (
                "Vertex job monitoring is rate-limited for this active job.\n\n"
                f"**Job:** `{job_name}`\n"
                f"**Last known state:** {last_state or 'unknown'}\n"
                f"Please wait about {remaining} seconds before calling "
                "`gcp_vertex_jobs` with `operation='inspect'` or `operation='logs'` "
                "for this same job again. Do not use sandbox `bash`/`sleep` for this "
                "Google Cloud wait; continue other work or summarize that the job is "
                "running in Vertex AI."
            ),
            "totalResults": 0,
            "resultsShared": 0,
        }

    def _record_monitor_poll(self, job_name: str, state: str) -> None:
        if self.session is None:
            return
        cache = getattr(self.session, "_gcp_vertex_monitor_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self.session, "_gcp_vertex_monitor_cache", cache)
        cache[str(job_name)] = {"monotonic": time.monotonic(), "state": state}

    @staticmethod
    def _error(message: str) -> ToolResult:
        return {
            "formatted": redact_text(message),
            "totalResults": 0,
            "resultsShared": 0,
            "isError": True,
        }


def _state_name(state: Any) -> str:
    if hasattr(state, "name"):
        return state.name
    return str(state)


def _safe_job_resource_name(job: Any) -> str:
    for attr in ("resource_name", "name"):
        try:
            value = getattr(job, attr, "")
        except Exception:
            continue
        if value:
            return str(value)
    return ""


def _vertex_console_url(project: str, region: str, resource_name: str) -> str:
    job_id = resource_name.rstrip("/").split("/")[-1] if resource_name else ""
    return (
        "https://console.cloud.google.com/vertex-ai/training/custom-jobs/"
        f"locations/{region}/customJobs/{job_id}?project={project}"
    )


GCP_VERTEX_JOBS_TOOL_SPEC = {
    "name": "gcp_vertex_jobs",
    "description": (
        "Execute ML training and fine-tuning jobs on Google Cloud Vertex AI Custom Training.\n\n"
        "Use this when the user asks for Google Cloud, GCP, Vertex AI, enterprise GCP infra, "
        "or GCS-backed training. Use hf_jobs when the user explicitly asks for Hugging Face Jobs.\n\n"
        "For normal supervised fine-tuning, prefer {'operation': 'run', 'template': 'sft', ...} "
        "instead of hand-writing an inline script. The SFT template uses the stable Liga ML "
        "runtime, conservative defaults, GCS output, and output-policy-aware final storage. "
        "Use raw script mode only for advanced workflows that the template does not support.\n\n"
        "Vertex AI run operations are billable and approval-gated. Include max_run_hours "
        "on run calls so approval and auto-approval budget checks can estimate a conservative "
        "upper bound. If max_run_hours is omitted, manual approval is required.\n\n"
        "Before submitting training jobs: inspect the dataset, choose template parameters, "
        "and run a tiny smoke test in the sandbox when possible. Respect training_goal "
        "(smoke-test, production, agent-decide) and output_policy (cloud-private, hf-hub, "
        "cloud-and-hf-hub). Vertex AI writes checkpoints and intermediate artifacts to GCS via "
        "AIP_MODEL_DIR/LIGA_ML_OUTPUT_DIR. For sensitive domains such as medical, finance, legal, "
        "insurance, government, or internal company data, recommend cloud-private unless the user "
        "explicitly chooses otherwise.\n\n"
        "Required deployment config: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_REGION, GCS_BUCKET. "
        "Cloud Run should use an attached service account with Vertex AI, GCS, and logging permissions.\n\n"
        "Monitoring discipline: after a run is submitted, call inspect/logs once, then wait. "
        "Do not call sandbox bash/read/write/edit just to sleep or poll a Vertex job. "
        "If this tool returns a monitoring cooldown message, stop polling that job until "
        "the cooldown expires and tell the user the job is still running in Vertex AI.\n\n"
        "Operations: run, ps, logs, inspect, cancel.\n"
        "Examples:\n"
        "{'operation': 'run', 'script': '/app/train.py', 'display_name': 'gst-sft', "
        "'machine_type': 'n1-standard-8', 'accelerator_type': 'NVIDIA_TESLA_T4', "
        "'accelerator_count': 1, 'max_run_hours': 2, "
        "'env': {'HF_MODEL_ID': 'ligaments/gst-model'}}\n"
        "{'operation': 'run', 'template': 'sft', 'display_name': 'medical-sft', "
        "'dataset_name': 'FreedomIntelligence/medical-o1-reasoning-SFT', "
        "'dataset_config': 'en', 'model_name': 'Qwen/Qwen2.5-0.5B-Instruct', "
        "'hub_model_id': 'ligaments-dev/medical-qwen2.5-0.5b-sft', "
        "'column_mapping': {'user': 'Question', 'assistant': ['Complex_CoT', 'Response']}, "
        "'training_goal': 'smoke-test', 'output_policy': 'cloud-private'}\n"
        "{'operation': 'inspect', 'job_name': 'projects/.../locations/.../customJobs/123'}"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["run", "ps", "logs", "inspect", "cancel"],
                "description": "Operation to execute.",
            },
            "script": {
                "type": "string",
                "description": (
                    "Python code or sandbox file path to execute on Vertex AI. "
                    "Mutually exclusive with command."
                ),
            },
            "template": {
                "type": "string",
                "enum": ["sft"],
                "description": "Use a stable Liga ML training template. Prefer 'sft' for normal supervised fine-tuning jobs.",
            },
            "dataset_name": {
                "type": "string",
                "description": "Hugging Face dataset id for template='sft'.",
            },
            "dataset_config": {
                "type": "string",
                "description": "Optional Hugging Face dataset config for template='sft'.",
            },
            "dataset_split": {
                "type": "string",
                "description": "Dataset split for template='sft'. Default: train.",
            },
            "dataset_source": {
                "type": "string",
                "enum": ["hf", "gcs_jsonl"],
                "description": (
                    "Dataset source for template='sft'. Defaults to hf. The backend "
                    "sets gcs_jsonl automatically for session-uploaded Vertex datasets "
                    "after staging normalized train.jsonl to GCS."
                ),
            },
            "train_gcs_uri": {
                "type": "string",
                "description": "GCS JSONL input URI for dataset_source='gcs_jsonl'.",
            },
            "staged_train_uri": {
                "type": "string",
                "description": "Alias/metadata for the staged uploaded train.jsonl GCS URI.",
            },
            "source_format": {
                "type": "string",
                "description": "Original uploaded source format when the dataset was staged from a session upload.",
            },
            "eval_dataset_split": {
                "type": "string",
                "description": (
                    "Optional explicit evaluation split for template='sft'. If omitted, "
                    "the template creates a deterministic validation split for train "
                    "datasets with at least 20 rows."
                ),
            },
            "validation_split_ratio": {
                "type": "number",
                "description": (
                    "Validation ratio used when eval_dataset_split is omitted and the "
                    "training split has at least 20 rows. Must be > 0 and < 1. Default: 0.1."
                ),
            },
            "model_name": {
                "type": "string",
                "description": "Base model id for template='sft'.",
            },
            "hub_model_id": {
                "type": "string",
                "description": (
                    "Final Hugging Face model repo id for template='sft'. Required when "
                    "output_policy is hf-hub or cloud-and-hf-hub; optional for cloud-private."
                ),
            },
            "training_goal": {
                "type": "string",
                "enum": ["smoke-test", "production", "agent-decide"],
                "description": (
                    "Training intent selected by the user. smoke-test means choose small "
                    "sample settings and short runtime when practical; production means "
                    "production-ready fine-tuning; agent-decide lets the agent choose."
                ),
            },
            "output_policy": {
                "type": "string",
                "enum": ["cloud-private", "hf-hub", "cloud-and-hf-hub"],
                "description": (
                    "Final model storage destination. cloud-private stores final artifacts in "
                    "GCS only and must not push to Hugging Face Hub. hf-hub pushes to Hub. "
                    "cloud-and-hf-hub saves to GCS and pushes to Hub. Default for Vertex: cloud-private."
                ),
            },
            "column_mapping": {
                "type": "object",
                "description": "Column mapping for template='sft', e.g. {'user': 'Question', 'assistant': ['Complex_CoT', 'Response']}.",
            },
            "max_train_samples": {
                "type": "integer",
                "description": "Optional cap for template='sft' smoke or small runs.",
            },
            "dataset_rows": {
                "type": "integer",
                "description": "Optional total dataset row count for scale guardrails. Datasets over 10,000 rows are pilot-capped unless full_dataset_approved=True.",
            },
            "dataset_num_rows": {
                "type": "integer",
                "description": "Alias for dataset_rows.",
            },
            "full_dataset_approved": {
                "type": "boolean",
                "description": "Set true only after explicit separate approval to train the full dataset when it exceeds 10,000 rows.",
            },
            "max_eval_samples": {
                "type": "integer",
                "description": "Optional cap for evaluation rows in template='sft'.",
            },
            "num_train_epochs": {
                "type": "integer",
                "description": "Epoch count for template='sft'. Default: 1.",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum sequence length for template='sft'. Default: 1024.",
            },
            "learning_rate": {
                "type": "number",
                "description": "Learning rate for template='sft'. Default: 2e-4.",
            },
            "per_device_train_batch_size": {
                "type": "integer",
                "description": "Per-device train batch size for template='sft'. Default: 1.",
            },
            "gradient_accumulation_steps": {
                "type": "integer",
                "description": "Gradient accumulation steps for template='sft'. Default: 8.",
            },
            "run_name": {
                "type": "string",
                "description": "Optional Trainer/Trackio run name for template='sft'.",
            },
            "trackio_project": {
                "type": "string",
                "description": "Trackio project name for template='sft'.",
            },
            "trackio_mode": {
                "type": "string",
                "enum": ["disabled", "reuse-existing", "create-if-allowed"],
                "description": "Trackio behavior. Default: disabled for Vertex jobs. Space creation is allowed only with create-if-allowed.",
            },
            "hf_token_secret_resource": {
                "type": "string",
                "description": "Optional Secret Manager resource containing an HF token, e.g. projects/<project>/secrets/<secret>/versions/latest. Preferred over raw env tokens.",
            },
            "allow_insecure_hf_token_env": {
                "type": "boolean",
                "description": "Emergency compatibility escape hatch. If true, pass the session HF token as a raw Vertex env var; avoid for production.",
            },
            "trackio_space_id": {
                "type": "string",
                "description": "Trackio dashboard Space id for template='sft'.",
            },
            "script_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional argv values passed to the Python script.",
            },
            "command": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Container command. Mutually exclusive with script.",
            },
            "image": {
                "type": "string",
                "description": "Container image URI. Defaults to a Vertex AI PyTorch GPU image.",
            },
            "display_name": {
                "type": "string",
                "description": "Human-readable Vertex AI job name.",
            },
            "machine_type": {
                "type": "string",
                "description": "Vertex AI machine type. Default: n1-standard-8.",
            },
            "accelerator_type": {
                "type": "string",
                "description": "Optional accelerator type, e.g. NVIDIA_TESLA_T4 or NVIDIA_TESLA_A100.",
            },
            "accelerator_count": {
                "type": "integer",
                "description": "Number of accelerators. Default: 1 when accelerator_type is set.",
            },
            "replica_count": {
                "type": "integer",
                "description": "Worker replica count. Default: 1.",
            },
            "env": {
                "type": "object",
                "description": "Environment variables for the Vertex AI job. HF_TOKEN is auto-included from the session when available.",
            },
            "staging_bucket": {
                "type": "string",
                "description": "Optional gs:// staging bucket. Defaults to VERTEX_AI_STAGING_BUCKET or gs://GCS_BUCKET/vertex-staging.",
            },
            "output_dir": {
                "type": "string",
                "description": "Optional gs:// output root. Defaults to VERTEX_AI_OUTPUT_DIR or gs://GCS_BUCKET/vertex-outputs.",
            },
            "service_account": {
                "type": "string",
                "description": "Optional Vertex AI runtime service account email.",
            },
            "max_run_hours": {
                "type": "number",
                "description": (
                    "Expected maximum runtime in hours, used for approval/cost "
                    "guardrails. Required for auto-approval; if omitted, manual "
                    "approval is required."
                ),
            },
            "job_name": {
                "type": "string",
                "description": "Full Vertex AI custom job resource name. Required for logs, inspect, cancel.",
            },
            "job_id": {
                "type": "string",
                "description": "Alias for job_name.",
            },
            "filter": {
                "type": "string",
                "description": "Optional Vertex AI list filter for ps.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum jobs/log entries to return.",
            },
        },
        "required": ["operation"],
    },
}


async def gcp_vertex_jobs_handler(
    arguments: dict[str, Any], session: Any = None, tool_call_id: str | None = None
) -> tuple[str, bool]:
    """Handler for agent tool router."""
    try:
        script = arguments.get("script", "")
        sandbox = getattr(session, "sandbox", None) if session else None
        if sandbox and script:
            from agent.tools.sandbox_tool import resolve_sandbox_script

            content, error = await resolve_sandbox_script(sandbox, script)
            if error:
                return error, False
            if content:
                arguments = {**arguments, "script": content}

        tool = GcpVertexJobsTool(session=session, tool_call_id=tool_call_id)
        result = await tool.execute(arguments)
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        return f"Error executing Vertex AI Jobs tool: {e}", False
