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
import time
from datetime import datetime, timezone
from typing import Any, Callable

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
        hf_model_target = ""
        scale_warning = ""
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

        display_name = _slug(args.get("display_name") or f"liga-ml-{_now_suffix()}")
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

        env = {str(k): str(v) for k, v in (args.get("env") or {}).items()}
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
        elif (
            self.session is not None
            and getattr(self.session, "hf_token", None)
            and str(args.get("allow_insecure_hf_token_env") or "").lower()
            in {"1", "true", "yes"}
        ):
            env.setdefault("HF_TOKEN", self.session.hf_token)

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
        client = self._job_service_client(config["region"])
        job = await asyncio.to_thread(client.get_custom_job, name=job_name)
        state = _state_name(job.state)
        self._record_monitor_poll(job_name, state)
        return {
            "formatted": (
                "**Vertex AI job details:**\n\n"
                f"**Job:** `{job.name}`\n"
                f"**Display name:** {job.display_name}\n"
                f"**State:** {state}\n"
                f"**Create time:** {getattr(job, 'create_time', '')}\n"
                f"**Update time:** {getattr(job, 'update_time', '')}"
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
            "formatted": message,
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
