"""Safe AWS SageMaker Jobs tool."""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from agent.core.aws_dataset_staging import stage_hf_dataset_to_s3
from agent.core.aws_readiness import build_aws_sagemaker_readiness_snapshot
from agent.core.cost_estimation import estimate_aws_sagemaker_job_cost
from agent.core.session import Event
from agent.tools.types import ToolResult
from agent.training_templates.aws_sft import (
    AwsSftTemplateConfig,
    build_aws_sft_training_script,
)
from agent.training_templates.aws_validation import validate_aws_sft_template_request

AWS_OUTPUT_POLICIES = {"aws-private", "cloud-private", "hf-hub", "cloud-and-hf-hub"}
AWS_REQUIRED_ENV_HELP = (
    "Set AWS_REGION, AWS_S3_BUCKET, and AWS_SAGEMAKER_ROLE_ARN in the backend "
    "environment. Optional defaults: AWS_S3_PREFIX, AWS_DEFAULT_INSTANCE_TYPE, "
    "AWS_DEFAULT_INSTANCE_COUNT, AWS_DEFAULT_MAX_RUN_SECONDS, AWS_OUTPUT_POLICY, "
    "AWS_SAGEMAKER_TRAINING_IMAGE_URI. AWS credentials must be discoverable by "
    "boto3's default provider chain. Check /api/health/providers for the current "
    "non-sensitive readiness snapshot."
)
DEFAULT_VOLUME_SIZE_GB = 30
DEFAULT_LOG_GROUP = "/aws/sagemaker/TrainingJobs"
MAX_LOG_EVENTS = 150
SAGEMAKER_STATUS_MAP = {
    "Completed": "succeeded",
    "Failed": "failed",
    "Stopped": "stopped",
    "Stopping": "stopping",
    "InProgress": "running",
}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9-]+", "-", value.strip()).strip("-").lower()
    return slug[:63] or "liga-ml-sagemaker-job"


def _now_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _positive_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _request_value(args: dict[str, Any], key: str, readiness: dict[str, Any]) -> Any:
    if args.get(key) not in (None, ""):
        return args[key]
    readiness_key = {
        "s3_bucket": "s3_bucket",
        "s3_prefix": "s3_prefix",
        "role_arn": "sagemaker_role_arn",
        "image_uri": "training_image_uri",
        "instance_type": "default_instance_type",
        "instance_count": "default_instance_count",
        "max_run_seconds": "default_max_run_seconds",
        "output_policy": "output_policy",
    }.get(key)
    return readiness.get(readiness_key) if readiness_key else None


def map_sagemaker_status(status: str | None) -> str:
    if not status:
        return "unknown"
    return SAGEMAKER_STATUS_MAP.get(status, status.lower())


def _load_sagemaker_client(region: str | None = None):
    import boto3

    return boto3.client("sagemaker", region_name=region)


def _load_logs_client(region: str | None = None):
    import boto3

    return boto3.client("logs", region_name=region)


def _load_s3_client(region: str | None = None):
    import boto3

    return boto3.client("s3", region_name=region)


def _split_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket_and_key = uri.removeprefix("s3://")
    bucket, _, key = bucket_and_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def _s3_dir_for_file(uri: str) -> str:
    bucket, key = _split_s3_uri(uri)
    prefix = key.rsplit("/", 1)[0]
    return f"s3://{bucket}/{prefix}/"


def _console_url(region: str, job_name: str) -> str:
    encoded_job = quote(job_name, safe="")
    return (
        f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}"
        f"#/jobs/{encoded_job}"
    )


def _cloudwatch_logs_url(region: str, job_name: str) -> str:
    encoded_group = quote(DEFAULT_LOG_GROUP, safe="")
    encoded_prefix = quote(job_name, safe="")
    return (
        f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}"
        f"#logsV2:log-groups/log-group/{encoded_group}/log-events/{encoded_prefix}"
    )


def _format_time(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _safe_log_message(message: Any) -> str:
    text = str(message)
    redactions = [
        (r"(?i)(aws_secret_access_key\s*=\s*)\S+", r"\1[redacted]"),
        (r"(?i)(aws_session_token\s*=\s*)\S+", r"\1[redacted]"),
        (r"(?i)(hf_token\s*=\s*)\S+", r"\1[redacted]"),
        (r"(?i)(huggingface_hub_token\s*=\s*)\S+", r"\1[redacted]"),
        (r"AKIA[0-9A-Z]{16}", "[redacted-access-key]"),
    ]
    for pattern, replacement in redactions:
        text = re.sub(pattern, replacement, text)
    return text


class AwsSageMakerJobsTool:
    """Manage controlled AWS SageMaker training job submission for Liga ML."""

    def __init__(
        self,
        *,
        session: Any = None,
        tool_call_id: str | None = None,
        sagemaker_client: Any | None = None,
        logs_client: Any | None = None,
        s3_client: Any | None = None,
    ) -> None:
        self.session = session
        self.tool_call_id = tool_call_id
        self.sagemaker_client = sagemaker_client
        self.logs_client = logs_client
        self.s3_client = s3_client

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        operation = str(params.get("operation", "")).lower().strip()
        if not operation:
            return self._error("'operation' parameter is required.")
        if operation == "run":
            return await self._run_job(params)
        if operation == "ps":
            return await self._list_jobs()
        if operation == "inspect":
            return await self._inspect_job(params)
        if operation == "logs":
            return await self._logs(params)
        if operation == "cancel":
            return await self._cancel_job(params)
        return self._error(
            f'Unknown operation: "{operation}". Available operations: run, ps, logs, inspect, cancel.'
        )

    async def _run_job(self, args: dict[str, Any]) -> ToolResult:
        readiness = build_aws_sagemaker_readiness_snapshot()
        if not readiness.get("configured"):
            return self._missing_config_error(readiness)

        errors = self._validate_run_request(args)
        errors.extend(validate_aws_sft_template_request(args))
        if errors:
            return self._error("; ".join(dict.fromkeys(errors)))

        region = str(readiness.get("region") or "us-east-1")
        job_name = _slug(
            str(args.get("job_name") or f"liga-ml-sagemaker-{_now_suffix()}")
        )
        s3_bucket = str(_request_value(args, "s3_bucket", readiness) or "")
        s3_prefix = str(_request_value(args, "s3_prefix", readiness) or "liga-ml")
        role_arn = str(_request_value(args, "role_arn", readiness) or "")
        image_uri = str(_request_value(args, "image_uri", readiness) or "").strip()
        if not image_uri:
            return self._error(
                "image_uri is required before staging or submitting SageMaker jobs. "
                "Pass image_uri on the tool call or set AWS_SAGEMAKER_TRAINING_IMAGE_URI."
            )

        instance_type = str(
            _request_value(args, "instance_type", readiness) or "ml.g5.xlarge"
        )
        instance_count = _positive_int(
            _request_value(args, "instance_count", readiness), 1
        )
        max_run_seconds = _positive_int(
            _request_value(args, "max_run_seconds", readiness), 3600
        )
        volume_size_gb = _positive_int(
            args.get("volume_size_gb"), DEFAULT_VOLUME_SIZE_GB
        )
        if instance_count is None or max_run_seconds is None or volume_size_gb is None:
            return self._error(
                "instance_count, max_run_seconds, and volume_size_gb must be positive integers."
            )

        output_policy = str(
            _request_value(args, "output_policy", readiness) or "aws-private"
        )
        if output_policy not in AWS_OUTPUT_POLICIES:
            return self._error(
                "output_policy must be one of: "
                + ", ".join(sorted(AWS_OUTPUT_POLICIES))
                + "."
            )

        dataset_name = str(args.get("dataset_name") or "").strip()
        dataset_config = (
            str(args.get("dataset_config")).strip()
            if args.get("dataset_config") not in (None, "")
            else None
        )
        dataset_split = str(args.get("dataset_split") or "train").strip() or "train"
        cost = await estimate_aws_sagemaker_job_cost(
            {
                "operation": "run",
                "instance_type": instance_type,
                "instance_count": instance_count,
                "max_run_seconds": max_run_seconds,
            }
        )
        cost_line = (
            f"**Conservative cost estimate:** ${cost.estimated_cost_usd:.2f}\n"
            if cost.estimated_cost_usd is not None
            else f"**Conservative cost estimate:** unavailable ({cost.block_reason})\n"
        )

        hf_token = self._hf_token()
        try:
            staged = await stage_hf_dataset_to_s3(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                dataset_split=dataset_split,
                s3_bucket=s3_bucket,
                s3_prefix=s3_prefix,
                job_name=job_name,
                session_id=getattr(self.session, "session_id", None),
                hf_token=hf_token,
                s3_client=self.s3_client,
            )
        except Exception as exc:
            return self._error(str(exc))

        try:
            script = build_aws_sft_training_script(self._template_config(args))
        except Exception as exc:
            return self._error(str(exc))

        script_key = f"{s3_prefix.strip('/')}/jobs/{job_name}/code/train.py"
        script_s3_uri = f"s3://{s3_bucket}/{script_key}"
        try:
            await self._upload_training_script(
                bucket=s3_bucket,
                key=script_key,
                script=script,
                region=region,
            )
        except Exception as exc:
            return self._error(
                f"Could not upload SageMaker training script to S3: {exc}"
            )

        console_url = _console_url(region, job_name)
        cloudwatch_logs_url = _cloudwatch_logs_url(region, job_name)
        s3_model_artifact = staged.s3_output_uri.rstrip("/") + "/model.tar.gz"
        train_input_uri = _s3_dir_for_file(staged.s3_train_uri)

        request = {
            "TrainingJobName": job_name,
            "RoleArn": role_arn,
            "AlgorithmSpecification": {
                "TrainingImage": image_uri,
                "TrainingInputMode": "File",
            },
            "InputDataConfig": [
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": train_input_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "ContentType": "application/jsonl",
                    "InputMode": "File",
                },
                {
                    "ChannelName": "code",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": script_s3_uri.rsplit("/", 1)[0] + "/",
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "InputMode": "File",
                },
            ],
            "OutputDataConfig": {"S3OutputPath": staged.s3_output_uri},
            "ResourceConfig": {
                "InstanceType": instance_type,
                "InstanceCount": instance_count,
                "VolumeSizeInGB": volume_size_gb,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": max_run_seconds},
            "Environment": {
                "LIGA_PROVIDER": "aws-sagemaker",
                "LIGA_AWS_TRAINING_JOB_NAME": job_name,
                "LIGA_AWS_REGION": region,
                "LIGA_S3_MODEL_ARTIFACT": s3_model_artifact,
                "LIGA_S3_OUTPUT_DIR": staged.s3_output_uri,
                "LIGA_CLOUDWATCH_LOGS_URL": cloudwatch_logs_url,
                "LIGA_OUTPUT_POLICY": output_policy,
            },
            "HyperParameters": {
                "sagemaker_program": "train.py",
                "sagemaker_submit_directory": script_s3_uri,
                "model_name": str(args.get("model_name") or ""),
                "output_model_id": str(args.get("output_model_id") or ""),
                "output_policy": output_policy,
            },
        }

        sagemaker_client = self.sagemaker_client or _load_sagemaker_client(region)
        try:
            await asyncio.to_thread(sagemaker_client.create_training_job, **request)
        except Exception as exc:
            return self._error(f"Could not submit SageMaker training job: {exc}")

        await self._emit_running_state(
            job_name=job_name,
            job_url=console_url,
            s3_train_uri=staged.s3_train_uri,
            s3_output_uri=staged.s3_output_uri,
            s3_model_artifact=s3_model_artifact,
            cloudwatch_logs_url=cloudwatch_logs_url,
            region=region,
        )

        metadata = {
            "state": "running",
            "job_name": job_name,
            "region": region,
            "s3_train_uri": staged.s3_train_uri,
            "s3_output_uri": staged.s3_output_uri,
            "s3_model_artifact": s3_model_artifact,
            "script_s3_uri": script_s3_uri,
            "console_url": console_url,
            "cloudwatch_logs_url": cloudwatch_logs_url,
            "instance_type": instance_type,
            "instance_count": instance_count,
            "max_run_seconds": max_run_seconds,
            "output_policy": output_policy,
            "estimated_cost_usd": cost.estimated_cost_usd,
        }

        return {
            "formatted": (
                "AWS SageMaker training job submitted.\n\n"
                f"**Job name:** `{job_name}`\n"
                f"**Region:** `{region}`\n"
                f"**S3 train URI:** `{staged.s3_train_uri}`\n"
                f"**S3 output URI:** `{staged.s3_output_uri}`\n"
                f"**S3 model artifact:** `{s3_model_artifact}`\n"
                f"**Training script:** `{script_s3_uri}`\n"
                f"**Instance type:** `{instance_type}`\n"
                f"**Instance count:** `{instance_count}`\n"
                f"**Max run seconds:** `{max_run_seconds}`\n"
                f"**Output policy:** `{output_policy}`\n"
                f"{cost_line}"
                f"**SageMaker console:** {console_url}\n"
                f"**CloudWatch logs:** {cloudwatch_logs_url}\n\n"
                "The job was submitted after approval through the existing "
                "SageMaker run guardrails. Use `aws_sagemaker_jobs` inspect/logs "
                "for live status, CloudWatch logs, and final artifact details."
            ),
            "totalResults": 1,
            "resultsShared": 1,
            "metadata": metadata,
        }

    async def _list_jobs(self) -> ToolResult:
        readiness = build_aws_sagemaker_readiness_snapshot()
        if not readiness.get("configured"):
            return self._missing_config_error(readiness)
        region = str(readiness.get("region") or "us-east-1")
        sagemaker_client = self.sagemaker_client or _load_sagemaker_client(region)
        try:
            response = await asyncio.to_thread(
                sagemaker_client.list_training_jobs,
                SortBy="CreationTime",
                SortOrder="Descending",
                MaxResults=20,
            )
        except Exception as exc:
            return self._error(f"Could not list SageMaker training jobs: {exc}")

        summaries = response.get("TrainingJobSummaries") or []
        if not summaries:
            return {
                "formatted": (
                    "No recent AWS SageMaker training jobs found.\n\n"
                    f"**Region:** `{region}`"
                ),
                "totalResults": 0,
                "resultsShared": 0,
            }
        rows = [
            "| Job name | Status | Created |",
            "| --- | --- | --- |",
        ]
        for summary in summaries:
            name = summary.get("TrainingJobName") or ""
            status = summary.get("TrainingJobStatus") or ""
            created = _format_time(summary.get("CreationTime"))
            rows.append(f"| `{name}` | `{status}` | `{created}` |")
        return {
            "formatted": (
                "Recent AWS SageMaker training jobs.\n\n"
                f"**Region:** `{region}`\n\n" + "\n".join(rows)
            ),
            "totalResults": len(summaries),
            "resultsShared": len(summaries),
        }

    async def _inspect_job(self, args: dict[str, Any]) -> ToolResult:
        job_name = str(args.get("job_name") or args.get("job_id") or "").strip()
        if not job_name:
            return self._error("job_name is required for inspect.")
        readiness = build_aws_sagemaker_readiness_snapshot()
        if not readiness.get("configured"):
            return self._missing_config_error(readiness)
        region = str(readiness.get("region") or "us-east-1")
        sagemaker_client = self.sagemaker_client or _load_sagemaker_client(region)
        try:
            job = await asyncio.to_thread(
                sagemaker_client.describe_training_job,
                TrainingJobName=job_name,
            )
        except Exception as exc:
            return self._error(
                f"Could not inspect SageMaker training job `{job_name}`: {exc}"
            )

        status = str(job.get("TrainingJobStatus") or "")
        output_uri = (job.get("OutputDataConfig") or {}).get("S3OutputPath")
        model_artifact = (job.get("ModelArtifacts") or {}).get("S3ModelArtifacts")
        console_url = _console_url(region, job_name)
        logs_url = _cloudwatch_logs_url(region, job_name)
        rows = [
            ("TrainingJobName", job.get("TrainingJobName") or job_name),
            ("TrainingJobStatus", status),
            ("SecondaryStatus", job.get("SecondaryStatus")),
            ("TrainingStartTime", _format_time(job.get("TrainingStartTime"))),
            ("TrainingEndTime", _format_time(job.get("TrainingEndTime"))),
            (
                "ResourceConfig",
                _json_value(job.get("ResourceConfig"))
                if job.get("ResourceConfig")
                else None,
            ),
            ("S3OutputPath", output_uri),
            ("S3ModelArtifacts", model_artifact),
            ("FailureReason", job.get("FailureReason")),
            ("SageMaker console", console_url),
            ("CloudWatch logs", logs_url),
        ]
        formatted_rows = [
            f"**{label}:** {value}" for label, value in rows if value not in (None, "")
        ]
        await self._emit_job_state(
            state=map_sagemaker_status(status),
            job_name=job_name,
            job_url=console_url,
            s3_output_uri=output_uri,
            s3_model_artifact=model_artifact,
            cloudwatch_logs_url=logs_url,
            region=region,
        )
        return {
            "formatted": "AWS SageMaker training job details.\n\n"
            + "\n".join(formatted_rows),
            "totalResults": 1,
            "resultsShared": 1,
        }

    async def _logs(self, args: dict[str, Any]) -> ToolResult:
        job_name = str(args.get("job_name") or args.get("job_id") or "").strip()
        if not job_name:
            return self._error("job_name is required for logs.")
        readiness = build_aws_sagemaker_readiness_snapshot()
        if not readiness.get("configured"):
            return self._missing_config_error(readiness)
        region = str(readiness.get("region") or "us-east-1")
        log_group = str(args.get("log_group") or DEFAULT_LOG_GROUP)
        logs_client = self.logs_client or _load_logs_client(region)
        limit = _positive_int(args.get("limit"), MAX_LOG_EVENTS) or MAX_LOG_EVENTS
        limit = min(max(limit, 1), 200)
        try:
            streams_response = await asyncio.to_thread(
                logs_client.describe_log_streams,
                logGroupName=log_group,
                logStreamNamePrefix=job_name,
                orderBy="LastEventTime",
                descending=True,
                limit=5,
            )
        except Exception as exc:
            return self._error(
                f"Could not discover CloudWatch log streams for `{job_name}`: {exc}"
            )

        streams = streams_response.get("logStreams") or []
        if not streams:
            return {
                "formatted": (
                    f"No CloudWatch log streams found yet for `{job_name}`.\n\n"
                    f"**Log group:** `{log_group}`\n"
                    f"**CloudWatch logs:** {_cloudwatch_logs_url(region, job_name)}"
                ),
                "totalResults": 0,
                "resultsShared": 0,
            }

        events: list[dict[str, Any]] = []
        for stream in streams:
            stream_name = stream.get("logStreamName")
            if not stream_name:
                continue
            try:
                response = await asyncio.to_thread(
                    logs_client.get_log_events,
                    logGroupName=log_group,
                    logStreamName=stream_name,
                    limit=limit,
                    startFromHead=False,
                )
            except Exception as exc:
                return self._error(
                    f"Could not fetch CloudWatch log events for `{job_name}`: {exc}"
                )
            for event in response.get("events") or []:
                events.append({**event, "logStreamName": stream_name})
                if len(events) >= limit:
                    break
            if len(events) >= limit:
                break

        if not events:
            return {
                "formatted": (
                    f"No CloudWatch log events found yet for `{job_name}`.\n\n"
                    f"**Log group:** `{log_group}`\n"
                    f"**CloudWatch logs:** {_cloudwatch_logs_url(region, job_name)}"
                ),
                "totalResults": 0,
                "resultsShared": 0,
            }

        lines = [
            f"CloudWatch logs for `{job_name}`.",
            "",
            f"**Log group:** `{log_group}`",
            f"**CloudWatch logs:** {_cloudwatch_logs_url(region, job_name)}",
            "",
            "```text",
        ]
        for event in events[-limit:]:
            timestamp = event.get("timestamp")
            prefix = f"{timestamp} " if timestamp is not None else ""
            lines.append(prefix + _safe_log_message(event.get("message", "")))
        lines.append("```")
        return {
            "formatted": "\n".join(lines),
            "totalResults": len(events),
            "resultsShared": len(events),
        }

    async def _cancel_job(self, args: dict[str, Any]) -> ToolResult:
        job_name = str(args.get("job_name") or args.get("job_id") or "").strip()
        if not job_name:
            return self._error("job_name is required for cancel.")
        readiness = build_aws_sagemaker_readiness_snapshot()
        if not readiness.get("configured"):
            return self._missing_config_error(readiness)
        region = str(readiness.get("region") or "us-east-1")
        sagemaker_client = self.sagemaker_client or _load_sagemaker_client(region)
        try:
            await asyncio.to_thread(
                sagemaker_client.stop_training_job,
                TrainingJobName=job_name,
            )
        except Exception as exc:
            return self._error(
                f"Could not cancel SageMaker training job `{job_name}`: {exc}"
            )
        console_url = _console_url(region, job_name)
        return {
            "formatted": (
                f"Cancellation requested for AWS SageMaker training job `{job_name}`.\n\n"
                f"**SageMaker console:** {console_url}"
            ),
            "totalResults": 1,
            "resultsShared": 1,
        }

    @staticmethod
    def _validate_run_request(args: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        template = str(args.get("template") or "sft").strip().lower()
        if template != "sft":
            errors.append(
                "Unsupported template: " + template + ". Available templates: sft"
            )
        for key in ("dataset_name", "model_name", "output_model_id"):
            if not str(args.get(key) or "").strip():
                errors.append(f"{key} is required for SageMaker training")
        return errors

    @staticmethod
    def _template_config(args: dict[str, Any]) -> AwsSftTemplateConfig:
        return AwsSftTemplateConfig(
            dataset_split=str(args.get("dataset_split") or "train"),
            model_name=str(args.get("model_name") or ""),
            output_model_id=str(args.get("output_model_id") or ""),
            output_policy=str(args.get("output_policy") or "aws-private"),
            hub_model_id=args.get("hub_model_id"),
            column_mapping=dict(args.get("column_mapping") or {}),
            max_train_samples=args.get("max_train_samples"),
            max_eval_samples=args.get("max_eval_samples"),
            validation_split_ratio=float(args.get("validation_split_ratio") or 0.1),
            num_train_epochs=int(args.get("num_train_epochs") or 1),
            max_length=int(args.get("max_length") or 1024),
            learning_rate=float(args.get("learning_rate") or 2e-4),
            per_device_train_batch_size=int(
                args.get("per_device_train_batch_size") or 1
            ),
            gradient_accumulation_steps=int(
                args.get("gradient_accumulation_steps") or 8
            ),
            trackio_project=args.get("trackio_project"),
            trackio_space_id=args.get("trackio_space_id"),
            run_name=args.get("run_name"),
        )

    def _hf_token(self) -> str | None:
        session_token = getattr(self.session, "hf_token", None)
        if session_token:
            return str(session_token)
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    async def _upload_training_script(
        self, *, bucket: str, key: str, script: str, region: str | None = None
    ) -> None:
        s3_client = self.s3_client or _load_s3_client(region)
        await asyncio.to_thread(
            s3_client.put_object,
            Bucket=bucket,
            Key=key,
            Body=script.encode("utf-8"),
            ContentType="text/x-python; charset=utf-8",
        )

    async def _emit_running_state(
        self,
        *,
        job_name: str,
        job_url: str,
        s3_train_uri: str,
        s3_output_uri: str,
        s3_model_artifact: str,
        cloudwatch_logs_url: str,
        region: str,
    ) -> None:
        await self._emit_job_state(
            state="running",
            job_name=job_name,
            job_url=job_url,
            s3_train_uri=s3_train_uri,
            s3_output_uri=s3_output_uri,
            s3_model_artifact=s3_model_artifact,
            cloudwatch_logs_url=cloudwatch_logs_url,
            region=region,
        )

    async def _emit_job_state(
        self,
        *,
        state: str,
        job_name: str,
        job_url: str,
        s3_train_uri: str | None = None,
        s3_output_uri: str | None = None,
        s3_model_artifact: str | None = None,
        cloudwatch_logs_url: str | None = None,
        region: str | None = None,
    ) -> None:
        if self.session is None or not self.tool_call_id:
            return
        send_event = getattr(self.session, "send_event", None)
        if send_event is None:
            return
        try:
            await send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": self.tool_call_id,
                        "tool": "aws_sagemaker_jobs",
                        "state": state,
                        "jobName": job_name,
                        "jobUrl": job_url,
                        **({"s3TrainUri": s3_train_uri} if s3_train_uri else {}),
                        **({"s3OutputUri": s3_output_uri} if s3_output_uri else {}),
                        **(
                            {"s3ModelArtifact": s3_model_artifact}
                            if s3_model_artifact
                            else {}
                        ),
                        **(
                            {"cloudWatchLogsUrl": cloudwatch_logs_url}
                            if cloudwatch_logs_url
                            else {}
                        ),
                        **({"region": region} if region else {}),
                    },
                )
            )
        except Exception:
            return

    @staticmethod
    def _missing_config_error(readiness: dict[str, Any]) -> ToolResult:
        missing = readiness.get("missing_env") or []
        details = []
        if missing:
            details.append(
                "Missing required AWS configuration: " + ", ".join(missing) + "."
            )
        if not readiness.get("credentials_detected"):
            details.append("AWS credentials were not detected by boto3.")
        for error in readiness.get("errors") or []:
            if error not in details:
                details.append(str(error))
        return AwsSageMakerJobsTool._error(" ".join(details + [AWS_REQUIRED_ENV_HELP]))

    @staticmethod
    def _error(message: str) -> ToolResult:
        return {
            "formatted": message,
            "totalResults": 0,
            "resultsShared": 0,
            "isError": True,
        }


AWS_SAGEMAKER_JOBS_TOOL_SPEC = {
    "name": "aws_sagemaker_jobs",
    "description": (
        "Submit controlled AWS SageMaker AI training jobs after approval.\n\n"
        "Use this when the session provider is AWS SageMaker AI or the user asks "
        "for AWS/SageMaker training, fine-tuning, SFT, model adaptation, or cloud "
        "compute. For normal SFT, prefer {'operation': 'run', 'template': 'sft', ...}; "
        "the tool stages normalized Hugging Face datasets to S3, uploads the stable "
        "AWS SFT script, and can call SageMaker create_training_job when readiness "
        "and training image config are present.\n\n"
        "Run and cancel operations are approval-gated and billable. Include "
        "max_run_seconds so approval and auto-approval budget checks can estimate "
        "a conservative upper bound. A training image is required via image_uri or "
        "AWS_SAGEMAKER_TRAINING_IMAGE_URI; the tool never guesses a live image.\n\n"
        "Operations: run, ps, logs, inspect, cancel."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["run", "ps", "inspect", "logs", "cancel"],
                "description": "Operation to execute.",
            },
            "template": {
                "type": "string",
                "enum": ["sft"],
                "description": "Stable Liga ML SageMaker training template.",
            },
            "dataset_name": {
                "type": "string",
                "description": "Hugging Face dataset id to load and stage to S3.",
            },
            "dataset_config": {
                "type": "string",
                "description": "Optional dataset config.",
            },
            "dataset_split": {
                "type": "string",
                "description": "Dataset split. Default: train.",
            },
            "model_name": {"type": "string", "description": "Base model id for SFT."},
            "output_model_id": {
                "type": "string",
                "description": "Output model/artifact id for the training result.",
            },
            "hub_model_id": {
                "type": "string",
                "description": "Optional Hugging Face model repo id for hub output policies.",
            },
            "output_policy": {
                "type": "string",
                "enum": ["aws-private", "cloud-private", "hf-hub", "cloud-and-hf-hub"],
                "description": "Artifact policy. Use cloud-private/aws-private for S3-only output. Default: aws-private.",
            },
            "column_mapping": {"type": "object", "description": "SFT column mapping."},
            "max_train_samples": {
                "type": "integer",
                "description": "Optional train cap.",
            },
            "max_eval_samples": {
                "type": "integer",
                "description": "Optional eval cap.",
            },
            "validation_split_ratio": {
                "type": "number",
                "description": "Deterministic eval split ratio. Default: 0.1.",
            },
            "num_train_epochs": {
                "type": "integer",
                "description": "Epoch count. Default: 1.",
            },
            "max_length": {
                "type": "integer",
                "description": "Max sequence length. Default: 1024.",
            },
            "learning_rate": {
                "type": "number",
                "description": "Learning rate. Default: 2e-4.",
            },
            "per_device_train_batch_size": {
                "type": "integer",
                "description": "Per-device train batch size. Default: 1.",
            },
            "gradient_accumulation_steps": {
                "type": "integer",
                "description": "Gradient accumulation steps. Default: 8.",
            },
            "run_name": {
                "type": "string",
                "description": "Optional Trainer/Trackio run name.",
            },
            "trackio_project": {
                "type": "string",
                "description": "Optional Trackio project.",
            },
            "trackio_space_id": {
                "type": "string",
                "description": "Optional Trackio Space id.",
            },
            "image_uri": {
                "type": "string",
                "description": "Required SageMaker training image URI unless configured by env.",
            },
            "s3_bucket": {
                "type": "string",
                "description": "Optional S3 bucket override.",
            },
            "s3_prefix": {
                "type": "string",
                "description": "Optional S3 prefix override. Default: liga-ml.",
            },
            "role_arn": {
                "type": "string",
                "description": "Optional SageMaker execution role ARN override.",
            },
            "instance_type": {
                "type": "string",
                "description": "SageMaker training instance type. Default: ml.g5.xlarge.",
            },
            "instance_count": {
                "type": "integer",
                "description": "SageMaker training instance count. Default: 1.",
            },
            "volume_size_gb": {
                "type": "integer",
                "description": "SageMaker EBS volume size in GB. Default: 30.",
            },
            "max_run_seconds": {
                "type": "integer",
                "description": "Maximum runtime seconds for cost guardrails. Default: 3600.",
            },
            "job_name": {"type": "string", "description": "SageMaker job name."},
            "job_id": {"type": "string", "description": "Alias for job_name."},
        },
        "required": ["operation"],
    },
}


async def aws_sagemaker_jobs_handler(
    arguments: dict[str, Any], session: Any = None, tool_call_id: str | None = None
) -> tuple[str, bool]:
    try:
        tool = AwsSageMakerJobsTool(session=session, tool_call_id=tool_call_id)
        result = await tool.execute(arguments)
        return result["formatted"], not result.get("isError", False)
    except Exception as exc:
        return f"Error executing AWS SageMaker Jobs tool: {exc}", False
