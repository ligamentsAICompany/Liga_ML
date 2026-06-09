"""Mocked tests for Phase 7b AWS SageMaker read-only preflight probes."""

from __future__ import annotations

from typing import Any

import pytest

from agent.core.preflight_aws_sagemaker import run_aws_sagemaker_preflight_checks
from agent.core.training_preflight import PreflightStatus, run_training_preflight


@pytest.fixture(autouse=True)
def clear_aws_env(monkeypatch):
    for name in (
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_S3_BUCKET",
        "AWS_SAGEMAKER_ROLE_ARN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
    ):
        monkeypatch.delenv(name, raising=False)


class FakeAwsClient:
    def __init__(
        self,
        *,
        credentials: Any | None = object(),
        credentials_error: Exception | None = None,
        identity: dict[str, str] | Exception | None = None,
        sagemaker_api: dict[str, Any] | Exception | None = None,
        bucket_read: dict[str, Any] | Exception | None = None,
        s3_write_permission: bool | Exception | None = None,
        role: dict[str, Any] | Exception | None = None,
        quota: dict[str, Any] | Exception | None = None,
    ) -> None:
        self.credentials = credentials
        self.credentials_error = credentials_error
        self.identity = (
            identity
            if identity is not None
            else {
                "Account": "123456789012",
                "Arn": "arn:aws:iam::123456789012:user/tester",
                "UserId": "AIDAEXAMPLE",
            }
        )
        self.sagemaker_api = (
            sagemaker_api if sagemaker_api is not None else {"TrainingJobSummaries": []}
        )
        self.bucket_read = (
            bucket_read
            if bucket_read is not None
            else {"LocationConstraint": "us-east-1"}
        )
        self.s3_write_permission = s3_write_permission
        self.role = (
            role
            if role is not None
            else {"Role": {"Arn": "arn:aws:iam::123456789012:role/TestRole"}}
        )
        self.quota = quota
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def discover_credentials(self) -> Any | None:
        self.calls.append(("discover_credentials", {}))
        if self.credentials_error:
            raise self.credentials_error
        return self.credentials

    def get_caller_identity(self) -> dict[str, str]:
        self.calls.append(("get_caller_identity", {}))
        if isinstance(self.identity, Exception):
            raise self.identity
        return dict(self.identity or {})

    def list_training_jobs(self, *, max_results: int) -> dict[str, Any]:
        self.calls.append(("list_training_jobs", {"max_results": max_results}))
        if isinstance(self.sagemaker_api, Exception):
            raise self.sagemaker_api
        return dict(self.sagemaker_api or {})

    def head_bucket(self, *, bucket: str) -> dict[str, Any]:
        self.calls.append(("head_bucket", {"bucket": bucket}))
        if isinstance(self.bucket_read, Exception):
            raise self.bucket_read
        return dict(self.bucket_read or {})

    def get_bucket_location(self, *, bucket: str) -> dict[str, Any]:
        self.calls.append(("get_bucket_location", {"bucket": bucket}))
        if isinstance(self.bucket_read, Exception):
            raise self.bucket_read
        return dict(self.bucket_read or {})

    def simulate_s3_write_permission(
        self, *, bucket: str, role_arn: str | None
    ) -> bool | None:
        self.calls.append(
            ("simulate_s3_write_permission", {"bucket": bucket, "role_arn": role_arn})
        )
        if isinstance(self.s3_write_permission, Exception):
            raise self.s3_write_permission
        return self.s3_write_permission

    def get_role(self, *, role_name: str) -> dict[str, Any]:
        self.calls.append(("get_role", {"role_name": role_name}))
        if isinstance(self.role, Exception):
            raise self.role
        return dict(self.role or {})

    def get_service_quota(self, *, instance_type: str | None) -> dict[str, Any] | None:
        self.calls.append(("get_service_quota", {"instance_type": instance_type}))
        if isinstance(self.quota, Exception):
            raise self.quota
        return self.quota

    def create_training_job(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_training_job must not be called")

    def create_processing_job(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_processing_job must not be called")

    def create_model(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_model must not be called")

    def create_endpoint(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_endpoint must not be called")

    def create_endpoint_config(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_endpoint_config must not be called")

    def create_transform_job(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_transform_job must not be called")

    def start_pipeline_execution(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("start_pipeline_execution must not be called")

    def create_bucket(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_bucket must not be called")

    def put_object(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("put_object must not be called")

    def upload_file(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("upload_file must not be called")

    def upload_fileobj(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("upload_fileobj must not be called")

    def put_role_policy(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("put_role_policy must not be called")

    def attach_role_policy(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("attach_role_policy must not be called")

    def create_role(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_role must not be called")

    def create_policy(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_policy must not be called")


def _recommendation(
    *,
    provider: str = "aws-sagemaker",
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    hardware_id: str = "aws-sagemaker:ml.g5.xlarge",
    output_policy: str = "cloud-private",
) -> dict[str, Any]:
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


def _checks_by_id(result: Any) -> dict[str, Any]:
    return {check.check_id: check for check in result.primary.checks}


@pytest.mark.asyncio
async def test_missing_aws_credentials_for_sagemaker_blocks_launch():
    result = await run_aws_sagemaker_preflight_checks(
        provider="aws-sagemaker",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="aws-sagemaker:ml.g5.xlarge",
        output_policy="cloud-private",
        target_bucket="s3://training-bucket",
        aws_region="us-east-1",
        execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
        aws_client_factory=lambda _region: FakeAwsClient(credentials=None),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["aws.credentials.present"].status == PreflightStatus.FAILED
    assert result.launch_ready is False
    assert result.metadata["provider_jobs_launched"] is False
    assert result.metadata["resources_created"] is False


@pytest.mark.asyncio
async def test_missing_region_blocks_launch():
    result = await run_aws_sagemaker_preflight_checks(
        provider="aws-sagemaker",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="aws-sagemaker:ml.g5.xlarge",
        output_policy="cloud-private",
        target_bucket="s3://training-bucket",
        aws_region=None,
        execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
        aws_client_factory=lambda _region: FakeAwsClient(),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["aws.region.configured"].status == PreflightStatus.FAILED
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_sts_sagemaker_s3_role_and_hardware_checks_pass_with_mocks():
    fake_client = FakeAwsClient(s3_write_permission=True)

    result = await run_training_preflight(
        session_id="s1",
        recommendation=_recommendation(),
        dataset_summary={"rows": 10},
        target_bucket="s3://training-bucket",
        aws_region="us-east-1",
        aws_execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
        aws_client_factory=lambda region: fake_client,
    )

    checks = _checks_by_id(result)
    assert checks["aws.credentials.present"].status == PreflightStatus.PASSED
    assert checks["aws.identity.sts"].status == PreflightStatus.PASSED
    assert checks["aws.region.configured"].status == PreflightStatus.PASSED
    assert checks["aws.sagemaker.api"].status == PreflightStatus.PASSED
    assert checks["aws.s3.bucket_read"].status == PreflightStatus.PASSED
    assert checks["aws.s3.write_permission"].status == PreflightStatus.PASSED
    assert checks["aws.sagemaker.execution_role"].status == PreflightStatus.PASSED
    assert checks["aws.sagemaker.hardware_catalog"].status == PreflightStatus.PASSED
    assert checks["aws.sagemaker.quota_availability"].status == PreflightStatus.UNKNOWN
    assert result.launch_ready is False
    assert ("list_training_jobs", {"max_results": 1}) in fake_client.calls


@pytest.mark.asyncio
async def test_sts_auth_failure_maps_to_auth_failed_and_blocks():
    result = await run_aws_sagemaker_preflight_checks(
        provider="aws-sagemaker",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="aws-sagemaker:ml.g5.xlarge",
        output_policy="cloud-private",
        target_bucket="s3://training-bucket",
        aws_region="us-east-1",
        execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
        aws_client_factory=lambda _region: FakeAwsClient(
            identity=RuntimeError("401 unauthorized invalid token")
        ),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["aws.identity.sts"].status == PreflightStatus.FAILED
    assert checks["aws.identity.sts"].error_code == "auth_failed"
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_sagemaker_api_access_denied_blocks_safely():
    result = await run_aws_sagemaker_preflight_checks(
        provider="aws-sagemaker",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="aws-sagemaker:ml.g5.xlarge",
        output_policy="cloud-private",
        target_bucket="s3://training-bucket",
        aws_region="us-east-1",
        execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
        aws_client_factory=lambda _region: FakeAwsClient(
            sagemaker_api=RuntimeError("AccessDeniedException: access denied")
        ),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["aws.sagemaker.api"].status == PreflightStatus.FAILED
    assert checks["aws.sagemaker.api"].error_code == "permission_denied"
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_s3_bucket_missing_or_access_denied_blocks_storage_readiness():
    result = await run_aws_sagemaker_preflight_checks(
        provider="aws-sagemaker",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="aws-sagemaker:ml.g5.xlarge",
        output_policy="cloud-private",
        target_bucket="s3://training-bucket",
        aws_region="us-east-1",
        execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
        aws_client_factory=lambda _region: FakeAwsClient(
            bucket_read=RuntimeError("403 access denied")
        ),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["aws.s3.bucket_read"].status == PreflightStatus.FAILED
    assert checks["aws.s3.bucket_read"].error_code == "permission_denied"
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_s3_write_permission_unprovable_is_unknown_not_passed():
    result = await run_aws_sagemaker_preflight_checks(
        provider="aws-sagemaker",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="aws-sagemaker:ml.g5.xlarge",
        output_policy="cloud-private",
        target_bucket="s3://training-bucket",
        aws_region="us-east-1",
        execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
        aws_client_factory=lambda _region: FakeAwsClient(s3_write_permission=None),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["aws.s3.write_permission"].status == PreflightStatus.UNKNOWN
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_execution_role_missing_blocks_sagemaker_launch_readiness():
    result = await run_aws_sagemaker_preflight_checks(
        provider="aws-sagemaker",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="aws-sagemaker:ml.g5.xlarge",
        output_policy="cloud-private",
        target_bucket="s3://training-bucket",
        aws_region="us-east-1",
        execution_role_arn=None,
        aws_client_factory=lambda _region: FakeAwsClient(),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["aws.sagemaker.execution_role"].status == PreflightStatus.FAILED
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_execution_role_present_but_iam_unavailable_is_unknown_warning():
    result = await run_aws_sagemaker_preflight_checks(
        provider="aws-sagemaker",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="aws-sagemaker:ml.g5.xlarge",
        output_policy="cloud-private",
        target_bucket="s3://training-bucket",
        aws_region="us-east-1",
        execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
        aws_client_factory=lambda _region: FakeAwsClient(
            role=RuntimeError("AccessDeniedException: access denied")
        ),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["aws.sagemaker.execution_role"].status == PreflightStatus.UNKNOWN
    assert checks["aws.sagemaker.execution_role"].severity.value == "warning"
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_selected_non_sagemaker_hardware_fails_compatibility():
    result = await run_aws_sagemaker_preflight_checks(
        provider="aws-sagemaker",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="hf-jobs:t4-small",
        output_policy="cloud-private",
        target_bucket="s3://training-bucket",
        aws_region="us-east-1",
        execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
        aws_client_factory=lambda _region: FakeAwsClient(),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["aws.sagemaker.hardware_catalog"].status == PreflightStatus.FAILED
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_service_quotas_unavailable_returns_unknown_warning():
    result = await run_aws_sagemaker_preflight_checks(
        provider="aws-sagemaker",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="aws-sagemaker:ml.g5.xlarge",
        output_policy="cloud-private",
        target_bucket="s3://training-bucket",
        aws_region="us-east-1",
        execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
        aws_client_factory=lambda _region: FakeAwsClient(
            quota=RuntimeError("Service Quotas API unavailable")
        ),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["aws.sagemaker.quota_availability"].status == PreflightStatus.UNKNOWN
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_aws_secrets_paths_redacted_and_no_mutating_methods_called():
    fake_key = "AKIA" + "ABCDEFGHIJKLMNOP"
    fake_client = FakeAwsClient(
        credentials_error=RuntimeError(
            r"credentials at C:\Users\me\.aws\credentials "
            f"contained {fake_key} and AWS_SESSION_TOKEN=secret-token"
        )
    )

    result = await run_aws_sagemaker_preflight_checks(
        provider="aws-sagemaker",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="aws-sagemaker:ml.g5.xlarge",
        output_policy="cloud-private",
        target_bucket="s3://training-bucket",
        aws_region="us-east-1",
        execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
        aws_client_factory=lambda _region: fake_client,
    )

    payload = result.to_dict()
    assert fake_key not in str(payload)
    assert "secret-token" not in str(payload)
    assert ".aws\\credentials" not in str(payload)
    assert "[REDACTED]" in str(payload)
    assert not any(
        call[0]
        in {
            "create_training_job",
            "create_processing_job",
            "create_model",
            "create_endpoint",
            "create_endpoint_config",
            "create_transform_job",
            "start_pipeline_execution",
            "create_bucket",
            "put_object",
            "upload_file",
            "upload_fileobj",
            "put_role_policy",
            "attach_role_policy",
            "create_role",
            "create_policy",
        }
        for call in fake_client.calls
    )
