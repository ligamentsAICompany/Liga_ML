"""Mocked tests for Phase 7b Google Vertex read-only preflight probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agent.core.preflight_gcp_vertex import (
    _probe_vertex_custom_jobs,
    run_gcp_vertex_preflight_checks,
)
from agent.core.training_preflight import PreflightStatus, run_training_preflight


@pytest.fixture(autouse=True)
def clear_gcp_env(monkeypatch):
    for name in (
        "GOOGLE_CLOUD_PROJECT",
        "GCP_PROJECT",
        "GCLOUD_PROJECT",
        "GOOGLE_CLOUD_REGION",
        "GOOGLE_CLOUD_LOCATION",
        "GCP_REGION",
        "GCS_BUCKET",
        "VERTEX_AI_STAGING_BUCKET",
    ):
        monkeypatch.delenv(name, raising=False)


@dataclass
class FakeCredentials:
    refresh_error: Exception | None = None

    def refresh(self, _request: Any) -> None:
        if self.refresh_error:
            raise self.refresh_error


class FakeBucket:
    def __init__(
        self,
        *,
        exists: bool = True,
        exists_error: Exception | None = None,
        permissions: list[str] | Exception | None = None,
    ) -> None:
        self._exists = exists
        self._exists_error = exists_error
        self._permissions = permissions if permissions is not None else []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def exists(self) -> bool:
        self.calls.append(("exists", {}))
        if self._exists_error:
            raise self._exists_error
        return self._exists

    def test_iam_permissions(self, permissions: list[str]) -> list[str]:
        self.calls.append(("test_iam_permissions", {"permissions": permissions}))
        if isinstance(self._permissions, Exception):
            raise self._permissions
        return list(self._permissions)

    def create(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("bucket.create must not be called")

    def blob(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("bucket.blob must not be called")


class FakeGcpClient:
    def __init__(
        self,
        *,
        credentials: FakeCredentials | None = None,
        credentials_error: Exception | None = None,
        project_id: str | None = "proj-1",
        vertex_api: bool | Exception | None = True,
        bucket: FakeBucket | None = None,
    ) -> None:
        self.credentials = credentials or FakeCredentials()
        self.credentials_error = credentials_error
        self.project_id = project_id
        self.vertex_api = vertex_api
        self.bucket = bucket or FakeBucket(permissions=["storage.objects.create"])
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def discover_credentials(self) -> tuple[FakeCredentials, str | None]:
        self.calls.append(("discover_credentials", {}))
        if self.credentials_error:
            raise self.credentials_error
        return self.credentials, self.project_id

    def refresh_credentials(self, credentials: FakeCredentials) -> None:
        self.calls.append(("refresh_credentials", {}))
        credentials.refresh(None)

    def check_vertex_api(self, *, project_id: str, region: str) -> bool | None:
        self.calls.append(
            ("check_vertex_api", {"project_id": project_id, "region": region})
        )
        if isinstance(self.vertex_api, Exception):
            raise self.vertex_api
        return self.vertex_api

    def get_bucket(self, bucket_name: str) -> FakeBucket:
        self.calls.append(("get_bucket", {"bucket_name": bucket_name}))
        return self.bucket

    def create_custom_job(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_custom_job must not be called")

    def run(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("run must not be called")

    def submit(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("submit must not be called")

    def create_training_pipeline(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_training_pipeline must not be called")

    def create_model(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_model must not be called")

    def upload_model(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("upload_model must not be called")

    def deploy_model(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("deploy_model must not be called")

    def create_endpoint(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_endpoint must not be called")

    def batch_predict(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("batch_predict must not be called")

    def create_bucket(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_bucket must not be called")


class FakeListCustomJobsClientWithoutPageSize:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def list_custom_jobs(self, *, parent: str):
        self.calls.append({"parent": parent})
        return iter([])

    def create_custom_job(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_custom_job must not be called")


def _recommendation(
    *,
    provider: str = "gcp-vertex",
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    hardware_id: str = "gcp-vertex:n1-standard-8-t4",
    output_policy: str = "cloud-private",
    training_goal: str = "smoke-test",
    estimated_cost_usd: float = 1.1,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "training_goal": training_goal,
        "recommended_model": model_id,
        "output_policy": output_policy,
        "recommendation": {
            "training_goal": training_goal,
            "estimated_cost_usd": estimated_cost_usd,
            "selected_provider": {"provider_id": provider},
            "selected_model": {"model_id": model_id},
            "selected_hardware": {"hardware_id": hardware_id},
            "output_policy": output_policy,
        },
    }


def _checks_by_id(result) -> dict[str, Any]:
    return {check.check_id: check for check in result.primary.checks}


def test_vertex_custom_jobs_probe_retries_without_page_size_for_older_sdk():
    client = FakeListCustomJobsClientWithoutPageSize()

    result = _probe_vertex_custom_jobs(
        client,
        parent="projects/proj-1/locations/us-central1",
    )

    assert result is True
    assert client.calls == [{"parent": "projects/proj-1/locations/us-central1"}]


@pytest.mark.asyncio
async def test_missing_gcp_credentials_for_vertex_blocks_launch():
    result = await run_gcp_vertex_preflight_checks(
        provider="gcp-vertex",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="gcp-vertex:n1-standard-8-t4",
        output_policy="cloud-private",
        project_id="proj-1",
        region="us-central1",
        target_bucket="gs://bucket",
        gcp_client_factory=lambda: FakeGcpClient(
            credentials_error=RuntimeError("ADC credentials not found")
        ),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["gcp.credentials.present"].status == PreflightStatus.FAILED
    assert result.launch_ready is False
    assert result.metadata["provider_jobs_launched"] is False
    assert result.metadata["resources_created"] is False


@pytest.mark.asyncio
async def test_missing_project_id_blocks_launch():
    result = await run_gcp_vertex_preflight_checks(
        provider="gcp-vertex",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="gcp-vertex:n1-standard-8-t4",
        output_policy="cloud-private",
        project_id=None,
        region="us-central1",
        target_bucket="gs://bucket",
        gcp_client_factory=lambda: FakeGcpClient(project_id=None),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["gcp.project.configured"].status == PreflightStatus.FAILED
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_missing_region_blocks_launch():
    result = await run_gcp_vertex_preflight_checks(
        provider="gcp-vertex",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="gcp-vertex:n1-standard-8-t4",
        output_policy="cloud-private",
        project_id="proj-1",
        region=None,
        target_bucket="gs://bucket",
        gcp_client_factory=lambda: FakeGcpClient(),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["gcp.region.configured"].status == PreflightStatus.FAILED
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_mocked_refresh_vertex_api_bucket_and_write_permission_checks():
    fake_client = FakeGcpClient(
        bucket=FakeBucket(permissions=["storage.objects.create"]),
        vertex_api=True,
    )

    result = await run_training_preflight(
        session_id="s1",
        recommendation=_recommendation(),
        dataset_summary={"rows": 10},
        target_bucket="gs://bucket",
        gcp_project_id="proj-1",
        gcp_region="us-central1",
        gcp_client_factory=lambda: fake_client,
    )

    checks = _checks_by_id(result)
    assert checks["gcp.credentials.present"].status == PreflightStatus.PASSED
    assert checks["gcp.credentials.refresh"].status == PreflightStatus.PASSED
    assert checks["gcp.project.configured"].status == PreflightStatus.PASSED
    assert checks["gcp.region.configured"].status == PreflightStatus.PASSED
    assert checks["gcp.vertex.api"].status == PreflightStatus.PASSED
    assert checks["gcp.gcs.bucket_read"].status == PreflightStatus.PASSED
    assert checks["gcp.gcs.write_permission"].status == PreflightStatus.PASSED
    assert checks["gcp.vertex.hardware_catalog"].status == PreflightStatus.PASSED
    assert checks["gcp.vertex.quota_availability"].status == PreflightStatus.UNKNOWN
    assert result.launch_ready is False
    assert result.manual_approval_allowed is True
    assert result.approval_required is True
    assert ("get_bucket", {"bucket_name": "bucket"}) in fake_client.calls


@pytest.mark.asyncio
async def test_credential_refresh_failure_maps_to_auth_failed():
    result = await run_gcp_vertex_preflight_checks(
        provider="gcp-vertex",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="gcp-vertex:n1-standard-8-t4",
        output_policy="cloud-private",
        project_id="proj-1",
        region="us-central1",
        target_bucket="gs://bucket",
        gcp_client_factory=lambda: FakeGcpClient(
            credentials=FakeCredentials(RuntimeError("401 invalid credential"))
        ),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["gcp.credentials.refresh"].status == PreflightStatus.FAILED
    assert checks["gcp.credentials.refresh"].error_code == "auth_failed"
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_vertex_api_disabled_blocks_launch():
    result = await run_gcp_vertex_preflight_checks(
        provider="gcp-vertex",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="gcp-vertex:n1-standard-8-t4",
        output_policy="cloud-private",
        project_id="proj-1",
        region="us-central1",
        target_bucket="gs://bucket",
        gcp_client_factory=lambda: FakeGcpClient(vertex_api=False),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["gcp.vertex.api"].status == PreflightStatus.FAILED
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_vertex_api_auth_or_permission_error_still_blocks_launch():
    result = await run_gcp_vertex_preflight_checks(
        provider="gcp-vertex",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="gcp-vertex:n1-standard-8-t4",
        output_policy="cloud-private",
        project_id="proj-1",
        region="us-central1",
        target_bucket="gs://bucket",
        gcp_client_factory=lambda: FakeGcpClient(
            vertex_api=RuntimeError("403 permission denied")
        ),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["gcp.vertex.api"].status == PreflightStatus.FAILED
    assert checks["gcp.vertex.api"].error_code == "permission_denied"
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_gcs_bucket_missing_or_access_denied_blocks_storage_readiness():
    result = await run_gcp_vertex_preflight_checks(
        provider="gcp-vertex",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="gcp-vertex:n1-standard-8-t4",
        output_policy="cloud-private",
        project_id="proj-1",
        region="us-central1",
        target_bucket="gs://missing",
        gcp_client_factory=lambda: FakeGcpClient(
            bucket=FakeBucket(exists_error=RuntimeError("403 permission denied"))
        ),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["gcp.gcs.bucket_read"].status == PreflightStatus.FAILED
    assert checks["gcp.gcs.bucket_read"].error_code == "permission_denied"
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_gcs_write_permission_unprovable_is_unknown_not_passed():
    result = await run_gcp_vertex_preflight_checks(
        provider="gcp-vertex",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="gcp-vertex:n1-standard-8-t4",
        output_policy="cloud-private",
        project_id="proj-1",
        region="us-central1",
        target_bucket="gs://bucket",
        gcp_client_factory=lambda: FakeGcpClient(
            bucket=FakeBucket(
                permissions=RuntimeError("testIamPermissions unsupported")
            )
        ),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["gcp.gcs.write_permission"].status == PreflightStatus.UNKNOWN
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_selected_non_vertex_hardware_fails_compatibility():
    result = await run_gcp_vertex_preflight_checks(
        provider="gcp-vertex",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="hf-jobs:t4-small",
        output_policy="cloud-private",
        project_id="proj-1",
        region="us-central1",
        target_bucket="gs://bucket",
        gcp_client_factory=lambda: FakeGcpClient(),
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["gcp.vertex.hardware_catalog"].status == PreflightStatus.FAILED
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_credential_paths_are_redacted_and_no_mutating_methods_called():
    fake_client = FakeGcpClient(
        credentials_error=RuntimeError(
            r"credentials at C:\Users\me\service-account.json were denied"
        ),
    )

    result = await run_gcp_vertex_preflight_checks(
        provider="gcp-vertex",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="gcp-vertex:n1-standard-8-t4",
        output_policy="cloud-private",
        project_id="proj-1",
        region="us-central1",
        target_bucket="gs://bucket",
        gcp_client_factory=lambda: fake_client,
    )

    payload = result.to_dict()
    assert "service-account.json" not in str(payload)
    assert "[REDACTED]" in str(payload)
    assert not any(
        call[0]
        in {
            "create_custom_job",
            "run",
            "submit",
            "create_training_pipeline",
            "create_model",
            "upload_model",
            "deploy_model",
            "create_endpoint",
            "batch_predict",
            "create_bucket",
        }
        for call in fake_client.calls
    )
